"""Przygotowanie PMData do prognozowania następnego dnia kalendarzowego.

Źródła łączone są na niezależnym kalendarzu UTC, bez interpolacji pomiarów
ani celu. Zero sRPE oznacza brak zapisanej sesji przy dostępnym pliku, a nie
potwierdzony odpoczynek. Brak pliku lub niepełna sesja pozostawia obciążenie
nieznane. Nazwy kolumn i kluczy metadanych zachowano dla zgodności z modelami.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import PMDATA_DIR, PolishArgumentParser, print_summary
from src.data.pmdata.features import (
    add_composite_wellness,
    add_ewma_load_features,
    add_lag_features,
    add_relative_rhr,
    add_rolling_load_features,
    validate_daily_calendar,
)

PIPELINE_VERSION = "pmdata-observed-calendar-v2"
WELLNESS_COLUMNS = ["readiness", "fatigue", "soreness", "sleep_duration_h", "sleep_quality", "mood", "stress"]
SLEEP_COLUMNS = ["sleep_overall_score", "sleep_deep_minutes", "sleep_restlessness"]
SOURCE_FILES = ["pmsys/wellness.csv", "pmsys/srpe.csv", "fitbit/resting_heart_rate.json", "fitbit/sleep_score.csv", "fitbit/time_in_heart_rate_zones.json"]


def _empty_daily(columns):
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), **{c: pd.Series(dtype=float) for c in columns}})


def _utc_timestamps(values):
    result = pd.to_datetime(values, format="mixed", utc=True, errors="raise")
    if result.isna().any():
        raise ValueError("Brak znacznika czasu w danych źródłowych")
    return result


def _latest_daily(frame, timestamp_col, value_cols):
    """Zachowuje cały ostatni raport według czasu; remisy rozstrzyga kolejność w pliku."""
    if frame.empty:
        return _empty_daily(value_cols)
    frame = frame.copy()
    frame["_timestamp"] = _utc_timestamps(frame[timestamp_col])
    frame["date"] = frame["_timestamp"].dt.tz_localize(None).dt.normalize()
    for col in value_cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise") if col in frame else np.nan
        if not np.isfinite(frame[col].dropna()).all():
            raise ValueError(f"Nieskończony pomiar źródłowy: {col}")
    return (frame.sort_values("_timestamp", kind="stable").drop_duplicates("date", keep="last")
            [["date"] + list(value_cols)].reset_index(drop=True))


def load_resting_heart_rate(participant_dir: str) -> pd.DataFrame:
    """Wczytuje ostatni dostępny pomiar tętna dla każdego dnia."""
    path = Path(participant_dir) / "fitbit/resting_heart_rate.json"
    if not path.exists():
        return _empty_daily(["resting_heart_rate"])
    raw = json.loads(path.read_text())
    records = [{"timestamp": r["dateTime"], "resting_heart_rate": r.get("value", {}).get("value")}
               for r in raw if r.get("dateTime")]
    return _latest_daily(pd.DataFrame(records), "timestamp", ["resting_heart_rate"])


def load_sleep_score(participant_dir: str) -> pd.DataFrame:
    """Wczytuje ostatni dzienny raport jakości i parametrów snu."""
    path = Path(participant_dir) / "fitbit/sleep_score.csv"
    if not path.exists():
        return _empty_daily(SLEEP_COLUMNS)
    frame = pd.read_csv(path).rename(columns={"overall_score": "sleep_overall_score", "deep_sleep_in_minutes": "sleep_deep_minutes", "restlessness": "sleep_restlessness"})
    return _latest_daily(frame, "timestamp", SLEEP_COLUMNS)


def load_hr_zones(participant_dir: str) -> pd.DataFrame:
    """Wczytuje czas aktywności w strefach tętna bez uzupełniania braków."""
    path = Path(participant_dir) / "fitbit/time_in_heart_rate_zones.json"
    if not path.exists():
        return _empty_daily(["hr_zone_active_minutes"])
    raw = json.loads(path.read_text())
    records = []
    for record in raw:
        if not record.get("dateTime"):
            continue
        zones = record.get("value", {}).get("valuesInZones", {})
        records.append({"timestamp": record["dateTime"], "hr_zone_active_minutes":
                        zones.get("IN_DEFAULT_ZONE_2", np.nan) + zones.get("IN_DEFAULT_ZONE_3", np.nan)})
    return _latest_daily(pd.DataFrame(records), "timestamp", ["hr_zone_active_minutes"])


def load_participant_data(participant_dir: str) -> pd.DataFrame | None:
    """Łączy źródła jednej osoby na pełnym kalendarzu UTC i tworzy audyt."""
    root = Path(participant_dir)
    wellness_path, srpe_path = root / SOURCE_FILES[0], root / SOURCE_FILES[1]
    wellness_raw = pd.read_csv(wellness_path) if wellness_path.exists() else pd.DataFrame()
    wellness = _latest_daily(wellness_raw, "effective_time_frame", WELLNESS_COLUMNS)
    srpe = pd.read_csv(srpe_path) if srpe_path.exists() else pd.DataFrame()
    if srpe.empty:
        sessions = _empty_daily(["daily_srpe", "n_sessions"])
    else:
        srpe = srpe.copy()
        srpe["date"] = _utc_timestamps(srpe["end_date_time"]).dt.tz_localize(None).dt.normalize()
        for col in ["perceived_exertion", "duration_min"]:
            srpe[col] = pd.to_numeric(srpe[col], errors="raise")
            if srpe[col].dropna().lt(0).any() or not np.isfinite(srpe[col].dropna()).all():
                raise ValueError(f"Nieprawidłowa wartość {col} dla {root.name}")
        srpe["session_srpe"] = srpe.perceived_exertion * srpe.duration_min
        sessions = srpe.groupby("date").agg(
            daily_srpe=("session_srpe", lambda s: s.sum(min_count=len(s))),
            n_sessions=("session_srpe", "size"),
        ).reset_index()
    sources = [wellness, sessions, load_resting_heart_rate(str(root)), load_sleep_score(str(root)), load_hr_zones(str(root))]
    dates = pd.concat([source["date"] for source in sources], ignore_index=True).dropna()
    if dates.empty:
        return None
    daily = pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq="D")})
    for source in sources:
        daily = daily.merge(source, on="date", how="left", validate="one_to_one")
    daily["participant"] = root.name
    daily["wellness_reported"] = daily.date.isin(wellness.date)
    daily["srpe_file_available"] = srpe_path.exists()
    daily["srpe_reported"] = daily.date.isin(sessions.date)
    # Uzupełniamy tylko dni bez sesji, nie sesje o nieznanym obciążeniu.
    no_record = ~daily.srpe_reported
    if srpe_path.exists():
        daily.loc[no_record, ["daily_srpe", "n_sessions"]] = 0
    daily.attrs["source_audit"] = {
        "participant": root.name,
        "wellness_reports": len(wellness_raw), "wellness_days": len(wellness),
        "extra_same_day_wellness_reports": len(wellness_raw) - len(wellness),
        "session_records": len(srpe), "session_days": len(sessions),
        "session_days_without_wellness": int((~sessions.date.isin(wellness.date)).sum()),
        "srpe_file_available": srpe_path.exists(),
        "unknown_recorded_load_days": int(daily.loc[daily.srpe_reported, "daily_srpe"].isna().sum()),
    }
    return daily


def load_all_participants(data_dir: str) -> pd.DataFrame:
    """Wczytuje osoby z katalogów p01, p02 itd. wraz z audytem źródeł."""
    roots = sorted(p for p in Path(data_dir).glob("p*") if p.is_dir() and p.name[1:].isdigit())
    if not roots:
        raise ValueError(f"Brak folderów uczestników (p01, p02, ...) w {data_dir}")
    frames = [load_participant_data(str(root)) for root in roots]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise ValueError("Nie znaleziono rekordów źródłowych z datą")
    audits = [frame.attrs["source_audit"] for frame in frames]
    result = pd.concat(frames, ignore_index=True)
    result.attrs["source_audits"] = audits
    return result


def build_full_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnia brakujące daty kalendarza, zachowując brakujące pomiary."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if df[["participant", "date"]].isna().any().any() or df.duplicated(["participant", "date"]).any():
        raise ValueError("Przed utworzeniem kalendarza zagreguj powtarzające się raporty dzienne")
    frames = []
    for participant, group in df.groupby("participant"):
        dates = pd.date_range(group.date.min(), group.date.max(), freq="D")
        full = pd.DataFrame({"participant": participant, "date": dates})
        frames.append(full.merge(group, on=["participant", "date"], how="left", validate="one_to_one"))
    return validate_daily_calendar(pd.concat(frames, ignore_index=True))


def compute_gap_lengths(df: pd.DataFrame, value_col: str = "readiness") -> pd.DataFrame:
    """Zestawia początki i długości luk w obserwacjach każdej osoby."""
    records = []
    for participant, group in validate_daily_calendar(df).groupby("participant"):
        missing = group[value_col].isna()
        blocks = missing.ne(missing.shift()).cumsum()
        for _, gap in group.loc[missing].groupby(blocks[missing]):
            records.append({"participant": participant, "gap_length_days": len(gap), "gap_start_date": gap.date.iloc[0]})
    return pd.DataFrame(records, columns=["participant", "gap_length_days", "gap_start_date"])


def mark_long_gaps(df: pd.DataFrame, gaps: pd.DataFrame, max_short_gap: int = 2) -> pd.DataFrame:
    """Oznacza długie luki wyłącznie do audytu; znacznik nie jest cechą modelu."""
    df = df.copy()
    df["in_long_gap"] = False
    for row in gaps.loc[gaps.gap_length_days > max_short_gap].itertuples():
        end = row.gap_start_date + pd.Timedelta(days=row.gap_length_days - 1)
        mask = df.participant.eq(row.participant) & df.date.between(row.gap_start_date, end)
        df.loc[mask, "in_long_gap"] = True
    return df


def add_target_column(df: pd.DataFrame, target_col: str, horizon: int = 1, predict_delta: bool = False) -> pd.DataFrame:
    """Wyznacza poziom lub zmianę wskaźnika po zadanej liczbie dni."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Horyzont musi być dodatnią całkowitą liczbą dni kalendarzowych")
    df = validate_daily_calendar(df)
    future = df.groupby("participant")[target_col].shift(-horizon)
    df["target"] = future - df[target_col] if predict_delta else future
    df["target_date"] = df.date + pd.Timedelta(days=horizon)
    return df


def build_ml_dataset(df: pd.DataFrame, target_col: str = "composite_wellness", horizon: int = 1,
                     lag_cols: list[str] | None = None, lags: list[int] | None = None,
                     predict_delta: bool = False) -> pd.DataFrame:
    """Tworzy cechy i wybiera wiersze z zaobserwowanym stanem i celem."""
    lag_cols = lag_cols if lag_cols is not None else ["daily_srpe", "readiness", "fatigue", "soreness", "sleep_duration_h", "sleep_quality", "rhr_relative", "sleep_overall_score", "sleep_deep_minutes", "hr_zone_active_minutes"]
    lags = lags if lags is not None else [1, 2, 3]
    if not lags or any(not isinstance(lag, int) or isinstance(lag, bool) or lag < 1 for lag in lags):
        raise ValueError("Opóźnienia muszą być dodatnimi całkowitymi liczbami dni")
    df = validate_daily_calendar(df)
    df["day_of_week"] = df.date.dt.dayofweek
    df = add_lag_features(df, cols=lag_cols, lags=lags)
    df = add_rolling_load_features(df, missing_policy="preserve")
    df = add_ewma_load_features(df, missing_policy="preserve")
    df = add_target_column(df, target_col, horizon, predict_delta)
    features = ([f"{col}_lag{lag}" for col in lag_cols for lag in lags]
                + ["acute_load", "chronic_load", "acwr", "ewma_acute_load", "ewma_chronic_load", "ewma_acwr", "day_of_week", target_col])
    # Bieżąca wartość i cel pochodzą z obserwacji. Braki historii pozostają NaN.
    # Modele mogą je odrzucić lub imputować wyłącznie na podstawie treningu.
    keep = df[target_col].notna() & df.target.notna()
    dataset = df.loc[keep, ["participant", "date", "target"] + features].replace([np.inf, -np.inf], np.nan).copy()
    dataset = dataset.dropna(subset=[target_col, "target"])
    return dataset.reset_index(drop=True)


def summarize_ml_dataset(dataset: pd.DataFrame) -> None:
    """Wyświetla kompletność cech i statystyki celu po polsku."""
    print(f"Liczba cech: {len(dataset.columns) - 3}")
    print(f"Wiersze z kompletnymi cechami: {len(dataset.dropna())}")
    labels = {"count": "Liczba", "mean": "Średnia", "std": "Odchylenie standardowe",
              "min": "Minimum", "25%": "Kwartyl 25%", "50%": "Mediana", "75%": "Kwartyl 75%", "max": "Maksimum"}
    print("Statystyki zmiennej celu:")
    print(dataset.target.describe().rename(index=labels).round(3).to_string())


def read_dataset_metadata(path):
    """Odczytuje plik metadanych towarzyszący zbiorowi CSV."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    return json.loads(sidecar.read_text()) if sidecar.exists() else {}


def resolve_target_mode(metadata, requested=None, default=False):
    """Sprawdza zgodność wariantu celu, aby zapobiec użyciu niewłaściwego punktu odniesienia."""
    recorded = metadata.get("predict_delta")
    if requested is not None and recorded is not None and requested != recorded:
        raise ValueError("Wybrany wariant celu jest sprzeczny z metadanymi zbioru")
    return requested if requested is not None else metadata.get("predict_delta", default)


def prepare_dataset(data_dir, output, target_col="composite_wellness", horizon=1, predict_delta=False):
    """Przygotowuje zbiór i zapisuje kalendarz, luki oraz metadane."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Horyzont musi być dodatnią liczbą całkowitą")
    root, output = Path(data_dir), Path(output)
    loaded = load_all_participants(str(root))
    audits = loaded.attrs["source_audits"]
    daily = build_full_calendar(loaded)
    gaps = compute_gap_lengths(daily)
    daily = mark_long_gaps(daily, gaps)
    # Obie funkcje korzystają z obserwacji i statystyk z dni wcześniejszych.
    daily = add_composite_wellness(daily)
    daily = add_relative_rhr(daily)
    dataset = build_ml_dataset(daily, target_col=target_col, horizon=horizon, predict_delta=predict_delta)
    feature_cols = [c for c in dataset if c not in {"participant", "date", "target"}]
    complete = dataset.dropna(subset=feature_cols + ["target"])
    metadata = {
        "pipeline_version": PIPELINE_VERSION, "predict_delta": predict_delta, "level_col": target_col, "horizon": horizon,
        "calendar_rows": len(daily), "calendar_participants": int(daily.participant.nunique()),
        "dataset_rows": len(dataset), "dataset_participants": int(dataset.participant.nunique()),
        "complete_rows": len(complete), "complete_participants": int(complete.participant.nunique()),
        "feature_cols": feature_cols, "source_audits": audits,
        "duplicate_policy": "Ostatni cały raport dnia według czasu UTC; remisy według kolejności w pliku; sesje sumowane bez usuwania duplikatów",
        "missing_policy": "Bez interpolacji, uzupełniania z przyszłości ani imputacji celu; statystyki historyczne pomijają braki przy zachowaniu minimalnej liczby obserwacji",
        "load_policy": "Tylko zapisane sRPE: brak sesji przy dostępnym pliku oznacza 0 zapisanego obciążenia, nie potwierdzony odpoczynek; brak pliku lub niepełna sesja oznacza nieznane obciążenie",
        "load_feature_policy": "Nieznane obciążenie unieważnia sumę, dopóki jest w oknie; EWMA rozpoczyna się ponownie po brakującym dniu",
        "gap_policy": "Długie luki oznaczane tylko do audytu; brak sztucznych wartości w cechach i celu",
        "composite_definition": "Zachowano definicję: z(readiness) - z(fatigue) - z(soreness); 21 poprzednich dni, minimum 5 obserwacji; zerowe odchylenie daje NaN; wskaźnik nie jest zwalidowaną miarą regeneracji",
        "time_policy": "Dni kalendarzowe UTC, prognoza na koniec dnia t; zakłada się dostępność raportów tego dnia; eksport nie zawiera osobnego audytu czasu przesłania",
        "downstream_imputation": "Imputację cech dopasowuj tylko na części treningowej; nigdy nie imputuj celu",
        "compatibility": "Nowy zbiór wymaga ponownego treningu i zgodnego przetwarzania w aplikacji. Przygotowanie danych nie aktualizuje wdrożonego modelu.",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for participant in sorted(root.glob("p*")) if participant.is_dir() and participant.name[1:].isdigit()
                          for relative in SOURCE_FILES if (p := participant / relative).exists()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    daily_path = output.parent / "daily_calendar.csv"
    gaps_path = output.parent / "gaps.csv"
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    daily.to_csv(daily_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary("PMData", len(dataset), dataset.participant.nunique(), {
        "Zbiór modelowy": output, "Kalendarz dzienny": daily_path,
        "Audyt luk": gaps_path, "Metadane": metadata_path,
    })
    summarize_ml_dataset(dataset)
    return dataset, daily, metadata


def main():
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder źródłowy z katalogami uczestników PMData.")
    parser.add_argument("--target-col", default="composite_wellness", choices=["composite_wellness", "readiness"], help="Wskaźnik prognozowany przez model.")
    parser.add_argument("--horizon", type=int, default=1, help="Horyzont prognozy w dniach (domyślnie: 1).")
    parser.add_argument("--predict-delta", action="store_true", help="Prognozuj zmianę względem dnia bieżącego zamiast poziomu.")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path, default=PMDATA_DIR, help="Folder wynikowy (domyślnie: data/processed/pmdata).")
    destination.add_argument("--output", type=Path, help="Opcjonalna ścieżka CSV zamiast folderu wynikowego.")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("Horyzont --horizon musi być dodatni.")
    name = "ml_dataset_delta.csv" if args.predict_delta else "ml_dataset.csv"
    output = args.output if args.output is not None else args.output_dir / name
    prepare_dataset(args.data_dir, output, args.target_col, args.horizon, args.predict_delta)


if __name__ == "__main__":
    main()
