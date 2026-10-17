"""Przygotowanie dziennych danych modelowych (SoccerMon)
======================================================

1. Wczytuje raporty wellness, opublikowane obciążenia i sesje treningowe.
2. Łączy je na pełnym kalendarzu, sprawdza skale i audytuje zgodność obciążeń.
3. Tworzy cechy bieżące i historyczne oraz cel fatigue następnego dnia.
4. Zachowuje rzeczywiste raporty celu i wymaganą historię, bez interpolacji.
5. Zapisuje dataset.csv, daily_calendar.csv i audit.json do
   data/processed/soccermon/.

Prognoza dotyczy końca dnia t. Zero obciążenia oznacza brak zapisanego
obciążenia, nie potwierdzony odpoczynek.
Użycie: python -m src.data.soccermon.preparation --data-dir /sciezka/do/soccermon
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import SOCCERMON_DIR, PolishArgumentParser, print_summary

WELLNESS = ["fatigue", "soreness", "readiness", "sleep_duration", "sleep_quality", "mood", "stress"]


def read_daily(path: Path, name: str) -> pd.DataFrame:
    """Przekształca dzienną tabelę źródłową do wierszy osoba–data."""
    wide = pd.read_csv(path)
    dates = pd.to_datetime(wide.iloc[:, 0], format="%d.%m.%Y", errors="raise")
    if dates.duplicated().any():
        raise ValueError(f"Powtarzające się daty w {path}")
    wide = wide.rename(columns={wide.columns[0]: "date"})
    wide["date"] = dates
    return wide.melt(id_vars="date", var_name="participant", value_name=name)


def load_daily(root: Path) -> tuple[pd.DataFrame, dict]:
    """Łączy dzienne źródła na pełnym kalendarzu i audytuje obciążenia."""
    paths = [root / "wellness" / f"{name}.csv" for name in WELLNESS]
    paths += [root / "training-load" / "daily_load.csv", root / "training-load" / "session.json"]
    frames = [read_daily(path, name) for path, name in zip(paths, WELLNESS)]
    frames.append(read_daily(paths[-2], "daily_srpe"))
    daily = frames[0]
    for frame in frames[1:]:
        daily = daily.merge(frame, on=["participant", "date"], how="outer", validate="one_to_one")
    # Pełny kalendarz zapewnia przesunięcia o dni, również przy brakujących datach źródeł.
    calendar = pd.MultiIndex.from_product(
        [sorted(daily.participant.unique()), pd.date_range(daily.date.min(), daily.date.max())],
        names=["participant", "date"],
    )
    daily = daily.set_index(["participant", "date"]).reindex(calendar).reset_index()
    daily["team"] = daily.participant.str.split("-").str[0]
    for name in WELLNESS + ["daily_srpe"]:
        daily[name] = pd.to_numeric(daily[name], errors="raise")
    for name in ["fatigue", "soreness", "sleep_quality", "mood", "stress"]:
        if not daily[name].dropna().between(1, 5).all():
            raise ValueError(f"Nieprawidłowy zakres skali {name}")
    if not daily.readiness.dropna().between(1, 10).all():
        raise ValueError("Nieprawidłowy zakres skali readiness")
    if (daily.daily_srpe.dropna() < 0).any():
        raise ValueError("Ujemne obciążenie treningowe")

    with paths[-1].open() as stream:
        sessions = json.load(stream)
    records = pd.DataFrame([{"participant": person, **record} for person, values in sessions.items() for record in values])
    records["date"] = pd.to_datetime(records.date, format="%d.%m.%Y")
    summed = records.groupby(["participant", "date"]).srpe.sum().rename("session_sum").reset_index()
    check = summed.merge(daily[["participant", "date", "daily_srpe"]], on=["participant", "date"], how="outer", validate="one_to_one")
    mismatches = check.session_sum.notna() & check.daily_srpe.notna() & ~np.isclose(check.session_sum, check.daily_srpe)
    audit = {
        "participants": int(daily.participant.nunique()),
        "calendar_rows": len(daily),
        "date_start": str(daily.date.min().date()), "date_end": str(daily.date.max().date()),
        "observed_values": {name: int(daily[name].notna().sum()) for name in WELLNESS},
        "session_records": len(records),
        "identical_session_records_beyond_first": int(records.duplicated().sum()),
        "session_srpe_formula_mismatches": int((records.srpe != records.rpe * records.duration).sum()),
        "daily_session_sum_mismatch_days": int(mismatches.sum()),
        "positive_daily_load_without_session_day": int((check.session_sum.isna() & check.daily_srpe.gt(0)).sum()),
        "session_days_outside_daily_calendar": int((check.session_sum.notna() & check.daily_srpe.isna()).sum()),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "load_policy": "Opublikowany daily_load.csv; zero oznacza brak zapisanego obciążenia, nie potwierdzony odpoczynek; nierozróżnialne sesje nie są usuwane.",
    }
    return daily, audit


def make_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Tworzy cechy historyczne, próg intensywności i cel następnego dnia."""
    daily = daily.copy().sort_values(["participant", "date"]).reset_index(drop=True)
    if daily.duplicated(["participant", "date"]).any():
        raise ValueError("Wymagany jeden wiersz na osobę i datę")
    steps = daily.groupby("participant").date.diff().dropna()
    if not steps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Wymagane kolejne dni kalendarzowe")
    groups = daily.groupby("participant", sort=False)
    wellness_features = []
    for name in WELLNESS:
        wellness_features.append(name)
        for lag in [1, 2, 3]:
            feature = f"{name}_lag{lag}"
            daily[feature] = groups[name].shift(lag)
            wellness_features.append(feature)
    for window in [7, 21]:
        for stat in ["mean", "std"]:
            feature = f"fatigue_{stat}_{window}"
            daily[feature] = groups.fatigue.transform(lambda s: getattr(s.rolling(window, min_periods=3), stat)())
            wellness_features.append(feature)
    daily["weekday_sin"] = np.sin(2 * np.pi * daily.date.dt.dayofweek / 7)
    daily["weekday_cos"] = np.cos(2 * np.pi * daily.date.dt.dayofweek / 7)
    wellness_features += ["weekday_sin", "weekday_cos"]
    load_features = ["daily_srpe"]
    for lag in [1, 2, 3]:
        feature = f"daily_srpe_lag{lag}"
        daily[feature] = groups.daily_srpe.shift(lag)
        load_features.append(feature)
    for window in [7, 21]:
        feature = f"load_mean_{window}"
        daily[feature] = groups.daily_srpe.transform(lambda s: s.rolling(window, min_periods=window).mean())
        load_features.append(feature)
    daily["acwr"] = daily.load_mean_7 / daily.load_mean_21.replace(0, np.nan)
    load_features.append("acwr")
    # Próg intensywności wykorzystuje tylko zapisane dodatnie obciążenia sprzed dnia t.
    daily["intense_threshold"] = groups.daily_srpe.transform(
        lambda s: s.where(s > 0).shift(1).rolling(28, min_periods=7).quantile(0.75)
    )
    daily["intense_day"] = daily.daily_srpe.gt(daily.intense_threshold) & daily.intense_threshold.notna()
    daily["fatigue_history_count"] = groups.fatigue.transform(lambda s: s.shift(1).rolling(21, min_periods=1).count())
    daily["target"] = groups.fatigue.shift(-1)
    daily["target_date"] = daily.date + pd.Timedelta(days=1)
    sets = {"today_only": ["fatigue"], "wellness": wellness_features, "wellness_load": wellness_features + load_features}
    return daily, sets


def prepare(root: Path, output: Path) -> dict:
    """Przygotowuje zbiór i zapisuje kalendarz oraz audyt źródeł."""
    daily, audit = load_daily(root)
    featured, feature_sets = make_features(daily)
    dataset = featured.loc[featured.fatigue.notna() & featured.target.notna() & featured.fatigue_history_count.ge(7)].copy()
    audit.update({
        "dataset_rows": len(dataset), "dataset_participants": int(dataset.participant.nunique()),
        "intense_day_rows": int(dataset.intense_day.sum()),
        "feature_sets": feature_sets,
        "target": "Zaobserwowane fatigue następnego dnia kalendarzowego, oryginalna skala 1–5",
        "prediction_time": "Koniec dnia t; uwzględnia raport wellness i zapisane obciążenie z dnia t",
        "imputation": "Bez interpolacji ani imputacji celu; imputację cech należy dopasować wyłącznie na części treningowej",
        "minimum_history": "Co najmniej 7 raportów fatigue w poprzednich 21 dniach kalendarzowych",
    })
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_calendar.csv", index=False)
    dataset.to_csv(output / "dataset.csv", index=False)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary("SoccerMon", len(dataset), dataset.participant.nunique(), {
        "Zbiór modelowy": output / "dataset.csv", "Kalendarz dzienny": output / "daily_calendar.csv",
        "Audyt": output / "audit.json",
    })
    print(f"Liczba cech wellness i obciążenia: {len(feature_sets['wellness_load'])}")
    print(f"Dni intensywne w zbiorze: {audit['intense_day_rows']}")
    return audit


def main():
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder źródłowy z wellness i training-load.")
    parser.add_argument("--output-dir", type=Path, default=SOCCERMON_DIR, help="Folder wynikowy (domyślnie: data/processed/soccermon).")
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
