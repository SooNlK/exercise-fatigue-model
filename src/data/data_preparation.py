"""
Przygotowanie danych PMData pod predykcję samopoczucia na kolejny dzień
========================================================================

Pipeline:
1. Wczytuje i scala wellness.csv + srpe.csv dla wszystkich uczestników.
2. Uzupełnia pełny kalendarz dni, wykrywa i klasyfikuje braki danych.
3. Interpoluje krótkie luki (<=2 dni), oznacza długie luki (>=3 dni) do wykluczenia.
4. Liczy złożony wskaźnik samopoczucia (composite_wellness).
5. Buduje cechy oparte na historii obciążeń treningowych (lagi, sumy kroczące,
   acute:chronic workload ratio) oraz cechy z samego kwestionariusza wellness.
6. Wyznacza zmienną celu: wartość wskaźnika samopoczucia w dniu t+horizon.
7. Zapisuje gotowy zbiór do ml_dataset.csv, gotowy pod trenowanie modeli
   (z zachowaniem kolumny 'participant' do grupowej walidacji krzyżowej).


Użycie:
    python data_preparation.py --data-dir /sciezka/do/pmdata --horizon 1
"""

import argparse
import glob
import json
import os


import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Wczytywanie danych
# ---------------------------------------------------------------------------

def load_resting_heart_rate(participant_dir: str) -> pd.DataFrame:
    """
    Wczytuje dzienne tętno spoczynkowe z fitbit/resting_heart_rate.json.
    Zwraca DataFrame z kolumnami: date, resting_heart_rate.
    """
    path = os.path.join(participant_dir, "fitbit", "resting_heart_rate.json")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "resting_heart_rate"])

    with open(path, "r") as f:
        raw = json.load(f)

    records = []
    for rec in raw:
        date_str = rec.get("dateTime", "")[:10]
        rhr = rec.get("value", {}).get("value")
        if date_str and rhr is not None:
            records.append(
                {"date": pd.to_datetime(date_str).date(), "resting_heart_rate": rhr}
            )

    return pd.DataFrame(records)


def load_sleep_score(participant_dir: str) -> pd.DataFrame:
    """
    Wczytuje fitbit/sleep_score.csv - dzienny wynik snu z rozbiciem na
    komponenty: overall_score, głęboki sen (minuty), niespokojność snu.
    Timestamp reprezentuje moment przebudzenia (rano), więc data z niego
    odpowiada dniu, w którym powstał tego ranka raport wellness.
    """
    path = os.path.join(participant_dir, "fitbit", "sleep_score.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "sleep_overall_score", "sleep_deep_minutes", "sleep_restlessness"])

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.rename(columns={
        "overall_score": "sleep_overall_score",
        "deep_sleep_in_minutes": "sleep_deep_minutes",
        "restlessness": "sleep_restlessness",
    })
    # jeśli kilka wpisów w tym samym dniu (rzadkie), bierzemy ostatni
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df[["date", "sleep_overall_score", "sleep_deep_minutes", "sleep_restlessness"]]


def load_hr_zones(participant_dir: str) -> pd.DataFrame:
    """
    Wczytuje fitbit/time_in_heart_rate_zones.json - dzienna liczba minut
    spędzonych w poszczególnych strefach tętna. Sumujemy strefy 2 i 3
    (cardio + peak) jako obiektywną miarę intensywnego wysiłku danego dnia,
    niezależną od subiektywnego sRPE.
    """
    path = os.path.join(participant_dir, "fitbit", "time_in_heart_rate_zones.json")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "hr_zone_active_minutes"])

    with open(path, "r") as f:
        raw = json.load(f)

    records = []
    for rec in raw:
        date_str = rec.get("dateTime", "")[:10]
        zones = rec.get("value", {}).get("valuesInZones", {})
        active_minutes = zones.get("IN_DEFAULT_ZONE_2", 0) + zones.get("IN_DEFAULT_ZONE_3", 0)
        if date_str:
            records.append({"date": pd.to_datetime(date_str).date(), "hr_zone_active_minutes": active_minutes})

    return pd.DataFrame(records)


def load_participant_data(participant_dir: str) -> pd.DataFrame | None:
    """Wczytuje i scala wellness.csv oraz srpe.csv dla jednego uczestnika."""
    participant_id = os.path.basename(participant_dir.rstrip("/"))

    pmsys_dir = os.path.join(participant_dir, "pmsys")
    wellness_path = os.path.join(pmsys_dir, "wellness.csv")
    srpe_path = os.path.join(pmsys_dir, "srpe.csv")

    if not os.path.exists(wellness_path):
        print(f"[UWAGA] Brak pliku {wellness_path} dla {participant_id}, pomijam.")
        return None

    wellness = pd.read_csv(wellness_path)
    wellness["date"] = pd.to_datetime(wellness["effective_time_frame"]).dt.date
    wellness["participant"] = participant_id

    if os.path.exists(srpe_path):
        srpe = pd.read_csv(srpe_path)
        srpe["date"] = pd.to_datetime(srpe["end_date_time"]).dt.date
        # sRPE = odczuwany wysiłek (RPE) * czas trwania sesji w minutach
        srpe["session_srpe"] = srpe["perceived_exertion"] * srpe["duration_min"]
        srpe_daily = (
            srpe.groupby("date")
            .agg(daily_srpe=("session_srpe", "sum"), n_sessions=("session_srpe", "count"))
            .reset_index()
        )
        merged = wellness.merge(srpe_daily, on="date", how="left")
    else:
        merged = wellness.copy()
        merged["daily_srpe"] = np.nan
        merged["n_sessions"] = 0

    rhr = load_resting_heart_rate(participant_dir)
    if not rhr.empty:
        merged = merged.merge(rhr, on="date", how="left")
    else:
        merged["resting_heart_rate"] = np.nan

    sleep_score = load_sleep_score(participant_dir)
    if not sleep_score.empty:
        merged = merged.merge(sleep_score, on="date", how="left")
    else:
        for col in ["sleep_overall_score", "sleep_deep_minutes", "sleep_restlessness"]:
            merged[col] = np.nan

    hr_zones = load_hr_zones(participant_dir)
    if not hr_zones.empty:
        merged = merged.merge(hr_zones, on="date", how="left")
    else:
        merged["hr_zone_active_minutes"] = np.nan

    return merged


def load_all_participants(data_dir: str) -> pd.DataFrame:
    participant_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "p*"))
        if os.path.isdir(d) and os.path.basename(d.rstrip("/"))[1:].isdigit()
    )
    if not participant_dirs:
        raise SystemExit(
            f"Nie znaleziono folderów uczestników (p01, p02, ...) w {data_dir}."
        )
    print(f"Znaleziono {len(participant_dirs)} folderów uczestników.")

    frames = [load_participant_data(d) for d in participant_dirs]
    frames = [f for f in frames if f is not None]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"Wczytano {len(df)} wierszy (dni z raportem) przed uzupełnieniem kalendarza.")
    return df


# ---------------------------------------------------------------------------
# 2-3. Kalendarz, braki, interpolacja
# ---------------------------------------------------------------------------

def build_full_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnia pełny zakres dat per uczestnik, żeby braki były widoczne jako NaN."""
    filled_frames = []
    for participant, group in df.groupby("participant"):
        group = group.sort_values("date")
        full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D").date
        full_df = pd.DataFrame({"date": full_range, "participant": participant})
        merged = full_df.merge(group, on=["date", "participant"], how="left")
        filled_frames.append(merged)
    return pd.concat(filled_frames, ignore_index=True)


def compute_gap_lengths(df: pd.DataFrame, value_col: str = "readiness") -> pd.DataFrame:
    """Liczy długości kolejnych luk (ciągów brakujących dni) per uczestnik."""
    gap_records = []
    for participant, group in df.groupby("participant"):
        group = group.sort_values("date").reset_index(drop=True)
        is_missing = group[value_col].isna()

        gap_length = 0
        gap_start_idx = None
        for idx, missing in enumerate(is_missing):
            if missing:
                if gap_length == 0:
                    gap_start_idx = idx
                gap_length += 1
            else:
                if gap_length > 0:
                    gap_records.append(
                        {
                            "participant": participant,
                            "gap_length_days": gap_length,
                            "gap_start_date": group.loc[gap_start_idx, "date"],
                        }
                    )
                gap_length = 0
        if gap_length > 0:
            gap_records.append(
                {
                    "participant": participant,
                    "gap_length_days": gap_length,
                    "gap_start_date": group.loc[gap_start_idx, "date"],
                }
            )

    return pd.DataFrame(gap_records)


def mark_long_gaps(df: pd.DataFrame, gaps: pd.DataFrame, max_short_gap: int = 2) -> pd.DataFrame:
    """Dodaje kolumnę 'in_long_gap' - True dla dni należących do luki > max_short_gap dni."""
    df = df.copy()
    df["in_long_gap"] = False
    long_gaps = gaps[gaps["gap_length_days"] > max_short_gap]

    for _, row in long_gaps.iterrows():
        gap_dates = pd.date_range(
            row["gap_start_date"], periods=int(row["gap_length_days"])
        ).date
        mask = (df["participant"] == row["participant"]) & (df["date"].isin(gap_dates))
        df.loc[mask, "in_long_gap"] = True

    return df


def interpolate_short_gaps(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Liniowa interpolacja braków, osobno per uczestnik."""
    df = df.copy().sort_values(["participant", "date"])
    for col in value_cols:
        df[col] = df.groupby("participant")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
    return df


def filter_low_quality_participants(
    df: pd.DataFrame, value_col: str = "readiness", max_missing_pct: float = 25.0
) -> pd.DataFrame:
    """Usuwa uczestników z procentem braków w value_col powyżej progu."""
    missing_pct = df.groupby("participant")[value_col].apply(lambda s: s.isna().mean() * 100)
    keep = missing_pct[missing_pct <= max_missing_pct].index
    dropped = sorted(set(df["participant"]) - set(keep))
    if dropped:
        print(
            f"Wykluczam {len(dropped)} uczestników z brakami > {max_missing_pct}%: {dropped}"
        )
    return df[df["participant"].isin(keep)].copy()


# ---------------------------------------------------------------------------
# 4. Złożony wskaźnik samopoczucia
# ---------------------------------------------------------------------------

from src.features import (
    add_composite_wellness,
    add_ewma_load_features,
    add_lag_features,
    add_relative_rhr,
    add_rolling_load_features,
)


# ---------------------------------------------------------------------------
# 5. Inżynieria cech
# ---------------------------------------------------------------------------
# UWAGA: add_lag_features, add_rolling_load_features, add_ewma_load_features,
# add_composite_wellness, add_relative_rhr są zaimportowane z src/features.py -
# to WSPÓLNY moduł używany też przez app.py, żeby backend liczył cechy
# dokładnie tak samo jak podczas treningu (patrz docstring src/features.py).


def add_target_column(
    df: pd.DataFrame, target_col: str, horizon: int = 1, predict_delta: bool = False
) -> pd.DataFrame:
    """
    Dodaje kolumnę 'target'.
    - predict_delta=False (domyślnie): target = wartość target_col za `horizon` dni (poziom bezwzględny)
    - predict_delta=True: target = ZMIANA względem dnia bieżącego,
      tzn. target_col(t+horizon) - target_col(t). Wymusza to na modelu
      przewidywanie wpływu dzisiejszego bodźca (trening/sen), zamiast
      głównie odtwarzać dzisiejszy poziom (autoregresję).
    Dodaje też 'target_in_long_gap' - True, jeśli dzień docelowy (t+horizon)
    należy do długiej luki (wtedy target jest niewiarygodny mimo interpolacji).
    """
    df = df.copy().sort_values(["participant", "date"])
    future_val = df.groupby("participant")[target_col].shift(-horizon)
    if predict_delta:
        df["target"] = future_val - df[target_col]
    else:
        df["target"] = future_val
    df["target_in_long_gap"] = df.groupby("participant")["in_long_gap"].shift(-horizon)
    return df


def build_ml_dataset(
    df: pd.DataFrame,
    target_col: str = "composite_wellness",
    horizon: int = 1,
    lag_cols: list[str] | None = None,
    lags: list[int] | None = None,
    predict_delta: bool = False,
) -> pd.DataFrame:
    """Składa wszystkie kroki inżynierii cech w jeden gotowy zbiór pod ML."""
    if lag_cols is None:
        lag_cols = [
            "daily_srpe", "readiness", "fatigue", "soreness",
            "sleep_duration_h", "sleep_quality", "rhr_relative",
            "sleep_overall_score", "sleep_deep_minutes", "hr_zone_active_minutes",
        ]
    if lags is None:
        lags = [1, 2, 3]

    df = df.sort_values(["participant", "date"]).reset_index(drop=True)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

    df = add_lag_features(df, cols=lag_cols, lags=lags)
    df = add_rolling_load_features(df, load_col="daily_srpe")
    df = add_ewma_load_features(df, load_col="daily_srpe")
    df = add_target_column(df, target_col=target_col, horizon=horizon, predict_delta=predict_delta)

    feature_cols = (
        [f"{col}_lag{lag}" for col in lag_cols for lag in lags]
        + ["acute_load", "chronic_load", "acwr"]
        + ["ewma_acute_load", "ewma_chronic_load", "ewma_acwr"]
        + ["day_of_week"]
        + [target_col]  # dzisiejszy poziom samopoczucia jako cecha wejściowa
    )

    keep_cols = ["participant", "date", "target"] + feature_cols
    dataset = df[keep_cols].copy()

    # wykluczamy wiersze, gdzie:
    # - dzień docelowy (t+horizon) należy do długiej luki -> target niewiarygodny
    # - dzień bieżący (t) należy do długiej luki -> cechy niewiarygodne
    # - target jest NaN (koniec serii danego uczestnika)
    n_before = len(dataset)
    dataset = dataset[~df["target_in_long_gap"].fillna(True).astype(bool)]
    dataset = dataset[~df.loc[dataset.index, "in_long_gap"].astype(bool)]
    dataset = dataset.dropna(subset=["target"])
    n_after = len(dataset)

    print(
        f"\nZbiór ML: {n_after} wierszy (z {n_before} przed filtrowaniem długich luk / braków targetu)."
    )

    # ile braków zostaje w cechach lagowych (na początku serii każdego uczestnika
    # pierwsze `max(lags)` dni nie mają pełnej historii)
    n_incomplete_features = dataset[feature_cols].isna().any(axis=1).sum()
    print(
        f"Wiersze z niekompletnymi cechami (początek serii uczestnika, brak historii): "
        f"{n_incomplete_features} ({n_incomplete_features / max(n_after, 1) * 100:.1f}%)"
    )
    print(
        "Te wiersze można albo usunąć, albo zaimputować (np. 0 dla obciążenia, "
        "mediana dla wellness) - decyzja zależy od tego, ile ich zostanie."
    )

    return dataset


def summarize_ml_dataset(dataset: pd.DataFrame) -> None:
    print("\n=== Podsumowanie zbioru ML (predykcja next-day wellness) ===\n")
    print(f"Liczba wierszy: {len(dataset)}")
    print(f"Liczba uczestników: {dataset['participant'].nunique()}")
    print("\nLiczba wierszy per uczestnik:")
    print(dataset["participant"].value_counts().sort_index().to_string())
    print("\nStatystyki zmiennej celu (target):")
    print(dataset["target"].describe().round(3).to_string())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def diagnostic_trajectory(
    df: pd.DataFrame,
    target_col: str,
    load_col: str = "daily_srpe",
    load_percentile: float = 75.0,
    days_before: int = 3,
    days_after: int = 7,
) -> None:
    """
    Dla wszystkich dni intensywnego treningu, wyrównuje w czasie wartości
    target_col od (event - days_before) do (event + days_after), centruje
    względem własnej średniej każdego uczestnika i uśrednia po wszystkich
    epizodach. Pozwala ocenić, czy dany wskaźnik w ogóle reaguje na
    obciążenie treningowe, zanim zaufamy mu jako cesze/targetowi.
    """
    offsets = range(-days_before, days_after + 1)
    trajectory = {offset: [] for offset in offsets}

    for participant, group in df.groupby("participant"):
        group = group.sort_values("date").reset_index(drop=True)
        loads = group[load_col].dropna()
        loads = loads[loads > 0]
        if len(loads) < 5:
            continue
        load_threshold = np.percentile(loads, load_percentile)
        participant_mean = group[target_col].mean()
        if pd.isna(participant_mean):
            continue

        n = len(group)
        for i in range(n):
            row = group.iloc[i]
            if pd.notna(row[load_col]) and row[load_col] >= load_threshold:
                for offset in offsets:
                    j = i + offset
                    if 0 <= j < n:
                        val = group.iloc[j][target_col]
                        if pd.notna(val):
                            trajectory[offset].append(val - participant_mean)

    print(f"\n=== Diagnostyka trajektorii: {target_col} wokół intensywnego treningu ===")
    print("(dzień 0 = dzień intensywnego treningu, wartości scentrowane wzgl. średniej uczestnika)\n")
    for offset in offsets:
        values = trajectory[offset]
        if values:
            label = f"{offset:+d}" if offset != 0 else "0 (trening)"
            print(f"dzień {label:>12}: średnia = {np.mean(values):>8.3f}  (n={len(values)})")


def main():
    parser = argparse.ArgumentParser(
        description="Przygotowanie danych PMData pod predykcję next-day wellness"
    )
    parser.add_argument("--data-dir", required=True, help="Ścieżka do katalogu głównego PMData")
    parser.add_argument(
        "--target-col",
        default="composite_wellness",
        choices=["composite_wellness", "readiness"],
        help="Zmienna, którą przewidujemy na dzień t+horizon (domyślnie: composite_wellness)",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Ile dni w przód przewidujemy (domyślnie: 1)")
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=100.0,
        help=(
            "Próg wykluczenia CAŁEGO uczestnika ze względu na procent braków w readiness "
            "(domyślnie: 100 = wyłączone). Maskowanie długich luk (in_long_gap) i tak "
            "wyklucza niewiarygodne DNI niezależnie od tego progu, więc domyślnie "
            "pozwalamy zachować dobre odcinki danych nawet u uczestników z wysokim "
            "ogólnym procentem braków, zamiast wyrzucać ich w całości."
        ),
    )
    parser.add_argument(
        "--predict-delta",
        action="store_true",
        help="Przewiduj ZMIANĘ (t+horizon minus t) zamiast poziomu bezwzględnego targetu",
    )
    parser.add_argument(
        "--output",
        default="ml_dataset.csv",
        help="Nazwa pliku wyjściowego (domyślnie: ml_dataset.csv) - zmień przy porównywaniu wariantów targetu",
    )
    args = parser.parse_args()

    df = load_all_participants(args.data_dir)
    df_full = build_full_calendar(df)
    print(f"Po uzupełnieniu pełnym kalendarzem: {len(df_full)} wierszy.")

    df_full = filter_low_quality_participants(
        df_full, value_col="readiness", max_missing_pct=args.max_missing_pct
    )

    gaps = compute_gap_lengths(df_full, value_col="readiness")
    df_marked = mark_long_gaps(df_full, gaps, max_short_gap=2)
    df_interp = interpolate_short_gaps(
        df_marked,
        value_cols=[
            "readiness", "fatigue", "soreness", "daily_srpe",
            "sleep_duration_h", "sleep_quality", "resting_heart_rate",
            "sleep_overall_score", "sleep_deep_minutes", "sleep_restlessness",
            "hr_zone_active_minutes",
        ],
    )
    df_interp = add_composite_wellness(df_interp)
    df_interp = add_relative_rhr(df_interp)

    if df_interp["resting_heart_rate"].notna().any():
        diagnostic_trajectory(df_interp, target_col="resting_heart_rate")
        print(
            "\nUwaga interpretacyjna: w przeciwieństwie do readiness, dla RHR "
            "'regeneracja' oznacza SPADEK z powrotem do normy (podwyższone RHR "
            "po treningu = oznaka niepełnej regeneracji), więc oczekiwany "
            "kierunek efektu jest odwrotny niż dla wskaźników subiektywnych."
        )
    else:
        print("\n[UWAGA] Brak danych resting_heart_rate - pomijam diagnostykę RHR.")

    for col, note in [
        ("sleep_overall_score", "wyższy wynik = lepszy sen; spadek po treningu oznaczałby zaburzenie snu przez wysiłek"),
        ("sleep_deep_minutes", "więcej głębokiego snu = zwykle lepsza regeneracja"),
        ("sleep_restlessness", "WYŻSZA niespokojność = GORSZY sen (kierunek jak dla RHR, nie jak dla readiness)"),
        ("hr_zone_active_minutes", "obiektywna miara intensywności dnia - oczekiwany szczyt w dniu 0 (definicja treningu), interesujące jest zachowanie PO"),
    ]:
        if df_interp[col].notna().any():
            diagnostic_trajectory(df_interp, target_col=col)
            print(f"Uwaga interpretacyjna ({col}): {note}")
        else:
            print(f"\n[UWAGA] Brak danych {col} - pomijam diagnostykę.")

    dataset = build_ml_dataset(
        df_interp,
        target_col=args.target_col,
        horizon=args.horizon,
        predict_delta=args.predict_delta,
    )
    summarize_ml_dataset(dataset)

    out_path = args.output
    dataset.to_csv(out_path, index=False)
    print(f"\nZapisano gotowy zbiór ML do: {out_path}")


if __name__ == "__main__":
    main()