"""
Wspólna logika inżynierii cech - jedyna zależność między pipeline'em
treningowym a produkcyjną aplikacją.


WAŻNE: ten moduł nie zawiera żadnej logiki wczytywania/czyszczenia danych
PMData, diagnostyki ani budowy zbioru ML.
"""

import numpy as np
import pandas as pd


def validate_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Sortuje i sprawdza jeden rekord na osobę i dzień kalendarzowy."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["participant", "date"]).reset_index(drop=True)
    if df[["participant", "date"]].isna().any().any():
        raise ValueError("Osoba i data nie mogą być puste")
    if df.duplicated(["participant", "date"]).any():
        raise ValueError("Wymagany jeden rekord na osobę i datę")
    steps = df.groupby("participant")["date"].diff().dropna()
    if not steps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Przed przesuwaniem cech wymagany jest pełny kalendarz dzienny")
    return df


def add_relative_rhr(
    df: pd.DataFrame,
    rhr_col: str = "resting_heart_rate",
    window: int = 21,
    min_periods: int = 5,
) -> pd.DataFrame:
    """
    Dodaje 'rhr_relative' = odchylenie dzisiejszego RHR od kroczącej średniej
    uczestnika z ostatnich `window` dni sprzed dnia t (shift(1) przed rolling).

    Rolling + shift(1) gwarantuje, że baseline pochodzi wyłącznie z dni < t
    (bez wycieku danych z przyszłości). Surowe wartości RHR nie są
    porównywalne między zawodnikami, więc do modelu generalizującego na
    nowych sportowców używamy wartości względnej.
    """
    df = df.copy().sort_values(["participant", "date"])
    rolling_mean = df.groupby("participant")[rhr_col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean()
    )
    df["rhr_relative"] = df[rhr_col] - rolling_mean
    return df


def add_composite_wellness(
    df: pd.DataFrame,
    readiness_col: str = "readiness",
    fatigue_col: str = "fatigue",
    soreness_col: str = "soreness",
    window: int = 21,
    min_periods: int = 5,
) -> pd.DataFrame:
    """
    composite_wellness = zscore(readiness) - zscore(fatigue) - zscore(soreness).

    Z-score liczony względem kroczącej średniej/odchylenia z ostatnich
    `window` dni SPRZED dnia t (shift(1) przed rolling) - bez wycieku danych
    z przyszłości. Pierwsze `min_periods` dni serii każdego uczestnika nie
    mają zdefiniowanej wartości (brak wystarczającej historii).
    """
    df = df.copy().sort_values(["participant", "date"])

    def rolling_zscore(col: str) -> pd.Series:
        shifted = df.groupby("participant")[col].shift(1)
        rolling_mean = shifted.groupby(df["participant"]).transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).mean()
        )
        rolling_std = shifted.groupby(df["participant"]).transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).std()
        )
        rolling_std = rolling_std.replace(0, np.nan)
        return (df[col] - rolling_mean) / rolling_std

    df["z_readiness"] = rolling_zscore(readiness_col)
    df["z_fatigue"] = rolling_zscore(fatigue_col)
    df["z_soreness"] = rolling_zscore(soreness_col)
    df["composite_wellness"] = df["z_readiness"] - df["z_fatigue"] - df["z_soreness"]
    return df


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """Dodaje kolumny {col}_lag{k} = wartość col sprzed k dni, per uczestnik."""
    df = df.copy().sort_values(["participant", "date"])
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("participant")[col].shift(lag)
    return df


def add_rolling_load_features(
    df: pd.DataFrame,
    load_col: str = "daily_srpe",
    acute_window: int = 7,
    chronic_window: int = 21,
    missing_policy: str = "zero",
) -> pd.DataFrame:
    """
    Dodaje:
    - acute_load: suma obciążenia z ostatnich `acute_window` dni (włącznie z dniem t)
    - chronic_load: suma obciążenia z ostatnich `chronic_window` dni
    - acwr: acute:chronic workload ratio (znormalizowany do tygodni)

    missing_policy="preserve": NaN unieważnia sumę, dopóki jest w oknie.
    Domyślne "zero" zachowuje kontrakt starego modelu API; poprawiony pipeline
    przygotowania danych używa jawnie "preserve".
    """
    if missing_policy not in {"zero", "preserve"}:
        raise ValueError("Polityka braków missing_policy musi mieć wartość zero lub preserve")
    df = df.copy().sort_values(["participant", "date"])
    # Wariant zero zachowuje zgodność z dotychczasowym modelem API.
    # Przygotowanie danych używa preserve: nieznane obciążenie nie oznacza odpoczynku.
    load_filled = df[load_col].fillna(0) if missing_policy == "zero" else df[load_col]

    def observed_sum(series, window):
        rolling = series.rolling(window=window, min_periods=1)
        result = rolling.sum()
        if missing_policy == "preserve":
            available = pd.Series(1., index=series.index).rolling(window, min_periods=1).sum()
            result = result.where(rolling.count().eq(available))
        return result

    df["acute_load"] = (
        load_filled.groupby(df["participant"])
        .transform(lambda s: observed_sum(s, acute_window))
    )
    df["chronic_load"] = (
        load_filled.groupby(df["participant"])
        .transform(lambda s: observed_sum(s, chronic_window))
    )

    chronic_weekly_avg = df["chronic_load"] / (chronic_window / acute_window)
    df["acwr"] = df["acute_load"] / chronic_weekly_avg.replace(0, np.nan)
    return df


def add_ewma_load_features(
    df: pd.DataFrame,
    load_col: str = "daily_srpe",
    acute_span: int = 7,
    chronic_span: int = 28,
    missing_policy: str = "zero",
) -> pd.DataFrame:
    """
    Dodaje EWMA-ACWR (wykładniczo ważona wersja acute:chronic workload ratio).

    W odróżnieniu od prostej sumy lub średniej kroczącej EWMA nadaje starszym
    obciążeniom wykładniczo malejące wagi. Samo zastosowanie takiego ważenia
    nie potwierdza zgodności wskaźnika z rzeczywistym przebiegiem zmęczenia.

    missing_policy="preserve": brak obciążenia pozostaje NaN, a EWMA jest
    inicjalizowana ponownie po przerwie w obserwowanej historii.
    """
    if missing_policy not in {"zero", "preserve"}:
        raise ValueError("Polityka braków missing_policy musi mieć wartość zero lub preserve")
    df = df.copy().sort_values(["participant", "date"])
    load_filled = df[load_col].fillna(0) if missing_policy == "zero" else df[load_col]

    def ewma(series, span):
        if missing_policy == "zero":
            return series.ewm(span=span, adjust=False).mean()
        # Nieznany dzień przerywa historię obciążenia. Kolejny znany odcinek
        # rozpoczyna EWMA ponownie, zamiast przenosić stan przez lukę.
        segments = series.isna().cumsum()
        return series.groupby(segments).transform(
            lambda s: s.ewm(span=span, adjust=False).mean()
        ).where(series.notna())

    df["ewma_acute_load"] = load_filled.groupby(df["participant"]).transform(
        lambda s: ewma(s, acute_span)
    )
    df["ewma_chronic_load"] = load_filled.groupby(df["participant"]).transform(
        lambda s: ewma(s, chronic_span)
    )
    df["ewma_acwr"] = df["ewma_acute_load"] / df["ewma_chronic_load"].replace(0, np.nan)
    return df
