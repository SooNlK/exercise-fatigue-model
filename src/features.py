"""
Wspólna logika inżynierii cech - jedyna zależność między pipeline'em
treningowym a produkcyjną aplikacją.


WAŻNE: ten moduł nie zawiera żadnej logiki wczytywania/czyszczenia danych
PMData, diagnostyki ani budowy zbioru ML.
"""

import numpy as np
import pandas as pd


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
) -> pd.DataFrame:
    """
    Dodaje:
    - acute_load: suma obciążenia z ostatnich `acute_window` dni (włącznie z dniem t)
    - chronic_load: suma obciążenia z ostatnich `chronic_window` dni
    - acwr: acute:chronic workload ratio (znormalizowany do tygodni)
    """
    df = df.copy().sort_values(["participant", "date"])
    load_filled = df.groupby("participant")[load_col].transform(lambda s: s.fillna(0))

    df["acute_load"] = (
        load_filled.groupby(df["participant"])
        .transform(lambda s: s.rolling(window=acute_window, min_periods=1).sum())
    )
    df["chronic_load"] = (
        load_filled.groupby(df["participant"])
        .transform(lambda s: s.rolling(window=chronic_window, min_periods=1).sum())
    )

    chronic_weekly_avg = df["chronic_load"] / (chronic_window / acute_window)
    df["acwr"] = df["acute_load"] / chronic_weekly_avg.replace(0, np.nan)
    return df


def add_ewma_load_features(
    df: pd.DataFrame,
    load_col: str = "daily_srpe",
    acute_span: int = 7,
    chronic_span: int = 28,
) -> pd.DataFrame:
    """
    Dodaje EWMA-ACWR (wykładniczo ważona wersja acute:chronic workload ratio).

    Klasyczny ACWR liczony jako prosta suma/średnia z okna ma znaną wadę:
    przypisuje taką samą wagę treningowi sprzed 7 dni co wczorajszemu. EWMA
    naturalnie "zapomina" starsze obciążenie wykładniczo, lepiej oddając
    kumulację i wygasanie zmęczenia w czasie (Banister i in.).
    """
    df = df.copy().sort_values(["participant", "date"])
    load_filled = df.groupby("participant")[load_col].transform(lambda s: s.fillna(0))

    df["ewma_acute_load"] = load_filled.groupby(df["participant"]).transform(
        lambda s: s.ewm(span=acute_span, adjust=False).mean()
    )
    df["ewma_chronic_load"] = load_filled.groupby(df["participant"]).transform(
        lambda s: s.ewm(span=chronic_span, adjust=False).mean()
    )
    df["ewma_acwr"] = df["ewma_acute_load"] / df["ewma_chronic_load"].replace(0, np.nan)
    return df