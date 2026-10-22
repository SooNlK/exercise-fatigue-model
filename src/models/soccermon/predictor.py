"""Generowanie prognozy ryzyka dla jednej osoby na koniec dnia.

1. Sprawdza wersję i definicję celu zapisanego modelu.
2. Odtwarza pełny kalendarz dostępnej historii, minimum 29 dni.
3. Tworzy cechy tą samą funkcją, co przygotowanie danych SoccerMon.
4. Zwraca ryzyko pogorszenia następnego dnia i informację o metodzie.

Historia musi obejmować początek tego okna; może zawierać jawne braki.
Brakujące dni i pomiary pozostają NaN. Nie przyjmujemy braku obciążenia za zero.
Wymagane jest dzisiejsze fatigue i 7 raportów fatigue z poprzednich 21 dni.
"""

from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.soccermon.preparation import WELLNESS, make_features
from src.models.soccermon.risk import PIPELINE_VERSION, TARGET_DEFINITION, risk_probabilities


def daily_date(value):
    """Daty raportów są datami kalendarzowymi bez godziny i strefy czasowej."""
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("Wymagana data kalendarzowa bez godziny i strefy czasowej.")
    return day


def prediction_row(history, user_id, prediction_date):
    """Przygotowuje pełny wiersz, w tym znacznik intensywnego dnia."""
    day = daily_date(prediction_date)
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Wymagany niepusty identyfikator osoby.")
    if history.empty or not {"date", "fatigue"}.issubset(history.columns):
        raise ValueError("Historia musi zawierać daty i fatigue.")
    if "participant" in history and not history.participant.eq(user_id).all():
        raise ValueError("Historia musi dotyczyć wyłącznie wskazanej osoby.")
    frame = history.reindex(columns=["date", *WELLNESS, "daily_srpe"]).copy()
    frame["date"] = pd.to_datetime(frame.date.map(daily_date))
    if frame.date.duplicated().any() or frame.date.gt(day).any():
        raise ValueError("Daty muszą być unikalne i nie mogą wykraczać poza dzień prognozy.")
    start = day - pd.Timedelta(days=28)
    if frame.date.min() > start:
        raise ValueError("Historia musi obejmować dzień prognozy i 28 wcześniejszych dni; nieznane pomiary mogą być puste.")
    if not frame.date.eq(day).any():
        raise ValueError("Brak raportu z dnia prognozy.")
    for column in [*WELLNESS, "daily_srpe"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].dropna()
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Nieskończone wartości w {column}.")
        if column in ["fatigue", "soreness", "sleep_quality", "mood", "stress"]:
            if not values.isin([1, 2, 3, 4, 5]).all():
                raise ValueError(f"Wymagana skala całkowita 1–5 dla {column}.")
        elif column == "readiness" and not values.between(1, 10).all():
            raise ValueError("Readiness musi być w zakresie 1–10.")
        elif column in ["daily_srpe", "sleep_duration"] and values.lt(0).any():
            raise ValueError(f"Ujemne wartości w {column}.")
    # Zachowujemy całą dostępną historię. Skrócenie jej może zmienić ostatnie
    # bity obliczeń rolling.std i przez to wybór gałęzi w drzewie.
    frame = frame.set_index("date").reindex(pd.date_range(frame.date.min(), day)).rename_axis("date").reset_index()
    frame["participant"] = user_id
    featured, sets = make_features(frame)
    row = featured.iloc[[-1]].copy()
    if row.fatigue.isna().any() or row.fatigue_history_count.iloc[0] < 7:
        raise ValueError("Wymagane fatigue dzisiaj i co najmniej 7 raportów z poprzednich 21 dni.")
    # Eksperyment uczył się na dataset.csv odczytanym domyślnym parserem pandas.
    # Odtwarzamy tę samą reprezentację liczb także dla cech liczonych na żywo.
    columns = sets["wellness_load"]
    parsed = pd.read_csv(StringIO(row[columns].to_csv(index=False)))
    row[columns] = parsed.to_numpy()
    return row, sets["wellness_load"]


def validate_artifact(artifact):
    """Odrzuca artefakty regresji i niezgodny format przetwarzania."""
    if (artifact.get("pipeline_version") != PIPELINE_VERSION
            or artifact.get("target_definition") != TARGET_DEFINITION
            or artifact.get("horizon_days") != 1):
        raise ValueError("Niezgodna wersja modelu lub definicja prognozy ryzyka.")
    if artifact.get("method") not in ["personal_intensity_frequency", "boosting_load"]:
        raise ValueError("Nieznana metoda prognozy ryzyka.")


def load_risk_model(path: str | Path):
    """Wczytuje lokalny artefakt zapisany przez train_risk."""
    artifact = joblib.load(path)
    validate_artifact(artifact)
    return artifact


def predict_risk(artifact, history, user_id, prediction_date):
    """Zwraca prawdopodobieństwo; nie określa czasu regeneracji ani zaleceń."""
    validate_artifact(artifact)
    day = daily_date(prediction_date)
    if day < daily_date(artifact["training_max_target_date"]):
        raise ValueError("Model korzysta z celów późniejszych niż dzień prognozy; użyj wcześniejszego treningu.")
    row, feature_cols = prediction_row(history, user_id, day)
    if artifact["feature_cols"] != feature_cols:
        raise ValueError("Cechy modelu są niezgodne z przygotowaniem danych.")
    probability, source = risk_probabilities(artifact, row)
    count = int(artifact["training_counts_by_participant"].get(user_id, 0))
    threshold = row.intense_threshold.iloc[0]
    return {
        "user_id": user_id,
        "prediction_date": day.date().isoformat(),
        "target_date": (day + pd.Timedelta(days=1)).date().isoformat(),
        "current_fatigue": int(row.fatigue.iloc[0]),
        "deterioration_probability": float(probability[0]),
        "model_id": artifact["model_id"],
        "method": artifact["method"],
        "probability_source": str(source[0]),
        "training_rows_for_person": count,
        "known_person_with_20_training_pairs": count >= 20,
        "intense_day": bool(row.intense_day.iloc[0]),
        "intensity_status": "available" if pd.notna(threshold) and pd.notna(row.daily_srpe.iloc[0]) else "unknown_treated_as_false",
        "target_definition": TARGET_DEFINITION,
    }
