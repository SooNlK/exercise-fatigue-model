from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.experiments.preparation import WELLNESS, make_features
from src.models.soccermon.predictor import daily_date

FEATURES = [
    "fatigue_gap", "event_srpe", "daily_srpe", "soreness", "sleep_duration",
]


@lru_cache(maxsize=1)
def load_recovery_model(path):
    artifact = joblib.load(path)
    if (
        artifact.get("pipeline_version") != "soccermon-recovery-v1"
        or artifact.get("feature_cols") != FEATURES
        or artifact.get("horizon_days") != 1
        or artifact.get("target_definition")
        != "observed_fatigue_day2 >= baseline_fatigue"
        or list(artifact["model"].classes_) != [0, 1]
    ):
        raise RuntimeError("Niezgodny model regeneracji.")
    return artifact


def predict_recovery(model_path, history, user_id, prediction_date):
    """Korzysta z historii wcześniej sprawdzonej przez predict_risk."""
    day = daily_date(prediction_date)
    frame = history.reindex(columns=["date", *WELLNESS, "daily_srpe"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").reindex(
        pd.date_range(frame["date"].min(), day)
    ).rename_axis("date").reset_index()
    frame["participant"] = user_id
    featured, _ = make_features(frame)
    previous = featured.iloc[-2]
    current = featured.iloc[-1]

    result = {
        "event_date": (day - pd.Timedelta(days=1)).date().isoformat(),
        "status": "insufficient_data",
        "message": "Uzupełnij obciążenie z dnia poprzedzającego prognozę.",
        "baseline_fatigue": None,
        "recovery_probability": None,
    }
    if pd.isna(previous.daily_srpe):
        return result
    if pd.isna(previous.intense_threshold):
        result["message"] = (
            "Do oceny intensywności potrzeba co najmniej 7 dodatnich "
            "obciążeń z 28 dni poprzedzających dzień wysiłku."
        )
        return result
    if not previous.intense_day:
        result.update(
            status="not_applicable",
            message="Dzień poprzedzający prognozę był poniżej progu intensywności lub równy temu progowi.",
        )
        return result

    baseline_values = featured.loc[
        featured["date"].between(
            day - pd.Timedelta(days=4), day - pd.Timedelta(days=2)
        ),
        "fatigue",
    ].dropna()
    if len(baseline_values) < 2:
        result["message"] = "Potrzebne są 2 oceny zmęczenia z 3 dni przed wysiłkiem."
        return result

    baseline = float(baseline_values.mean())
    result["baseline_fatigue"] = baseline
    if current.fatigue >= baseline:
        result.update(
            status="already_at_baseline",
            message="Ocena zmęczenia osiągnęła już poziom sprzed wysiłku lub wyższy.",
        )
        return result

    artifact = load_recovery_model(str(model_path))
    if day < daily_date(artifact["training_max_target_date"]):
        result["status"] = "unavailable"
        result["message"] = "Ta data wymaga modelu wytrenowanego na wcześniejszych danych."
        return result

    values = pd.DataFrame([{
        "fatigue_gap": baseline - current.fatigue,
        "event_srpe": previous.daily_srpe,
        "daily_srpe": current.daily_srpe,
        "soreness": current.soreness,
        "sleep_duration": current.sleep_duration,
    }], columns=FEATURES)
    probability = float(artifact["model"].predict_proba(values)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Model zwrócił nieprawidłowe prawdopodobieństwo.")
    result.update(
        status="available",
        message="Szansa powrotu do poziomu zmęczenia sprzed wysiłku na kolejny dzień.",
        recovery_probability=probability,
    )
    return result
