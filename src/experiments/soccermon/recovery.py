"""Trening i ocena prognozy powrotu do poziomu zmęczenia sprzed wysiłku."""

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.common import SOCCERMON_DIR
from src.experiments.common import SOCCERMON_ARTIFACTS, print_outputs


FEATURES = [
    "fatigue_gap",
    "event_srpe",
    "daily_srpe",
    "soreness",
    "sleep_duration",
]
TARGET = "recovered_day2"
CUTOFF = "2021-07-01"


def main():
    data = pd.read_csv(
        SOCCERMON_DIR / "recovery_dataset.csv",
        parse_dates=["event_date", "prediction_date", "target_date"],
    )

    boundary = pd.Timestamp(CUTOFF)

    # Wszystkie wyniki treningowe muszą być znane przed granicą testu.
    train = data.loc[data["target_date"] < boundary].copy()
    test = data.loc[data["prediction_date"] >= boundary].copy()

    if train.empty or test.empty:
        raise ValueError("Podział musi zawierać dane treningowe i testowe.")

    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError("Ten eksperyment wymaga obu klas w treningu i teście.")

    # Uzupełnianie braków i skalowanie są dopasowywane tylko na treningu.
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
    model.fit(train[FEATURES], train[TARGET])

    # Punkt odniesienia przypisuje każdemu odsetek powrotów z treningu.
    baseline_probability = float(train[TARGET].mean())

    probabilities = {
        "train_frequency": np.full(len(test), baseline_probability),
        "logistic_regression": model.predict_proba(test[FEATURES])[:, 1],
    }

    predictions = test[
        ["participant", "event_date", "prediction_date", "target_date", TARGET]
    ].copy()

    results = []
    for name, probability in probabilities.items():
        results.append({
            "model": name,
            "test_rows": len(test),
            "brier_score": brier_score_loss(test[TARGET], probability),
            "roc_auc": roc_auc_score(test[TARGET], probability),
        })
        predictions[name] = probability

    metrics = pd.DataFrame(results)
    output = SOCCERMON_ARTIFACTS / "recovery"
    output.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)

    # Zapis obejmuje pipeline oraz informacje potrzebne do jego użycia.
    artifact = {
        "model": model,
        "pipeline_version": "soccermon-recovery-v1",
        "feature_cols": FEATURES,
        "target": TARGET,
        "target_definition": "observed_fatigue_day2 >= baseline_fatigue",
        "prediction_time": "Koniec pierwszego dnia po intensywnym wysiłku",
        "horizon_days": 1,
        "evaluation_cutoff": CUTOFF,
        "training_rows": len(train),
        "training_max_target_date": str(train["target_date"].max().date()),
        "baseline_probability": baseline_probability,
    }
    joblib.dump(artifact, output / "logistic_recovery.joblib")

    print(f"Trening: {len(train)} epizodów")
    print(f"Test: {len(test)} epizodów")
    print(f"Częstość powrotów w treningu: {baseline_probability:.2%}")
    print(metrics.round(4).to_string(index=False))

    print_outputs(
        "Prognozowanie regeneracji",
        output,
        ["metrics.csv", "predictions.csv", "logistic_recovery.joblib"],
    )


if __name__ == "__main__":
    main()