"""
Minimalny backend Flask serwujący predykcję next-day wellness.
=================================================================

Realizuje wymagania funkcjonalne z pracy:
- WF-01 / WF-02: przyjmowanie codziennych danych treningowych i wellness
- WF-03: predykcja na podstawie modelu ML
- WF-04: wyświetlanie wyniku wraz z opisową interpretacją

WAŻNE: logika liczenia cech (lagi, acute/chronic load, ACWR, composite_wellness,
rhr_relative) jest importowana BEZPOŚREDNIO z src/features.py, żeby
backend liczył cechy dokładnie tak samo, jak podczas treningu modelu. Osobna,
zduplikowana implementacja tej logiki w dwóch miejscach byłaby źródłem
trudnych do wykrycia błędów (rozjazd między danymi treningowymi a produkcyjnymi).

To jest PROTOTYP: historia użytkowników trzymana w pamięci procesu (słownik),
zgodnie z zakresem pracy inżynierskiej. W wersji docelowej zgodnie z sekcją 2.4
("warstwa danych") powinna to być trwała baza danych (np. SQLite/PostgreSQL).

Uruchomienie:
    python app.py
Przykładowe użycie (curl):
    curl -X POST http://localhost:5001/users/ala/daily-log \
         -H "Content-Type: application/json" \
         -d '{"date":"2026-09-10","readiness":6,"fatigue":2,"soreness":1,
              "sleep_duration_h":7.5,"sleep_quality":4,"daily_srpe":350,
              "resting_heart_rate":54.2}'

    curl http://localhost:5001/users/ala/predict
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from src.features import (
    add_composite_wellness,
    add_ewma_load_features,
    add_lag_features,
    add_relative_rhr,
    add_rolling_load_features,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
FEATURE_COLS = artifact["feature_cols"]
THRESHOLDS = artifact["thresholds"]
PREDICT_DELTA = artifact.get("predict_delta", False)
LEVEL_COL = artifact.get("level_col", "composite_wellness")

app = Flask(__name__)

# "Baza danych" w pamięci - patrz uwaga w docstringu modułu.
USER_HISTORY: dict[str, list[dict]] = {}

REQUIRED_DAILY_FIELDS = [
    "date", "readiness", "fatigue", "soreness",
    "sleep_duration_h", "sleep_quality",
]

MIN_DAYS_FOR_PREDICTION = 9  # 5 dni do średniej z-score + 3 dni na lagi rhr_relative_lag1..3 + dzień bieżący


@app.route("/users/<user_id>/daily-log", methods=["POST"])
def add_daily_log(user_id):
    """Zapisuje codzienny wpis użytkownika (WF-01 + WF-02)."""
    payload = request.get_json(force=True)

    missing = [f for f in REQUIRED_DAILY_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Brak wymaganych pól: {missing}"}), 400

    payload.setdefault("daily_srpe", np.nan)
    payload.setdefault("resting_heart_rate", np.nan)

    USER_HISTORY.setdefault(user_id, []).append(payload)
    return jsonify({"status": "ok", "n_zapisanych_dni": len(USER_HISTORY[user_id])})


def build_today_features(user_id: str) -> pd.DataFrame:
    """
    Odtwarza inżynierię cech z data_preparation.py dla pojedynczego
    użytkownika, zwracając wiersz cech dla najnowszego zapisanego dnia
    (traktowanego jako "dziś", z którego przewidujemy "jutro").
    """
    records = USER_HISTORY.get(user_id, [])
    if not records:
        raise ValueError("Brak historii dla tego użytkownika - zaloguj co najmniej jeden dzień.")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["participant"] = user_id
    df = df.sort_values("date").reset_index(drop=True)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

    df = add_composite_wellness(df)
    df = add_relative_rhr(df)
    df = add_lag_features(
        df,
        cols=["daily_srpe", "readiness", "fatigue", "soreness", "sleep_duration_h", "sleep_quality", "rhr_relative"],
        lags=[1, 2, 3],
    )
    df = add_rolling_load_features(df, load_col="daily_srpe")
    df = add_ewma_load_features(df, load_col="daily_srpe")

    return df.iloc[[-1]]  # ostatni zapisany dzień


def categorize(prediction: float) -> str:
    """Mapuje ciągły wynik na opisową kategorię zgodnie z WF-04."""
    if prediction < THRESHOLDS["dluga_regeneracja_ponizej"]:
        return "długa regeneracja potrzebna"
    if prediction > THRESHOLDS["krotka_regeneracja_powyzej"]:
        return "krótka regeneracja / gotowy do treningu"
    return "umiarkowana regeneracja"


@app.route("/users/<user_id>/predict", methods=["GET"])
def predict(user_id):
    """Zwraca predykcję samopoczucia na jutro (WF-03) + interpretację (WF-04)."""
    try:
        today_row = build_today_features(user_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    n_days = len(USER_HISTORY[user_id])
    if n_days < MIN_DAYS_FOR_PREDICTION:
        return jsonify({
            "status": "zbieranie_danych",
            "message": (
                f"Zebrano {n_days} dni danych. Pełna predykcja wymaga min. "
                f"{MIN_DAYS_FOR_PREDICTION} dni historii (problem \"zimnego startu\" - "
                "patrz ograniczenia systemu w pracy)."
            ),
        }), 200

    missing_features = [c for c in FEATURE_COLS if c not in today_row.columns]
    if missing_features:
        return jsonify({"error": f"Brakujące cechy w danych: {missing_features}"}), 500

    X = today_row[FEATURE_COLS]
    if X.isna().any(axis=1).iloc[0]:
        return jsonify({
            "status": "niepelne_dane",
            "message": "Część wymaganych cech ma braki (np. brak RHR z ostatnich dni).",
        }), 200

    raw_prediction = float(MODEL.predict(X)[0])

    if PREDICT_DELTA:
        today_level = float(today_row[LEVEL_COL].iloc[0])
        predicted_level = today_level + raw_prediction
        response = {
            "user_id": user_id,
            "predicted_delta": round(raw_prediction, 3),
            "today_level": round(today_level, 3),
            "predicted_level_tomorrow": round(predicted_level, 3),
            "category": categorize(predicted_level),
            "n_days_history": n_days,
        }
    else:
        predicted_level = raw_prediction
        response = {
            "user_id": user_id,
            "predicted_composite_wellness": round(predicted_level, 3),
            "category": categorize(predicted_level),
            "n_days_history": n_days,
        }

    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Uruchamianie aplikacji na porcie {port}...")
    app.run(debug=True, port=port)