import os
from datetime import date

import joblib
import pandas as pd
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.db.database import  Base, engine, get_db
from src.db.models import DailyLog
from src.features import (
    add_relative_rhr,
    add_rolling_load_features,
    add_lag_features,
    add_ewma_load_features,
    add_composite_wellness,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
FEATURE_COLS = artifact["feature_cols"]
THRESHOLDS = artifact["thresholds"]
PREDICT_DELTA = artifact.get("predict_delta", False)
LEVEL_COL = artifact.get("level_col", "composite_wellness")

app = FastAPI()

Base.metadata.create_all(engine)


class DailyLogInput(BaseModel):
    date: date
    readiness: float = Field(ge=0, le=10)
    fatigue: float = Field(ge=0, le=5)
    soreness: float = Field(ge=0, le=5)
    sleep_duration_h: float = Field(ge=0, le=24)
    sleep_quality: float = Field(ge=0, le=5)
    daily_srpe: float | None = Field(default=None, ge=0)
    resting_heart_rate: float | None = Field(default=None, ge=0)
    sleep_overall_score: float | None = Field(default=None, ge=0)
    sleep_deep_minutes: float | None = Field(default=None, ge=0)
    hr_zone_active_minutes: float | None = Field(default=None, ge=0)


def build_today_features(user_id: str, db: Session) -> pd.DataFrame:
    stmt = (select(DailyLog).where(DailyLog.user_id == user_id).order_by(DailyLog.date.asc()))
    df = pd.read_sql(stmt, db.bind)

    df["participant"] = df["user_id"]
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

    df = add_relative_rhr(df)
    df = add_lag_features(df, cols=["daily_srpe", "readiness", "fatigue", "soreness", "sleep_duration_h", "sleep_quality", "rhr_relative", "sleep_overall_score", "sleep_deep_minutes", "hr_zone_active_minutes"], lags=[1,2,3])
    df = add_rolling_load_features(df, load_col="daily_srpe")
    df = add_ewma_load_features(df, load_col="daily_srpe")
    df = add_composite_wellness(df)

    return df.iloc[[-1]]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/users/{user_id}/daily-log")
def add_daily_log(user_id: str, entry: DailyLogInput, db: Session = Depends(get_db)):
    daily_log = DailyLog(user_id=user_id, **entry.model_dump())
    db.add(daily_log)
    db.commit()
    db.refresh(daily_log)

    return daily_log

@app.get("/users/{user_id}/predict")
def predict(user_id: str, db: Session = Depends(get_db)):
    today_row = build_today_features(user_id=user_id, db=db)
    X = today_row[FEATURE_COLS]

    predicted_delta = float(MODEL.predict(X)[0])
    current_wellness = float(today_row["composite_wellness"].values[0])
    reconstructed_level = current_wellness + predicted_delta

    if reconstructed_level < THRESHOLDS["dluga_regeneracja_ponizej"]:
        recommendation = "DŁUGA_REGENERACJA"
    elif reconstructed_level > THRESHOLDS["krotka_regeneracja_powyzej"]:
        recommendation = "KROTKA_REGENERACJA"
    else:
        recommendation = "UMIARKOWANA_REGENERACJA"
    return {
        "user_id": user_id,
        "date": str(today_row["date"].values[0]),
        "current_wellness": round(current_wellness, 3),
        "predicted_delta": round(predicted_delta, 3),
        "predicted_wellness_tomorrow": round(reconstructed_level, 3),
        "recommendation": recommendation,
    }