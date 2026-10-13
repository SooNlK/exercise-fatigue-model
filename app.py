import os

import joblib
import logging
from fastapi import FastAPI, Depends, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.repositories.daily_log_repository import DailyLogRepository
from src.schemas.daily_log import DailyLogCreate, DailyLogResponse
from src.schemas.prediction import PredictionResponse
from src.db.database import  Base, engine, get_db
from src.services.daily_log_service import DailyLogService
from src.services.prediction_service import PredictionService

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
FEATURE_COLS = artifact["feature_cols"]
THRESHOLDS = artifact["thresholds"]
PREDICT_DELTA = artifact.get("predict_delta", False)
LEVEL_COL = artifact.get("level_col", "composite_wellness")
logger = logging.getLogger(__name__)

app = FastAPI()

Base.metadata.create_all(engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.put("/users/{user_id}/daily-log", response_model=DailyLogResponse)
def add_or_update_daily_log(user_id: str, entry: DailyLogCreate, response: Response, db: Session = Depends(get_db)):
    service = DailyLogService(db)

    try:
        daily_log, created = service.create_or_update(user_id=user_id, entry=entry)
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Nie udało się zapisać danych")

    response.status_code = 201 if created else 200

    return daily_log


@app.get("/users/{user_id}/predict", response_model=PredictionResponse)
def predict(user_id: str, db: Session = Depends(get_db)):
    repository = DailyLogRepository(db)

    service = PredictionService(repository=repository, model=MODEL, feature_cols=FEATURE_COLS, thresholds=THRESHOLDS)

    return service.predict(user_id=user_id)
