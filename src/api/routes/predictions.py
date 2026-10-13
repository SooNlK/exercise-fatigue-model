import logging
import os

import joblib
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.repositories.daily_log_repository import DailyLogRepository
from src.schemas.prediction import PredictionResponse
from src.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
FEATURE_COLS = artifact["feature_cols"]
THRESHOLDS = artifact["thresholds"]

router = APIRouter(prefix="/users/", tags=["predictions"])

@router.get("/{user_id}/predict", response_model=PredictionResponse)
def predict(user_id: str, db: Session = Depends(get_db)):
    repository = DailyLogRepository(db)

    service = PredictionService(repository=repository, model=MODEL, feature_cols=FEATURE_COLS, thresholds=THRESHOLDS)

    try:
        return service.predict(user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception(f"Nie udało się wykonać predykcji dla usera {user_id}")
        raise HTTPException(status_code=500, detail="Nie udało się wykonać predykcji")