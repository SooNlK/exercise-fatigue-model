import logging

import joblib
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.app.db.database import get_db
from src.app.repositories.daily_log_repository import DailyLogRepository
from src.app.schemas.prediction import PredictionResponse
from src.app.services.prediction_service import PredictionService
from src.app.config import settings
from src.app.exceptions import (
    InsufficientHistoryError,
    NoUserDataError,
    PredictionError,
)

logger = logging.getLogger(__name__)

artifact = joblib.load(settings.model_path)
MODEL = artifact["model"]
FEATURE_COLS = artifact["feature_cols"]
THRESHOLDS = artifact["thresholds"]

router = APIRouter(prefix="/users", tags=["predictions"])

@router.get("/{user_id}/predict", response_model=PredictionResponse)
def predict(user_id: str, db: Session = Depends(get_db)):
    repository = DailyLogRepository(db)

    service = PredictionService(repository=repository, model=MODEL, feature_cols=FEATURE_COLS, thresholds=THRESHOLDS)

    try:
        return service.predict(user_id=user_id)
    except NoUserDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))