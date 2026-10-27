from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.app.api.dependencies import get_current_user
from src.app.config import settings
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.exceptions import (
    InsufficientHistoryError,
    NoUserDataError,
    PredictionError,
)
from src.app.repositories.daily_log_repository import (
    DailyLogRepository,
)
from src.app.schemas.prediction import PredictionResponse
from src.app.services.prediction_service import PredictionService
from src.models.soccermon.predictor import load_risk_model


ARTIFACT = load_risk_model(settings.model_path)

router = APIRouter(prefix="/me", tags=["predictions"])


@router.get(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    prediction_date: date,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repository = DailyLogRepository(db)
    service = PredictionService(
        repository=repository,
        artifact=ARTIFACT,
    )

    try:
        return service.predict(
            user_id=current_user.id,
            prediction_date=prediction_date,
        )
    except NoUserDataError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except InsufficientHistoryError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except PredictionError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
