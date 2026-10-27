from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.app.schemas.daily_log import DailyLogCreate, DailyLogResponse
from src.app.services.daily_log_service import DailyLogService
from src.app.db.database import get_db
from src.app.api.dependencies import get_current_user
from src.app.db.models import User

router = APIRouter(prefix="/me", tags=["daily-log"])

@router.put("/daily-log", response_model=DailyLogResponse)
def add_or_update_daily_log(
    entry: DailyLogCreate,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = DailyLogService(db)

    try:
        daily_log, created = service.create_or_update(
            user_id=current_user.id,
            entry=entry,
        )
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Nie udało się zapisać danych")

    response.status_code = 201 if created else 200

    return daily_log