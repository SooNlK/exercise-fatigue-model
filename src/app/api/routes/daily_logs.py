from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from datetime import date

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

@router.get(
    "/daily-logs",
    response_model=list[DailyLogResponse],
)
def get_daily_logs(date_from: date | None = None, date_to: date | None = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(
            status_code=422,
            detail="Data początkowa nie może być późniejsza od końcowej.",
        )

    service = DailyLogService(db)

    try:
        return service.get_history(
            user_id=current_user.id,
            date_from=date_from,
            date_to=date_to,
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Nie udało się pobrać historii raportów.",
        ) from exc