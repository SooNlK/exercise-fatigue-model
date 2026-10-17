from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.app.schemas.daily_log import DailyLogCreate, DailyLogResponse
from src.app.services.daily_log_service import DailyLogService
from src.app.db.database import get_db

router = APIRouter(prefix="/users", tags=["daily-log"])

@router.put("/{user_id}/daily-log", response_model=DailyLogResponse)
def add_or_update_daily_log(user_id: str, entry: DailyLogCreate, response: Response, db: Session = Depends(get_db)):
    service = DailyLogService(db)

    try:
        daily_log, created = service.create_or_update(user_id=user_id, entry=entry)
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Nie udało się zapisać danych")

    response.status_code = 201 if created else 200

    return daily_log