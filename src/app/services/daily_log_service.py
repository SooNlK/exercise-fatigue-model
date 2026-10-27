from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from datetime import date

from src.app.db.models import DailyLog
from src.app.repositories.daily_log_repository import DailyLogRepository
from src.app.schemas.daily_log import DailyLogCreate

class DailyLogService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = DailyLogRepository(db)

    def create_or_update(self, user_id: str, entry: DailyLogCreate) -> tuple[DailyLog, bool]:
        existing_log = self.repository.get_by_user_and_date(user_id, entry.date)

        try:
            if existing_log:
                data = entry.model_dump()

                for field, value in data.items():
                    setattr(existing_log, field, value)

                daily_log = existing_log
                created = False
            else:
                daily_log = DailyLog(user_id=user_id, **entry.model_dump())
                self.repository.add(daily_log)
                created = True

            self.db.commit()
            self.repository.refresh(daily_log)

            return daily_log, created

        except SQLAlchemyError:
            self.db.rollback()
            raise

    def get_history(self, user_id: str, date_from: date | None = None, date_to: date | None = None) -> list[DailyLog]:
        return self.repository.get_history(
            user_id=user_id,
            date_from=date_from,
            date_to=date_to,
        )