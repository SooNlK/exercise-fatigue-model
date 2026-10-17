from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.app.db.models import DailyLog


class DailyLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_user_and_date(self, user_id: str, log_date: date) -> DailyLog | None:
        stmt = select(DailyLog).where(DailyLog.user_id == user_id, DailyLog.date == log_date)

        return self.db.execute(stmt).scalar_one_or_none()

    def get_by_user(self, user_id: str) -> pd.DataFrame:
        stmt = select(DailyLog).where(DailyLog.user_id == user_id).order_by(DailyLog.date.asc())

        return pd.read_sql(stmt, self.db.bind)

    def add(self, daily_log: DailyLog) -> DailyLog:
        self.db.add(daily_log)
        return daily_log

    def refresh(self, daily_log: DailyLog) -> None:
        self.db.refresh(daily_log)