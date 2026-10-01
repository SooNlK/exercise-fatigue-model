from datetime import date
from sqlalchemy import String, Float, Date
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class DailyLog(Base):
    __tablename__ = 'daily_log'

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    date: Mapped[date] = mapped_column(Date)
    readiness: Mapped[float] = mapped_column(Float)
    fatigue: Mapped[float] = mapped_column(Float)
    soreness: Mapped[float] = mapped_column(Float)
    sleep_duration_h: Mapped[float] = mapped_column(Float)
    sleep_quality: Mapped[float] = mapped_column(Float)
    daily_srpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    resting_heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_deep_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    hr_zone_active_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
