from datetime import date
from sqlalchemy import String, Float, Date, UniqueConstraint, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.app.db.database import Base


class DailyLog(Base):
    __tablename__ = 'daily_log'
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_user_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    date: Mapped[date] = mapped_column(Date)
    fatigue: Mapped[int] = mapped_column(Integer)
    readiness: Mapped[int | None] = mapped_column(Integer)
    soreness: Mapped[int | None] = mapped_column(Integer)
    sleep_duration: Mapped[float | None] = mapped_column(Float)
    sleep_quality: Mapped[int | None] = mapped_column(Integer)
    mood: Mapped[int | None] = mapped_column(Integer)
    stress: Mapped[int | None] = mapped_column(Integer)
    daily_srpe: Mapped[float | None] = mapped_column(Float, nullable=True)
