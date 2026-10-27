from datetime import date, datetime
from uuid import uuid4
from sqlalchemy import String, Float, Date, DateTime, ForeignKey, func, UniqueConstraint, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.app.db.database import Base


class DailyLog(Base):
    __tablename__ = 'daily_log'
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_user_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    date: Mapped[date] = mapped_column(Date)
    fatigue: Mapped[int] = mapped_column(Integer)
    readiness: Mapped[int | None] = mapped_column(Integer)
    soreness: Mapped[int | None] = mapped_column(Integer)
    sleep_duration: Mapped[float | None] = mapped_column(Float)
    sleep_quality: Mapped[int | None] = mapped_column(Integer)
    mood: Mapped[int | None] = mapped_column(Integer)
    stress: Mapped[int | None] = mapped_column(Integer)
    daily_srpe: Mapped[float | None] = mapped_column(Float, nullable=True)


class User(Base):
    __tablename__ = 'user'
    __table_args__ = (UniqueConstraint("email", name="uq_user_email"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[date] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)