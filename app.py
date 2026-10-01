from datetime import date
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.database import  Base, engine, get_db
from src.db.models import DailyLog

class DailyLogInput(BaseModel):
    date: date
    readiness: float = Field(ge=0, le=10)
    fatigue: float = Field(ge=0, le=5)
    soreness: float = Field(ge=0, le=5)
    sleep_duration_h: float = Field(ge=0, le=24)
    sleep_quality: float = Field(ge=0, le=5)
    daily_srpe: float | None = Field(default=None, ge=0)
    resting_heart_rate: float | None = Field(default=None, ge=0)
    sleep_overall_score: float | None = Field(default=None, ge=0)
    sleep_deep_minutes: float | None = Field(default=None, ge=0)
    hr_zone_active_minutes: float | None = Field(default=None, ge=0)

app = FastAPI()

Base.metadata.create_all(engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/users/{user_id}/daily-log")
def add_daily_log(user_id: str, entry: DailyLogInput, db: Session = Depends(get_db)):
    daily_log = DailyLog(user_id=user_id, **entry.model_dump())
    db.add(daily_log)
    db.commit()
    db.refresh(daily_log)

    return daily_log