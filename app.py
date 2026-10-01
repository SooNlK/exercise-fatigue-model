from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import date

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

@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.post("/users/{user_id}/daily-log")
def add_daily_log(user_id: str, entry: DailyLogInput):
    return {"user_id": user_id, "entry": entry}