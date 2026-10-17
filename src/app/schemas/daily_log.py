from datetime import date
from pydantic import BaseModel, Field, ConfigDict

class DailyLogCreate(BaseModel):
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


class DailyLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    date: date

    readiness: float
    fatigue: float
    soreness: float
    sleep_duration_h: float
    sleep_quality: float

    daily_srpe: float | None
    resting_heart_rate: float | None
    sleep_overall_score: float | None
    sleep_deep_minutes: float | None
    hr_zone_active_minutes: float | None