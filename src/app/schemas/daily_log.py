from datetime import date
from pydantic import BaseModel, Field, ConfigDict

class DailyLogCreate(BaseModel):
    date: date
    fatigue: int = Field(ge=1, le=5, description="Liczba całkowita 1-5; niżej = większe zmęczenie")

    soreness: int | None = Field(default=None, ge=1, le=5, description="Liczba całkowita 1-5")
    readiness: int | None = Field(default=None, ge=1, le=10, description="Liczba całkowita 1-10")
    sleep_duration: float | None = Field(default=None, ge=0, le=24, description="Liczba 0-24")
    sleep_quality: int | None = Field(default=None, ge=1, le=5 ,description="Liczba całkowita 1-5")
    mood: int | None = Field(default=None, ge=1, le=5, description="Liczba całkowita 1-5")
    stress: int | None = Field(default=None, ge=1, le=5, description="Liczba całkowita 1-5")
    daily_srpe: float | None = Field(default=None, ge=0)


class DailyLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    date: date
    fatigue: float

    readiness: float | None
    soreness: float | None
    sleep_duration: float | None
    sleep_quality: float | None
    mood: float | None
    stress: float | None
    daily_srpe: float | None