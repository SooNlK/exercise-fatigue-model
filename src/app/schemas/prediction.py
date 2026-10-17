from datetime import date
from pydantic import BaseModel
from enum import Enum

class RecoveryRecommendation(str, Enum):
    LONG = "DŁUGA_REGENERACJA"
    MODERATE = "UMIARKOWANA_REGENERACJA"
    SHORT = "KROTKA_REGENERACJA"

class PredictionResponse(BaseModel):
    user_id: str
    date: date
    current_wellness: float
    predicted_delta: float
    predicted_wellness_tomorrow: float
    recommendation: RecoveryRecommendation