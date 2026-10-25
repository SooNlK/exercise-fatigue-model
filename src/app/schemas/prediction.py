from datetime import date
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    user_id: str
    prediction_date: date
    target_date: date

    current_fatigue: int = Field(ge=1, le=5)
    deterioration_probability: float = Field(ge=0, le=1)

    model_id: str
    method: str
    probability_source: str

    training_rows_for_person: int
    known_person_with_20_training_pairs: bool

    intense_day: bool
    intensity_status: str
    target_definition: str