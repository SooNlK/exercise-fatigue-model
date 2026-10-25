import logging
import pandas as pd

from datetime import date

from src.app.exceptions import (
    InsufficientHistoryError,
    NoUserDataError,
    PredictionError,
)
from src.models.soccermon.predictor import (
    predict_risk,
    validate_artifact,
)

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, repository, artifact):
        validate_artifact(artifact)
        self.artifact = artifact
        self.repository = repository

    def predict(self, user_id: str, prediction_date: date) -> dict:
        history = self.repository.get_by_user(user_id)

        if history.empty:
            raise NoUserDataError("Brak raportów dla danego użytkownika")

        try:
            history = history.copy()
            history["date"] = pd.to_datetime(history["date"], errors="raise")

            history = history.loc[history["date"].le(pd.Timestamp(prediction_date))].copy()

            if history.empty:
                raise NoUserDataError("Brak raportów dla wskazanej daty")

            if "user_id" in history:
                history["participant"] = history["user_id"]

            return predict_risk(
                artifact=self.artifact,
                history=history,
                user_id=user_id,
                prediction_date=prediction_date,
            )

        except NoUserDataError:
            raise

        except ValueError as ext:
            raise InsufficientHistoryError(str(ext)) from ext

        except Exception as exc:
            logger.exception(
                "Błąd prognozy ryzyka dla użytkownika %s",
                user_id,
            )
            raise PredictionError(
                "Nie udało się wykonać prognozy."
            ) from exc
