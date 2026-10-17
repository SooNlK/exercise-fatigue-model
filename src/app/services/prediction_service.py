import logging
import pandas as pd

from src.app.repositories.daily_log_repository import DailyLogRepository
from src.data.pmdata.features import (
    add_relative_rhr,
    add_rolling_load_features,
    add_lag_features,
    add_ewma_load_features,
    add_composite_wellness
)
from src.app.exceptions import (
    InsufficientHistoryError,
    NoUserDataError,
    PredictionError,
)

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, repository: DailyLogRepository, model, feature_cols: list[str], thresholds: dict ):
        self.repository = repository
        self.model = model
        self.feature_cols = feature_cols
        self.thresholds = thresholds

    def _build_today_features(self, user_id: str) -> pd.DataFrame:
        df = self.repository.get_by_user(user_id)

        if df.empty:
            raise NoUserDataError("Brak wpisów dla tego użytkownika")

        df["participant"] = df["user_id"]
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

        df = add_relative_rhr(df)
        df = add_lag_features(df, cols=["daily_srpe", "readiness", "fatigue", "soreness", "sleep_duration_h",
                                        "sleep_quality", "rhr_relative", "sleep_overall_score", "sleep_deep_minutes",
                                        "hr_zone_active_minutes"], lags=[1, 2, 3])
        df = add_rolling_load_features(df, load_col="daily_srpe")
        df = add_ewma_load_features(df, load_col="daily_srpe")
        df = add_composite_wellness(df)

        return df.iloc[[-1]]


    def _get_recommendation(self, wellness: float) -> str:
        if wellness < self.thresholds["dluga_regeneracja_ponizej"]:
            return "DŁUGA_REGENERACJA"

        if wellness > self.thresholds["krotka_regeneracja_powyzej"]:
            return "KROTKA_REGENERACJA"

        return "UMIARKOWANA_REGENERACJA"


    def predict(self, user_id: str):
        today_row = self._build_today_features(user_id)

        X = today_row[self.feature_cols]

        missing_cols = X.columns[X.isna().any()].tolist()

        if missing_cols:
            raise InsufficientHistoryError(f"Za mało danych historycznych do wykonania predykcji. missing features: {missing_cols}")

        try:
            predicted_delta = float(self.model.predict(X)[0])
        except Exception:
            logger.exception(f"Błąd podczas wykonania predykcji dla usera {user_id}")
            raise PredictionError("Nie udało się wykonać predykcji")

        current_wellness = float(today_row["composite_wellness"].values[0])
        reconstructed_level = current_wellness + predicted_delta

        recommendation = self._get_recommendation(reconstructed_level)

        return {
            "user_id": user_id,
            "date": today_row["date"].iloc[0],
            "current_wellness": round(current_wellness, 3),
            "predicted_delta": round(predicted_delta, 3),
            "predicted_wellness_tomorrow": round(
                reconstructed_level, 3
            ),
            "recommendation": recommendation,
        }