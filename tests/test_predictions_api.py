import importlib
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.data.soccermon.preparation import make_features
from src.models.soccermon.risk import (
    PIPELINE_VERSION,
    TARGET_DEFINITION,
)


class TestRiskModel:
    """Model testowy zwraca stałe ryzyko, żeby test był przewidywalny."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        self.last_input = X.copy()
        return np.tile([0.35, 0.65], (len(X), 1))


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame({
            "user_id": "test-user",
            "date": pd.date_range(
                "2026-09-27",
                "2026-10-25",
            ),
            "fatigue": 3,
            "readiness": 7,
            "soreness": 2,
            "sleep_duration": 7.5,
            "sleep_quality": 4,
            "mood": 4,
            "stress": 2,
            "daily_srpe": 300.0,
        })

        _, feature_sets = make_features(
            self.history.assign(participant="test-user")
        )

        self.model = TestRiskModel()
        self.artifact = {
            "pipeline_version": PIPELINE_VERSION,
            "target_definition": TARGET_DEFINITION,
            "horizon_days": 1,
            "method": "boosting_load",
            "model_id": "test-model",
            "model": self.model,
            "feature_cols": feature_sets["wellness_load"],
            "training_max_target_date": "2021-12-31",
            "training_counts_by_participant": {},
        }

        # Konfiguracja testowa. Engine może powstać, ale nie łączymy z bazą.
        test_env = {
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "test",
            "JWT_SECRET_KEY": "test-only-secret-0123456789abcdef0123456789abcdef",
        }

        # Router ładuje model podczas importu, więc zastępujemy loader.
        with patch.dict(os.environ, test_env):
            with patch(
                "src.models.soccermon.predictor.load_risk_model",
                return_value=self.artifact,
            ):
                self.routes = importlib.import_module(
                    "src.app.api.routes.predictions"
                )

        from src.app.config import settings
        from src.app.db.database import Base
        from src.app.db.models import User
        from src.app.security import create_access_token

        secret_patch = patch.object(
            settings,
            "jwt_secret_key",
            SecretStr("test-only-secret-0123456789abcdef0123456789abcdef"),
        )
        secret_patch.start()
        self.addCleanup(secret_patch.stop)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            session.add(User(
                id="test-user",
                email="test-user@example.com",
                password_hash="unused-in-token-tests",
            ))
            session.commit()

        def test_db():
            with Session(self.engine) as session:
                yield session

        self.headers = {
            "Authorization": f"Bearer {create_access_token('test-user')}"
        }

        # Działa również wtedy, gdy router zaimportowano wcześniej.
        artifact_patch = patch.object(
            self.routes,
            "ARTIFACT",
            self.artifact,
        )
        artifact_patch.start()
        self.addCleanup(artifact_patch.stop)

        self.repository = Mock()
        self.repository.get_by_user.return_value = self.history

        repository_patch = patch.object(
            self.routes,
            "DailyLogRepository",
            return_value=self.repository,
        )
        repository_patch.start()
        self.addCleanup(repository_patch.stop)

        # Nie importujemy main.py, żeby nie uruchamiać create_all().
        app = FastAPI()
        app.include_router(self.routes.router)
        app.dependency_overrides[self.routes.get_db] = test_db

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def get_prediction(self):
        return self.client.get(
            "/me/predict",
            params={"prediction_date": "2026-10-25"},
            headers=self.headers,
        )

    def test_correct_prediction(self):
        response = self.get_prediction()

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["prediction_date"], "2026-10-25")
        self.assertEqual(data["target_date"], "2026-10-26")
        self.assertEqual(data["current_fatigue"], 3)
        self.assertAlmostEqual(
            data["deterioration_probability"],
            0.65,
        )
        self.assertFalse(
            data["known_person_with_20_training_pairs"]
        )

    def test_user_without_reports_returns_404(self):
        self.repository.get_by_user.return_value = pd.DataFrame()

        response = self.get_prediction()

        self.assertEqual(response.status_code, 404)
        self.assertIn("detail", response.json())

    def test_short_history_returns_400(self):
        self.repository.get_by_user.return_value = (
            self.history.tail(5).copy()
        )

        response = self.get_prediction()

        self.assertEqual(response.status_code, 400)

    def test_missing_prediction_day_returns_400(self):
        self.repository.get_by_user.return_value = (
            self.history.iloc[:-1].copy()
        )

        response = self.get_prediction()

        self.assertEqual(response.status_code, 400)

    def test_too_few_previous_fatigue_reports_returns_400(self):
        history = self.history.copy()

        # Zostaje dzisiejsza ocena i tylko 6 wcześniejszych.
        history.loc[history.index[:22], "fatigue"] = np.nan
        self.repository.get_by_user.return_value = history

        response = self.get_prediction()

        self.assertEqual(response.status_code, 400)

    def test_fatigue_one_has_zero_deterioration_risk(self):
        history = self.history.copy()
        history.loc[history.index[-1], "fatigue"] = 1
        self.repository.get_by_user.return_value = history

        response = self.get_prediction()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["deterioration_probability"],
            0.0,
        )
        self.assertEqual(
            response.json()["probability_source"],
            "scale_floor",
        )

    def test_optional_measurements_can_be_missing(self):
        self.repository.get_by_user.return_value = self.history[
            ["user_id", "date", "fatigue"]
        ].copy()

        response = self.get_prediction()

        self.assertEqual(response.status_code, 200)

        # Sprawdzamy, że brak obciążenia pozostał NaN.
        self.assertTrue(
            self.model.last_input["daily_srpe"].isna().all()
        )

    def test_missing_token_returns_401_without_reading_reports(self):
        response = self.client.get(
            "/me/predict",
            params={"prediction_date": "2026-10-25"},
        )

        self.assertEqual(response.status_code, 401)
        self.repository.get_by_user.assert_not_called()

    def test_invalid_token_returns_401_without_reading_reports(self):
        response = self.client.get(
            "/me/predict",
            params={"prediction_date": "2026-10-25"},
            headers={"Authorization": "Bearer invalid-token"},
        )

        self.assertEqual(response.status_code, 401)
        self.repository.get_by_user.assert_not_called()

    def test_query_cannot_choose_another_owner(self):
        response = self.client.get(
            "/me/predict",
            params={"prediction_date": "2026-10-25", "user_id": "user-b"},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test-user")
        self.repository.get_by_user.assert_called_once_with("test-user")

    def test_legacy_user_route_is_not_available(self):
        response = self.client.get(
            "/users/user-b/predict",
            params={"prediction_date": "2026-10-25"},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 404)
        self.repository.get_by_user.assert_not_called()


if __name__ == "__main__":
    unittest.main()
