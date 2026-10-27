import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Konfiguracja importu modułów. Z tym PostgreSQL nie łączymy się.
with patch.dict(os.environ, {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
    "JWT_SECRET_KEY": "test-only-secret-0123456789abcdef0123456789abcdef",
}):
    from src.app.api.routes import daily_logs
    from src.app.db.database import Base, get_db
    from src.app.db.models import DailyLog, User
    from src.app.config import settings
    from src.app.security import create_access_token


class DailyLogApiTests(unittest.TestCase):
    def setUp(self):
        secret_patch = patch.object(
            settings,
            "jwt_secret_key",
            SecretStr("test-only-secret-0123456789abcdef0123456789abcdef"),
        )
        secret_patch.start()
        self.addCleanup(secret_patch.stop)
        # Każdy test otrzymuje nową bazę wyłącznie w pamięci.
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        event.listen(
            self.engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine)

        with session_factory() as session:
            for user_id in ["test-user", "user-a", "user-b"]:
                session.add(User(
                    id=user_id,
                    email=f"{user_id}@example.com",
                    # Testujemy tokeny, bez logowania hasłem.
                    password_hash="unused-in-token-tests",
                ))
            session.commit()

        def override_get_db():
            with session_factory() as session:
                yield session

        # Nie importujemy main.py, które tworzy tabele w bazie aplikacji.
        app = FastAPI()
        app.include_router(daily_logs.router)
        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

        self.payload = {
            "date": "2026-10-27",
            "fatigue": 3,
            "readiness": 7,
            "soreness": 2,
            "sleep_duration": 7.5,
            "sleep_quality": 4,
            "mood": 4,
            "stress": 2,
            "daily_srpe": 420,
        }

    def tearDown(self):
        self.client.close()
        self.engine.dispose()

    def save(self, payload, user_id="test-user"):
        return self.client.put(
            "/me/daily-log",
            json=payload,
            headers={"Authorization": f"Bearer {create_access_token(user_id)}"},
        )

    def row_count(self):
        with Session(self.engine) as session:
            return session.scalar(
                select(func.count()).select_from(DailyLog)
            )

    def test_full_report_is_saved(self):
        response = self.save(self.payload)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["user_id"], "test-user")
        self.assertEqual(self.row_count(), 1)

        # Sprawdzamy zapis w bazie, nie tylko odpowiedź HTTP.
        with Session(self.engine) as session:
            row = session.scalars(select(DailyLog)).one()
            self.assertEqual(row.fatigue, 3)
            self.assertEqual(row.mood, 4)
            self.assertEqual(row.daily_srpe, 420)

    def test_minimal_report_preserves_nulls(self):
        response = self.save({
            "date": "2026-10-27",
            "fatigue": 3,
        })

        self.assertEqual(response.status_code, 201)
        for field in [
            "readiness", "soreness", "sleep_duration",
            "sleep_quality", "mood", "stress", "daily_srpe",
        ]:
            self.assertIsNone(response.json()[field])

        with Session(self.engine) as session:
            row = session.scalars(select(DailyLog)).one()
            self.assertIsNone(row.daily_srpe)
            self.assertIsNone(row.mood)

    def test_update_does_not_create_duplicate(self):
        first = self.save(self.payload)
        changed = {**self.payload, "fatigue": 2, "mood": 3}
        second = self.save(changed)

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(self.row_count(), 1)

        with Session(self.engine) as session:
            row = session.scalars(select(DailyLog)).one()
            self.assertEqual(row.fatigue, 2)
            self.assertEqual(row.mood, 3)

    def test_same_date_for_different_users(self):
        first = self.save(self.payload, user_id="user-a")
        second = self.save(self.payload, user_id="user-b")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertNotEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(self.row_count(), 2)

    def test_invalid_report_is_not_saved(self):
        for field, value in [
            ("fatigue", 0),
            ("mood", 6),
            ("sleep_duration", -1),
            ("daily_srpe", -10),
        ]:
            with self.subTest(field=field):
                response = self.save({**self.payload, field: value})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(self.row_count(), 0)

    def test_missing_token_returns_401_without_saving(self):
        response = self.client.put("/me/daily-log", json=self.payload)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.row_count(), 0)

    def test_invalid_token_returns_401_without_saving(self):
        response = self.client.put(
            "/me/daily-log",
            json=self.payload,
            headers={"Authorization": "Bearer invalid-token"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.row_count(), 0)

    def test_body_cannot_choose_another_owner(self):
        first = self.save(self.payload, user_id="user-b")
        second = self.save(
            {**self.payload, "user_id": "user-b", "fatigue": 1},
            user_id="user-a",
        )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.json()["user_id"], "user-a")
        with Session(self.engine) as session:
            logs = {row.user_id: row.fatigue for row in session.scalars(select(DailyLog))}
            self.assertEqual(logs, {"user-a": 1, "user-b": 3})

    def test_legacy_user_route_is_not_available(self):
        response = self.client.put(
            "/users/user-b/daily-log",
            json=self.payload,
            headers={"Authorization": f"Bearer {create_access_token('user-a')}"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.row_count(), 0)


if __name__ == "__main__":
    unittest.main()
