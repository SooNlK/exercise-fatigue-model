"""Sprawdza sesję w ciasteczku, wylogowanie, CSRF i właściciela raportów.

Testy używają rzeczywistego haszowania i JWT oraz osobnej bazy SQLite
w pamięci. Nie łączą się z PostgreSQL aplikacji.
"""

from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
import os
import unittest
from unittest.mock import patch

import jwt
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

with patch.dict(os.environ, {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
    "JWT_SECRET_KEY": "test-only-secret-0123456789abcdef0123456789abcdef",
}):
    from src.app.api.dependencies import verify_browser_request
    from src.app.api.routes import daily_logs, login, me, register
    from src.app.config import settings
    from src.app.db.database import Base, get_db
    from src.app.db.models import DailyLog, User
    from src.app.security import ALGORITHM, AUTH_COOKIE_NAME, password_hasher


class AuthSessionApiTests(unittest.TestCase):
    password = "test-password-for-cookie-session"
    headers = {
        "X-CSRF-Protection": "1",
        "Origin": "http://localhost:5173",
    }

    @classmethod
    def setUpClass(cls):
        cls.password_hash = password_hasher.hash(cls.password)

    def setUp(self):
        for name, value in {
            "jwt_secret_key": SecretStr(
                "test-only-secret-0123456789abcdef0123456789abcdef"
            ),
            "auth_cookie_secure": False,
            "access_token_expire_minutes": 30,
            "csrf_trusted_origins": ["http://localhost:5173"],
        }.items():
            setting_patch = patch.object(settings, name, value)
            setting_patch.start()
            self.addCleanup(setting_patch.stop)

        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)

        with Session(self.engine) as db:
            for user_id in ["user-a", "user-b"]:
                db.add(User(
                    id=user_id,
                    email=f"{user_id}@example.com",
                    password_hash=self.password_hash,
                ))
            db.commit()

        def test_db():
            with Session(self.engine) as db:
                yield db

        self.app = FastAPI(dependencies=[Depends(verify_browser_request)])
        for router in [login.router, register.router, me.router, daily_logs.router]:
            self.app.include_router(router)
        self.app.dependency_overrides[get_db] = test_db
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def sign_in(self, user_id="user-a", **kwargs):
        return self.client.post(
            "/auth/login",
            json={"email": f"{user_id}@example.com", "password": self.password},
            headers=kwargs.get("headers", self.headers),
        )

    def test_login_sets_httponly_cookie_without_returning_token(self):
        response = self.sign_in()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Zalogowano."})
        cookie = SimpleCookie(response.headers["set-cookie"])[AUTH_COOKIE_NAME]
        self.assertTrue(cookie["httponly"])
        self.assertFalse(cookie["secure"])
        self.assertEqual(cookie["samesite"], "lax")
        self.assertEqual(cookie["path"], "/")
        self.assertEqual(cookie["max-age"], "1800")

    def test_new_client_restores_user_from_cookie(self):
        self.assertEqual(self.sign_in().status_code, 200)
        # Nowy klient odtwarza ponowne uruchomienie frontendu z tym samym cookie.
        with TestClient(self.app) as reloaded:
            reloaded.cookies.update(self.client.cookies)
            response = reloaded.get("/me")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "user-a")
        self.assertNotIn("password_hash", response.json())
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_logout_removes_cookie_and_access_to_me(self):
        self.sign_in()
        response = self.client.post("/auth/logout", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(AUTH_COOKIE_NAME, self.client.cookies)
        self.assertEqual(self.client.get("/me").status_code, 401)
        self.assertEqual(
            self.client.post("/auth/logout", headers=self.headers).status_code,
            200,
        )

    def test_incorrect_password_does_not_create_session(self):
        response = self.client.post(
            "/auth/login",
            json={"email": "user-a@example.com", "password": "wrong-password"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 401)
        self.assertNotIn(AUTH_COOKIE_NAME, self.client.cookies)

    def test_missing_invalid_expired_and_unknown_user_cookies_are_rejected(self):
        now = datetime.now(timezone.utc)
        for token in [
            None,
            "invalid-token",
            jwt.encode(
                {"sub": "user-a", "iat": now - timedelta(hours=2),
                 "exp": now - timedelta(hours=1)},
                settings.jwt_secret_key.get_secret_value(), algorithm=ALGORITHM,
            ),
            jwt.encode(
                {"sub": "unknown-user", "iat": now, "exp": now + timedelta(minutes=5)},
                settings.jwt_secret_key.get_secret_value(), algorithm=ALGORITHM,
            ),
        ]:
            with self.subTest(kind="missing" if token is None else "invalid-session"):
                self.client.cookies.clear()
                if token is not None:
                    self.client.cookies.set(AUTH_COOKIE_NAME, token)
                self.assertEqual(self.client.get("/me").status_code, 401)

    def test_login_requires_csrf_header_and_trusted_origin(self):
        for headers in [
            {},
            {"X-CSRF-Protection": "wrong"},
            {**self.headers, "Origin": "https://untrusted.example"},
            {**self.headers, "Origin": "null"},
        ]:
            with self.subTest(headers=headers):
                self.assertEqual(self.sign_in(headers=headers).status_code, 403)
                self.assertNotIn(AUTH_COOKIE_NAME, self.client.cookies)

    def test_cookie_writes_require_csrf_and_cannot_change_report_owner(self):
        self.sign_in()
        entry = {"date": "2026-10-28", "fatigue": 3, "user_id": "user-b"}
        response = self.client.put("/me/daily-log", json=entry)
        self.assertEqual(response.status_code, 403)
        response = self.client.put(
            "/me/daily-log", json=entry,
            headers={**self.headers, "Origin": "https://untrusted.example"},
        )
        self.assertEqual(response.status_code, 403)
        with Session(self.engine) as db:
            self.assertEqual(db.scalar(select(func.count()).select_from(DailyLog)), 0)

        response = self.client.put("/me/daily-log", json=entry, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["user_id"], "user-a")

        self.assertEqual(self.sign_in("user-b").status_code, 200)
        self.assertEqual(self.client.get("/me/daily-logs").json(), [])

    def test_logout_without_csrf_does_not_remove_session(self):
        self.sign_in()
        self.assertEqual(self.client.post("/auth/logout").status_code, 403)
        self.assertEqual(self.client.get("/me").status_code, 200)

    def test_registration_still_works_and_duplicate_email_is_rejected(self):
        entry = {"email": "new-user@example.com", "password": self.password}
        first = self.client.post("/auth/register", json=entry, headers=self.headers)
        second = self.client.post("/auth/register", json=entry, headers=self.headers)
        self.assertEqual(first.status_code, 201)
        self.assertNotIn("password_hash", first.json())
        self.assertEqual(second.status_code, 409)

    def test_https_configuration_sets_secure_cookie(self):
        with patch.object(settings, "auth_cookie_secure", True):
            response = self.sign_in()
        cookie = SimpleCookie(response.headers["set-cookie"])[AUTH_COOKIE_NAME]
        self.assertTrue(cookie["secure"])


if __name__ == "__main__":
    unittest.main()
