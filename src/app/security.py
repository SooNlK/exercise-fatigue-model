from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe

import jwt
from pwdlib import PasswordHash

from src.app.config import settings


ALGORITHM = "HS256"
AUTH_COOKIE_NAME = "fatigue_access_token"
password_hasher = PasswordHash.recommended()

# Umożliwia wykonanie weryfikacji również dla nieistniejącego konta.
DUMMY_PASSWORD_HASH = password_hasher.hash(token_urlsafe(32))


def verify_password(password: str, password_hash: str) -> bool:
    return password_hasher.verify(password, password_hash)


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(
            minutes=settings.access_token_expire_minutes
        ),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )