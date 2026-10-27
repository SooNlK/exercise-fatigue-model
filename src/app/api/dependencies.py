import jwt
from fastapi import Depends, HTTPException
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session

from src.app.config import settings
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.security import ALGORITHM


bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=401,
        detail="Zaloguj się ponownie.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
            options={"require": ["sub", "exp", "iat"]},
        )

        user_id = payload["sub"]

        if not isinstance(user_id, str) or not user_id:
            raise unauthorized

    except InvalidTokenError as exc:
        raise unauthorized from exc

    user = db.get(User, user_id)

    if user is None:
        raise unauthorized

    return user