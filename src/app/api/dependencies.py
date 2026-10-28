import jwt
from fastapi import Depends, Header, HTTPException, Request, Response
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session

from src.app.config import settings
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.security import ALGORITHM, AUTH_COOKIE_NAME


bearer = HTTPBearer(auto_error=False)


def verify_browser_request(
    request: Request,
    response: Response,
    csrf_protection: str | None = Header(
        default=None,
        alias="X-CSRF-Protection",
    ),
) -> None:
    """Chroni żądania zmieniające dane przed CSRF."""
    response.headers["Cache-Control"] = "no-store"

    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return

    if csrf_protection != "1":
        raise HTTPException(
            status_code=403,
            detail="Brak wymaganego nagłówka ochronnego.",
        )

    origin = request.headers.get("origin")

    if (
        origin is not None
        and origin not in settings.csrf_trusted_origins
    ):
        raise HTTPException(
            status_code=403,
            detail="Niedozwolone źródło żądania.",
        )


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=401,
        detail="Zaloguj się ponownie.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Zachowujemy obsługę Bearer dla dotychczasowych klientów i testów.
    # Frontend będzie korzystał z ciasteczka.
    token = (
        credentials.credentials
        if credentials is not None
        else request.cookies.get(AUTH_COOKIE_NAME)
    )

    if not token:
        raise unauthorized

    try:
        payload = jwt.decode(
            token,
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