from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.app.config import settings
from src.app.db.database import get_db
from src.app.exceptions import InvalidCredentialsError
from src.app.schemas.user import UserLogin
from src.app.security import AUTH_COOKIE_NAME
from src.app.services.auth_service import AuthService


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=dict[str, str])
def login(
    entry: UserLogin,
    response: Response,
    db: Session = Depends(get_db),
):
    service = AuthService(db)

    try:
        result = service.login(entry)

    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Nie udało się wykonać logowania.",
        ) from exc

    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=result["access_token"],
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
    )

    return {"message": "Zalogowano."}


@router.post("/logout", response_model=dict[str, str])
def logout(response: Response):
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
    )

    return {"message": "Wylogowano."}