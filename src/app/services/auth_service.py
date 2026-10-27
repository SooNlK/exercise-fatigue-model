from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.app.db.models import User
from src.app.exceptions import InvalidCredentialsError
from src.app.schemas.user import UserLogin
from src.app.security import (
    DUMMY_PASSWORD_HASH,
    create_access_token,
    verify_password,
)


class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def login(self, entry: UserLogin) -> dict:
        email = str(entry.email).lower()

        statement = select(User).where(
            func.lower(User.email) == email
        )
        user = self.db.scalar(statement)

        stored_hash = (
            user.password_hash
            if user is not None
            else DUMMY_PASSWORD_HASH
        )

        password_valid = verify_password(
            entry.password.get_secret_value(),
            stored_hash,
        )

        if user is None or not password_valid:
            raise InvalidCredentialsError(
                "Niepoprawny e-mail lub hasło."
            )

        return {
            "access_token": create_access_token(user.id),
            "token_type": "bearer",
        }