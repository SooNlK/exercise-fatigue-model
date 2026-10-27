from pwdlib import PasswordHash
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from src.app.db.models import User
from src.app.exceptions import EmailAlreadyExistsError
from src.app.schemas.user import UserCreate


password_hasher = PasswordHash.recommended()


class RegistrationService:
    def __init__(self, db: Session):
        self.db = db

    def _email_exists(self, email: str) -> bool:
        statement = select(User.id).where(
            func.lower(User.email) == email
        )
        return self.db.scalar(statement) is not None

    def register(self, entry: UserCreate) -> User:
        # W naszej aplikacji e-maile porównujemy bez wielkości liter.
        email = str(entry.email).lower()

        if self._email_exists(email):
            raise EmailAlreadyExistsError(
                "Ten adres e-mail jest już zajęty."
            )

        hashed_password = password_hasher.hash(
            entry.password.get_secret_value()
        )

        user = User(
            email=email,
            password_hash=hashed_password,
        )

        self.db.add(user)

        try:
            self.db.commit()

        except IntegrityError as exc:
            self.db.rollback()

            # Obsługuje również dwie równoczesne rejestracje
            # tego samego adresu.
            if self._email_exists(email):
                raise EmailAlreadyExistsError(
                    "Ten adres e-mail jest już zajęty."
                ) from exc

            raise

        except SQLAlchemyError:
            self.db.rollback()
            raise

        self.db.refresh(user)
        return user