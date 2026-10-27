from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.app.db.database import get_db
from src.app.exceptions import EmailAlreadyExistsError
from src.app.schemas.user import UserCreate, UserResponse
from src.app.services.registration_service import RegistrationService


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
)
def register(
    entry: UserCreate,
    db: Session = Depends(get_db),
):
    service = RegistrationService(db)

    try:
        return service.register(entry)

    except EmailAlreadyExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Nie udało się utworzyć konta.",
        ) from exc