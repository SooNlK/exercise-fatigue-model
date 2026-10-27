from fastapi import APIRouter, Depends

from src.app.api.dependencies import get_current_user
from src.app.db.models import User
from src.app.schemas.user import UserResponse


router = APIRouter(prefix="/me", tags=["me"])


@router.get("", response_model=UserResponse)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user