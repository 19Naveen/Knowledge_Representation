from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.dependencies import get_current_user
from modules.auth.service import login_user, register_user
from modules.auth.repository import get_user_by_email
from modules.auth.schemas import (
    LoginRequest,
    OAuthProviderConfig,
    SignupRequest,
    TokenResponse,
)
from modules.auth.schemas import UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return register_user(
        db=db,
        username=payload.username,
        email=payload.email,
        password=payload.password,
    )


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return login_user(db=db, email=payload.email, password=payload.password)


@router.get("/me", response_model=UserPublic)
async def me(
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
) -> UserPublic:
    email = str(user.get("email", ""))

    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserPublic(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        auth_provider=user.auth_provider,
    )


@router.get("/oauth/providers", response_model=list[OAuthProviderConfig])
async def oauth_providers() -> list[OAuthProviderConfig]:
    # OAuth provider metadata contract exposed now so provider setup can be added later.
    return [
        OAuthProviderConfig(provider="google", enabled=False),
        OAuthProviderConfig(provider="github", enabled=False),
        OAuthProviderConfig(provider="microsoft", enabled=False),
    ]
