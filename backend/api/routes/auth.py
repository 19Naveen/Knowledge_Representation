from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core.database import get_db
from auth.security import decode_access_token
from auth.service import login_user, register_user
from auth.store import get_user_by_email
from schema.auth import (
    LoginRequest,
    OAuthProviderConfig,
    SignupRequest,
    TokenResponse,
)
from schema.user import UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer_scheme = HTTPBearer(auto_error=True)


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
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> UserPublic:
    token_payload = decode_access_token(credentials.credentials)
    email = str(token_payload.get("email", ""))

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
