from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config.config import settings
from auth.security import create_access_token, get_password_hash, verify_password
from auth.store import create_user, get_user_by_email, get_user_by_username
from schema.auth import TokenResponse
from schema.user import UserInternal, UserPublic


def register_user(db: Session, username: str, email: str, password: str) -> TokenResponse:
    if get_user_by_username(db, username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    if get_user_by_email(db, email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists",
        )

    user = UserInternal(
        id=str(uuid4()),
        username=username,
        email=email,
        password_hash=get_password_hash(password),
        created_at=datetime.now(timezone.utc),
    )

    try:
        created_user = create_user(db, user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists",
        )

    access_token = create_access_token(
        subject=created_user.id,
        username=created_user.username,
        email=created_user.email,
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserPublic(
            id=created_user.id,
            username=created_user.username,
            email=created_user.email,
            created_at=created_user.created_at,
            auth_provider=created_user.auth_provider,
        ),
    )


def login_user(db: Session, email: str, password: str) -> TokenResponse:
    user = get_user_by_email(db, email)
    if not user or not user.password_hash or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(
        subject=user.id,
        username=user.username,
        email=user.email,
        provider=user.auth_provider,
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserPublic(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            auth_provider=user.auth_provider,
        ),
    )
