from typing import Optional

from sqlalchemy.orm import Session

from models import UserModel
from schema.user import UserInternal


def get_user_by_username(db: Session, username: str) -> Optional[UserInternal]:
    user = db.query(UserModel).filter(UserModel.username == username).first()
    return _model_to_internal(user) if user else None


def get_user_by_email(db: Session, email: str) -> Optional[UserInternal]:
    user = db.query(UserModel).filter(
        UserModel.email == email.lower()
    ).first()
    return _model_to_internal(user) if user else None


def create_user(db: Session, user: UserInternal) -> UserInternal:
    user.email = user.email.lower()
    
    db_user = UserModel(
        id=user.id,
        username=user.username,
        email=user.email,
        password_hash=user.password_hash,
        auth_provider=user.auth_provider,
        is_active=user.is_active,
        created_at=user.created_at,
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return _model_to_internal(db_user)


def _model_to_internal(model: UserModel) -> UserInternal:
    return UserInternal(
        id=model.id,
        username=model.username,
        email=model.email,
        password_hash=model.password_hash,
        created_at=model.created_at,
        is_active=model.is_active,
        auth_provider=model.auth_provider,
        oauth_identities=[],
    )
