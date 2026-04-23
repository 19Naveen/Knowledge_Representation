from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


AuthProvider = Literal["local", "google", "github", "microsoft"]


class UserPublic(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime
    auth_provider: AuthProvider = "local"


class OAuthIdentity(BaseModel):
    provider: AuthProvider
    provider_subject: str
    provider_email: str | None = None


class UserInternal(BaseModel):
    id: str
    username: str
    email: str
    password_hash: str | None = None
    created_at: datetime
    is_active: bool = True
    auth_provider: AuthProvider = "local"
    oauth_identities: list[OAuthIdentity] = Field(default_factory=list)
