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


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic


class OAuthInitResponse(BaseModel):
    provider: AuthProvider
    detail: str


class OAuthProviderConfig(BaseModel):
    provider: AuthProvider
    enabled: bool = False
