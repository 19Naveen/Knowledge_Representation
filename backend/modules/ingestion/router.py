from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import decode_access_token
from modules.auth.service import login_user, register_user
from modules.auth.repository import get_user_by_email
from modules.auth.schemas import (
    LoginRequest,
    OAuthProviderConfig,
    SignupRequest,
    TokenResponse,
)
from modules.auth.schemas import UserPublic

router = APIRouter(
    prefix="/data-ingest",
    tags=["data-ingest"],
)

@router.post("/ingest")
async def ingest_data():
    return {"status": "ok"}
