from fastapi import APIRouter

from modules.auth.router import router as auth_router
from modules.ingestion.router import router as ingest_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(ingest_router)
