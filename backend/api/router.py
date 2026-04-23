from fastapi import APIRouter

from api.routes.auth import router as auth_router


api_router = APIRouter()
api_router.include_router(auth_router)
