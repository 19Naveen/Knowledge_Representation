from api.router import api_router
from core.database import Base, engine
import modules.workspace.models  # noqa: F401 — ensures table is registered before create_all
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize database tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to Knowledge Representation API!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
