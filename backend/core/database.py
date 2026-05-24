from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from core.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={} if "postgresql" not in DATABASE_URL else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for all models to inherit from
Base = declarative_base()

def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
