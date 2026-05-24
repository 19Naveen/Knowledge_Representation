from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, Text

from core.database import Base


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(500), nullable=True)
    auth_provider = Column(String(50), default="local", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    oauth_identities = Column(Text, default="[]", nullable=False)

    def __repr__(self):
        return f"<UserModel(id={self.id}, username={self.username}, email={self.email})>"
