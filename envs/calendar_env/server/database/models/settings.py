# [MODULE DOC]
# SQLAlchemy ORM model for per-user calendar settings (table "settings").
"""
Settings database model
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .base import Base


class Settings(Base):
    # [CLASS DOC]
    # ORM row for one setting: id (string primary key, e.g. "timezone"), value (string),
    # optional etag, optional user_id foreign key to users.user_id, and a "user" relationship.
    # KNOWN ISSUE: id alone is the primary key, so two users cannot both have a setting with the same id (e.g. "timezone") despite the per-user user_id column.
    """Settings database model"""

    __tablename__ = "settings"

    id = Column(String(255), primary_key=True, nullable=False)  # e.g., "timezone"
    value = Column(String(255), nullable=False)  # e.g., "Asia/Karachi"
    etag = Column(String(255), nullable=True)
    user_id = Column(
        String(255), ForeignKey("users.user_id"), nullable=True, index=True
    )

    # Relationships
    user = relationship("User")

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns a debug string showing id, value and user_id.
        return (
            f"<Setting(id='{self.id}', value='{self.value}', user_id='{self.user_id}')>"
        )
