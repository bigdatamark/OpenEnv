# [MODULE DOC]
# SQLAlchemy ORM model for Google Calendar color palette entries (table "colors"), plus the ColorType enum.
"""
Color database model for Google Calendar API v3 color definitions
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .base import Base


class ColorType(enum.Enum):
    # [CLASS DOC]
    # Enum distinguishing palette kinds: CALENDAR ("calendar") and EVENT ("event").
    """Color type enumeration"""

    CALENDAR = "calendar"
    EVENT = "event"


class Color(Base):
    # [CLASS DOC]
    # ORM row for one palette entry: autoincrement id, color_id string ("1", "2", ...),
    # color_type (ColorType), background/foreground hex strings, created/updated UTC
    # timestamps, and an "events" relationship back-populating Event.color.
    # KNOWN ISSUE: the "Composite unique constraint" comment does not match __table_args__, which defines no constraint and only conditionally (via an always-false hasattr check on __builtins__) sets a MySQL charset.
    """Color database model for calendar and event color definitions"""

    __tablename__ = "colors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    color_id = Column(String(10), nullable=False, index=True)  # "1", "2", "3", etc.
    color_type = Column(Enum(ColorType), nullable=False, index=True)
    background = Column(String(7), nullable=False)  # Hex color like "#ac725e"
    foreground = Column(String(7), nullable=False)  # Hex color like "#1d1d1d"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    events = relationship("Event", back_populates="color")

    # Composite unique constraint
    __table_args__ = (
        {"mysql_charset": "utf8mb4"} if hasattr(__builtins__, "mysql") else {}
    )

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns a debug string with color_id, type value, background and foreground.
        return f"<Color(id='{self.color_id}', type='{self.color_type.value}', bg='{self.background}', fg='{self.foreground}')>"

    def to_dict(self):
        # [FUNCTION DOC]
        # Returns a dict with keys "background" and "foreground" (the API color shape).
        """Convert to dictionary format for API responses"""
        return {"background": self.background, "foreground": self.foreground}
