# [PACKAGE DOC]
# Database models package: imports and re-exports the ORM classes (Base, User, Calendar,
# Event, Attendees, Attachment, WorkingLocationProperties, Color, Settings, ACLs, Scope,
# WatchChannel). Importing it registers all tables on Base.metadata.
"""
Database models package
"""

from .acl import ACLs, Scope
from .base import Base
from .calendar import Calendar
from .color import Color
from .event import Attachment, Attendees, Event, WorkingLocationProperties
from .settings import Settings
from .user import User
from .watch_channel import WatchChannel

__all__ = [
    "Base",
    "User",
    "Calendar",
    "Event",
    "Attendees",
    "Attachment",
    "WorkingLocationProperties",
    "Color",
    "Settings",
    "ACLs",
    "Scope",
    "WatchChannel",
]
