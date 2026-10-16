# [MODULE DOC]
# Shared Pydantic response models and plain string-constant classes used across the calendar API.
"""
Common models and response schemas for Calendar API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    # [CLASS DOC]
    # Pydantic model with success (bool), optional message and optional timestamp string.
    """Base response model"""

    success: bool
    message: Optional[str] = None
    timestamp: Optional[str] = None


class ErrorResponse(BaseResponse):
    # [CLASS DOC]
    # BaseResponse with success defaulting to False, a required error string and optional error_code.
    """Error response model"""

    success: bool = False
    error: str
    error_code: Optional[str] = None


class SuccessResponse(BaseResponse):
    # [CLASS DOC]
    # BaseResponse with success defaulting to True and an optional arbitrary data payload.
    """Success response model"""

    success: bool = True
    data: Optional[Any] = None


class PaginationModel(BaseModel):
    # [CLASS DOC]
    # Pagination parameters: limit (1..1000, default 100) and offset (>=0, default 0).
    """Pagination model"""

    limit: int = Field(
        default=100, ge=1, le=1000, description="Number of records to return"
    )
    offset: int = Field(default=0, ge=0, description="Number of records to skip")


class DatabaseStateResponse(BaseModel):
    # [CLASS DOC]
    # Response describing a database snapshot: success, service name, database_info dict,
    # per-table row counts and per-table row data lists, plus a timestamp string.
    """Database state response model"""

    success: bool
    service: str
    database_info: Dict[str, Any]
    table_counts: Dict[str, int]
    table_data: Dict[str, List[Dict[str, Any]]]
    timestamp: str


# Enums for calendar status and visibility
class EventStatus:
    # [CLASS DOC]
    # Namespace of event status string constants: confirmed, tentative, cancelled.
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EventVisibility:
    # [CLASS DOC]
    # Namespace of event visibility constants: default, public, private, confidential.
    DEFAULT = "default"
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


class CalendarAccessRole:
    # [CLASS DOC]
    # Namespace of calendar access role constants: owner, reader, writer, freeBusyReader.
    OWNER = "owner"
    READER = "reader"
    WRITER = "writer"
    FREE_BUSY_READER = "freeBusyReader"


class EventTransparency:
    # [CLASS DOC]
    # Namespace of event transparency constants: opaque, transparent.
    OPAQUE = "opaque"
    TRANSPARENT = "transparent"
