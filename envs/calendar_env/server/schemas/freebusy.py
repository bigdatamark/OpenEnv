# [MODULE DOC]
# Pydantic models for the FreeBusy API (request, per-calendar/group results, response)
# plus internal helper models for time range / overlap / query validation.
"""
FreeBusy models following Google Calendar API v3 structure
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class FreeBusyError(BaseModel):
    # [CLASS DOC]
    # Error entry with domain and reason strings.
    """Error model for FreeBusy API responses"""

    domain: str = Field(..., description="Error domain")
    reason: str = Field(..., description="Error reason")


class TimePeriod(BaseModel):
    # [CLASS DOC]
    # Busy period with RFC3339 start and end strings.
    """Time period model for busy times"""

    start: str = Field(..., description="Start time in RFC3339 format")
    end: str = Field(..., description="End time in RFC3339 format")


class CalendarItem(BaseModel):
    # [CLASS DOC]
    # Requested calendar reference with an id.
    """Calendar item for FreeBusy query request"""

    id: str = Field(..., description="Calendar identifier")


class FreeBusyQueryRequest(BaseModel):
    # [CLASS DOC]
    # FreeBusy request body: timeMin, timeMax (strings, not parsed here), timeZone (default
    # "UTC"), optional groupExpansionMax/calendarExpansionMax (>=1), and non-empty items.
    """Request model for FreeBusy query"""

    timeMin: str = Field(..., description="Lower bound for the query (RFC3339)")
    timeMax: str = Field(..., description="Upper bound for the query (RFC3339)")
    timeZone: Optional[str] = Field("UTC", description="Time zone for the query")
    groupExpansionMax: Optional[int] = Field(
        None, ge=1, description="Maximum number of calendars to expand for groups"
    )
    calendarExpansionMax: Optional[int] = Field(
        None, ge=1, description="Maximum number of events to expand for calendars"
    )
    items: List[CalendarItem] = Field(..., description="List of calendars to query")

    @field_validator("timeZone")
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> Optional[str]:
        # [FUNCTION DOC]
        # Validator: returns None unchanged; otherwise returns v if dateutil recognizes it as a timezone.
        # Raises: ValueError "Invalid timeZone; validation failed" for unknown zones (the more
        # specific message is swallowed by the broad except).
        if v is None:
            return v
        try:
            from dateutil.tz import gettz

            if gettz(v) is None:
                raise ValueError("Invalid timeZone; must be a valid IANA timezone name")
        except Exception:
            # If dateutil is unavailable or another error occurs
            raise ValueError("Invalid timeZone; validation failed")
        return v

    @field_validator("items")
    @classmethod
    def validate_items_not_empty(cls, v):
        # [FUNCTION DOC]
        # Validator: raises ValueError if items is empty, else returns it.
        """Validate that items list is not empty."""
        if not v:
            raise ValueError("At least one calendar item is required")
        return v


class FreeBusyCalendarResult(BaseModel):
    # [CLASS DOC]
    # Per-calendar result with optional errors list and optional busy list.
    """FreeBusy result for a single calendar"""

    errors: Optional[List[FreeBusyError]] = Field(None, description="List of errors")
    busy: Optional[List[TimePeriod]] = Field(
        None, description="List of busy time periods"
    )


class FreeBusyGroupResult(BaseModel):
    # [CLASS DOC]
    # Per-group result with optional calendar id list and optional errors.
    """FreeBusy result for a group"""

    calendars: Optional[List[str]] = Field(
        None, description="List of calendar IDs in the group"
    )
    errors: Optional[List[FreeBusyError]] = Field(None, description="List of errors")


class FreeBusyQueryResponse(BaseModel):
    # [CLASS DOC]
    # Response: kind "calendar#freeBusy", timeMin, timeMax, calendars dict id -> result, optional groups.
    """Response model for FreeBusy query"""

    kind: str = Field(default="calendar#freeBusy", description="Resource type")
    timeMin: str = Field(..., description="Lower bound for the query (RFC3339)")
    timeMax: str = Field(..., description="Upper bound for the query (RFC3339)")
    calendars: Dict[str, FreeBusyCalendarResult] = Field(
        default_factory=dict, description="Calendar-specific results"
    )
    groups: Optional[Dict[str, FreeBusyGroupResult]] = Field(
        None, description="Group-specific results"
    )


# Additional helper models for internal use


class FreeBusyCalendarInput(BaseModel):
    # [CLASS DOC]
    # Internal (unused here) pair of calendar_id and user_id.
    """Internal model for calendar input validation"""

    calendar_id: str = Field(..., description="Calendar identifier")
    user_id: str = Field(..., description="User identifier")


class FreeBusyTimeRange(BaseModel):
    # [CLASS DOC]
    # Internal model with start/end datetimes and a timezone string (default "UTC").
    """Internal model for time range validation"""

    start: datetime = Field(..., description="Start datetime")
    end: datetime = Field(..., description="End datetime")
    timezone: str = Field(default="UTC", description="Timezone")

    def validate_range(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if end is after start.
        """Validate that end time is after start time"""
        return self.end > self.start


class FreeBusyEventOverlap(BaseModel):
    # [CLASS DOC]
    # Internal event slice: event_id, start, end, transparency (default "opaque").
    """Internal model for event overlap calculation"""

    event_id: str = Field(..., description="Event identifier")
    start: datetime = Field(..., description="Event start time")
    end: datetime = Field(..., description="Event end time")
    transparency: Optional[str] = Field("opaque", description="Event transparency")

    def is_busy(self) -> bool:
        # [FUNCTION DOC]
        # Returns True unless transparency is "transparent".
        """Check if event blocks time (is not transparent)"""
        return self.transparency != "transparent"


class FreeBusyQueryValidation(BaseModel):
    # [CLASS DOC]
    # Internal parsed query: time_min, time_max, calendar_ids, user_id.
    """Internal model for query validation"""

    time_min: datetime = Field(..., description="Query start time")
    time_max: datetime = Field(..., description="Query end time")
    calendar_ids: List[str] = Field(..., description="Calendar IDs to query")
    user_id: str = Field(..., description="User making the query")

    def validate_time_range(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if time_max is strictly after time_min and the span is at most 366 days.
        """Validate time range is reasonable"""
        max_duration_days = 366  # Maximum 1 year + 1 day
        # Calculate total duration in seconds to handle same-day queries
        duration_timedelta = self.time_max - self.time_min
        duration_seconds = duration_timedelta.total_seconds()

        # Ensure the time range is valid (end after start)
        if duration_seconds <= 0:
            return False

        # Convert to days for comparison with maximum limit
        duration_days = duration_seconds / (24 * 60 * 60)  # seconds per day

        return duration_days <= max_duration_days

    def validate_calendar_count(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if there are between 1 and 50 calendar ids.
        """Validate number of calendars is reasonable"""
        max_calendars = 50  # Maximum calendars per query
        return 0 < len(self.calendar_ids) <= max_calendars
