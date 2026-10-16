# [MODULE DOC]
# Pydantic models for the Calendars resource (Google Calendar API v3): calendar body,
# create/update requests, list response and conference properties with the allowed
# conference solution types.
"""
Calendar models for Calendar API following Google Calendar API v3 structure
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Allowed conference solution types per Google Calendar API v3 spec
ALLOWED_CONFERENCE_SOLUTION_TYPES = {
    "eventHangout",
    "eventNamedHangout",
    "hangoutsMeet",
}


class ConferenceProperties(BaseModel):
    # [CLASS DOC]
    # Conference settings with optional allowedConferenceSolutionTypes list.
    """Conference properties for calendar"""

    allowedConferenceSolutionTypes: Optional[List[str]] = Field(
        default=None,
        description=(
            "Conference solution types. Allowed values: 'eventHangout', 'eventNamedHangout', 'hangoutsMeet'"
        ),
    )

    @field_validator("allowedConferenceSolutionTypes")
    @classmethod
    def validate_allowed_conference_solution_types(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        # [FUNCTION DOC]
        # Validator: allows None or a list whose non-empty entries are all in
        # {eventHangout, eventNamedHangout, hangoutsMeet}; raises ValueError listing invalid values otherwise.
        """Ensure provided values are restricted to the API-supported set.

        The Google Calendar API v3 permits only the following values for
        conferenceProperties.allowedConferenceSolutionTypes:
        - "eventHangout"
        - "eventNamedHangout"
        - "hangoutsMeet"
        """
        if v is None:
            return v
        # Allow empty list, but every provided value must be valid
        invalid = [
            item
            for item in v
            if item not in ALLOWED_CONFERENCE_SOLUTION_TYPES and item not in [None, ""]
        ]
        if invalid:
            allowed_sorted = sorted(ALLOWED_CONFERENCE_SOLUTION_TYPES)
            raise ValueError(
                "Invalid values for conferenceProperties.allowedConferenceSolutionTypes: "
                f"{invalid}. Allowed values are: {allowed_sorted}"
            )
        return v


class Calendar(BaseModel):
    # [CLASS DOC]
    # Calendar resource: kind "calendar#calendar", optional etag/id, summary (1..255 chars),
    # optional description (<=1000) and location (<=500), timeZone (default "UTC"),
    # optional conferenceProperties.
    """Calendar model following Google Calendar API v3 structure"""

    kind: str = Field(default="calendar#calendar", description="Calendar resource type")
    etag: Optional[str] = Field(None, description="ETag of the resource")
    id: Optional[str] = Field(None, description="Unique calendar identifier")
    summary: str = Field(
        ..., min_length=1, max_length=255, description="Calendar title"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Calendar description"
    )
    location: Optional[str] = Field(
        None, max_length=500, description="Calendar location"
    )
    timeZone: str = Field(default="UTC", description="Calendar timezone in IANA format")
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )


class CalendarCreateRequest(BaseModel):
    # [CLASS DOC]
    # POST /calendars body: required summary (1..255), optional description, location,
    # timeZone and conferenceProperties.
    """Request model for creating a calendar (POST /calendars)"""

    summary: str = Field(
        ..., min_length=1, max_length=255, description="Calendar title (required)"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Calendar description"
    )
    location: Optional[str] = Field(
        None, max_length=500, description="Calendar location"
    )
    timeZone: Optional[str] = Field(
        None, description="Calendar timezone in IANA format"
    )
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )

    @field_validator("summary")
    @classmethod
    def validate_summary_not_blank(cls, v: str) -> str:
        # [FUNCTION DOC]
        # Validator: raises ValueError if summary is whitespace only; returns it otherwise.
        if v is None:
            return v
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("summary must not be blank")
        return v

    @field_validator("timeZone")
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> Optional[str]:
        # [FUNCTION DOC]
        # Validator: None passes; otherwise must be a timezone known to dateutil.
        # Raises ValueError "Invalid timeZone; validation failed" (specific message swallowed).
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


class CalendarUpdateRequest(BaseModel):
    # [CLASS DOC]
    # PATCH/PUT body: all fields optional (summary typed str but defaults to None).
    """Request model for updating a calendar (PATCH/PUT /calendars/{calendarId})"""

    summary: str = Field(
        None, min_length=1, max_length=255, description="Calendar title"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Calendar description"
    )
    location: Optional[str] = Field(
        None, max_length=500, description="Calendar location"
    )
    timeZone: Optional[str] = Field(
        None, description="Calendar timezone in IANA format"
    )
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )

    @field_validator("summary")
    @classmethod
    def validate_update_summary_not_blank(cls, v: Optional[str]) -> Optional[str]:
        # [FUNCTION DOC]
        # Validator: raises ValueError if summary is provided but blank.
        if v is None:
            return v
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("summary must not be blank when provided")
        return v

    @field_validator("timeZone")
    @classmethod
    def validate_update_timezone(cls, v: Optional[str]) -> Optional[str]:
        # [FUNCTION DOC]
        # Validator: same timezone check as the create request.
        if v is None:
            return v
        try:
            from dateutil.tz import gettz

            if gettz(v) is None:
                raise ValueError("Invalid timeZone; must be a valid IANA timezone name")
        except Exception:
            raise ValueError("Invalid timeZone; validation failed")
        return v


class CalendarListResponse(BaseModel):
    # [CLASS DOC]
    # List response: kind "calendar#calendarList", optional etag and items list of Calendar.
    """Response model for listing calendars"""

    kind: str = Field(
        default="calendar#calendarList", description="Calendar list resource type"
    )
    etag: Optional[str] = Field(None, description="ETag of the collection")
    items: List[Calendar] = Field(default_factory=list, description="List of calendars")
