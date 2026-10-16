# [MODULE DOC]
# Pydantic models for the CalendarList resource (Google Calendar API v3): enums for
# access role, reminder/notification methods and notification types; reminder,
# notification and conference sub-models (strict versions and lenient "Patch" versions
# that accept empty strings); entry, insert/update/patch requests, list response, and
# watch channel request/response models.
"""
CalendarList models following Google Calendar API v3 CalendarList structure
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class AccessRole(str, Enum):
    # [CLASS DOC]
    # String enum of calendar-list access roles: freeBusyReader, reader, writer, owner.
    """Access roles for calendar list entries"""

    FREE_BUSY_READER = "freeBusyReader"
    READER = "reader"
    WRITER = "writer"
    OWNER = "owner"


class ReminderMethod(str, Enum):
    # [CLASS DOC]
    # String enum of reminder methods: email, popup.
    """Allowed reminder delivery methods per Google Calendar API v3"""

    EMAIL = "email"
    POPUP = "popup"


class NotificationMethod(str, Enum):
    # [CLASS DOC]
    # String enum of notification methods: only email.
    """Allowed notification delivery methods (CalendarList notifications)"""

    EMAIL = "email"


class NotificationType(str, Enum):
    # [CLASS DOC]
    # String enum of notification types: eventCreation, eventChange, eventCancellation, eventResponse, agenda.
    """Allowed CalendarList notification types per Google Calendar API v3"""

    EVENT_CREATION = "eventCreation"
    EVENT_CHANGE = "eventChange"
    EVENT_CANCELLATION = "eventCancellation"
    EVENT_RESPONSE = "eventResponse"
    AGENDA = "agenda"


class EventReminder(BaseModel):
    # [CLASS DOC]
    # Strict default reminder: method (ReminderMethod) and non-negative minutes.
    """Event reminder settings"""

    method: ReminderMethod = Field(
        ..., description="Reminder delivery method (email, popup)"
    )
    minutes: int = Field(
        ..., description="Minutes before event to trigger reminder (>= 0)"
    )

    @field_validator("minutes")
    @classmethod
    def _validate_minutes_non_negative(cls, v: int) -> int:
        # [FUNCTION DOC]
        # HELPER validator: raises ValueError if minutes is None or negative; returns it otherwise.
        if v is None:
            raise ValueError("minutes is required for defaultReminders items")
        if int(v) < 0:
            raise ValueError("defaultReminders[].minutes must be >= 0")
        return v


class PatchCalendarListEventReminder(BaseModel):
    # [CLASS DOC]
    # Lenient reminder for PATCH: method may be "" (default) or a ReminderMethod; minutes required.
    """Event reminder settings"""

    method: Union[ReminderMethod, str] = ""
    minutes: int = Field(
        ..., description="Minutes before event to trigger reminder (>= 0)"
    )

    @field_validator("minutes")
    @classmethod
    def _validate_minutes_non_negative(cls, v: int) -> int:
        # [FUNCTION DOC]
        # HELPER validator: intended to enforce 0..40320 minutes; returns v.
        # KNOWN ISSUE: the condition "v < 0 and v >= 40320" can never be true, so no range check happens (negative or very large values pass).
        if v is None:
            raise ValueError("minutes is required for defaultReminders items")
        if int(v) < 0 and int(v) >= 40320:
            raise ValueError("defaultReminders[].minutes must be between 0 and 40320")
        return v

    @field_validator("method")
    @classmethod
    def _validate_method(cls, v):
        # [FUNCTION DOC]
        # HELPER validator: "" passes through; otherwise converts to ReminderMethod or raises ValueError listing allowed values.
        if v == "":
            return v
        try:
            return ReminderMethod(v)
        except ValueError:
            raise ValueError(
                f"Reminder method must be one of {[rm.value for rm in ReminderMethod]} or empty string"
            )


class Notification(BaseModel):
    # [CLASS DOC]
    # Strict notification setting: method (email) and type (NotificationType).
    """Notification setting item for CalendarList notificationSettings.notifications[]"""

    method: NotificationMethod = Field(
        ..., description="Notification delivery method (only 'email' supported)"
    )
    type: NotificationType = Field(
        ...,
        description=(
            "Notification type (eventCreation, eventChange, eventCancellation, eventResponse, agenda)"
        ),
    )


class PatchCalendarListNotification(BaseModel):
    # [CLASS DOC]
    # Lenient notification for PATCH: method and type may each be "" (default) or a valid enum value.
    """Notification setting item for CalendarList notificationSettings.notifications[]"""

    method: Union[NotificationMethod, str] = ""
    type: Union[NotificationType, str] = ""

    @field_validator("method")
    @classmethod
    def _validate_method(cls, v):
        # [FUNCTION DOC]
        # HELPER validator: "" passes; otherwise converts to NotificationMethod or raises ValueError.
        if v == "":
            return v
        try:
            return NotificationMethod(v)
        except ValueError:
            raise ValueError(
                f"Notification method must be {[nm.value for nm in NotificationMethod]} or empty string"
            )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v):
        # [FUNCTION DOC]
        # HELPER validator: "" passes; otherwise converts to NotificationType or raises ValueError.
        if v == "":
            return v
        try:
            return NotificationType(v)
        except ValueError:
            raise ValueError(
                f"Notification type must be one of the {[nt.value for nt in NotificationType]} or empty string"
            )


class NotificationSettings(BaseModel):
    # [CLASS DOC]
    # Wrapper holding an optional list of strict Notification items.
    """Notification settings for calendar"""

    notifications: Optional[List[Notification]] = Field(
        default=None,
        description="List of notification settings (each requires method and type)",
    )


class PatchCalendarListNotificationSettings(BaseModel):
    # [CLASS DOC]
    # Wrapper holding an optional list of lenient PatchCalendarListNotification items.
    """Notification settings for calendar"""

    notifications: Optional[List[PatchCalendarListNotification]] = Field(
        default=None,
        description="List of notification settings (each requires method and type)",
    )


class ConferenceProperties(BaseModel):
    # [CLASS DOC]
    # Conference properties with allowedConferenceSolutionTypes (defaults to empty list).
    """Conference properties for calendar"""

    allowedConferenceSolutionTypes: Optional[List[str]] = Field(
        default_factory=list, description="List of supported conference solution types"
    )

    @field_validator("allowedConferenceSolutionTypes")
    @classmethod
    def _validate_conference_solution_types(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        # [FUNCTION DOC]
        # HELPER validator: None passes; otherwise every entry must be eventHangout,
        # eventNamedHangout or hangoutsMeet, else ValueError.
        if v is None:
            return v

        allowed_types = {"eventHangout", "eventNamedHangout", "hangoutsMeet"}
        for solution_type in v:
            if solution_type not in allowed_types:
                raise ValueError(
                    f"Invalid conference solution type '{solution_type}'. Must be one of: {', '.join(sorted(allowed_types))}"
                )

        return v


class CalendarListEntry(BaseModel):
    # [CLASS DOC]
    # Calendar list entry resource: kind "calendar#calendarListEntry", etag, id, summary,
    # description, location, timeZone, summaryOverride, colorId, background/foreground colors,
    # hidden (False), selected (True), required accessRole, defaultReminders,
    # notificationSettings, primary (False), deleted (False), conferenceProperties.
    """CalendarListEntry following Google Calendar API v3 structure"""

    kind: str = Field(default="calendar#calendarListEntry", description="Resource type")
    etag: Optional[str] = Field(None, description="ETag of the resource")
    id: str = Field(..., description="Calendar identifier")
    summary: str = Field(..., description="Calendar title")
    description: Optional[str] = Field(None, description="Calendar description")
    location: Optional[str] = Field(None, description="Calendar location")
    timeZone: str = Field(..., description="Calendar timezone")
    summaryOverride: Optional[str] = Field(
        None, description="Custom calendar title override"
    )
    colorId: Optional[str] = Field(None, description="Calendar color ID")
    backgroundColor: Optional[str] = Field(
        None, description="Calendar background color (hex)"
    )
    foregroundColor: Optional[str] = Field(
        None, description="Calendar foreground color (hex)"
    )
    hidden: Optional[bool] = Field(
        False, description="Whether calendar is hidden from list"
    )
    selected: Optional[bool] = Field(
        True, description="Whether calendar is selected in UI"
    )
    accessRole: AccessRole = Field(
        ..., description="User's access level to this calendar"
    )
    defaultReminders: Optional[List[EventReminder]] = Field(
        None, description="Default reminders for events"
    )
    notificationSettings: Optional[NotificationSettings] = Field(
        None, description="Notification settings"
    )
    primary: Optional[bool] = Field(
        False, description="Whether this is the user's primary calendar"
    )
    deleted: Optional[bool] = Field(False, description="Whether calendar is deleted")
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )


class CalendarListInsertRequest(BaseModel):
    # [CLASS DOC]
    # Body for adding a calendar to the user's list: required id plus optional display
    # fields (summaryOverride, colors, hidden default False, selected default True,
    # defaultReminders, notificationSettings).
    """Request model for inserting calendar into user's list"""

    id: str = Field(..., description="Calendar ID to add to user's list")
    summaryOverride: Optional[str] = Field(
        None, description="Custom calendar title override"
    )
    colorId: Optional[str] = Field(None, description="Calendar color ID")
    backgroundColor: Optional[str] = Field(
        None, description="Calendar background color (hex)"
    )
    foregroundColor: Optional[str] = Field(
        None, description="Calendar foreground color (hex)"
    )
    hidden: Optional[bool] = Field(
        False, description="Whether calendar is hidden from list"
    )
    selected: Optional[bool] = Field(
        True, description="Whether calendar is selected in UI"
    )
    defaultReminders: Optional[List[EventReminder]] = Field(
        None, description="Default reminders for events"
    )
    notificationSettings: Optional[NotificationSettings] = Field(
        None, description="Notification settings"
    )


class CalendarListUpdateRequest(BaseModel):
    # [CLASS DOC]
    # Body for full update (PUT): all display fields optional with None defaults, strict reminder/notification types.
    """Request model for updating calendar list entry"""

    summaryOverride: Optional[str] = Field(
        None, description="Custom calendar title override"
    )
    colorId: Optional[str] = Field(None, description="Calendar color ID")
    backgroundColor: Optional[str] = Field(
        None, description="Calendar background color (hex)"
    )
    foregroundColor: Optional[str] = Field(
        None, description="Calendar foreground color (hex)"
    )
    hidden: Optional[bool] = Field(
        None, description="Whether calendar is hidden from list"
    )
    selected: Optional[bool] = Field(
        None, description="Whether calendar is selected in UI"
    )
    defaultReminders: Optional[List[EventReminder]] = Field(
        None, description="Default reminders for events"
    )
    notificationSettings: Optional[NotificationSettings] = Field(
        None, description="Notification settings"
    )


class CalendarListPatchRequest(BaseModel):
    # [CLASS DOC]
    # Body for PATCH: like the update request but uses the lenient Patch reminder/notification models and also accepts conferenceProperties.
    """Request model for updating calendar list entry"""

    summaryOverride: Optional[str] = Field(
        None, description="Custom calendar title override"
    )
    colorId: Optional[str] = Field(None, description="Calendar color ID")
    backgroundColor: Optional[str] = Field(
        None, description="Calendar background color (hex)"
    )
    foregroundColor: Optional[str] = Field(
        None, description="Calendar foreground color (hex)"
    )
    hidden: Optional[bool] = Field(
        None, description="Whether calendar is hidden from list"
    )
    selected: Optional[bool] = Field(
        None, description="Whether calendar is selected in UI"
    )
    defaultReminders: Optional[List[PatchCalendarListEventReminder]] = Field(
        None, description="Default reminders for events"
    )
    notificationSettings: Optional[PatchCalendarListNotificationSettings] = Field(
        None, description="Notification settings"
    )
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )


class CalendarListResponse(BaseModel):
    # [CLASS DOC]
    # List response: kind "calendar#calendarList", etag, nextPageToken, nextSyncToken and items.
    """Response model for calendar list collection"""

    kind: str = Field(
        default="calendar#calendarList", description="Collection resource type"
    )
    etag: Optional[str] = Field(None, description="ETag of the collection")
    nextPageToken: Optional[str] = Field(
        None, description="Token for next page of results"
    )
    nextSyncToken: Optional[str] = Field(None, description="Token for incremental sync")
    items: List[CalendarListEntry] = Field(
        default_factory=list, description="List of calendar entries"
    )


class Channel(BaseModel):
    # [CLASS DOC]
    # Watch channel resource: kind "api#channel", id, resourceId, resourceUri, token,
    # expiration (integer), type (default "web_hook") and required address.
    """Channel resource for watch notifications"""

    kind: str = Field(default="api#channel", description="Resource type identifier")
    id: str = Field(..., description="Channel identifier")
    resourceId: Optional[str] = Field(None, description="Resource being watched")
    resourceUri: Optional[str] = Field(None, description="Resource URI")
    token: Optional[str] = Field(None, description="Verification token")
    expiration: Optional[int] = Field(None, description="Channel expiration time")
    type: str = Field(default="web_hook", description="Channel type")
    address: str = Field(..., description="Notification delivery address")


class WatchParams(BaseModel):
    # [CLASS DOC]
    # Watch params with optional ttl string of positive integer seconds.
    """Watch parameters"""

    ttl: Optional[str] = Field(None, description="Time to live (seconds)")

    @field_validator("ttl")
    @classmethod
    def _validate_ttl(cls, v: Optional[str]) -> Optional[str]:
        # [FUNCTION DOC]
        # HELPER validator: None passes; otherwise stripped value must be digits and > 0; returns stripped string; else ValueError.
        if v is None:
            return v
        s = str(v).strip()
        if not s.isdigit():
            raise ValueError(
                "params.ttl must be an integer string representing seconds"
            )
        if int(s) <= 0:
            raise ValueError("params.ttl must be greater than 0")
        return s


class WatchRequest(BaseModel):
    # [CLASS DOC]
    # Watch request body: non-empty id, type, non-empty https address, optional token and params.
    """Request model for setting up watch notifications"""

    id: str = Field(..., min_length=1, description="Channel identifier (non-empty)")
    type: str = Field(
        ..., description="Channel type: must be 'web_hook' (alias 'webhook' accepted)"
    )
    address: str = Field(
        ..., min_length=1, description="Notification delivery address (HTTPS URL)"
    )
    token: Optional[str] = Field(None, description="Verification token")
    params: Optional[WatchParams] = Field(
        None,
        description="Optional parameters object; supports 'ttl' as string seconds per Google spec",
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        # [FUNCTION DOC]
        # HELPER validator: accepts "web_hook"/"webhook" (case-insensitive) and returns "web_hook"; else ValueError.
        if v is None:
            raise ValueError("type is required")
        s = str(v).strip().lower()
        if s not in ("web_hook", "webhook"):
            raise ValueError("Only channel type 'web_hook' is supported")
        # Normalize to canonical 'web_hook'
        return "web_hook"

    @field_validator("address")
    @classmethod
    def _validate_address(cls, v: str) -> str:
        # [FUNCTION DOC]
        # HELPER validator: requires an https URL with a host; returns trimmed string; else ValueError.
        if v is None:
            raise ValueError("address is required")
        s = str(v).strip()
        parsed = urlparse(s)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Invalid 'address': must be an https URL")
        return s
