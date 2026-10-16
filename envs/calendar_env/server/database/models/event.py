# [MODULE DOC]
# SQLAlchemy ORM models for events and their related tables: attendees, attachments,
# office locations, working location properties, conference data, birthday properties,
# extended properties, reminders and recurring event series, plus related enums and a
# before_update listener that makes Event.eventType immutable.
"""
Event database model
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    event,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import object_session, relationship

from .base import Base


class ExtendedPropertyScope(enum.Enum):
    # [CLASS DOC]
    # Enum of extended property scopes: private, shared.
    private = "private"
    shared = "shared"


class EventTypeEnum(enum.Enum):
    # [CLASS DOC]
    # Enum of Google event types: birthday, default, focusTime, fromGmail, outOfOffice, workingLocation.
    birthday = "birthday"
    default = "default"
    focusTime = "focusTime"
    fromGmail = "fromGmail"
    outOfOffice = "outOfOffice"
    workingLocation = "workingLocation"


class ReminderMethodEnum(enum.Enum):
    # [CLASS DOC]
    # Enum of reminder methods: email, popup.
    email = "email"
    popup = "popup"


class RecurrenceFrequency(enum.Enum):
    # [CLASS DOC]
    # Enum of RFC 5545 RRULE FREQ values (SECONDLY .. YEARLY).
    """RRULE frequency values from RFC 5545"""

    SECONDLY = "SECONDLY"
    MINUTELY = "MINUTELY"
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


class RecurrenceWeekday(enum.Enum):
    # [CLASS DOC]
    # Enum of RRULE BYDAY weekday codes MO..SU.
    """Days of the week for RRULE BYDAY parameter"""

    MO = "MO"  # Monday
    TU = "TU"  # Tuesday
    WE = "WE"  # Wednesday
    TH = "TH"  # Thursday
    FR = "FR"  # Friday
    SA = "SA"  # Saturday
    SU = "SU"  # Sunday


class Attendees(Base):
    # [CLASS DOC]
    # ORM row (table "attendees") for an event attendee: attendees_id PK, comment,
    # displayName, additionalGuests, optional/resource flags, responseStatus (default
    # "needsAction"), event_id FK (cascade delete), optional user_id FK; relationships to Event and User.
    """Attendees database model"""

    __tablename__ = "attendees"

    attendees_id = Column(String(255), primary_key=True, nullable=False)
    comment = Column(String(255), nullable=True)
    displayName = Column(String(255), nullable=True)
    additionalGuests = Column(Integer, default=0)
    optional = Column(Boolean, default=False, nullable=False)
    resource = Column(Boolean, default=False, nullable=False)
    responseStatus = Column(String(50), nullable=False, default="needsAction")
    event_id = Column(
        String(255), ForeignKey("events.event_id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(String(255), ForeignKey("users.user_id"), nullable=True)

    # Relationships
    event = relationship("Event", back_populates="attendees")
    user = relationship("User", back_populates="attendees")


class Attachment(Base):
    # [CLASS DOC]
    # ORM row (table "attachments") linking a file URL (up to 2000 chars) to an event.
    """Attachment model for storing file URLs linked to events"""

    __tablename__ = "attachments"

    attachment_id = Column(String(255), primary_key=True, nullable=False)
    event_id = Column(
        String(255), ForeignKey("events.event_id", ondelete="CASCADE"), nullable=False
    )
    file_url = Column(String(2000), nullable=False)  # long enough for Drive links

    # Relationship
    event = relationship("Event", back_populates="attachments")

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns a debug string with attachment_id, event_id and file_url.
        return f"<Attachment(attachment_id='{self.attachment_id}', event_id='{self.event_id}', file_url='{self.file_url}')>"


class OfficeLocation(Base):
    # [CLASS DOC]
    # ORM row (table "office_locations") describing an office place: id, optional building,
    # desk, floor and floor section ids, required label; one-to-one back reference to WorkingLocationProperties.
    __tablename__ = "office_locations"

    id = Column(String(255), primary_key=True, nullable=False)
    buildingId = Column(String(255), nullable=True)
    deskId = Column(String(255), nullable=True)
    floorId = Column(String(255), nullable=True)
    floorSectionId = Column(String(255), nullable=True)
    label = Column(String(255), nullable=False)

    workingLocation = relationship(
        "WorkingLocationProperties", back_populates="officeLocation", uselist=False
    )


class WorkingLocationProperties(Base):
    # [CLASS DOC]
    # ORM row (table "working_location_properties") for a workingLocation event: id, event_id
    # FK, type ("homeOffice", "officeLocation" or "customLocation"), homeOffice JSON,
    # customLocationLabel, unique optional officeLocationId FK; relationships to OfficeLocation and Event.
    """Working location properties model for events"""

    __tablename__ = "working_location_properties"

    working_location_id = Column(String(255), primary_key=True, nullable=False)
    event_id = Column(
        String(255), ForeignKey("events.event_id", ondelete="CASCADE"), nullable=False
    )
    type = Column(
        String(50),
        nullable=False,
        doc=(
            "Type of working location. Required. Values: "
            "'homeOffice', 'officeLocation', 'customLocation'"
        ),
    )
    homeOffice = Column(
        JSON,
        nullable=True,
        doc="If present, specifies that the user is working at home.",
    )
    customLocationLabel = Column(
        String(255),
        nullable=True,
        doc="Optional extra label for custom location additional information.",
    )
    officeLocationId = Column(
        String(255), ForeignKey("office_locations.id"), unique=True, nullable=True
    )

    # Relationship
    officeLocation = relationship(
        "OfficeLocation", back_populates="workingLocation", uselist=False
    )
    event = relationship("Event", back_populates="workingLocationProperties")

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns a debug string with working_location_id, event_id and type.
        return f"<WorkingLocationProperties(working_location_id='{self.working_location_id}', event_id='{self.event_id}', type='{self.type}')>"


class ConferenceData(Base):
    # [CLASS DOC]
    # ORM row (table "conference_data", one per event): solution type/name/icon, create
    # request id/type/status code, entry points JSON, notes, signature, legacy meeting_uri and label.
    __tablename__ = "conference_data"

    id = Column(String(50), primary_key=True, nullable=False)
    event_id = Column(
        String(255),
        ForeignKey("events.event_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Conference solution fields
    solution_type = Column(String(100), nullable=True)  # conferenceSolution.key.type
    solution_name = Column(String(255), nullable=True)  # conferenceSolution.name
    solution_icon_uri = Column(String(500), nullable=True)  # conferenceSolution.iconUri

    # Create request fields
    request_id = Column(String(255), nullable=True)  # createRequest.requestId
    create_solution_type = Column(
        String(100), nullable=True
    )  # createRequest.conferenceSolution.key.type
    status_code = Column(String(50), nullable=True)  # createRequest.status.statusCode

    # Entry points (stored as JSON array for flexibility)
    entry_points = Column(JSON, nullable=True)  # entryPoints array

    # Additional conference fields
    notes = Column(Text, nullable=True)  # notes
    signature = Column(String(500), nullable=True)  # signature

    # Legacy field for backward compatibility
    meeting_uri = Column(String(500), nullable=True)  # Primary meeting URI
    label = Column(String(255), nullable=True)  # Deprecated - use solution_name

    # Relationships
    event = relationship("Event", back_populates="conferenceData")


class BirthdayProperties(Base):
    # [CLASS DOC]
    # ORM row (table "birthday_properties", one per event) with a type string (default "birthday").
    __tablename__ = "birthday_properties"

    id = Column(String(50), primary_key=True, nullable=False)
    event_id = Column(
        String(255),
        ForeignKey("events.event_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    type = Column(String(50), nullable=False, default="birthday")  # Default enforced

    # Relationship back to event
    event = relationship("Event", back_populates="birthdayProperties")


class ExtendedProperty(Base):
    # [CLASS DOC]
    # ORM row (table "extended_properties"): id, event_id FK, scope (private/shared) and properties JSON.
    __tablename__ = "extended_properties"

    id = Column(String(255), primary_key=True, nullable=False)
    event_id = Column(
        String(255), ForeignKey("events.event_id", ondelete="CASCADE"), nullable=False
    )
    scope = Column(Enum(ExtendedPropertyScope), nullable=False)  # private or shared
    properties = Column(JSON, nullable=True)

    event = relationship("Event", back_populates="extendedProperties")


class Reminder(Base):
    # [CLASS DOC]
    # ORM row (table "reminders"): id, event_id FK, method (email/popup), minutes before the event, use_default flag (default True).
    __tablename__ = "reminders"

    id = Column(String(255), primary_key=True, nullable=False)
    event_id = Column(
        String(255), ForeignKey("events.event_id", ondelete="CASCADE"), nullable=False
    )

    method = Column(
        Enum(ReminderMethodEnum, name="reminder_method_enum"), nullable=False
    )
    minutes = Column(
        Integer, nullable=False, doc="Minutes before the event to trigger reminder"
    )
    use_default = Column(Boolean, nullable=False, default=True)

    event = relationship("Event", back_populates="reminders")


class Event(Base):
    # [CLASS DOC]
    # ORM row (table "events"): event_id PK, calendar_id FK, owner user_id FK, organizer
    # fields, optional recurring_event_id FK, summary, description, location, naive
    # start/end datetimes with optional timezone names, recurrence text (RRULE/JSON), status
    # (default "confirmed"), visibility (default "default"), color_id, eventType (default
    # "default", immutable), focusTime/outOfOffice/source JSON, guest permission flags,
    # sequence, transparency (default "opaque"), iCalUID, privateCopy, locked, hangoutLink,
    # originalStartTime date/dateTime/timeZone, timestamps, and cascading relationships to
    # attendees, attachments, conference data, birthday properties, extended properties
    # (eager joined), reminders and working location properties.
    # KNOWN ISSUE: color_id is a string palette id (e.g. "1") but the foreign key points to colors.id, the autoincrement integer key, not colors.color_id.
    """Event database model"""

    __tablename__ = "events"

    event_id = Column(String(255), primary_key=True, nullable=False)
    calendar_id = Column(
        String(255), ForeignKey("calendars.calendar_id"), nullable=False
    )
    user_id = Column(
        String(255), ForeignKey("users.user_id"), nullable=False, index=True
    )

    # Organizer information stored separately from user_id
    organizer_id = Column(String(255), nullable=True, doc="Organizer user ID")
    organizer_email = Column(String(255), nullable=True, doc="Organizer email address")
    organizer_display_name = Column(
        String(255), nullable=True, doc="Organizer display name"
    )
    organizer_self = Column(
        Boolean, default=False, doc="Whether the organizer is the current user"
    )

    # Foreign key to recurring event (if this is an instance of a recurring event)
    recurring_event_id = Column(
        String(255),
        ForeignKey("recurring_events.recurring_event_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    summary = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    location = Column(String(500), nullable=True)
    start_datetime = Column(DateTime, nullable=False)
    end_datetime = Column(DateTime, nullable=False)
    start_timezone = Column(String(100), nullable=True)
    end_timezone = Column(String(100), nullable=True)
    recurrence = Column(Text, nullable=True)  # RRULE format
    status = Column(String(50), nullable=False, default="confirmed")
    visibility = Column(String(50), nullable=False, default="default")
    color_id = Column(
        String(50), ForeignKey("colors.id", ondelete="SET NULL"), nullable=True
    )

    eventType = Column(
        Enum(EventTypeEnum, name="event_type_enum"),
        nullable=False,
        default=EventTypeEnum.default,
        doc=(
            "Specific type of the event. Cannot be modified after creation. "
            "Allowed values: 'birthday', 'default', 'focusTime', "
            "'fromGmail', 'outOfOffice', 'workingLocation'."
        ),
    )

    focusTimeProperties = Column(
        JSON,
        nullable=True,
        doc="Focus Time event data. Used if eventType is focusTime.",
    )
    guestsCanInviteOthers = Column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether attendees other than the organizer can invite others to the event.",
    )
    guestsCanModify = Column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether attendees other than the organizer can modify the event.",
    )
    guestsCanSeeOtherGuests = Column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether attendees other than the organizer can see who the event's attendees are.",
    )
    outOfOfficeProperties = Column(
        JSON,
        nullable=True,
        doc="Out of office event data. Used if eventType is outOfOffice.",
    )
    sequence = Column(
        Integer,
        nullable=True,
        doc="Sequence number as per iCalendar.",
    )
    source = Column(
        JSON,
        nullable=True,
        doc=(
            "Source information for the event. Contains 'title' and 'url' properties. "
            "URL scheme must be HTTP or HTTPS."
        ),
    )

    transparency = Column(
        String(50),
        nullable=True,
        default="opaque",
        doc="Whether event blocks time. Values: 'opaque', 'transparent'",
    )
    iCalUID = Column(
        String(255), nullable=True, doc="Event iCalendar UID for external integration"
    )
    privateCopy = Column(
        Boolean,
        nullable=True,
        default=False,
        doc="Whether this is a private copy of the event",
    )
    locked = Column(
        Boolean,
        nullable=True,
        default=False,
        doc="Whether the event is locked against changes",
    )
    hangoutLink = Column(String(500), nullable=True, doc="Hangout video call link")

    # Original start time fields for recurring events and event tracking
    originalStartTime_date = Column(
        Date,
        nullable=True,
        doc="The date, in the format 'yyyy-mm-dd', if this is an all-day event.",
    )
    originalStartTime_dateTime = Column(
        DateTime,
        nullable=True,
        doc="The time, as a combined date-time value (formatted according to RFC3339). A time zone offset is required unless a time zone is explicitly specified in timeZone.",
    )
    originalStartTime_timeZone = Column(
        String(100),
        nullable=True,
        doc="The time zone in which the time is specified. (Formatted as an IANA Time Zone Database name, e.g. 'Europe/Zurich'.) For recurring events this field is required and specifies the time zone in which the recurrence is expanded. For single events this field is optional and indicates a custom time zone for the event start/end.",
    )

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    calendar = relationship("Calendar", back_populates="events")
    user = relationship("User")
    recurring_event = relationship("RecurringEvent", back_populates="event_instances")
    attendees = relationship(
        "Attendees", back_populates="event", cascade="all, delete-orphan"
    )
    attachments = relationship(
        "Attachment", back_populates="event", cascade="all, delete-orphan"
    )
    conferenceData = relationship(
        "ConferenceData",
        uselist=False,
        back_populates="event",
        cascade="all, delete-orphan",
    )
    birthdayProperties = relationship(
        "BirthdayProperties",
        uselist=False,
        back_populates="event",
        cascade="all, delete-orphan",
    )
    color = relationship("Color", back_populates="events")
    extendedProperties = relationship(
        "ExtendedProperty",
        back_populates="event",
        cascade="all, delete-orphan",
        lazy="joined",
    )
    reminders = relationship(
        "Reminder", back_populates="event", cascade="all, delete-orphan"
    )
    workingLocationProperties = relationship(
        "WorkingLocationProperties",
        uselist=False,
        back_populates="event",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns a debug string with event_id, user_id, summary and calendar_id.
        return f"<Event(event_id='{self.event_id}', user_id='{self.user_id}', summary='{self.summary}', calendar_id='{self.calendar_id}')>"


class RecurringEvent(Base):
    # [CLASS DOC]
    # ORM row (table "recurring_events") for a recurring series: recurring_event_id PK,
    # original_recurrence text, and event_instances relationship (cascade delete).
    """
    RecurringEvent database model
    """

    __tablename__ = "recurring_events"

    # Primary key
    recurring_event_id = Column(String(255), primary_key=True, nullable=False)

    # Original recurrence string for reference
    original_recurrence = Column(
        Text, nullable=True, doc="Original recurrence string array as provided by user"
    )

    # One-to-many relationship with Event instances
    event_instances = relationship(
        "Event", back_populates="recurring_event", cascade="all, delete-orphan"
    )


@event.listens_for(Event, "before_update", propagate=True)
def prevent_event_type_update(mapper, connection, target):
    # [FUNCTION DOC]
    # SQLAlchemy before_update listener on Event: reads the stored eventType from the DB via
    # the target's session and raises IntegrityError if the new value differs (compared as
    # enum value strings). No-op if the object has no session or no stored type.
    # Side effects: issues a SELECT during flush.
    """Ensure eventType is immutable after creation"""
    session = object_session(target)

    if session:
        db_event_type = (
            session.query(Event.eventType)
            .filter(Event.event_id == target.event_id)
            .scalar()
        )

        if db_event_type:
            # Normalize both values to strings for comparison
            db_type_str = (
                db_event_type.value
                if hasattr(db_event_type, "value")
                else str(db_event_type)
            )
            target_type_str = (
                target.eventType.value
                if hasattr(target.eventType, "value")
                else str(target.eventType)
            )

            if db_type_str != target_type_str:
                raise IntegrityError(
                    None,
                    None,
                    f"eventType cannot be modified once set (was '{db_type_str}', tried to change to '{target_type_str}')",
                )
