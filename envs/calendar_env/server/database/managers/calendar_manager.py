# [MODULE DOC]
# SQLAlchemy-based operations on calendars (Calendars API): create (with owner/writer
# ACLs), get, update, delete, clear, list (owned + shared via user ACL), and primary
# calendar integrity helpers. Defines ALLOWED_CONFERENCE_SOLUTION_TYPES.
"""
Calendar Manager - Database operations for calendar management using SQLAlchemy
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from database.models import Calendar, User
from database.models.acl import ACLs, Scope
from database.session_utils import get_session, init_database
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Allowed conference solution types per Google Calendar API v3 spec
ALLOWED_CONFERENCE_SOLUTION_TYPES = {
    "eventHangout",
    "eventNamedHangout",
    "hangoutsMeet",
}


class CalendarManager:
    # [CLASS DOC]
    # Manager bound to one database_id. Unlike other managers it opens one long-lived
    # Session at construction (self.session) used by most methods and never closed explicitly;
    # some integrity helpers open their own short-lived sessions instead.
    """Manager for calendar database operations using SQLAlchemy"""

    def __init__(self, database_id: str):
        # [FUNCTION DOC]
        # Stores database_id and opens self.session. Does not create tables.
        # Initialize database on first use
        self.database_id = database_id
        self.session = self.get_session()

    def get_session(self):
        # [FUNCTION DOC]
        # HELPER: returns a new Session from a freshly constructed CalendarSessionManager for this
        # database_id (a new engine per call, since the manager cache is per-instance).
        from database.session_manager import CalendarSessionManager

        return CalendarSessionManager().get_session(self.database_id)

    def _has_acl_role(
        self, user_id: str, calendar: Calendar, allowed_roles: List[str]
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns True if any ACL on the calendar has a "user" scope whose value equals
        # the user's email and a role in allowed_roles; False (with a warning log) if the user
        # does not exist, there is no matching ACL, or no role matches.
        user = self.session.query(User).filter(User.user_id == user_id).first()
        if not user:
            logger.warning(f"No user found with ID: {user_id}")
            return False

        acls = (
            self.session.query(ACLs)
            .join(Scope, ACLs.scope_id == Scope.id)
            .filter(
                ACLs.calendar_id == calendar.calendar_id,
                Scope.type == "user",
                Scope.value == user.email,
            )
            .all()
        )

        if not acls:
            logger.warning(
                f"No ACL found for user {user.email} on calendar {calendar.calendar_id}"
            )
            return False

        for acl in acls:
            if acl.role.value in allowed_roles:
                return True

        logger.warning(
            f"User {user.email} has ACLs but lacks required roles: {allowed_roles}"
        )
        return False

    def create_calendar(self, user_id: str, calendar_data: dict):
        # [FUNCTION DOC]
        # Validates conferenceProperties.allowedConferenceSolutionTypes, inserts a new secondary
        # calendar (UUID id, summary, description, location, timeZone default "UTC", conference
        # properties as JSON) owned by user_id, commits, then creates owner and writer ACLs.
        # Returns the Calendar ORM object.
        # Raises: ValueError for invalid conference types or unknown user (from ACL creation).
        # KNOWN ISSUE: an explicit conferenceProperties of None causes AttributeError; the calendar is committed before ACL creation, so a failed ACL step leaves a calendar without ACLs.
        """Create a new calendar"""
        conf_types = calendar_data.get("conferenceProperties", {}).get(
            "allowedConferenceSolutionTypes"
        )
        if conf_types:
            for ctype in conf_types:
                if ctype not in ALLOWED_CONFERENCE_SOLUTION_TYPES:
                    raise ValueError(f"Invalid conference solution type: {ctype}")
        # This endpoint strictly creates secondary calendars per Google Calendar API semantics.
        # Do not auto-promote to primary here.
        mapped_data = {
            "calendar_id": str(uuid.uuid4()),  # Generate unique calendar ID
            "user_id": user_id,
            "summary": calendar_data.get("summary"),
            "description": calendar_data.get("description"),
            "location": calendar_data.get("location"),
            "time_zone": calendar_data.get("timeZone", "UTC"),
            "conference_properties": json.dumps(
                calendar_data.get("conferenceProperties", {})
            ),
            "is_primary": False,
        }
        # Create Calendar model instance
        new_calendar = Calendar(**mapped_data)
        self.session.add(new_calendar)
        self.session.commit()
        self.session.refresh(new_calendar)

        # Automatically create ACL entry for the calendar owner
        self._create_owner_acl(user_id, new_calendar.calendar_id)

        # Return the created calendar
        return new_calendar

    def _create_owner_acl(self, user_id: str, calendar_id: str):
        # [FUNCTION DOC]
        # HELPER: finds or creates a "user" Scope for the user's email and adds two ACL rows
        # (roles owner and writer, random quoted etags) for the calendar, then commits.
        # Raises: ValueError if the user does not exist.
        """Create ACL entries with owner and writer roles for the calendar creator"""
        from database.models.acl import AclRole, ScopeType

        # Get the user to access their email
        user = self.session.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        # Create or get the scope for this user
        scope = (
            self.session.query(Scope)
            .filter(Scope.type == ScopeType.user, Scope.value == user.email)
            .first()
        )

        if not scope:
            scope = Scope(id=str(uuid.uuid4()), type=ScopeType.user, value=user.email)
            self.session.add(scope)
            self.session.flush()  # Flush to get the scope ID

        # Create the owner ACL entry
        owner_acl = ACLs(
            id=str(uuid.uuid4()),
            calendar_id=calendar_id,
            user_id=user_id,
            scope_id=scope.id,
            role=AclRole.owner,
            etag=f'"{uuid.uuid4()}"',
        )

        # Create the writer ACL entry
        writer_acl = ACLs(
            id=str(uuid.uuid4()),
            calendar_id=calendar_id,
            user_id=user_id,
            scope_id=scope.id,
            role=AclRole.writer,
            etag=f'"{uuid.uuid4()}"',
        )

        self.session.add(owner_acl)
        self.session.add(writer_acl)
        self.session.commit()

        logger.info(
            f"Created owner and writer ACL entries for user {user.email} on calendar {calendar_id}"
        )

    def get_calendar_by_id(
        self, user_id: str, calendar_id: str, allowed_roles: List[str] = None
    ):
        # [FUNCTION DOC]
        # Returns the Calendar with calendar_id owned by user_id.
        # Raises: HTTPException 404 if the calendar does not exist at all, or not for this user;
        # PermissionError if allowed_roles is given and the user lacks them.
        """Get a calendar by ID for a specific user"""
        calendar = (
            self.session.query(Calendar)
            .filter(Calendar.calendar_id == calendar_id)
            .first()
        )
        if not calendar:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calendar with id '{calendar_id}' does not exist",
            )
        calendar = (
            self.session.query(Calendar)
            .filter(Calendar.calendar_id == calendar_id, Calendar.user_id == user_id)
            .first()
        )
        if not calendar:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calendar with id '{calendar_id}' does not exist for user {user_id}",
            )
        if allowed_roles and not self._has_acl_role(user_id, calendar, allowed_roles):
            raise PermissionError(
                f"User '{user_id}' lacks required roles: {allowed_roles}"
            )
        return calendar

    def update_calendar(self, user_id: str, calendar_id: str, update_data: dict):
        # [FUNCTION DOC]
        # Updates summary, description, location, timeZone (only if not None) and
        # conferenceProperties (validated; stored as JSON or None when falsy) from update_data,
        # sets updated_at, commits and returns the refreshed Calendar; None if not found.
        # Raises: ValueError if update_data contains "is_primary" or conference types are invalid.
        # KNOWN ISSUE: user_id is ignored; no ownership or ACL check is performed.
        """Update a calendar (cannot modify primary status)"""
        calendar = (
            self.session.query(Calendar)
            .filter(Calendar.calendar_id == calendar_id)
            .first()
        )
        if not calendar:
            return None

        # Prevent changing primary status via update
        if "is_primary" in update_data:
            raise ValueError("Cannot modify primary calendar status")

        # Update fields
        if "summary" in update_data:
            calendar.summary = update_data["summary"]

        if "description" in update_data:
            calendar.description = update_data["description"]

        if "location" in update_data:
            calendar.location = update_data["location"]

        if "timeZone" in update_data and update_data["timeZone"] is not None:
            calendar.time_zone = update_data["timeZone"]

        if "conferenceProperties" in update_data:
            self._validate_conference_properties(
                update_data.get("conferenceProperties")
            )
            calendar.conference_properties = (
                json.dumps(update_data["conferenceProperties"])
                if update_data["conferenceProperties"]
                else None
            )

        calendar.updated_at = datetime.now(timezone.utc)

        self.session.commit()
        self.session.refresh(calendar)
        return calendar

    def delete_calendar(self, user_id: str, calendar_id: str):
        # [FUNCTION DOC]
        # Deletes a calendar (cascading events/ACLs); returns True, or False if not found.
        # Raises: ValueError for primary calendars; PermissionError if user is not an owner via ACL.
        """Delete a calendar (cannot delete primary calendar)"""
        calendar = (
            self.session.query(Calendar)
            .filter(Calendar.calendar_id == calendar_id)
            .first()
        )
        if not calendar:
            return False
        if calendar.is_primary:
            raise ValueError("Cannot delete primary calendar.")
        if not self._has_acl_role(user_id, calendar, ["owner"]):
            raise PermissionError("Only owners can delete the calendar")
        self.session.delete(calendar)
        self.session.commit()
        return True

    def clear_calendar(self, user_id: str, calendar_id: str):
        # [FUNCTION DOC]
        # Deletes all events of a primary calendar and returns how many were deleted; 0 if the calendar is not found.
        # Raises: PermissionError if the user lacks owner/writer ACL; ValueError if not primary.
        """Clear all events from a calendar and return the number of events deleted"""
        calendar = (
            self.session.query(Calendar)
            .filter(Calendar.calendar_id == calendar_id)
            .first()
        )
        if not calendar:
            return 0
        if not self._has_acl_role(user_id, calendar, ["owner", "writer"]):
            raise PermissionError(
                "User does not have permission to clear this calendar"
            )
        if not calendar.is_primary:
            raise ValueError("Can only clear primary calendars")
        count = len(calendar.events)
        for event in calendar.events:
            self.session.delete(event)
        self.session.commit()
        return count

    def list_calendars(self, user_id: str):
        # [FUNCTION DOC]
        # Returns the union of calendars owned by user_id and calendars shared via an ACL with a
        # "user" scope matching the user's email; empty list if the user does not exist.
        """List all calendars for a user"""
        user = self.session.query(User).filter(User.user_id == user_id).first()
        if not user:
            return []
        owned = self.session.query(Calendar).filter(Calendar.user_id == user_id)
        shared = (
            self.session.query(Calendar)
            .join(ACLs, Calendar.calendar_id == ACLs.calendar_id)
            .join(Scope, ACLs.scope_id == Scope.id)
            .filter(Scope.type == "user", Scope.value == user.email)
        )
        return owned.union(shared).all()

    def get_primary_calendar(self, user_id: str):
        # [FUNCTION DOC]
        # Returns the user's primary Calendar or None.
        return (
            self.session.query(Calendar)
            .filter_by(user_id=user_id, is_primary=True)
            .first()
        )

    def _format_calendar_response(self, calendar: Calendar) -> Dict:
        # [FUNCTION DOC]
        # HELPER: returns a "calendar#calendar" dict with etag "etag-<id>-<updated_at iso>", id,
        # summary, timeZone, and description/location/conferenceProperties (JSON-decoded, ignored
        # if invalid) when present.
        """Format calendar model for API response"""
        formatted = {
            "kind": "calendar#calendar",
            "etag": f"etag-{calendar.calendar_id}-{calendar.updated_at.isoformat() if calendar.updated_at else ''}",
            "id": calendar.calendar_id,
            "summary": calendar.summary,
            "timeZone": calendar.time_zone,
        }

        # Add optional fields if present
        if calendar.description:
            formatted["description"] = calendar.description

        if calendar.location:
            formatted["location"] = calendar.location
        if calendar.conference_properties:
            try:
                formatted["conferenceProperties"] = json.loads(
                    calendar.conference_properties
                )
            except:
                pass

        return formatted

    def ensure_primary_calendar_constraint(self, user_id: str) -> bool:
        # [FUNCTION DOC]
        # Using a separate session: if the user has no primary calendar, marks the oldest one
        # primary; if several, keeps the oldest primary and demotes the rest; commits. Returns
        # True in all non-error cases. Rolls back and re-raises on error.
        """Ensure user has exactly one primary calendar"""
        session = get_session(self.database_id)
        try:
            # Get all primary calendars for user
            primary_calendars = (
                session.query(Calendar)
                .filter(Calendar.user_id == user_id, Calendar.is_primary == True)
                .all()
            )

            # Case 1: No primary calendar - make first calendar primary
            if not primary_calendars:
                first_calendar = (
                    session.query(Calendar)
                    .filter(Calendar.user_id == user_id)
                    .order_by(Calendar.created_at.asc())
                    .first()
                )

                if first_calendar:
                    first_calendar.is_primary = True
                    session.commit()
                    logger.info(
                        f"Made calendar {first_calendar.calendar_id} primary for user {user_id}"
                    )
                    return True
                else:
                    # User has no calendars - this is valid
                    return True

            # Case 2: Multiple primary calendars - keep oldest, make others secondary
            elif len(primary_calendars) > 1:
                # Sort by creation date, keep the oldest as primary
                primary_calendars.sort(key=lambda c: c.created_at)
                primary_calendar = primary_calendars[0]

                # Make all others secondary
                for calendar in primary_calendars[1:]:
                    calendar.is_primary = False
                    logger.warning(
                        f"Made calendar {calendar.calendar_id} secondary for user {user_id}"
                    )

                session.commit()
                logger.info(
                    f"Ensured single primary calendar {primary_calendar.calendar_id} for user {user_id}"
                )
                return True

            # Case 3: Exactly one primary calendar - already correct
            return True

        except Exception as e:
            session.rollback()
            logger.error(
                f"Error ensuring primary calendar constraint for user {user_id}: {e}"
            )
            raise
        finally:
            session.close()

    def _validate_conference_properties(
        self, conference_properties: Optional[Dict]
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: no-op for falsy input or missing list; raises ValueError if
        # allowedConferenceSolutionTypes is not a list or has entries outside the allowed set
        # (None/"" entries are tolerated).
        """Validate conferenceProperties.allowedConferenceSolutionTypes values.

        Raises ValueError if any provided value is not allowed per API spec.
        """
        if not conference_properties:
            return
        allowed_list = conference_properties.get("allowedConferenceSolutionTypes")
        if allowed_list is None:
            return
        if not isinstance(allowed_list, list):
            raise ValueError(
                "conferenceProperties.allowedConferenceSolutionTypes must be a list if provided"
            )
        invalid = [
            item
            for item in allowed_list
            if item not in ALLOWED_CONFERENCE_SOLUTION_TYPES and item not in [None, ""]
        ]
        if invalid:
            allowed_sorted = sorted(ALLOWED_CONFERENCE_SOLUTION_TYPES)
            raise ValueError(
                "Invalid values for conferenceProperties.allowedConferenceSolutionTypes: "
                f"{invalid}. Allowed values are: {allowed_sorted}"
            )

    def validate_primary_calendar_integrity(self, user_id: str) -> Dict[str, any]:
        # [FUNCTION DOC]
        # Returns {user_id, total_calendars, primary_calendars_count, is_valid (exactly one
        # primary when the user has calendars), primary_calendar_ids}. Re-raises DB errors.
        """Validate primary calendar integrity for a user"""
        session = get_session(self.database_id)
        try:
            primary_calendars = (
                session.query(Calendar)
                .filter(Calendar.user_id == user_id, Calendar.is_primary == True)
                .all()
            )

            total_calendars = (
                session.query(Calendar).filter(Calendar.user_id == user_id).count()
            )

            result = {
                "user_id": user_id,
                "total_calendars": total_calendars,
                "primary_calendars_count": len(primary_calendars),
                "is_valid": len(primary_calendars) == 1
                if total_calendars > 0
                else True,
                "primary_calendar_ids": [c.calendar_id for c in primary_calendars],
            }

            return result

        except Exception as e:
            logger.error(
                f"Error validating primary calendar integrity for user {user_id}: {e}"
            )
            raise
        finally:
            session.close()

    def _ensure_user_has_primary_calendar(
        self, user_id: str, session: Session = None
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: on the given session (or a new one), makes the oldest calendar primary if none
        # is, or demotes all but the oldest primary if several are. Does not commit.
        # KNOWN ISSUE: when it opens its own session it closes it without committing, so changes are lost in that mode.
        """Private method to ensure user has exactly one primary calendar

        This enforces the business rule: If user has calendars, exactly one must be primary
        """
        close_session = session is None
        if session is None:
            session = get_session(self.database_id)

        try:
            # Get all calendars for user
            all_calendars = (
                session.query(Calendar)
                .filter(Calendar.user_id == user_id)
                .order_by(Calendar.created_at.asc())
                .all()
            )

            if not all_calendars:
                # User has no calendars - nothing to enforce
                return

            # Get primary calendars
            primary_calendars = [c for c in all_calendars if c.is_primary]

            if len(primary_calendars) == 0:
                # No primary calendar - make the oldest one primary
                oldest_calendar = all_calendars[0]
                oldest_calendar.is_primary = True
                logger.info(
                    f"Made calendar {oldest_calendar.calendar_id} primary for user {user_id} (no primary existed)"
                )

            elif len(primary_calendars) > 1:
                # Multiple primary calendars - keep oldest, make others secondary
                primary_calendars.sort(key=lambda c: c.created_at)
                primary_calendar = primary_calendars[0]

                for calendar in primary_calendars[1:]:
                    calendar.is_primary = False
                    logger.warning(
                        f"Made calendar {calendar.calendar_id} secondary for user {user_id} (multiple primaries existed)"
                    )

                logger.info(
                    f"Kept calendar {primary_calendar.calendar_id} as primary for user {user_id}"
                )

            # If exactly one primary exists, no action needed

        except Exception as e:
            logger.error(f"Error ensuring primary calendar for user {user_id}: {e}")
            raise
        finally:
            if close_session:
                session.close()

    def enforce_primary_calendar_constraint_for_all_users(self) -> Dict[str, int]:
        # [FUNCTION DOC]
        # For every user that owns calendars, validates integrity and fixes invalid ones via
        # _ensure_user_has_primary_calendar on a shared session, then commits. Returns stats
        # {users_processed, users_fixed, calendars_made_primary, calendars_made_secondary}.
        # KNOWN ISSUE: calendars_made_primary/secondary are never incremented (always 0).
        """Enforce primary calendar constraint for all users in the database

        Returns:
            Dict with statistics about fixes applied
        """
        session = get_session(self.database_id)
        try:
            # Get all unique user IDs
            user_ids = session.query(Calendar.user_id).distinct().all()
            user_ids = [uid[0] for uid in user_ids]

            stats = {
                "users_processed": 0,
                "users_fixed": 0,
                "calendars_made_primary": 0,
                "calendars_made_secondary": 0,
            }

            for user_id in user_ids:
                stats["users_processed"] += 1

                # Check if user needs fixing
                validation_result = self.validate_primary_calendar_integrity(user_id)

                if not validation_result["is_valid"]:
                    stats["users_fixed"] += 1
                    logger.info(
                        f"Fixing primary calendar constraint for user {user_id}"
                    )

                    # Fix the constraint
                    self._ensure_user_has_primary_calendar(user_id, session)

            session.commit()
            return stats

        except Exception as e:
            session.rollback()
            logger.error(f"Error enforcing primary calendar constraints: {e}")
            raise
        finally:
            session.close()
