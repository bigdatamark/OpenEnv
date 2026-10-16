# [MODULE DOC]
# Webhook notification service for calendar watch channels. Looks up WatchChannel
# rows via a SQLAlchemy session and POSTs JSON notifications (Google "api#channel"
# shape) to the registered webhook addresses using requests.
"""
Webhook notification service for calendar watch channels
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from database.models.watch_channel import WatchChannel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class NotificationService:
    # [CLASS DOC]
    # Sends webhook notifications for watch channels and maintains channel state
    # (notification counters, deactivation of expired channels) in the database.
    # All public methods swallow exceptions and log them instead of raising.
    """Service for sending webhook notifications to watch channels"""

    def __init__(self, db: Session):
        # [FUNCTION DOC]
        # Stores the SQLAlchemy session used for all queries and commits.
        self.db = db

    def send_notification(
        self,
        channel_id: str,
        event_type: str,
        resource_data: Dict[str, Any],
        resource_state: str = "sync",
    ) -> bool:
        # [FUNCTION DOC]
        # Sends one notification to the active channel with the given id.
        # Inputs: channel_id; event_type (placed in "eventType"); resource_data (placed in
        # "data"); resource_state (placed in "resourceState", default "sync").
        # Behavior: loads the active channel (is_active == "true"); returns False if missing.
        # If the channel is expired it is deactivated and False is returned. Otherwise builds
        # a payload with kind, id, resourceId, resourceUri, eventType, resourceState, a UTC ISO
        # timestamp ending in "Z" and data; sets JSON content type, a User-Agent and the
        # X-Goog-Channel-Token header when the channel has a token; POSTs with a 30s timeout.
        # Returns: True on HTTP 200/201/202/204 (and updates channel stats), else False.
        # Raises: nothing; timeouts, request errors and any other exceptions are logged and give False.
        # Side effects: network request, database commit, logging.
        """
        Send a notification to a specific watch channel

        Args:
            channel_id: The watch channel ID
            event_type: Type of event (e.g., "sync", "exists", "not_exists")
            resource_data: The actual resource data that changed
            resource_state: State of the resource

        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            # Get the watch channel from database
            channel = (
                self.db.query(WatchChannel)
                .filter(WatchChannel.id == channel_id, WatchChannel.is_active == "true")
                .first()
            )

            if not channel:
                logger.warning(f"Watch channel {channel_id} not found or inactive")
                return False

            # Check if channel has expired
            if channel.is_expired():
                logger.info(f"Watch channel {channel_id} has expired")
                self._deactivate_channel(channel)
                return False

            # Prepare notification payload
            notification_payload = {
                "kind": "api#channel",
                "id": channel.id,
                "resourceId": channel.resource_id,
                "resourceUri": channel.resource_uri,
                "eventType": event_type,
                "resourceState": resource_state,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": resource_data,
            }

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Calendar-Notifications/1.0",
            }

            # Add token to headers if present
            if channel.webhook_token:
                headers["X-Goog-Channel-Token"] = channel.webhook_token

            # Send the webhook notification
            response = requests.post(
                channel.webhook_address,
                json=notification_payload,
                headers=headers,
                timeout=30,
            )

            # Check if notification was successful
            if response.status_code in [200, 201, 202, 204]:
                # Update channel statistics
                self._update_channel_stats(channel)
                logger.info(
                    f"Notification sent successfully to {channel.webhook_address}"
                )
                return True
            else:
                logger.error(
                    f"Webhook failed with status {response.status_code}: {response.text}"
                )
                return False

        except requests.exceptions.Timeout:
            logger.error(f"Webhook timeout for channel {channel_id}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Webhook request failed for channel {channel_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending notification: {e}")
            return False

    def notify_acl_change(
        self, calendar_id: str, change_type: str, acl_rule_data: Dict[str, Any]
    ) -> int:
        # [FUNCTION DOC]
        # Notifies every active "acl" watch channel of a calendar about an ACL change.
        # Expired channels are deactivated and skipped; others get send_notification with
        # event type change_type, the ACL rule data and state "sync".
        # Returns: number of successful notifications (0 on any error or when none exist).
        # Side effects: network requests, database commits, logging.
        """
        Send ACL change notifications to all active watch channels for a calendar

        Args:
            calendar_id: The calendar ID
            change_type: Type of change ("insert", "update", "delete")
            acl_rule_data: The ACL rule data

        Returns:
            Number of successful notifications sent
        """
        try:
            # Find all active watch channels for this calendar's ACL
            channels = (
                self.db.query(WatchChannel)
                .filter(
                    WatchChannel.calendar_id == calendar_id,
                    WatchChannel.resource_type == "acl",
                    WatchChannel.is_active == "true",
                )
                .all()
            )
            if not channels:
                logger.debug(
                    f"No active watch channels found for calendar {calendar_id}"
                )
                return 0

            successful_notifications = 0

            for channel in channels:
                # Check if channel has expired
                if channel.is_expired():
                    self._deactivate_channel(channel)
                    continue

                # Send notification
                success = self.send_notification(
                    channel.id, change_type, acl_rule_data, "sync"
                )

                if success:
                    successful_notifications += 1

            logger.info(
                f"Sent {successful_notifications} ACL change notifications for calendar {calendar_id}"
            )
            return successful_notifications

        except Exception as e:
            logger.error(f"Error sending ACL change notifications: {e}")
            return 0

    def notify_settings_change(
        self, user_id: str, change_type: str, setting_data: Dict[str, Any]
    ) -> int:
        # [FUNCTION DOC]
        # Same as notify_acl_change but for the active "settings" watch channels of a user.
        # Returns: number of successful notifications (0 on error or none).
        """
        Send settings change notifications to all active watch channels for a user

        Args:
            user_id: The user ID whose settings changed
            change_type: Type of change ("insert", "update", "delete")
            setting_data: The setting data

        Returns:
            Number of successful notifications sent
        """
        try:
            # Find all active watch channels for this user's settings
            channels = (
                self.db.query(WatchChannel)
                .filter(
                    WatchChannel.user_id == user_id,
                    WatchChannel.resource_type == "settings",
                    WatchChannel.is_active == "true",
                )
                .all()
            )

            if not channels:
                logger.debug(
                    f"No active settings watch channels found for user {user_id}"
                )
                return 0

            successful_notifications = 0

            for channel in channels:
                # Check if channel has expired
                if channel.is_expired():
                    self._deactivate_channel(channel)
                    continue

                # Send notification
                success = self.send_notification(
                    channel.id, change_type, setting_data, "sync"
                )

                if success:
                    successful_notifications += 1

            logger.info(
                f"Sent {successful_notifications} settings change notifications for user {user_id}"
            )
            return successful_notifications

        except Exception as e:
            logger.error(f"Error sending settings change notifications: {e}")
            return 0

    def cleanup_expired_channels(self) -> int:
        # [FUNCTION DOC]
        # Deactivates all active channels whose expires_at is before the current UTC time.
        # Returns: number of channels deactivated; 0 on error (after a rollback).
        # Side effects: database commits, logging.
        """
        Clean up expired watch channels

        Returns:
            Number of channels cleaned up
        """
        try:
            current_time = datetime.utcnow()

            # Find expired channels
            expired_channels = (
                self.db.query(WatchChannel)
                .filter(
                    WatchChannel.expires_at < current_time,
                    WatchChannel.is_active == "true",
                )
                .all()
            )

            cleanup_count = 0
            for channel in expired_channels:
                self._deactivate_channel(channel)
                cleanup_count += 1

            if cleanup_count > 0:
                self.db.commit()
                logger.info(f"Cleaned up {cleanup_count} expired watch channels")

            return cleanup_count

        except Exception as e:
            logger.error(f"Error cleaning up expired channels: {e}")
            self.db.rollback()
            return 0

    def _update_channel_stats(self, channel: WatchChannel):
        # [FUNCTION DOC]
        # HELPER: Sets last_notification_at to now (UTC), increments notification_count and
        # commits. On error logs and rolls back.
        """Update channel statistics after successful notification"""
        try:
            channel.last_notification_at = datetime.utcnow()
            channel.notification_count += 1
            self.db.commit()
        except Exception as e:
            logger.error(f"Error updating channel stats: {e}")
            self.db.rollback()

    def _deactivate_channel(self, channel: WatchChannel):
        # [FUNCTION DOC]
        # HELPER: Sets the channel's is_active to "false" and commits. On error logs and rolls back.
        """Deactivate a watch channel"""
        try:
            channel.is_active = "false"
            self.db.commit()
            logger.info(f"Deactivated watch channel {channel.id}")
        except Exception as e:
            logger.error(f"Error deactivating channel {channel.id}: {e}")
            self.db.rollback()


def get_notification_service(db: Session) -> NotificationService:
    # [FUNCTION DOC]
    # HELPER: Factory returning a new NotificationService bound to the given session.
    """Factory function to get notification service instance"""
    return NotificationService(db)
