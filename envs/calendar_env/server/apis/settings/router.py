# [MODULE DOC]
# FastAPI router for /settings (Google Calendar v3 settings: list, get, watch), scoped by user context from headers.
# A module-level CalendarSessionManager is created but unused.
"""
Calendar Settings API endpoints following Google Calendar API v3 structure
Handles GET and LIST operations for calendar settings
"""

import logging

from database.managers.settings_manager import SettingManager
from database.session_manager import CalendarSessionManager
from fastapi import APIRouter, Depends, HTTPException, Query, status
from middleware.auth import get_user_context
from schemas.settings import (
    Channel,
    SettingItem,
    SettingsListResponse,
    SettingsWatchRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

# Initialize managers
session_manager = CalendarSessionManager()


def get_setting_manager(database_id: str) -> SettingManager:
    # [FUNCTION DOC]
    # HELPER: Returns a new SettingManager for the database id.
    return SettingManager(database_id)


@router.get("", response_model=SettingsListResponse, operation_id="list_settings")
async def list_settings(user_context: tuple[str, str] = Depends(get_user_context)):
    # [FUNCTION DOC]
    # GET /settings: returns a SettingsListResponse with the user's settings and a fixed etag. Raises: HTTPException 500
    # on error.
    """
    Lists all user-visible settings

    GET /settings
    """
    try:
        database_id, user_id = user_context
        manager = get_setting_manager(database_id)

        # Pass user_id to manager
        settings = manager.list_settings(user_id=user_id)

        return SettingsListResponse(items=settings, etag="settings-collection-etag")
    except Exception as e:
        logger.error(f"Error listing settings: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{settingId}", response_model=SettingItem, operation_id="get_settings")
async def get_settings(
    settingId: str, user_context: tuple[str, str] = Depends(get_user_context)
):
    # [FUNCTION DOC]
    # GET /settings/{settingId}: returns the user's setting with that id. Raises: HTTPException 404 if not found, 500 on
    # other errors.
    """
    Returns a setting for the user

    GET /settings/{settingId}
    """
    try:
        database_id, user_id = user_context
        manager = get_setting_manager(database_id)

        logger.info(f"Fetching setting {settingId} for user {user_id}")
        setting = manager.get_setting_by_id(settingId, user_id=user_id)

        if not setting:
            raise HTTPException(
                status_code=404,
                detail=f"Setting {settingId} not found for user {user_id}",
            )
        return setting
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting setting {settingId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/watch", response_model=Channel, operation_id="watch_settings")
async def watch_settings(
    watch_request: SettingsWatchRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    # [FUNCTION DOC]
    # POST /settings/watch: requires non-empty address and id (else 400), creates a notification channel via the manager
    # and returns it. Raises: 400 for ValueError, 500 for unexpected errors. Side effects: persists a watch channel.
    """
    Watch for changes to settings

    POST /settings/watch

    Sets up a notification channel to receive updates when settings change.
    Following the Google Calendar API v3 pattern.
    """
    try:
        database_id, user_id = user_context
        manager = get_setting_manager(database_id)

        logger.info(
            f"Setting up settings watch for user {user_id} with channel {watch_request.id}"
        )

        # Validate the watch request
        if not watch_request.address:
            raise HTTPException(status_code=400, detail="Webhook address is required")

        if not watch_request.id:
            raise HTTPException(status_code=400, detail="Channel ID is required")

        # Create the watch channel
        channel = manager.watch_settings(watch_request, user_id)

        logger.info(
            f"Successfully created settings watch channel {watch_request.id} for user {user_id}"
        )
        return channel

    except ValueError as verr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"{str(verr)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up settings watch: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
