# [MODULE DOC]
# FastAPI router for GET /colors (Google Calendar v3 colors resource). Colors are read per database selected by the
# x-database-id header.
"""
Colors API endpoints following Google Calendar API v3 structure
Provides color definitions for calendars and events
"""

import logging
from typing import Any, Dict

from database.managers.color_manager import ColorManager
from fastapi import APIRouter, Header, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/colors", tags=["colors"])


def get_color_manager(database_id: str) -> ColorManager:
    # [FUNCTION DOC]
    # HELPER: Returns a new ColorManager bound to the given database id.
    """Get color manager for the specified database"""
    return ColorManager(database_id)


@router.get("", response_model=Dict[str, Any])
async def get_colors(x_database_id: str = Header(alias="x-database-id")):
    # [FUNCTION DOC]
    # GET /colors handler. Input: x-database-id header (required). Returns the manager's colors response dict (calendar and
    # event palettes). Raises: HTTPException 500 with the error text on any failure.
    """
    Returns the color definitions for calendars and events

    GET /colors

    Returns a global palette of color definitions for calendars and events.
    Color data is dynamically loaded from database with exact Google Calendar API v3 format.
    Colors are global/shared across all users - no user_id required.
    Use POST /api/load-sample-colors to populate database with Google's color definitions.
    """
    try:
        color_manager = get_color_manager(x_database_id)

        colors_response = color_manager.get_colors_response()

        logger.info("Retrieved calendar and event color definitions from database")
        return colors_response

    except Exception as e:
        logger.error(f"Error getting color definitions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )
