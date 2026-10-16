# [MODULE DOC]
# Router exposing the MCP JSON-RPC endpoint POST /mcp for the Calendar API; delegates to handlers.mcp_handler.
"""
MCP (Model Context Protocol) API Router
Handles MCP protocol messages ONLY for Calendar API
Database management APIs are in apis.database_router
"""

import logging

from fastapi import APIRouter, Request, Response
from handlers.mcp_handler import handle_mcp_request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/mcp")
async def handle_mcp(request: Request):
    # [FUNCTION DOC]
    # POST /mcp: passes the raw request to handle_mcp_request. Returns an empty 204 response when the handler returns None
    # (notifications), the model_dump() of a pydantic response, or the response as-is otherwise.
    """Handle MCP protocol messages"""
    response = await handle_mcp_request(request)
    if response is None:
        # For notifications, return 204 No Content
        return Response(content="", status_code=204, headers={"Content-Length": "0"})

    # Check if response is already a dict or needs to be converted
    if hasattr(response, "model_dump"):
        return response.model_dump()
    else:
        return response


# MCP protocol endpoint only - all other routes are at root level (no /api prefix)
