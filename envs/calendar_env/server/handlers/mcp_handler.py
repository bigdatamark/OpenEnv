# [MODULE DOC]
# JSON-RPC (MCP protocol) request handling for the calendar server's FastAPI /mcp route:
# dispatches initialize, tools/list, tools/call and stub resources methods.
"""
MCP Protocol Handler using official MCP library
"""

import json
import logging
from typing import Optional, Union

from database.managers.user_manager import UserManager
from fastapi import Request
from handlers.tool_handlers import MCP_TOOLS_LIST, TOOL_HANDLERS
from mcp.types import (
    CallToolResult,
    Implementation,
    InitializeResult,
    JSONRPCRequest,
    JSONRPCResponse,
    ListToolsResult,
    ServerCapabilities,
    TextContent,
)

logger = logging.getLogger(__name__)


async def handle_mcp_request(request: Request) -> Optional[JSONRPCResponse]:
    # [FUNCTION DOC]
    # Parses the request JSON body and dispatches by "method". Returns None for
    # notifications (no id) and for "notifications/initialized" (router turns this into 204).
    # For requests: initialize, tools/list, tools/call go to dedicated handlers;
    # resources/list and resources/templates/list return empty lists; unknown methods return a
    # JSONRPCResponse whose result is {"error": "Method not found: ..."} (not a JSON-RPC error object).
    # On invalid JSON returns result {"error": "Invalid JSON"} with id None; on other
    # exceptions re-reads the body to recover the id and returns result {"error": "Internal error: ..."}.
    # Side effects: logging.
    # KNOWN ISSUE: errors are reported inside "result" instead of a JSON-RPC "error" member; and the bare except hides failures when re-reading the body.
    """Handle MCP protocol messages from FastAPI Request"""
    try:
        # Parse JSON body from FastAPI Request
        body = await request.json()

        # Extract JSON-RPC fields
        jsonrpc = body.get("jsonrpc", "2.0")
        method = body.get("method")
        params = body.get("params")
        request_id = body.get("id")  # None for notifications

        # Check if it's a notification (no id field)
        is_notification = request_id is None

        logger.info(
            f"Received MCP request: method={method}, id={request_id}, is_notification={is_notification}"
        )

        # Handle notifications/initialized - return 204 No Content
        if method == "notifications/initialized":
            logger.info("MCP client initialized")
            return None  # This triggers 204 response in router

        # Create JSONRPCRequest object for requests with id
        if not is_notification:
            mcp_request = JSONRPCRequest(
                jsonrpc=jsonrpc, method=method, params=params, id=request_id
            )

            if method == "initialize":
                return await handle_initialize(mcp_request)
            elif method == "tools/list":
                return await handle_tools_list(mcp_request)
            elif method == "tools/call":
                return await handle_tools_call(mcp_request, request)
            elif method == "resources/list":
                logger.info("Resources list request received")
                return JSONRPCResponse(
                    jsonrpc="2.0", id=request_id, result={"resources": []}
                )
            elif method == "resources/templates/list":
                logger.info("Resource templates list request received")
                return JSONRPCResponse(
                    jsonrpc="2.0", id=request_id, result={"resourceTemplates": []}
                )
            else:
                return JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request_id,
                    result={"error": f"Method not found: {method}"},
                )
        else:
            # Handle other notifications if needed
            logger.info(f"Received notification: {method}")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {e}")
        return JSONRPCResponse(jsonrpc="2.0", id=None, result={"error": "Invalid JSON"})
    except Exception as e:
        logger.error(f"MCP error: {e}")
        # Try to get request_id if available
        try:
            body = await request.json()
            request_id = body.get("id")
        except:
            request_id = None

        return JSONRPCResponse(
            jsonrpc="2.0", id=request_id, result={"error": f"Internal error: {str(e)}"}
        )


async def handle_initialize(request: JSONRPCRequest) -> JSONRPCResponse:
    # [FUNCTION DOC]
    # Returns a JSONRPCResponse with an InitializeResult: protocol "2024-11-05", empty
    # tools/logging/prompts/resources capabilities, serverInfo "calendar-api-clone-server" 1.0.0.
    """Handle MCP initialize request"""
    result = InitializeResult(
        protocolVersion="2024-11-05",
        capabilities=ServerCapabilities(tools={}, logging={}, prompts={}, resources={}),
        serverInfo=Implementation(name="calendar-api-clone-server", version="1.0.0"),
    )
    return JSONRPCResponse(
        jsonrpc="2.0", id=request.id, result=result.model_dump(exclude_none=True)
    )


async def handle_tools_list(request: JSONRPCRequest) -> JSONRPCResponse:
    # [FUNCTION DOC]
    # Returns a JSONRPCResponse whose result is ListToolsResult over MCP_TOOLS_LIST. Logs the count.
    """Handle MCP tools/list request"""
    logger.info(
        f"Tools list request received. Calendar tools: {len(MCP_TOOLS_LIST)} tools"
    )

    # Use the official ListToolsResult class
    result = ListToolsResult(tools=MCP_TOOLS_LIST)
    return JSONRPCResponse(
        jsonrpc="2.0", id=request.id, result=result.model_dump(exclude_none=True)
    )


async def handle_tools_call(
    request: JSONRPCRequest, fastapi_request: Request
) -> JSONRPCResponse:
    # [FUNCTION DOC]
    # Executes a tools/call: reads name/arguments from params, requires header
    # x-database-id, resolves the user from header x-access-token via
    # UserManager.get_user_by_access_token, or falls back to the first user in the DB when no
    # token is given. Unknown tools produce a CallToolResult text "Unknown tool: ...".
    # Otherwise awaits TOOL_HANDLERS[name](name, arguments, database_id, user_id) and wraps its
    # "text" in a CallToolResult (isError is not propagated to the MCP result).
    # All exceptions (missing db id, user not found, handler errors) are caught and
    # returned as a CallToolResult text "Tool execution failed: ...". Always returns a JSONRPCResponse.
    # Side effects: DB reads/writes via handlers, logging.
    # KNOWN ISSUE: the CallToolResult never sets isError, so clients cannot distinguish failures from successes except by text.
    """Handle MCP tools/call request"""
    try:
        params = request.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"Tool call: {tool_name} with arguments: {arguments}")

        # Extract database_id from headers (following Ola pattern)
        headers = dict(fastapi_request.headers)
        database_id = headers.get("x-database-id")

        # Require database_id early (needed for user manager)
        if not database_id:
            raise ValueError("database_id is required")

        user_manager = UserManager(database_id)

        # Extract access token from headers
        access_token = headers.get("x-access-token")
        if not access_token:
            user = user_manager.get_first_user_from_db()
            if not user:
                raise ValueError("User not found")
        else:
            # Validate user using static token mapped to user
            user = user_manager.get_user_by_access_token(access_token)
            if not user:
                raise ValueError(f"User not found for given access token")

        # user_id is used to scope operations to this user's calendars
        user_id = user["id"]

        # Check if we have a handler for this tool
        if tool_name not in TOOL_HANDLERS:
            result = CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")]
            )
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                result=result.model_dump(exclude_none=True),
            )

        # Execute the tool using the generic handler
        tool_result = await TOOL_HANDLERS[tool_name](
            tool_name, arguments, database_id, user_id
        )

        # Format the response
        if tool_result.get("isError", False):
            text_content = tool_result.get("text", "Unknown error")
        else:
            text_content = tool_result.get("text", "Operation completed successfully")

        result = CallToolResult(content=[TextContent(type="text", text=text_content)])

        return JSONRPCResponse(
            jsonrpc="2.0", id=request.id, result=result.model_dump(exclude_none=True)
        )

    except Exception as e:
        logger.error(f"Error in tools/call: {e}")
        result = CallToolResult(
            content=[TextContent(type="text", text=f"Tool execution failed: {str(e)}")]
        )
        return JSONRPCResponse(
            jsonrpc="2.0", id=request.id, result=result.model_dump(exclude_none=True)
        )
