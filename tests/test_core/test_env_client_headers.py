# [MODULE DOC]
# Tests that GenericEnvClient and MCPToolClient send their configured headers on every
# connection (needed for private Spaces). Serves the echo env behind a token guard;
# skipped if echo_env is not importable.
# SPDX-License-Identifier: BSD-3-Clause

"""Clients send `headers` on every connection, so they can reach a private Space."""

import pytest
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient

echo_app = pytest.importorskip("echo_env.server.app").app

HEADERS = {"Authorization": "Bearer secret"}
AUTH_HEADER = (b"authorization", b"Bearer secret")


def require_token(app):
    # [FUNCTION DOC]
    # HELPER: Wraps an ASGI app so requests lacking the bearer Authorization header are
    # rejected (WebSocket closed with 1008, HTTP 401); lifespan events pass through.
    # Returns: the wrapped ASGI callable.
    """Reject HTTP requests and WebSocket upgrades without the token, like a private Space."""

    async def guarded(scope, receive, send):
        # [FUNCTION DOC]
        # HELPER: The guarding ASGI callable.
        if scope["type"] == "lifespan" or AUTH_HEADER in scope["headers"]:
            await app(scope, receive, send)
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})

    return guarded


@pytest.fixture
def server_url(serve):
    # [FUNCTION DOC]
    # Fixture serving the token-guarded echo app. Returns: its URL.
    return serve(require_token(echo_app))


async def test_websocket_without_headers_is_rejected(server_url):
    # [FUNCTION DOC]
    # Without headers, reset raises ConnectionError.
    with pytest.raises(ConnectionError):
        async with GenericEnvClient(base_url=server_url) as env:
            await env.reset()


async def test_generic_client_sends_headers(server_url):
    # [FUNCTION DOC]
    # With headers, reset works, and a new_session child also connects and resets.
    async with GenericEnvClient(base_url=server_url, headers=HEADERS) as env:
        await env.reset()
        child = await env.new_session()
        await child.reset()


async def test_mcp_tool_client_sends_headers(server_url):
    # [FUNCTION DOC]
    # MCPToolClient with headers can reset and list tools.
    async with MCPToolClient(base_url=server_url, headers=HEADERS) as env:
        await env.reset()
        assert await env.list_tools()
