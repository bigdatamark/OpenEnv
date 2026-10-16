# [MODULE DOC]
# Pytest fixtures for the calendar server tests: FastAPI apps/TestClients for the MCP,
# calendars and calendarList routers and the full main app, a per-test database id,
# a seeded multi-user database, and builders for MCP JSON-RPC requests and API headers.
# Side effects at import: inserts the server directory at the front of sys.path.
# KNOWN ISSUE: sys.path is modified after the server-local imports (apis, data) already
# ran, so the path insert cannot help those imports.
"""
Pytest configuration and shared fixtures for Calendar API tests
"""

import sys
import uuid
from pathlib import Path

import pytest
from apis.calendarList.router import router as calendar_list_router
from apis.calendars.router import router as calendars_router
from apis.mcp.router import router as mcp_router
from data.multi_user_sample import get_multi_user_sql
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


@pytest.fixture
def app():
    # [FUNCTION DOC]
    # Fixture: FastAPI app containing only the MCP router.
    """Create a FastAPI app with the MCP router for testing"""
    app = FastAPI()
    app.include_router(mcp_router)
    return app


@pytest.fixture
def client(app):
    # [FUNCTION DOC]
    # Fixture: TestClient for the MCP-only app.
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def calendar_app():
    # [FUNCTION DOC]
    # Fixture: FastAPI app containing only the calendars router.
    """Create a FastAPI app with the Calendar router for testing"""
    app = FastAPI()
    app.include_router(calendars_router)
    return app


@pytest.fixture
def calendar_client(calendar_app):
    # [FUNCTION DOC]
    # Fixture: TestClient for the calendars app.
    """Create a test client for calendar endpoints"""
    return TestClient(calendar_app)


@pytest.fixture
def calendar_list_app():
    # [FUNCTION DOC]
    # Fixture: FastAPI app containing only the calendarList router.
    """Create a FastAPI app with the CalendarList router for testing"""
    app = FastAPI()
    app.include_router(calendar_list_router)
    return app


@pytest.fixture
def calendar_list_client(calendar_list_app):
    # [FUNCTION DOC]
    # Fixture: TestClient for the calendarList app.
    """Create a test client for calendarList endpoints"""
    return TestClient(calendar_list_app)


@pytest.fixture
def full_app():
    # [FUNCTION DOC]
    # Fixture: imports and returns the complete application from the server's main module.
    """Create a complete FastAPI app with all routers (like main.py)"""
    from main import app

    return app


@pytest.fixture
def full_client(full_app):
    # [FUNCTION DOC]
    # Fixture: TestClient for the complete application.
    """Create a test client for the complete application"""
    return TestClient(full_app)


@pytest.fixture(scope="function")
def test_database_id():
    # [FUNCTION DOC]
    # Fixture (function scope): returns a unique id "test_" plus 8 random hex chars.
    """Generate unique test database ID for each test"""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="function")
def seeded_database(full_client, test_database_id):
    # [FUNCTION DOC]
    # Fixture: POSTs the multi-user sample SQL to /api/seed-database for the test database id
    # and asserts HTTP 200. Returns a dict with the client, the database_id and a mapping of
    # short names (alice, bob, carol, dave) to seeded user ids.
    """Setup a seeded test database"""
    # Get sample SQL data
    sql_content = get_multi_user_sql()

    # Seed the database
    seed_response = full_client.post(
        "/api/seed-database",
        json={"database_id": test_database_id, "sql_content": sql_content},
    )

    assert seed_response.status_code == 200, (
        f"Failed to seed database: {seed_response.text}"
    )

    return {
        "client": full_client,
        "database_id": test_database_id,
        "users": {
            "alice": "alice_manager",
            "bob": "bob_developer",
            "carol": "carol_designer",
            "dave": "dave_sales",
        },
    }


@pytest.fixture
def mcp_request_helper():
    # [FUNCTION DOC]
    # Fixture returning a builder for MCP "tools/call" JSON-RPC requests.
    """Helper function to create MCP requests"""

    def create_mcp_request(method_name, arguments, request_id=1):
        # [FUNCTION DOC]
        # HELPER: Returns a JSON-RPC 2.0 dict with the given id (default 1), method "tools/call"
        # and params containing the tool name and arguments.
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": method_name, "arguments": arguments},
        }

    return create_mcp_request


@pytest.fixture
def api_headers():
    # [FUNCTION DOC]
    # Fixture returning a builder for request headers.
    """Helper function to create API headers"""

    def create_headers(database_id, user_id):
        # [FUNCTION DOC]
        # HELPER: Returns a dict with x-database-id and x-user-id headers.
        return {"x-database-id": database_id, "x-user-id": user_id}

    return create_headers
