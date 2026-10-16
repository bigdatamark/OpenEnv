# [MODULE DOC]
# Tests for the DIPG safety environment client's connection error handling.
# Prepends repo root to sys.path.
import os
import sys

import pytest

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.dipg_safety_env.client import DIPGSafetyEnv


@pytest.mark.asyncio
async def test_invalid_url():
    # [FUNCTION DOC]
    # Async: reset against an unresolvable host raises ConnectionError.
    """Test that the client raises an error for an invalid URL."""
    with pytest.raises(ConnectionError):
        env = DIPGSafetyEnv(base_url="http://invalid-url:9999")
        await env.reset()


@pytest.mark.asyncio
async def test_server_not_running():
    # [FUNCTION DOC]
    # Async: reset against an unused localhost port raises ConnectionError.
    """Test that the client raises an error when the server is not running."""
    with pytest.raises(ConnectionError):
        env = DIPGSafetyEnv(base_url="http://localhost:9999")
        await env.reset()


def test_invalid_action():
    # [FUNCTION DOC]
    # Placeholder that does nothing (requires a running server).
    """Test that the client raises an error for an invalid action."""
    # This test requires a running server, so we'll skip it for now.
    pass


def test_server_timeout():
    # [FUNCTION DOC]
    # Placeholder that does nothing (requires a hanging server).
    """Test that the client raises an error for a server timeout."""
    # This test requires a running server that can be made to hang, so we'll skip it for now.
    pass
