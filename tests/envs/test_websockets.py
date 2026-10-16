# [MODULE DOC]
# Integration tests for the WebSocket EnvClient architecture and the create_app factory
# pattern: smoke tests (in-process), HTTP/WebSocket protocol tests and concurrency tests
# against uvicorn subprocesses on fixed localhost ports (8100-8201). Inserts the repo
# root into sys.path at import time.
# SPDX-License-Identifier: BSD-3-Clause

"""
Integration tests for OpenEnv environments.

This module tests the new WebSocket-based client architecture and factory pattern
to ensure all environments work correctly after the migration from HTTPEnvClient.

Test Categories:
- Smoke: Factory pattern validation and basic server startup
- Protocol: WebSocket and HTTP endpoint verification
- Concurrency: Multiple simultaneous session handling

Run with: pytest tests/envs/test_websockets.py -v
Run specific category: pytest tests/envs/test_websockets.py -v -k "smoke"
"""

import asyncio
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Generator

import pytest
import requests
from websockets.protocol import State

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================


@contextmanager
def run_server(
    module_path: str,
    port: int = 8000,
    startup_timeout: float = 10.0,
    env_vars: dict = None,
) -> Generator[subprocess.Popen, None, None]:
    # [FUNCTION DOC]
    # HELPER: Context manager that launches uvicorn serving "<module_path>:app" on
    # 127.0.0.1:<port> with extra env vars, polls /health every 0.5 s until 200 or
    # startup_timeout, then yields the Popen; always terminates (kills after 5 s) and closes pipes.
    # Raises: TimeoutError with captured stderr if the server never becomes healthy.
    # Side effects: subprocess, HTTP requests.
    # KNOWN ISSUE: only ConnectionError is caught while polling; a requests Timeout would
    # propagate. Reading stderr after timeout can block since the process is still running.
    """
    Context manager to start and stop a server process.

    Args:
        module_path: Python module path (e.g., "envs.echo_env.server.app")
        port: Port to run the server on
        startup_timeout: Max seconds to wait for server startup
        env_vars: Additional environment variables

    Yields:
        The subprocess.Popen instance
    """
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    # Start the server
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            f"{module_path}:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < startup_timeout:
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
        else:
            # Print stderr for debugging
            stderr = process.stderr.read().decode() if process.stderr else ""
            raise TimeoutError(
                f"Server failed to start within {startup_timeout}s. Stderr: {stderr}"
            )

        yield process

    finally:
        # Clean shutdown
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        # Close pipes
        for stream in [process.stdin, process.stdout, process.stderr]:
            if stream and not stream.closed:
                stream.close()


def wait_for_server(base_url: str, timeout: float = 10.0) -> bool:
    # [FUNCTION DOC]
    # HELPER: Polls <base_url>/health until 200 or timeout. Returns True if healthy, else False.
    # KNOWN ISSUE: unused in this module.
    """Wait for a server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    return False


# =============================================================================
# Smoke Tests - Factory Pattern and Basic Functionality
# =============================================================================


class TestSmokeFactoryPattern:
    # [CLASS DOC]
    # In-process checks that envs construct/reset and that create_app takes a factory.
    """Test that the factory pattern works correctly for all environments."""

    def test_smoke_echo_env_factory_pattern(self):
        # [FUNCTION DOC]
        # EchoEnvironment constructs, resets, and closes.
        """Test that EchoEnvironment can be created via factory."""
        from envs.echo_env.server.echo_environment import EchoEnvironment

        # Should be callable
        env = EchoEnvironment()
        assert env is not None

        # Test basic operations
        obs = env.reset()
        assert obs is not None

        env.close()

    def test_smoke_connect4_env_factory_pattern(self):
        # [FUNCTION DOC]
        # Connect4Environment constructs, resets, and closes.
        """Test that Connect4Environment can be created via factory."""
        from envs.connect4_env.server.connect4_environment import Connect4Environment

        env = Connect4Environment()
        assert env is not None

        obs = env.reset()
        assert obs is not None

        env.close()

    def test_smoke_create_app_accepts_class(self):
        # [FUNCTION DOC]
        # create_app accepts an environment class.
        """Test that create_app accepts a class (not instance)."""
        from envs.echo_env.server.echo_environment import EchoEnvironment
        from openenv.core.env_server.http_server import create_app
        from openenv.core.env_server.mcp_types import (
            CallToolAction,
            CallToolObservation,
        )

        # Should not raise TypeError
        app = create_app(
            EchoEnvironment, CallToolAction, CallToolObservation, env_name="test"
        )
        assert app is not None

    def test_smoke_create_app_accepts_factory_function(self):
        # [FUNCTION DOC]
        # create_app accepts a zero-arg factory function.
        """Test that create_app accepts a factory function."""
        from envs.echo_env.server.echo_environment import EchoEnvironment
        from openenv.core.env_server.http_server import create_app
        from openenv.core.env_server.mcp_types import (
            CallToolAction,
            CallToolObservation,
        )

        def create_echo_env():
            # [FUNCTION DOC]
            # HELPER: Factory returning a new EchoEnvironment.
            return EchoEnvironment()

        # Should not raise TypeError
        app = create_app(
            create_echo_env, CallToolAction, CallToolObservation, env_name="test"
        )
        assert app is not None

    def test_smoke_create_app_rejects_instance(self):
        # [FUNCTION DOC]
        # create_app raises TypeError "must be a callable" for an instance.
        """Test that create_app rejects an instance (not callable)."""
        from envs.echo_env.server.echo_environment import EchoEnvironment
        from openenv.core.env_server.http_server import create_app
        from openenv.core.env_server.mcp_types import (
            CallToolAction,
            CallToolObservation,
        )

        # Create an instance (wrong pattern)
        instance = EchoEnvironment()

        # Should raise TypeError
        with pytest.raises(TypeError, match="must be a callable"):
            create_app(instance, CallToolAction, CallToolObservation, env_name="test")

        instance.close()


# =============================================================================
# Protocol Tests - WebSocket and HTTP Endpoints
# =============================================================================


@pytest.mark.integration
class TestProtocolHttpEndpoints:
    # [CLASS DOC]
    # HTTP endpoint tests against an echo server on port 8100.
    """Test that HTTP endpoints work correctly."""

    @pytest.fixture
    def echo_server(self):
        # [FUNCTION DOC]
        # Fixture: runs the echo server on port 8100 and yields its URL.
        """Start echo environment server."""
        with run_server("envs.echo_env.server.app", port=8100) as proc:
            yield "http://127.0.0.1:8100"

    def test_protocol_health_endpoint(self, echo_server):
        # [FUNCTION DOC]
        # /health returns 200 with status "healthy".
        """Test /health endpoint."""
        response = requests.get(f"{echo_server}/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_protocol_schema_endpoint(self, echo_server):
        # [FUNCTION DOC]
        # /schema returns action and observation schemas.
        """Test /schema endpoint."""
        response = requests.get(f"{echo_server}/schema")
        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "observation" in data

    def test_protocol_reset_endpoint(self, echo_server):
        # [FUNCTION DOC]
        # POST /reset returns an observation.
        """Test /reset endpoint."""
        response = requests.post(f"{echo_server}/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data

    def test_protocol_step_endpoint(self, echo_server):
        # [FUNCTION DOC]
        # POST /step with a call_tool action returns an observation.
        """Test /step endpoint with MCP action."""
        # First reset
        requests.post(f"{echo_server}/reset", json={})

        # Then step with MCP CallToolAction format
        response = requests.post(
            f"{echo_server}/step",
            json={
                "action": {
                    "type": "call_tool",
                    "tool_name": "echo_message",
                    "arguments": {"message": "Hello"},
                }
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data

    def test_protocol_state_endpoint(self, echo_server):
        # [FUNCTION DOC]
        # GET /state after reset includes step_count.
        """Test /state endpoint."""
        # First reset
        requests.post(f"{echo_server}/reset", json={})

        response = requests.get(f"{echo_server}/state")
        assert response.status_code == 200
        data = response.json()
        assert "step_count" in data


@pytest.mark.integration
class TestProtocolWebSocketClient:
    # [CLASS DOC]
    # WebSocket client tests against an echo server on port 8101.
    """Test that WebSocket client (EnvClient) works correctly."""

    @pytest.fixture
    def echo_server(self):
        # [FUNCTION DOC]
        # Fixture: runs the echo server on port 8101 and yields its URL.
        """Start echo environment server."""
        with run_server("envs.echo_env.server.app", port=8101) as proc:
            yield "http://127.0.0.1:8101"

    def test_protocol_client_connect_and_reset(self, echo_server):
        # [FUNCTION DOC]
        # Sync client connects and reset returns an observation.
        """Test client can connect and reset via WebSocket."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=echo_server).sync() as client:
            result = client.reset()
            assert result is not None
            assert result.observation is not None

    def test_protocol_client_step(self, echo_server):
        # [FUNCTION DOC]
        # call_tool echo_message returns the message.
        """Test client can step via WebSocket."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=echo_server).sync() as client:
            client.reset()
            result = client.call_tool("echo_message", message="Hello")
            assert result is not None
            assert result == "Hello"

    def test_protocol_client_state(self, echo_server):
        # [FUNCTION DOC]
        # state().step_count is 1 after one tool call.
        """Test client can get state via WebSocket."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=echo_server).sync() as client:
            client.reset()
            client.call_tool("echo_message", message="Test")

            state = client.state()
            assert state is not None
            assert state.step_count == 1

    def test_protocol_client_reconnects_after_close(self, echo_server):
        # [FUNCTION DOC]
        # Closes the inner websocket on the sync wrapper's loop behind the client's back,
        # then checks reset and call_tool still work (client reconnects).
        """A socket closed by the far end must not disable the client."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=echo_server).sync() as client:
            client.reset()
            assert client.call_tool("echo_message", message="before") == "before"

            # Close the socket the way a keepalive timeout or a dropped tunnel
            # would: from underneath the client, without telling it. The sync
            # wrapper drives its own loop on a background thread, so the close
            # has to be scheduled onto that loop.
            inner = client._async
            asyncio.run_coroutine_threadsafe(inner._ws.close(), client._loop).result(
                timeout=10
            )
            assert inner._ws.state in (State.CLOSING, State.CLOSED)

            client.reset()
            assert client.call_tool("echo_message", message="after") == "after"

    def test_protocol_client_multiple_episodes(self, echo_server):
        # [FUNCTION DOC]
        # step_count is 2 after two calls, 0 after reset, 1 after another call.
        """Test client can run multiple episodes."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=echo_server).sync() as client:
            # Episode 1
            client.reset()
            client.call_tool("echo_message", message="E1S1")
            client.call_tool("echo_message", message="E1S2")

            state1 = client.state()
            assert state1.step_count == 2

            # Episode 2 - reset should clear state
            client.reset()
            state2 = client.state()
            assert state2.step_count == 0

            client.call_tool("echo_message", message="E2S1")
            state3 = client.state()
            assert state3.step_count == 1


# =============================================================================
# Concurrency Tests - Multiple Sessions
# =============================================================================


@pytest.mark.integration
class TestConcurrencyMultipleSessions:
    # [CLASS DOC]
    # Concurrent session tests (skipped by default) on port 8102 with MAX_CONCURRENT_ENVS=10.
    """Test that multiple concurrent sessions work correctly.

    NOTE: These tests require the server to be configured with max_concurrent_envs > 1.
    By default, environments only allow 1 concurrent session, so these tests are
    marked to skip unless concurrency is explicitly configured.
    """

    @pytest.fixture
    def echo_server_concurrent(self):
        # [FUNCTION DOC]
        # Fixture: runs the echo server on port 8102 with MAX_CONCURRENT_ENVS=10.
        """Start echo environment server with concurrent sessions enabled."""
        # Pass MAX_CONCURRENT_ENVS env var to enable multiple sessions
        with run_server(
            "envs.echo_env.server.app",
            port=8102,
            env_vars={"MAX_CONCURRENT_ENVS": "10"},
        ) as proc:
            yield "http://127.0.0.1:8102"

    @pytest.mark.skip(
        reason="Concurrency requires server configuration - run manually with MAX_CONCURRENT_ENVS > 1"
    )
    def test_concurrency_two_independent_sessions(self, echo_server_concurrent):
        # [FUNCTION DOC]
        # Two clients keep independent step counts (3 and 1). Skipped.
        """Test that two clients can run independently."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=echo_server_concurrent).sync() as client1:
            with EchoEnv(base_url=echo_server_concurrent).sync() as client2:
                # Both reset
                client1.reset()
                client2.reset()

                # Client 1 takes 3 steps
                for i in range(3):
                    client1.call_tool("echo_message", message=f"C1-{i}")

                # Client 2 takes 1 step
                client2.call_tool("echo_message", message="C2-0")

                # Check states are independent
                state1 = client1.state()
                state2 = client2.state()

                assert state1.step_count == 3
                assert state2.step_count == 1

    @pytest.mark.skip(
        reason="Concurrency requires server configuration - run manually with MAX_CONCURRENT_ENVS > 1"
    )
    def test_concurrency_session_isolation(self, echo_server_concurrent):
        # [FUNCTION DOC]
        # Echo results are not mixed between two sessions. Skipped.
        """Test that session state is isolated between clients."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=echo_server_concurrent).sync() as client1:
            client1.reset()
            result1 = client1.call_tool("echo_message", message="Secret from C1")

            with EchoEnv(base_url=echo_server_concurrent).sync() as client2:
                client2.reset()
                result2 = client2.call_tool("echo_message", message="Secret from C2")

                # Messages should not leak between sessions
                assert result1 == "Secret from C1"
                assert result2 == "Secret from C2"


# =============================================================================
# Environment-Specific Tests
# =============================================================================


@pytest.mark.integration
class TestEchoEnvironment:
    # [CLASS DOC]
    # Echo-specific tool tests on port 8200.
    """Test EchoEnvironment specifically."""

    @pytest.fixture
    def server(self):
        # [FUNCTION DOC]
        # Fixture: runs the echo server on port 8200 and yields its URL.
        with run_server("envs.echo_env.server.app", port=8200) as proc:
            yield "http://127.0.0.1:8200"

    def test_echo_message_echoed(self, server):
        # [FUNCTION DOC]
        # echo_message returns the input.
        """Test that messages are echoed correctly."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=server).sync() as client:
            client.reset()
            result = client.call_tool("echo_message", message="Hello World!")
            assert result == "Hello World!"

    def test_echo_with_length(self, server):
        # [FUNCTION DOC]
        # echo_with_length returns the message and its length.
        """Test that echo_with_length returns message and length."""
        from envs.echo_env.client import EchoEnv

        with EchoEnv(base_url=server).sync() as client:
            client.reset()
            result = client.call_tool("echo_with_length", message="Hello World!")
            assert result["message"] == "Hello World!"
            assert result["length"] == len("Hello World!")


@pytest.mark.integration
class TestConnect4Environment:
    # [CLASS DOC]
    # Connect4-specific tests on port 8201.
    """Test Connect4Environment specifically."""

    @pytest.fixture
    def server(self):
        # [FUNCTION DOC]
        # Fixture: runs the Connect4 server on port 8201 and yields its URL.
        with run_server("envs.connect4_env.server.app", port=8201) as proc:
            yield "http://127.0.0.1:8201"

    def test_connect4_initial_board(self, server):
        # [FUNCTION DOC]
        # Reset board is an all-zero 6x7 grid.
        """Test that initial board is empty."""
        from envs.connect4_env.client import Connect4Env

        with Connect4Env(base_url=server).sync() as client:
            result = client.reset()

            # Board should be 6x7 and empty (all zeros)
            assert len(result.observation.board) == 6
            assert all(len(row) == 7 for row in result.observation.board)
            assert all(cell == 0 for row in result.observation.board for cell in row)

    def test_connect4_legal_actions(self, server):
        # [FUNCTION DOC]
        # All 7 columns are legal after reset.
        """Test that all columns are legal initially."""
        from envs.connect4_env.client import Connect4Env

        with Connect4Env(base_url=server).sync() as client:
            result = client.reset()

            # All 7 columns should be legal
            assert len(result.observation.legal_actions) == 7


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
