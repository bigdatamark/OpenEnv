# [MODULE DOC]
# Integration tests for the maze_env server: a module-scoped fixture launches it with gunicorn on a
# free port and the tests drive it through the async MazeEnv client (reset, step, state, step count,
# metadata). All tests are skipped when gunicorn is not installed. Side effect at import: prepends the
# repo root to sys.path.
# KNOWN ISSUE: the module docstring says "OpenSpiel environment server" but the tests target the Maze env.
"""Unit tests for OpenSpiel environment server."""

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest
import requests

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.maze_env.client import MazeEnv
from envs.maze_env.models import MazeAction

# Skip all tests if gunicorn is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("gunicorn") is None, reason="gunicorn not installed"
)


@pytest.fixture(scope="module")
def server():
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture. Picks a free port, starts gunicorn with one uvicorn worker serving
    # envs.maze_env.server.app:app (PYTHONPATH = repo root and src), and polls /health up to 12 times
    # (10s sleeps after connection errors). If never healthy, prints logs, kills the process and skips.
    # Yields the base URL and kills the process afterwards. Side effects: subprocess, network, prints.
    # KNOWN ISSUE: sets BROWSERGYM_* environment variables that are irrelevant to the maze server
    # (apparently copied from another env's test); a non-200 health response retries without sleeping.
    """Starts the Maze environment server as a background process."""
    # Define paths for subprocess environment
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    SRC_PATH = os.path.join(ROOT_DIR, "src")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        PORT = sock.getsockname()[1]
    localhost = f"http://localhost:{PORT}"

    print(f"\n--- Starting Maze server on port {PORT} ---")

    server_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([ROOT_DIR, SRC_PATH]),
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
        "BROWSERGYM_HEADLESS": "true",
    }

    gunicorn_command = [
        "gunicorn",
        "-w",
        "1",  # Single worker for testing
        "-k",
        "uvicorn.workers.UvicornWorker",
        "-b",
        f"0.0.0.0:{PORT}",
        "envs.maze_env.server.app:app",
    ]

    server_process = subprocess.Popen(
        gunicorn_command,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to become healthy
    print("\n--- Waiting for server to become healthy... ---")
    is_healthy = False
    for i in range(12):
        try:
            response = requests.get(f"{localhost}/health", timeout=5)
            if response.status_code == 200:
                is_healthy = True
                print("✅ Server is running and healthy!")
                break
        except requests.exceptions.RequestException:
            print(f"Attempt {i + 1}/12: Server not ready, waiting 10 seconds...")
            time.sleep(10)

    if not is_healthy:
        print("❌ Server did not become healthy in time. Aborting.")
        print("\n--- Server Logs ---")
        stdout, stderr = server_process.communicate(timeout=5)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        try:
            server_process.kill()
        except ProcessLookupError:
            # The process is already dead; nothing to clean up.
            pass
        pytest.skip("Server failed to start")

    yield localhost

    # Cleanup
    print("\n--- Cleaning up server ---")
    try:
        server_process.kill()
        print("✅ Server process killed")
    except ProcessLookupError:
        print("✅ Server process was already killed")


def test_health_endpoint(server):
    # [FUNCTION DOC]
    # GET /health returns 200 with a "status" key.
    """Test that the health endpoint works."""
    response = requests.get(f"{server}/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_reset(server):
    # [FUNCTION DOC]
    # reset returns a non-done observation with legal_actions and current/previous positions.
    """Test that reset() returns a valid observation."""

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Async body of the test using a MazeEnv context manager.
        async with MazeEnv(base_url=server) as env:
            result = await env.reset()

            assert result.observation is not None
            assert hasattr(result.observation, "legal_actions")
            assert hasattr(result.observation, "current_position")
            assert hasattr(result.observation, "previous_position")
            assert result.observation.done is False
            await env.close()

    asyncio.run(_run())


def test_reset_multiple_times(server):
    # [FUNCTION DOC]
    # Repeated resets return valid observations and produce different episode ids.
    """Test that reset() can be called multiple times."""

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Async body of the test.
        async with MazeEnv(base_url=server) as env:
            result1 = await env.reset()
            result2 = await env.reset()

            # Both should be valid observations
            assert result1.observation is not None
            assert result2.observation is not None

            # Episode IDs should be different (new episodes)
            state1 = await env.state()
            await env.reset()
            state2 = await env.state()
            assert state1.episode_id != state2.episode_id
            await env.close()

    asyncio.run(_run())


def test_step(server):
    # [FUNCTION DOC]
    # Stepping with the first legal action returns an observation, a numeric or None reward and bool done.
    """Test that step() returns a valid result."""

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Async body of the test.
        async with MazeEnv(base_url=server) as env:
            result = await env.reset()

            # Take a simple action
            action = MazeAction(action=result.observation.legal_actions[0])
            result = await env.step(action)

            assert result.observation is not None
            assert isinstance(result.reward, (int, float)) or result.reward is None
            assert isinstance(result.done, bool)
            await env.close()

    asyncio.run(_run())


def test_step_multiple_times(server):
    # [FUNCTION DOC]
    # Two consecutive steps (actions 0 and 1) both return observations.
    """Test that step() can be called multiple times."""

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Async body of the test.
        async with MazeEnv(base_url=server) as env:
            await env.reset()

            # Take multiple actions
            action1 = MazeAction(action=0)
            result1 = await env.step(action1)

            action2 = MazeAction(action=1)
            result2 = await env.step(action2)

            # Both should be valid
            assert result1.observation is not None
            assert result2.observation is not None
            await env.close()

    asyncio.run(_run())


def test_state_endpoint(server):
    # [FUNCTION DOC]
    # state after reset has current_position, exit_cell and status.
    """Test that the state endpoint returns valid state."""

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Async body of the test.
        async with MazeEnv(base_url=server) as env:
            await env.reset()

            state = await env.state()

            assert state is not None
            assert hasattr(state, "current_position")
            assert hasattr(state, "exit_cell")
            assert hasattr(state, "status")
            await env.close()

    asyncio.run(_run())


def test_step_count_increments(server):
    # [FUNCTION DOC]
    # step_count goes 0, 1, 2 across reset and two steps.
    """Test that step count increments correctly."""

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Async body of the test.
        async with MazeEnv(base_url=server) as env:
            await env.reset()

            state1 = await env.state()
            assert state1.step_count == 0

            action = MazeAction(action=0)
            await env.step(action)

            state2 = await env.state()
            assert state2.step_count == 1

            await env.step(action)

            state3 = await env.state()
            assert state3.step_count == 2
            await env.close()

    asyncio.run(_run())


def test_action_with_metadata(server):
    # [FUNCTION DOC]
    # An action carrying metadata is accepted.
    """Test that actions with metadata work."""

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Async body of the test.
        async with MazeEnv(base_url=server) as env:
            await env.reset()

            action = MazeAction(action=0, metadata={"test": "value", "number": 42})
            result = await env.step(action)

            assert result.observation is not None
            await env.close()

    asyncio.run(_run())
