# [MODULE DOC]
# Integration tests for the DIPG safety environment server launched under
# gunicorn (4 workers, port 8009) with the mock dataset. Skipped if gunicorn
# is missing; prepends repo root to sys.path.
# tests/envs/test_dipg_environment.py
import os
import shutil
import subprocess
import sys
import time

import pytest
import requests

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.dipg_safety_env.client import DIPGSafetyEnv
from envs.dipg_safety_env.models import DIPGAction

# Skip all tests if gunicorn is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("gunicorn") is None, reason="gunicorn not installed"
)


@pytest.fixture(scope="module")
def server():
    # [FUNCTION DOC]
    # Module fixture: launches gunicorn with PYTHONPATH and DIPG_DATASET_PATH set,
    # polls /health up to 12 times for a "healthy" 200, raises RuntimeError (after
    # printing stderr and killing it) if it never becomes healthy, yields the
    # base URL, then kills the process. Side effects: subprocess, network, prints.
    # KNOWN ISSUE: the poll only sleeps on connection errors, so non-healthy
    # responses are retried 12 times with no delay.
    """Starts the environment server as a background process."""
    # --- Define Absolute Paths & Port ---
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    SRC_PATH = os.path.join(ROOT_DIR, "src")
    DATASET_SOURCE_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "mock_dataset.jsonl")
    )
    PORT = 8009

    # --- Launch the Server using Gunicorn ---
    localhost = f"http://localhost:{PORT}"
    print(f"--- Starting DIPGSafetyEnv server with Gunicorn on port {PORT} ---")

    server_env = {
        **os.environ,
        "PYTHONPATH": SRC_PATH,
        "DIPG_DATASET_PATH": DATASET_SOURCE_PATH,
    }

    gunicorn_command = [
        "gunicorn",
        "-w",
        "4",
        "-k",
        "uvicorn.workers.UvicornWorker",
        "-b",
        f"0.0.0.0:{PORT}",
        "envs.dipg_safety_env.server.app:app",
    ]
    openenv_process = subprocess.Popen(
        gunicorn_command,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # --- Wait and Verify ---
    print("\n--- Waiting for server to become healthy... ---")
    is_healthy = False
    for i in range(12):
        try:
            response = requests.get(f"{localhost}/health", timeout=5)
            if response.status_code == 200 and "healthy" in response.text:
                is_healthy = True
                print("✅ Server is running and healthy!")
                break
        except requests.exceptions.RequestException:
            print(f"Attempt {i + 1}/12: Server not ready, waiting 10 seconds...")
            time.sleep(10)

    if not is_healthy:
        print("❌ Server did not become healthy in time. Aborting.")
        print("\n--- Server Logs ---")
        print(openenv_process.stderr.read())
        try:
            openenv_process.kill()
        except ProcessLookupError:
            pass
        raise RuntimeError("Server failed to start.")

    yield localhost

    # --- Clean up ---
    print("\n--- Cleaning up ---")
    try:
        openenv_process.kill()
        print("✅ Server process killed.")
    except ProcessLookupError:
        print("✅ Server process was already killed.")


def test_reset(server):
    # [FUNCTION DOC]
    # Two resets return different questions.
    """Test that reset() returns a valid observation."""
    env = DIPGSafetyEnv(base_url=server, timeout=300)
    obs1 = env.reset()
    obs2 = env.reset()
    assert obs1.observation.question != obs2.observation.question


def test_step(server):
    # [FUNCTION DOC]
    # A well-formed analysis/final response yields a float reward and done.
    """Test that step() returns a valid result."""
    env = DIPGSafetyEnv(base_url=server, timeout=300)
    env.reset()
    action = DIPGAction(
        llm_response="<|channel|>analysis<|message|>This is an analysis.<|end|>\n<|channel|>final<|message|>This is the final answer.<|end|>"
    )
    result = env.step(action)
    assert isinstance(result.reward, float)
    assert result.done is True


def test_malformed_step(server):
    # [FUNCTION DOC]
    # A malformed response still yields a float reward and done.
    """Test that a malformed step() does not crash the server."""
    env = DIPGSafetyEnv(base_url=server, timeout=300)
    env.reset()
    action = DIPGAction(llm_response="This is a malformed response")
    result = env.step(action)
    assert isinstance(result.reward, float)
    assert result.done is True
