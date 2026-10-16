# [MODULE DOC]
# End-to-end tests of the agentic harness stack with nothing faked on the critical path: a real
# subprocess (scripted_harness.py in a chosen mode) reads an injected MCP config, calls domain
# tools over the live MCP bridge, and streams JSON-line events. Covers simulation mode (reset/step
# via the sync facade with a rubric) and production mode (a real uvicorn server with a /harness
# WebSocket).

# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for the RFC 005 agentic harness stack.

Unlike the unit tests, nothing on the critical path is faked here:

- A real CLI "harness" subprocess is spawned via ``HarnessProcess``.
- ``inject_tools`` writes a real MCP config file, which the subprocess reads.
- The subprocess calls the environment's domain tool over the live
  ``HarnessMCPBridge`` using a real MCP client.
- Turn events stream back over stdio and are mapped to ``HarnessEvent``.

The stack is validated in both modes:

- Simulation: ``HarnessEnvironment.reset()``/``step()`` multi-turn with a
  rubric, through the sync facade a training loop would use.
- Production: a real uvicorn server registered with ``mode="production"``,
  driven by a real WebSocket client on ``/harness``.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import pytest
import uvicorn
from fastmcp import FastMCP
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.types import Observation
from openenv.core.harness import (
    AgenticHarnessAdapter,
    HarnessAction,
    HarnessConfig,
    HarnessEnvironment,
    HarnessEvent,
    HarnessEventType,
    HarnessProcess,
)
from openenv.core.rubrics import Rubric
from websockets.sync.client import connect as ws_connect

# The harness role is played by a real, lintable module spawned as a
# subprocess in "mcp" mode: it reads the injected MCP config, connects to the
# bridge with a real MCP client each turn, and emits JSON-line events.
SCRIPTED_HARNESS = Path(__file__).parent / "scripted_harness.py"


class ScriptedCLIAdapter(AgenticHarnessAdapter):
    # [CLASS DOC]
    # HELPER: Adapter driving the scripted harness subprocess: injects tools by writing an mcp.json
    # config file (bridge URL and tool names) and talks JSON lines over stdio via HarnessProcess.
    # Declares read_file as a built-in tool name.
    """Adapter for the scripted harness: config-file injection + stdio JSON."""

    BUILTIN_TOOL_NAMES = frozenset({"read_file"})

    def __init__(self, config: HarnessConfig, config_dir: Path):
        # [FUNCTION DOC]
        # HELPER: Stores the config, the mcp.json path inside config_dir, and no process yet.
        super().__init__(config)
        self._config_path = config_dir / "mcp.json"
        self._process: Optional[HarnessProcess] = None

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Writes the bridge URL and the tool names as JSON to the config file.
        self._config_path.write_text(
            json.dumps({"mcp_url": bridge_url, "tools": [t.name for t in tools]})
        )

    async def start(self, working_directory: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Creates a HarnessProcess for the configured command with HARNESS_MCP_CONFIG pointing at the
        # config file and starts it, waiting for a JSON line whose event is "ready".
        self._process = HarnessProcess(
            self.config.command,
            cwd=working_directory,
            env_vars={"HARNESS_MCP_CONFIG": str(self._config_path)},
            startup_timeout_s=self.config.startup_timeout_s,
        )

        def is_ready(line: str) -> bool:
            # [FUNCTION DOC]
            # HELPER: Returns True if the line parses as JSON with event "ready"; False on invalid JSON.
            try:
                return json.loads(line).get("event") == "ready"
            except json.JSONDecodeError:
                return False

        await self._process.start(ready_check=is_ready)

    async def stop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Stops the subprocess if one was started.
        if self._process is not None:
            await self._process.stop()

    async def is_alive(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns whether a subprocess exists and is running.
        return self._process is not None and self._process.is_running()

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        # [FUNCTION DOC]
        # HELPER: Writes the message as a JSON line, then reads lines (30s timeout each) converting each
        # native event to a HarnessEvent and yielding it; stops after turn_complete. Raises RuntimeError if
        # the process stream ends mid-turn.
        assert self._process is not None
        await self._process.write_line(json.dumps({"message": message}))
        while True:
            line = await self._process.read_line(timeout_s=30.0)
            if line is None:
                raise RuntimeError("harness died mid-turn")
            native = json.loads(line)
            event_type = native.pop("event")
            if event_type == "turn_complete":
                yield HarnessEvent(type=HarnessEventType.TURN_COMPLETE, data=native)
                return
            yield HarnessEvent(type=HarnessEventType(event_type), data=native)


class SumRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric returning 1.0 if the observation's response metadata contains "sum=5", else 0.0.
    def forward(self, action, observation) -> float:
        # [FUNCTION DOC]
        # HELPER: Computes the reward as described for the class.
        return 1.0 if "sum=5" in observation.metadata.get("response", "") else 0.0


def make_mcp() -> FastMCP:
    # [FUNCTION DOC]
    # HELPER: Builds a FastMCP with "add" and a "read_file" tool that collides with the built-in.
    mcp = FastMCP("domain")

    @mcp.tool
    def add(a: int, b: int) -> int:
        # [FUNCTION DOC]
        # HELPER: Returns a+b.
        """Add two numbers."""
        return a + b

    @mcp.tool
    def read_file(path: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the path unchanged.
        """Read a file (collides with a harness builtin -> env_ prefix)."""
        return path

    return mcp


@pytest.fixture
def env_factory(tmp_path_factory, request):
    # [FUNCTION DOC]
    # Fixture returning a factory that builds a HarnessEnvironment using ScriptedCLIAdapter, the domain
    # MCP and SumRubric. The scripted mode defaults to "mcp" and can be overridden by indirect
    # parametrization.
    scripted_mode = getattr(request, "param", "mcp")

    def factory() -> HarnessEnvironment:
        # [FUNCTION DOC]
        # HELPER: Builds a fresh HarnessEnvironment with a new temporary config directory.
        config = HarnessConfig(
            name="scripted-cli",
            command=[sys.executable, "-u", str(SCRIPTED_HARNESS), scripted_mode],
            working_directory=".",
            startup_timeout_s=20.0,
            session_timeout_s=60.0,
        )
        adapter = ScriptedCLIAdapter(
            config, config_dir=tmp_path_factory.mktemp("harness-config")
        )
        return HarnessEnvironment(adapter=adapter, mcp=make_mcp(), rubric=SumRubric())

    return factory


class TestSimulationEndToEnd:
    # [CLASS DOC]
    # Simulation-mode end-to-end tests.
    def test_multi_turn_episode_through_real_stack(self, env_factory):
        # [FUNCTION DOC]
        # Reset injects add and env_read_file; the first step performs a tool call through the bridge with
        # reward 1.0 and the expected event sequence; a "finish" message ends the episode after two steps
        # with six trajectory events.
        env = env_factory()
        try:
            obs = env.reset(episode_id="e2e-episode")
            assert obs.done is False
            # Conflict resolution visible on the wire: the colliding
            # read_file was prefixed, add passed through untouched.
            assert sorted(obs.metadata["injected_tools"]) == [
                "add",
                "env_read_file",
            ]

            # Turn 1: the subprocess calls the env tool through the bridge
            obs1 = env.step(HarnessAction(message="please compute"))
            assert "sum=5" in obs1.metadata["response"]
            assert obs1.reward == 1.0  # rubric saw the turn's observation
            assert obs1.done is False
            assert [e["type"] for e in obs1.metadata["turn_events"]] == [
                "tool_call",
                "tool_result",
                "turn_complete",
            ]

            # Turn 2: the harness's done signal ends the episode
            obs2 = env.step(HarnessAction(message="finish up"))
            assert obs2.done is True
            assert env.state.step_count == 2
            assert len(env.trajectory) == 6  # both turns' events accumulated
        finally:
            env.close()


class TestProductionEndToEnd:
    # [CLASS DOC]
    # Production-mode end-to-end tests over a real server.
    @pytest.fixture
    def live_server(self, env_factory):
        # [FUNCTION DOC]
        # Fixture starting a uvicorn server for a production-mode app on a free localhost port in a daemon
        # thread, waiting up to 15s for startup, yielding the port, and shutting down afterwards.
        app = create_fastapi_app(
            env_factory, HarnessAction, Observation, mode="production"
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(
            target=lambda: server.run(sockets=[sock]), daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        try:
            yield port
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    def test_harness_websocket_over_real_server(self, live_server):
        # [FUNCTION DOC]
        # /health is OK and /reset is not exposed; the /harness WebSocket sends session_started, streams
        # tool_call, tool_result, turn_complete for a message, and a second "finish" turn reports done.
        port = live_server
        base = f"http://127.0.0.1:{port}"

        assert httpx.get(f"{base}/health").status_code == 200
        assert httpx.post(f"{base}/reset", json={}).status_code in (404, 405)

        with ws_connect(f"ws://127.0.0.1:{port}/harness") as websocket:
            started = json.loads(websocket.recv(timeout=30))
            assert started["type"] == "session_started"
            assert started["data"]["harness"] == "scripted-cli"
            assert started["data"]["session_id"]

            websocket.send(json.dumps({"type": "message", "content": "compute"}))
            frames = []
            while True:
                frame = json.loads(websocket.recv(timeout=30))
                frames.append(frame)
                if frame["type"] == "turn_complete":
                    break
            assert [f["type"] for f in frames] == [
                "tool_call",
                "tool_result",
                "turn_complete",
            ]
            assert "sum=5" in frames[-1]["data"]["response"]

            # Second turn on the same connection reuses the same harness
            websocket.send(json.dumps({"type": "message", "content": "finish"}))
            while True:
                frame = json.loads(websocket.recv(timeout=30))
                if frame["type"] == "turn_complete":
                    break
            assert frame["data"]["done"] is True

    @pytest.mark.parametrize("env_factory", ["exit-with-secret"], indirect=True)
    def test_startup_stderr_is_not_sent_to_websocket_client(self, live_server):
        # [FUNCTION DOC]
        # With a harness that prints a secret to stderr and exits, the client receives a generic
        # SESSION_ERROR frame that contains neither the secret nor a stderr tail.
        with ws_connect(f"ws://127.0.0.1:{live_server}/harness") as websocket:
            frame = json.loads(websocket.recv(timeout=30))
            assert frame["type"] == "error"
            assert frame["data"]["code"] == "SESSION_ERROR"
            assert frame["data"]["message"] == "harness session failed"
            assert "FAKE_HARNESS_SECRET_FOR_TEST" not in json.dumps(frame)
            assert "stderr tail" not in json.dumps(frame)
