# [MODULE DOC]
# Tests the served_probe fixture app (loaded by file path at import time from
# tests/fixtures/validation/runtime/served_probe/app.py) and the real Echo
# environment over the production OpenEnv HTTP/WebSocket server, including
# fault modes visible on the wire and the runtime collector/graders against a
# live uvicorn server.
# Side effects at import: executes the fixture module.
"""Test fixture faults over the production OpenEnv WebSocket endpoint."""

import importlib.util
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from starlette.testclient import TestClient


FIXTURE = Path(__file__).parents[2] / "fixtures/validation/runtime/served_probe/app.py"
spec = importlib.util.spec_from_file_location("served_probe_app", FIXTURE)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def observation(ws, request):
    # [FUNCTION DOC]
    # HELPER: Sends a JSON request over a test WebSocket, receives one reply,
    # asserts its type is "observation" and returns its "data" payload.
    # Raises: AssertionError if the reply type differs.
    ws.send_json(request)
    result = ws.receive_json()
    assert result["type"] == "observation"
    return result["data"]


def test_real_session_retains_counter_and_requested_identity():
    # [FUNCTION DOC]
    # Checks /health and /schema, then over one WebSocket session resets with an
    # episode id and seed and steps twice, verifying the counter increments, done
    # becomes true on step 2 and state reports the episode id and step count; a new
    # session reset starts again at counter 0.
    with TestClient(fixture.make_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/schema").json()["observation"]["properties"]["counter"]
        with client.websocket_connect("/ws") as ws:
            initial = observation(
                ws, {"type": "reset", "data": {"episode_id": "test", "seed": 42}}
            )
            assert initial["observation"]["counter"] == 0
            for index in (1, 2):
                result = observation(ws, {"type": "step", "data": {"increment": 1}})
                assert result["observation"]["counter"] == index
                assert result["done"] is (index == 2)
                ws.send_json({"type": "state"})
                assert ws.receive_json()["data"] == {
                    "episode_id": "test",
                    "step_count": index,
                }
        with client.websocket_connect("/ws") as ws:
            result = observation(ws, {"type": "reset", "data": {"episode_id": "fresh"}})
            assert result["observation"]["counter"] == 0


@pytest.mark.parametrize(
    "mode", ["bad_reward", "bad_observation", "missing_done", "bad_state"]
)
def test_fault_is_visible_in_raw_wire_response(mode):
    # [FUNCTION DOC]
    # Parametrized over fault modes; resets and checks the raw response shows the
    # fault: reward True, a non-integer counter, a missing done field, or a wrong
    # episode id in state.
    with TestClient(fixture.make_app(mode)) as client:
        with client.websocket_connect("/ws") as ws:
            result = observation(ws, {"type": "reset", "data": {"episode_id": "test"}})
            if mode == "bad_reward":
                assert result["reward"] is True
            elif mode == "bad_observation":
                assert result["observation"]["counter"] == "not-an-integer"
            elif mode == "missing_done":
                assert "done" not in result
            else:
                ws.send_json({"type": "state"})
                assert ws.receive_json()["data"]["episode_id"] == "wrong-episode"


def test_failed_start_is_explicit():
    # [FUNCTION DOC]
    # Expects building the app in startup_failure mode to raise RuntimeError with
    # "Deliberate startup failure".
    with pytest.raises(RuntimeError, match="Deliberate startup failure"):
        fixture.make_app("startup_failure")


@pytest.mark.parametrize("legacy_schema", [False, True])
def test_real_echo_reset_schema_and_unscored_rewards(legacy_schema, monkeypatch):
    # [FUNCTION DOC]
    # Parametrized on legacy_schema. Builds the real Echo FastAPI app (with a
    # distinct reset observation class), optionally wrapped so /schema responses
    # drop the reset_observation key (emulating an old server). Serves it with
    # uvicorn on a free localhost port in a thread, collects runtime evidence with
    # a one-action plan, and stops the server. Asserts no failure, reset schema
    # presence matches the mode, the step response has null reward / done False and
    # echoes the message, then runs graders: reward well-formed fails, state
    # contract passes, observation schema fails only with the legacy schema.
    # Side effects: prepends envs/ to sys.path; binds a local socket; starts a
    # server thread.
    # The lab deliberately clears PYTHONPATH and tests the installed core wheel.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "envs"))
    from echo_env.server.echo_environment import EchoEnvironment
    from openenv.core.env_server.http_server import create_fastapi_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Observation
    from openenv.validation.graders.runtime import (
        ObservationSchemaGrader,
        RewardWellFormedGrader,
        StateContractGrader,
    )
    from openenv.validation.runtime.collector import collect_runtime_evidence
    from openenv.validation.runtime.contracts import RuntimePlan

    app = create_fastapi_app(
        EchoEnvironment,
        CallToolAction,
        CallToolObservation,
        reset_observation_cls=Observation,
    )

    async def served_app(scope, receive, send):
        # [FUNCTION DOC]
        # HELPER: ASGI wrapper that forwards to the Echo app, substituting legacy_send
        # for the send callable on /schema requests when legacy_schema is set.
        async def legacy_send(message):
            # [FUNCTION DOC]
            # HELPER: ASGI send wrapper: strips the content-length header from the
            # response start and removes "reset_observation" from the JSON body before
            # forwarding.
            # Raises: KeyError if the body lacks reset_observation.
            # Reproduce an older server's actual single-schema HTTP response.
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        (key, value)
                        for key, value in message["headers"]
                        if key != b"content-length"
                    ],
                }
            elif message["type"] == "http.response.body":
                schema = json.loads(message["body"])
                schema.pop("reset_observation")
                message = {**message, "body": json.dumps(schema).encode()}
            await send(message)

        await app(
            scope,
            receive,
            legacy_send if legacy_schema and scope.get("path") == "/schema" else send,
        )

    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "echo-contract", "seed": 42},
            "actions": [
                {
                    "tool_name": "echo_message",
                    "arguments": {"message": "real contract probe"},
                }
            ],
        }
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(served_app, log_level="error", lifespan="on")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            evidence = collect_runtime_evidence(
                f"http://127.0.0.1:{port}", plan, episode_timeout_s=5
            )
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert evidence.failure_reason is None
    assert (evidence.reset_observation_schema_json is None) is legacy_schema
    step = next(row for row in evidence.exchanges if row.operation == "step")
    response = json.loads(step.response_json)["data"]
    assert response["reward"] is None and response["done"] is False
    assert "real contract probe" in json.dumps(response["observation"]["result"])
    subject = SimpleNamespace(
        runtime_evidence=evidence,
        manifest=SimpleNamespace(reward=SimpleNamespace(range=(0, 1))),
    )
    assert RewardWellFormedGrader().run(subject).status.value == "fail"
    assert StateContractGrader().run(subject).status.value == "pass"
    assert ObservationSchemaGrader().run(subject).status.value == (
        "fail" if legacy_schema else "pass"
    )
