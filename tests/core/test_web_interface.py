# [MODULE DOC]
# Tests for create_web_interface_app (Gradio web wrapper): /web/reset with
# environments whose reset takes no kwargs, root redirects to /web/,
# functools.partial env factories, and the REPL environment's web flow
# (conflict before reset, reset kwargs, hf_token never echoed).
# Skipped when gradio or smolagents are not installed.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OpenEnv Gradio web interface helpers."""

from __future__ import annotations

import json
from functools import partial

import pytest
from fastapi.testclient import TestClient
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.web_interface import create_web_interface_app

pytest.importorskip("gradio", reason="gradio is not installed")
pytest.importorskip("smolagents", reason="smolagents is not installed")

from repl_env.models import REPLAction, REPLObservation
from repl_env.server.repl_environment import REPLEnvironment


class NoKwargAction(Action):
    # [CLASS DOC]
    # Action with an optional message (default "noop").
    """Minimal action for exercising the web wrapper."""

    message: str = "noop"


class NoKwargObservation(Observation):
    # [CLASS DOC]
    # Observation with a required response string.
    """Minimal observation for exercising the web wrapper."""

    response: str
    reward: float | None = None
    done: bool = False


class NoKwargState(State):
    # [CLASS DOC]
    # State with step_count and a last_reset_marker string (default "default").
    """Minimal state for exercising the web wrapper."""

    step_count: int = 0
    last_reset_marker: str = "default"


class NoKwargEnvironment(Environment):
    # [CLASS DOC]
    # Environment whose reset accepts no keyword arguments.
    """Environment whose reset signature intentionally accepts no kwargs."""

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes a default NoKwargState.
        super().__init__()
        self._state = NoKwargState()

    def reset(self) -> NoKwargObservation:
        # [FUNCTION DOC]
        # Replaces the state with a fresh one and returns response "reset".
        self._state = NoKwargState(step_count=0, last_reset_marker="default")
        return NoKwargObservation(response="reset")

    def step(self, action: NoKwargAction) -> NoKwargObservation:
        # [FUNCTION DOC]
        # Increments step_count and echoes the action message with reward 0.0.
        self._state.step_count += 1
        return NoKwargObservation(response=action.message, reward=0.0, done=False)

    @property
    def state(self) -> NoKwargState:
        # [FUNCTION DOC]
        # Property returning the current state.
        return self._state

    def close(self) -> None:
        # [FUNCTION DOC]
        # No-op.
        pass


def test_web_reset_accepts_no_body_and_ignores_unsupported_kwargs() -> None:
    # [FUNCTION DOC]
    # POST /web/reset works with no body and with an unknown key (dropped);
    # /web/state still shows the default marker.
    """POST /web/reset should preserve old behavior and ignore unsupported kwargs."""
    app = create_web_interface_app(
        NoKwargEnvironment,
        NoKwargAction,
        NoKwargObservation,
    )
    client = TestClient(app)

    no_body = client.post("/web/reset")
    assert no_body.status_code == 200
    assert no_body.json()["observation"]["response"] == "reset"

    extra_body = client.post("/web/reset", json={"unused": "value"})
    assert extra_body.status_code == 200
    assert extra_body.json()["observation"]["response"] == "reset"

    state = client.get("/web/state")
    assert state.status_code == 200
    assert state.json()["last_reset_marker"] == "default"


def test_web_root_redirects_to_gradio_interface() -> None:
    # [FUNCTION DOC]
    # GET / and GET /web return 307 redirects to /web/.
    """GET / should redirect to /web/ so HF Space embeds have a live root page."""
    app = create_web_interface_app(
        NoKwargEnvironment,
        NoKwargAction,
        NoKwargObservation,
    )
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/web/"

    web_response = client.get("/web", follow_redirects=False)
    assert web_response.status_code == 307
    assert web_response.headers["location"] == "/web/"


def test_web_interface_instantiates_partial_factory() -> None:
    # [FUNCTION DOC]
    # A partial factory is instantiated; reset works and /web/metadata names the
    # underlying class.
    """A `functools.partial` factory should be instantiated, like a class or function."""
    app = create_web_interface_app(
        partial(NoKwargEnvironment),
        NoKwargAction,
        NoKwargObservation,
    )
    client = TestClient(app)

    reset_response = client.post("/web/reset")
    assert reset_response.status_code == 200
    assert reset_response.json()["observation"]["response"] == "reset"

    metadata = client.get("/web/metadata").json()
    assert metadata["name"] == "NoKwargEnvironment"


def test_repl_web_state_before_reset_returns_conflict() -> None:
    # [FUNCTION DOC]
    # For REPLEnvironment, /web/state before reset returns 409 mentioning
    # "Call reset() first".
    """GET /web/state should fail cleanly before reset instead of crashing."""
    app = create_web_interface_app(
        REPLEnvironment,
        REPLAction,
        REPLObservation,
        env_name="repl_env",
    )
    client = TestClient(app)

    response = client.get("/web/state")
    assert response.status_code == 409
    assert "Call reset() first" in response.json()["detail"]


def test_repl_web_reset_passes_context_and_task_prompt_without_echoing_hf_token() -> (
    None
):
    # [FUNCTION DOC]
    # Resets the REPL env with context, task_prompt and hf_token, runs a code step
    # counting words (expects "3"), checks state fields, and asserts the token
    # appears nowhere in the combined reset/step/state JSON.
    """The REPL web flow should accept reset kwargs and keep the token out of state."""
    app = create_web_interface_app(
        REPLEnvironment,
        REPLAction,
        REPLObservation,
        env_name="repl_env",
    )
    client = TestClient(app)

    reset_response = client.post(
        "/web/reset",
        json={
            "context": "alpha beta gamma",
            "task_prompt": "Count the words",
            "hf_token": "super-secret-token",
        },
    )
    assert reset_response.status_code == 200
    reset_json = reset_response.json()
    assert reset_json["observation"]["context_preview"] == "alpha beta gamma"
    assert "context" in reset_json["observation"]["available_variables"]

    step_response = client.post(
        "/web/step",
        json={"action": {"code": "count = len(context.split())"}},
    )
    assert step_response.status_code == 200
    step_json = step_response.json()
    assert step_json["observation"]["result"]["success"] is True
    assert step_json["observation"]["result"]["locals_snapshot"]["count"] == "3"

    state_response = client.get("/web/state")
    assert state_response.status_code == 200
    state_json = state_response.json()
    assert state_json["context"] == "alpha beta gamma"
    assert state_json["task_prompt"] == "Count the words"
    assert "count" in state_json["namespace_keys"]

    combined_output = json.dumps(
        {
            "reset": reset_json,
            "step": step_json,
            "state": state_json,
        }
    )
    assert "super-secret-token" not in combined_output
