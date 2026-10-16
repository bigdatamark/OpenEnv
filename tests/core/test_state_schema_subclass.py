# [MODULE DOC]
# Regression tests that /schema and /state serve an environment's declared
# State subclass (passed as state_cls) instead of the base State model, and
# fall back to base State when no subclass is declared.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the HTTP surface serves an environment's own `State` subclass.

Regression coverage for the case where `/schema` and `/state` were wired to the
base `State` model: every field an environment declared on its `State` subclass
was missing from the published schema and stripped from the response body, while
the WebSocket `state` frame kept them. The two transports disagreed about the
same object.
"""

import pytest
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import (
    create_app,
    create_fastapi_app,
    HTTPEnvServer,
)
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class EchoAction(Action):
    # [CLASS DOC]
    # Action with an optional string "message".
    """Action for the fixture environment."""

    message: str = ""


class EchoObservation(Observation):
    # [CLASS DOC]
    # Observation with an optional string "response".
    """Observation for the fixture environment."""

    response: str = ""


class EchoState(State):
    # [CLASS DOC]
    # State subclass adding an integer counter and a list-of-strings history.
    """State subclass declaring fields the base model does not have."""

    counter: int = 0
    history: list[str] = Field(default_factory=list)


class EchoEnvironment(Environment[EchoAction, EchoObservation, EchoState]):
    # [CLASS DOC]
    # Environment whose state is a fixed EchoState (episode "ep-1", step 3,
    # counter 42, history a,b).
    """Minimal environment whose state is an `EchoState`."""

    def __init__(self):
        # [FUNCTION DOC]
        # Calls the base initializer and builds the fixed EchoState.
        super().__init__()
        self._state = EchoState(
            episode_id="ep-1", step_count=3, counter=42, history=["a", "b"]
        )

    @property
    def state(self) -> EchoState:
        # [FUNCTION DOC]
        # Property returning the stored EchoState.
        return self._state

    def reset(self) -> EchoObservation:
        # [FUNCTION DOC]
        # Returns an EchoObservation with empty response; state is not reset.
        return EchoObservation(response="")

    def step(self, action: EchoAction) -> EchoObservation:
        # [FUNCTION DOC]
        # Increments state counter (mutation) and echoes the action message.
        self._state.counter += 1
        return EchoObservation(response=action.message)


@pytest.fixture
def declared_state_client() -> TestClient:
    # [FUNCTION DOC]
    # Fixture: TestClient for create_fastapi_app with state_cls=EchoState.
    """Client for an app that declares its `State` subclass."""
    app = create_fastapi_app(
        EchoEnvironment,
        EchoAction,
        EchoObservation,
        env_name="echo_env",
        state_cls=EchoState,
    )
    return TestClient(app)


@pytest.fixture
def default_state_client() -> TestClient:
    # [FUNCTION DOC]
    # Fixture: TestClient for create_fastapi_app without state_cls.
    """Client for an app that declares no `State` subclass."""
    app = create_fastapi_app(
        EchoEnvironment,
        EchoAction,
        EchoObservation,
        env_name="echo_env",
    )
    return TestClient(app)


class TestDeclaredStateClass:
    # [CLASS DOC]
    # Tests for apps that declare their State subclass.
    """An environment that passes its `State` subclass gets it served."""

    @pytest.mark.parametrize("web_enabled", [False, True])
    def test_create_app_serves_declared_state_in_both_modes(
        self, monkeypatch, web_enabled
    ):
        # [FUNCTION DOC]
        # Parametrized over ENABLE_WEB_INTERFACE (set via monkeypatch); create_app must
        # serve counter in /state and in the /schema state properties.
        monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true" if web_enabled else "false")
        app = create_app(
            EchoEnvironment,
            EchoAction,
            EchoObservation,
            env_name="echo_env",
            state_cls=EchoState,
        )
        client = TestClient(app)

        assert client.get("/state").json()["counter"] == 42
        assert "counter" in client.get("/schema").json()["state"]["properties"]

    def test_schema_publishes_subclass_fields(self, declared_state_client):
        # [FUNCTION DOC]
        # /schema state properties equal exactly the base plus subclass field names.
        response = declared_state_client.get("/schema")

        assert response.status_code == 200
        properties = response.json()["state"]["properties"]
        assert set(properties) == {"episode_id", "step_count", "counter", "history"}

    def test_state_response_keeps_subclass_fields(self, declared_state_client):
        # [FUNCTION DOC]
        # /state body contains subclass and base field values.
        response = declared_state_client.get("/state")

        assert response.status_code == 200
        body = response.json()
        assert body["counter"] == 42
        assert body["history"] == ["a", "b"]
        assert body["episode_id"] == "ep-1"
        assert body["step_count"] == 3

    def test_action_and_observation_schemas_are_unaffected(self, declared_state_client):
        # [FUNCTION DOC]
        # /schema still lists action "message" and observation "response".
        payload = declared_state_client.get("/schema").json()

        assert "message" in payload["action"]["properties"]
        assert "response" in payload["observation"]["properties"]


class TestDefaultStateClass:
    # [CLASS DOC]
    # Tests for apps that do not declare state_cls.
    """Omitting `state_cls` keeps the previous base-model behaviour."""

    def test_schema_falls_back_to_base_state(self, default_state_client):
        # [FUNCTION DOC]
        # /schema state properties are only episode_id and step_count.
        properties = default_state_client.get("/schema").json()["state"]["properties"]

        assert set(properties) == {"episode_id", "step_count"}

    def test_state_response_is_serialized_through_the_base_model(
        self, default_state_client
    ):
        # [FUNCTION DOC]
        # /state body keeps base fields and drops subclass-only fields.
        body = default_state_client.get("/state").json()

        assert body["episode_id"] == "ep-1"
        assert body["step_count"] == 3
        assert "counter" not in body
        assert "history" not in body

    def test_server_defaults_to_base_state(self):
        # [FUNCTION DOC]
        # HTTPEnvServer.state_cls is the base State when not given.
        server = HTTPEnvServer(EchoEnvironment, EchoAction, EchoObservation)

        assert server.state_cls is State
