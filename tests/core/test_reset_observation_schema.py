# [MODULE DOC]
# Tests that the /schema endpoint publishes an explicit reset observation schema separate from the
# step observation schema, defaults it to the step schema without instantiating the environment,
# keeps legacy SchemaResponse payloads valid, and that the reference echo env advertises its base
# reset observation.

"""Reset schemas are explicit declarations, independent of step observations."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_app, create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, SchemaResponse, State


class StepObservation(Observation):
    # [CLASS DOC]
    # HELPER: Observation with a required integer value.
    value: int


class ResetObservation(Observation):
    # [CLASS DOC]
    # HELPER: Observation with a required boolean ready.
    ready: bool


class DistinctResetEnvironment(Environment[Action, StepObservation, State]):
    # [CLASS DOC]
    # HELPER: Environment whose reset returns a ResetObservation and whose step returns a done
    # StepObservation with reward 1.0.
    def reset(self, seed=None, episode_id=None):
        # [FUNCTION DOC]
        # HELPER: Returns ResetObservation(ready=True).
        return ResetObservation(ready=True, reward=None)

    def step(self, action):
        # [FUNCTION DOC]
        # HELPER: Returns StepObservation(value=1) with reward 1.0 and done.
        return StepObservation(value=1, reward=1.0, done=True)

    @property
    def state(self):
        # [FUNCTION DOC]
        # HELPER: Returns a fixed State.
        return State(episode_id="reset-schema", step_count=0)


@pytest.mark.parametrize("web_enabled", [False, True])
def test_explicit_reset_model_is_published_and_matches_raw_reset(
    monkeypatch, web_enabled
):
    # [FUNCTION DOC]
    # Parametrized web interface on/off: /schema publishes distinct reset and step schemas and a
    # WebSocket reset returns the reset observation fields only.
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", str(web_enabled).lower())
    app = create_app(
        DistinctResetEnvironment,
        Action,
        StepObservation,
        reset_observation_cls=ResetObservation,
    )
    with TestClient(app) as client:
        schema = client.get("/schema").json()
        assert schema["reset_observation"] == ResetObservation.model_json_schema()
        assert schema["observation"] == StepObservation.model_json_schema()
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            response = ws.receive_json()
            assert response["data"]["observation"]["ready"] is True
            assert "value" not in response["data"]["observation"]


def test_default_reset_schema_keeps_declared_model_without_creating_environment():
    # [FUNCTION DOC]
    # Without an explicit reset class the reset schema equals the step schema and the factory is never called.
    def factory():
        # [FUNCTION DOC]
        # HELPER: Factory that fails the test if called.
        raise AssertionError("Schema discovery must not instantiate the environment")

    with TestClient(create_fastapi_app(factory, Action, StepObservation)) as client:
        schema = client.get("/schema").json()
    assert schema["reset_observation"] == schema["observation"]
    assert schema["reset_observation"]["required"] == ["value"]


def test_schema_response_still_accepts_legacy_three_schema_payload():
    # [FUNCTION DOC]
    # A SchemaResponse without reset_observation omits the field in dumps and over HTTP.
    response = SchemaResponse(action={}, observation={}, state={})
    assert response.reset_observation is None
    assert "reset_observation" not in response.model_dump()
    assert "reset_observation" not in json.loads(response.model_dump_json())
    app = FastAPI()

    @app.get("/schema", response_model=SchemaResponse)
    def schema():
        # [FUNCTION DOC]
        # HELPER: Route returning the legacy response.
        return response

    with TestClient(app) as client:
        assert "reset_observation" not in client.get("/schema").json()


def test_reference_echo_explicitly_advertises_its_base_reset_observation():
    # [FUNCTION DOC]
    # The echo app publishes the base Observation as reset schema and its reset omits tool_name.
    from echo_env.server.app import app

    with TestClient(app) as client:
        schema = client.get("/schema").json()
        assert schema["reset_observation"] == Observation.model_json_schema()
        assert "tool_name" in schema["observation"]["required"]
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"episode_id": "reset-schema"}})
            assert "tool_name" not in ws.receive_json()["data"]["observation"]
