# [MODULE DOC]
# Tests for ORS-compatible task/split HTTP routes registered by HTTPEnvServer
# (/list_environments, /<env>/splits, tasks, num_tasks, task, task_range), plus
# /step and /mcp handling of environments whose step_async is a
# functools.partial and of MCP calls returning the wrong observation type.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ORS-compatible task and split endpoints."""

import functools

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_app, HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation, State


class TaskAction(Action):
    # [CLASS DOC]
    # Action with an optional string "value".
    value: str = ""


class TaskObservation(Observation):
    # [CLASS DOC]
    # Observation with an optional string "message".
    message: str = ""


class TaskEnvironment(Environment):
    # [CLASS DOC]
    # Environment implementing the task API with splits "train" and "holdout",
    # two tasks per split.
    def reset(self, **kwargs) -> TaskObservation:
        # [FUNCTION DOC]
        # Returns an observation with message "ready".
        return TaskObservation(message="ready")

    def step(self, action: TaskAction, **kwargs) -> TaskObservation:
        # [FUNCTION DOC]
        # Echoes action.value as message with reward 1.0.
        return TaskObservation(message=action.value, reward=1.0)

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning a fresh default State.
        return State()

    def list_splits(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the split names ["train", "holdout"].
        return ["train", "holdout"]

    def list_tasks(self, split: str) -> list[dict[str, str]]:
        # [FUNCTION DOC]
        # Returns two task dicts with ids "<split>-0" and "<split>-1".
        return [{"id": f"{split}-0"}, {"id": f"{split}-1"}]

    def num_tasks(self, split: str) -> int:
        # [FUNCTION DOC]
        # Always returns 2.
        return 2

    def get_task(self, split: str, index: int) -> dict[str, str | int]:
        # [FUNCTION DOC]
        # Returns a dict with id "<split>-<index>" and the index.
        return {"id": f"{split}-{index}", "index": index}

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, str | int]]:
        # [FUNCTION DOC]
        # Returns task dicts for indices start..stop-1 (start defaults 0, stop 2).
        start = 0 if start is None else start
        stop = 2 if stop is None else stop
        return [{"id": f"{split}-{i}", "index": i} for i in range(start, stop)]


class UnsupportedTaskEnvironment(Environment):
    # [CLASS DOC]
    # Environment without task API methods (routes should return 501).
    def reset(self, **kwargs) -> TaskObservation:
        # [FUNCTION DOC]
        # Returns an observation with message "ready".
        return TaskObservation(message="ready")

    def step(self, action: TaskAction, **kwargs) -> TaskObservation:
        # [FUNCTION DOC]
        # Echoes action.value as message.
        return TaskObservation(message=action.value)

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning a default State.
        return State()


class PartialStepAsyncEnvironment(Environment):
    # [CLASS DOC]
    # Environment whose instance step_async is a functools.partial wrapping an
    # async function, to test detection of non-plain coroutine functions.
    def __init__(self):
        # [FUNCTION DOC]
        # Defines the nested async step and assigns its partial to self.step_async.
        async def step_async(action: CallToolAction, **kwargs) -> CallToolObservation:
            # [FUNCTION DOC]
            # Async: returns CallToolObservation for the action's tool with result ok=True.
            return CallToolObservation(tool_name=action.tool_name, result={"ok": True})

        self.step_async = functools.partial(step_async)  # type: ignore[method-assign]

    def reset(self, **kwargs) -> TaskObservation:
        # [FUNCTION DOC]
        # Returns an observation with message "ready".
        return TaskObservation(message="ready")

    def step(self, action: CallToolAction, **kwargs) -> CallToolObservation:
        # [FUNCTION DOC]
        # Sync fallback returning result sync=True (should not be used by /step).
        return CallToolObservation(tool_name=action.tool_name, result={"sync": True})

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning a default State.
        return State()


class PartialMcpStepAsyncEnvironment(Environment):
    # [CLASS DOC]
    # Environment with a partial-wrapped async step_async handling MCP actions.
    def __init__(self):
        # [FUNCTION DOC]
        # Defines the nested async step and assigns its partial to self.step_async.
        async def step_async(action: Action, **kwargs) -> Observation:
            # [FUNCTION DOC]
            # Async: returns an empty ListToolsObservation for ListToolsAction, otherwise
            # a CallToolObservation for "unknown" with no result.
            if isinstance(action, ListToolsAction):
                return ListToolsObservation(tools=[])
            return CallToolObservation(tool_name="unknown", result=None)

        self.step_async = functools.partial(step_async)  # type: ignore[method-assign]

    def reset(self, **kwargs) -> TaskObservation:
        # [FUNCTION DOC]
        # Returns an observation with message "ready".
        return TaskObservation(message="ready")

    def step(self, action: Action, **kwargs) -> Observation:
        # [FUNCTION DOC]
        # Sync fallback returning a TaskObservation "sync".
        return TaskObservation(message="sync")

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning a default State.
        return State()


class WrongMcpStyleEnvironment(Environment):
    # [CLASS DOC]
    # Environment whose step always returns a TaskObservation, i.e. the wrong
    # type for MCP list/call actions.
    def reset(self, **kwargs) -> TaskObservation:
        # [FUNCTION DOC]
        # Returns an observation with message "ready".
        return TaskObservation(message="ready")

    def step(self, action: Action, **kwargs) -> Observation:
        # [FUNCTION DOC]
        # Returns TaskObservation "wrong:<action class name>".
        return TaskObservation(message=f"wrong:{type(action).__name__}")

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning a default State.
        return State()


def test_task_routes_expose_ors_compatible_shapes() -> None:
    # [FUNCTION DOC]
    # Checks exact JSON shapes of all task routes; "holdout" split maps to type
    # "validation", "train" to "train".
    app = FastAPI()
    server = HTTPEnvServer(
        env=TaskEnvironment,
        action_cls=TaskAction,
        observation_cls=TaskObservation,
        env_name="task_env",
    )
    server.register_routes(app)
    client = TestClient(app)

    assert client.get("/list_environments").json() == ["task_env"]
    assert client.get("/task_env/splits").json() == [
        {"name": "train", "type": "train"},
        {"name": "holdout", "type": "validation"},
    ]
    assert client.post("/task_env/tasks", json={"split": "train"}).json() == {
        "tasks": [{"id": "train-0"}, {"id": "train-1"}],
        "env_name": "task_env",
    }
    assert client.post("/task_env/num_tasks", json={"split": "train"}).json() == {
        "num_tasks": 2
    }
    assert client.post(
        "/task_env/task", json={"split": "train", "index": 1}
    ).json() == {"task": {"id": "train-1", "index": 1}}
    assert client.post(
        "/task_env/task_range",
        json={"split": "train", "start": 0, "stop": 2},
    ).json() == {
        "tasks": [{"id": "train-0", "index": 0}, {"id": "train-1", "index": 1}]
    }


def test_task_routes_reject_unknown_environment_name() -> None:
    # [FUNCTION DOC]
    # Requests for a different env name return 404.
    app = FastAPI()
    server = HTTPEnvServer(
        env=TaskEnvironment,
        action_cls=TaskAction,
        observation_cls=TaskObservation,
        env_name="task_env",
    )
    server.register_routes(app)
    client = TestClient(app)

    response = client.get("/other_env/splits")

    assert response.status_code == 404


def test_task_routes_return_501_when_environment_does_not_support_tasks() -> None:
    # [FUNCTION DOC]
    # Splits route returns 501 for environments without the task API.
    app = FastAPI()
    server = HTTPEnvServer(
        env=UnsupportedTaskEnvironment,
        action_cls=TaskAction,
        observation_cls=TaskObservation,
        env_name="plain_env",
    )
    server.register_routes(app)
    client = TestClient(app)

    response = client.get("/plain_env/splits")

    assert response.status_code == 501


def test_create_app_threads_env_name_to_task_routes() -> None:
    # [FUNCTION DOC]
    # create_app passes env_name through to the task routes.
    app = create_app(
        TaskEnvironment,
        TaskAction,
        TaskObservation,
        env_name="created_env",
    )
    client = TestClient(app)

    assert client.get("/list_environments").json() == ["created_env"]
    assert client.get("/created_env/splits").status_code == 200


def test_step_route_handles_partial_step_async() -> None:
    # [FUNCTION DOC]
    # /step awaits the partial-wrapped step_async (result ok=True, not sync).
    app = create_app(
        PartialStepAsyncEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="partial_env",
    )
    client = TestClient(app)

    response = client.post(
        "/step",
        json={"action": {"tool_name": "submit", "arguments": {}}},
    )

    assert response.status_code == 200
    assert response.json()["observation"]["result"] == {"ok": True}


def test_mcp_style_step_handles_partial_step_async() -> None:
    # [FUNCTION DOC]
    # HTTP POST /mcp tools/list uses the partial step_async and returns empty tools.
    app = create_app(
        PartialMcpStepAsyncEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="partial_mcp_env",
    )
    client = TestClient(app)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
    )

    assert response.status_code == 200
    assert response.json()["result"] == {"tools": []}


def test_mcp_style_tools_list_reports_wrong_observation_type() -> None:
    # [FUNCTION DOC]
    # /mcp tools/list returns a JSON-RPC error mentioning expected ListToolsObservation.
    app = create_app(
        WrongMcpStyleEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="wrong_mcp_env",
    )
    client = TestClient(app)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
    )

    assert response.status_code == 200
    assert "expected ListToolsObservation" in response.json()["error"]["message"]


def test_mcp_style_tools_call_reports_wrong_observation_type() -> None:
    # [FUNCTION DOC]
    # /mcp tools/call returns a JSON-RPC error mentioning expected CallToolObservation.
    app = create_app(
        WrongMcpStyleEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="wrong_mcp_env",
    )
    client = TestClient(app)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "submit", "arguments": {}},
            "id": 1,
        },
    )

    assert response.status_code == 200
    assert "expected CallToolObservation" in response.json()["error"]["message"]
