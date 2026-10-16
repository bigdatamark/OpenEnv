# [MODULE DOC]
# Tests that BrowserGymEnvironment.reset(task_name=...) rebuilds the gym env
# for a new task, and keeps the old env if building the new one fails. If
# gymnasium is missing, a temporary stub module is inserted into sys.modules
# for the import and removed afterwards; prepends repo root to sys.path.
"""Unit tests for BrowserGym task switching behavior."""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    sys.modules["gymnasium"] = types.SimpleNamespace(
        make=lambda *_args, **_kwargs: None
    )
    _INSERTED_GYMNASIUM_STUB = True
else:
    _INSERTED_GYMNASIUM_STUB = False

from envs.browsergym_env.server import browsergym_environment
from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment

if _INSERTED_GYMNASIUM_STUB:
    sys.modules.pop("gymnasium", None)


class _FakeGymEnv:
    # [CLASS DOC]
    # HELPER: fake gym env recording its id, reset kwargs and closed state.
    def __init__(self, env_id: str) -> None:
        # [FUNCTION DOC]
        # Stores env_id, closed False and an empty reset-call log.
        self.env_id = env_id
        self.closed = False
        self.reset_calls: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any):
        # [FUNCTION DOC]
        # Records kwargs; returns an obs dict with a goal naming the env id, and empty info.
        self.reset_calls.append(dict(kwargs))
        return {"goal": f"goal for {self.env_id}", "url": "http://example.test"}, {}

    def close(self) -> None:
        # [FUNCTION DOC]
        # Marks the env closed.
        self.closed = True


def test_reset_rebuilds_browsergym_env_when_task_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # With gym.make and importlib patched, reset to a new task creates the new env
    # id, closes the old env, updates task_name/state, forwards the seed and
    # returns the new goal.
    created_envs: list[tuple[str, dict[str, Any], _FakeGymEnv]] = []

    monkeypatch.setattr(
        browsergym_environment.importlib,
        "import_module",
        lambda _name: object(),
    )

    def fake_make(env_id: str, **kwargs: Any) -> _FakeGymEnv:
        # [FUNCTION DOC]
        # Creates a _FakeGymEnv and records (id, kwargs, env).
        env = _FakeGymEnv(env_id)
        created_envs.append((env_id, dict(kwargs), env))
        return env

    monkeypatch.setattr(browsergym_environment.gym, "make", fake_make)

    env = BrowserGymEnvironment(
        benchmark="miniwob",
        task_name="click-test",
        headless=True,
    )
    original_env = env.gym_env

    observation = env.reset(task_name="enter-text", seed=7)

    assert [env_id for env_id, _, _ in created_envs] == [
        "browsergym/miniwob.click-test",
        "browsergym/miniwob.enter-text",
    ]
    assert original_env.closed is True
    assert env.gym_env is created_envs[1][2]
    assert env.task_name == "enter-text"
    assert env.state.task_name == "enter-text"
    assert created_envs[1][2].reset_calls == [{"seed": 7}]
    assert observation.goal == "goal for browsergym/miniwob.enter-text"


def test_failed_task_switch_keeps_previous_browsergym_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # When making the new task's env fails, reset raises ValueError naming the env
    # id and the previous env/task stay active and open; once making succeeds the
    # switch happens.
    fail_enter_text = True
    created_envs: list[_FakeGymEnv] = []

    monkeypatch.setattr(
        browsergym_environment.importlib,
        "import_module",
        lambda _name: object(),
    )

    def fake_make(env_id: str, **_kwargs: Any) -> _FakeGymEnv:
        # [FUNCTION DOC]
        # Raises RuntimeError for the enter-text env while the enclosing failure flag
        # is set; otherwise creates and records a _FakeGymEnv.
        if env_id == "browsergym/miniwob.enter-text" and fail_enter_text:
            raise RuntimeError("task is unavailable")
        env = _FakeGymEnv(env_id)
        created_envs.append(env)
        return env

    monkeypatch.setattr(browsergym_environment.gym, "make", fake_make)

    env = BrowserGymEnvironment(benchmark="miniwob", task_name="click-test")
    original_env = env.gym_env

    with pytest.raises(ValueError, match="browsergym/miniwob.enter-text"):
        env.reset(task_name="enter-text")

    assert env.gym_env is original_env
    assert original_env.closed is False
    assert env.task_name == "click-test"
    assert env.env_id == "browsergym/miniwob.click-test"
    assert env.state.task_name == "click-test"

    fail_enter_text = False
    observation = env.reset(task_name="enter-text")

    assert original_env.closed is True
    assert env.gym_env is created_envs[-1]
    assert env.task_name == "enter-text"
    assert env.env_id == "browsergym/miniwob.enter-text"
    assert observation.goal == "goal for browsergym/miniwob.enter-text"
