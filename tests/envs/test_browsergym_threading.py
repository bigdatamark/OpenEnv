# [MODULE DOC]
# Tests that BrowserGymEnvironment runs every gym call on a single dedicated
# thread regardless of the calling thread (Playwright sync API requirement).
# Uses a gymnasium stub when gymnasium is not installed.
"""Unit tests for BrowserGym thread affinity."""

from __future__ import annotations

import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
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

from envs.browsergym_env.models import BrowserGymAction
from envs.browsergym_env.server import browsergym_environment
from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment

if _INSERTED_GYMNASIUM_STUB:
    sys.modules.pop("gymnasium", None)


class _ThreadRecordingGymEnv:
    # [CLASS DOC]
    # HELPER: fake gym env appending the current thread id to a shared list on
    # every reset/step/close.
    """Records the thread of every call, like Playwright's sync API cares about."""

    def __init__(self, threads: list[int]) -> None:
        # [FUNCTION DOC]
        # Stores the shared thread-id list.
        self.threads = threads

    def reset(self, **_kwargs: Any):
        # [FUNCTION DOC]
        # Records thread id; returns a fixed obs and empty info.
        self.threads.append(threading.get_ident())
        return {"goal": "goal", "url": "http://example.test"}, {}

    def step(self, _action: str):
        # [FUNCTION DOC]
        # Records thread id; returns obs, reward 0.0, not terminated/truncated, empty info.
        self.threads.append(threading.get_ident())
        return {"goal": "goal", "url": "http://example.test"}, 0.0, False, False, {}

    def close(self) -> None:
        # [FUNCTION DOC]
        # Records thread id.
        self.threads.append(threading.get_ident())


def test_browsergym_calls_run_on_one_thread_whatever_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Two environments driven from a 4-worker thread pool (4 episodes) then closed
    # produce 10 recorded calls all on one thread id.
    """The HTTP API, WebSocket sessions and /web call the env from different threads."""
    threads: list[int] = []
    monkeypatch.setattr(
        browsergym_environment.importlib, "import_module", lambda _name: object()
    )
    monkeypatch.setattr(
        browsergym_environment.gym,
        "make",
        lambda *_args, **_kwargs: _ThreadRecordingGymEnv(threads),
    )

    envs = [BrowserGymEnvironment(task_name="click-test") for _ in range(2)]

    def episode(env: BrowserGymEnvironment) -> None:
        # [FUNCTION DOC]
        # Resets the env and takes one noop step.
        env.reset()
        env.step(BrowserGymAction(action_str="noop()"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(episode, envs * 2))
    for env in envs:
        env.close()

    assert len(threads) == 10
    assert len(set(threads)) == 1
