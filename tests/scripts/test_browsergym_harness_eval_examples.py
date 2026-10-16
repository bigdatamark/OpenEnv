# [MODULE DOC]
# Offline smoke tests for examples/browsergym_harness_eval_common: action extraction,
# the session MCP HTTP bridge, the white-box episode loop, episode finalisation and
# OpenAI client creation. Adds src/, envs/ and examples/ to sys.path at import time.
"""Smoke tests for the BrowserGym harness evaluation examples."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import requests

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "envs"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))

from browsergym_env import BrowserGymAction, BrowserGymObservation, BrowserGymState
from browsergym_env.harness import BrowserGymSessionFactory
from browsergym_harness_eval_common import (
    _finalize_episode,
    build_openai_model_step,
    create_openai_client,
    extract_browsergym_action,
    run_white_box_episode,
    SessionMCPHttpServer,
    summarize_episodes,
)
from openenv.core.client_types import StepResult
from openenv.core.harness import (
    HarnessRolloutResult,
    HarnessRunLimits,
    SessionMCPBridge,
    VerifyResult,
)


class FakeBrowserGymClient:
    # [CLASS DOC]
    # HELPER: In-memory BrowserGym client. Episode ends with reward 1.0 when the action is
    # exactly click("13"); records action strings and close state.
    """Small BrowserGym-like client used for example tests."""

    def __init__(self):
        # [FUNCTION DOC]
        # Initialises closed flag, step count, cumulative reward and recorded actions.
        self.closed = False
        self._step_count = 0
        self._cum_reward = 0.0
        self.step_actions: list[str] = []

    def reset(self, **kwargs: Any) -> StepResult[BrowserGymObservation]:
        # [FUNCTION DOC]
        # Ignores kwargs, resets counters. Returns: StepResult with the fixed goal/axtree, reward 0.
        del kwargs
        self._step_count = 0
        self._cum_reward = 0.0
        return StepResult(
            observation=BrowserGymObservation(
                goal="Click the highlighted button",
                axtree_txt="[13] button 'Continue'",
                text="[13] button 'Continue'",
                url="http://example.test",
                done=False,
                reward=0.0,
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: BrowserGymAction) -> StepResult[BrowserGymObservation]:
        # [FUNCTION DOC]
        # Records the action, increments step count, done/reward 1.0 iff action is click("13"),
        # accumulates reward. Returns: StepResult with the after-URL observation.
        self.step_actions.append(action.action_str)
        self._step_count += 1
        done = action.action_str == 'click("13")'
        reward = 1.0 if done else 0.0
        self._cum_reward += reward
        return StepResult(
            observation=BrowserGymObservation(
                goal="Click the highlighted button",
                axtree_txt="[13] button 'Continue'",
                text="[13] button 'Continue'",
                url="http://example.test/after",
                done=done,
                reward=reward,
                last_action_error=False,
                error="",
            ),
            reward=reward,
            done=done,
        )

    def state(self) -> BrowserGymState:
        # [FUNCTION DOC]
        # Returns: BrowserGymState with current step count and cumulative reward.
        return BrowserGymState(
            episode_id="browsergym-episode",
            step_count=self._step_count,
            benchmark="miniwob",
            task_name="click-test",
            goal="Click the highlighted button",
            current_url="http://example.test/after",
            cum_reward=self._cum_reward,
        )

    def close(self) -> None:
        # [FUNCTION DOC]
        # Sets closed True.
        self.closed = True


class FakeOpenAIClient:
    # [CLASS DOC]
    # HELPER: Fake OpenAI client exposing chat.completions.create that pops canned outputs.
    """Minimal fake OpenAI client for build_openai_model_step tests."""

    def __init__(self, outputs: list[str]):
        # [FUNCTION DOC]
        # Stores a copy of outputs, a call log, and chat.completions pointing to itself.
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any):
        # [FUNCTION DOC]
        # Logs kwargs and returns a response object whose first choice holds the next output.
        # Raises IndexError when outputs run out.
        self.calls.append(kwargs)
        content = self._outputs.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_extract_browsergym_action_tolerates_common_outputs():
    # [FUNCTION DOC]
    # Bare, "Action:"-prefixed and fenced actions are extracted; unparseable text gives noop().
    assert extract_browsergym_action("click('13')") == "click('13')"
    assert extract_browsergym_action("Action: click('13')") == "click('13')"
    assert extract_browsergym_action("```text\nclick('13')\n```") == "click('13')"
    assert extract_browsergym_action("I am not sure what to do.") == "noop()"


def test_session_http_bridge_lists_tools_and_calls_tool():
    # [FUNCTION DOC]
    # Serves a BrowserGym session over SessionMCPHttpServer and checks JSON-RPC tools/list
    # returns click, tools/call click(13) finishes with reward 1.0, and an unknown tool yields
    # an "Unknown tool" error. Side effects: local HTTP server; session closed afterwards.
    factory = BrowserGymSessionFactory(client_factory=FakeBrowserGymClient)
    session = factory.create(task=None)
    bridge = SessionMCPBridge(session)

    try:
        with SessionMCPHttpServer(bridge).start() as server:
            list_response = requests.post(
                server.url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                timeout=5,
            ).json()
            call_response = requests.post(
                server.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "click", "arguments": {"bid": "13"}},
                },
                timeout=5,
            ).json()
            missing_response = requests.post(
                server.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "missing", "arguments": {}},
                },
                timeout=5,
            ).json()
    finally:
        session.close()

    assert list_response["result"]["tools"][0]["name"] == "click"
    assert call_response["result"]["structuredContent"]["done"] is True
    assert call_response["result"]["data"]["reward"] == 1.0
    assert missing_response["error"]["message"] == "Unknown tool: missing"


def test_white_box_example_loop_runs_one_episode_and_aggregates_metrics():
    # [FUNCTION DOC]
    # Runs one white-box episode with a fake model answering click('13'); checks reward,
    # success, done, 1 step, summary averages, and that the first prompt message is system.
    fake_client = FakeOpenAIClient(["Action: click('13')"])
    session_factory = BrowserGymSessionFactory(
        client_factory=FakeBrowserGymClient,
        default_task="click-test",
    )
    model_step = build_openai_model_step(fake_client, model="fake-model")

    episode = run_white_box_episode(
        session_factory=session_factory,
        model_step=model_step,
        limits=HarnessRunLimits(max_turns=3),
        episode_id="example-episode",
    )
    summary = summarize_episodes([episode])

    assert episode.reward == 1.0
    assert episode.success is True
    assert episode.done is True
    assert episode.step_count == 1
    assert summary["avg_reward"] == 1.0
    assert summary["success_rate"] == 1.0
    assert summary["avg_steps"] == 1.0
    assert fake_client.calls[0]["messages"][0]["role"] == "system"


def test_finalize_episode_rejects_missing_env_reward():
    # [FUNCTION DOC]
    # _finalize_episode raises ValueError when verify() returns no env reward.
    class MissingRewardSession:
        # [CLASS DOC]
        # HELPER: Session stub whose verify returns a done result without env reward.
        def verify(self, transcript, final_state=None):
            # [FUNCTION DOC]
            # Returns VerifyResult(env_reward None, done True).
            return VerifyResult(env_reward=None, done=True)

    with pytest.raises(ValueError, match="did not produce an environment reward"):
        _finalize_episode(
            MissingRewardSession(),
            HarnessRolloutResult(
                messages=[{"role": "assistant", "content": "done"}],
                done=True,
            ),
        )


def test_create_openai_client_does_not_fallback_to_hf_token(monkeypatch):
    # [FUNCTION DOC]
    # With only HF_TOKEN set, create_openai_client must not pass api_key to OpenAI
    # (OpenAI patched to capture kwargs).
    import browsergym_harness_eval_common as common

    captured_kwargs: dict[str, Any] = {}

    def fake_openai(**kwargs):
        # [FUNCTION DOC]
        # HELPER: Captures kwargs and returns a dummy object.
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setattr(common, "OpenAI", fake_openai)

    client = create_openai_client()

    assert client is not None
    assert "api_key" not in captured_kwargs
