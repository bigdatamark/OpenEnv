# [MODULE DOC]
# Tests for BrowserGym's harness adapter: BrowserGymSessionFactory sessions
# (tools, initial messages, rewards, verify), the action-string <-> tool-call
# builders/parsers with quoting, and use with build_harness_rollout_func.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BrowserGym's harness-oriented session adapter."""

from __future__ import annotations

from typing import Any

import pytest
from browsergym_env import BrowserGymAction, BrowserGymObservation, BrowserGymState
from browsergym_env.harness import (
    BrowserGymSessionFactory,
    build_browsergym_action_str,
    build_browsergym_action_tool_call,
)
from openenv.core.client_types import StepResult
from openenv.core.harness import (
    build_harness_rollout_func,
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
    ResourceSessionFactory,
)
from openenv.core.llm_client import LLMResponse


class FakeBrowserGymClient:
    # [CLASS DOC]
    # In-memory BrowserGym-like client: a single page with button 13; clicking
    # "13" finishes the episode with reward 1.0. Records action strings.
    """Small BrowserGym-like client used for harness tests."""

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes closed flag, step count, cumulative reward and action log.
        self.closed = False
        self._step_count = 0
        self._cum_reward = 0.0
        self.step_actions: list[str] = []

    def reset(self, **kwargs: Any) -> StepResult[BrowserGymObservation]:
        # [FUNCTION DOC]
        # Resets counters and returns a StepResult with the goal/axtree observation.
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
        # Appends the action string, increments steps; done and reward 1.0 only for
        # click("13"); accumulates reward; returns a StepResult.
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
        # Returns a BrowserGymState with current step count and cumulative reward.
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
        # Sets closed to True.
        self.closed = True


def test_browsergym_session_factory_exposes_expected_tools():
    # [FUNCTION DOC]
    # Factory is a ResourceSessionFactory; sessions list click/fill/send_keys/
    # scroll/noop and one initial message with goal and axtree.
    factory = BrowserGymSessionFactory(client_factory=FakeBrowserGymClient)
    assert isinstance(factory, ResourceSessionFactory)

    session = factory.create(task="ignored-task")

    tool_names = [tool.name for tool in session.list_tools()]
    assert tool_names == ["click", "fill", "send_keys", "scroll", "noop"]

    messages = session.initial_messages()
    assert len(messages) == 1
    assert "Click the highlighted button" in messages[0]["content"]
    assert "[13] button 'Continue'" in messages[0]["content"]

    session.close()


def test_browsergym_tool_calls_match_plain_env_rewards():
    # [FUNCTION DOC]
    # Calling the click tool via a session gives the same done/reward (in
    # metadata and verify result) as stepping the plain client.
    plain_client = FakeBrowserGymClient()
    plain_client.reset()
    plain_result = plain_client.step(BrowserGymAction(action_str='click("13")'))

    factory = BrowserGymSessionFactory(client_factory=FakeBrowserGymClient)
    session = factory.create(task="ignored-task")
    tool_result = session.call_tool("click", {"bid": "13"})
    verify_result = session.verify(transcript=[{"role": "assistant", "content": ""}])

    assert tool_result.done is True
    assert tool_result.metadata["reward"] == plain_result.reward
    assert verify_result.env_reward == plain_result.reward
    assert verify_result.done == plain_result.done
    session.close()


def test_browsergym_action_text_parser():
    # [FUNCTION DOC]
    # Parses click/fill/scroll/noop strings into tool names and argument mappings.
    click = build_browsergym_action_tool_call("click('13')")
    fill = build_browsergym_action_tool_call("fill('42', 'hello')")
    scroll = build_browsergym_action_tool_call("scroll('down')")
    noop = build_browsergym_action_tool_call("noop()")

    assert click.name == "click"
    assert click.args == {"bid": "13"}
    assert fill.name == "fill"
    assert fill.args == {"bid": "42", "text": "hello"}
    assert scroll.name == "scroll"
    assert scroll.args == {"direction": "down"}
    assert noop.name == "noop"
    assert noop.args == {}


def test_browsergym_fill_parser_supports_embedded_quote():
    # [FUNCTION DOC]
    # Double-quoted text may contain a single quote.
    fill = build_browsergym_action_tool_call('fill("42", "O\'Brien")')

    assert fill.name == "fill"
    assert fill.args == {"bid": "42", "text": "O'Brien"}


def test_browsergym_fill_parser_supports_single_quoted_embedded_quote():
    # [FUNCTION DOC]
    # Escaped single quote inside single-quoted text is parsed.
    fill = build_browsergym_action_tool_call("fill('42', 'O\\'Brien')")

    assert fill.name == "fill"
    assert fill.args == {"bid": "42", "text": "O'Brien"}


def test_browsergym_action_builder_quotes_round_trip_through_parser():
    # [FUNCTION DOC]
    # build_browsergym_action_str escapes quotes/newlines and parses back exactly.
    action = build_browsergym_action_str(
        "fill",
        {"bid": 'button"42', "text": "O'Brien\nNext"},
    )
    tool_call = build_browsergym_action_tool_call(action)

    assert action == 'fill("button\\"42", "O\'Brien\\nNext")'
    assert tool_call.name == "fill"
    assert tool_call.args == {"bid": 'button"42', "text": "O'Brien\nNext"}


def test_browsergym_action_parser_rejects_malformed_strings():
    # [FUNCTION DOC]
    # Unbalanced quoting raises ValueError "Unsupported BrowserGym action".
    with pytest.raises(ValueError, match="Unsupported BrowserGym action"):
        build_browsergym_action_tool_call("fill('42', 'O'Brien')")


def test_browsergym_session_factory_works_with_generic_rollout_helper():
    # [FUNCTION DOC]
    # A rollout with a model step that always clicks 13 returns the token ids,
    # logprobs and env_reward 1.0.
    factory = BrowserGymSessionFactory(client_factory=FakeBrowserGymClient)
    adapter = MCPHarnessAdapter()

    def model_step_builder(trainer, session):
        # [FUNCTION DOC]
        # Returns a model-step callable producing a fixed click tool call with fixed
        # prompt/completion ids and logprobs.
        tool_call = build_browsergym_action_tool_call("click('13')")
        return lambda messages, tools, sampling: ModelStepResult(
            response=LLMResponse(content="click", tool_calls=[tool_call]),
            prompt_ids=[3, 4],
            completion_ids=[5, 6],
            logprobs=[-0.3, -0.4],
        )

    rollout_func = build_harness_rollout_func(
        session_factory=factory,
        harness_adapter=adapter,
        model_step_builder=model_step_builder,
        limits=HarnessRunLimits(max_turns=3),
    )

    result = rollout_func(["browsergym prompt"], trainer=object())

    assert result["prompt_ids"] == [[3, 4]]
    assert result["completion_ids"] == [[5, 6]]
    assert result["logprobs"] == [[-0.3, -0.4]]
    assert result["env_reward"] == [1.0]
