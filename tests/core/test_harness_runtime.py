# [MODULE DOC]
# Tests for the harness/session runtime layer: StepEnvSessionAdapter (wrapping a step-based env as
# a ResourceSession), SessionMCPBridge (JSON-RPC tools/list and tools/call with error mapping),
# MCPHarnessAdapter white-box runs, build_harness_rollout_func (TRL-style rollout function and the
# reward-forwarding check), and CLIHarnessAdapter black-box runs.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the harness/session runtime layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import Tool
from openenv.core.env_server.types import State
from openenv.core.harness import (
    build_harness_rollout_func,
    CLIHarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
    RESERVED_TOOL_NAMES,
    RolloutEvent,
    SessionMCPBridge,
    StepEnvSessionAdapter,
    ToolResult,
    VerifyResult,
)
from openenv.core.llm_client import LLMResponse, ToolCall


class FakeSyncEnv:
    # [CLASS DOC]
    # HELPER: Synchronous fake env: reset records kwargs and returns a "Ready" observation with the
    # task; step records the action, returns done with reward 1.0 for "finish" else reward 0.25;
    # state reports a fixed episode id and step count; close sets closed.
    """Simple sync environment used to test session adapters."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes closed flag, step counter and call logs.
        self.closed = False
        self.step_count = 0
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[str] = []

    def reset(self, **kwargs: Any) -> StepResult[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Records kwargs, zeroes the step count, returns a StepResult with text "Ready" and the task.
        self.reset_calls.append(kwargs)
        self.step_count = 0
        return StepResult(
            observation={"text": "Ready", "task": kwargs.get("task")},
            reward=0.0,
            done=False,
        )

    def step(self, action: str) -> StepResult[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Records the action, increments steps, returns "ran:<action>" with done/reward as above.
        self.step_calls.append(action)
        self.step_count += 1
        done = action == "finish"
        reward = 1.0 if done else 0.25
        return StepResult(
            observation={
                "text": f"ran:{action}",
                "done": done,
                "reward": reward,
            },
            reward=reward,
            done=done,
        )

    def state(self) -> State:
        # [FUNCTION DOC]
        # HELPER: Returns a State with id "fake-episode" and the step count.
        return State(episode_id="fake-episode", step_count=self.step_count)

    def close(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Marks closed.
        self.closed = True


@dataclass
class RecordingSessionFactory:
    # [CLASS DOC]
    # HELPER: Dataclass factory always returning the given session and recording (task, seed, episode_id).
    """Factory used to test rollout helper ordering."""

    session: Any
    created_tasks: list[Any] = field(default_factory=list)

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> Any:
        # [FUNCTION DOC]
        # HELPER: Records the call and returns the stored session.
        self.created_tasks.append((task, seed, episode_id))
        return self.session


class TestStepEnvSessionAdapter:
    # [CLASS DOC]
    # Tests StepEnvSessionAdapter.
    """Tests for the generic step-based resource session adapter."""

    def test_session_exposes_tools_and_lifecycle(self):
        # [FUNCTION DOC]
        # Initial messages come from the builder, tools are listed, call_tool steps the env and reports
        # observation, done and reward, verify reports the last reward and step count, close closes the env.
        env = FakeSyncEnv()
        tools = [
            Tool(
                name="advance",
                description="Advance the fake environment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                    },
                    "required": ["value"],
                },
            ),
        ]

        session = StepEnvSessionAdapter(
            client=env,
            task="test-task",
            tool_specs=tools,
            action_builder=lambda name, arguments: arguments["value"],
            initial_messages_builder=lambda result, task: [
                {"role": "user", "content": f"{task}:{result.observation['text']}"}
            ],
        )

        assert session.initial_messages() == [
            {"role": "user", "content": "test-task:Ready"}
        ]
        assert session.list_tools() == tools

        tool_result = session.call_tool("advance", {"value": "next"})

        assert tool_result.data["observation"]["text"] == "ran:next"
        assert tool_result.done is False
        assert tool_result.metadata["reward"] == 0.25

        verify_result = session.verify(
            transcript=[{"role": "assistant", "content": ""}]
        )
        assert verify_result.env_reward == 0.25
        assert verify_result.done is False
        assert verify_result.metrics["step_count"] == 1

        session.close()
        assert env.closed is True

    def test_session_uses_custom_verify_builder(self):
        # [FUNCTION DOC]
        # A custom verify builder receives transcript, final state, last result and state, and its result is returned.
        env = FakeSyncEnv()
        seen: dict[str, Any] = {}

        session = StepEnvSessionAdapter(
            client=env,
            task="verify-me",
            tool_specs=[],
            action_builder=lambda name, arguments: None,
            initial_messages_builder=lambda result, task: [],
            verify_builder=lambda transcript, final_state, last_result, state: (
                seen.update(
                    {
                        "transcript": transcript,
                        "final_state": final_state,
                        "last_result": last_result,
                        "state": state,
                    }
                )
                or VerifyResult(env_reward=9.0, done=True, metrics={"kind": "custom"})
            ),
        )

        result = session.verify(
            transcript=[{"role": "assistant", "content": "done"}],
            final_state={"terminal": True},
        )

        assert result.env_reward == 9.0
        assert result.done is True
        assert result.metrics == {"kind": "custom"}
        assert seen["final_state"] == {"terminal": True}

    def test_session_rejects_reserved_tool_names(self):
        # [FUNCTION DOC]
        # A tool named reset raises ValueError ("reserved").
        env = FakeSyncEnv()

        with pytest.raises(ValueError, match="reserved"):
            StepEnvSessionAdapter(
                client=env,
                task="reserved-tool",
                tool_specs=[
                    Tool(
                        name="reset",
                        description="Must remain an orchestration control",
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
                action_builder=lambda name, arguments: name,
                initial_messages_builder=lambda result, task: [],
            )

    def test_session_closes_client_when_reset_fails(self):
        # [FUNCTION DOC]
        # If the initial reset raises, the error propagates and the client is closed.
        class FailingResetEnv(FakeSyncEnv):
            # [CLASS DOC]
            # HELPER: FakeSyncEnv whose reset raises RuntimeError.
            def reset(self, **kwargs: Any) -> StepResult[dict[str, Any]]:
                # [FUNCTION DOC]
                # HELPER: Raises RuntimeError("reset failed").
                raise RuntimeError("reset failed")

        env = FailingResetEnv()

        with pytest.raises(RuntimeError, match="reset failed"):
            StepEnvSessionAdapter(
                client=env,
                task="bad-reset",
                tool_specs=[],
                action_builder=lambda name, arguments: None,
                initial_messages_builder=lambda result, task: [],
            )

        assert env.closed is True


class TestSessionMCPBridge:
    # [CLASS DOC]
    # Tests SessionMCPBridge.handle_request.
    """Tests for exposing sessions through an MCP-style JSON-RPC bridge."""

    def test_tools_list_and_tools_call(self):
        # [FUNCTION DOC]
        # tools/list returns tool specs and tools/call returns the tool result data and done flag.
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="bridge-task",
            tool_specs=[
                Tool(
                    name="finish",
                    description="Finish the fake task",
                    input_schema={
                        "type": "object",
                        "properties": {},
                    },
                )
            ],
            action_builder=lambda name, arguments: name,
            initial_messages_builder=lambda result, task: [],
        )
        bridge = SessionMCPBridge(session)

        list_response = bridge.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        assert list_response["result"]["tools"][0]["name"] == "finish"

        call_response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "finish", "arguments": {}},
            }
        )
        assert call_response["result"]["data"]["reward"] == 1.0
        assert call_response["result"]["done"] is True

    def test_unknown_method_returns_jsonrpc_error(self):
        # [FUNCTION DOC]
        # An unknown method yields a "Method not found" error with the request id.
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="bridge-task",
            tool_specs=[],
            action_builder=lambda name, arguments: None,
            initial_messages_builder=lambda result, task: [],
        )
        bridge = SessionMCPBridge(session)

        response = bridge.handle_request(
            {"jsonrpc": "2.0", "id": 7, "method": "resources/list", "params": {}}
        )

        assert response["error"]["message"] == "Method not found"
        assert response["id"] == 7

    def test_unknown_tool_returns_jsonrpc_error(self):
        # [FUNCTION DOC]
        # An unknown tool yields "Unknown tool: <name>".
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="bridge-task",
            tool_specs=[],
            action_builder=lambda name, arguments: None,
            initial_messages_builder=lambda result, task: [],
        )
        bridge = SessionMCPBridge(session)

        response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "missing", "arguments": {}},
            }
        )

        assert response["error"]["message"] == "Unknown tool: missing"
        assert response["id"] == 9

    def test_reserved_tool_returns_jsonrpc_error(self):
        # [FUNCTION DOC]
        # Each reserved tool name is rejected by the bridge without reaching the session.
        class ReservedSession:
            # [CLASS DOC]
            # HELPER: Session with no tools whose call_tool fails the test if reached.
            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns no tools.
                return []

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Raises AssertionError.
                raise AssertionError("reserved names must be blocked by the bridge")

        bridge = SessionMCPBridge(ReservedSession())

        for name in RESERVED_TOOL_NAMES:
            response = bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {}},
                }
            )

            assert (
                response["error"]["message"] == f"Reserved orchestration tool: {name}"
            )
            assert response["id"] == 12

    def test_value_error_returns_invalid_params_error(self):
        # [FUNCTION DOC]
        # A ValueError from the session becomes an error response with its message.
        class ValueErrorSession:
            # [CLASS DOC]
            # HELPER: Session whose call_tool raises ValueError.
            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns no tools.
                return []

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Raises ValueError("bad arguments").
                raise ValueError("bad arguments")

        bridge = SessionMCPBridge(ValueErrorSession())

        response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "finish", "arguments": {}},
            }
        )

        assert response["error"]["message"] == "bad arguments"
        assert response["id"] == 10

    def test_unexpected_tool_error_returns_internal_error(self):
        # [FUNCTION DOC]
        # Any other exception becomes an error response with its message.
        class FailingSession:
            # [CLASS DOC]
            # HELPER: Session whose call_tool raises RuntimeError.
            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns no tools.
                return []

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Raises RuntimeError("bridge exploded").
                raise RuntimeError("bridge exploded")

        bridge = SessionMCPBridge(FailingSession())

        response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {"name": "finish", "arguments": {}},
            }
        )

        assert response["error"]["message"] == "bridge exploded"
        assert response["id"] == 11


class TestMCPHarnessAdapter:
    # [CLASS DOC]
    # Tests MCPHarnessAdapter.run_white_box.
    """Tests for the white-box MCP-first harness adapter."""

    def test_run_white_box_accumulates_model_tokens_and_tool_trace(self):
        # [FUNCTION DOC]
        # A model step calling finish ends the run; prompt/completion ids and logprobs are accumulated,
        # the tool trace is recorded, and sampling defaults to an empty dict.
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="white-box-task",
            tool_specs=[
                Tool(
                    name="finish",
                    description="Finish the task",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            action_builder=lambda name, arguments: name,
            initial_messages_builder=lambda result, task: [
                {"role": "user", "content": "Solve the task"}
            ],
        )

        adapter = MCPHarnessAdapter()

        model_calls: list[tuple[list[dict[str, Any]], list[Tool], dict[str, Any]]] = []

        def model_step(messages, tools, sampling):
            # [FUNCTION DOC]
            # HELPER: Records the call and returns a response calling finish with fixed token ids and logprobs.
            model_calls.append((messages, tools, sampling))
            return ModelStepResult(
                response=LLMResponse(
                    content="Calling finish",
                    tool_calls=[ToolCall(id="tool-1", name="finish", args={})],
                ),
                prompt_ids=[11, 12],
                completion_ids=[21, 22],
                logprobs=[-0.1, -0.2],
            )

        result = adapter.run_white_box(
            model_step=model_step,
            session=session,
            limits=HarnessRunLimits(max_turns=2),
        )

        assert result.done is True
        assert result.prompt_ids == [11, 12]
        assert result.completion_ids == [21, 22]
        assert result.logprobs == [-0.1, -0.2]
        assert len(result.tool_trace) == 1
        assert result.tool_trace[0].tool_name == "finish"
        assert result.tool_trace[0].result.done is True
        assert model_calls[0][2] == {}

    def test_run_white_box_does_not_verify_session(self):
        # [FUNCTION DOC]
        # A white-box run never calls session.verify.
        class VerifyingSession:
            # [CLASS DOC]
            # HELPER: Session without tools that records whether verify was called.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: verified starts False.
                self.verified = False

            def initial_messages(self):
                # [FUNCTION DOC]
                # HELPER: Returns one user message.
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns no tools.
                return []

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Raises AssertionError (no tools expected).
                raise AssertionError("No tools expected")

            def verify(self, transcript, final_state=None):
                # [FUNCTION DOC]
                # HELPER: Sets verified and returns reward 1.0.
                self.verified = True
                return VerifyResult(env_reward=1.0)

            def close(self):
                # [FUNCTION DOC]
                # HELPER: No-op.
                pass

        session = VerifyingSession()
        adapter = MCPHarnessAdapter()

        adapter.run_white_box(
            model_step=lambda messages, tools, sampling: ModelStepResult(
                response=LLMResponse(content="Final answer", tool_calls=[]),
                prompt_ids=[1],
                completion_ids=[2],
                logprobs=[-0.5],
            ),
            session=session,
            limits=HarnessRunLimits(max_turns=1),
        )

        assert session.verified is False


class TestBuildHarnessRolloutFunc:
    # [CLASS DOC]
    # Tests build_harness_rollout_func.
    """Tests for the TRL rollout helper built on sessions/harnesses."""

    def test_rollout_func_verifies_after_harness_completes(self):
        # [FUNCTION DOC]
        # The rollout function runs model, tool, then verify, then close, and returns per-task lists of
        # token ids, logprobs, env reward and verify metrics; the factory receives the task.
        order: list[str] = []

        class RecordingSession:
            # [CLASS DOC]
            # HELPER: Minimal session stub for this test (single user message, scripted tools/verify).
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: closed starts False.
                self.closed = False

            def initial_messages(self):
                # [FUNCTION DOC]
                # HELPER: Returns one user message.
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns a finish tool.
                return [
                    Tool(
                        name="finish",
                        description="Finish the task",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Records "tool" and returns a done result with data reward 1.0.
                order.append("tool")
                return ToolResult(data={"reward": 1.0}, done=True)

            def verify(self, transcript, final_state=None):
                # [FUNCTION DOC]
                # HELPER: Records "verify" and returns reward 1.0 with a verified metric.
                order.append("verify")
                return VerifyResult(
                    env_reward=1.0,
                    done=True,
                    metrics={"verified": True},
                )

            def close(self):
                # [FUNCTION DOC]
                # HELPER: Records "close" and sets closed.
                order.append("close")
                self.closed = True

        session = RecordingSession()
        factory = RecordingSessionFactory(session=session)
        adapter = MCPHarnessAdapter()

        def model_step_builder(trainer, session):
            # [FUNCTION DOC]
            # HELPER: Returns a model step that records "model" and returns a finish tool call with token data.
            return lambda messages, tools, sampling: (
                order.append("model")
                or ModelStepResult(
                    response=LLMResponse(
                        content="calling tool",
                        tool_calls=[ToolCall(id="call-1", name="finish", args={})],
                    ),
                    prompt_ids=[101],
                    completion_ids=[202],
                    logprobs=[-0.25],
                )
            )

        rollout_func = build_harness_rollout_func(
            session_factory=factory,
            harness_adapter=adapter,
            model_step_builder=model_step_builder,
            limits=HarnessRunLimits(max_turns=2),
        )

        result = rollout_func(["task-a"], trainer=object())

        assert result["prompt_ids"] == [[101]]
        assert result["completion_ids"] == [[202]]
        assert result["logprobs"] == [[-0.25]]
        assert result["env_reward"] == [1.0]
        assert result["verify_metrics"] == [{"verified": True}]
        assert factory.created_tasks == [("task-a", None, None)]
        assert order == ["model", "tool", "verify", "close"]

    def test_rollout_func_rejects_verify_reward_mismatch(self):
        # [FUNCTION DOC]
        # A verify reward differing from the tool-reported reward raises ValueError.
        class RecordingSession:
            # [CLASS DOC]
            # HELPER: Minimal session stub for this test (single user message, scripted tools/verify).
            def initial_messages(self):
                # [FUNCTION DOC]
                # HELPER: Returns one user message.
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns a finish tool.
                return [
                    Tool(
                        name="finish",
                        description="Finish the task",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Returns a done result with reward 1.0 in data and metadata.
                return ToolResult(
                    data={"reward": 1.0},
                    done=True,
                    metadata={"reward": 1.0},
                )

            def verify(self, transcript, final_state=None):
                # [FUNCTION DOC]
                # HELPER: Returns a mismatching reward 0.5.
                return VerifyResult(env_reward=0.5, done=True)

            def close(self):
                # [FUNCTION DOC]
                # HELPER: No-op.
                pass

        factory = RecordingSessionFactory(session=RecordingSession())
        rollout_func = build_harness_rollout_func(
            session_factory=factory,
            harness_adapter=MCPHarnessAdapter(),
            model_step_builder=lambda trainer, session: (
                lambda messages, tools, sampling: ModelStepResult(
                    response=LLMResponse(
                        content="calling tool",
                        tool_calls=[ToolCall(id="call-1", name="finish", args={})],
                    ),
                )
            ),
            limits=HarnessRunLimits(max_turns=1),
        )

        with pytest.raises(
            ValueError,
            match="verify.env_reward must forward the environment reward",
        ):
            rollout_func(["task-a"], trainer=object())

    def test_rollout_func_accepts_float_rounding_in_reward_check(self):
        # [FUNCTION DOC]
        # A verify reward differing only by float rounding (0.1+0.2 vs 0.3) is accepted.
        class RecordingSession:
            # [CLASS DOC]
            # HELPER: Minimal session stub for this test (single user message, scripted tools/verify).
            def initial_messages(self):
                # [FUNCTION DOC]
                # HELPER: Returns one user message.
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns a finish tool.
                return [
                    Tool(
                        name="finish",
                        description="Finish the task",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Returns a done result with reward 0.3.
                return ToolResult(
                    data={"reward": 0.3},
                    done=True,
                    metadata={"reward": 0.3},
                )

            def verify(self, transcript, final_state=None):
                # [FUNCTION DOC]
                # HELPER: Returns reward 0.1+0.2.
                return VerifyResult(env_reward=0.1 + 0.2, done=True)

            def close(self):
                # [FUNCTION DOC]
                # HELPER: No-op.
                pass

        rollout_func = build_harness_rollout_func(
            session_factory=RecordingSessionFactory(session=RecordingSession()),
            harness_adapter=MCPHarnessAdapter(),
            model_step_builder=lambda trainer, session: (
                lambda messages, tools, sampling: ModelStepResult(
                    response=LLMResponse(
                        content="calling tool",
                        tool_calls=[ToolCall(id="call-1", name="finish", args={})],
                    ),
                )
            ),
            limits=HarnessRunLimits(max_turns=1),
        )

        result = rollout_func(["task-a"], trainer=object())

        assert result["env_reward"] == [0.3]

    def test_rollout_func_rejects_missing_reward(self):
        # [FUNCTION DOC]
        # A None verify reward with no tool reward raises ValueError.
        class RecordingSession:
            # [CLASS DOC]
            # HELPER: Minimal session stub for this test (single user message, scripted tools/verify).
            def initial_messages(self):
                # [FUNCTION DOC]
                # HELPER: Returns one user message.
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                # [FUNCTION DOC]
                # HELPER: Returns no tools.
                return []

            def call_tool(self, name, arguments):
                # [FUNCTION DOC]
                # HELPER: Raises AssertionError (no tools expected).
                raise AssertionError("No tools expected")

            def verify(self, transcript, final_state=None):
                # [FUNCTION DOC]
                # HELPER: Returns a None reward.
                return VerifyResult(env_reward=None, done=True)

            def close(self):
                # [FUNCTION DOC]
                # HELPER: No-op.
                pass

        rollout_func = build_harness_rollout_func(
            session_factory=RecordingSessionFactory(session=RecordingSession()),
            harness_adapter=MCPHarnessAdapter(),
            model_step_builder=lambda trainer, session: (
                lambda messages, tools, sampling: ModelStepResult(
                    response=LLMResponse(content="done", tool_calls=[]),
                )
            ),
            limits=HarnessRunLimits(max_turns=1),
        )

        with pytest.raises(ValueError, match="did not produce an environment reward"):
            rollout_func(["task-a"], trainer=object())


class TestCLIHarnessAdapter:
    # [CLASS DOC]
    # Tests CLIHarnessAdapter.
    """Tests for black-box CLI-style evaluation harnesses."""

    def test_black_box_runner_can_use_session_mcp_bridge(self):
        # [FUNCTION DOC]
        # The runner receives a SessionMCPBridge it can use to list and call tools; its result is returned.
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="cli-task",
            tool_specs=[
                Tool(
                    name="finish",
                    description="Finish the task",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            action_builder=lambda name, arguments: name,
            initial_messages_builder=lambda result, task: [],
        )

        def runner(bridge, current_session, limits):
            # [FUNCTION DOC]
            # HELPER: Lists tools and calls finish through the bridge, returning a done rollout with one event
            # holding the responses.
            list_response = bridge.handle_request(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            )
            call_response = bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "finish", "arguments": {}},
                }
            )
            return HarnessRolloutResult(
                messages=[{"role": "assistant", "content": "done"}],
                tool_trace=[],
                events=[
                    RolloutEvent(
                        type="cli_runner",
                        payload={
                            "tools": list_response["result"]["tools"],
                            "result": call_response["result"],
                        },
                    )
                ],
                done=True,
                metrics={"mode": "black_box"},
            )

        adapter = CLIHarnessAdapter(runner=runner)
        result = adapter.run_black_box(session=session, limits=HarnessRunLimits())

        assert result.done is True
        assert result.metrics["mode"] == "black_box"
        assert result.events[0].payload["result"]["done"] is True
