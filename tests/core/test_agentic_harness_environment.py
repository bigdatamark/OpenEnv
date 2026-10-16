# [MODULE DOC]
# Unit tests for HarnessEnvironment, the turn-based environment wrapping an agentic harness adapter.
# Uses a recording FakeAdapter and an autouse fixture replacing HarnessMCPBridge with FakeBridge.
# Covers reset ordering and conflict resolution, steps, rubrics, transforms, sync facade, close,
# cleanup on failure/cancellation, error classification and mode-specific tool exclusion.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for HarnessEnvironment (RFC 005 turn-based wrapper)."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

import pytest
from fastmcp import FastMCP
from openenv.core.env_server._utils import overrides_method
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import ListToolsAction, Tool
from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import Action
from openenv.core.harness import (
    AgenticHarnessAdapter,
    HarnessAction,
    HarnessConfig,
    HarnessEnvironment,
    HarnessError,
    HarnessEvent,
    HarnessEventType,
    HarnessNotRunningError,
    HarnessTurnTimeoutError,
)
from openenv.core.rubrics import Rubric


def turn_events(response: str, done: bool = False) -> list[HarnessEvent]:
    # [FUNCTION DOC]
    # HELPER: Returns the events for one turn: a TOOL_CALL for "shell" followed by TURN_COMPLETE
    # carrying the given response and done flag.
    return [
        HarnessEvent(type=HarnessEventType.TOOL_CALL, data={"tool_name": "shell"}),
        HarnessEvent(
            type=HarnessEventType.TURN_COMPLETE,
            data={"response": response, "done": done},
        ),
    ]


class FakeAdapter(AgenticHarnessAdapter):
    # [CLASS DOC]
    # HELPER: Recording fake adapter with built-in tool read_file. Records method names in calls,
    # stores injected tools and bridge URL, and has knobs: fail_on_start, fail_on_send,
    # raise_turn_timeout, send_delay_s, and a queue of scripted_turns.
    """Recording fake with failure knobs."""

    BUILTIN_TOOL_NAMES = frozenset({"read_file"})

    def __init__(self, config: Optional[HarnessConfig] = None):
        # [FUNCTION DOC]
        # HELPER: Initializes with the given or a dummy config and default knob values.
        super().__init__(config or HarnessConfig(name="fake", command=["fake"]))
        self.calls: list[str] = []
        self.injected_tools: Optional[list[Tool]] = None
        self.injected_bridge_url: Optional[str] = "unset"
        self.alive = False
        self.fail_on_start = False
        self.fail_on_send = False
        self.raise_turn_timeout = False
        self.send_delay_s = 0.0
        self.scripted_turns: list[list[HarnessEvent]] = []

    async def start(self, working_directory: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Records "start"; raises HarnessError if fail_on_start, else marks alive.
        self.calls.append("start")
        if self.fail_on_start:
            raise HarnessError("boom on start")
        self.alive = True

    async def stop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Records "stop" and marks not alive.
        self.calls.append("stop")
        self.alive = False

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Records "inject_tools" and stores the tools list and bridge URL.
        self.calls.append("inject_tools")
        self.injected_tools = list(tools)
        self.injected_bridge_url = bridge_url

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        # [FUNCTION DOC]
        # HELPER: Records "send", optionally sleeps send_delay_s, raises HarnessTurnTimeoutError or
        # HarnessError per knobs, then yields the next scripted turn's events (or a default turn).
        self.calls.append("send")
        if self.send_delay_s:
            await asyncio.sleep(self.send_delay_s)
        if self.raise_turn_timeout:
            raise HarnessTurnTimeoutError("adapter timed the turn out itself")
        if self.fail_on_send:
            raise HarnessError("harness exploded mid-turn")
        events = (
            self.scripted_turns.pop(0)
            if self.scripted_turns
            else turn_events("default response")
        )
        for event in events:
            yield event

    async def is_alive(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns the alive flag.
        return self.alive


class FakeBridge:
    # [CLASS DOC]
    # HELPER: In-process stand-in for HarnessMCPBridge; counts start/stop calls, can fail on stop,
    # and records every instance in the class-level instances list.
    """In-process stand-in for HarnessMCPBridge (real one tested separately)."""

    instances: list["FakeBridge"] = []

    def __init__(self, mcp_server):
        # [FUNCTION DOC]
        # HELPER: Stores the MCP server, zeroes counters and registers the instance.
        self.mcp_server = mcp_server
        self.started = 0
        self.stopped = 0
        self.fail_on_stop = False
        FakeBridge.instances.append(self)

    def start(self, timeout_s: float = 10.0) -> str:
        # [FUNCTION DOC]
        # HELPER: Increments started and returns a fixed loopback URL.
        self.started += 1
        return "http://127.0.0.1:9/mcp"

    def stop(self, timeout_s: float = 5.0) -> None:
        # [FUNCTION DOC]
        # HELPER: Increments stopped; raises HarnessError if fail_on_stop is set.
        self.stopped += 1
        if self.fail_on_stop:
            raise HarnessError("bridge shutdown timed out")


@pytest.fixture(autouse=True)
def fake_bridge(monkeypatch):
    # [FUNCTION DOC]
    # Autouse fixture clearing FakeBridge.instances and patching the environment module's
    # HarnessMCPBridge with FakeBridge; returns FakeBridge.
    FakeBridge.instances = []
    monkeypatch.setattr("openenv.core.harness.environment.HarnessMCPBridge", FakeBridge)
    return FakeBridge


class SpyRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric returning 0.75, recording (action, observation) pairs and counting resets.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes the record list and reset counter.
        super().__init__()
        self.seen: list[tuple] = []
        self.resets = 0

    def forward(self, action, observation) -> float:
        # [FUNCTION DOC]
        # HELPER: Records the pair and returns 0.75.
        self.seen.append((action, observation))
        return 0.75

    def reset(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Increments the reset counter.
        self.resets += 1


def make_env(**kwargs) -> tuple[HarnessEnvironment, FakeAdapter]:
    # [FUNCTION DOC]
    # HELPER: Builds a HarnessEnvironment with a FakeAdapter. Unless an mcp keyword is given, uses a
    # FastMCP with query_db and read_file tools. Extra keywords go to the environment constructor.
    # Returns (env, adapter).
    adapter = FakeAdapter()
    mcp = kwargs.pop("mcp", "default")
    if mcp == "default":
        mcp = FastMCP("domain")

        @mcp.tool
        def query_db(sql: str) -> str:
            # [FUNCTION DOC]
            # HELPER: Tool returning the SQL string.
            """Run a SQL query."""
            return sql

        @mcp.tool
        def read_file(path: str) -> str:
            # [FUNCTION DOC]
            # HELPER: Tool returning the path.
            """Read a task file."""
            return path

    env = HarnessEnvironment(adapter=adapter, mcp=mcp, **kwargs)
    return env, adapter


class TestReset:
    # [CLASS DOC]
    # Tests reset behavior.
    async def test_inject_before_start_with_conflict_resolution(self):
        # [FUNCTION DOC]
        # Reset calls stop, inject_tools, start in order; read_file is renamed env_read_file; the
        # observation lists injected tools; state is fresh; a single bridge was started and its URL injected.
        env, adapter = make_env()
        obs = await env.reset_async()

        assert adapter.calls == ["stop", "inject_tools", "start"]
        names = sorted(t.name for t in adapter.injected_tools)
        assert names == ["env_read_file", "query_db"]
        assert sorted(obs.metadata["injected_tools"]) == names
        assert obs.done is False
        assert env.state.step_count == 0
        assert env.state.episode_id
        assert adapter.injected_bridge_url == "http://127.0.0.1:9/mcp"
        assert len(FakeBridge.instances) == 1
        assert FakeBridge.instances[0].started == 1

    async def test_reset_stops_live_adapter_first(self):
        # [FUNCTION DOC]
        # A second reset again calls stop, inject_tools, start.
        env, adapter = make_env()
        await env.reset_async()
        adapter.calls.clear()
        await env.reset_async()
        assert adapter.calls == ["stop", "inject_tools", "start"]

    async def test_reset_does_not_replace_bridge_after_shutdown_failure(self):
        # [FUNCTION DOC]
        # If the old bridge fails to stop, reset raises, keeps the same bridge, does not inject/start, and
        # steps fail; once stop works a reset creates a new bridge.
        env, adapter = make_env()
        await env.reset_async()
        bridge = env._bridge
        bridge.fail_on_stop = True
        stop_count = bridge.stopped
        adapter.calls.clear()

        with pytest.raises(HarnessError, match="bridge shutdown timed out"):
            await env.reset_async()

        assert env._bridge is bridge
        assert bridge.stopped == stop_count + 1
        assert FakeBridge.instances == [bridge]
        assert "inject_tools" not in adapter.calls
        assert "start" not in adapter.calls
        assert adapter.alive is False
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="go"))

        bridge.fail_on_stop = False
        await env.reset_async()
        assert env._bridge is not bridge
        assert adapter.alive is True
        env.close()

    async def test_cancelled_bridge_stop_still_cleans_up(self, monkeypatch):
        # [FUNCTION DOC]
        # If bridge stop raises CancelledError once, reset propagates it, retries stop, keeps the bridge and
        # leaves the adapter stopped.
        env, adapter = make_env()
        await env.reset_async()
        bridge = env._bridge
        stop = bridge.stop

        def cancel_once():
            # [FUNCTION DOC]
            # HELPER: Calls the real stop, then raises CancelledError on the first call.
            stop()
            if bridge.stopped == 1:
                raise asyncio.CancelledError()

        monkeypatch.setattr(bridge, "stop", cancel_once)
        with pytest.raises(asyncio.CancelledError):
            await env.reset_async()

        assert bridge.stopped == 2
        assert env._bridge is bridge
        assert FakeBridge.instances == [bridge]
        assert adapter.alive is False
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="go"))
        env.close()

    async def test_reset_clears_trajectory_and_resets_rubric(self):
        # [FUNCTION DOC]
        # Reset with a new episode id clears the trajectory, sets the id, zeroes steps and resets the rubric.
        rubric = SpyRubric()
        env, adapter = make_env(rubric=rubric)
        await env.reset_async()
        await env.step_async(HarnessAction(message="turn 1"))
        assert env.trajectory
        await env.reset_async(episode_id="ep-2")
        assert env.trajectory == []
        assert env.state.episode_id == "ep-2"
        assert env.state.step_count == 0
        assert rubric.resets == 2

    async def test_no_mcp_tools_injects_empty_list_and_no_bridge(self):
        # [FUNCTION DOC]
        # Without an MCP server, an empty tool list and None bridge URL are injected and no bridge is created.
        env, adapter = make_env(mcp=None)
        await env.reset_async()
        assert adapter.injected_tools == []
        assert adapter.injected_bridge_url is None
        assert FakeBridge.instances == []

    @pytest.mark.parametrize("async_rubric", [False, True])
    async def test_rubric_reset_failure_stops_resources_and_allows_retry(
        self, async_rubric
    ):
        # [FUNCTION DOC]
        # Parametrized sync/async rubric reset: a failing rubric reset propagates the same exception and
        # stops the adapter and bridge; steps then fail; a retry reset after fixing it succeeds.
        failure = RuntimeError("rubric reset failed")

        class FailingRubric(SpyRubric):
            # [CLASS DOC]
            # HELPER: SpyRubric whose reset asserts the adapter and bridge are running, then raises while fail is set.
            fail = True

            def reset(self):
                # [FUNCTION DOC]
                # HELPER: Asserts resources are live; raises the prepared failure if fail, else counts the reset.
                assert adapter.alive
                assert FakeBridge.instances[-1].started == 1
                if self.fail:
                    raise failure
                super().reset()

        class AsyncFailingRubric(FailingRubric):
            # [CLASS DOC]
            # HELPER: Variant exposing reset_async that delegates to reset.
            async def reset_async(self):
                # [FUNCTION DOC]
                # HELPER: Calls reset.
                self.reset()

        rubric = AsyncFailingRubric() if async_rubric else FailingRubric()
        env, adapter = make_env(rubric=rubric)
        with pytest.raises(RuntimeError) as caught:
            await env.reset_async()

        assert caught.value is failure
        assert adapter.alive is False
        assert adapter.calls[-1] == "stop"
        assert FakeBridge.instances[0].stopped >= 1
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="after failed reset"))

        rubric.fail = False
        try:
            obs = await env.reset_async()
            assert obs.done is False
            assert adapter.alive
            assert rubric.resets == 1
            assert len(FakeBridge.instances) == 2
            obs = await env.step_async(HarnessAction(message="retry"))
            assert obs.metadata["response"] == "default response"
        finally:
            env.close()

    async def test_start_failure_leaves_episode_inactive(self):
        # [FUNCTION DOC]
        # A failing adapter start raises, stops the bridge, and leaves steps failing with HarnessNotRunningError.
        env, adapter = make_env()
        adapter.fail_on_start = True
        with pytest.raises(HarnessError):
            await env.reset_async()
        assert FakeBridge.instances[0].stopped >= 1
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="hi"))


class TestStep:
    # [CLASS DOC]
    # Tests step behavior.
    async def test_turn_updates_state_and_trajectory(self):
        # [FUNCTION DOC]
        # Steps return the scripted responses, turn numbers and done flags, update step_count and accumulate
        # trajectory events.
        env, adapter = make_env()
        adapter.scripted_turns = [turn_events("first"), turn_events("second", True)]
        await env.reset_async()

        obs1 = await env.step_async(HarnessAction(message="fix the bug"))
        assert obs1.done is False
        assert obs1.metadata["response"] == "first"
        assert obs1.metadata["turn_number"] == 1
        assert env.state.step_count == 1

        obs2 = await env.step_async(HarnessAction(message="tests still fail"))
        assert obs2.done is True
        assert obs2.metadata["response"] == "second"
        assert env.state.step_count == 2
        assert len(env.trajectory) == 4

    async def test_observation_serializes_to_json(self):
        # [FUNCTION DOC]
        # The step observation serializes to JSON including turn_events.
        env, adapter = make_env()
        await env.reset_async()
        obs = await env.step_async(HarnessAction(message="go"))
        payload = serialize_observation(obs)
        assert json.loads(json.dumps(payload))["metadata"]["turn_events"]

    async def test_rubric_applied_after_turn(self):
        # [FUNCTION DOC]
        # The rubric receives the same action and the turn observation; its value becomes the reward.
        rubric = SpyRubric()
        env, adapter = make_env(rubric=rubric)
        await env.reset_async()
        action = HarnessAction(message="go")
        obs = await env.step_async(action)
        assert obs.reward == 0.75
        seen_action, seen_obs = rubric.seen[0]
        assert seen_action is action
        assert seen_obs.metadata["response"] == "default response"

    async def test_step_before_reset_raises(self):
        # [FUNCTION DOC]
        # Stepping before reset raises HarnessNotRunningError.
        env, _ = make_env()
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="hi"))

    async def test_dead_adapter_returns_terminal_error_observation(self):
        # [FUNCTION DOC]
        # A dead adapter yields a done observation with error_type harness_crashed.
        env, adapter = make_env()
        await env.reset_async()
        adapter.alive = False
        obs = await env.step_async(HarnessAction(message="hi"))
        assert obs.done is True
        assert obs.metadata["error_type"] == "harness_crashed"

    async def test_mid_turn_crash_stops_adapter(self):
        # [FUNCTION DOC]
        # A HarnessError mid-turn yields harness_crashed, stops the adapter and ends the episode.
        env, adapter = make_env()
        await env.reset_async()
        adapter.fail_on_send = True
        obs = await env.step_async(HarnessAction(message="hi"))
        assert obs.done is True
        assert obs.metadata["error_type"] == "harness_crashed"
        assert adapter.calls[-1] == "stop"
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="again"))

    async def test_turn_timeout_returns_terminal_error_observation(self):
        # [FUNCTION DOC]
        # A turn exceeding session_timeout_s yields a done turn_timeout observation and stops the adapter.
        env, adapter = make_env()
        adapter.config.session_timeout_s = 0.05
        adapter.send_delay_s = 5.0
        await env.reset_async()
        obs = await env.step_async(HarnessAction(message="hi"))
        assert obs.done is True
        assert obs.metadata["error_type"] == "turn_timeout"
        assert adapter.calls[-1] == "stop"

    async def test_unknown_action_type_raises(self):
        # [FUNCTION DOC]
        # A plain Action raises TypeError.
        env, _ = make_env()
        await env.reset_async()
        with pytest.raises(TypeError):
            await env.step_async(Action())

    async def test_list_tools_routing_preserved(self):
        # [FUNCTION DOC]
        # ListToolsAction lists the original source tool names even before reset.
        env, _ = make_env()
        obs = await env.step_async(ListToolsAction())
        assert sorted(t.name for t in obs.tools) == ["query_db", "read_file"]


class TestSyncFacadeAndClose:
    # [CLASS DOC]
    # Tests sync wrappers, close, construction and server async-path selection.
    def test_sync_reset_and_step(self):
        # [FUNCTION DOC]
        # Sync reset and step work and update state.
        env, adapter = make_env()
        adapter.scripted_turns = [turn_events("sync response", True)]
        obs = env.reset()
        assert obs.done is False
        obs = env.step(HarnessAction(message="go"))
        assert obs.metadata["response"] == "sync response"
        assert obs.done is True
        assert env.state.step_count == 1

    def test_close_is_idempotent_and_stops_adapter_and_bridge(self):
        # [FUNCTION DOC]
        # Close stops the adapter and bridge; a second close does nothing.
        env, adapter = make_env()
        env.reset()
        env.close()
        assert "stop" in adapter.calls
        assert FakeBridge.instances[0].stopped >= 1
        stop_count = adapter.calls.count("stop")
        env.close()
        assert adapter.calls.count("stop") == stop_count

    def test_close_retries_bridge_shutdown_after_failure(self):
        # [FUNCTION DOC]
        # After a failed bridge stop, close keeps the bridge and retries next time; once stopped further
        # closes do not stop again.
        env, _ = make_env()
        env.reset()
        bridge = env._bridge
        bridge.fail_on_stop = True

        env.close()

        assert env._bridge is bridge
        stop_count = bridge.stopped
        bridge.fail_on_stop = False
        env.close()
        assert bridge.stopped == stop_count + 1
        assert env._bridge is None
        env.close()
        assert bridge.stopped == stop_count + 1

    def test_constructor_starts_nothing(self):
        # [FUNCTION DOC]
        # Construction calls no adapter methods.
        env, adapter = make_env()
        assert adapter.calls == []

    def test_server_selects_async_paths(self):
        # [FUNCTION DOC]
        # reset_async and step_async are overridden and concurrent sessions are unsupported.
        env, _ = make_env()
        assert overrides_method(env.reset_async, Environment.reset_async)
        assert overrides_method(env.step_async, Environment.step_async)
        assert env.SUPPORTS_CONCURRENT_SESSIONS is False


class TestTransforms:
    # [CLASS DOC]
    # Tests observation transforms.
    @pytest.mark.parametrize("sync", [False, True])
    async def test_reset_and_turn_apply_transform_once_after_rubric(self, sync):
        # [FUNCTION DOC]
        # Parametrized sync/async: the transform runs once per reset and turn, after the rubric, and the
        # rubric sees the untransformed observation.
        seen = []

        def transform(observation):
            # [FUNCTION DOC]
            # HELPER: Records the observation and returns a copy with reward +1 and metadata transformed=True.
            seen.append(observation)
            return observation.model_copy(
                update={
                    "reward": observation.reward + 1.0,
                    "metadata": {**observation.metadata, "transformed": True},
                }
            )

        rubric = SpyRubric()
        env, adapter = make_env(rubric=rubric, transform=transform)
        adapter.scripted_turns = [turn_events("first"), turn_events("last", True)]
        try:
            reset_obs = env.reset() if sync else await env.reset_async()
            assert reset_obs.metadata["transformed"] is True
            assert reset_obs.reward == 1.0
            assert len(seen) == 1

            for message, done in [("first", False), ("last", True)]:
                action = HarnessAction(message=message)
                obs = env.step(action) if sync else await env.step_async(action)
                assert obs.metadata["transformed"] is True
                assert obs.metadata["response"] == message
                assert obs.done is done
                assert obs.reward == 1.75
                assert rubric.seen[-1][1] is seen[-1]
                assert "transformed" not in rubric.seen[-1][1].metadata
            assert len(seen) == 3
        finally:
            env.close()

    @pytest.mark.parametrize(
        "failure, value, error_type",
        [
            ("alive", False, "harness_crashed"),
            ("fail_on_send", True, "harness_crashed"),
            ("raise_turn_timeout", True, "turn_timeout"),
            ("send_delay_s", 5.0, "turn_timeout"),
        ],
    )
    async def test_terminal_errors_apply_transform(self, failure, value, error_type):
        # [FUNCTION DOC]
        # Parametrized failure modes producing harness_crashed or turn_timeout: the transform is applied
        # once to the terminal error observation.
        seen = []

        def transform(observation):
            # [FUNCTION DOC]
            # HELPER: Records the observation and returns a copy with metadata transformed=True.
            seen.append(observation)
            return observation.model_copy(
                update={"metadata": {**observation.metadata, "transformed": True}}
            )

        env, adapter = make_env(transform=transform)
        try:
            await env.reset_async()
            seen.clear()
            setattr(adapter, failure, value)
            obs = await env.step_async(HarnessAction(message="go"), timeout_s=0.05)
            assert obs.done is True
            assert obs.metadata["error_type"] == error_type
            assert obs.metadata["transformed"] is True
            assert len(seen) == 1
            assert seen[0].metadata["turn_events"] == obs.metadata["turn_events"]
        finally:
            env.close()


class TestCleanupAndErrorClassification:
    # [CLASS DOC]
    # Tests resource cleanup and error classification for unexpected failures and cancellation.
    @pytest.mark.parametrize(
        "stage, failure",
        [
            ("is_alive", RuntimeError("adapter liveness check failed")),
            ("send_message", RuntimeError("unexpected adapter failure")),
            ("send_message", json.JSONDecodeError("invalid harness JSON", "{", 1)),
            ("rubric", RuntimeError("turn evaluation failed")),
        ],
    )
    @pytest.mark.parametrize("stop_fails", [False, True])
    async def test_unexpected_turn_failure_stops_resources_and_allows_retry(
        self, stage, failure, stop_fails, monkeypatch
    ):
        # [FUNCTION DOC]
        # Parametrized by failing stage (is_alive, send_message, rubric) and whether stop also fails: the
        # original exception propagates, the adapter and bridge are stopped, the episode ends, and after
        # restoring methods a new reset and step succeed.
        env, adapter = make_env(rubric=SpyRubric())
        await env.reset_async()
        target = env if stage == "rubric" else adapter
        method = "_apply_rubric_async" if stage == "rubric" else stage
        original = getattr(target, method)
        original_stop = adapter.stop

        async def fail(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Raises the prepared failure.
            raise failure

        async def failing_stop():
            # [FUNCTION DOC]
            # HELPER: Runs the real stop then raises RuntimeError.
            await original_stop()
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(target, method, fail)
        if stop_fails:
            monkeypatch.setattr(adapter, "stop", failing_stop)
        try:
            with pytest.raises(type(failure)) as caught:
                await env.step_async(HarnessAction(message="go"))

            assert caught.value is failure
            assert adapter.alive is False
            assert adapter.calls[-1] == "stop"
            assert FakeBridge.instances[0].stopped >= 1
            with pytest.raises(HarnessNotRunningError, match="No active episode"):
                await env.step_async(HarnessAction(message="again"))

            monkeypatch.setattr(target, method, original)
            monkeypatch.setattr(adapter, "stop", original_stop)
            await env.reset_async()
            assert adapter.alive
            assert len(FakeBridge.instances) == 2
            obs = await env.step_async(HarnessAction(message="fresh turn"))
            assert obs.metadata["response"] == "default response"
        finally:
            env.close()

    @pytest.mark.parametrize("stage", ["is_alive", "send_message", "rubric"])
    @pytest.mark.parametrize("stop_fails", [False, True])
    async def test_cancelled_turn_cleans_up_resources(
        self, stage, stop_fails, monkeypatch
    ):
        # [FUNCTION DOC]
        # Parametrized by stage and stop failure: cancelling a stalled step propagates CancelledError,
        # stops resources and ends the episode; a later reset and step succeed.
        env, adapter = make_env(rubric=SpyRubric())
        await env.reset_async()
        entered = asyncio.Event()
        target = env if stage == "rubric" else adapter
        method = "_apply_rubric_async" if stage == "rubric" else stage
        original = getattr(target, method)
        original_stop = adapter.stop

        async def stall(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Runs the original method, signals entry, then waits forever.
            await original(*args, **kwargs)
            entered.set()
            await asyncio.Event().wait()

        async def failing_stop():
            # [FUNCTION DOC]
            # HELPER: Runs the real stop then raises HarnessError.
            await original_stop()
            raise HarnessError("cleanup failed")

        monkeypatch.setattr(target, method, stall)
        if stop_fails:
            monkeypatch.setattr(adapter, "stop", failing_stop)
        turn_task = asyncio.create_task(env.step_async(HarnessAction(message="go")))
        try:
            await asyncio.wait_for(entered.wait(), timeout=10.0)
            turn_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn_task

            assert adapter.alive is False
            assert adapter.calls[-1] == "stop"
            assert FakeBridge.instances[0].stopped >= 1
            with pytest.raises(HarnessNotRunningError):
                await env.step_async(HarnessAction(message="again"))

            monkeypatch.setattr(target, method, original)
            monkeypatch.setattr(adapter, "stop", original_stop)
            await env.reset_async()
            obs = await env.step_async(HarnessAction(message="fresh turn"))
            assert obs.metadata["response"] == "default response"
        finally:
            if not turn_task.done():
                turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
            env.close()

    @pytest.mark.parametrize("stage", ["inject_tools", "start", "rubric"])
    async def test_cancelled_reset_cleans_up_resources(self, stage, monkeypatch):
        # [FUNCTION DOC]
        # Parametrized by stage (inject_tools, start, rubric reset): cancelling a stalled reset stops the
        # adapter and bridge and leaves no active episode.
        env, adapter = make_env(rubric=SpyRubric())
        entered = asyncio.Event()
        target = env if stage == "rubric" else adapter
        method = "_reset_rubric_async" if stage == "rubric" else stage
        original = getattr(target, method)

        async def stall(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Runs the original method, signals entry, then waits forever.
            await original(*args, **kwargs)
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(target, method, stall)
        reset_task = asyncio.create_task(env.reset_async())
        try:
            await asyncio.wait_for(entered.wait(), timeout=10.0)
            reset_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reset_task

            assert adapter.alive is False
            assert adapter.calls[-1] == "stop"
            assert FakeBridge.instances[0].stopped >= 1
            with pytest.raises(HarnessNotRunningError):
                await env.step_async(HarnessAction(message="go"))
        finally:
            if not reset_task.done():
                reset_task.cancel()
            await asyncio.gather(reset_task, return_exceptions=True)
            env.close()

    async def test_reset_stops_adapter_even_when_not_alive(self):
        # [FUNCTION DOC]
        # Reset calls stop first even when the adapter reports not alive.
        # A harness that died on its own reports is_alive() False while still
        # holding reapable resources; skipping stop() leaked them.
        env, adapter = make_env()
        await env.reset_async()
        adapter.alive = False
        adapter.calls.clear()

        await env.reset_async()

        assert adapter.calls[0] == "stop"

    async def test_start_failure_stops_the_adapter(self):
        # [FUNCTION DOC]
        # A failing start is followed by stop.
        env, adapter = make_env()
        adapter.fail_on_start = True
        with pytest.raises(HarnessError):
            await env.reset_async()
        # inject -> start (raises) -> stop
        assert adapter.calls[-1] == "stop"

    async def test_adapter_raised_timeout_is_not_reported_as_a_crash(self):
        # [FUNCTION DOC]
        # An adapter-raised HarnessTurnTimeoutError is reported as turn_timeout.
        env, adapter = make_env()
        await env.reset_async()
        adapter.raise_turn_timeout = True

        obs = await env.step_async(HarnessAction(message="hi"))

        assert obs.done is True
        assert obs.metadata["error_type"] == "turn_timeout"


class TestModeSpecificTools:
    # [CLASS DOC]
    # Tests that mode-specific tools are not injected into the harness.
    async def test_mode_tools_are_not_injected(self, caplog):
        # [FUNCTION DOC]
        # A production-only tool registered on the env is listed in production mode but not injected, and a
        # warning naming it is logged.
        # Tools registered with tool(mode=...) are tracked by the environment
        # rather than the FastMCP server, so the bridge cannot serve them.
        # Advertising them would promise the harness a tool it cannot call.
        env, adapter = make_env()

        @env.tool(mode="production")
        def prod_only(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Production-only tool returning x.
            """Production-only tool."""
            return x

        # A mode tool is only advertised when the env is in that mode; without
        # this the tool never reaches the injectable list and the test would
        # pass vacuously.
        env._mode = "production"
        listed = await env._async_handle_list_tools()
        assert "prod_only" in [t.name for t in listed.tools]

        with caplog.at_level("WARNING"):
            obs = await env.reset_async()

        assert "prod_only" not in obs.metadata["injected_tools"]
        assert all(t.name != "prod_only" for t in adapter.injected_tools)
        assert "prod_only" in caplog.text
