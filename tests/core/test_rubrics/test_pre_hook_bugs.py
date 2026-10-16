# [MODULE DOC]
# Regression tests for rubric hook and container execution-order bugs: pre-forward hooks must run
# before forward (sync and async), Sequential must call each child exactly once when switching from
# sync to async children mid-chain, and containers (Sequential, Gate, WeightedSum) must run
# pre-hooks in both sync and async paths.
# KNOWN ISSUE: the module and several test docstrings say the tests currently fail and cite specific
# source line numbers; these statements describe a past state and are likely stale.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for pre-hook and container execution bugs.

These tests verify:
1. Pre-hooks are called BEFORE forward() executes (not after)
2. Sequential doesn't call rubrics twice when async is detected mid-way
3. Container pre-hooks work in sync path

These tests currently FAIL due to implementation bugs.
"""

from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import Gate, Sequential, WeightedSum


class TrackingRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Sync rubric appending "<name>.forward" to its execution log and returning a fixed score.
    """Rubric that tracks execution order."""

    def __init__(self, name: str, score: float = 1.0):
        # [FUNCTION DOC]
        # HELPER: Stores name and score and creates an empty log.
        super().__init__()
        self.name = name
        self.score = score
        self.execution_log = []

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Logs the call and returns the score.
        self.execution_log.append(f"{self.name}.forward")
        return self.score


class AsyncTrackingRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Async rubric appending "<name>.forward" to its execution log and returning a fixed score.
    """Async rubric that tracks execution order."""

    def __init__(self, name: str, score: float = 1.0):
        # [FUNCTION DOC]
        # HELPER: Stores name and score and creates an empty log.
        super().__init__()
        self.name = name
        self.score = score
        self.execution_log = []

    async def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Logs the call and returns the score.
        self.execution_log.append(f"{self.name}.forward")
        return self.score


class TestPreHookExecutionOrder:
    # [CLASS DOC]
    # Tests that pre-hooks run before forward.
    """Test that pre-hooks are called BEFORE forward(), not after."""

    def test_pre_hook_before_forward_sync(self):
        # [FUNCTION DOC]
        # In the sync path the pre-hook runs while forward has not yet logged anything.
        """Pre-hook must be called BEFORE forward() executes (sync path).

        BUG: Currently pre-hook is called AFTER forward() in base.py:78-88.
        The sync path calls forward() at line 69, then _call_sync() at line 76,
        which calls pre-hooks at line 81. This is backwards.
        """
        rubric = TrackingRubric("test", 0.8)
        execution_order = []

        def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Records the call and asserts forward has not run yet.
            # Pre-hook should see that forward hasn't executed yet
            execution_order.append("pre_hook")
            # At this point, execution_log should be EMPTY
            assert len(rubric.execution_log) == 0, "Pre-hook called AFTER forward()!"

        rubric.register_forward_pre_hook(pre_hook)
        result = rubric("action", "obs")

        # Verify correct execution order
        assert execution_order == ["pre_hook"], "Pre-hook not called"
        assert rubric.execution_log == ["test.forward"], "Forward not called"
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_pre_hook_before_forward_async(self):
        # [FUNCTION DOC]
        # In the async path an async pre-hook runs before forward.
        """Pre-hook must be called BEFORE forward() executes (async path)."""
        rubric = AsyncTrackingRubric("test", 0.8)
        execution_order = []

        async def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Records the call and asserts forward has not run yet.
            execution_order.append("pre_hook")
            # At this point, execution_log should be EMPTY
            assert len(rubric.execution_log) == 0, "Pre-hook called AFTER forward()!"

        rubric.register_forward_pre_hook(pre_hook)
        result = await rubric("action", "obs")

        # Verify correct execution order
        assert execution_order == ["pre_hook"], "Pre-hook not called"
        assert rubric.execution_log == ["test.forward"], "Forward not called"
        assert result == 0.8

    def test_pre_hook_can_modify_state(self):
        # [FUNCTION DOC]
        # State set by a pre-hook is visible to a post-hook.
        # KNOWN ISSUE: this cannot detect a pre-hook running after forward, since the post-hook only checks
        # that the pre-hook ran at some point.
        """Pre-hook should be able to set up state before forward() runs.

        This is a realistic use case: pre-hook sets up caching or logging
        that forward() relies on.
        """
        rubric = TrackingRubric("test", 0.5)
        state = {"initialized": False}

        def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Marks the shared state initialized.
            state["initialized"] = True

        def post_hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Asserts the shared state was initialized.
            # Verify state was set by pre-hook BEFORE forward ran
            assert state["initialized"], "Pre-hook didn't run before forward"

        rubric.register_forward_pre_hook(pre_hook)
        rubric.register_forward_hook(post_hook)
        rubric("action", "obs")


class TestSequentialDoubleCallBug:
    # [CLASS DOC]
    # Tests that Sequential calls each child once across sync/async transitions.
    """Test that Sequential doesn't call rubrics twice when async detected mid-way."""

    @pytest.mark.asyncio
    async def test_sequential_async_third_position_no_double_call(self):
        # [FUNCTION DOC]
        # With an async child third of four, every child runs once and the last score is returned.
        """When async is detected at position 2 (third rubric), no double-call.

        BUG: In containers.py:95-101, when we detect async mid-iteration,
        we call `score = rubric(action, observation)` at line 96, which
        returns a coroutine. Then at line 99-101, we check if it's a coroutine,
        and if so, we call `_call_async_mid` with `self._rubric_list[1:]`.

        The problem: We've already called the async rubric at line 96, but
        we discard its coroutine and then call it AGAIN in _call_async_mid
        because we pass `self._rubric_list[1:]` which includes all rubrics
        after the first one, not just the ones after the current position.
        """
        # Create sequence: sync, sync, async, sync
        r1 = TrackingRubric("sync1", 1.0)
        r2 = TrackingRubric("sync2", 0.9)
        r3 = AsyncTrackingRubric("async1", 0.8)  # Third position - triggers switch
        r4 = TrackingRubric("sync3", 0.7)

        rubric = Sequential(r1, r2, r3, r4)
        result = await rubric("action", "obs")

        # Each rubric should be called exactly once
        assert len(r1.execution_log) == 1, (
            f"r1 called {len(r1.execution_log)} times: {r1.execution_log}"
        )
        assert len(r2.execution_log) == 1, (
            f"r2 called {len(r2.execution_log)} times: {r2.execution_log}"
        )
        assert len(r3.execution_log) == 1, (
            f"r3 called {len(r3.execution_log)} times: {r3.execution_log}"
        )
        assert len(r4.execution_log) == 1, (
            f"r4 called {len(r4.execution_log)} times: {r4.execution_log}"
        )
        assert result == 0.7

    @pytest.mark.asyncio
    async def test_sequential_async_detected_midway_no_double_call(self):
        # [FUNCTION DOC]
        # With an async child second of three, every child runs once.
        """When async is detected mid-way, rubrics shouldn't be called twice.

        This test has async at the second position (index 1), which means
        the loop at line 95 hasn't executed yet, so the bug may not manifest.
        """
        # Create a mixed sync/async sequence
        r1 = TrackingRubric("sync1", 1.0)  # Sync
        r2 = AsyncTrackingRubric("async1", 0.8)  # Async (triggers switch)
        r3 = TrackingRubric("sync2", 0.9)  # Sync

        rubric = Sequential(r1, r2, r3)
        result = await rubric("action", "obs")

        # Each rubric should be called exactly once
        assert r1.execution_log == ["sync1.forward"], (
            f"r1 called wrong: {r1.execution_log}"
        )
        assert r2.execution_log == ["async1.forward"], (
            f"r2 called wrong: {r2.execution_log}"
        )
        assert r3.execution_log == ["sync2.forward"], (
            f"r3 called wrong: {r3.execution_log}"
        )
        assert result == 0.9

    @pytest.mark.asyncio
    async def test_sequential_async_at_second_position_no_double_call(self):
        # [FUNCTION DOC]
        # A sync then async pair each run once.
        """Specific case: async at position 1 (second rubric).

        When async is at position 1, it's detected immediately after the first
        rubric, so _call_async_detected is called, not _call_async_mid.
        This test ensures no double-calls in that path.
        """
        call_counts = {"sync": 0, "async": 0}

        class CountingSync(Rubric):
            # [CLASS DOC]
            # HELPER: Sync rubric incrementing the shared sync counter and returning 1.0.
            def forward(self, action, obs):
                # [FUNCTION DOC]
                # HELPER: Increments the sync count and returns 1.0.
                call_counts["sync"] += 1
                return 1.0

        class CountingAsync(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric incrementing the shared async counter and returning 0.8.
            async def forward(self, action, obs):
                # [FUNCTION DOC]
                # HELPER: Increments the async count and returns 0.8.
                call_counts["async"] += 1
                return 0.8

        rubric = Sequential(CountingSync(), CountingAsync())
        result = await rubric("action", "obs")

        # Each should be called exactly once
        assert call_counts["sync"] == 1, f"Sync called {call_counts['sync']} times"
        assert call_counts["async"] == 1, f"Async called {call_counts['async']} times"
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_sequential_multiple_async_transitions(self):
        # [FUNCTION DOC]
        # Alternating sync/async children each run once.
        """Test multiple sync->async transitions don't cause double calls."""
        r1 = TrackingRubric("sync1", 1.0)
        r2 = AsyncTrackingRubric("async1", 0.8)
        r3 = TrackingRubric("sync2", 0.9)
        r4 = AsyncTrackingRubric("async2", 0.7)

        rubric = Sequential(r1, r2, r3, r4)
        result = await rubric("action", "obs")

        # Verify each called exactly once
        assert len(r1.execution_log) == 1
        assert len(r2.execution_log) == 1
        assert len(r3.execution_log) == 1
        assert len(r4.execution_log) == 1
        assert result == 0.7


class TestContainerPreHooksSyncPath:
    # [CLASS DOC]
    # Tests container pre-hooks in the sync path.
    """Test that container pre-hooks work in sync path."""

    def test_sequential_pre_hooks_called_sync(self):
        # [FUNCTION DOC]
        # Sequential runs its pre-hook in the sync path.
        """Sequential should call pre-hooks in sync path.

        BUG: Looking at containers.py:68-116, the Sequential.__call__ sync
        path never calls _forward_pre_hooks. Pre-hooks are only called in
        the async helper methods (_empty_async, _wrap_sync_result, etc.).
        """
        rubric = Sequential(
            TrackingRubric("r1", 1.0),
            TrackingRubric("r2", 0.8),
        )

        pre_hook_called = {"called": False}

        def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Marks the pre-hook as called.
            pre_hook_called["called"] = True

        rubric.register_forward_pre_hook(pre_hook)
        result = rubric("action", "obs")

        assert pre_hook_called["called"], "Pre-hook not called in sync path"
        assert result == 0.8

    def test_gate_pre_hooks_called_sync(self):
        # [FUNCTION DOC]
        # Gate runs its pre-hook in the sync path.
        """Gate should call pre-hooks in sync path."""
        rubric = Gate(TrackingRubric("child", 0.8), threshold=0.5)

        pre_hook_called = {"called": False}

        def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Marks the pre-hook as called.
            pre_hook_called["called"] = True

        rubric.register_forward_pre_hook(pre_hook)
        result = rubric("action", "obs")

        assert pre_hook_called["called"], "Gate pre-hook not called in sync path"
        assert result == 0.8

    def test_weighted_sum_pre_hooks_called_sync(self):
        # [FUNCTION DOC]
        # WeightedSum runs its pre-hook in the sync path.
        """WeightedSum should call pre-hooks in sync path."""
        rubric = WeightedSum(
            [TrackingRubric("r1", 0.6), TrackingRubric("r2", 0.8)],
            [0.5, 0.5],
        )

        pre_hook_called = {"called": False}

        def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Marks the pre-hook as called.
            pre_hook_called["called"] = True

        rubric.register_forward_pre_hook(pre_hook)
        result = rubric("action", "obs")

        assert pre_hook_called["called"], "WeightedSum pre-hook not called in sync path"
        assert result == pytest.approx(0.7)

    def test_sequential_post_hooks_still_work_sync(self):
        # [FUNCTION DOC]
        # Sequential post-hooks receive the result (control).
        """Verify post-hooks still work (as control test)."""
        rubric = Sequential(
            TrackingRubric("r1", 1.0),
            TrackingRubric("r2", 0.8),
        )

        post_hook_called = {"called": False, "result": None}

        def post_hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Records that it was called and the result.
            post_hook_called["called"] = True
            post_hook_called["result"] = result

        rubric.register_forward_hook(post_hook)
        result = rubric("action", "obs")

        assert post_hook_called["called"], "Post-hook not called"
        assert post_hook_called["result"] == 0.8
        assert result == 0.8


class TestContainerPreHooksAsyncPath:
    # [CLASS DOC]
    # Tests container pre-hooks in the async path (controls).
    """Test that container pre-hooks work correctly in async path (control tests)."""

    @pytest.mark.asyncio
    async def test_sequential_pre_hooks_called_async(self):
        # [FUNCTION DOC]
        # Sequential runs an async pre-hook in the async path.
        """Sequential should call pre-hooks in async path (this should work)."""
        rubric = Sequential(
            AsyncTrackingRubric("r1", 1.0),
            AsyncTrackingRubric("r2", 0.8),
        )

        pre_hook_called = {"called": False}

        async def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Marks the pre-hook as called.
            pre_hook_called["called"] = True

        rubric.register_forward_pre_hook(pre_hook)
        result = await rubric("action", "obs")

        assert pre_hook_called["called"], "Pre-hook not called in async path"
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_gate_pre_hooks_called_async(self):
        # [FUNCTION DOC]
        # Gate runs an async pre-hook in the async path.
        """Gate should call pre-hooks in async path (this should work)."""
        rubric = Gate(AsyncTrackingRubric("child", 0.8), threshold=0.5)

        pre_hook_called = {"called": False}

        async def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Marks the pre-hook as called.
            pre_hook_called["called"] = True

        rubric.register_forward_pre_hook(pre_hook)
        result = await rubric("action", "obs")

        assert pre_hook_called["called"], "Gate pre-hook not called in async path"
        assert result == 0.8
