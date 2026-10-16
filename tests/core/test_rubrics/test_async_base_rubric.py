# [MODULE DOC]
# Tests that the Rubric base class supports async forward implementations: awaiting calls,
# last_score tracking, async composites, sync and async forward/pre-forward hooks, child traversal
# (children, named_rubrics, get_rubric) and mixing with sync rubrics.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for async Rubric functionality.

This test file verifies that the Rubric base class supports async operations:
- async forward() method
- async __call__() with hooks
- Async hook execution
- Integration with async environments
"""

from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric


class AsyncRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric with an async forward returning a fixed score (default 1.0).
    """Concrete async rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0):
        # [FUNCTION DOC]
        # HELPER: Stores the score.
        super().__init__()
        self.score = score

    async def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns the stored score.
        """Async forward implementation."""
        # Simulate async work (e.g., API call, DB query)
        return self.score


class AsyncCompositeRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric with two AsyncRubric children (0.5 and 0.7) whose async forward averages them.
    """Rubric with async child rubrics."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Creates the two children.
        super().__init__()
        self.child1 = AsyncRubric(0.5)
        self.child2 = AsyncRubric(0.7)

    async def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Awaits both children and returns the mean.
        """Async forward that awaits children."""
        score1 = await self.child1(action, observation)
        score2 = await self.child2(action, observation)
        return (score1 + score2) / 2


class TestAsyncRubricBasics:
    # [CLASS DOC]
    # Tests basic async behavior.
    """Test basic async Rubric functionality."""

    @pytest.mark.asyncio
    async def test_async_forward_is_awaitable(self):
        # [FUNCTION DOC]
        # forward can be awaited directly.
        """Async forward() can be awaited."""
        rubric = AsyncRubric(0.8)
        result = await rubric.forward("action", "observation")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_async_call_invokes_forward(self):
        # [FUNCTION DOC]
        # Awaiting the rubric call returns the forward result.
        """Calling an async rubric invokes async forward()."""
        rubric = AsyncRubric(0.8)
        result = await rubric("action", "observation")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_last_score_tracked_async(self):
        # [FUNCTION DOC]
        # last_score is None initially and set after an awaited call.
        """last_score is updated after async call."""
        rubric = AsyncRubric(0.6)
        assert rubric.last_score is None

        await rubric("action", "observation")
        assert rubric.last_score == 0.6

    @pytest.mark.asyncio
    async def test_async_composite_rubric(self):
        # [FUNCTION DOC]
        # An async composite averages its children.
        """Composite rubric with async children works."""
        rubric = AsyncCompositeRubric()
        result = await rubric("action", "observation")
        assert result == pytest.approx(0.6)  # (0.5 + 0.7) / 2


class TestAsyncRubricHooks:
    # [CLASS DOC]
    # Tests hooks with async rubrics.
    """Test async hook functionality."""

    @pytest.mark.asyncio
    async def test_forward_hook_called_async(self):
        # [FUNCTION DOC]
        # A forward hook receives action, observation and result.
        """Forward hooks are called after async forward()."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        def hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Records (action, observation, result).
            hook_calls.append((action, obs, result))

        rubric.register_forward_hook(hook)
        await rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs", 0.9)

    @pytest.mark.asyncio
    async def test_forward_pre_hook_called_async(self):
        # [FUNCTION DOC]
        # A pre-forward hook receives action and observation.
        """Pre-forward hooks are called before async forward()."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Records (action, observation).
            hook_calls.append((action, obs))

        rubric.register_forward_pre_hook(pre_hook)
        await rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs")

    @pytest.mark.asyncio
    async def test_multiple_hooks_async(self):
        # [FUNCTION DOC]
        # Multiple hooks run in registration order.
        """Multiple hooks work with async rubrics."""
        rubric = AsyncRubric(0.5)
        results = []

        rubric.register_forward_hook(lambda r, a, o, res: results.append(1))
        rubric.register_forward_hook(lambda r, a, o, res: results.append(2))

        await rubric("action", "obs")

        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_async_hooks(self):
        # [FUNCTION DOC]
        # Async forward hooks are awaited.
        """Async hooks are supported."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        async def async_hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Records the result.
            # Simulate async work in hook (e.g., logging to API)
            hook_calls.append(result)

        rubric.register_forward_hook(async_hook)
        await rubric("action", "obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == 0.9

    @pytest.mark.asyncio
    async def test_async_pre_hooks(self):
        # [FUNCTION DOC]
        # Async pre-forward hooks are awaited.
        """Async pre-hooks are supported."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        async def async_pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Records (action, observation).
            # Simulate async pre-processing
            hook_calls.append((action, obs))

        rubric.register_forward_pre_hook(async_pre_hook)
        await rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs")


class TestAsyncChildTraversal:
    # [CLASS DOC]
    # Tests child traversal on async rubrics.
    """Test async rubric child traversal works correctly."""

    @pytest.mark.asyncio
    async def test_children_still_iterable(self):
        # [FUNCTION DOC]
        # children() yields both child rubrics.
        """children() works the same for async rubrics."""
        rubric = AsyncCompositeRubric()

        children = list(rubric.children())
        assert len(children) == 2
        assert rubric.child1 in children
        assert rubric.child2 in children

    @pytest.mark.asyncio
    async def test_named_rubrics_async(self):
        # [FUNCTION DOC]
        # named_rubrics() yields dotted paths for nested children.
        """named_rubrics() works with async rubrics."""

        class NestedAsyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Rubric wrapping an AsyncCompositeRubric as attribute inner.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Creates the inner composite.
                super().__init__()
                self.inner = AsyncCompositeRubric()

            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Awaits and returns the inner rubric's score.
                return await self.inner(action, observation)

        rubric = NestedAsyncRubric()

        paths = dict(rubric.named_rubrics())
        assert "inner" in paths
        assert "inner.child1" in paths
        assert "inner.child2" in paths

    @pytest.mark.asyncio
    async def test_get_rubric_by_path_async(self):
        # [FUNCTION DOC]
        # get_rubric resolves dotted paths to the child objects.
        """get_rubric() works with async rubrics."""

        class NestedAsyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Rubric wrapping an AsyncCompositeRubric as attribute inner.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Creates the inner composite.
                super().__init__()
                self.inner = AsyncCompositeRubric()

            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Awaits and returns the inner rubric's score.
                return await self.inner(action, observation)

        rubric = NestedAsyncRubric()

        assert rubric.get_rubric("inner") is rubric.inner
        assert rubric.get_rubric("inner.child1") is rubric.inner.child1


class TestBackwardCompatibility:
    # [CLASS DOC]
    # Tests sync rubrics alongside async ones.
    """Test that sync rubrics still work (backward compatibility)."""

    @pytest.mark.asyncio
    async def test_sync_rubric_still_works_sync(self):
        # [FUNCTION DOC]
        # A sync rubric call returns its value directly.
        """Synchronous rubrics can still be called synchronously."""

        class SyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Sync rubric returning 0.5.
            def forward(self, action: Any, observation: Any) -> float:
                # [FUNCTION DOC]
                # HELPER: Returns 0.5.
                return 0.5

        rubric = SyncRubric()
        # Should work synchronously
        result = rubric("action", "obs")
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_sync_and_async_rubrics_mixed(self):
        # [FUNCTION DOC]
        # An async composite can call a sync child directly and await an async child.
        """Mixing sync and async rubrics in a composite."""

        class SyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Sync rubric returning 0.3.
            def forward(self, action: Any, observation: Any) -> float:
                # [FUNCTION DOC]
                # HELPER: Returns 0.3.
                return 0.3

        class MixedComposite(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric averaging a sync child (0.3) and an async child (0.7).
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Creates both children.
                super().__init__()
                self.sync_child = SyncRubric()
                self.async_child = AsyncRubric(0.7)

            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Calls the sync child, awaits the async child, and returns their mean.
                # Can call sync child directly
                sync_score = self.sync_child(action, observation)
                # Must await async child
                async_score = await self.async_child(action, observation)
                return (sync_score + async_score) / 2

        rubric = MixedComposite()
        result = await rubric("action", "obs")
        assert result == pytest.approx(0.5)  # (0.3 + 0.7) / 2
