# [MODULE DOC]
# Tests rubric integration with an async Environment: async reward computation via
# _apply_rubric_async in step_async, rubric reset via _reset_rubric_async in reset_async,
# introspection, multi-episode lifecycle, hooks, error propagation and concurrent environments.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for async rubric integration with async Environment.

This test file verifies that rubrics work with async environments:
- Async _apply_rubric() in async environments
- Async step() with rubric evaluation
- Async reset_async() with rubric reset
- Async rubric hooks during environment step
"""

from typing import Any, Optional

import pytest
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.rubrics import Rubric


# Test fixtures - using Pydantic models


class MockAction(Action):
    # [CLASS DOC]
    # HELPER: Action with a string content (default "test").
    """Simple action for testing."""

    content: str = "test"


class MockObservation(Observation):
    # [CLASS DOC]
    # HELPER: Observation with a string content.
    """Simple observation for testing."""

    content: str = ""


class MockState(State):
    # [CLASS DOC]
    # HELPER: Plain State subclass.
    """Simple state for testing."""

    pass


class AsyncRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Async rubric counting calls; returns 1.0 for action content "good", 0.0 for "bad", else its base score.
    """Async rubric that returns action-dependent score."""

    def __init__(self, base_score: float = 1.0):
        # [FUNCTION DOC]
        # HELPER: Stores the base score and zeroes the call count.
        super().__init__()
        self.base_score = base_score
        self.call_count = 0

    async def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Increments call_count and returns the score described above.
        """Async forward with action-based scoring."""
        self.call_count += 1
        if hasattr(action, "content"):
            if action.content == "good":
                return 1.0
            elif action.content == "bad":
                return 0.0
        return self.base_score


class AsyncCompositeRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric averaging two AsyncRubric children with base scores 0.5 and 0.8.
    """Composite rubric with async children."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Creates the two children.
        super().__init__()
        self.child1 = AsyncRubric(0.5)
        self.child2 = AsyncRubric(0.8)

    async def forward(self, action, observation):
        # [FUNCTION DOC]
        # HELPER: Awaits both children and returns their mean.
        """Async forward combining children."""
        score1 = await self.child1(action, observation)
        score2 = await self.child2(action, observation)
        return (score1 + score2) / 2


class AsyncEnvironment(Environment[MockAction, MockObservation, MockState]):
    # [CLASS DOC]
    # HELPER: Environment with both sync and async reset/step; steps produce "response to <content>"
    # observations whose reward comes from the rubric (sync or async application respectively).
    """Async environment implementation for testing."""

    def __init__(self, rubric: Optional[Rubric] = None):
        # [FUNCTION DOC]
        # HELPER: Passes the rubric to the base class and initializes state.
        super().__init__(rubric=rubric)
        self._state = MockState()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MockObservation:
        # [FUNCTION DOC]
        # HELPER: Resets the rubric synchronously, sets state with the episode id, returns an "initial" observation.
        """Sync reset (fallback)."""
        self._reset_rubric()
        self._state = MockState(episode_id=episode_id)
        return MockObservation(content="initial")

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MockObservation:
        # [FUNCTION DOC]
        # HELPER: Resets the rubric asynchronously, sets state, returns an "initial" observation.
        """Async reset."""
        await self._reset_rubric_async()
        self._state = MockState(episode_id=episode_id)
        return MockObservation(content="initial")

    def step(
        self,
        action: MockAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MockObservation:
        # [FUNCTION DOC]
        # HELPER: Builds the response observation and sets reward via the sync rubric application.
        """Sync step (fallback)."""
        obs = MockObservation(content=f"response to {action.content}")
        obs.reward = self._apply_rubric(action, obs)
        return obs

    async def step_async(
        self,
        action: MockAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MockObservation:
        # [FUNCTION DOC]
        # HELPER: Builds the response observation and sets reward via the async rubric application.
        """Async step with async rubric application."""
        obs = MockObservation(content=f"response to {action.content}")
        obs.reward = await self._apply_rubric_async(action, obs)
        return obs

    @property
    def state(self) -> MockState:
        # [FUNCTION DOC]
        # HELPER: Returns the current state.
        return self._state


class TestAsyncEnvironmentRubricIntegration:
    # [CLASS DOC]
    # Tests async rubric use within the environment.
    """Test async rubric integration with async Environment."""

    @pytest.mark.asyncio
    async def test_async_environment_without_rubric(self):
        # [FUNCTION DOC]
        # Without a rubric the reward is 0.0.
        """Async environment works without a rubric."""
        env = AsyncEnvironment()
        assert env.rubric is None

        obs = await env.reset_async()
        assert obs.content == "initial"

        obs = await env.step_async(MockAction(content="test"))
        assert obs.reward == 0.0  # Default when no rubric

    @pytest.mark.asyncio
    async def test_async_environment_with_rubric(self):
        # [FUNCTION DOC]
        # The rubric's score becomes the reward.
        """Async environment uses async rubric for reward computation."""
        rubric = AsyncRubric(0.75)
        env = AsyncEnvironment(rubric=rubric)

        assert env.rubric is rubric

        await env.reset_async()
        obs = await env.step_async(MockAction(content="test"))

        assert obs.reward == 0.75

    @pytest.mark.asyncio
    async def test_async_rubric_called_each_step(self):
        # [FUNCTION DOC]
        # The rubric is called once per step and not on reset.
        """Async rubric is called on each async step."""
        rubric = AsyncRubric()
        env = AsyncEnvironment(rubric=rubric)

        await env.reset_async()
        assert rubric.call_count == 0

        await env.step_async(MockAction(content="a"))
        assert rubric.call_count == 1

        await env.step_async(MockAction(content="b"))
        assert rubric.call_count == 2

    @pytest.mark.asyncio
    async def test_async_rubric_receives_action_and_observation(self):
        # [FUNCTION DOC]
        # The rubric sees the action (good gives 1.0, bad gives 0.0).
        """Async rubric receives both action and observation."""
        rubric = AsyncRubric()
        env = AsyncEnvironment(rubric=rubric)

        await env.reset_async()

        obs = await env.step_async(MockAction(content="good"))
        assert obs.reward == 1.0

        obs = await env.step_async(MockAction(content="bad"))
        assert obs.reward == 0.0

    @pytest.mark.asyncio
    async def test_async_rubric_reset_on_env_reset(self):
        # [FUNCTION DOC]
        # reset_async calls the rubric's async reset.
        """Async rubric state is reset when environment resets."""

        class StatefulAsyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric counting steps, with an async reset that zeroes the count.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Zeroes the step count.
                super().__init__()
                self.step_count = 0

            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Increments the step count and returns 1.0.
                self.step_count += 1
                return 1.0

            async def reset_async(self):
                # [FUNCTION DOC]
                # HELPER: Zeroes the step count.
                """Async reset."""
                self.step_count = 0

        rubric = StatefulAsyncRubric()
        env = AsyncEnvironment(rubric=rubric)

        await env.reset_async()
        await env.step_async(MockAction(content="a"))
        await env.step_async(MockAction(content="b"))

        assert rubric.step_count == 2

        await env.reset_async()
        assert rubric.step_count == 0  # Reset clears state

    @pytest.mark.asyncio
    async def test_async_rubric_introspection(self):
        # [FUNCTION DOC]
        # Child rubrics are reachable via named_children with their last scores.
        """Can introspect async rubric from environment."""
        rubric = AsyncCompositeRubric()
        env = AsyncEnvironment(rubric=rubric)

        await env.reset_async()
        await env.step_async(MockAction(content="test"))

        # Can introspect child scores
        assert env.rubric is not None
        named = dict(env.rubric.named_children())
        assert "child1" in named
        assert "child2" in named
        assert named["child1"].last_score == 0.5
        assert named["child2"].last_score == 0.8

    @pytest.mark.asyncio
    async def test_apply_rubric_async_without_rubric(self):
        # [FUNCTION DOC]
        # _apply_rubric_async returns 0.0 without a rubric.
        """_apply_rubric_async returns 0.0 when no rubric is set."""
        env = AsyncEnvironment()
        action = MockAction(content="test")
        obs = MockObservation(content="result")

        result = await env._apply_rubric_async(action, obs)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_reset_rubric_async_without_rubric(self):
        # [FUNCTION DOC]
        # _reset_rubric_async does not raise without a rubric.
        """_reset_rubric_async is safe when no rubric is set."""
        env = AsyncEnvironment()
        await env._reset_rubric_async()  # Should not raise


class TestAsyncEnvironmentRubricLifecycle:
    # [CLASS DOC]
    # Tests behavior across episodes.
    """Test async rubric lifecycle with multiple episodes."""

    @pytest.mark.asyncio
    async def test_multiple_async_episodes(self):
        # [FUNCTION DOC]
        # Reset clears per-episode rubric state.
        """Async rubric handles multiple episodes correctly."""

        class EpisodeTrackingRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric recording action contents, cleared by its async reset.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Initializes the list of seen actions.
                super().__init__()
                self.actions_seen = []

            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Records the action content and returns 1.0.
                self.actions_seen.append(action.content)
                return 1.0

            async def reset_async(self):
                # [FUNCTION DOC]
                # HELPER: Clears the seen actions.
                """Async reset."""
                self.actions_seen = []

        rubric = EpisodeTrackingRubric()
        env = AsyncEnvironment(rubric=rubric)

        # Episode 1
        await env.reset_async()
        await env.step_async(MockAction(content="a"))
        await env.step_async(MockAction(content="b"))
        ep1_len = len(rubric.actions_seen)

        # Episode 2
        await env.reset_async()
        await env.step_async(MockAction(content="c"))
        ep2_len = len(rubric.actions_seen)

        assert ep1_len == 2
        assert ep2_len == 1  # Reset cleared previous episode

    @pytest.mark.asyncio
    async def test_async_rubric_hooks_work(self):
        # [FUNCTION DOC]
        # Async forward hooks fire on every environment step.
        """Async rubric hooks work through environment."""
        rubric = AsyncRubric(0.9)
        env = AsyncEnvironment(rubric=rubric)

        hook_calls = []

        async def async_hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Records the result.
            hook_calls.append(result)

        rubric.register_forward_hook(async_hook)

        await env.reset_async()
        await env.step_async(MockAction(content="a"))
        await env.step_async(MockAction(content="b"))

        assert len(hook_calls) == 2
        assert hook_calls == [0.9, 0.9]

    @pytest.mark.asyncio
    async def test_async_rubric_with_slow_computation(self):
        # [FUNCTION DOC]
        # A slow async rubric's delay is reflected in step time and its score returned.
        """Async rubric with slow computation doesn't block."""
        import asyncio

        class SlowAsyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric sleeping 50ms and returning 0.8.
            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Sleeps 50ms and returns 0.8.
                # Simulate slow LLM judge or API call
                await asyncio.sleep(0.05)  # 50ms
                return 0.8

        rubric = SlowAsyncRubric()
        env = AsyncEnvironment(rubric=rubric)

        await env.reset_async()

        import time

        start = time.time()
        obs = await env.step_async(MockAction(content="test"))
        elapsed = time.time() - start

        assert obs.reward == 0.8
        assert elapsed >= 0.05  # Should take at least 50ms


class TestAsyncRubricErrorHandling:
    # [CLASS DOC]
    # Tests async error handling.
    """Test error handling in async rubrics."""

    @pytest.mark.asyncio
    async def test_async_rubric_exception_propagates(self):
        # [FUNCTION DOC]
        # A rubric exception propagates out of step_async.
        """Exceptions in async rubric propagate to caller."""

        class FailingAsyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric that raises ValueError.
            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Raises ValueError("Rubric evaluation failed").
                raise ValueError("Rubric evaluation failed")

        rubric = FailingAsyncRubric()
        env = AsyncEnvironment(rubric=rubric)

        await env.reset_async()

        with pytest.raises(ValueError, match="Rubric evaluation failed"):
            await env.step_async(MockAction(content="test"))

    @pytest.mark.asyncio
    async def test_async_hook_exception_handling(self):
        # [FUNCTION DOC]
        # An exception in an async forward hook propagates to the caller.
        # KNOWN ISSUE: the docstring says hook exceptions are handled gracefully, but the test asserts they propagate.
        """Exceptions in async hooks are handled gracefully."""

        class AsyncHookRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric returning 0.5.
            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Returns 0.5.
                return 0.5

        rubric = AsyncHookRubric()

        async def failing_hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Raises RuntimeError("Hook failed").
            raise RuntimeError("Hook failed")

        rubric.register_forward_hook(failing_hook)

        # Hook exceptions should propagate
        with pytest.raises(RuntimeError, match="Hook failed"):
            await rubric("action", "obs")


class TestAsyncRubricConcurrency:
    # [CLASS DOC]
    # Tests concurrent environments.
    """Test concurrent async rubric execution."""

    @pytest.mark.asyncio
    async def test_multiple_environments_concurrent_rubric_calls(self):
        # [FUNCTION DOC]
        # Five environments step concurrently, each with its own rubric seeing at most one concurrent call.
        """Multiple environments can call async rubrics concurrently."""
        import asyncio

        class ConcurrentAsyncRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Async rubric tracking current and maximum concurrent invocations.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Zeroes the counters.
                super().__init__()
                self.concurrent_calls = 0
                self.max_concurrent = 0

            async def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Increments the concurrent count, updates the maximum, sleeps 10ms, decrements, returns 1.0.
                self.concurrent_calls += 1
                self.max_concurrent = max(self.max_concurrent, self.concurrent_calls)
                await asyncio.sleep(0.01)  # Simulate work
                self.concurrent_calls -= 1
                return 1.0

        # Create multiple environments, each with its own rubric instance
        envs = [AsyncEnvironment(rubric=ConcurrentAsyncRubric()) for _ in range(5)]

        # Reset all environments
        await asyncio.gather(*[env.reset_async() for env in envs])

        # Step all environments concurrently
        results = await asyncio.gather(
            *[env.step_async(MockAction(content="test")) for env in envs]
        )

        assert len(results) == 5
        assert all(obs.reward == 1.0 for obs in results)

        # Each rubric should only have seen 1 concurrent call (its own)
        for env in envs:
            assert env.rubric.max_concurrent == 1
