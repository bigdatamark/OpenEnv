# [MODULE DOC]
# Tests for TrajectoryRubric (accumulates (action, observation) steps, returns an intermediate
# reward until an observation is done, then the trajectory score) and
# ExponentialDiscountingTrajectoryRubric (per-step rewards discounted by gamma from the final
# score): basics, gamma validation and discount patterns, state_dict, hooks and edge cases.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TrajectoryRubric and ExponentialDiscountingTrajectoryRubric."""

from dataclasses import dataclass
from typing import Any, List, Tuple

import pytest
from openenv.core.rubrics.trajectory import (
    ExponentialDiscountingTrajectoryRubric,
    TrajectoryRubric,
)


@dataclass
class MockObservation:
    # [CLASS DOC]
    # HELPER: Dataclass observation with done flag and a metadata dict (None replaced by an empty dict).
    """Mock observation for testing."""

    done: bool = False
    metadata: dict = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # HELPER: Replaces a None metadata with an empty dict.
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockAction:
    # [CLASS DOC]
    # HELPER: Dataclass action with a string value and a metadata dict (None replaced by an empty dict).
    """Mock action for testing."""

    value: str = "move"
    metadata: dict = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # HELPER: Replaces a None metadata with an empty dict.
        if self.metadata is None:
            self.metadata = {}


class WinLossRubric(ExponentialDiscountingTrajectoryRubric):
    # [CLASS DOC]
    # HELPER: Discounting rubric scoring the final observation's metadata outcome: win 1.0, loss 0.0,
    # anything else 0.5; empty trajectory 0.0.
    """Example rubric that scores 1.0 for win, 0.0 for loss, 0.5 for draw."""

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns the outcome-based score described above.
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        outcome = getattr(final_obs, "metadata", {}).get("outcome")
        if outcome == "win":
            return 1.0
        elif outcome == "loss":
            return 0.0
        else:
            return 0.5


class EqualCreditRubric(TrajectoryRubric):
    # [CLASS DOC]
    # HELPER: TrajectoryRubric scoring the final observation's metadata "score" (default 0.0) and
    # giving that score to every step.
    """Rubric that gives equal credit to all steps."""

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns the final observation's metadata score, or 0.0 for an empty trajectory.
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        return final_obs.metadata.get("score", 0.0)

    def compute_step_rewards(self) -> List[float]:
        # [FUNCTION DOC]
        # HELPER: Returns the trajectory score repeated once per step (empty list if no steps).
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class TestTrajectoryRubricBasics:
    # [CLASS DOC]
    # Tests TrajectoryRubric basics.
    """Test basic TrajectoryRubric functionality."""

    def test_abstract_methods_required(self):
        # [FUNCTION DOC]
        # TrajectoryRubric cannot be instantiated directly.
        """Cannot instantiate TrajectoryRubric without implementing abstract methods."""
        with pytest.raises(TypeError):
            TrajectoryRubric()

    def test_returns_intermediate_until_done(self):
        # [FUNCTION DOC]
        # A non-done step returns the intermediate reward and is recorded.
        """Returns intermediate_reward for non-terminal steps."""
        rubric = EqualCreditRubric(intermediate_reward=0.0)

        obs1 = MockObservation(done=False)
        result = rubric(MockAction(), obs1)

        assert result == 0.0
        assert len(rubric._trajectory) == 1

    def test_returns_score_when_done(self):
        # [FUNCTION DOC]
        # A done step returns the trajectory score.
        """Returns trajectory score when done=True."""
        rubric = EqualCreditRubric(intermediate_reward=0.0)

        obs1 = MockObservation(done=False)
        obs2 = MockObservation(done=True, metadata={"score": 0.8})

        rubric(MockAction(), obs1)
        result = rubric(MockAction(), obs2)

        assert result == 0.8
        assert len(rubric._trajectory) == 2

    def test_custom_intermediate_reward(self):
        # [FUNCTION DOC]
        # A custom intermediate reward is returned for non-done steps.
        """Intermediate reward can be customized."""
        rubric = EqualCreditRubric(intermediate_reward=0.1)

        obs = MockObservation(done=False)
        result = rubric(MockAction(), obs)

        assert result == 0.1

    def test_reset_clears_trajectory(self):
        # [FUNCTION DOC]
        # reset clears the recorded steps.
        """reset() clears the accumulated trajectory."""
        rubric = EqualCreditRubric()

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        assert len(rubric._trajectory) == 2

        rubric.reset()
        assert len(rubric._trajectory) == 0

    def test_trajectory_property_returns_copy(self):
        # [FUNCTION DOC]
        # The trajectory property returns a copy.
        """trajectory property returns a copy."""
        rubric = EqualCreditRubric()

        rubric(MockAction(), MockObservation(done=False))
        trajectory = rubric.trajectory

        # Modifying the copy should not affect internal state
        trajectory.clear()
        assert len(rubric._trajectory) == 1


class TestExponentialDiscounting:
    # [CLASS DOC]
    # Tests exponential discounting.
    """Test ExponentialDiscountingTrajectoryRubric."""

    def test_gamma_validation(self):
        # [FUNCTION DOC]
        # gamma outside [0, 1] raises ValueError.
        """Gamma must be in [0, 1]."""
        with pytest.raises(ValueError):
            WinLossRubric(gamma=-0.1)

        with pytest.raises(ValueError):
            WinLossRubric(gamma=1.5)

    def test_gamma_one_equal_credit(self):
        # [FUNCTION DOC]
        # gamma 1 gives every step the final score.
        """With gamma=1.0, all steps get equal credit."""
        rubric = WinLossRubric(gamma=1.0)

        # Simulate 3-step episode with win
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert len(step_rewards) == 3
        assert step_rewards[0] == 1.0
        assert step_rewards[1] == 1.0
        assert step_rewards[2] == 1.0

    def test_gamma_zero_final_only(self):
        # [FUNCTION DOC]
        # gamma 0 rewards only the final step.
        """With gamma=0.0, only final step gets reward."""
        rubric = WinLossRubric(gamma=0.0)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [0.0, 0.0, 1.0]

    def test_gamma_discounting_pattern(self):
        # [FUNCTION DOC]
        # gamma 0.5 gives 0.25, 0.5, 1.0 over three steps.
        """With 0 < gamma < 1, later steps get higher reward."""
        rubric = WinLossRubric(gamma=0.5)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        # r_t = gamma^(T-1-t) * R_final, T=3, R_final=1.0
        # t=0: 0.5^2 = 0.25
        # t=1: 0.5^1 = 0.5
        # t=2: 0.5^0 = 1.0
        assert step_rewards[0] == pytest.approx(0.25)
        assert step_rewards[1] == pytest.approx(0.5)
        assert step_rewards[2] == pytest.approx(1.0)

    def test_gamma_099_standard_discounting(self):
        # [FUNCTION DOC]
        # gamma 0.99 gives strictly increasing rewards ending at 1.0.
        """With gamma=0.99, standard RL discounting pattern."""
        rubric = WinLossRubric(gamma=0.99)

        # 5-step episode with win
        for _ in range(4):
            rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        # Verify discounting order: later steps get more
        for i in range(len(step_rewards) - 1):
            assert step_rewards[i] < step_rewards[i + 1]

        # Final step gets full reward
        assert step_rewards[-1] == pytest.approx(1.0)

    def test_loss_outcome(self):
        # [FUNCTION DOC]
        # A loss gives 0.0 to every step.
        """Loss returns 0.0 for all steps."""
        rubric = WinLossRubric(gamma=0.99)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "loss"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [0.0, 0.0]

    def test_draw_outcome(self):
        # [FUNCTION DOC]
        # A draw with gamma 1 gives 0.5 to every step.
        """Draw returns 0.5 for all steps (with discounting)."""
        rubric = WinLossRubric(gamma=1.0)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "draw"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [0.5, 0.5]

    def test_empty_trajectory(self):
        # [FUNCTION DOC]
        # No steps yields an empty reward list.
        """compute_step_rewards() returns empty list for empty trajectory."""
        rubric = WinLossRubric(gamma=0.99)

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == []


class TestTrajectoryRubricStateSerialization:
    # [CLASS DOC]
    # Tests state_dict support.
    """Test state_dict serialization for trajectory rubrics."""

    def test_trajectory_rubric_state_dict(self):
        # [FUNCTION DOC]
        # state_dict includes intermediate_reward.
        """TrajectoryRubric serializes intermediate_reward."""
        rubric = EqualCreditRubric(intermediate_reward=0.2)

        state = rubric.state_dict()

        assert state["intermediate_reward"] == 0.2

    def test_trajectory_rubric_load_state_dict(self):
        # [FUNCTION DOC]
        # load_state_dict restores intermediate_reward.
        """TrajectoryRubric loads intermediate_reward from state."""
        rubric = EqualCreditRubric(intermediate_reward=0.0)

        rubric.load_state_dict({"intermediate_reward": 0.3})

        assert rubric.intermediate_reward == 0.3

    def test_exponential_discounting_state_dict(self):
        # [FUNCTION DOC]
        # state_dict includes gamma and intermediate_reward.
        """ExponentialDiscountingTrajectoryRubric serializes gamma."""
        rubric = WinLossRubric(gamma=0.95, intermediate_reward=0.1)

        state = rubric.state_dict()

        assert state["gamma"] == 0.95
        assert state["intermediate_reward"] == 0.1

    def test_exponential_discounting_load_state_dict(self):
        # [FUNCTION DOC]
        # load_state_dict restores gamma and intermediate_reward.
        """ExponentialDiscountingTrajectoryRubric loads gamma from state."""
        rubric = WinLossRubric(gamma=0.99)

        rubric.load_state_dict({"gamma": 0.9, "intermediate_reward": 0.2})

        assert rubric.gamma == 0.9
        assert rubric.intermediate_reward == 0.2


class TestTrajectoryRubricHooks:
    # [CLASS DOC]
    # Tests hooks on trajectory rubrics.
    """Test that hooks work with trajectory rubrics."""

    def test_hooks_called_each_step(self):
        # [FUNCTION DOC]
        # Forward hooks see the intermediate reward then the final score.
        """Forward hooks are called on each step."""
        rubric = EqualCreditRubric()
        hook_calls = []

        def hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Records the result.
            hook_calls.append(result)

        rubric.register_forward_hook(hook)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"score": 0.7}))

        assert len(hook_calls) == 2
        assert hook_calls[0] == 0.0  # intermediate
        assert hook_calls[1] == 0.7  # final


class TestTrajectoryRubricEdgeCases:
    # [CLASS DOC]
    # Tests edge cases.
    """Test edge cases."""

    def test_single_step_episode(self):
        # [FUNCTION DOC]
        # A single done step gets the full score.
        """Single-step episode (immediately done)."""
        rubric = WinLossRubric(gamma=0.99)

        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [1.0]

    def test_very_long_episode(self):
        # [FUNCTION DOC]
        # In a 100-step episode the first step gets 0.99 to the 99th power and the last gets 1.0.
        """Long episode (100 steps)."""
        rubric = WinLossRubric(gamma=0.99)

        for _ in range(99):
            rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert len(step_rewards) == 100
        # First step should have gamma^99 reward
        assert step_rewards[0] == pytest.approx(0.99**99)
        # Last step should have full reward
        assert step_rewards[-1] == 1.0

    def test_observation_without_done_attribute(self):
        # [FUNCTION DOC]
        # An observation lacking done is treated as not done.
        """Handles observations without done attribute (defaults to False)."""
        rubric = EqualCreditRubric()

        class NoDoneObs:
            # [CLASS DOC]
            # HELPER: Empty class without a done attribute.
            pass

        obs = NoDoneObs()
        result = rubric(MockAction(), obs)

        # Should treat as not done
        assert result == 0.0
        assert len(rubric._trajectory) == 1

    def test_multiple_episodes_with_reset(self):
        # [FUNCTION DOC]
        # Rewards are computed independently per episode after reset.
        """Multiple episodes with reset between them."""
        rubric = WinLossRubric(gamma=1.0)

        # Episode 1: win
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))
        ep1_rewards = rubric.compute_step_rewards()

        rubric.reset()

        # Episode 2: loss
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "loss"}))
        ep2_rewards = rubric.compute_step_rewards()

        assert ep1_rewards == [1.0, 1.0]
        assert ep2_rewards == [0.0]
