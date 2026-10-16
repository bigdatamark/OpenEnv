# [MODULE DOC]
# Trajectory rubrics for delayed rewards: accumulate (action, observation) pairs over an
# episode, return an intermediate reward until the observation is done, then score the
# whole trajectory; subclasses define per-step credit assignment (e.g. exponential
# discounting).
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-based rubrics for delayed reward computation.

These rubrics accumulate trajectory data and compute rewards based on
episode outcomes rather than individual steps. This supports scenarios
where reward signals depend on future events:

- Terminal games (chess, Go): Win/loss known only at game end
- Plan execution: Plan quality depends on execution success
- Multi-agent games: One player's action quality depends on opponent response

See RFC 004 "Delayed Rewards" section for design rationale.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from openenv.core.rubrics.base import Rubric


class TrajectoryRubric(Rubric):
    # [CLASS DOC]
    # Abstract rubric that stores the episode trajectory in memory and returns
    # intermediate_reward for non-terminal steps and score_trajectory() at the terminal step.
    # Subclasses implement score_trajectory and compute_step_rewards. The trajectory is only
    # cleared by reset().
    """Abstract base for rubrics that score based on full trajectories.

    Subclasses implement:
    - score_trajectory(): Compute final score from trajectory
    - compute_step_rewards(): Define credit assignment strategy

    The __call__ method accumulates steps and returns rewards according
    to the subclass's implementation.

    IMPORTANT: Trajectories are stored in CPU memory to avoid GPU pressure.
    Environments with GPU tensors in observations must move them to CPU
    before returning from step().

    Known limitation: Very long episodes (thousands of steps) may consume
    significant CPU memory. For such cases, consider streaming rubrics.

    Examples:

        ```python
        class WinLossRubric(TrajectoryRubric):
            def score_trajectory(self, trajectory):
                _, final_obs = trajectory[-1]
                return 1.0 if final_obs.metadata.get('won') else 0.0

            def compute_step_rewards(self):
                # Equal credit to all steps
                score = self.score_trajectory(self._trajectory)
                return [score] * len(self._trajectory)

        rubric = WinLossRubric()
        for action, obs in episode:
            reward = rubric(action, obs)  # 0.0 until done
        step_rewards = rubric.compute_step_rewards()  # Credit assignment
        ```
    """

    _trajectory: List[Tuple[Any, Any]]
    intermediate_reward: float

    def __init__(self, intermediate_reward: float = 0.0):
        # [FUNCTION DOC]
        # Stores intermediate_reward (default 0.0) and an empty trajectory.
        """Initialize trajectory rubric.

        Args:
            intermediate_reward (`float`, *optional*, defaults to `0.0`):
                Value to return for non-terminal steps.
        """
        super().__init__()
        self.intermediate_reward = intermediate_reward
        self._trajectory = []

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Appends (action, observation); returns score_trajectory(trajectory) if observation.done
        # is truthy (missing attribute counts as not done), else intermediate_reward.
        """Accumulate step and return reward.

        Returns intermediate_reward until done, then computes trajectory score.

        Args:
            action: The action taken.
            observation: The resulting observation. Must have a 'done' attribute.

        Returns:
            `float`: intermediate_reward if not done, else score_trajectory() result.
        """
        self._trajectory.append((action, observation))

        if getattr(observation, "done", False):
            return self.score_trajectory(self._trajectory)
        else:
            return self.intermediate_reward

    @abstractmethod
    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        # [FUNCTION DOC]
        # Abstract. Must return the final score for a list of (action, observation) pairs.
        """Score the complete trajectory. Return 0.0-1.0.

        Called when observation.done=True.

        Args:
            trajectory (`list`):
                List of (action, observation) tuples.

        Returns:
            `float`: Final trajectory score (typically 0.0 to 1.0).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_step_rewards(self) -> List[float]:
        # [FUNCTION DOC]
        # Abstract. Must return one reward per accumulated step.
        """Compute per-step rewards from the accumulated trajectory.

        Define your credit assignment strategy here (e.g., discounting,
        assigning all credit to specific steps, etc.).

        Returns:
            `list[float]`: Rewards, one per step. Length matches len(trajectory).
        """
        raise NotImplementedError

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clears the accumulated trajectory.
        """Clear accumulated trajectory. Call on env.reset()."""
        self._trajectory = []

    @property
    def trajectory(self) -> List[Tuple[Any, Any]]:
        # [FUNCTION DOC]
        # Returns a copy of the trajectory list.
        """Current trajectory (read-only copy)."""
        return list(self._trajectory)

    def state_dict(self) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Returns {"intermediate_reward": value}; trajectory is not serialized.
        """Serialize configuration (not trajectory data)."""
        return {"intermediate_reward": self.intermediate_reward}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Restores intermediate_reward if present.
        """Load configuration from checkpoint."""
        if "intermediate_reward" in state:
            self.intermediate_reward = state["intermediate_reward"]


class ExponentialDiscountingTrajectoryRubric(TrajectoryRubric):
    # [CLASS DOC]
    # Trajectory rubric whose per-step rewards are R_final * gamma^(T-1-t), where R_final is
    # score_trajectory() and T the trajectory length.
    """TrajectoryRubric with exponential discounting for credit assignment.

    Per-step reward:

    ```text
    r_t = gamma^(T-1-t) * R_final
    ```

    With gamma=0.99, later steps get higher reward (they're "closer" to the outcome).
    With gamma=1.0, all steps get equal reward.
    With gamma=0.0, only the final step gets reward.

    This is the standard temporal discounting used in reinforcement learning,
    applied retroactively once the episode outcome is known.

    Examples:

        ```python
        class ChessRubric(ExponentialDiscountingTrajectoryRubric):
            def score_trajectory(self, trajectory):
                _, final_obs = trajectory[-1]
                outcome = final_obs.metadata.get('winner')
                if outcome == 'agent': return 1.0
                elif outcome == 'opponent': return 0.0
                else: return 0.5  # Draw

        rubric = ChessRubric(gamma=0.99)
        reward = rubric(action, obs)  # 0.0 until done, then final score
        step_rewards = rubric.compute_step_rewards()  # Discounted per-step rewards
        ```
    """

    gamma: float

    def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
        # [FUNCTION DOC]
        # Stores intermediate_reward and gamma.
        # Raises: ValueError if gamma is outside [0, 1].
        """Initialize with discount factor.

        Args:
            gamma (`float`, *optional*, defaults to `0.99`):
                Discount factor in [0, 1]. Higher values give more credit to early moves.
            intermediate_reward (`float`, *optional*, defaults to `0.0`):
                Value to return for non-terminal steps.
        """
        super().__init__(intermediate_reward=intermediate_reward)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma

    def compute_step_rewards(self) -> List[float]:
        # [FUNCTION DOC]
        # Returns [] for an empty trajectory; otherwise computes R_final once and returns
        # R_final * gamma^(T-1-t) for t = 0..T-1.
        """Apply exponential discounting from final reward.

        Returns:
            `list[float]`: Discounted rewards where `step_rewards[t] = gamma^(T-1-t) * R_final`,
            T is the trajectory length and R_final is score_trajectory().
        """
        if not self._trajectory:
            return []

        final_score = self.score_trajectory(self._trajectory)
        T = len(self._trajectory)
        return [final_score * (self.gamma ** (T - 1 - t)) for t in range(T)]

    def state_dict(self) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Returns the base state plus gamma.
        """Serialize configuration."""
        state = super().state_dict()
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Restores base state and gamma if present (no range validation).
        """Load configuration from checkpoint."""
        super().load_state_dict(state)
        if "gamma" in state:
            self.gamma = state["gamma"]
