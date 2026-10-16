# [MODULE DOC]
# Chess rubric giving discounted win/loss/draw rewards (RFC 004). Falls back to a minimal local
# ExponentialDiscountingTrajectoryRubric when the installed openenv core lacks rubrics.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chess-specific rubrics for reward computation.

Provides ChessWinLossRubric that extends ExponentialDiscountingTrajectoryRubric
to compute temporally-discounted rewards based on chess game outcomes.

See RFC 004 for rubric design: rfcs/004-rubrics.md
"""

from typing import Any, List, Tuple

try:
    from openenv.core.rubrics.trajectory import ExponentialDiscountingTrajectoryRubric
except ModuleNotFoundError:

    class ExponentialDiscountingTrajectoryRubric:
        # [CLASS DOC]
        # HELPER: Compatibility fallback that records (action, observation) pairs, returns intermediate_reward
        # until an observation is done, then the trajectory score; can compute discounted per-step rewards.
        """Compatibility fallback when the installed core lacks rubrics."""

        def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
            # [FUNCTION DOC]
            # Stores gamma (0.99), intermediate reward (0.0) and an empty trajectory.
            self.gamma = gamma
            self.intermediate_reward = intermediate_reward
            self._trajectory: List[Tuple[Any, Any]] = []

        def __call__(self, action: Any, observation: Any) -> float:
            # [FUNCTION DOC]
            # Appends the step; returns score_trajectory when observation.done, else intermediate_reward.
            self._trajectory.append((action, observation))
            if getattr(observation, "done", False):
                return self.score_trajectory(self._trajectory)
            return self.intermediate_reward

        def reset(self) -> None:
            # [FUNCTION DOC]
            # Clears the trajectory.
            self._trajectory = []

        def compute_step_rewards(self) -> List[float]:
            # [FUNCTION DOC]
            # Returns gamma^(T-1-t) * final score for each recorded step (empty list if none).
            if not self._trajectory:
                return []

            final_score = self.score_trajectory(self._trajectory)
            total_steps = len(self._trajectory)
            return [
                self.gamma ** (total_steps - 1 - step_index) * final_score
                for step_index in range(total_steps)
            ]


class ChessWinLossRubric(ExponentialDiscountingTrajectoryRubric):
    # [CLASS DOC]
    # Trajectory rubric scoring the final observation's reward (+1 win, -1 loss, 0 draw), discounted per step.
    """Score chess game based on win/loss/draw outcome with temporal discounting.

    Per-step reward: r_t = gamma^(T-1-t) * R_final

    Terminal rewards:
    - Win:  +1.0
    - Loss: -1.0
    - Draw:  0.0

    Note: Unlike the base class convention of 0.0-1.0 scores, chess uses
    -1.0 for losses to match standard chess RL conventions.

    Usage:
        rubric = ChessWinLossRubric(gamma=0.99)
        rubric.reset()
        for action, obs in episode:
            reward = rubric(action, obs)  # 0.0 until done
        step_rewards = rubric.compute_step_rewards()
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        # [FUNCTION DOC]
        # Returns the last observation's reward attribute (0.0 for an empty trajectory or missing attribute).
        """Score based on game outcome from final observation.

        Reads the reward from the terminal observation, which encodes:
        +1.0 for agent win, -1.0 for agent loss, 0.0 for draw.

        Args:
            trajectory: List of (action, observation) tuples.

        Returns:
            Terminal reward: +1.0 (win), -1.0 (loss), 0.0 (draw).
        """
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        return getattr(final_obs, "reward", 0.0)
