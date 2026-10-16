# [MODULE DOC]
# Reward rubrics for the REPL environment: outcome rubrics comparing the final answer (from
# observation metadata "final_answer") with an expected answer set at reset (exact, fuzzy, custom
# metric), a per-step process rubric based on execution success, and the composite REPLRubric.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rubrics for the REPL environment.

Follows the OpenEnv Rubric system (RFC 004) to provide composable,
outcome-based rewards suitable for RL training (GRPO, etc.).

The key insight from DSPy GRPO and Daytona RL guides: the RLM is a pure
inference engine. Reward computation is external — it compares the final
answer against ground truth. The environment provides the reward via rubrics;
the training framework consumes it.
"""

from __future__ import annotations

from typing import Any, Callable

from openenv.core.rubrics.base import Rubric


class ExactMatchRubric(Rubric):
    # [CLASS DOC]
    # Outcome rubric: 1.0 when the observation is done and its final answer equals the expected answer
    # (case- and whitespace-insensitive when normalize is True), else 0.0; 0.0 with no expected answer.
    """Outcome rubric: 1.0 if final answer matches expected, 0.0 otherwise.

    This is the standard outcome-based reward used by GRPO-style training.
    The expected answer is set via `set_expected()` at reset time.
    """

    def __init__(self, normalize: bool = True) -> None:
        # [FUNCTION DOC]
        # Stores the normalise flag; no expected answer yet.
        super().__init__()
        self._expected: str | None = None
        self._normalize = normalize

    def set_expected(self, expected: str | None) -> None:
        # [FUNCTION DOC]
        # Sets (or clears with None) the expected answer.
        self._expected = expected

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns the exact-match score as described for the class.
        if self._expected is None:
            return 0.0
        if not getattr(observation, "done", False):
            return 0.0
        final = getattr(observation, "metadata", {}).get("final_answer")
        if final is None:
            return 0.0
        if self._normalize:
            return (
                1.0
                if str(final).strip().lower() == str(self._expected).strip().lower()
                else 0.0
            )
        return 1.0 if str(final) == str(self._expected) else 0.0

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clears the expected answer.
        self._expected = None


class FuzzyMatchRubric(Rubric):
    # [CLASS DOC]
    # Outcome rubric: 1.0 for a normalised exact match, 0.5 when one normalised string contains the other,
    # else 0.0; 0.0 when not done, no final answer or no expected answer.
    """Outcome rubric: partial credit based on string containment.

    Returns 1.0 for exact match, 0.5 if expected is contained in the answer
    (or vice versa), 0.0 otherwise. Useful for tasks where partial matches
    are acceptable.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialises with no expected answer.
        super().__init__()
        self._expected: str | None = None

    def set_expected(self, expected: str | None) -> None:
        # [FUNCTION DOC]
        # Sets the expected answer.
        self._expected = expected

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns the fuzzy score as described for the class. An empty final answer counts as contained.
        if self._expected is None:
            return 0.0
        if not getattr(observation, "done", False):
            return 0.0
        final = getattr(observation, "metadata", {}).get("final_answer")
        if final is None:
            return 0.0
        final_norm = str(final).strip().lower()
        expected_norm = str(self._expected).strip().lower()
        if final_norm == expected_norm:
            return 1.0
        if expected_norm in final_norm or final_norm in expected_norm:
            return 0.5
        return 0.0

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clears the expected answer.
        self._expected = None


class CustomMetricRubric(Rubric):
    # [CLASS DOC]
    # Outcome rubric delegating to a user metric(expected, predicted) once the episode is done.
    """Outcome rubric using a user-provided metric function.

    This mirrors the DSPy GRPO pattern where the user provides
    `metric(expected, predicted) -> float`.
    """

    def __init__(self, metric_fn: Callable[[str, str], float]) -> None:
        # [FUNCTION DOC]
        # Stores the metric callable; no expected answer yet.
        super().__init__()
        self._metric_fn = metric_fn
        self._expected: str | None = None

    def set_expected(self, expected: str | None) -> None:
        # [FUNCTION DOC]
        # Sets the expected answer.
        self._expected = expected

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns metric(expected, final answer) as strings when done with both present, else 0.0.
        # Exceptions from the metric propagate.
        if self._expected is None:
            return 0.0
        if not getattr(observation, "done", False):
            return 0.0
        final = getattr(observation, "metadata", {}).get("final_answer")
        if final is None:
            return 0.0
        return self._metric_fn(str(self._expected), str(final))

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clears the expected answer.
        self._expected = None


class CodeExecutionRubric(Rubric):
    # [CLASS DOC]
    # Process rubric: error_penalty (default -0.05) when observation.result.success is False, else
    # success_reward (default 0.0); 0.0 when there is no result.
    # KNOWN ISSUE: the docstring promises a positive reward for success and 0.0 for non-terminal steps,
    # but the default success reward is 0.0 and the rubric does not look at done.
    """Process rubric: per-step signal based on code execution success.

    Returns a small positive reward for successful execution,
    a negative reward for errors, 0.0 for non-terminal steps.
    """

    def __init__(
        self,
        success_reward: float = 0.0,
        error_penalty: float = -0.05,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the success reward and error penalty.
        super().__init__()
        self.success_reward = success_reward
        self.error_penalty = error_penalty

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns the process reward as described for the class.
        result = getattr(observation, "result", None)
        if result is None:
            return 0.0
        if not getattr(result, "success", True):
            return self.error_penalty
        return self.success_reward


class REPLRubric(Rubric):
    # [CLASS DOC]
    # Composite rubric: on terminal steps returns the outcome rubric score if a final answer exists, else
    # failure_reward (default -0.1); on non-terminal steps returns the process rubric score.
    # Defaults: ExactMatchRubric outcome and CodeExecutionRubric process.
    """Composite rubric for the REPL environment.

    Combines outcome-based reward (final answer correctness) with
    optional process-based reward (code execution quality).

    The outcome rubric is only evaluated on terminal steps (done=True).
    The process rubric is evaluated on every step.
    """

    def __init__(
        self,
        outcome: Rubric | None = None,
        process: Rubric | None = None,
        failure_reward: float = -0.1,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the outcome and process rubrics (creating defaults) and the failure reward.
        super().__init__()
        self.outcome = outcome or ExactMatchRubric()
        self.process = process or CodeExecutionRubric()
        self.failure_reward = failure_reward

    def set_expected(self, expected: str | None) -> None:
        # [FUNCTION DOC]
        # Forwards the expected answer to the outcome rubric if it supports it.
        """Pass expected answer to the outcome rubric."""
        if hasattr(self.outcome, "set_expected"):
            self.outcome.set_expected(expected)

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns the composite reward as described for the class.
        done = getattr(observation, "done", False)
        if done:
            final = getattr(observation, "metadata", {}).get("final_answer")
            if final is not None:
                return self.outcome(action, observation)
            # Done but no final answer (max iterations exhausted)
            return self.failure_reward
        # Non-terminal step: process reward only
        return self.process(action, observation)

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Resets both child rubrics.
        self.outcome.reset()
        self.process.reset()
