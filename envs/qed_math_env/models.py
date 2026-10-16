# [MODULE DOC]
# Pydantic models for the QED Math environment: marker base action/observation classes, three
# action types (submit proof, get problem, get grading guidelines), and two observation types
# (problem statement and graded submission). RewardValue is a union of bool, int, float or None.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the QED Math Environment.

Defines action and observation types for mathematical proof submission
and grading.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


RewardValue = bool | int | float | None


class QEDMathAction(Action):
    # [CLASS DOC]
    # Base marker action class with no extra fields.
    """Base action for the QED Math environment."""


class SubmitProof(QEDMathAction):
    # [CLASS DOC]
    # Action carrying proof text (required) and an attempt_number (default 1).
    """Submit a proof attempt for the current problem."""

    proof: str = Field(..., description="The proof text submitted by agent")
    attempt_number: int = Field(default=1, description="Attempt counter")


class GetProblem(QEDMathAction):
    # [CLASS DOC]
    # Marker action requesting the current problem; no fields.
    """Request the current problem statement."""


class GetGradingGuidelines(QEDMathAction):
    # [CLASS DOC]
    # Marker action requesting the grading guidelines; no fields.
    """Request the grading guidelines/rubric for current problem."""


class QEDMathObservation(Observation):
    # [CLASS DOC]
    # Base marker observation class with no extra fields.
    """Base observation for the QED Math environment."""


class ProblemObservation(QEDMathObservation):
    # [CLASS DOC]
    # Problem observation: problem text, reference_solution, grading_guidelines, problem_id,
    # dataset_source, problem_type (default "proof"; also "answer" or "multi_step") and max_attempts (default 1).
    """Observation containing the problem statement."""

    problem: str = Field(default="", description="The mathematical problem")
    reference_solution: str = Field(default="", description="Ground truth solution")
    grading_guidelines: str = Field(
        default="", description="Rubric for grading (0-7 scale)"
    )
    problem_id: str = Field(default="", description="Unique problem identifier")
    dataset_source: str = Field(default="", description="Source dataset name")
    problem_type: str = Field(
        default="proof",
        description="Problem type: proof, answer, or multi_step",
    )
    max_attempts: int = Field(
        default=1,
        description="Maximum number of allowed submission attempts",
    )


class ProofSubmissionObservation(QEDMathObservation):
    # [CLASS DOC]
    # Graded submission: proof, score (0-7 integer), feedback, reward (default 0.0), done (default True),
    # problem_type, attempt_number, attempts_remaining and is_correct.
    """Observation returned after submitting a proof."""

    proof: str = Field(default="", description="The submitted proof")
    score: int = Field(default=0, description="Grade from rubric (0-7)")
    feedback: str = Field(default="", description="Grader feedback")
    reward: RewardValue = Field(
        default=0.0,
        description="Normalized reward (score/7)",
    )
    done: bool = Field(default=True, description="Episode ends after proof submission")
    problem_type: str = Field(
        default="proof",
        description="Problem type used to evaluate this submission",
    )
    attempt_number: int = Field(
        default=1, description="1-based submission attempt index"
    )
    attempts_remaining: int = Field(
        default=0,
        description="Remaining submission attempts in the current episode",
    )
    is_correct: bool = Field(
        default=False,
        description="Whether the submission is considered fully correct",
    )
