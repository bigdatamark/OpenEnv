# [MODULE DOC]
# Reward rubric tree for Pelican SVG. Leaf rubrics only read scores already computed and stored on
# the observation by the scoring module; the tree is a gate followed by a weighted sum.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward composition for the Pelican SVG environment.

The children are thin readers: all the work happens once in
[`~envs.pelican_svg_env.server.scoring.evaluate_submission`], which hangs its
results on the observation. Composing them therefore costs nothing and the tree
stays introspectable via `env.rubric.named_rubrics()`.

See RFC 004 for the rubric design: `rfcs/004-rubrics.md`.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum

from .scoring import component_weights


class GatePassed(Rubric):
    # [CLASS DOC]
    # Rubric returning 1.0 if observation.gate_passed is true, else 0.0.
    """One if the submission cleared the deterministic checks, zero otherwise."""

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns the gate verdict as 1.0 or 0.0; action ignored.
        """Return the gate verdict as a score."""
        return 1.0 if observation.gate_passed else 0.0


class StructureScore(Rubric):
    # [CLASS DOC]
    # Rubric returning the observation's structure_score.
    """Fraction of structural checks the drawing passed."""

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns observation.structure_score; action ignored.
        """Return the structural score carried on the observation."""
        return observation.structure_score


class SemanticScore(Rubric):
    # [CLASS DOC]
    # Rubric returning the observation's semantic_score (vision judge result).
    """The vision judge's verdict on whether the drawing shows the subject."""

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Returns observation.semantic_score; action ignored.
        """Return the semantic score carried on the observation."""
        return observation.semantic_score


def build_rubric(judge_enabled: bool = True) -> Rubric:
    # [FUNCTION DOC]
    # Builds the rubric: a Sequential of a Gate on GatePassed (threshold 1.0, zeroes the reward on
    # failure) then a WeightedSum of structure and semantic scores, with weights from
    # component_weights(judge_enabled). Returns the composed Rubric.
    """Assemble the reward tree.

    The gate comes first and zeroes everything when it fails, so a submission
    that embedded a bitmap cannot collect structural credit for the bitmap's
    contents. Past the gate, structure and semantics are combined by weight.

    Args:
        judge_enabled (`bool`, *optional*, defaults to `True`):
            Whether a vision judge is configured. Without one the semantic
            score is unmeasured rather than bad, so structure takes the full
            weight. The split comes from
            [`~envs.pelican_svg_env.server.scoring.component_weights`], the
            same function the observation's reward uses.

    Returns:
        [`~openenv.core.rubrics.Rubric`]: The composed rubric.

    Examples:

    ```python
    rubric = build_rubric(judge_enabled=False)
    reward = rubric(action, observation)
    ```
    """
    structure_weight, semantic_weight = component_weights(judge_enabled)
    return Sequential(
        Gate(GatePassed(), threshold=1.0),
        WeightedSum(
            [StructureScore(), SemanticScore()],
            weights=[structure_weight, semantic_weight],
        ),
    )
