# [MODULE DOC]
# Generic observation transforms: a composite that chains transforms and a
# pass-through no-op.
# SPDX-License-Identifier: BSD-3-Clause

"""Base transform implementations for composing environment-specific transforms."""

from .interfaces import Transform
from .types import Observation


class CompositeTransform(Transform):
    # [CLASS DOC]
    # Transform applying a list of transforms in order, feeding each output into
    # the next.
    """Combines multiple transforms into a single transform."""

    def __init__(self, transforms: list[Transform]):
        # [FUNCTION DOC]
        # Stores the transforms list (not copied).
        self.transforms = transforms

    def __call__(self, observation: Observation) -> Observation:
        # [FUNCTION DOC]
        # Applies each transform sequentially and returns the final observation.
        for transform in self.transforms:
            observation = transform(observation)
        return observation


class NullTransform(Transform):
    # [CLASS DOC]
    # Transform that returns the observation unchanged.
    """Default transform that passes through unchanged."""

    def __call__(self, observation: Observation) -> Observation:
        # [FUNCTION DOC]
        # Returns observation as-is.
        return observation
