# [MODULE DOC]
# Observation transforms for the Julia environment: a regex-based safety check
# that penalizes executed code containing dangerous Julia calls, plus a factory.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
envs/julia_env/julia_transforms.py
--------------------------------
Safety and quality transforms for Julia code.
"""

import re

# Support both in-repo and standalone imports
try:
    # In-repo imports
    from openenv.core.env_server.interfaces import Transform

    from ..models import JuliaObservation
except ImportError:
    from models import JuliaObservation

    # Standalone imports
    from openenv.core.env_server.interfaces import Transform


# -------------------------
# Safety Transform
# -------------------------
class JuliaSafetyTransform(Transform):
    # [CLASS DOC]
    # Transform that adds a penalty (default -3.0) to the reward of a
    # JuliaObservation whose executed code (from metadata core_code and test_code)
    # matches any dangerous pattern: run(, read(, write(, unsafe_, ccall(,
    # Base.exit, Base.kill, rm(, download(.
    # KNOWN ISSUE: patterns are unanchored substrings, so e.g. "thread(" or
    # "rerun(" also match and are penalized.
    """Detects dangerous Julia operations and penalizes them with a negative reward."""

    def __init__(self, penalty: float = -3.0):
        # [FUNCTION DOC]
        # Stores the penalty and the list of dangerous regex patterns.
        self.penalty = penalty
        self.dangerous_patterns = [
            r"run\(",
            r"read\(",
            r"write\(",
            r"unsafe_",
            r"ccall\(",
            r"Base\.exit",
            r"Base\.kill",
            r"rm\(",  # file deletion
            r"download\(",  # downloading
        ]

    def __call__(self, observation):
        # [FUNCTION DOC]
        # Returns non-JuliaObservation inputs unchanged. Otherwise builds the code text
        # from metadata; on the first matching pattern adds the penalty to the reward
        # (None treated as 0.0) and records the pattern under
        # metadata["safety_violation"]; otherwise coerces a None reward to 0.0.
        # Mutates and returns the same observation.
        # Only act on JuliaObservation objects
        if not isinstance(observation, JuliaObservation):
            return observation

        # Extract executed code from metadata (core_code + test_code)
        if observation.metadata:
            code = (
                observation.metadata.get("core_code", "")
                + "\n"
                + observation.metadata.get("test_code", "")
            )
        else:
            code = ""

        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
                # Apply penalty and record violation
                observation.reward = (observation.reward or 0.0) + self.penalty
                observation.metadata = observation.metadata or {}
                observation.metadata["safety_violation"] = pattern
                return observation

        # Safe code gets neutral reward
        observation.reward = observation.reward or 0.0
        return observation


# -------------------------
# Factory
# -------------------------
def create_safe_julia_transform():
    # [FUNCTION DOC]
    # Returns a new JuliaSafetyTransform with the default penalty.
    """Creates safety transform for Julia code."""
    return JuliaSafetyTransform()
