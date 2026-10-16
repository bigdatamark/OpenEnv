# [MODULE DOC]
# Pydantic models for Grid World: the move enum, the action and the observation.
# Falls back to a top-level "core" package if openenv is not importable.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

try:
    from openenv.core.env_server.types import Action, Observation
    from pydantic import Field
except ImportError:
    from core.env_server.types import Action, Observation
    from pydantic import Field


# --- Action Models ---
class MoveAction(str, Enum):
    # [CLASS DOC]
    # String enum of the four moves: UP, DOWN, LEFT, RIGHT.
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class GridWorldAction(Action):
    # [CLASS DOC]
    # Action carrying a single required MoveAction.
    action: MoveAction = Field(..., description="The direction to move the agent.")


# --- Observation Model ---
class GridWorldObservation(Observation):
    # [CLASS DOC]
    # Observation with required x and y position, a status message (default ""),
    # and reward/done overridden to non-optional float 0.0 / bool False.
    x: int = Field(..., description="Current X position")
    y: int = Field(..., description="Current Y position")
    message: str = Field("", description="Status message")
    # Reward must be a float, default to 0.0 (not None)

    reward: float = Field(0.0, description="Reward received from the last action")
    done: bool = Field(False, description="Whether the episode has ended")
