# [MODULE DOC]
# Pydantic models for TextArena games: a message, the action, the observation and the state.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the TextArena Environment.

This module defines the action, observation, and state models for interacting
with TextArena game environments (e.g., Wordle-v0).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class TextArenaMessage(BaseModel):
    # [CLASS DOC]
    # One message seen by a player: sender id (-1 for the game), content text and category name (e.g.
    # PROMPT, MESSAGE).
    """Single message observed by a player."""

    sender_id: int
    content: str
    category: str


class TextArenaAction(Action):
    # [CLASS DOC]
    # Action carrying the agent's raw text move in "message" (e.g. a bracketed Wordle guess).
    """Action issued by the agent for TextArena games."""

    message: str


class TextArenaObservation(Observation):
    # [CLASS DOC]
    # Observation with the prompt (game instructions), converted messages, current player id, sorted legal
    # player ids and an info mapping (TextArena step info plus any reward_signals).
    """Observation returned from any TextArena game."""

    prompt: str
    messages: List[TextArenaMessage] = Field(default_factory=list)
    current_player_id: int = 0
    legal_players: List[int] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)


class TextArenaState(State):
    # [CLASS DOC]
    # Server state: episode id, step count, TextArena env id, number of players, optional max turns, current
    # turn, last reward, last info and a raw snapshot of the underlying TextArena state.
    """Structured state snapshot for the server."""

    episode_id: Optional[str] = None
    step_count: int = 0
    env_id: str
    num_players: int
    max_turns: Optional[int] = None
    turn: int = 0
    last_reward: float = 0.0
    last_info: Dict[str, Any] = Field(default_factory=dict)
    raw_state: Dict[str, Any] = Field(default_factory=dict)
