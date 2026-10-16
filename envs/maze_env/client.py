# [MODULE DOC]
# Client for the maze environment. Defines MazeEnv, an EnvClient subclass that serializes
# MazeAction into the step payload and parses server JSON into MazeObservation/MazeState.
# Imports models relatively when part of the package, falling back to a top-level "models"
# import when run standalone.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maze Environment Client.

This module provides the client for connecting to a Maze Environment server
via WebSocket for persistent sessions.
"""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

# Support both in-repo and standalone imports
try:
    from .models import MazeAction, MazeObservation, MazeState
except ImportError:
    from models import MazeAction, MazeObservation, MazeState


class MazeEnv(EnvClient[MazeAction, MazeObservation, MazeState]):
    # [CLASS DOC]
    # WebSocket client for one maze session. Implements the three EnvClient hooks: build the step
    # payload, parse a step/reset response, and parse a state response. All transport is inherited.
    """
    Client for the Maze Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: MazeAction) -> Dict:
        # [FUNCTION DOC]
        # HELPER: Builds the step message body.
        # Inputs: action - a MazeAction. Returns: a mapping with a single key "action" holding the integer action index.
        """
        Convert MazeAction to JSON payload for step message.

        Args:
            action: MazeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MazeObservation]:
        # [FUNCTION DOC]
        # HELPER: Converts a server step/reset response into a StepResult.
        # Inputs: payload - mapping with optional "observation" (holding legal_actions, current_position,
        # previous_position, metadata), "reward", "done", "metadata".
        # Returns: StepResult wrapping a MazeObservation; missing lists default to empty, done to False,
        # reward to None; observation metadata prefers the top-level "metadata" over the nested one.
        """
        Parse server response into StepResult[MazeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MazeObservation
        """
        obs_data = payload.get("observation", {})
        observation = MazeObservation(
            legal_actions=obs_data.get("legal_actions", []),
            current_position=obs_data.get("current_position", []),
            previous_position=obs_data.get("previous_position", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> MazeState:
        # [FUNCTION DOC]
        # HELPER: Converts a state response into a MazeState, defaulting step_count to 0, done to False,
        # position/exit lists to empty and status to "playing".
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return MazeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
            current_position=payload.get("current_position", []),
            exit_cell=payload.get("exit_cell", []),
            status=payload.get("status", "playing"),
        )
