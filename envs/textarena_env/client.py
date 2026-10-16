# [MODULE DOC]
# Client for the TextArena environment server: serialises TextArenaAction and parses observation/state
# payloads.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TextArena Environment HTTP Client.

This module provides the client for connecting to a TextArena Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    TextArenaAction,
    TextArenaMessage,
    TextArenaObservation,
    TextArenaState,
)


class TextArenaEnv(EnvClient[TextArenaAction, TextArenaObservation, TextArenaState]):
    # [CLASS DOC]
    # EnvClient for TextArena; implements only the serialisation hooks.
    """
    HTTP client for the TextArena Environment.

    This client connects to a TextArenaEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = TextArenaEnv(base_url="<ENV_HTTP_URL_HERE>")
        >>> result = client.reset()
        >>> print(result.observation.prompt)
        >>>
        >>> # Send an action
        >>> result = client.step(TextArenaAction(message="[crane]"))
        >>> print(result.observation.messages)
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TextArenaEnv.from_docker_image("textarena-env:latest")
        >>> result = client.reset()
        >>> result = client.step(TextArenaAction(message="Test"))
    """

    def _step_payload(self, action: TextArenaAction) -> Dict:
        # [FUNCTION DOC]
        # Returns a mapping with the action's "message".
        """
        Convert TextArenaAction to JSON payload for step request.

        Args:
            action: TextArenaAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TextArenaObservation]:
        # [FUNCTION DOC]
        # Builds TextArenaMessage objects from the observation's "messages" (non-mapping items skipped; defaults
        # sender -1, empty content, category "MESSAGE") and a TextArenaObservation with prompt, current player,
        # legal players and info, reward/done from the envelope and metadata from envelope or observation.
        # Returns: StepResult with observation, reward, done (default False) and envelope metadata.
        """
        Parse server response into StepResult[TextArenaObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with TextArenaObservation
        """
        obs_data = payload.get("observation", {})
        messages_payload = obs_data.get("messages", [])
        messages = [
            TextArenaMessage(
                sender_id=item.get("sender_id", -1),
                content=item.get("content", ""),
                category=item.get("category", "MESSAGE"),
            )
            for item in messages_payload
            if isinstance(item, dict)
        ]

        observation = TextArenaObservation(
            prompt=obs_data.get("prompt", ""),
            messages=messages,
            current_player_id=obs_data.get("current_player_id", 0),
            legal_players=obs_data.get("legal_players", []),
            info=obs_data.get("info", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TextArenaState:
        # [FUNCTION DOC]
        # Returns a TextArenaState from the payload with defaults (env_id "unknown", 1 player, turn 0, etc.).
        return TextArenaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            env_id=payload.get("env_id", "unknown"),
            num_players=payload.get("num_players", 1),
            max_turns=payload.get("max_turns"),
            turn=payload.get("turn", 0),
            last_reward=payload.get("last_reward", 0.0),
            last_info=payload.get("last_info", {}),
            raw_state=payload.get("raw_state", {}),
        )
