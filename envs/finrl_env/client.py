# [MODULE DOC]
# WebSocket client for the FinRL environment: translates FinRLAction into step
# payloads and server JSON responses into typed observations and state.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FinRL Environment Client.

This module provides the client for connecting to a FinRL Environment server
via WebSocket for persistent sessions.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import FinRLAction, FinRLObservation


class FinRLEnv(EnvClient[FinRLAction, FinRLObservation, State]):
    # [CLASS DOC]
    # EnvClient specialization for FinRL. Only implements the three serialization
    # hooks; connection handling, reset/step/close come from EnvClient.
    """
    Client for the FinRL Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions for stock trading RL tasks.

    Example:
        >>> # Connect to a running server
        >>> with FinRLEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.state)
        ...     print(result.observation.portfolio_value)
        ...
        ...     # Execute a trading action
        ...     action = FinRLAction(actions=[0.5, -0.3])  # Buy stock 0, sell stock 1
        ...     result = client.step(action)
        ...     print(result.reward)
        ...     print(result.observation.portfolio_value)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = FinRLEnv.from_docker_image("finrl-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(FinRLAction(actions=[0.1]))
        ... finally:
        ...     client.close()

    Example training loop:
        >>> import numpy as np
        >>> from envs.finrl_env import FinRLEnv, FinRLAction
        >>>
        >>> with FinRLEnv(base_url="http://localhost:8000") as client:
        ...     # Training loop
        ...     for episode in range(10):
        ...         result = client.reset()
        ...         done = False
        ...         episode_reward = 0
        ...
        ...         while not done:
        ...             # Get state
        ...             state = result.observation.state
        ...
        ...             # Simple random policy (replace with your RL agent)
        ...             num_stocks = len(state) // 7  # Simplified calculation
        ...             actions = np.random.uniform(-1, 1, size=num_stocks).tolist()
        ...
        ...             # Execute action
        ...             result = client.step(FinRLAction(actions=actions))
        ...
        ...             episode_reward += result.reward or 0
        ...             done = result.done
        ...
        ...         print(f"Episode {episode}: reward={episode_reward:.2f}, "
        ...               f"final value={result.observation.portfolio_value:.2f}")
    """

    def _step_payload(self, action: FinRLAction) -> Dict:
        # [FUNCTION DOC]
        # Builds the step payload: a dict with the single key "actions" holding the
        # action's list of floats.
        """
        Convert FinRLAction to JSON payload for step request.

        Args:
            action: FinRLAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "actions": action.actions,
        }

    def _parse_result(self, payload: Dict) -> StepResult[FinRLObservation]:
        # [FUNCTION DOC]
        # Converts a server step/reset response into a StepResult. Reads the nested
        # "observation" dict for state (default empty list), portfolio_value (default
        # 0.0) and date (default ""); done and reward come from the top level; metadata
        # from top level, falling back to observation metadata, then empty dict.
        # Returns a StepResult carrying the observation, reward, done and top-level
        # metadata (may be None).
        """
        Parse server response into StepResult[FinRLObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with FinRLObservation
        """
        obs_data = payload.get("observation", {})
        observation = FinRLObservation(
            state=obs_data.get("state", []),
            portfolio_value=obs_data.get("portfolio_value", 0.0),
            date=obs_data.get("date", ""),
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

    def _parse_state(self, payload: Dict) -> State:
        # [FUNCTION DOC]
        # Builds a State from the payload's episode_id (may be None) and step_count
        # (default 0).
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
