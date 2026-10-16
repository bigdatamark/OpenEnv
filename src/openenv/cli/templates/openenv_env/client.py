# [MODULE DOC]
# Template client for a generated echo-style environment: an EnvClient subclass
# that converts actions to JSON payloads and parses step results and state from
# server responses.
# SPDX-License-Identifier: BSD-3-Clause

"""__ENV_TITLE_NAME__ Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation


class __ENV_CLASS_NAME__Env(
    EnvClient[__ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation, State]
):
    # [CLASS DOC]
    # Typed EnvClient for the generated environment (action, observation, State).
    # Connection and session handling are inherited from EnvClient.
    # KNOWN ISSUE: docstring examples use ">>>" prompts, which the project's
    # doc-builder style disallows.
    """
    Client for the __ENV_TITLE_NAME__ Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server (.sync() for sync use)
        >>> with __ENV_CLASS_NAME__Env(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(__ENV_CLASS_NAME__Action(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect (.sync() for sync use)
        >>> client = __ENV_CLASS_NAME__Env.from_docker_image("__ENV_IMAGE_NAME__:latest").sync()
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(__ENV_CLASS_NAME__Action(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: __ENV_CLASS_NAME__Action) -> Dict:
        # [FUNCTION DOC]
        # Returns a dict with the action's "message" field for the step message.
        """
        Convert __ENV_CLASS_NAME__Action to JSON payload for step message.

        Args:
            action: __ENV_CLASS_NAME__Action instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[__ENV_CLASS_NAME__Observation]:
        # [FUNCTION DOC]
        # Builds a StepResult from a server payload: the observation is built from
        # payload["observation"] (echoed_message default "", message_length default 0)
        # with done/reward from the top-level payload and metadata from the top-level
        # payload, falling back to the observation's metadata. StepResult carries the
        # top-level reward, done (default False) and metadata (may be None).
        """
        Parse server response into StepResult[__ENV_CLASS_NAME__Observation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with __ENV_CLASS_NAME__Observation
        """
        obs_data = payload.get("observation", {})
        observation = __ENV_CLASS_NAME__Observation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
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
        # Returns a State with episode_id (may be None) and step_count (default 0) taken
        # from the payload.
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
