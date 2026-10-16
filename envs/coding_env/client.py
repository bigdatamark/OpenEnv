# [MODULE DOC]
# WebSocket EnvClient for the coding environment.
"""
CodingEnv
---------
Client-side wrapper for the Coding environment server.

This client maintains a persistent WebSocket connection to the environment
server, enabling efficient multi-step interactions with lower latency.

- users instantiate CodingEnv with a base_url provided by the higher-level
  vector/orchestration layer.
- Environment authors ship the Docker image that serves the API.

(Seeds, episode IDs, request IDs, capabilities can be added later in the payloads.)
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import CodeAction, CodeObservation, CodeState


class CodingEnv(EnvClient[CodeAction, CodeObservation, CodeState]):
    # [CLASS DOC]
    # Typed client sending code strings and parsing execution observations and state.
    # --- HTTPEnvClient abstract hooks ---

    def _step_payload(self, action: CodeAction) -> dict:
        # [FUNCTION DOC]
        # HELPER: Returns {"code": action.code}.
        # Shape expected by the server's /step endpoint under "action"
        return {
            "code": action.code,
        }

    def _parse_result(self, payload: dict) -> StepResult[CodeObservation]:
        # [FUNCTION DOC]
        # HELPER: Builds a CodeObservation from payload["observation"] and a StepResult with reward and done.
        # Raises: KeyError if "observation" is missing.
        # Expecting: { "observation": {...}, "reward": <float|null>, "done": <bool>, "info": {...} }
        obs = CodeObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> CodeState:
        # [FUNCTION DOC]
        # HELPER: Builds a CodeState with episode_id, step_count (default 0) and last_exit_code (default 0).
        """
        Parse server response into CodeState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            CodeState object with episode_id, step_count, and last_exit_code
        """
        return CodeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            last_exit_code=payload.get("last_exit_code", 0),
        )
