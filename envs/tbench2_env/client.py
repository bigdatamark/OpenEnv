# [MODULE DOC]
# Client for the TB2 environment server: serialises Tbench2Action and parses observation/state payloads.
# Supports in-repo and standalone imports.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TB2 Environment Client."""

from __future__ import annotations

from typing import Any


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import Tbench2Action, Tbench2Observation, Tbench2State
except ImportError:
    from models import Tbench2Action, Tbench2Observation, Tbench2State

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class Tbench2Env(EnvClient[Tbench2Action, Tbench2Observation, Tbench2State]):
    # [CLASS DOC]
    # EnvClient for TB2; implements only the serialisation hooks.
    """HTTP client for the TB2 environment."""

    def _step_payload(self, action: Tbench2Action) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a mapping with all action fields: action_type, command, session_id, block, wait_seconds,
        # file_path, content.
        return {
            "action_type": action.action_type,
            "command": action.command,
            "session_id": action.session_id,
            "block": action.block,
            "wait_seconds": action.wait_seconds,
            "file_path": action.file_path,
            "content": action.content,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Tbench2Observation]:
        # [FUNCTION DOC]
        # Builds a Tbench2Observation from the payload's "observation" sub-mapping with defaults (success True,
        # empty strings, empty info), reward/done from the envelope, metadata from envelope or observation.
        # Returns: StepResult with observation, reward, done (default False) and envelope metadata.
        obs_data = payload.get("observation", {})
        observation = Tbench2Observation(
            instruction=obs_data.get("instruction", ""),
            output=obs_data.get("output", ""),
            success=obs_data.get("success", True),
            error=obs_data.get("error", ""),
            task_id=obs_data.get("task_id", ""),
            task_path=obs_data.get("task_path", ""),
            session_id=obs_data.get("session_id"),
            action_type=obs_data.get("action_type", ""),
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

    def _parse_state(self, payload: dict[str, Any]) -> Tbench2State:
        # [FUNCTION DOC]
        # Returns a Tbench2State from the payload with defaults.
        # KNOWN ISSUE: the server state's session_id field is not copied into the parsed state.
        return Tbench2State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_path=payload.get("task_path", ""),
            terminal_ready=payload.get("terminal_ready", False),
            last_action_type=payload.get("last_action_type", ""),
            last_command=payload.get("last_command", ""),
            last_output=payload.get("last_output", ""),
        )
