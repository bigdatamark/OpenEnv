# [MODULE DOC]
# Typed OpenEnv client for the Sophistry-Bench Sprint environment. Handles serialisation of the
# advocacy action and reconstruction of the observation (including reward-component metadata that the
# HTTP layer strips).
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import AdvocacyAction, AdvocacyObservation


class SophistryBenchSprintEnv(EnvClient[AdvocacyAction, AdvocacyObservation, State]):
    # [CLASS DOC]
    # EnvClient subclass for AdvocacyAction / AdvocacyObservation / generic State, with a convenience
    # step_text method.
    """Typed client for the sophistry-bench sprint OpenEnv environment."""

    async def step_text(self, text: str) -> StepResult[AdvocacyObservation]:
        # [FUNCTION DOC]
        # Async convenience: wraps a raw argument string in an AdvocacyAction and awaits the base step.
        # Returns: StepResult with an AdvocacyObservation. Side effects: one network step on the session.
        """Convenience: submit a raw argument string as an AdvocacyAction.

        ``async`` so it returns a ``StepResult`` (not a coroutine) and is wrapped
        by ``.sync()`` like the base ``step``/``reset`` — call ``await
        env.step_text(...)``, or ``env.sync().step_text(...)`` on a sync client.
        """
        return await super().step(AdvocacyAction(text=text))

    def _step_payload(self, action: AdvocacyAction) -> dict:
        # [FUNCTION DOC]
        # Returns the action's pydantic dump (a mapping with "text" plus base Action fields).
        return action.model_dump()

    def _parse_result(self, data: dict) -> StepResult[AdvocacyObservation]:
        # [FUNCTION DOC]
        # Inputs: data - response envelope with "observation", "reward", "done".
        # Raises ValueError if "observation" is missing or not a mapping. Removes any wire "metadata" from the
        # observation; rebuilds metadata as that wire metadata if non-empty, else a copy of "components"; adds
        # the "error" string under "error" if present and not already set. Copies envelope reward and done
        # (bool, default False) onto the observation.
        # Returns: StepResult(observation, reward, done) (no metadata on the StepResult).
        # Fail loudly on a malformed payload rather than silently building an
        # empty observation (a missing/null ``observation`` is a real protocol error).
        observation_payload = data.get("observation")
        if not isinstance(observation_payload, dict):
            raise ValueError(
                "malformed step result: 'observation' must be a dict, got "
                f"{type(observation_payload).__name__}"
            )
        obs_data = dict(observation_payload)
        # The framework's HTTP layer strips the base ``metadata`` dict from the
        # serialized observation, so the reward components arrive in the declared
        # ``components`` field (and the diagnostic message in ``error``). Rebuild
        # ``metadata`` here so the public contract holds — ``observation.metadata``
        # carries the eight components — preferring any metadata that survived
        # (in-process callers), else the mirrored ``components``.
        wire_metadata = obs_data.pop("metadata", None)
        metadata = (
            dict(wire_metadata)
            if wire_metadata
            else dict(obs_data.get("components") or {})
        )
        error = obs_data.get("error") or ""
        if error and "error" not in metadata:
            metadata["error"] = error
        # ``serialize_observation`` keeps reward and done on the envelope, not in
        # the observation dict, so set them on the observation too. Without this
        # ``result.observation.reward`` stayed None while ``result.reward`` was
        # correct, which traps callers that read the observation.
        reward = data.get("reward")
        done = bool(data.get("done", False))
        # Construct once with metadata set, rather than mutating the model after.
        observation = AdvocacyObservation(
            **obs_data, metadata=metadata, reward=reward, done=done
        )
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, data: dict) -> State:
        # [FUNCTION DOC]
        # Returns a State constructed from all keys of the payload.
        return State(**data)
