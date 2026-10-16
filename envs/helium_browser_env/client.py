# [MODULE DOC]
# WebSocket client for the Helium browser environment.
"""OpenEnv client serialization for browser actions and observations."""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import BrowserAction, BrowserObservation, BrowserState


class BrowserClient(EnvClient[BrowserAction, BrowserObservation, BrowserState]):
    # [CLASS DOC]
    # EnvClient specialization implementing the serialization hooks for browser
    # actions, observations and state.
    def _step_payload(self, action):
        # [FUNCTION DOC]
        # Returns the action serialized with pydantic model_dump.
        return action.model_dump()

    def _parse_result(self, payload):
        # [FUNCTION DOC]
        # Validates payload["observation"] into a BrowserObservation, computes done as
        # the top-level done OR the observation's done, and returns a StepResult with a
        # copy of the observation whose done is overwritten, top-level reward and done.
        # Raises: KeyError if "observation" is missing; pydantic ValidationError.
        observation = BrowserObservation.model_validate(payload["observation"])
        done = bool(payload.get("done", False) or observation.done)
        return StepResult(
            observation=observation.model_copy(update={"done": done}),
            reward=payload.get("reward"),
            done=done,
        )

    def _parse_state(self, payload):
        # [FUNCTION DOC]
        # Validates the payload into a BrowserState.
        return BrowserState.model_validate(payload)
