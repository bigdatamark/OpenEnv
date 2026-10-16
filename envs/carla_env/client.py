# [MODULE DOC]
# WebSocket EnvClient for the CARLA environment server.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Client for CARLA environment.

Provides EnvClient wrapper for remote or local CARLA instances.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from .models import CarlaAction, CarlaObservation, CarlaState


class CarlaEnv(EnvClient[CarlaAction, CarlaObservation, CarlaState]):
    # [CLASS DOC]
    # Typed EnvClient that serializes CarlaAction, and parses CarlaObservation and CarlaState.
    # reset accepts extra keyword args (scenario_name, scenario_config) forwarded by the base class.
    """
    Client for CARLA environment.

    Connects to a running CARLA environment server via WebSocket.

    Example:
        >>> from carla_env import CarlaEnv, CarlaAction
        >>> env = CarlaEnv(base_url="http://localhost:8000")
        >>> result = env.reset()
        >>> print(result.observation.scene_description)
        >>> result = env.step(CarlaAction(action_type="emergency_stop"))
        >>> env.close()

    Override scenario config at reset time (no new scenario name needed):
        >>> result = env.reset(scenario_config={"weather": "HardRainNoon", "max_steps": 100})

    Switch scenario AND override config:
        >>> result = env.reset(
        ...     scenario_name="free_roam_Town05",
        ...     scenario_config={"num_npc_vehicles": 30, "route_distance_max": 300.0},
        ... )

    For async usage:
        >>> async with CarlaEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()
        ...     result = await env.step(CarlaAction(action_type="observe"))

    Launch from Docker (delegates to base class):
        >>> env = await CarlaEnv.from_docker_image(
        ...     "carla-env:latest",
        ...     environment={"CARLA_SCENARIO": "trolley_saves", "CARLA_MODE": "mock"},
        ... )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        **kwargs: Any,
    ):
        # [FUNCTION DOC]
        # Passes base_url (default http://localhost:8000) and other kwargs to EnvClient.
        """
        Initialize CARLA environment client.

        Args:
            base_url: Base URL of the CARLA environment server
            **kwargs: Additional arguments for EnvClient
        """
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: CarlaAction) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Returns the action's model_dump dict as the step payload.
        """Convert CarlaAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CarlaObservation]:
        # [FUNCTION DOC]
        # HELPER: Builds a StepResult from a response dict: CarlaObservation from "observation",
        # reward from "reward" (may be None), done from "done" (default False).
        # Raises: KeyError if "observation" is missing; pydantic ValidationError for bad fields.
        """Parse JSON response to StepResult."""
        observation = CarlaObservation(**payload["observation"])
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CarlaState:
        # [FUNCTION DOC]
        # HELPER: Builds a CarlaState from the state payload dict.
        """Parse JSON response to CarlaState."""
        return CarlaState(**payload)
