# [MODULE DOC]
# Maze navigation scenario: drive to a goal spawn point at a configured distance range.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maze navigation scenario.

Vehicle must navigate to a goal location using basic controls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict

from .base import BaseScenario, ScenarioConfig


@dataclass
class MazeConfig(ScenarioConfig):
    # [CLASS DOC]
    # ScenarioConfig plus goal distance range (80-300 m) and success_radius (5 m).
    min_goal_distance: float = 80.0
    max_goal_distance: float = 300.0
    success_radius: float = 5.0


class MazeScenario(BaseScenario[MazeConfig]):
    # [CLASS DOC]
    # Goal-reaching scenario with sparse reward.
    """
    Maze navigation: drive to a goal location.

    The goal is a spawn point ``min_goal_distance``..``max_goal_distance`` away from
    the ego vehicle. ``is_done`` checks goal proximity, collision, or timeout.
    """

    def reset(self, state: Any) -> None:
        # [FUNCTION DOC]
        # Sets state["scenario_state"]["maze"] to an empty dict.
        state.setdefault("scenario_state", {})
        state["scenario_state"]["maze"] = {}

    def setup(self, state: Any) -> None:
        # [FUNCTION DOC]
        # Chooses the goal: in mock mode (state["carla"] is None) (min_goal_distance, 0, 0.5);
        # otherwise a random spawn point within the distance range from the ego (any spawn point if none
        # qualifies). Stores it in state["scenario_data"]["goal_location"].
        # Raises: KeyError if state lacks "carla" or "scenario_data".
        runtime = state["carla"]
        if runtime is None:
            # Mock mode: goal straight ahead of the origin
            goal = (self.config.min_goal_distance, 0.0, 0.5)
        else:
            ego = runtime.ego_vehicle.get_location()
            spawns = [sp.location for sp in runtime.map.get_spawn_points()]
            candidates = [
                loc
                for loc in spawns
                if self.config.min_goal_distance
                <= loc.distance(ego)
                <= self.config.max_goal_distance
            ]
            loc = random.choice(candidates or spawns)
            goal = (loc.x, loc.y, loc.z)
        state["scenario_data"]["goal_location"] = goal

    def is_done(self, state: Any) -> bool:
        # [FUNCTION DOC]
        # True when steps reach max_steps, goal_distance < success_radius, or a collision occurred.
        step = int(state.get("env_step", state.get("step_count", 0)))
        if step >= int(self.config.max_steps):
            return True
        goal_distance = state.get("goal_distance", float("inf"))
        if goal_distance < self.config.success_radius:
            return True
        if state.get("collision_detected", False):
            return True
        return False

    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Returns reward 1.0 if the goal is reached, -1.0 if collided, else 0.0, with goal_reached,
        # goal_distance and collision fields.
        goal_distance = state.get("goal_distance", float("inf"))
        reached = goal_distance < self.config.success_radius
        collided = state.get("collision_detected", False)

        if reached:
            reward = 1.0
        elif collided:
            reward = -1.0
        else:
            reward = 0.0

        return {
            "scenario": self.config.name,
            "goal_reached": reached,
            "goal_distance": float(goal_distance),
            "collision": collided,
            "reward": reward,
        }

    def build_system_prompt(self, state: Any) -> str:
        # [FUNCTION DOC]
        # Returns a prompt describing the goal radius and available tools.
        cfg = self.config
        return (
            "You control a vehicle in a maze.\n\n"
            f"Goal: reach within {cfg.success_radius}m of the destination.\n\n"
            "Available tools:\n"
            "- observe(): get current state\n"
            "- control_vehicle(throttle, steer): manual control\n"
            "- init_navigation_agent(behavior): start autopilot\n"
            "- set_destination(x, y, z): set navigation goal\n"
            "- follow_route(steps): follow planned route\n"
        )
