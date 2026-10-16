# [MODULE DOC]
# Server-side Grid World environment: a 5x5 grid, agent starts at (0, 0), goal
# at (4, 4). Also provides optional web playground hooks (move buttons and an
# HTML board rendering).
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server.types import State
except ImportError:
    from core.env_server import Environment
    from core.env_server.types import State
from ..models import GridWorldAction, GridWorldObservation, MoveAction


class GridWorldEnvironment(Environment):
    # [CLASS DOC]
    # Environment with agent coordinates (agent_x is the row, agent_y the column),
    # fixed grid size 5 and goal [4, 4]. Supports concurrent sessions (each session
    # gets its own instance).
    """
    A simple 5x5 Grid World environment.

    The agent starts at [0, 0] and must navigate to [4, 4].
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # [FUNCTION DOC]
        # Sets grid size, goal, agent at (0, 0) and a State with a new UUID and step 0.
        super().__init__()

        # ---Environment Configuration ---
        self.grid_size = 5
        self.goal_pos = [4, 4]

        # --- Internal State Variable ---
        self.agent_x = 0
        self.agent_y = 0

        # Initialize the base OpenEnv State Container
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self) -> GridWorldObservation:
        # [FUNCTION DOC]
        # Moves the agent back to (0, 0), resets step_count and assigns a new episode
        # UUID. Returns the welcome observation with reward 0.0 and done False.
        # Update State
        self.agent_x = 0
        self.agent_y = 0

        self._state.step_count = 0
        self._state.episode_id = str(uuid.uuid4())
        # Return initial observation (reward must be float 0.0, not None)
        return GridWorldObservation(
            x=self.agent_x,
            y=self.agent_y,
            message="Welcome to Grid World! Goal is at [4, 4].",
            reward=0.0,
            done=False,
        )

    def step(self, action: GridWorldAction) -> GridWorldObservation:
        # [FUNCTION DOC]
        # Increments step_count, applies the move (UP/DOWN change agent_x i.e. row,
        # LEFT/RIGHT change agent_y i.e. column), clamps to the grid, and returns an
        # observation. Reward is -0.1 per step and 1.0 with done True on reaching the
        # goal. Episodes never end otherwise (no step limit).
        # Increment step counter in the base State
        self._state.step_count += 1
        # =============================================
        move = action.action

        # self._state.episode_steps += 1

        # Use current state
        # current_x = self._state.agent_x
        # current_y = self._state.agent_y

        move = action.action

        if move == MoveAction.UP:
            self.agent_x -= 1
        elif move == MoveAction.DOWN:
            self.agent_x += 1
        elif move == MoveAction.LEFT:
            self.agent_y -= 1
        elif move == MoveAction.RIGHT:
            self.agent_y += 1

        # Clamp to boundaries
        self.agent_x = max(0, min(self.agent_x, self.grid_size - 1))
        self.agent_y = max(0, min(self.agent_y, self.grid_size - 1))

        # # Update State
        # self._state.agent_x = current_x
        # self._state.agent_y = current_y

        # Logic
        done = False
        message = "Keep going..."
        reward = -0.1

        if [self.agent_x, self.agent_y] == self.goal_pos:
            reward = 1.0
            done = True
            message = "You found the goal!"

        return GridWorldObservation(
            x=self.agent_x, y=self.agent_y, message=message, reward=reward, done=done
        )

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        # [FUNCTION DOC]
        # Web playground hook: returns one (label, action-dict) button per MoveAction,
        # label being an arrow plus the lowercase move name, action dict {"action": move}.
        # The observation argument is unused.
        """The four moves as buttons."""
        arrows = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
        return [
            (f"{arrows[m.value]} {m.value.lower()}", {"action": m.value})
            for m in MoveAction
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        # [FUNCTION DOC]
        # Web playground hook: returns an HTML string drawing the grid as styled spans,
        # marking the agent cell with a dot and the goal with a star (agent wins if both).
        # Returns None if the observation dict has no "x" key.
        """Draw the grid with the agent (a dot) and the goal (a star)."""
        if "x" not in observation:
            return None
        cells = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                agent = [row, col] == [observation["x"], observation["y"]]
                goal = [row, col] == self.goal_pos
                border = "var(--color-accent)" if goal else "transparent"
                dot = ""
                if agent:
                    dot = '<span style="width:60%;height:60%;border-radius:50%;background:var(--color-accent)"></span>'
                elif goal:
                    dot = "★"
                cells.append(
                    '<span style="display:flex;align-items:center;justify-content:center;'
                    "background:var(--border-color-primary);border-radius:6px;"
                    f'border:2px solid {border};color:var(--color-accent);font-size:20px">{dot}</span>'
                )
        return (
            '<div role="img" aria-label="Grid World board" style="display:inline-grid;'
            f"grid-template-columns:repeat({self.grid_size},40px);grid-auto-rows:40px;gap:4px;padding:10px;"
            'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
            + "".join(cells)
            + "</div>"
        )

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning the current State.
        return self._state
