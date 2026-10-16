# [MODULE DOC]
# Two-player Connect4 environment on a 6x7 numpy board. Both players move through step (alternating 1 and
# -1); there is no built-in opponent.
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openenv.core.env_server import Environment

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import Connect4Action, Connect4Observation, Connect4State
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import Connect4Action, Connect4Observation, Connect4State


class Connect4Environment(Environment):
    # [CLASS DOC]
    # Connect4 game supporting concurrent sessions. Board values: 1 / -1 for players, 0 empty; row 0 is the top.
    ROWS = 6
    COLUMNS = 7
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, opponent=None):
        # [FUNCTION DOC]
        # Stores the opponent argument and resets.
        # KNOWN ISSUE: the opponent argument is never used.
        super().__init__()
        self._opponent = opponent
        self.reset()

    def reset(self):
        # [FUNCTION DOC]
        # Clears the board, sets player 1 to move, creates a new state with a fresh episode id, and returns the
        # initial observation (reward 0, not done).
        self.board = np.zeros((self.ROWS, self.COLUMNS), dtype=np.int8)
        self.next_player = 1
        self.invalid_move_played = False

        self._state = Connect4State(
            board=self.board.copy().tolist(),
            next_player=self.next_player,
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        return self._make_observation()

    def step(self, action: Connect4Action):
        # [FUNCTION DOC]
        # Drops the current player's piece in the column. Invalid columns (out of range or full) end the game with
        # reward -1. Otherwise reward 1.0 and done on a four-in-a-row for the mover, done with 0.0 on a full board,
        # else 0.0. Always switches player and increments step_count.
        # Returns: Connect4Observation (reward from the mover's perspective).
        # KNOWN ISSUE: steps are still accepted after the game is over.
        col = action.column
        # reward = 0.0
        done = False

        # check action validity
        if col < 0 or col >= self.COLUMNS or self.board[0, col] != 0:
            self.invalid_move_played = True
            reward = -1  # penalty for invalid move
            done = True
        else:
            # drop piece
            for row in range(self.ROWS - 1, -1, -1):
                if self.board[row, col] == 0:
                    self.board[row, col] = self.next_player
                    break

            # check win / full board
            reward, done = self._check_win_or_draw(row, col)

        self.next_player *= -1

        self._state = Connect4State(
            board=self.board.copy().tolist(),
            next_player=self.next_player,
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + 1,
        )

        return self._make_observation(reward, done)

    def _make_observation(self, reward=0.0, done=False):
        # [FUNCTION DOC]
        # HELPER: Builds the observation with a board copy, non-full columns as legal actions, reward, done and
        # metadata next_player.
        legal_actions = [c for c in range(self.COLUMNS) if self.board[0, c] == 0]
        return Connect4Observation(
            board=self.board.copy().tolist(),
            legal_actions=legal_actions,
            reward=reward,
            done=done,
            metadata={"next_player": self.next_player},
        )

    def _check_win_or_draw(self, row, col):
        # [FUNCTION DOC]
        # HELPER: Checks the 7-cell windows through (row, col) in four directions for 4 consecutive pieces of the
        # placed player. Returns (1.0, True) on a win, (0.0, True) when the board is full, else (0.0, False).
        # Implement 4-in-a-row check (like your Gymnasium code)
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            for step in range(-3, 4):
                r, c = row + step * dr, col + step * dc
                if (
                    0 <= r < self.ROWS
                    and 0 <= c < self.COLUMNS
                    and self.board[r, c] == player
                ):
                    count += 1
                    if count >= 4:
                        return 1.0, True
                else:
                    count = 0
        if np.all(self.board != 0):
            return 0.0, True
        return 0.0, False

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        # [FUNCTION DOC]
        # Web playground hook: one "col N" button per legal column.
        """The legal columns as buttons."""
        return [
            (f"col {c}", {"column": c}) for c in observation.get("legal_actions") or []
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        # [FUNCTION DOC]
        # Web playground hook: HTML grid of discs (player 1 accent color, player -1 text color, empty border color)
        # with column labels; None if the observation has no board.
        """Draw the 6x7 board: player 1 in the accent colour, player -1 in the text colour."""
        if "board" not in observation:
            return None
        colours = {
            1: "var(--color-accent)",
            -1: "var(--body-text-color)",
            0: "var(--border-color-primary)",
        }
        cells = [
            f'<span style="background:{colours[value]};border-radius:50%"></span>'
            for row in observation["board"]
            for value in row
        ]
        cells += [
            f'<span style="text-align:center;font-size:12px;line-height:26px">{c}</span>'
            for c in range(self.COLUMNS)
        ]
        return (
            '<div role="img" aria-label="Connect4 board" style="display:inline-grid;'
            "grid-template-columns:repeat(7,26px);grid-auto-rows:26px;gap:4px;padding:10px;"
            'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
            + "".join(cells)
            + "</div>"
        )

    @property
    def state(self):
        # [FUNCTION DOC]
        # Property: returns the Connect4State.
        return self._state
