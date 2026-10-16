# [MODULE DOC]
# Test of the snake environment's web playground hooks (move buttons and board HTML). Skipped
# without marlenv.
# SPDX-License-Identifier: BSD-3-Clause

"""The Snake board and moves in the web playground."""

import pytest

pytest.importorskip("marlenv")

from snake_env.models import SnakeAction
from snake_env.server.snake_environment import SnakeEnvironment


def test_snake_web_playground():
    # [FUNCTION DOC]
    # On a 10x10 board, web_actions offers straight/left/right; the rendered board after a step has an
    # aria label, 100 cells and no hex colours; a grid value 13 renders a rounded head; an empty
    # observation renders None.
    env = SnakeEnvironment(height=10, width=10)
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [
        ("0 · straight", {"action": 0}),
        ("1 · turn left", {"action": 1}),
        ("2 · turn right", {"action": 2}),
    ]
    obs = env.step(SnakeAction(action=0)).model_dump()
    board = env.render_web(obs)
    assert 'aria-label="Snake board"' in board
    assert board.count("<span") == 100
    assert "radius:4px" in env.render_web({"grid": [[13]]})  # a second snake's head
    assert "#" not in board  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None
