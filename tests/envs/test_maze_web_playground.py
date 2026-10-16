# [MODULE DOC]
# Tests for the maze environment's web playground hooks (web_actions quick-move buttons and
# render_web board HTML). Skipped without numpy.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Maze drawing and move buttons in the web playground."""

import pytest

pytest.importorskip("numpy")

from maze_env.models import MazeAction
from maze_env.server.maze_env_environment import MazeEnvironment


def test_maze_web_playground():
    # [FUNCTION DOC]
    # After reset only "right" is offered, after moving right "left" and "right"; the rendered board has
    # an aria label, one agent circle, one exit shadow and no hex colours; render_web of an empty
    # observation returns None.
    env = MazeEnvironment()
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [("3 · right", {"action": 3})]
    obs = env.step(MazeAction(action=3)).model_dump()
    assert env.web_actions(obs) == [
        ("2 · left", {"action": 2}),
        ("3 · right", {"action": 3}),
    ]
    board = env.render_web(obs)
    assert 'aria-label="Maze board"' in board
    assert board.count("border-radius:50%") == 1  # the agent
    assert board.count("box-shadow") == 1  # the exit
    assert "#" not in board  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None
