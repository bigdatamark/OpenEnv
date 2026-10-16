# [MODULE DOC]
# Tests for openspiel_env.harness: render_tic_tac_toe_board and OpenSpielSessionFactory (initial
# prompt, play_move tool mapping to env steps, client closing, reset argument forwarding) using an
# in-memory fake client.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OpenSpiel harness session adapter."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openspiel_env.harness import OpenSpielSessionFactory, render_tic_tac_toe_board
from openspiel_env.models import OpenSpielAction, OpenSpielObservation, OpenSpielState


class FakeOpenSpielClient:
    # [CLASS DOC]
    # HELPER: In-memory OpenSpiel client recording reset kwargs and step actions; any first step ends the
    # game with reward 1.0 if winning else 0.0.
    """In-memory OpenSpiel client that plays a scripted TTT game."""

    def __init__(self, winning: bool = True):
        # [FUNCTION DOC]
        # HELPER: Stores whether the game is won and initialises call logs and closed flag.
        self._winning = winning
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[OpenSpielAction] = []
        self.closed = False

    def reset(self, **kwargs: Any) -> StepResult[OpenSpielObservation]:
        # [FUNCTION DOC]
        # HELPER: Records kwargs and returns an empty-board playing observation with all nine legal actions.
        self.reset_calls.append(kwargs)
        return StepResult(
            observation=OpenSpielObservation(
                info_state=[1.0] * 9 + [0.0] * 18,
                legal_actions=list(range(9)),
                game_phase="playing",
                current_player_id=0,
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: OpenSpielAction) -> StepResult[OpenSpielObservation]:
        # [FUNCTION DOC]
        # HELPER: Records the action and returns a terminal observation (done after the first step) with
        # reward depending on the winning flag.
        self.step_calls.append(action)
        done = len(self.step_calls) >= 1
        reward = 1.0 if (done and self._winning) else 0.0
        info_state = (
            [0.0] * 9 + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 9
        )
        return StepResult(
            observation=OpenSpielObservation(
                info_state=info_state,
                legal_actions=[] if done else list(range(1, 9)),
                game_phase="terminal" if done else "playing",
                current_player_id=0,
            ),
            reward=reward,
            done=done,
        )

    def state(self) -> OpenSpielState:
        # [FUNCTION DOC]
        # HELPER: Returns state with the number of steps and game tic_tac_toe.
        return OpenSpielState(
            step_count=len(self.step_calls),
            game_name="tic_tac_toe",
        )

    def close(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Marks the client closed.
        self.closed = True


class TestRenderTicTacToeBoard:
    # [CLASS DOC]
    # Board rendering tests.
    def test_renders_empty_board_with_action_ids(self):
        # [FUNCTION DOC]
        # An empty board shows action ids 0..8.
        empty = [1.0] * 9 + [0.0] * 18
        board = render_tic_tac_toe_board(empty)
        assert "0" in board and "8" in board

    def test_renders_x_and_o_pieces(self):
        # [FUNCTION DOC]
        # X and O planes in info_state render as X and O.
        info_state = list([0.0] * 27)
        # cell 0 -> X
        info_state[9 + 0] = 1.0
        # cell 4 -> O
        info_state[18 + 4] = 1.0
        # remaining empties
        for idx in (1, 2, 3, 5, 6, 7, 8):
            info_state[idx] = 1.0
        board = render_tic_tac_toe_board(info_state)
        assert "X" in board
        assert "O" in board

    def test_empty_string_when_info_state_wrong_shape(self):
        # [FUNCTION DOC]
        # A wrong-length info_state renders as the empty string.
        assert render_tic_tac_toe_board([0.0] * 5) == ""


class TestOpenSpielSessionFactory:
    # [CLASS DOC]
    # Session factory tests.
    def test_initial_message_includes_legal_actions(self):
        # [FUNCTION DOC]
        # The single initial message names the game, lists legal actions and includes a board.
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client, game_name="tic_tac_toe")

        session = factory.create(episode_id="ttt-000000")
        try:
            initial = session.initial_messages()
        finally:
            session.close()

        assert len(initial) == 1
        content = initial[0]["content"]
        assert "tic_tac_toe" in content
        assert "Legal actions: [0, 1, 2, 3, 4, 5, 6, 7, 8]" in content
        assert "Board:" in content

    def test_play_move_tool_call_invokes_step_with_action_id(self):
        # [FUNCTION DOC]
        # play_move with action_id 4 steps the client with that id and game name; the tool result is done
        # with reward 1.0 in metadata and a board in data.
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client)

        session = factory.create(episode_id="ttt-000001")
        try:
            tool_result = session.call_tool("play_move", {"action_id": 4})
        finally:
            session.close()

        assert client.step_calls[0].action_id == 4
        assert client.step_calls[0].game_name == "tic_tac_toe"
        assert tool_result.done is True
        assert tool_result.metadata["reward"] == 1.0
        assert "board" in tool_result.data

    def test_session_closes_underlying_client(self):
        # [FUNCTION DOC]
        # Closing the session closes the client.
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client)

        session = factory.create()
        session.close()

        assert client.closed is True

    def test_reset_forwards_seed_and_episode_id(self):
        # [FUNCTION DOC]
        # create forwards seed and episode_id to the client's reset.
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client)

        factory.create(seed=7, episode_id="ttt-000002")

        assert client.reset_calls[0]["seed"] == 7
        assert client.reset_calls[0]["episode_id"] == "ttt-000002"
