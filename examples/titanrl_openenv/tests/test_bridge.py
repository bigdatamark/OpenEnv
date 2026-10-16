# [MODULE DOC]
# Pytest suite for the torch-free OpenEnv bridge, task profiles and chess launcher.
# Adds the test dir, package dir and repo root to sys.path at import. Unit tests drive
# OpenEnvBridge with a scripted stub client; integration-marked tests run live
# round-trips against an in-process tiny env and (if chess deps exist) the real chess env.
# Async tests rely on an async-capable pytest plugin/config.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TitanRL <-> OpenEnv bridge.

These exercise the OpenEnv side of the integration and need only ``openenv``
(no ``torch`` / ``torchtitan``):

* fast, deterministic unit tests that drive :class:`OpenEnvBridge` with a
  scripted stub client, covering observation rendering, tool-call normalization,
  text/tool action translation, and the chess task profile;
* ``integration``-marked live round-trips over the actual HTTP/WebSocket
  protocol against a tiny in-process env (always) and against OpenEnv's real
  ``envs/chess_env`` (skipped unless ``python-chess``/``moonfish`` are
  installed).
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
from types import SimpleNamespace

import pytest

# Make the example package importable as ``import openenv_bridge`` / ``import
# tasks`` (mirrors running with the example dir on PYTHONPATH), and this test
# directory importable as ``import tiny_env``.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TEST_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
for _path in (_TEST_DIR, _PKG_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from openenv.core.client_types import StepResult  # noqa: E402
from openenv_bridge import (  # noqa: E402
    BridgeTurn,
    DEFAULT_ACT_TOOL,
    normalize_tool_call,
    OpenEnvBridge,
    render_observation,
)
from tasks import (  # noqa: E402
    CHESS_EVAL_SCALE,
    CHESS_MOVE_TOOL,
    CHESS_POSITION_REWARD,
    chess_position_rewards,
    CHESS_TASK,
    get_task_profile,
    render_chess_observation,
)


# --------------------------------------------------------------------------- #
# Test double: an EnvClient-shaped stub that replays scripted StepResults.
# --------------------------------------------------------------------------- #


class ScriptedClient:
    # [CLASS DOC]
    # HELPER: duck-typed async EnvClient stub that pops canned StepResults in order for
    # reset and step, recording reset kwargs, step actions and whether close was called.
    """Duck-typed ``EnvClient`` that records calls and replays canned results."""

    def __init__(self, results: list[StepResult]) -> None:
        # [FUNCTION DOC]
        # Copies the canned results and initializes recording fields.
        self._results = list(results)
        self.actions: list[dict] = []
        self.reset_kwargs: dict | None = None
        self.closed = False

    async def reset(self, **kwargs):
        # [FUNCTION DOC]
        # Records kwargs and returns the next canned result. Raises IndexError when exhausted.
        self.reset_kwargs = kwargs
        return self._results.pop(0)

    async def step(self, action):
        # [FUNCTION DOC]
        # Records the action and returns the next canned result.
        self.actions.append(action)
        return self._results.pop(0)

    async def close(self):
        # [FUNCTION DOC]
        # Marks the stub closed.
        self.closed = True


def _obs(text=None, **extra):
    # [FUNCTION DOC]
    # HELPER: builds an observation dict with optional "text" plus any extra fields.
    payload = {}
    if text is not None:
        payload["text"] = text
    payload.update(extra)
    return payload


# --------------------------------------------------------------------------- #
# render_observation
# --------------------------------------------------------------------------- #


def test_render_observation_passthrough_string():
    # [FUNCTION DOC]
    # Checks that a string observation is returned unchanged.
    assert render_observation("hello") == "hello"


def test_render_observation_none_is_empty():
    # [FUNCTION DOC]
    # Checks that None renders to an empty string.
    assert render_observation(None) == ""


def test_render_observation_prefers_known_keys():
    # [FUNCTION DOC]
    # Checks that "text" wins over "message" and "message" is used when alone.
    assert render_observation({"text": "hi", "message": "no"}) == "hi"
    assert render_observation({"message": "hi"}) == "hi"


def test_render_observation_metadata_fallback():
    # [FUNCTION DOC]
    # Checks fallback to metadata["message"].
    assert render_observation({"metadata": {"message": "from-meta"}}) == "from-meta"


def test_render_observation_json_fallback_strips_control_fields():
    # [FUNCTION DOC]
    # Checks that the JSON fallback omits reward/done and keeps other fields.
    rendered = render_observation({"score": 3, "reward": 1.0, "done": True})
    assert "reward" not in rendered and "done" not in rendered
    assert '"score": 3' in rendered


# --------------------------------------------------------------------------- #
# normalize_tool_call
# --------------------------------------------------------------------------- #


def test_normalize_tool_call_object_like():
    # [FUNCTION DOC]
    # Checks normalization of an attribute-style tool call.
    tc = SimpleNamespace(name="openenv_act", arguments={"action": {"x": 1}})
    assert normalize_tool_call(tc) == ("openenv_act", {"action": {"x": 1}})


def test_normalize_tool_call_flat_dict():
    # [FUNCTION DOC]
    # Checks normalization of a flat dict tool call.
    tc = {"name": "act", "arguments": {"a": 1}}
    assert normalize_tool_call(tc) == ("act", {"a": 1})


def test_normalize_tool_call_openai_style_json_string_args():
    # [FUNCTION DOC]
    # Checks OpenAI-style tool call with JSON-string arguments is decoded.
    tc = {"function": {"name": "act", "arguments": '{"action": {"guess": "apple"}}'}}
    name, args = normalize_tool_call(tc)
    assert name == "act"
    assert args == {"action": {"guess": "apple"}}


def test_normalize_tool_call_bad_json_yields_empty_args():
    # [FUNCTION DOC]
    # Checks that undecodable JSON arguments yield an empty dict.
    tc = {"function": {"name": "act", "arguments": "not-json"}}
    assert normalize_tool_call(tc) == ("act", {})


# --------------------------------------------------------------------------- #
# OpenEnvBridge — configuration
# --------------------------------------------------------------------------- #


def test_invalid_action_mode_raises():
    # [FUNCTION DOC]
    # Checks that an invalid action_mode raises ValueError.
    with pytest.raises(ValueError):
        OpenEnvBridge(action_mode="nonsense")


def test_default_tool_schema_shape():
    # [FUNCTION DOC]
    # Checks the default act tool's name and that it has an "action" property.
    assert DEFAULT_ACT_TOOL["name"] == "openenv_act"
    assert "action" in DEFAULT_ACT_TOOL["parameters"]["properties"]


# --------------------------------------------------------------------------- #
# OpenEnvBridge — stepping (scripted stub client)
# --------------------------------------------------------------------------- #


async def test_start_returns_reset_turn_and_forwards_kwargs():
    # [FUNCTION DOC]
    # Checks that start forwards reset kwargs and returns a not-done turn with text and no reward.
    client = ScriptedClient([StepResult(observation=_obs("welcome"), reward=None)])
    bridge = OpenEnvBridge(client=client)
    turn = await bridge.start(seed=7)
    assert isinstance(turn, BridgeTurn)
    assert turn.text == "welcome"
    assert turn.reward is None and turn.done is False
    assert client.reset_kwargs == {"seed": 7}


async def test_text_mode_wraps_text_under_action_key():
    # [FUNCTION DOC]
    # Checks that text mode puts the text under the configured action key and returns
    # reward/done from the step result.
    client = ScriptedClient(
        [
            StepResult(observation=_obs("reset")),
            StepResult(observation=_obs("stepped"), reward=0.5, done=True),
        ]
    )
    bridge = OpenEnvBridge(client=client, action_mode="text", action_key="guess")
    await bridge.start()
    turn = await bridge.act_from_text("apple")
    assert client.actions == [{"guess": "apple"}]
    assert turn.text == "stepped"
    assert turn.reward == 0.5
    assert turn.done is True


async def test_tool_mode_extracts_action_from_tool_call():
    # [FUNCTION DOC]
    # Checks that tool mode unwraps the "action" argument into the env action.
    client = ScriptedClient(
        [
            StepResult(observation=_obs("reset")),
            StepResult(observation=_obs("ok"), reward=1.0, done=True),
        ]
    )
    bridge = OpenEnvBridge(
        client=client, action_mode="tool"
    )  # tool_action_key="action"
    await bridge.start()
    tool_calls = [
        {
            "function": {
                "name": "openenv_act",
                "arguments": '{"action": {"guess": "kiwi"}}',
            }
        }
    ]
    turn = await bridge.act_from_tool_calls(tool_calls)
    assert client.actions == [{"guess": "kiwi"}]
    assert turn.reward == 1.0 and turn.done is True


def test_action_from_tool_calls_without_wrapper_key_uses_whole_args():
    # [FUNCTION DOC]
    # Checks that tool arguments without the wrapper key are used whole as the action.
    bridge = OpenEnvBridge(action_mode="tool", tool_action_key="action")
    # Arguments with no "action" wrapper -> the whole arguments dict is the action.
    action = bridge.action_from_tool_calls(
        [{"name": "act", "arguments": {"guess": "x"}}]
    )
    assert action == {"guess": "x"}


def test_action_from_empty_tool_calls_is_empty():
    # [FUNCTION DOC]
    # Checks that an empty tool-call list yields an empty action.
    bridge = OpenEnvBridge(action_mode="tool")
    assert bridge.action_from_tool_calls([]) == {}


async def test_reward_and_done_fall_back_to_observation_fields():
    # [FUNCTION DOC]
    # Checks that reward/done fall back to fields in the observation dict.
    # StepResult carries no reward/done, but the observation dict does.
    client = ScriptedClient(
        [
            StepResult(observation=_obs("reset")),
            StepResult(observation=_obs("done-here", reward=2.0, done=True)),
        ]
    )
    bridge = OpenEnvBridge(client=client, action_mode="text")
    await bridge.start()
    turn = await bridge.act_from_text("go")
    assert turn.reward == 2.0
    assert turn.done is True


async def test_stop_closes_client_and_is_idempotent():
    # [FUNCTION DOC]
    # Checks that stop closes the client and can be called twice.
    client = ScriptedClient([StepResult(observation=_obs("reset"))])
    bridge = OpenEnvBridge(client=client)
    await bridge.start()
    await bridge.stop()
    await bridge.stop()
    assert client.closed is True


def test_bridge_turn_as_message():
    # [FUNCTION DOC]
    # Checks BridgeTurn.as_message output.
    turn = BridgeTurn(text="hi", done=False, reward=None)
    assert turn.as_message("tool") == {"role": "tool", "content": "hi"}


async def test_custom_observation_renderer_overrides_default():
    # [FUNCTION DOC]
    # Checks that a custom observation renderer is used instead of the default.
    client = ScriptedClient([StepResult(observation=_obs("ignored", score=3))])
    bridge = OpenEnvBridge(
        client=client,
        observation_renderer=lambda obs: f"score={obs['score']}",
    )
    turn = await bridge.start()
    assert turn.text == "score=3"


async def test_bridge_tells_the_chess_renderer_that_a_lost_game_is_over():
    # [FUNCTION DOC]
    # Checks that a lost chess game (reward -1, done) renders as game over rather than a
    # rejected move, and that shaping on the merged observation gives -1.0.
    # The wire format moves reward/done onto the StepResult; the bridge must put
    # them back for the renderer, or a lost game (-1.0) reads as a rejected move.
    start = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "legal_moves": ["f2f3", "e2e4"],
        "is_check": False,
        "result": None,
    }
    mated = {
        "fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "legal_moves": [],
        "is_check": True,
        "result": "0-1",
        "metadata": {"evaluation": 150.0},
    }
    client = ScriptedClient(
        [
            StepResult(observation=start),
            StepResult(observation=mated, reward=-1.0, done=True),
        ]
    )
    bridge = OpenEnvBridge(
        client=client,
        tool_action_key=CHESS_TASK.tool_action_key,
        observation_renderer=CHESS_TASK.render,
    )
    await bridge.start()
    call = {"name": "chess_move", "arguments": {"move": "g2g4"}}
    turn = await bridge.act_from_tool_calls([call])
    assert turn.done and turn.reward == -1.0
    assert "rejected" not in turn.text
    assert "Game over" in turn.text
    # Given the same view the TitanRL env builds, shaping scores the loss.
    merged = {**turn.raw_observation, "reward": turn.reward, "done": turn.done}
    assert chess_position_rewards(merged) == {CHESS_POSITION_REWARD: -1.0}


# --------------------------------------------------------------------------- #
# Task profiles
# --------------------------------------------------------------------------- #


def test_get_task_profile_known_and_unknown():
    # [FUNCTION DOC]
    # Checks profile lookup for chess and generic and ValueError for unknown names.
    assert get_task_profile("chess") is CHESS_TASK
    assert get_task_profile("generic").tool is DEFAULT_ACT_TOOL
    with pytest.raises(ValueError):
        get_task_profile("nope")


def test_chess_tool_schema_shape():
    # [FUNCTION DOC]
    # Checks the chess move tool's name and required arguments.
    assert CHESS_MOVE_TOOL["name"] == "chess_move"
    assert CHESS_MOVE_TOOL["parameters"]["required"] == ["move"]


def test_chess_profile_maps_tool_args_straight_to_action():
    # [FUNCTION DOC]
    # Checks that the chess profile uses tool arguments directly as the action.
    """``chess_move(move=...)`` is already a ChessAction — no wrapper key."""
    bridge = OpenEnvBridge(
        action_mode="tool", tool_action_key=CHESS_TASK.tool_action_key
    )
    tool_calls = [
        {"function": {"name": "chess_move", "arguments": json.dumps({"move": "e2e4"})}}
    ]
    assert bridge.action_from_tool_calls(tool_calls) == {"move": "e2e4"}


def _chess_obs(**overrides):
    # [FUNCTION DOC]
    # HELPER: returns a starting-position chess observation dict with overrides applied.
    obs = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "legal_moves": ["e2e4", "d2d4", "g1f3"],
        "is_check": False,
        "result": None,
        "done": False,
        "reward": 0.0,
    }
    obs.update(overrides)
    return obs


def test_render_chess_observation_includes_position_and_moves():
    # [FUNCTION DOC]
    # Checks that the chess renderer shows FEN, side to move and legal moves, with no rejection note.
    rendered = render_chess_observation(_chess_obs())
    assert "Position (FEN): rnbqkbnr" in rendered
    assert "Side to move: white" in rendered
    assert "Legal moves (3): e2e4, d2d4, g1f3" in rendered
    assert "rejected" not in rendered


def test_render_chess_observation_flags_rejected_move():
    # [FUNCTION DOC]
    # Checks that a mid-game negative reward produces the rejection note.
    # The env answers an illegal move with -0.1 rather than an error, so the
    # renderer has to say so or the model never learns what happened.
    rendered = render_chess_observation(_chess_obs(reward=-0.1))
    assert "rejected" in rendered
    assert "Legal moves" in rendered


def test_render_chess_observation_does_not_call_a_lost_game_a_rejected_move():
    # [FUNCTION DOC]
    # Checks that a lost game is reported as over and not as a rejected move.
    # A loss pays -1.0 and ends the game; only a negative reward mid-game means
    # the move was rejected.
    rendered = render_chess_observation(
        _chess_obs(reward=-1.0, done=True, result="0-1")
    )
    assert "rejected" not in rendered
    assert "Game over" in rendered


def test_render_chess_observation_reports_check_and_result():
    # [FUNCTION DOC]
    # Checks check and result lines in the chess rendering.
    rendered = render_chess_observation(
        _chess_obs(is_check=True, done=True, result="1-0", reward=1.0)
    )
    assert "in check" in rendered
    assert "Game over — result 1-0." in rendered


def test_render_chess_observation_falls_back_for_other_envs():
    # [FUNCTION DOC]
    # Checks that non-chess observations fall back to generic rendering.
    # A non-chess observation must not be mangled by the chess renderer.
    assert render_chess_observation({"text": "hello"}) == "hello"


# --------------------------------------------------------------------------- #
# chess shaping reward (tasks.chess_position_rewards)
# --------------------------------------------------------------------------- #


def test_chess_position_reward_is_signed_and_bounded():
    # [FUNCTION DOC]
    # Checks that the position reward is tanh-scaled, sign-preserving and within (-1, 1).
    # moonfish scores from the side to move's point of view, and the agent is
    # always the side to move, so a positive evaluation must stay positive.
    ahead = chess_position_rewards(_chess_obs(metadata={"evaluation": 300.0}))
    behind = chess_position_rewards(_chess_obs(metadata={"evaluation": -300.0}))
    assert ahead[CHESS_POSITION_REWARD] == pytest.approx(
        math.tanh(300 / CHESS_EVAL_SCALE)
    )
    assert behind[CHESS_POSITION_REWARD] == pytest.approx(-ahead[CHESS_POSITION_REWARD])
    assert -1.0 < behind[CHESS_POSITION_REWARD] < 0 < ahead[CHESS_POSITION_REWARD] < 1.0


@pytest.mark.parametrize(
    ("outcome", "expected"), [(1.0, 1.0), (0.0, 0.0), (-1.0, -1.0)]
)
def test_chess_position_reward_scores_a_finished_game_by_its_outcome(outcome, expected):
    # [FUNCTION DOC]
    # Checks that finished games are scored by outcome, ignoring the evaluation.
    # After the agent's own game-ending move the opponent never replies, so the
    # evaluation is from the opponent's side -- here it says the agent is losing
    # badly even when it just won. The outcome is what counts.
    finished = _chess_obs(done=True, reward=outcome, metadata={"evaluation": -900.0})
    assert chess_position_rewards(finished) == {CHESS_POSITION_REWARD: expected}


def test_chess_position_reward_skips_a_finished_game_without_a_reward():
    # [FUNCTION DOC]
    # Checks that a finished game without a reward yields no shaping entry.
    assert chess_position_rewards(_chess_obs(done=True, reward=None)) == {}


def test_chess_position_reward_separates_positions_the_env_scores_identically():
    # [FUNCTION DOC]
    # Checks that a better evaluation gives a higher shaping reward.
    # The point of the shaping reward: chess pays 0.0 for every legal
    # non-terminal move, so without this two rollouts that both played on are
    # indistinguishable and GRPO has no advantage to learn from.
    good = chess_position_rewards(_chess_obs(metadata={"evaluation": 120.0}))
    bad = chess_position_rewards(_chess_obs(metadata={"evaluation": -450.0}))
    assert good[CHESS_POSITION_REWARD] > bad[CHESS_POSITION_REWARD]


def test_chess_position_reward_is_monotonic_in_the_evaluation():
    # [FUNCTION DOC]
    # Checks monotonicity of shaping over evaluations and zero at evaluation 0.
    values = [
        chess_position_rewards(_chess_obs(metadata={"evaluation": e}))[
            CHESS_POSITION_REWARD
        ]
        for e in (-600.0, -60.0, 0.0, 60.0, 600.0)
    ]
    assert values == sorted(values)
    assert values[2] == 0.0


@pytest.mark.parametrize(
    "observation",
    [
        "not a dict",
        {"fen": "x"},  # no metadata at all
        {"fen": "x", "metadata": "not a dict"},
        {"fen": "x", "metadata": {}},  # no evaluation key
        {"fen": "x", "metadata": {"evaluation": None}},
        {"fen": "x", "metadata": {"evaluation": "nonsense"}},
        {"fen": "x", "metadata": {"evaluation": float("nan")}},
        {"fen": "x", "metadata": {"evaluation": float("inf")}},
    ],
)
def test_chess_position_reward_skips_unusable_observations(observation):
    # [FUNCTION DOC]
    # Checks that unusable observations (non-dict, missing/invalid/NaN/inf evaluation)
    # yield an empty dict.
    # A missing or unusable evaluation must drop the key rather than inject a
    # 0.0, which the rubric would otherwise read as "an even position".
    assert chess_position_rewards(observation) == {}


def test_chess_profile_carries_the_shaping_hook_and_generic_does_not():
    # [FUNCTION DOC]
    # Checks that chess has the shaping hook and the generic profile does not.
    assert CHESS_TASK.shaping is chess_position_rewards
    assert get_task_profile("generic").shaping is None


# --------------------------------------------------------------------------- #
# Live round-trips against real in-process OpenEnv servers.
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    # [FUNCTION DOC]
    # HELPER: returns a free localhost port via binding port 0.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.integration
async def test_live_roundtrip_tool_mode():
    # [FUNCTION DOC]
    # Integration: serves the tiny env, then guesses each word via tool calls until done,
    # asserting a win with reward 1.0. Always stops the bridge and server.
    """Serve the tiny env for real and win the game through the bridge."""
    pytest.importorskip("uvicorn")
    import tiny_env

    port = _free_port()
    server, _thread = tiny_env.serve_in_background(port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        bridge = OpenEnvBridge(base_url=base_url, action_mode="tool")
        reset_turn = await bridge.start(seed=0)
        assert reset_turn.text  # got an opening observation
        assert reset_turn.done is False

        won = False
        rewards = []
        try:
            for word in tiny_env.WORDS:
                tool_call = {
                    "function": {
                        "name": "openenv_act",
                        "arguments": json.dumps({"action": {"guess": word}}),
                    }
                }
                turn = await bridge.act_from_tool_calls([tool_call])
                rewards.append(turn.reward)
                if turn.done:
                    won = turn.reward == 1.0
                    break
        finally:
            await bridge.stop()

        assert won, f"expected to win within the word list, got rewards={rewards}"
    finally:
        server.should_exit = True


@pytest.mark.integration
async def test_live_roundtrip_text_mode():
    # [FUNCTION DOC]
    # Integration: same as tool mode but in text mode with action_key "guess".
    """The same env driven in text mode (message text is the action)."""
    pytest.importorskip("uvicorn")
    import tiny_env

    port = _free_port()
    server, _thread = tiny_env.serve_in_background(port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        bridge = OpenEnvBridge(
            base_url=base_url, action_mode="text", action_key="guess"
        )
        await bridge.start(seed=0)
        saw_reward_one = False
        try:
            for word in tiny_env.WORDS:
                turn = await bridge.act_from_text(word)
                if turn.done and turn.reward == 1.0:
                    saw_reward_one = True
                    break
        finally:
            await bridge.stop()
        assert saw_reward_one
    finally:
        server.should_exit = True


@pytest.mark.integration
async def test_live_roundtrip_chess_env_tool_mode():
    # [FUNCTION DOC]
    # Integration (needs uvicorn, chess, moonfish): serves the real chess demo app, checks
    # the opening observation, that an illegal move gives -0.1 without changing the FEN and
    # is flagged, that a few random legal moves change the FEN, and that shaping is in (-1, 1).
    """Play OpenEnv's real chess environment through the chess task profile."""
    pytest.importorskip("uvicorn")
    pytest.importorskip("chess")
    pytest.importorskip("moonfish")
    import random

    import run_bridge_demo as demo

    port = _free_port()
    server, _thread = demo.serve_in_background(port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        bridge = OpenEnvBridge(
            base_url=base_url,
            action_mode="tool",
            tool_action_key=CHESS_TASK.tool_action_key,
            observation_renderer=CHESS_TASK.render,
        )
        rng = random.Random(0)
        try:
            turn = await bridge.start()
            assert turn.done is False
            assert "Legal moves" in turn.text
            start_fen = turn.raw_observation["fen"]

            # An illegal move is rejected with -0.1 and leaves the position alone.
            turn = await bridge.act_from_tool_calls([demo.move_tool_call("a1a8")])
            assert turn.reward == pytest.approx(-0.1)
            assert turn.done is False
            assert turn.raw_observation["fen"] == start_fen
            assert "rejected" in turn.text

            # A few legal moves actually advance the game.
            for _ in range(3):
                move = demo.pick_move(turn.raw_observation, rng)
                assert move is not None
                turn = await bridge.act_from_tool_calls([demo.move_tool_call(move)])
                if turn.done:
                    break
            assert turn.raw_observation["fen"] != start_fen

            # The environment reports its evaluation of the position it just
            # reached, which is what the shaping reward is derived from.
            shaped = chess_position_rewards(
                {**turn.raw_observation, "reward": turn.reward, "done": turn.done}
            )
            assert -1.0 < shaped[CHESS_POSITION_REWARD] < 1.0
        finally:
            await bridge.stop()
    finally:
        server.should_exit = True


# --------------------------------------------------------------------------- #
# serve_chess: session capacity for concurrent rollouts
# --------------------------------------------------------------------------- #


def _serve_chess():
    # [FUNCTION DOC]
    # HELPER: skips unless chess and moonfish are installed, then imports and returns the
    # serve_chess module.
    pytest.importorskip("chess")
    pytest.importorskip("moonfish")
    import serve_chess

    return serve_chess


def test_serve_chess_opts_into_concurrent_sessions():
    # [FUNCTION DOC]
    # Checks that the stock ChessEnvironment does not opt into concurrency while the
    # launcher subclass does and is a subclass of it.
    # ``ChessEnvironment`` itself does not set the flag, so the stock server
    # refuses to allow more than one session; the launcher's subclass is what
    # makes a batch of concurrent rollouts possible.
    serve_chess = _serve_chess()
    from envs.chess_env.server.chess_environment import ChessEnvironment

    assert not getattr(ChessEnvironment, "SUPPORTS_CONCURRENT_SESSIONS", False)
    assert serve_chess.ConcurrentChessEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True
    assert issubclass(serve_chess.ConcurrentChessEnvironment, ChessEnvironment)


def test_serve_chess_build_app_accepts_many_sessions():
    # [FUNCTION DOC]
    # Checks that build_app succeeds with 64 sessions (create_app would raise without opt-in).
    # ``create_app`` raises ConcurrencyConfigurationError for
    # max_concurrent_envs > 1 unless the env class opted in, so building the
    # app at all is the assertion.
    serve_chess = _serve_chess()
    app = serve_chess.build_app(max_sessions=64, opponent_depth=1, max_moves=40)
    assert app is not None


@pytest.mark.parametrize("color", ["white", "black"])
def test_serve_chess_can_pin_the_agent_color(color):
    # [FUNCTION DOC]
    # Checks that pinning the agent color sets the env's agent color and side to move.
    # The environment otherwise flips a coin per episode, which makes the
    # rollouts in a GRPO group incomparable.
    serve_chess = _serve_chess()
    import chess

    env = serve_chess.ConcurrentChessEnvironment(
        opponent="moonfish", opponent_depth=1, agent_color=color
    )
    expected = chess.WHITE if color == "white" else chess.BLACK
    assert env._agent_color == expected
    # The agent is always the side to move: playing black means the opponent
    # has already replied by the time the first observation is handed over.
    assert env._board.turn == expected


def test_serve_chess_normalizes_the_alternate_color(monkeypatch):
    # [FUNCTION DOC]
    # Checks that agent_color "alternate" is normalized to None in build_app, by replacing
    # create_app with a capturing fake and building an env from the captured factory.
    # ``--agent-color alternate`` is the CLI spelling of the environment's own
    # default, which is ``None``. The normalization has to live in
    # ``build_app`` rather than in argument parsing, because worker processes
    # read the setting back out of the environment as a string -- and the
    # upstream environment does not recognize "alternate": an unknown setting
    # falls through to ``chess.WHITE``, so the literal would quietly pin the
    # color it was asked to stop pinning.
    serve_chess = _serve_chess()

    captured = {}

    def fake_create_app(factory, *args, **kwargs):
        # [FUNCTION DOC]
        # Fake create_app that records the factory and returns a dummy object.
        captured["factory"] = factory
        return object()

    monkeypatch.setattr(serve_chess, "create_app", fake_create_app)
    serve_chess.build_app(max_sessions=2, agent_color="alternate")
    env = captured["factory"]()
    assert env._agent_color_setting is None


def test_serve_chess_rejects_capacity_above_one_without_the_opt_in():
    # [FUNCTION DOC]
    # Checks that create_app with the stock ChessEnvironment and capacity 64 raises an
    # error mentioning concurrency.
    # Guard the reason the subclass exists: if this ever stops raising, the
    # upstream environment has opted in and the launcher can be simplified.
    _serve_chess()  # skips unless the chess extras are installed
    import functools

    from envs.chess_env.models import ChessAction, ChessObservation
    from envs.chess_env.server.chess_environment import ChessEnvironment
    from openenv.core.env_server import create_app

    with pytest.raises(Exception, match="(?i)concurren"):
        create_app(
            functools.partial(ChessEnvironment, opponent="moonfish"),
            ChessAction,
            ChessObservation,
            env_name="chess_env",
            max_concurrent_envs=64,
        )
