# [MODULE DOC]
# Tests for the default Gradio playground (gradio_ui.build_gradio_app and its
# private helpers _result_html, _params, _Form, _description, _readme_body)
# and the Environment render_web/web_actions hooks, including OpenSpiel
# drawings and an env scaffolded by "openenv init". Callbacks are pulled from
# the built Blocks by function name (reset_env, step_fn, show_state) and run
# with asyncio; their output tuples are indexed by position (3 = rendered
# observation HTML, 4 = raw JSON, 5 = state/log, 6 = episode entries).
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the default web playground at /web and its environment hooks."""

import asyncio
import importlib
import json
import sys
from typing import Literal

import gradio as gr
import pytest
from openenv.core.env_server.gradio_ui import (
    _description,
    _Form,
    _params,
    _readme_body,
    _result_html,
    build_gradio_app,
)
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import (
    Action,
    EnvironmentMetadata,
    Observation,
    State,
)
from openenv.core.env_server.web_interface import (
    _extract_action_fields,
    WebInterfaceManager,
)


class MoveAction(Action):
    # [CLASS DOC]
    # Action with a required integer action_id.
    action_id: int


class BoardObservation(Observation):
    # [CLASS DOC]
    # Observation with a legal_actions integer list.
    legal_actions: list[int] = []


class TinyEnv(Environment):
    # [CLASS DOC]
    # Minimal environment: reset returns legal actions [0, 1]; step returns reward
    # 1.0 and done.
    def reset(self, seed=None, episode_id=None, **kwargs):
        # [FUNCTION DOC]
        # Returns BoardObservation with legal actions [0, 1].
        return BoardObservation(legal_actions=[0, 1])

    def step(self, action, timeout_s=None, **kwargs):
        # [FUNCTION DOC]
        # Returns BoardObservation with reward 1.0, done True.
        return BoardObservation(legal_actions=[0, 1], reward=1.0, done=True)

    @property
    def state(self):
        # [FUNCTION DOC]
        # Property returning a default State.
        return State()


def test_environment_hooks_default_to_nothing():
    # [FUNCTION DOC]
    # Default render_web returns None and web_actions returns an empty list.
    env = TinyEnv()
    assert env.render_web({"legal_actions": [0, 1]}) is None
    assert env.web_actions({"legal_actions": [0, 1]}) == []


def test_result_shows_tool_output_and_stats():
    # [FUNCTION DOC]
    # _result_html includes tool result text and reward/done/step stats.
    data = {"observation": {"result": {"data": "Hello"}}, "reward": 0.0, "done": False}
    html = _result_html(data, step_count=1)
    assert "Hello" in html
    assert "reward" in html and "done" in html and "step" in html


def test_result_lists_observation_fields():
    # [FUNCTION DOC]
    # _result_html lists observation fields but hides empty metadata.
    data = {
        "observation": {"legal_actions": [0, 1], "metadata": {}},
        "reward": 1.0,
        "done": True,
    }
    html = _result_html(data, step_count=2)
    assert "legal_actions" in html
    assert "metadata" not in html


class ScreenObservation(Observation):
    # [CLASS DOC]
    # Observation with large screen and board lists plus legal actions.
    screen: list = []
    board: list = []
    legal_actions: list[int] = []


class ScreenEnv(TinyEnv):
    # [CLASS DOC]
    # TinyEnv variant returning a 1080x1920x3 screen and 6x7 board, and recording
    # whatever it is asked to draw.
    def reset(self, seed=None, episode_id=None, **kwargs):
        # [FUNCTION DOC]
        # Returns the large screen, a 6x7 zero board and 7 legal actions.
        return ScreenObservation(
            screen=[[[0, 0, 255]] * 1920] * 1080,
            board=[[0] * 7] * 6,
            legal_actions=list(range(7)),
        )

    def step(self, action, timeout_s=None, **kwargs):
        # [FUNCTION DOC]
        # Returns an empty ScreenObservation with reward 1.0 and done.
        return ScreenObservation(reward=1.0, done=True)

    def render_web(self, observation):
        # [FUNCTION DOC]
        # Stores the observation on self.drawn (side effect) and returns an empty SVG.
        self.drawn = observation
        return "<svg></svg>"


def test_big_arrays_are_shown_as_their_shape():
    # [FUNCTION DOC]
    # Large arrays are summarized as "list · HxWxC" in HTML and JSON (JSON under
    # 1000 chars), small boards keep cells, render_web still receives the full
    # observation with reward/done which update after a step.
    env = ScreenEnv()
    manager = WebInterfaceManager(env, MoveAction, ScreenObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    fns = {f.fn.__name__: f.fn for f in blocks.fns.values()}

    outputs = asyncio.run(fns["reset_env"]([]))

    assert "list · 1080×1920×3" in outputs[3]
    assert "[0, 0, 0, 0, 0, 0, 0]" in outputs[3]  # a 6x7 board keeps its cells
    assert json.loads(outputs[4])["observation"]["screen"] == "list · 1080×1920×3"
    assert len(outputs[4]) < 1000
    assert len(env.drawn["screen"]) == 1080  # the drawing gets the full observation
    assert env.drawn["reward"] is None and env.drawn["done"] is False

    asyncio.run(fns["step_fn"](outputs[6], 0))

    assert env.drawn["reward"] == 1.0 and env.drawn["done"] is True


def test_playground_builds_for_a_plain_env():
    # [FUNCTION DOC]
    # build_gradio_app returns gr.Blocks for a non-MCP environment.
    manager = WebInterfaceManager(TinyEnv(), MoveAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    assert isinstance(blocks, gr.Blocks)


def test_playground_lists_mcp_tools():
    # [FUNCTION DOC]
    # For the echo env (skipped if unavailable) the tool radio lists echo_message
    # and echo_with_length.
    echo = pytest.importorskip("echo_env.server.echo_environment")
    manager = WebInterfaceManager(
        echo.EchoEnvironment(), CallToolAction, CallToolObservation
    )
    blocks = build_gradio_app(
        manager, _extract_action_fields(CallToolAction), None, False
    )
    radios = [b for b in blocks.blocks.values() if isinstance(b, gr.Radio)]
    tool_names = {value for radio in radios for _, value in radio.choices}
    assert {"echo_message", "echo_with_length"} <= tool_names


def test_catch_offers_legal_moves_and_a_board():
    # [FUNCTION DOC]
    # OpenSpiel catch (skipped without open_spiel): web_actions labels and an SVG
    # board with no hard-coded hex colours.
    pytest.importorskip("open_spiel")
    from openspiel_env.server.openspiel_environment import OpenSpielEnvironment

    env = OpenSpielEnvironment(game_name="catch")
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [
        ("0 · left", {"action_id": 0}),
        ("1 · stay", {"action_id": 1}),
        ("2 · right", {"action_id": 2}),
    ]
    board = env.render_web(obs)
    assert 'aria-label="Catch board"' in board
    assert "#" not in board  # theme colours only, so it reads in dark mode


def test_openspiel_2048_web_playground():
    # [FUNCTION DOC]
    # OpenSpiel 2048: direction-labelled actions and a board showing every
    # non-zero tile value, no hex colours.
    pytest.importorskip("open_spiel")
    from openspiel_env.server.openspiel_environment import OpenSpielEnvironment

    env = OpenSpielEnvironment(game_name="2048")
    obs = env.reset().model_dump()
    names = {0: "up", 1: "right", 2: "down", 3: "left"}
    assert env.web_actions(obs) == [
        (f"{a} · {names[a]}", {"action_id": a}) for a in obs["legal_actions"]
    ]
    board = env.render_web(obs)
    assert 'aria-label="2048 board"' in board
    assert "#" not in board
    for value in obs["info_state"]:
        if value:
            assert f">{int(value)}<" in board


@pytest.mark.parametrize(
    "game, label, button",
    [
        ("tic_tac_toe", "Tic-Tac-Toe board", ("4 · centre", {"action_id": 4})),
        ("connect_four", "Connect Four board", ("col 3", {"action_id": 3})),
        ("blackjack", "Blackjack board", ("0 · hit", {"action_id": 0})),
        ("kuhn_poker", "Kuhn Poker board", ("1 · bet", {"action_id": 1})),
        ("cliff_walking", "Cliff Walking board", ("1 · up", {"action_id": 1})),
    ],
)
def test_openspiel_web_playground(game, label, button):
    # [FUNCTION DOC]
    # Parametrized over several OpenSpiel games: an expected button is offered,
    # and after stepping it the drawing has the game's aria-label and no hex colours.
    pytest.importorskip("open_spiel")
    from openspiel_env.models import OpenSpielAction
    from openspiel_env.server.openspiel_environment import OpenSpielEnvironment

    env = OpenSpielEnvironment(game_name=game)
    obs = env.reset().model_dump()
    assert button in env.web_actions(obs)
    obs = env.step(OpenSpielAction(**button[1])).model_dump()
    drawing = env.render_web(obs)
    assert f'aria-label="{label}"' in drawing
    assert "#" not in drawing


def _sum_env():
    # [FUNCTION DOC]
    # HELPER: builds an MCPEnvironment subclass instance backed by a FastMCP
    # server with a single undocumented "total" tool taking a list of ints.
    from fastmcp import FastMCP
    from openenv.core.env_server.mcp_environment import MCPEnvironment

    mcp = FastMCP("sum_env")

    @mcp.tool
    def total(numbers: list[int]) -> int:
        # [FUNCTION DOC]
        # MCP tool returning the sum of the numbers.
        return sum(numbers)

    class SumEnv(MCPEnvironment):
        # [CLASS DOC]
        # Local MCPEnvironment returning empty observations and default state.
        def reset(self, seed=None, episode_id=None, **kwargs):
            # [FUNCTION DOC]
            # Returns an empty Observation.
            return Observation()

        def _step_impl(self, action, timeout_s=None, **kwargs):
            # [FUNCTION DOC]
            # Returns an empty Observation.
            return Observation()

        @property
        def state(self):
            # [FUNCTION DOC]
            # Property returning a default State.
            return State()

    return SumEnv(mcp)


def test_tool_without_description_and_json_argument():
    # [FUNCTION DOC]
    # A tool without description is still listed; its array parameter gets a
    # JSON textbox labelled "numbers (array, JSON) · required".
    manager = WebInterfaceManager(_sum_env(), CallToolAction, CallToolObservation)
    blocks = build_gradio_app(
        manager, _extract_action_fields(CallToolAction), None, False
    )
    radios = [b for b in blocks.blocks.values() if isinstance(b, gr.Radio)]
    assert ("total", "total") in [c for r in radios for c in r.choices]
    labels = [b.label for b in blocks.blocks.values() if isinstance(b, gr.Textbox)]
    assert "numbers (array, JSON) · required" in labels


def test_failed_tool_call_shows_the_error():
    # [FUNCTION DOC]
    # _result_html shows "Error: <message>" and not "null" for failed calls.
    data = {
        "observation": {"result": None, "error": {"message": "unknown tool"}},
        "reward": None,
        "done": False,
    }
    html = _result_html(data, step_count=1)
    assert "Error: unknown tool" in html
    assert "null" not in html


class RunAction(Action):
    # [CLASS DOC]
    # Action with a required command and optional timeout/retries/mode/tags.
    command: str
    timeout: float | None = 30.0
    retries: int | None = None
    mode: str = "fast"
    tags: list[str] | None = None


def test_params_unwrap_optional_and_keep_defaults():
    # [FUNCTION DOC]
    # _params unwraps optional (anyOf null) schemas, keeps defaults and marks
    # required flags.
    params = {
        name: (schema, required)
        for name, schema, required in _params(RunAction.model_json_schema())
    }
    assert params["command"] == ({"title": "Command", "type": "string"}, True)
    assert (
        params["timeout"][0]["type"] == "number"
        and params["timeout"][0]["default"] == 30.0
    )
    assert params["retries"][0]["type"] == "integer" and not params["retries"][1]
    assert params["tags"][0]["type"] == "array"


def test_form_sends_defaults_skips_empty_optionals_and_requires_fields():
    # [FUNCTION DOC]
    # _Form uses an empty textbox for defaultless optional ints, prefills defaults,
    # values() drops empty optionals and parses JSON lists, and raises ValueError
    # "Fill in command" for a blank required field.
    with gr.Blocks():
        form = _Form(_params(RunAction.model_json_schema()))
    by_name = dict(zip([n for n, _ in form.names], form.inputs))
    assert isinstance(by_name["retries"], gr.Textbox)  # no default: empty, not 0
    assert by_name["timeout"].value == 30.0
    raw = {
        "command": "ls",
        "timeout": 30.0,
        "retries": "",
        "mode": "fast",
        "tags": '["a"]',
    }
    values = form.values([raw[n] for n, _ in form.names])
    assert values == {"command": "ls", "timeout": 30.0, "mode": "fast", "tags": ["a"]}
    raw["command"] = " "
    with pytest.raises(ValueError, match="Fill in command"):
        form.values([raw[n] for n, _ in form.names])


class TurnAction(Action):
    # [CLASS DOC]
    # Action with a required Literal direction and an optional Literal speed.
    direction: Literal["LEFT", "RIGHT"]
    speed: Literal["slow", "fast"] | None = None


def test_required_choice_starts_on_its_first_option():
    # [FUNCTION DOC]
    # Required choice inputs default to their first option; optional ones to None
    # and are omitted from values().
    with gr.Blocks():
        form = _Form(_params(TurnAction.model_json_schema()))
    assert [i.value for i in form.inputs] == ["LEFT", None]
    assert form.values([i.value for i in form.inputs]) == {"direction": "LEFT"}


def test_description_falls_back_to_the_readme():
    # [FUNCTION DOC]
    # _description uses the first README paragraph (front matter, headings, notes
    # and link markup stripped) when the description is the generic placeholder,
    # else the given description.
    readme = "---\ntitle: X\n---\n# Catch\n\n> [!NOTE]\n> hi\n\nMove the [paddle](x) to catch the ball.\n"
    meta = EnvironmentMetadata(
        name="catch_env", description="catch_env environment", readme_content=readme
    )
    assert _description(meta) == "Move the paddle to catch the ball."
    meta = EnvironmentMetadata(
        name="x", description="A custom description.", readme_content=readme
    )
    assert _description(meta) == "A custom description."


def test_reset_errors_and_rewards_render_readably():
    # [FUNCTION DOC]
    # _result_html shows metadata errors as "Error: ...", a dash for missing
    # reward, and rounds float rewards.
    html = _result_html(
        {
            "observation": {"metadata": {"error": "E2B_API_KEY is not set"}},
            "reward": None,
            "done": True,
        },
        0,
    )
    assert "Error: E2B_API_KEY is not set" in html
    assert "–" in html
    html = _result_html(
        {"observation": {"value": 1}, "reward": -0.45999999999999996}, 1
    )
    assert "-0.46" in html and "0.4599999" not in html


class LoginAction(Action):
    # [CLASS DOC]
    # Action with user and api_key strings.
    user: str
    api_key: str


def test_secret_arguments_are_masked_in_the_episode():
    # [FUNCTION DOC]
    # Episode log masks api_key as *** and the secret does not appear in output 5.
    manager = WebInterfaceManager(TinyEnv(), LoginAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(LoginAction), None, False)
    step_fn = next(f.fn for f in blocks.fns.values() if f.fn.__name__ == "step_fn")

    outputs = asyncio.run(step_fn([], "ana", "sk-123"))

    assert outputs[6][-1][1] == 'step(user="ana", api_key=***)'
    assert "sk-123" not in outputs[5]


def test_long_arguments_are_cut_in_the_episode():
    # [FUNCTION DOC]
    # Long argument values are truncated with an ellipsis in the episode entry
    # (under 100 chars).
    manager = WebInterfaceManager(TinyEnv(), LoginAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(LoginAction), None, False)
    step_fn = next(f.fn for f in blocks.fns.values() if f.fn.__name__ == "step_fn")
    svg = "<svg>" + "<circle r='1'/>" * 100 + "</svg>"

    outputs = asyncio.run(step_fn([], svg, "sk-123"))

    call = outputs[6][-1][1]
    assert call.startswith('step(user="<svg><circle') and "…, api_key=***)" in call
    assert len(call) < 100


def test_enter_in_a_text_field_runs_the_step():
    # [FUNCTION DOC]
    # Every textbox's submit event triggers step_fn.
    manager = WebInterfaceManager(TinyEnv(), RunAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(RunAction), None, False)
    step = next(f for f in blocks.fns.values() if f.fn and f.fn.__name__ == "step_fn")
    textboxes = [id for id, b in blocks.blocks.items() if isinstance(b, gr.Textbox)]

    assert textboxes and {(t, "submit") for t in textboxes} <= set(step.targets)


class NoStateEnv(TinyEnv):
     # [CLASS DOC]
     # TinyEnv whose state property raises RuntimeError.
    @property
    def state(self):
        # [FUNCTION DOC]
        # Always raises RuntimeError("no episode").
        raise RuntimeError("no episode")


def test_state_shows_nothing_when_unavailable():
    # [FUNCTION DOC]
    # show_state returns an empty string when state raises.
    manager = WebInterfaceManager(NoStateEnv(), MoveAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    show_state = next(
        f.fn for f in blocks.fns.values() if f.fn.__name__ == "show_state"
    )

    assert show_state() == ""


class FailingResetEnv(TinyEnv):
    # [CLASS DOC]
    # TinyEnv whose reset raises RuntimeError.
    def reset(self, seed=None, episode_id=None, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError("no sandbox").
        raise RuntimeError("no sandbox")


def test_failed_reset_keeps_the_episode():
    # [FUNCTION DOC]
    # A failing reset shows "reset() failed: ..." and leaves episode entries unchanged.
    manager = WebInterfaceManager(FailingResetEnv(), MoveAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    reset_env = next(f.fn for f in blocks.fns.values() if f.fn.__name__ == "reset_env")
    entries = [
        ["R", "reset()", "new episode"],
        ["1", "step(action_id=0)", "reward 1.0"],
    ]

    outputs = asyncio.run(reset_env(entries))

    assert "reset() failed: no sandbox" in outputs[3]
    assert outputs[6] == entries


def test_new_env_from_the_template_gets_a_working_playground(
    tmp_path, monkeypatch, request
):
    # [FUNCTION DOC]
    # Runs "openenv init fresh_env" via CliRunner in tmp_path, imports the
    # generated package (sys.path and ENABLE_WEB_INTERFACE patched), checks /web/
    # returns 200, then drives reset/step/show_state and verifies the echo and
    # step_count. Side effects: writes files in tmp_path; finalizer purges
    # fresh_env modules from sys.modules.
    """An env scaffolded by `openenv init` needs no extra code for /web to work."""
    from fastapi.testclient import TestClient
    from openenv.cli.__main__ import app as cli
    from typer.testing import CliRunner

    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(cli, ["init", "fresh_env"], input="\n").exit_code == 0
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")

    def forget_fresh_env():
        # [FUNCTION DOC]
        # HELPER: deletes all fresh_env* entries from sys.modules.
        for module in [m for m in sys.modules if m.startswith("fresh_env")]:
            del sys.modules[module]

    request.addfinalizer(forget_fresh_env)
    server = importlib.import_module("fresh_env.server.app")
    models = importlib.import_module("fresh_env.models")
    environment = importlib.import_module("fresh_env.server.fresh_env_environment")

    assert TestClient(server.app).get("/web/").status_code == 200

    manager = WebInterfaceManager(
        environment.FreshEnvironment, models.FreshAction, models.FreshObservation
    )
    blocks = build_gradio_app(
        manager, _extract_action_fields(models.FreshAction), None, False
    )
    fns = {f.fn.__name__: f.fn for f in blocks.fns.values()}
    entries = asyncio.run(fns["reset_env"]([]))[6]
    assert json.loads(fns["show_state"]())["step_count"] == 0
    outputs = asyncio.run(fns["step_fn"](entries, "hello"))

    assert "hello" in outputs[3]
    assert outputs[6][-1][1] == 'step(message="hello")'
    assert json.loads(fns["show_state"]())["step_count"] == 1


def test_readme_drops_the_space_front_matter():
    # [FUNCTION DOC]
    # _readme_body strips the YAML front matter block.
    readme = "---\ntitle: Catch\nsdk: docker\n---\n\n# Catch\n\nMove the paddle.\n"
    assert _readme_body(readme) == "\n# Catch\n\nMove the paddle.\n"
