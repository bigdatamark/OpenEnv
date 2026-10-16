# [MODULE DOC]
# Tests for create_web_interface_app layout options: custom gradio_builder,
# title_override, show_default_tab, custom_tab_name and custom_tab_primary.
# gr.mount_gradio_app (and sometimes gr.TabbedInterface) are monkeypatched to
# capture the built Blocks. Skipped without gradio.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for custom Gradio layout options."""

from __future__ import annotations

import pytest
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

gr = pytest.importorskip("gradio", reason="gradio is not installed")

from openenv.core.env_server import web_interface
from openenv.core.env_server.web_interface import create_web_interface_app


class LayoutAction(Action):
    # [CLASS DOC]
    # Action with optional message (default "noop").
    message: str = "noop"


class LayoutObservation(Observation):
    # [CLASS DOC]
    # Observation with required response string.
    response: str


class LayoutState(State):
    # [CLASS DOC]
    # State with step_count.
    step_count: int = 0


class LayoutEnvironment(Environment):
    # [CLASS DOC]
    # Simple echo environment used by the layout tests.
    def __init__(self):
        # [FUNCTION DOC]
        # Initializes a default LayoutState.
        super().__init__()
        self._state = LayoutState()

    def reset(self) -> LayoutObservation:
        # [FUNCTION DOC]
        # Resets state and returns response "reset".
        self._state = LayoutState()
        return LayoutObservation(response="reset")

    def step(self, action: LayoutAction) -> LayoutObservation:
        # [FUNCTION DOC]
        # Increments step_count and echoes the action message.
        self._state.step_count += 1
        return LayoutObservation(response=action.message)

    @property
    def state(self) -> LayoutState:
        # [FUNCTION DOC]
        # Property returning the current state.
        return self._state

    def close(self) -> None:
        # [FUNCTION DOC]
        # No-op.
        pass


def _capture_mounted_blocks(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: monkeypatches web_interface.gr.mount_gradio_app with a fake that
    # records the mounted blocks and the "path" kwarg into a dict and returns the
    # app unchanged. Returns that dict.
    captured = {}

    def fake_mount_gradio_app(app, blocks, *args, **kwargs):
        # [FUNCTION DOC]
        # Records blocks and path into the captured dict; returns app.
        captured["blocks"] = blocks
        captured["path"] = kwargs.get("path")
        return app

    monkeypatch.setattr(
        web_interface.gr,
        "mount_gradio_app",
        fake_mount_gradio_app,
    )
    return captured


def test_single_view_custom_builder_uses_default_display_title(monkeypatch) -> None:
    # [FUNCTION DOC]
    # With a custom builder and show_default_tab False, the builder receives and
    # the mounted Blocks carry title "OpenEnv Agentic Environment: layout_env",
    # mounted at /web.
    captured = _capture_mounted_blocks(monkeypatch)
    builder_args = {}

    def builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start):
        # [FUNCTION DOC]
        # Records the title argument and returns placeholder-titled Blocks.
        builder_args["title"] = title
        return gr.Blocks(title="placeholder")

    create_web_interface_app(
        LayoutEnvironment,
        LayoutAction,
        LayoutObservation,
        env_name="layout_env",
        gradio_builder=builder,
        show_default_tab=False,
    )

    expected_title = "OpenEnv Agentic Environment: layout_env"
    assert builder_args["title"] == expected_title
    assert captured["blocks"].title == expected_title
    assert captured["path"] == "/web"


def test_single_view_custom_builder_uses_title_override(monkeypatch) -> None:
    # [FUNCTION DOC]
    # title_override is passed to the builder and set on the mounted Blocks.
    captured = _capture_mounted_blocks(monkeypatch)
    builder_args = {}

    def builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start):
        # [FUNCTION DOC]
        # Records the title argument and returns placeholder-titled Blocks.
        builder_args["title"] = title
        return gr.Blocks(title="placeholder")

    create_web_interface_app(
        LayoutEnvironment,
        LayoutAction,
        LayoutObservation,
        env_name="layout_env",
        gradio_builder=builder,
        show_default_tab=False,
        title_override="Custom Layout",
    )

    assert builder_args["title"] == "Custom Layout"
    assert captured["blocks"].title == "Custom Layout"


def test_custom_tab_primary_controls_tab_order_and_title(monkeypatch) -> None:
    # [FUNCTION DOC]
    # With custom_tab_primary, tabs are ordered [custom, "Playground"] and the
    # outer Blocks get the override title.
    captured = _capture_mounted_blocks(monkeypatch)

    def fake_tabbed_interface(blocks, tab_names):
        # [FUNCTION DOC]
        # Records the tab blocks and names; returns empty Blocks.
        captured["tab_blocks"] = blocks
        captured["tab_names"] = tab_names
        return gr.Blocks()

    monkeypatch.setattr(web_interface.gr, "TabbedInterface", fake_tabbed_interface)

    def builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start):
        # [FUNCTION DOC]
        # Returns Blocks titled with the given title.
        return gr.Blocks(title=title)

    create_web_interface_app(
        LayoutEnvironment,
        LayoutAction,
        LayoutObservation,
        env_name="layout_env",
        gradio_builder=builder,
        custom_tab_name="REPL",
        custom_tab_primary=True,
        title_override="Tabbed Layout",
    )

    assert captured["tab_names"] == ["REPL", "Playground"]
    assert len(captured["tab_blocks"]) == 2
    assert captured["blocks"].title == "Tabbed Layout"


def test_custom_builder_defaults_to_playground_first(monkeypatch) -> None:
    # [FUNCTION DOC]
    # Without custom_tab_primary, tabs are ordered ["Playground", custom].
    captured = _capture_mounted_blocks(monkeypatch)

    def fake_tabbed_interface(blocks, tab_names):
        # [FUNCTION DOC]
        # Records the tab blocks and names; returns empty Blocks.
        captured["tab_blocks"] = blocks
        captured["tab_names"] = tab_names
        return gr.Blocks()

    monkeypatch.setattr(web_interface.gr, "TabbedInterface", fake_tabbed_interface)

    def builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start):
        # [FUNCTION DOC]
        # Returns Blocks titled with the given title.
        return gr.Blocks(title=title)

    create_web_interface_app(
        LayoutEnvironment,
        LayoutAction,
        LayoutObservation,
        env_name="layout_env",
        gradio_builder=builder,
        custom_tab_name="Custom View",
        title_override="Tabbed Layout",
    )

    assert captured["tab_names"] == ["Playground", "Custom View"]
    assert len(captured["tab_blocks"]) == 2
    assert captured["blocks"].title == "Tabbed Layout"


def test_title_override_applies_without_custom_builder(monkeypatch) -> None:
    # [FUNCTION DOC]
    # title_override also applies to the default (builder-less) interface.
    captured = _capture_mounted_blocks(monkeypatch)

    create_web_interface_app(
        LayoutEnvironment,
        LayoutAction,
        LayoutObservation,
        env_name="layout_env",
        title_override="Default Layout",
    )

    assert captured["blocks"].title == "Default Layout"
