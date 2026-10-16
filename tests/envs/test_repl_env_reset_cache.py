# [MODULE DOC]
# Unit tests for REPLEnvironment's reset-time caching of HF-backed LLM functions: rebuild on model
# change, no rebuild when the model resolves to the same default, and constructor-provided functions
# left alone. At import, prepends envs to sys.path and, if smolagents is not installed, temporarily
# registers a stub smolagents module only while importing REPLEnvironment.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for REPL reset-time LLM cache invalidation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class _DummyLocalPythonExecutor:
    # [CLASS DOC]
    # HELPER: Minimal stand-in for smolagents.LocalPythonExecutor with a state dict.
    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # HELPER: Initialises an empty state, ignoring arguments.
        self.state = {}

    def send_tools(self, tools) -> None:
        # [FUNCTION DOC]
        # HELPER: Stores the tools.
        self.tools = tools

    def send_variables(self, variables) -> None:
        # [FUNCTION DOC]
        # HELPER: Merges variables into state.
        self.state.update(variables)


def _install_smolagents_stub_if_needed() -> bool:
    # [FUNCTION DOC]
    # HELPER: If smolagents is not importable, registers a stub module exposing the dummy executor in
    # sys.modules and returns True; otherwise returns False.
    if importlib.util.find_spec("smolagents") is not None:
        return False

    module = ModuleType("smolagents")
    module.LocalPythonExecutor = _DummyLocalPythonExecutor
    sys.modules["smolagents"] = module
    return True


_ENVS_ROOT = Path(__file__).resolve().parents[2] / "envs"
sys.path.insert(0, str(_ENVS_ROOT))
_installed_stub = _install_smolagents_stub_if_needed()
try:
    from repl_env.server.repl_environment import REPLEnvironment
finally:
    if _installed_stub:
        sys.modules.pop("smolagents", None)


def test_reset_rebuilds_llm_on_model_change(monkeypatch):
    # [FUNCTION DOC]
    # With HF_TOKEN set and _create_llm_functions replaced by a tracker, resets with model-A then model-B
    # rebuild twice with those models.
    """Changing `llm_model` between resets must rebuild the LLM functions."""
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    env = REPLEnvironment()
    calls: list[tuple[str | None, str | None]] = []

    def tracked(token, model):
        # [FUNCTION DOC]
        # HELPER: Records (token, model) and installs a dummy llm_query_fn and runtime controller.
        calls.append((token, model))
        env.llm_query_fn = lambda *args, **kwargs: ""
        env._runtime_controller = object()

    monkeypatch.setattr(env, "_create_llm_functions", tracked)
    env.reset(llm_model="model-A")
    env.reset(llm_model="model-B")

    assert [model for _, model in calls] == ["model-A", "model-B"]


def test_reset_no_rebuild_when_model_resolves_to_same_default(monkeypatch):
    # [FUNCTION DOC]
    # reset with the explicit default model followed by a plain reset builds only once.
    """reset(llm_model=<default>) then reset() must not rebuild."""
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    env = REPLEnvironment()
    calls: list[tuple[str | None, str | None]] = []

    def tracked(token, model):
        # [FUNCTION DOC]
        # HELPER: Records (token, model) and installs a dummy llm_query_fn and runtime controller.
        calls.append((token, model))
        env.llm_query_fn = lambda *args, **kwargs: ""
        env._runtime_controller = object()

    monkeypatch.setattr(env, "_create_llm_functions", tracked)
    default = REPLEnvironment._resolve_model(None)
    env.reset(llm_model=default)
    env.reset()

    assert len(calls) == 1


def test_reset_preserves_constructor_llm_functions(monkeypatch):
    # [FUNCTION DOC]
    # A constructor-supplied llm_query_fn is not replaced on reset and no build happens.
    """Constructor-provided LLM functions are not HF runtime cache entries."""
    env = REPLEnvironment(llm_query_fn=lambda prompt: f"custom: {prompt}")
    calls: list[tuple[str | None, str | None]] = []

    def tracked(token, model):
        # [FUNCTION DOC]
        # HELPER: Records (token, model).
        calls.append((token, model))

    monkeypatch.setattr(env, "_create_llm_functions", tracked)
    env.reset()

    assert calls == []
    assert env.llm_query_fn("prompt") == "custom: prompt"
