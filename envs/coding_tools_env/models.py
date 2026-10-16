# [MODULE DOC]
# Pydantic models for the coding tools environment: tool results, todo items, multi-edit specs and the
# per-session state.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Models for coding_tools_env."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    # [CLASS DOC]
    # Normalized result of one tool call: tool name, ok flag, output text, optional error and metadata dict.
    """Normalized result for one tool execution."""

    tool: str
    ok: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TodoItem(BaseModel):
    # [CLASS DOC]
    # Todo entry managed by the todo_write tool: id, content, status and priority strings.
    """Todo item tracked by todo_write."""

    id: str
    content: str
    status: str
    priority: str


class CodingToolsState(State):
    # [CLASS DOC]
    # Session state: sandbox id, setup command results, verify commands and their results, todos, tool call
    # history, submitted flag, last reward, a note when a reward override was ignored, and last error.
    """Per-session state for coding_tools_env."""

    sandbox_id: str | None = None
    setup_results: list[CommandResult] = Field(default_factory=list)
    verify_commands: list[str] = Field(default_factory=list)
    verify_results: list[CommandResult] = Field(default_factory=list)
    todos: list[TodoItem] = Field(default_factory=list)
    tool_history: list[CommandResult] = Field(default_factory=list)
    submitted: bool = False
    last_reward: float | None = None
    reward_override_ignored: str | None = None
    last_error: str | None = None


class EditSpec(BaseModel):
    # [CLASS DOC]
    # One string replacement for multi_edit: old_string, new_string and replace_all flag.
    """One edit operation for multi_edit."""

    old_string: str
    new_string: str
    replace_all: bool = False
