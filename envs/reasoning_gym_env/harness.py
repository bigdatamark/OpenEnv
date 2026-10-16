# [MODULE DOC]
# Harness adapter for Reasoning Gym: exposes a client as a resource session with one MCP-style tool
# "answer" (string), using a one-question dataset per session.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Harness-oriented Reasoning Gym session adapters.

Follows the pattern introduced by ``openspiel_env.harness``: exposes a
ReasoningGymEnv client as a ``ResourceSession`` driven through MCP-style
tools, so it can be consumed by ``openenv.core.harness`` adapters (e.g.
the collect pipeline).
"""

from __future__ import annotations

import random
from typing import Any, Callable

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ResourceSessionFactory,
    StepEnvSessionAdapter,
    ToolResult,
)

from .client import ReasoningGymEnv
from .models import ReasoningGymAction

_REASONING_GYM_TOOLS: list[Tool] = [
    Tool(
        name="answer",
        description="Submit the final answer for the current question.",
        input_schema={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to submit.",
                },
            },
            "required": ["answer"],
        },
    ),
]


def _build_tool_result() -> Callable[..., ToolResult]:
    # [FUNCTION DOC]
    # HELPER: Returns a tool-result builder function.
    def builder(
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        state: Any,
    ) -> ToolResult:
        # [FUNCTION DOC]
        # HELPER: Converts a step result into a ToolResult with the submitted answer, score, correct answer,
        # reward and done in data; done flag; reward and state (dumped if it is a model) in metadata.
        observation = result.observation
        return ToolResult(
            data={
                "answer": arguments.get("answer"),
                "score": observation.score,
                "correct_answer": observation.correct_answer,
                "reward": result.reward,
                "done": result.done,
            },
            done=bool(result.done),
            metadata={
                "reward": result.reward,
                "state": state.model_dump() if hasattr(state, "model_dump") else state,
            },
        )

    return builder


class ReasoningGymSessionFactory(ResourceSessionFactory):
    # [CLASS DOC]
    # Resource session factory bound to a dataset name and optional dataset config.
    """Create ReasoningGym-backed resource sessions for harness rollouts."""

    def __init__(
        self,
        client_factory: Callable[[], ReasoningGymEnv],
        *,
        dataset_name: str,
        dataset_config: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC]
        # Stores the zero-argument client factory, dataset name and config (default empty mapping).
        self._client_factory = client_factory
        self._dataset_name = dataset_name
        self._dataset_config = dataset_config or {}

    def create(
        self,
        task: Any = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> StepEnvSessionAdapter:
        # [FUNCTION DOC]
        # Creates a client and returns a StepEnvSessionAdapter with the answer tool, an action builder that
        # stringifies the "answer" argument (KeyError if missing), an initial user message holding the
        # question, the tool result builder, and reset kwargs (dataset name, config, size 1). The seed is the
        # given one or a random 31-bit integer from the global random module.
        client = self._client_factory()
        effective_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        return StepEnvSessionAdapter(
            client=client,
            task=task,
            seed=effective_seed,
            episode_id=episode_id,
            tool_specs=list(_REASONING_GYM_TOOLS),
            action_builder=lambda name, arguments: ReasoningGymAction(
                answer=str(arguments["answer"]),
            ),
            initial_messages_builder=lambda result, current_task: [
                {
                    "role": "user",
                    "content": result.observation.question,
                }
            ],
            tool_result_builder=_build_tool_result(),
            reset_kwargs={
                "dataset_name": self._dataset_name,
                "dataset_config": self._dataset_config,
                "size": 1,
            },
        )


__all__ = [
    "ReasoningGymSessionFactory",
]
