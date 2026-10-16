# [MODULE DOC]
# In-process REPL helper that wraps a server-side REPLEnvironment directly (no network), returning
# StepResult objects like the remote client. Kept separate from client.py so the remote client does not
# import server code.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local in-process REPL helper.

This module is intentionally separate from `client.py` so the remote client
module does not import anything from `server/`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

try:
    from openenv.core.client_types import StepResult

    from .models import REPLAction, REPLObservation, REPLState
    from .rubrics import REPLRubric
    from .server.repl_environment import REPLEnvironment
except ImportError:
    from models import REPLAction, REPLObservation, REPLState
    from openenv.core.client_types import StepResult
    from rubrics import REPLRubric
    from server.repl_environment import REPLEnvironment


class LocalREPLEnv:
    # [CLASS DOC]
    # Synchronous local wrapper around one REPLEnvironment instance; usable as a context manager.
    """Explicit in-process REPL helper for local experimentation."""

    def __init__(
        self,
        *,
        llm_query_fn: Optional[Callable[[str], str]] = None,
        llm_batch_fn: Optional[Callable[[list[str]], list[str]]] = None,
        subcall_fn: Optional[Callable[[str, Optional[str]], str]] = None,
        subcall_batch_fn: Optional[
            Callable[[list[str], Optional[str]], list[str]]
        ] = None,
        max_output_length: int = 8192,
        context_preview_length: int = 500,
        rubric: Optional[REPLRubric] = None,
        rlm_max_depth: int = 1,
        rlm_max_iterations: int | None = None,
    ):
        # [FUNCTION DOC]
        # Creates the REPLEnvironment with the given LLM/sub-call callables, output and preview limits,
        # rubric and recursion limits (default depth 1).
        self._env = REPLEnvironment(
            max_output_length=max_output_length,
            context_preview_length=context_preview_length,
            rubric=rubric,
            llm_query_fn=llm_query_fn,
            llm_batch_fn=llm_batch_fn,
            subcall_fn=subcall_fn,
            subcall_batch_fn=subcall_batch_fn,
            rlm_max_depth=rlm_max_depth,
            rlm_max_iterations=rlm_max_iterations,
        )

    def reset(
        self,
        *,
        context: str = "",
        task_prompt: str = "",
        max_iterations: int = 30,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        llm_model: Optional[str] = None,
        expected_answer: Optional[str] = None,
        rlm_max_depth: Optional[int] = None,
        rlm_max_iterations: Optional[int] = None,
    ) -> StepResult[REPLObservation]:
        # [FUNCTION DOC]
        # Sets the environment's max_iterations, then resets it with context, task prompt, seed, episode id,
        # HF token, model and, only when given, expected answer and recursion overrides. Returns the wrapped
        # observation.
        self._env.max_iterations = max_iterations
        reset_kwargs = {}
        if rlm_max_depth is not None:
            reset_kwargs["rlm_max_depth"] = rlm_max_depth
        if rlm_max_iterations is not None:
            reset_kwargs["rlm_max_iterations"] = rlm_max_iterations
        if expected_answer is not None:
            reset_kwargs["expected_answer"] = expected_answer
        obs = self._env.reset(
            seed=seed,
            episode_id=episode_id,
            context=context,
            task_prompt=task_prompt,
            hf_token=hf_token,
            llm_model=llm_model,
            **reset_kwargs,
        )
        return self._wrap_observation(obs)

    def step(self, action: REPLAction) -> StepResult[REPLObservation]:
        # [FUNCTION DOC]
        # Steps the environment with the action and returns the wrapped observation.
        return self._wrap_observation(self._env.step(action))

    def execute(self, code: str) -> StepResult[REPLObservation]:
        # [FUNCTION DOC]
        # Steps with the given code.
        return self.step(REPLAction(code=code))

    def submit_final_answer(self, answer: str) -> StepResult[REPLObservation]:
        # [FUNCTION DOC]
        # Steps with is_final True and the answer, ending the episode.
        return self.step(REPLAction(code="", is_final=True, final_answer=answer))

    def get_variable(self, name: str) -> StepResult[REPLObservation]:
        # [FUNCTION DOC]
        # Executes code printing the repr of the named variable (name inserted unescaped).
        return self.execute(f"print(repr({name}))")

    def state(self) -> REPLState:
        # [FUNCTION DOC]
        # Returns the environment's current REPLState.
        return self._env.state

    def list_variables(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the namespace keys from the current state.
        return self.state().namespace_keys

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the underlying environment.
        self._env.close()

    def __enter__(self) -> "LocalREPLEnv":
        # [FUNCTION DOC]
        # Context manager entry; returns self.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # [FUNCTION DOC]
        # Context manager exit; closes the environment. Does not suppress exceptions.
        self.close()

    @staticmethod
    def _wrap_observation(obs: REPLObservation) -> StepResult[REPLObservation]:
        # [FUNCTION DOC]
        # HELPER: Wraps an observation in a StepResult with its reward and done (no metadata).
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )
