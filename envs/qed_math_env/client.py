# [MODULE DOC]
# MCP tool-calling client for the QED Math environment. Adds typed convenience wrappers around the
# get_problem, submit_proof and get_grading_guidelines tools and normalises reset/tool results into
# ProblemObservation / ProofSubmissionObservation.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QED Math Environment Client.

Provides tool-calling style interactions with the QED Math environment
via MCP (Model Context Protocol).

Examples:

```python
with QEDMathEnv(base_url="http://localhost:8000") as env:
    env.reset()
    tools = env.list_tools()
    result = env.call_tool("get_problem")
    result = env.call_tool("submit_proof", proof="By induction...")
```
"""

from typing import Any, Mapping, Optional

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import Observation, State
from openenv.core.mcp_client import MCPToolClient

from .models import ProblemObservation, ProofSubmissionObservation


class QEDMathEnv(MCPToolClient):
    # [CLASS DOC]
    # MCPToolClient subclass with a longer default message timeout (600 s, since grading can be slow)
    # and helper methods; each sync method dispatches its async counterpart through the base client.
    """
    Client for the QED Math Environment.

    Inherits MCP tool-calling interface from MCPToolClient:
    - ``list_tools()``: Discover available MCP tools
    - ``call_tool(name, **kwargs)``: Call a tool by name
    - ``reset(**kwargs)``: Reset the environment

    Examples:

    ```python
    with QEDMathEnv(base_url="http://localhost:8000") as env:
        env.reset()
        result = env.call_tool("get_problem")
        result = env.call_tool("submit_proof", proof="By induction...")
    ```
    """

    DEFAULT_MESSAGE_TIMEOUT_S = 600.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = DEFAULT_MESSAGE_TIMEOUT_S,
        websocket_ping_interval_s: Optional[float] = 20.0,
        websocket_ping_timeout_s: Optional[float] = 20.0,
        provider: Optional[Any] = None,
        mode: Optional[str] = None,
    ):
        # [FUNCTION DOC]
        # Passes connection settings (base_url, connect/message timeouts, websocket ping settings,
        # provider, mode) to MCPToolClient unchanged.
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            websocket_ping_interval_s=websocket_ping_interval_s,
            websocket_ping_timeout_s=websocket_ping_timeout_s,
            provider=provider,
            mode=mode,
        )

    @staticmethod
    def _as_problem_observation(value: Any) -> ProblemObservation:
        # [FUNCTION DOC]
        # HELPER: Returns value as a ProblemObservation: as-is if already one, built from a mapping or from
        # a model's dumped fields. Raises TypeError for other types (and pydantic errors for bad fields).
        """Normalize tool/reset outputs into a ProblemObservation instance."""
        if isinstance(value, ProblemObservation):
            return value
        if isinstance(value, Mapping):
            return ProblemObservation(**dict(value))
        if hasattr(value, "model_dump"):
            return ProblemObservation(**value.model_dump())
        raise TypeError(
            f"Unsupported problem observation payload type: {type(value).__name__}"
        )

    @staticmethod
    def _as_proof_submission_observation(value: Any) -> ProofSubmissionObservation:
        # [FUNCTION DOC]
        # HELPER: Same as the problem normaliser but for ProofSubmissionObservation.
        """Normalize tool outputs into a ProofSubmissionObservation instance."""
        if isinstance(value, ProofSubmissionObservation):
            return value
        if isinstance(value, Mapping):
            return ProofSubmissionObservation(**dict(value))
        if hasattr(value, "model_dump"):
            return ProofSubmissionObservation(**value.model_dump())
        raise TypeError(
            f"Unsupported proof submission payload type: {type(value).__name__}"
        )

    def reset(self, problem_id: Optional[str] = None, **kwargs: Any) -> Any:
        # [FUNCTION DOC]
        # Resets the remote environment, adding problem_id to the reset kwargs when given. Returns the
        # dispatched result: a StepResult whose observation is a ProblemObservation.
        """
        Reset the environment, optionally selecting a specific problem.

        Args:
            problem_id: Optional problem identifier to load a specific problem.
                        If None, a problem is chosen randomly from the dataset.
            **kwargs: Additional reset parameters (e.g., seed).

        Returns:
            StepResult with a normalized ProblemObservation in `observation`.
        """
        if problem_id is not None:
            kwargs["problem_id"] = problem_id
        return self._dispatch(lambda: self._reset_async(**kwargs))

    async def _reset_async(self, **kwargs: Any) -> StepResult[Observation]:
        # [FUNCTION DOC]
        # HELPER: Awaits the base async reset, normalises its observation to ProblemObservation and returns
        # a new StepResult with the same reward, done and metadata.
        result = await super()._reset_async(**kwargs)
        observation = result.observation if isinstance(result, StepResult) else result
        normalized_observation = self._as_problem_observation(observation)
        return StepResult(
            observation=normalized_observation,
            reward=result.reward,
            done=result.done,
            metadata=result.metadata,
        )

    def submit_proof(self, proof: str) -> Any:
        # [FUNCTION DOC]
        # Calls the submit_proof tool with the proof text and returns a ProofSubmissionObservation (via dispatch).
        """
        Submit a proof attempt for the current problem.

        Args:
            proof: The proof text to submit for grading.

        Returns:
            ProofSubmissionObservation with score (0-7), feedback, and reward.
        """
        return self._dispatch(lambda: self._submit_proof_async(proof))

    async def _submit_proof_async(self, proof: str) -> ProofSubmissionObservation:
        # [FUNCTION DOC]
        # HELPER: Async body of submit_proof: calls the tool and normalises the result.
        result = await self.call_tool("submit_proof", proof=proof)
        return self._as_proof_submission_observation(result)

    def get_current_problem(self) -> Any:
        # [FUNCTION DOC]
        # Calls the get_problem tool and returns a ProblemObservation for the active problem (via dispatch).
        """
        Retrieve the current problem statement without resetting.

        Returns:
            ProblemObservation for the active problem.
        """
        return self._dispatch(self._get_current_problem_async)

    async def _get_current_problem_async(self) -> ProblemObservation:
        # [FUNCTION DOC]
        # HELPER: Async body of get_current_problem.
        result = await self.call_tool("get_problem")
        return self._as_problem_observation(result)

    def get_problem(self) -> Any:
        # [FUNCTION DOC]
        # Alias for get_current_problem.
        """Compatibility alias for get_current_problem()."""
        return self.get_current_problem()

    def get_grading_feedback(self) -> Any:
        # [FUNCTION DOC]
        # Calls the get_grading_guidelines tool and returns its payload as a plain mapping (via dispatch).
        """
        Retrieve the grading guidelines/rubric for the current problem.

        Returns:
            Tool payload containing grading_guidelines and problem metadata.
        """
        return self._dispatch(self._get_grading_feedback_async)

    async def _get_grading_feedback_async(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Async body: converts a mapping or model result to a plain mapping. Raises TypeError otherwise.
        result = await self.call_tool("get_grading_guidelines")
        if isinstance(result, Mapping):
            return dict(result)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        raise TypeError(
            f"Unsupported grading feedback payload type: {type(result).__name__}"
        )

    def get_state(self) -> Any:
        # [FUNCTION DOC]
        # Returns the result of the base client's state call.
        """Return current environment state (episode_id, step_count)."""
        return self.state()

    def get_state_sync(self) -> State:
        # [FUNCTION DOC]
        # Returns the result of the base client's state call (same as get_state).
        """Synchronous helper for code paths that do not use async/await."""
        return self.state()
