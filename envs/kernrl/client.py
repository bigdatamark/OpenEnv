# [MODULE DOC]
# WebSocket client for the kernrl environment.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
kernrl Environment Client.

This module provides the client for connecting to a kernrl Environment server
via WebSocket for persistent sessions.

Example:
    >>> # Connect to a running server
    >>> with kernrl_env(base_url="http://localhost:8000") as client:
    ...     result = client.reset(problem_id="L1_23_Softmax")
    ...     print(result.observation.problem_description)
    ...
    ...     result = client.step(KernelAction(code=triton_kernel_code))
    ...     print(f"Speedup: {result.observation.speedup}x")

Example with Docker:
    >>> # Automatically start container and connect
    >>> client = kernrl_env.from_docker_image("kernrl:latest")
    >>> try:
    ...     result = client.reset()
    ...     result = client.step(KernelAction(code=my_kernel))
    ... finally:
    ...     client.close()
"""

from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State

    from .models import KernelAction, KernelObservation, KernelState
except ImportError:
    from models import KernelAction, KernelObservation, KernelState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State


class kernrl_env(EnvClient[KernelAction, KernelObservation, KernelState]):
    # [CLASS DOC]
    # EnvClient specialization for kernrl implementing the serialization hooks.
    """
    Client for the kernrl GPU kernel optimization environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Agents submit CUDA/Triton kernel code and receive feedback including:
    - Compilation status and errors
    - Correctness against reference implementation
    - Speedup compared to PyTorch baseline
    - Profiling data from NSight Systems/Compute

    Example:
        >>> # Connect to a running server
        >>> with kernrl_env(base_url="http://localhost:8000") as client:
        ...     result = client.reset(problem_id="L1_23_Softmax")
        ...     print(result.observation.problem_description)
        ...
        ...     result = client.step(KernelAction(code=triton_kernel_code))
        ...     print(f"Speedup: {result.observation.speedup}x")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = kernrl_env.from_docker_image("kernrl:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(KernelAction(code=my_kernel))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: KernelAction) -> Dict:
        # [FUNCTION DOC]
        # Builds the step payload with the single key "code".
        """
        Convert KernelAction to JSON payload for step request.

        Args:
            action: KernelAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "code": action.code,
        }

    def _parse_result(self, payload: Dict) -> StepResult[KernelObservation]:
        # [FUNCTION DOC]
        # Builds a KernelObservation from the nested "observation" dict (defaults: empty
        # strings, turn 0, max_turns 10, compilation_success False, optional fields
        # None) with top-level done/reward; metadata from top level, falling back to the
        # observation's, then {}. Returns a StepResult with reward, done and top-level
        # metadata.
        """
        Parse server response into StepResult[KernelObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with KernelObservation
        """
        obs_data = payload.get("observation", {})
        observation = KernelObservation(
            problem_id=obs_data.get("problem_id", ""),
            problem_description=obs_data.get("problem_description", ""),
            reference_code=obs_data.get("reference_code", ""),
            gpu_info=obs_data.get("gpu_info", ""),
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 10),
            feedback=obs_data.get("feedback", ""),
            compilation_success=obs_data.get("compilation_success", False),
            compilation_error=obs_data.get("compilation_error"),
            correctness_pass=obs_data.get("correctness_pass"),
            max_diff=obs_data.get("max_diff"),
            speedup=obs_data.get("speedup"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> KernelState:
        # [FUNCTION DOC]
        # Builds a KernelState from episode_id, step_count, problem_id, turn,
        # max_turns (default 10), best_speedup (default 0.0) and solved.
        """
        Parse server response into KernelState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            KernelState object with episode state
        """
        return KernelState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            problem_id=payload.get("problem_id"),
            turn=payload.get("turn", 0),
            max_turns=payload.get("max_turns", 10),
            best_speedup=payload.get("best_speedup", 0.0),
            solved=payload.get("solved", False),
        )
