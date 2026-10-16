# [MODULE DOC]
# Pydantic models for the coding environment.
"""
envs/coding_env/models.py
--------------------------------
Action/Observation types for the Coding environment.
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Action, Observation, State


class CodeAction(Action):
    # [CLASS DOC]
    # Action holding Python source code to execute.
    """
    Represents a single code execution request.
    """

    code: str
    # Optional: future fields like 'lint': bool, 'timeout_s': float, etc.


class CodeObservation(Observation):
    # [CLASS DOC]
    # Execution result: stdout, stderr and exit_code (0 on success); inherits reward, done, metadata.
    """
    Result of executing code in the environment.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class CodeState(State):
    # [CLASS DOC]
    # State adding last_exit_code to episode_id/step_count.
    """State for CodeAct environment with persistent execution context."""

    last_exit_code: int = 0
