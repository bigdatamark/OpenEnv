# [MODULE DOC]
# Pydantic models for the Jupyter environment's notebook cells, setup/verify
# command results and extended state.
"""Pydantic models for Jupyter environment state."""

from typing import List, Optional

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


class NotebookCell(BaseModel):
    # [CLASS DOC]
    # One executed cell: id, type ("code" or "shell"), source, plain-text output,
    # optional error/traceback, execution count, image flag and base64 PNG list,
    # and success flag.
    """Represents a single executed cell in the notebook."""

    cell_id: str
    cell_type: str  # "code" | "shell"
    code: str
    output: str  # plain text for LLM
    error: Optional[str] = None  # traceback string if failed
    execution_count: int
    has_image: bool = False  # True if result includes PNG
    images: List[str] = Field(default_factory=list)  # base64-encoded PNG strings
    success: bool = True


class CommandResult(BaseModel):
    # [CLASS DOC]
    # Result of one task setup or verify shell command: command, output, optional
    # error, success flag.
    """Outcome of one setup or verify shell command."""

    command: str
    output: str = ""
    error: Optional[str] = None
    success: bool = True


class JupyterState(State):
    # [CLASS DOC]
    # State extended with executed cells, sandbox id, last cell success, setup and
    # verify command results, verify commands, submitted answer, last reward, and
    # an optional note recording an ignored reward override.
    """Extended state tracking notebook cells and sandbox info."""

    cells: List[NotebookCell] = Field(default_factory=list)
    sandbox_id: Optional[str] = None
    last_cell_success: bool = True
    setup_results: List[CommandResult] = Field(default_factory=list)
    verify_results: List[CommandResult] = Field(default_factory=list)
    verify_commands: List[str] = Field(default_factory=list)
    submitted_answer: Optional[str] = None
    last_reward: Optional[float] = None
    reward_override_ignored: Optional[str] = None
