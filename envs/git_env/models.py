#!/usr/bin/env python3
# [MODULE DOC]
# Pydantic models for the Git environment: a single multi-purpose action, the
# observation, and the extended state.

"""
envs/git_env/models.py
--------------------------------
Action/Observation types for the Git environment with Gitea server.
"""

from __future__ import annotations

from typing import Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class GitAction(Action):
    # [CLASS DOC]
    # Unified action. action_type selects the operation ("clone_repo", "list_repos"
    # (default), "execute_git_command"); repo_name, target_dir (optional), command
    # and working_dir are used depending on the type.
    """
    Action for Git environment operations.

    This unified action class supports multiple operation types:
    - clone_repo: Clone a repository from Gitea to workspace
    - list_repos: List all available repositories
    - execute_git_command: Execute a git command in workspace

    Attributes:
        action_type: Type of operation ("clone_repo", "list_repos", "execute_git_command")
        repo_name: Name of repository (for clone_repo, execute_git_command)
        target_dir: Target directory for clone (optional)
        command: Git command to execute (for execute_git_command)
        working_dir: Working directory relative to workspace (for execute_git_command)
    """

    action_type: str = "list_repos"
    repo_name: str = ""
    target_dir: Optional[str] = None
    command: str = ""
    working_dir: str = ""


class GitObservation(Observation):
    # [CLASS DOC]
    # Action result: success flag, human message, output text, error text, and a list
    # of repository info dicts (populated for list_repos).
    """
    Result of executing a Git action.

    Attributes:
        success: Whether the action was successful
        message: Human-readable message about the result
        output: Command output or detailed result
        error: Error message if action failed
        repos: List of repositories (for list_repos action)
    """

    success: bool = False
    message: str = ""
    output: str = ""
    error: str = ""
    repos: list[dict[str, str]] = Field(default_factory=list)


class GitState(State):
    # [CLASS DOC]
    # State extended with gitea_ready (server reachable) and workspace_path
    # (default "/workspace").
    """
    State for Git environment.

    Attributes:
        episode_id: Unique identifier for the episode
        step_count: Number of steps taken
        gitea_ready: Whether Gitea server is accessible
        workspace_path: Path to the workspace directory
    """

    gitea_ready: bool = False
    workspace_path: str = "/workspace"
