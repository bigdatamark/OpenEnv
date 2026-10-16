# [MODULE DOC]
# Checks that the server Dockerfiles of first-party Hub canonical environments enable the web
# interface.
"""Deployment defaults for first-party Hub canonicals."""

from __future__ import annotations

from pathlib import Path

import pytest


CANONICAL_ENVS = (
    "browsergym_env",
    "chat_env",
    "echo_env",
    "repl_env",
    "sumo_rl_env",
    "tbench2_env",
    "textarena_env",
)


@pytest.mark.parametrize("env_name", CANONICAL_ENVS)
def test_canonical_dockerfiles_enable_web_interface(env_name: str) -> None:
    # [FUNCTION DOC]
    # Parametrized over CANONICAL_ENVS: reads envs/<name>/server/Dockerfile and asserts it contains
    # "ENV ENABLE_WEB_INTERFACE=true". Raises FileNotFoundError if the Dockerfile is missing.
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "envs" / env_name / "server" / "Dockerfile"

    dockerfile_text = dockerfile.read_text()

    assert "ENV ENABLE_WEB_INTERFACE=true" in dockerfile_text
