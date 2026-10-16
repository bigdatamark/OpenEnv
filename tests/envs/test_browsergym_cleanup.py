# [MODULE DOC]
# Tests that BrowserGymEnvironment.close swallows errors from the underlying
# gym env. Skipped without gymnasium; prepends repo root to sys.path.
"""Unit tests for BrowserGym cleanup behavior."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

pytest.importorskip("gymnasium", reason="gymnasium is not installed")

from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment


class _BrokenGymEnv:
    # [CLASS DOC]
    # HELPER: stand-in gym env whose close always fails.
    def close(self) -> None:
        # [FUNCTION DOC]
        # Raises RuntimeError("close failed").
        raise RuntimeError("close failed")


def test_browsergym_close_swallows_cleanup_errors() -> None:
    # [FUNCTION DOC]
    # Builds an environment without __init__, attaches the broken gym env and
    # asserts close() does not raise.
    """HTTP cleanup should not mask a successful reset/step result."""
    env = BrowserGymEnvironment.__new__(BrowserGymEnvironment)
    env.gym_env = _BrokenGymEnv()

    BrowserGymEnvironment.close(env)
