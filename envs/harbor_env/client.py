# [MODULE DOC]
# Re-exports the HarborEnv client from openenv.harbor.client; no logic.
"""Typed client for a deployed harbor_env."""

from openenv.harbor.client import HarborEnv

__all__ = ["HarborEnv"]
