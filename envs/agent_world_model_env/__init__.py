# [PACKAGE DOC]
# Agent World Model (AWM) environment package. Exposes the MCP-style client AWMEnv and the
# AWMAction/AWMObservation models. One OpenEnv environment fronts many AWM scenarios selected at reset time.
"""
Agent World Model Environment for OpenEnv.
Wraps 1,000 Agent World Model sub-environments into a single
OpenEnv environment with dynamic scenario selection via reset().
"""

from .client import AWMEnv
from .models import AWMAction, AWMObservation

__all__ = ["AWMEnv", "AWMAction", "AWMObservation"]
