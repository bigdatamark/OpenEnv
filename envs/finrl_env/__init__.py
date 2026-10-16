# [PACKAGE DOC]
# FinRL environment package: wraps FinRL's stock trading simulator behind the
# OpenEnv client/server API. Exposes FinRLEnv, FinRLAction and FinRLObservation
# lazily via a module-level attribute hook so importing the package does not
# import the client or models eagerly.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FinRL Environment for OpenEnv.

This package provides a wrapper around FinRL's StockTradingEnv that conforms
to the OpenEnv specification, enabling stock trading RL tasks through a
simple HTTP API.

Example:
    >>> from envs.finrl_env import FinRLEnv, FinRLAction
    >>>
    >>> # Connect to server
    >>> client = FinRLEnv(base_url="http://localhost:8000")
    >>>
    >>> # Reset environment
    >>> result = client.reset()
    >>> print(result.observation.portfolio_value)
    >>>
    >>> # Execute trading action
    >>> action = FinRLAction(actions=[0.5])  # Buy
    >>> result = client.step(action)
    >>> print(result.reward)
"""

__all__ = ["FinRLEnv", "FinRLAction", "FinRLObservation"]


def __getattr__(name: str):
    # [FUNCTION DOC]
    # HELPER: Module-level lazy attribute resolver. Imports and returns FinRLEnv from
    # .client or FinRLAction/FinRLObservation from .models on first access.
    # Raises AttributeError for any other name.
    if name == "FinRLEnv":
        from .client import FinRLEnv

        return FinRLEnv
    if name == "FinRLAction":
        from .models import FinRLAction

        return FinRLAction
    if name == "FinRLObservation":
        from .models import FinRLObservation

        return FinRLObservation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
