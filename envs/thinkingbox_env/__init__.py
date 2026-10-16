# [PACKAGE DOC]
# ThinkingBox environment package: exposes the public wire types (actions, observation, state, tool-call
# models, execution provenance, ListToolsAction) and the ThinkingBoxEnv client. The client is loaded lazily
# via module __getattr__ so server-side imports never pull in client code.
"""Expose the public ThinkingBox wire types and client.

The client is loaded lazily so importing server modules never crosses the
OpenEnv client-server boundary.
"""

from typing import Any

from openenv.core import ListToolsAction

from .models import (
    CallToolAction,
    SubmitMessageAction,
    SubmittedToolCall,
    ThinkingBoxAction,
    ThinkingBoxExecutionProvenance,
    ThinkingBoxObservation,
    ThinkingBoxState,
    ToolCallResult,
)

__all__ = [
    "CallToolAction",
    "ListToolsAction",
    "SubmitMessageAction",
    "SubmittedToolCall",
    "ThinkingBoxAction",
    "ThinkingBoxEnv",
    "ThinkingBoxExecutionProvenance",
    "ThinkingBoxObservation",
    "ThinkingBoxState",
    "ToolCallResult",
]


def __getattr__(name: str) -> Any:
    # [FUNCTION DOC]
    # Module attribute hook: on first access to "ThinkingBoxEnv" imports it from the client module, caches it
    # in module globals and returns it. Raises AttributeError for any other unknown name.
    """Load optional package exports without importing client code on servers."""
    if name == "ThinkingBoxEnv":
        from .client import ThinkingBoxEnv

        globals()[name] = ThinkingBoxEnv
        return ThinkingBoxEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # Returns the sorted union of module globals and __all__ (so lazy exports are listed).
    """Return package exports, including lazily loaded attributes."""
    return sorted(set(globals()) | set(__all__))
