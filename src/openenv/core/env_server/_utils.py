# [MODULE DOC]
# HELPER module with small utilities shared by env server code.
"""Shared env server implementation helpers."""

from __future__ import annotations

from typing import Any


def overrides_method(method: Any, base_method: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if the bound method's underlying function (or the object
    # itself if not bound) is not the given base implementation, i.e. a subclass
    # overrides it.
    """Return whether a bound method differs from the base implementation."""
    return getattr(method, "__func__", method) is not base_method
