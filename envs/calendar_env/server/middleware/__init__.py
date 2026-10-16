# [PACKAGE DOC]
# Middleware package; re-exports get_user_context from the auth module as its only public name.
"""
Middleware package
"""

from .auth import get_user_context

__all__ = ["get_user_context"]
