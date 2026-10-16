# [PACKAGE DOC]
# FreeBusy API package; re-exports its FastAPI router.
"""
FreeBusy API module
"""

from .router import router

__all__ = ["router"]
