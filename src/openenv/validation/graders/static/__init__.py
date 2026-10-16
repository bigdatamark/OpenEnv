# [PACKAGE DOC]
# Static-level validation graders (checks that need no running environment);
# re-exports StaticManifestGrader.
"""Static-level graders."""

from .manifest import StaticManifestGrader

__all__ = ["StaticManifestGrader"]
