# [PACKAGE DOC]
# Deprecated compatibility shim for the old openenv_core package. On import it imports
# openenv.core, emits a DeprecationWarning, mirrors its __all__, forwards attribute access and dir()
# to it, and registers openenv_core.<child> aliases in sys.modules for the submodules client_types,
# containers, env_client, env_server, rubrics, tools and utils (skipping any that are missing).

"""
Compatibility shim for the historical ``openenv_core`` package.

The core runtime now lives under ``openenv.core``. Importing from the old
package path will continue to work but emits a ``DeprecationWarning`` so
downstream users can migrate at their own pace.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from types import ModuleType
from typing import Dict

_TARGET_PREFIX = "openenv.core"
_TARGET_MODULE = importlib.import_module(_TARGET_PREFIX)

warnings.warn(
    "openenv_core is deprecated; import from openenv.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(_TARGET_MODULE, "__all__", [])


def __getattr__(name: str):
    # [FUNCTION DOC]
    # Module-level attribute fallback: returns the named attribute from openenv.core (raises
    # AttributeError if absent).
    return getattr(_TARGET_MODULE, name)


def __dir__():
    # [FUNCTION DOC]
    # Returns the sorted unique names from dir() of openenv.core.
    return sorted(set(dir(_TARGET_MODULE)))


def _alias(name: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Imports openenv.core.<name> and registers it in sys.modules as openenv_core.<name>.
    # Raises ModuleNotFoundError if the target is missing.
    target = f"{_TARGET_PREFIX}.{name}"
    sys.modules[f"{__name__}.{name}"] = importlib.import_module(target)


for _child in (
    "client_types",
    "containers",
    "env_client",
    "env_server",
    "rubrics",
    "tools",
    "utils",
):
    try:
        _alias(_child)
    except ModuleNotFoundError:  # pragma: no cover - defensive
        continue
