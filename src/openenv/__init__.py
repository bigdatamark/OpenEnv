# [PACKAGE DOC]
# Top-level openenv package. Sets __version__ from the installed distribution metadata
# ("openenv", then "openenv-core", else "0.0.0"). Submodules core and cli and the classes
# AutoEnv, AutoAction, GenericEnvClient, GenericAction and SyncEnvClient are imported lazily on
# first attribute access and then cached in the module globals.
"""Unified OpenEnv package bundling the CLI and core runtime."""

from __future__ import annotations

from importlib import import_module, metadata

__all__ = [
    "core",
    "cli",
    "AutoEnv",
    "AutoAction",
    "GenericEnvClient",
    "GenericAction",
    "SyncEnvClient",
]


def _load_package_version() -> str:
    # [FUNCTION DOC]
    # HELPER: returns the installed version of the first found distribution among "openenv" and
    # "openenv-core", or "0.0.0" if neither is installed.
    """Resolve the installed distribution version for the OpenEnv package."""
    for distribution_name in ("openenv", "openenv-core"):
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
    return "0.0.0"


__version__ = _load_package_version()


_LAZY_MODULES = {
    "core": ".core",
    "cli": ".cli",
}

_LAZY_ATTRS = {
    "AutoEnv": (".auto", "AutoEnv"),
    "AutoAction": (".auto", "AutoAction"),
    "GenericEnvClient": (".core", "GenericEnvClient"),
    "GenericAction": (".core", "GenericAction"),
    "SyncEnvClient": (".core", "SyncEnvClient"),
}


def __getattr__(name: str):
    # [FUNCTION DOC]
    # Module attribute hook: imports a lazy submodule or lazy attribute on first access, caches
    # it in globals and returns it. Raises: AttributeError for unknown names.
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module

    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # Returns the sorted union of module globals and __all__ (so lazy names appear in dir()).
    return sorted(set(globals().keys()) | set(__all__))
