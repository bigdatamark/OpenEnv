# [PACKAGE DOC]
# Environment discovery package: builds, serializes, loads, searches and diffs a
# versioned metadata-only catalog of environments (EnvironmentCard / DiscoveryEntry
# records in a CatalogSnapshot). Independent of AutoEnv resolution; nothing here
# imports or runs environments. Re-exports the public API from its submodules.
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned, metadata-only environment discovery, separate from AutoEnv resolution."""

from .errors import CatalogError
from .models import (
    CatalogChanges,
    CatalogSnapshot,
    DiscoveryEntry,
    ENVIRONMENT_MEDIA_TYPE,
    EnvironmentCard,
    PROFILE_VERSION,
    SearchMatch,
)
from .producer import build_catalog
from .search import compare_catalogs, resolve_entry, search_catalog
from .serialization import load_catalog, write_catalog

__all__ = [
    "CatalogChanges",
    "CatalogError",
    "CatalogSnapshot",
    "DiscoveryEntry",
    "ENVIRONMENT_MEDIA_TYPE",
    "EnvironmentCard",
    "PROFILE_VERSION",
    "SearchMatch",
    "build_catalog",
    "compare_catalogs",
    "load_catalog",
    "resolve_entry",
    "search_catalog",
    "write_catalog",
]
