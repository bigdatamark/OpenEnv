# [PACKAGE DOC]
# Package of deterministic source importers that convert external environment
# formats (ORS, verifiers) into OpenEnv environments. Re-exports the base types
# and defines DEFAULT_IMPORTERS, a list containing one instance of each built-in
# importer (ORS first, then verifiers).
"""Deterministic source importers for OpenEnv environments."""

from .base import DetectedEnvironment, EnvironmentImporter, ImporterRegistry
from .ors import ORSImporter
from .verifiers import VerifiersImporter

DEFAULT_IMPORTERS = [ORSImporter(), VerifiersImporter()]

__all__ = [
    "DEFAULT_IMPORTERS",
    "DetectedEnvironment",
    "EnvironmentImporter",
    "ImporterRegistry",
    "ORSImporter",
    "VerifiersImporter",
]
