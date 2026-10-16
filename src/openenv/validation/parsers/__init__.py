# [PACKAGE DOC]
# Parser protocol and registry mapping a detected package SignatureKind to the parser
# that turns that format into a NormalizedManifest.
"""Parser protocol and registry.

A parser owns everything format-specific; after `parse()` returns, the format has left
the pipeline. The registry is the only place a [`~openenv.validation.types.SignatureKind`]
selects behavior.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..manifest import NormalizedManifest
from ..signature import UnsupportedPackageError
from ..types import SignatureKind


@runtime_checkable
class Parser(Protocol):
    # [CLASS DOC]
    # Runtime-checkable protocol: a signature attribute and parse(package_root) returning a
    # NormalizedManifest without executing package code.
    """
    Parses one package format into the normalized manifest.

    A parse is a pure read: it never imports or executes package code.
    """

    signature: SignatureKind

    def parse(self, package_root: Path) -> NormalizedManifest:
        # [FUNCTION DOC]
        # Protocol method: reads the package at package_root and returns its normalized manifest.
        ...


class ParserRegistry:
    # [CLASS DOC]
    # Mutable mapping from SignatureKind to parser.
    """
    Maps detected signatures to registered parsers.

    A detected signature with no registered parser raises
    [`~openenv.validation.signature.UnsupportedPackageError`] with category
    `"parser-not-implemented"` — recognized, never guessed.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Creates an empty registry.
        self._parsers: dict[SignatureKind, Parser] = {}

    def register(self, parser: Parser) -> None:
        # [FUNCTION DOC]
        # Registers parser under its signature, replacing any previous one.
        """
        Register a parser for its signature.

        Args:
            parser ([`~openenv.validation.parsers.Parser`]):
                The parser to register. Replaces any parser previously registered for
                the same signature.
        """
        self._parsers[parser.signature] = parser

    def parser_for(self, signature: SignatureKind) -> Parser:
        # [FUNCTION DOC]
        # Returns the parser for signature; raises UnsupportedPackageError
        # ("parser-not-implemented") if none is registered.
        """
        Return the parser registered for a detected signature.

        Args:
            signature ([`~openenv.validation.types.SignatureKind`]):
                The detected package signature.

        Returns:
            [`~openenv.validation.parsers.Parser`]: the registered parser.
        """
        parser = self._parsers.get(signature)
        if parser is None:
            raise UnsupportedPackageError(
                "parser-not-implemented",
                f"no parser registered for signature {signature.value!r}",
            )
        return parser
