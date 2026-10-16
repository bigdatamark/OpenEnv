# [MODULE DOC]
# Detects which package format a directory contains based on well-known files, and defines errors
# for ambiguous/unrecognized packages and for recognized-but-unsupported package categories.
# WELL_KNOWN_FILES maps the formats with implemented parsers (only openenv.yaml) to filenames;
# UNSUPPORTED_CATEGORIES maps category keys to explanations.

"""Well-known files, signature detection, and unsupported-package errors."""

from pathlib import Path

from .types import SignatureKind

WELL_KNOWN_FILES: dict[SignatureKind, str] = {
    SignatureKind.OPENENV_SERVED: "openenv.yaml",
}
"""Formats this build can parse. Values are the enum filenames."""

UNSUPPORTED_CATEGORIES: dict[str, str] = {
    "hosted-verifier": (
        "the verifier calls an external service; the run cannot be hermetic and "
        "verifier portability is unmeasurable locally"
    ),
    "multi-agent": (
        "multi-agent scenes need an orchestration harness local validation does not define"
    ),
    "simulated-user": (
        "requires a user-simulator policy; no local reference simulator exists"
    ),
    "parser-not-implemented": (
        "defensive registry contract: a signature kind reached dispatch without a "
        "registered parser; unreachable when detection tracks implemented parsers"
    ),
}
"""RFC 008 unsupported-package categories. Maps to CLI exit code 2."""


class SignatureError(Exception):
    # [CLASS DOC]
    # Exception for a path that is not a directory, matches no parseable format, or matches several formats.
    """
    Ambiguous or unrecognized package: zero or two+ well-known files matched.

    Maps to CLI exit code 2.
    """


class UnsupportedPackageError(Exception):
    # [CLASS DOC]
    # Exception for a recognized package in an unsupported category; exposes category and reason.
    """
    A recognized package in a category local validation does not attempt.

    Raised at parse time where detectable. Maps to CLI exit code 2.

    Attributes:
        category (`str`):
            A key of [`~openenv.validation.signature.UNSUPPORTED_CATEGORIES`].
        reason (`str`):
            Human-readable explanation for this package.
    """

    def __init__(self, category: str, reason: str):
        # [FUNCTION DOC]
        # Validates category is a key of UNSUPPORTED_CATEGORIES (else raises ValueError), sets the message
        # "unsupported package (<category>): <reason>" and stores category and reason.
        if category not in UNSUPPORTED_CATEGORIES:
            raise ValueError(f"unknown unsupported-package category: {category!r}")
        super().__init__(f"unsupported package ({category}): {reason}")
        self.category = category
        self.reason = reason


def _has_yaml_frontmatter(path: Path) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if the file's UTF-8 text (leading whitespace stripped) starts with "---";
    # False on read or decode errors.
    try:
        return path.read_text(encoding="utf-8").lstrip().startswith("---")
    except (OSError, UnicodeError):
        return False


def _file_present(package_root: Path, kind: SignatureKind) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns whether the well-known file for the given kind exists as a regular file in the
    # package root; for task.md (POSTTRAIN_TASK) also requires YAML frontmatter.
    path = package_root / kind.value
    if not path.is_file():
        return False
    if kind is SignatureKind.POSTTRAIN_TASK:
        return _has_yaml_frontmatter(path)
    return True


def detect_signature(package_root: Path) -> SignatureKind:
    # [FUNCTION DOC]
    # Returns the single SignatureKind of the package directory.
    # Raises SignatureError if the path is not a directory, if more than one known format's file is
    # present (ambiguous), or if none of the parseable formats' files is present (unrecognized).
    """
    Detect a package's format from its well-known file. Never a guess.

    Only formats with implemented parsers are returned. A package carrying two or
    more well-known files of any named format is refused as ambiguous — even when
    only one of those formats has a parser in this build — because choosing the
    parseable one would be a guess. Zero implemented-parser matches is unrecognized.

    Args:
        package_root (`Path`):
            Directory to inspect.

    Returns:
        [`~openenv.validation.types.SignatureKind`]: the single matching signature.

    Raises:
        [`~openenv.validation.signature.SignatureError`]:
            The path is not a directory, zero implemented parsers match, or two or
            more named-format well-known files are present.
    """
    package_root = Path(package_root)
    if not package_root.is_dir():
        raise SignatureError(f"not a package directory: {package_root}")

    named_matches = [
        kind for kind in SignatureKind if _file_present(package_root, kind)
    ]
    if len(named_matches) > 1:
        found = ", ".join(sorted(kind.value for kind in named_matches))
        raise SignatureError(
            f"ambiguous package: {package_root} matches multiple signatures ({found}); "
            "a package must carry exactly one well-known file"
        )

    implemented = [
        kind for kind in WELL_KNOWN_FILES if _file_present(package_root, kind)
    ]
    if not implemented:
        expected = ", ".join(sorted(WELL_KNOWN_FILES.values()))
        raise SignatureError(
            f"unrecognized package: {package_root} contains none of the well-known "
            f"files this build can parse ({expected})"
        )
    return implemented[0]
