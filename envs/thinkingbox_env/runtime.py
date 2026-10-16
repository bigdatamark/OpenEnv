# [MODULE DOC]
# Shared (client/server/evaluator) helpers for ThinkingBox: loading and fingerprinting native YAML
# configuration files, and computing provenance of the installed thinkingbox package (VCS commit or source
# tree hash) and whether it matches the git pin declared by the thinkingbox-env distribution.
"""Load shared native configuration and installed ThinkingBox provenance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from thinkingbox.common.config_types import ConfigFile


@dataclass(frozen=True)
class ThinkingBoxRuntimeProvenance:
    # [CLASS DOC]
    # Frozen dataclass describing the installed ThinkingBox runtime: identity (commit id or
    # "source-sha256:<hash>"), version, source type (vcs, editable, local or installed), source tree SHA-256,
    # optional commit id, requested revision and normalised source URL, editable flag, and whether it is the
    # canonical pinned build.
    """Describe the installed ThinkingBox package without exposing local paths."""

    identity: str
    version: str
    source_type: str
    source_sha256: str
    commit_id: str | None = None
    requested_revision: str | None = None
    source_url: str | None = None
    editable: bool = False
    canonical: bool = False


def _normalized_vcs_url(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: strips a "git+" prefix and returns the URL with lowercased scheme and host (port kept), path
    # without trailing slash, and no query or fragment.
    raw = value.removeprefix("git+")
    parsed = urlsplit(raw)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            host.casefold(),
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )


def _parse_thinkingbox_requirement(requirement: str) -> tuple[str, str] | None:
    # [FUNCTION DOC]
    # HELPER: parses a requirement string of the form "thinkingbox @ git+<url>@<40-hex sha>" (optional
    # environment marker). Returns (normalised URL, lowercase sha) or None if it does not match.
    match = re.match(
        r"^\s*thinkingbox\s*@\s*git\+(.+)@([0-9a-fA-F]{40})(?:\s*;.*)?\s*$",
        requirement,
    )
    if match is None:
        return None
    return _normalized_vcs_url(match.group(1)), match.group(2).lower()


def _expected_thinkingbox_vcs_pin() -> tuple[str, str] | None:
    # [FUNCTION DOC]
    # HELPER: returns the first parsed thinkingbox git pin among the thinkingbox-env distribution's
    # requirements, or None if not installed or not found.
    try:
        requirements = metadata.distribution("thinkingbox-env").requires or []
    except metadata.PackageNotFoundError:
        requirements = []
    for requirement in requirements:
        parsed = _parse_thinkingbox_requirement(requirement)
        if parsed is not None:
            return parsed
    return None


def _thinkingbox_source_root() -> Path:
    # [FUNCTION DOC]
    # HELPER: returns the resolved directory of the installed thinkingbox package. Raises RuntimeError if it
    # cannot be located.
    spec = importlib.util.find_spec("thinkingbox")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        raise RuntimeError("Installed ThinkingBox package source is unavailable")
    return Path(next(iter(locations))).resolve()


def _source_tree_sha256(root: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: deterministic SHA-256 over all files under root (sorted by relative POSIX path; skips .git, cache
    # dirs, __pycache__ and .pyc/.pyo), hashing each relative path and its bytes with NUL separators. Raises
    # RuntimeError if no files are found.
    ignored = {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix not in {".pyc", ".pyo"}
            and not any(part in ignored for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise RuntimeError("Installed ThinkingBox package source is empty")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_thinkingbox_runtime_provenance() -> ThinkingBoxRuntimeProvenance:
    # [FUNCTION DOC]
    # Returns provenance of the installed thinkingbox distribution (cached after first call). Always hashes the
    # source tree. If PEP 610 direct_url.json has git vcs_info, validates url and 40-hex commit (RuntimeError on
    # invalid metadata), and marks canonical when (normalised URL, commit) equals the expected pin. Otherwise
    # returns a non-canonical record typed editable/local/installed from dir_info.
    # Raises RuntimeError if thinkingbox is not installed or metadata is malformed.
    """Load deterministic provenance for the installed ThinkingBox runtime."""
    try:
        distribution = metadata.distribution("thinkingbox")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("ThinkingBox distribution is not installed") from exc

    source_sha256 = _source_tree_sha256(_thinkingbox_source_root())
    raw_direct_url = distribution.read_text("direct_url.json")
    direct_url: dict[str, Any] | None = None
    if raw_direct_url is not None:
        try:
            parsed = json.loads(raw_direct_url)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ThinkingBox PEP 610 metadata is invalid") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("ThinkingBox PEP 610 metadata is invalid")
        direct_url = parsed

    if direct_url is not None and isinstance(direct_url.get("vcs_info"), dict):
        vcs_info = direct_url["vcs_info"]
        commit_id = vcs_info.get("commit_id")
        source_url = direct_url.get("url")
        vcs = vcs_info.get("vcs")
        if (
            vcs != "git"
            or not isinstance(source_url, str)
            or not isinstance(commit_id, str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", commit_id) is None
        ):
            raise RuntimeError("ThinkingBox PEP 610 VCS metadata is invalid")
        normalized_url = _normalized_vcs_url(source_url)
        commit_id = commit_id.lower()
        expected = _expected_thinkingbox_vcs_pin()
        canonical = expected == (normalized_url, commit_id)
        requested_revision = vcs_info.get("requested_revision")
        if requested_revision is not None and not isinstance(requested_revision, str):
            raise RuntimeError("ThinkingBox requested revision metadata is invalid")
        return ThinkingBoxRuntimeProvenance(
            identity=commit_id,
            version=distribution.version,
            source_type="vcs",
            source_sha256=source_sha256,
            commit_id=commit_id,
            requested_revision=requested_revision,
            source_url=normalized_url,
            canonical=canonical,
        )

    dir_info = direct_url.get("dir_info") if direct_url is not None else None
    editable = (
        bool(dir_info.get("editable", False)) if isinstance(dir_info, dict) else False
    )
    source_type = (
        "editable" if editable else ("local" if dir_info is not None else "installed")
    )
    return ThinkingBoxRuntimeProvenance(
        identity=f"source-sha256:{source_sha256}",
        version=distribution.version,
        source_type=source_type,
        source_sha256=source_sha256,
        editable=editable,
    )


def require_canonical_thinkingbox_runtime() -> ThinkingBoxRuntimeProvenance:
    # [FUNCTION DOC]
    # Returns the runtime provenance; raises RuntimeError if it is not the canonical pinned build.
    """Return installed runtime provenance only when it matches the VCS pin."""
    provenance = load_thinkingbox_runtime_provenance()
    if not provenance.canonical:
        raise RuntimeError(
            "Installed ThinkingBox runtime does not match the pinned VCS dependency"
        )
    return provenance


def _parse_thinkingbox_config(payload: bytes) -> ConfigFile:
    # [FUNCTION DOC]
    # HELPER: decodes UTF-8 YAML bytes and validates them into a ThinkingBox ConfigFile.
    return ConfigFile.model_validate(yaml.safe_load(payload.decode("utf-8")))


def load_thinkingbox_config_with_sha256(
    config_path: str | Path,
) -> tuple[ConfigFile, str]:
    # [FUNCTION DOC]
    # Reads the config file bytes once (user-expanded path) and returns (parsed ConfigFile, SHA-256 hex of the
    # bytes). Propagates I/O, YAML and validation errors.
    """Load one immutable config payload and its content fingerprint."""
    payload = Path(config_path).expanduser().read_bytes()
    return _parse_thinkingbox_config(payload), hashlib.sha256(payload).hexdigest()


def load_thinkingbox_config(config_path: str | Path) -> ConfigFile:
    # [FUNCTION DOC]
    # Returns only the parsed ConfigFile for config_path.
    """Load and validate a native ThinkingBox configuration."""
    parsed, _ = load_thinkingbox_config_with_sha256(config_path)
    return parsed
