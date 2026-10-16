# [MODULE DOC]
# Strict JSON parsing, canonical digesting, and bounded load/atomic write of catalog
# snapshots on the local filesystem. No network access.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, deterministic catalog serialization without network resolution."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .errors import CatalogError
from .models import CatalogSnapshot, MAX_CATALOG_BYTES


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    # [FUNCTION DOC]
    # HELPER: json object_pairs_hook that builds a dict and raises CatalogError on duplicate keys.
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError("JSON contains duplicate object keys")
        result[key] = value
    return result


def _non_json_number(value: str) -> None:
    # [FUNCTION DOC]
    # HELPER: json parse_constant hook that always raises CatalogError (rejects NaN/Infinity).
    raise CatalogError("JSON contains a non-finite number")


def parse_json(data: str) -> object:
    # [FUNCTION DOC]
    # Parses JSON text rejecting duplicate keys and non-finite numbers. Returns the value.
    # Raises CatalogError on invalid JSON or excessive nesting.
    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=_non_json_number,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise CatalogError("Invalid catalog JSON") from error


def catalog_digest(snapshot: CatalogSnapshot) -> str:
    # [FUNCTION DOC]
    # Computes "sha256:<hex>" over the snapshot dumped in JSON mode by alias, excluding
    # None fields and the digest field, encoded as compact sorted-key UTF-8 JSON.
    payload = snapshot.model_dump(
        by_alias=True, exclude_none=True, exclude={"digest"}, mode="json"
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def require_complete(snapshot: CatalogSnapshot) -> None:
    # [FUNCTION DOC]
    # Raises CatalogError if the snapshot is not complete or its digest does not match
    # the recomputed digest. Returns None otherwise.
    if not snapshot.complete:
        raise CatalogError("The configured catalog is incomplete")
    if catalog_digest(snapshot) != snapshot.digest:
        raise CatalogError("Catalog content does not match its snapshot digest")


def load_catalog(path: str | Path) -> CatalogSnapshot:
    # [FUNCTION DOC]
    # Reads a snapshot file (at most 16 MiB), parses strict JSON, validates as
    # CatalogSnapshot and requires completeness and a matching digest. Returns it.
    # Raises CatalogError on read errors, oversize, non-UTF-8, invalid JSON/profile,
    # incomplete or digest mismatch.
    """
    Read one complete, versioned local metadata snapshot.

    Args:
        path (`str` or `Path`):
            Explicitly configured local snapshot. References inside it are not fetched.
    """
    try:
        with Path(path).open("rb") as source:
            data = source.read(MAX_CATALOG_BYTES + 1)
    except OSError as error:
        raise CatalogError("Cannot read the configured catalog") from error
    if len(data) > MAX_CATALOG_BYTES:
        raise CatalogError("Catalog exceeds the metadata size limit")
    try:
        raw = parse_json(data.decode("utf-8"))
        snapshot = CatalogSnapshot.model_validate(raw)
    except (UnicodeDecodeError, ValidationError) as error:
        raise CatalogError("Catalog does not match the supported profile") from error
    require_complete(snapshot)
    return snapshot


def write_catalog(snapshot: CatalogSnapshot, path: str | Path) -> None:
    # [FUNCTION DOC]
    # Writes the snapshot as indented sorted-key JSON plus newline. Requires a matching
    # digest (incomplete snapshots are allowed) and size <= 16 MiB, else CatalogError.
    # Writes to a temp file in the destination directory, fsyncs, then atomically replaces
    # the destination; removes the temp file on failure. OSError -> CatalogError.
    """
    Atomically write a snapshot, including explicitly incomplete build reports.

    Args:
        snapshot ([`~openenv.discovery.CatalogSnapshot`]):
            Produced snapshot with a matching digest.
        path (`str` or `Path`):
            Destination file. Existing contents are not replaced by a partial write.
    """
    if catalog_digest(snapshot) != snapshot.digest:
        raise CatalogError("Cannot write a catalog with an invalid digest")
    payload = (
        json.dumps(
            snapshot.model_dump(by_alias=True, exclude_none=True, mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    if len(payload.encode("utf-8")) > MAX_CATALOG_BYTES:
        raise CatalogError("Catalog exceeds the metadata size limit")
    destination = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    except OSError as error:
        raise CatalogError("Cannot write the catalog snapshot") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
