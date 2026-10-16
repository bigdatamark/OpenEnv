# [MODULE DOC]
# Downloads, safely extracts, validates and caches the commit-pinned ThinkingBox benchmark data release.
# A bundle is a root with dataset/{agent,scenario,test_case}, servers, support, releases and LICENSE.txt,
# plus a YAML manifest of task UIDs. Canonical caches carry a JSON stamp and must match a pinned bundle
# hash. Module settings come from env vars: OPENENV_TB_DATASET (explicit data path), OPENENV_TB_DATA_TIMEOUT
# (default 120 s), OPENENV_TB_DATA_CACHE (default under XDG_CACHE_HOME/openenv). Size limits: 256 MiB
# archive, 2 GiB extracted. A thread lock plus a file lock serialise publication.
"""Download and validate the executable ThinkingBox benchmark release.

The loader safely extracts a commit-pinned archive, verifies every executable
release asset with one deterministic bundle hash, and publishes caches
atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import threading
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

import yaml
from filelock import FileLock

from .models import (
    DATA_ARCHIVE_URL,
    DATA_BUNDLE_SHA256,
    DATA_COMMIT,
    DATA_MANIFEST_PATH,
    DATA_RELEASE_NAME,
)


DATASET_NAME = "thinkingbox_bench"
DATASET_PATH = os.environ.get("OPENENV_TB_DATASET", "")
DATA_TIMEOUT = float(os.environ.get("OPENENV_TB_DATA_TIMEOUT", "120"))
_cache_home = Path(
    os.environ.get(
        "XDG_CACHE_HOME",
        str(Path.home() / ".cache"),
    )
)
DATA_CACHE = Path(
    os.environ.get(
        "OPENENV_TB_DATA_CACHE",
        str(_cache_home / "openenv" / "thinkingbox_bench"),
    )
)
_STAMP = ".openenv-thinkingbox-data.json"
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
_BUNDLE_TOP_LEVELS = (
    "dataset",
    "servers",
    "support",
    "releases",
)
_ALLOWED_TOP_LEVELS = {
    *_BUNDLE_TOP_LEVELS,
    "LICENSE.txt",
}
_MANIFEST_PATH = Path(DATA_MANIFEST_PATH)
_download_lock = threading.Lock()


class DatasetError(RuntimeError):
    # [CLASS DOC]
    # RuntimeError subclass raised whenever the pinned data bundle is unavailable or invalid.
    """The pinned ThinkingBox data bundle is unavailable or invalid."""


@dataclass(frozen=True)
class DataBundle:
    # [CLASS DOC]
    # Frozen dataclass identifying a validated bundle: root, dataset dir, manifest path, bundle hash, manifest
    # file hash, hash of the ordered UID list, task count, and release name / revision ("local" unless
    # canonical).
    """Identify one validated executable ThinkingBox data bundle.

    Args:
        root (`pathlib.Path`):
            Root directory containing release data and executable assets.
        dataset_dir (`pathlib.Path`):
            Dataset directory passed to the native ThinkingBox hydrator.
        manifest_path (`pathlib.Path`):
            Exact release manifest selected by the adapter.
        bundle_sha256 (`str`):
            Deterministic hash covering the release directories and license.
        manifest_sha256 (`str`):
            Hash of the exact manifest file bytes.
        manifest_uids_sha256 (`str`):
            Hash of the ordered manifest UID list.
        task_count (`int`):
            Number of unique UIDs derived from the manifest.
        release_name (`str`, *optional*, defaults to `"local"`):
            Human-readable release name for canonical cached data.
        revision (`str`, *optional*, defaults to `"local"`):
            Immutable data commit for canonical cached data.
    """

    root: Path
    dataset_dir: Path
    manifest_path: Path
    bundle_sha256: str
    manifest_sha256: str
    manifest_uids_sha256: str
    task_count: int
    release_name: str = "local"
    revision: str = "local"


def _read_test_uids(path: Path) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: loads the YAML manifest and returns its list of UIDs. Raises DatasetError if it cannot be parsed,
    # is not a non-empty list of non-blank strings, or contains duplicates.
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetError("ThinkingBox manifest could not be parsed") from exc
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and item.strip() for item in raw)
    ):
        raise DatasetError("ThinkingBox manifest must be a non-empty list of UIDs")
    if len(raw) != len(set(raw)):
        raise DatasetError("ThinkingBox manifest UIDs must be unique")
    return raw


def _sha256_file(path: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: SHA-256 hex digest of a file's bytes.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    # [FUNCTION DOC]
    # HELPER: SHA-256 hex digest of the compact, key-sorted, non-ASCII-preserving JSON encoding of value.
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bundle_sha256(root: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: deterministic hash over LICENSE.txt and all files under the four bundle top-level dirs (skipping
    # __pycache__ and .pyc/.pyo), sorted by relative path, hashing path and bytes with NUL separators.
    files = [root / "LICENSE.txt"]
    for top_level in _BUNDLE_TOP_LEVELS:
        files.extend(
            path
            for path in (root / top_level).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_stamp() -> dict[str, str | int]:
    # [FUNCTION DOC]
    # HELPER: returns the expected stamp mapping (schema version 2, release name, data commit, bundle hash,
    # manifest path).
    return {
        "schema_version": 2,
        "release_name": DATA_RELEASE_NAME,
        "data_commit": DATA_COMMIT,
        "bundle_sha256": DATA_BUNDLE_SHA256,
        "manifest_path": DATA_MANIFEST_PATH,
    }


def _require_canonical_content(bundle: DataBundle) -> None:
    # [FUNCTION DOC]
    # HELPER: raises DatasetError if the manifest is outside the bundle root, or the bundle hash or manifest
    # relative path differs from the pinned values.
    try:
        manifest_relative = bundle.manifest_path.relative_to(bundle.root).as_posix()
    except ValueError as exc:
        raise DatasetError("ThinkingBox manifest is outside the data bundle") from exc
    if (
        bundle.bundle_sha256 != DATA_BUNDLE_SHA256
        or manifest_relative != DATA_MANIFEST_PATH
    ):
        raise DatasetError("ThinkingBox data content does not match the pinned release")


def _validate_root(root: Path, *, require_stamp: bool) -> DataBundle:
    # [FUNCTION DOC]
    # HELPER: checks required directories, LICENSE.txt and the manifest exist (DatasetError otherwise), reads
    # UIDs, optionally requires the stamp file to equal the canonical stamp, computes hashes and builds a
    # DataBundle. With require_stamp, also enforces canonical content and fills release name and revision.
    # Returns the DataBundle.
    dataset = root / "dataset"
    for relative in (
        Path("dataset/agent"),
        Path("dataset/scenario"),
        Path("dataset/test_case"),
        Path("servers"),
        Path("support"),
        Path("releases"),
    ):
        if not (root / relative).is_dir():
            raise DatasetError(f"ThinkingBox data is missing {relative}")
    if not (root / "LICENSE.txt").is_file():
        raise DatasetError("ThinkingBox data is missing LICENSE.txt")
    manifest = root / _MANIFEST_PATH
    if not manifest.is_file():
        raise DatasetError(f"ThinkingBox manifest not found at {DATA_MANIFEST_PATH}")
    test_uids = _read_test_uids(manifest)
    if require_stamp:
        try:
            stamp = json.loads((root / _STAMP).read_text(encoding="utf-8"))
        except Exception as exc:
            raise DatasetError("ThinkingBox cache stamp is missing or invalid") from exc
        if stamp != _canonical_stamp():
            raise DatasetError("ThinkingBox cache stamp does not match pinned data")
    bundle = DataBundle(
        root=root,
        dataset_dir=dataset,
        manifest_path=manifest,
        bundle_sha256=_bundle_sha256(root),
        manifest_sha256=_sha256_file(manifest),
        manifest_uids_sha256=_sha256_json(test_uids),
        task_count=len(test_uids),
    )
    if require_stamp:
        _require_canonical_content(bundle)
        return replace(
            bundle,
            release_name=DATA_RELEASE_NAME,
            revision=DATA_COMMIT,
        )
    return bundle


def load_test_uids(bundle: DataBundle) -> list[str]:
    # [FUNCTION DOC]
    # Returns the bundle manifest's ordered unique UIDs (DatasetError if invalid).
    """Load the validated bundle's ordered task manifest.

    Args:
        bundle ([`DataBundle`]):
            Validated canonical or local executable data bundle.

    Returns:
        `list[str]`:
            Non-empty ordered unique task UIDs.
    """
    return _read_test_uids(bundle.manifest_path)


def _download(url: str, destination: Path, timeout: float) -> None:
    # [FUNCTION DOC]
    # HELPER: streams url to destination in 1 MiB chunks with a custom User-Agent and timeout. Raises
    # DatasetError if the size limit is exceeded or on any download error. Side effects: network, file write.
    size = 0
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OpenEnv-ThinkingBox/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > _MAX_ARCHIVE_BYTES:
                        raise DatasetError(
                            "ThinkingBox data archive is unexpectedly large"
                        )
                    output.write(chunk)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError("ThinkingBox data archive download failed") from exc


def _validated_archive_path(name: str) -> PurePosixPath:
    # [FUNCTION DOC]
    # HELPER: returns the member name as a PurePosixPath after rejecting backslashes, NULs, empty/./.. parts,
    # absolute POSIX/Windows paths and drive letters (DatasetError).
    if "\\" in name or "\0" in name:
        raise DatasetError("ThinkingBox archive contains an unsafe path")

    normalized = name.rstrip("/")
    raw_parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in raw_parts):
        raise DatasetError("ThinkingBox archive contains an unsafe path")

    path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise DatasetError("ThinkingBox archive contains an unsafe path")
    for part in path.parts:
        windows_part = PureWindowsPath(part)
        if windows_part.is_absolute() or windows_part.drive or windows_part.root:
            raise DatasetError("ThinkingBox archive contains an unsafe path")
    return path


def _archive_target(destination: Path, relative_parts: tuple[str, ...]) -> Path:
    # [FUNCTION DOC]
    # HELPER: returns destination joined with the relative parts; raises DatasetError if the resolved target is
    # not strictly inside destination.
    target = destination.joinpath(*relative_parts)
    resolved = target.resolve()
    if resolved == destination or destination not in resolved.parents:
        raise DatasetError("ThinkingBox archive contains an unsafe path")
    return target


def _extract_archive(archive: Path, destination: Path) -> None:
    # [FUNCTION DOC]
    # HELPER: safely extracts a .tar.gz: all members must share one root dir; only members under allowed
    # top-level names are extracted (others skipped); symlinks, hard links and special files are rejected;
    # total extracted file size is capped. Writes files and dirs under destination. Raises DatasetError on
    # violations.
    extracted_bytes = 0
    destination = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as source:
        members = source.getmembers()
        paths = [_validated_archive_path(member.name) for member in members]
        roots = {path.parts[0] for path in paths}
        if len(roots) != 1:
            raise DatasetError("ThinkingBox archive has an invalid root layout")
        archive_root = next(iter(roots))

        for member, path in zip(members, paths, strict=True):
            if path.parts[0] != archive_root:
                raise DatasetError("ThinkingBox archive root changed unexpectedly")
            relative_parts = path.parts[1:]
            if not relative_parts or relative_parts[0] not in _ALLOWED_TOP_LEVELS:
                continue
            if (
                member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise DatasetError("ThinkingBox archive contains an unsafe member")

            target = _archive_target(destination, relative_parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            extracted_bytes += member.size
            if extracted_bytes > _MAX_EXTRACTED_BYTES:
                raise DatasetError(
                    "ThinkingBox archive expands beyond the safety limit"
                )
            stream = source.extractfile(member)
            if stream is None:
                raise DatasetError("ThinkingBox archive member could not be read")
            with stream, target.open("wb") as output:
                shutil.copyfileobj(stream, output)


def _cache_destination(cache_root: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: returns the cache directory for the pinned release: resolved cache root joined with
    # "<release>-<commit 12 chars>-<bundle hash 16 chars>".
    key = f"{DATA_RELEASE_NAME}-{DATA_COMMIT[:12]}-{DATA_BUNDLE_SHA256[:16]}"
    return cache_root.expanduser().resolve() / key


def _publication_lock(destination: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: returns the sibling lock-file path ".<name>.lock" for a destination dir.
    return destination.parent / f".{destination.name}.lock"


def ensure_data(
    *,
    cache_root: Path | None = None,
    archive_url: str = DATA_ARCHIVE_URL,
    timeout: float = DATA_TIMEOUT,
) -> DataBundle:
    # [FUNCTION DOC]
    # Returns the canonical cached bundle, populating the cache if needed. Fast path: validate existing cache.
    # Otherwise, under the process thread lock and an inter-process file lock, re-checks, then downloads into a
    # unique partial workspace, extracts, validates (canonical content), writes the stamp, revalidates, and
    # atomically replaces the destination (removing an invalid existing one). The workspace is always deleted.
    # Raises DatasetError on any validation/download failure. Side effects: network, filesystem.
    """Download, verify, safely extract, and publish canonical data atomically.

    Args:
        cache_root (`pathlib.Path`, *optional*):
            Cache parent, defaults to the configured ThinkingBox cache.
        archive_url (`str`, *optional*, defaults to the pinned commit archive):
            Archive URL used to populate a missing cache.
        timeout (`float`, *optional*, defaults to the configured data timeout):
            Download timeout in seconds.

    Returns:
        [`DataBundle`]:
            Validated canonical bundle from the atomic cache.
    """
    destination = _cache_destination(cache_root or DATA_CACHE)
    try:
        return _validate_root(destination, require_stamp=True)
    except DatasetError:
        pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock:
        with FileLock(str(_publication_lock(destination))):
            try:
                return _validate_root(destination, require_stamp=True)
            except DatasetError:
                pass

            workspace = (
                destination.parent / f".{destination.name}.partial-{uuid4().hex}"
            )
            extracted = workspace / "content"
            archive = workspace / "archive.tar.gz"
            workspace.mkdir(parents=True)
            extracted.mkdir()
            try:
                _download(archive_url, archive, timeout)
                _extract_archive(archive, extracted)
                extracted_bundle = _validate_root(extracted, require_stamp=False)
                _require_canonical_content(extracted_bundle)
                (extracted / _STAMP).write_text(
                    json.dumps(_canonical_stamp(), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                _validate_root(extracted, require_stamp=True)

                try:
                    return _validate_root(destination, require_stamp=True)
                except DatasetError:
                    pass
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(extracted, destination)
                return _validate_root(destination, require_stamp=True)
            finally:
                shutil.rmtree(workspace, ignore_errors=True)


def resolve_data_bundle(
    configured: str | None = None,
    *,
    cache_root: Path | None = None,
    archive_url: str = DATA_ARCHIVE_URL,
    timeout: float = DATA_TIMEOUT,
) -> DataBundle:
    # [FUNCTION DOC]
    # Uses configured or OPENENV_TB_DATASET; if empty, returns ensure_data(). Otherwise accepts a data root
    # (containing dataset/) or a dataset dir itself (containing agent, scenario, test_case; root is its parent),
    # else raises DatasetError. Validates with the stamp requirement only if a stamp file exists.
    # Returns the DataBundle.
    """Resolve an explicit local bundle or the pinned atomic cache.

    Args:
        configured (`str`, *optional*):
            Data root or `dataset/` path. When omitted, use the canonical cache.

    Returns:
        [`DataBundle`]:
            Validated local or canonical executable data.
    """
    selected = (configured or DATASET_PATH).strip()
    if not selected:
        return ensure_data(
            cache_root=cache_root,
            archive_url=archive_url,
            timeout=timeout,
        )

    path = Path(selected).expanduser().resolve()
    if (path / "dataset").is_dir():
        root = path
    elif all((path / name).is_dir() for name in ("agent", "scenario", "test_case")):
        root = path.parent
    else:
        raise DatasetError(
            "OPENENV_TB_DATASET must point to a thinkingbox-data root or dataset/"
        )
    stamp_path = root / _STAMP
    if stamp_path.exists():
        return _validate_root(root, require_stamp=True)
    return _validate_root(root, require_stamp=False)


def data_ready(
    configured: str | None = None,
    *,
    cache_root: Path | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Returns True if the configured data (or the canonical cache) validates without downloading, False on
    # DatasetError. Other exceptions propagate.
    """Return whether configured or cached data is already valid.

    Args:
        configured (`str`, *optional*):
            Explicit data root or `dataset/` path.

    Returns:
        `bool`:
            `True` only when validation succeeds without downloading.
    """
    selected = (configured or DATASET_PATH).strip()
    try:
        if selected:
            resolve_data_bundle(selected)
        else:
            _validate_root(
                _cache_destination(cache_root or DATA_CACHE),
                require_stamp=True,
            )
    except DatasetError:
        return False
    return True
