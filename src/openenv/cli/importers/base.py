# [MODULE DOC]
# Shared building blocks for source importers: exclusion rules for copying a
# foreign source tree (VCS/cache/build/venv directories, compiled and key file
# suffixes, credential/secret file names and patterns), dependency collection
# and merging, vendoring helpers that copy a source tree into
# "<env>/vendor/<name>" as an importable package, template rendering from the
# importers.templates package, the DetectedEnvironment / ImportPackageContext
# records, the EnvironmentImporter protocol, and ImporterRegistry which looks up
# and runs importers by source type. Importers detect environments statically
# (without importing user code).
"""Shared importer registry types."""

from __future__ import annotations

import fnmatch
import hashlib
import shutil
import textwrap
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Template
from typing import Protocol

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

_EXCLUDED_FILE_SUFFIXES = {
    ".dll",
    ".dylib",
    ".key",
    ".p12",
    ".pfx",
    ".pem",
    ".pyd",
    ".pyc",
    ".pyo",
    ".so",
}

_EXCLUDED_FILE_NAMES = {
    ".env",
    ".netrc",
    "credentials.json",
    "secrets.json",
    "secrets.toml",
    "secrets.yaml",
    "secrets.yml",
}

_EXCLUDED_FILE_PATTERNS = {
    ".env.*",
    "*_secret.*",
    "*_secrets.*",
    "id_ed25519*",
    "id_rsa*",
}


def _is_excluded(path: Path) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if any path component is an excluded directory name, or
    # the file name is an excluded name, has an excluded suffix, or matches an
    # excluded glob pattern (e.g. ".env.*", "id_rsa*", "*_secret.*").
    if any(part in _EXCLUDED_DIRS for part in path.parts):
        return True
    name = path.name
    return (
        name in _EXCLUDED_FILE_NAMES
        or path.suffix in _EXCLUDED_FILE_SUFFIXES
        or any(fnmatch.fnmatch(name, pattern) for pattern in _EXCLUDED_FILE_PATTERNS)
    )


def _append_unique_dependency(dependencies: list[str], dependency: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Strips dependency and appends it to dependencies (in place) unless it
    # is empty or a dependency with the same normalized name is already present.
    dependency = dependency.strip()
    if not dependency:
        return
    dependency_names = {_dependency_name(existing) for existing in dependencies}
    if _dependency_name(dependency) not in dependency_names:
        dependencies.append(dependency)


def _requirement_line_dependency(line: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Converts one requirements.txt line into a portable requirement string
    # or None. Returns None for blank lines, comments, option lines starting with
    # "-", local paths (starting with "." or "/"), and VCS/URL requirements; strips
    # trailing " #" comments.
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return None
    if " #" in line:
        line = line.split(" #", 1)[0].rstrip()
    if line.startswith((".", "/")):
        return None
    if line.startswith(("git+", "http://", "https://")):
        return None
    return line or None


def collect_source_dependencies(source: Path) -> list[str]:
    # [FUNCTION DOC]
    # Collects dependencies from source/pyproject.toml ([project] dependencies,
    # string entries only; an invalid TOML file is treated as empty) and then
    # source/requirements.txt (filtered by _requirement_line_dependency),
    # de-duplicated by normalized name, preserving first-seen order.
    # Returns: list of requirement strings. Raises OSError on unreadable files.
    """Collect portable Python dependencies declared by a source tree."""
    dependencies: list[str] = []

    pyproject = source / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            data = {}
        project_dependencies = data.get("project", {}).get("dependencies") or []
        for dependency in project_dependencies:
            if isinstance(dependency, str):
                _append_unique_dependency(dependencies, dependency)

    requirements = source / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            dependency = _requirement_line_dependency(line)
            if dependency:
                _append_unique_dependency(dependencies, dependency)

    return dependencies


def iter_python_files(source: Path) -> list[Path]:
    # [FUNCTION DOC]
    # Returns all .py files under source recursively, sorted, excluding those whose
    # relative path is excluded by _is_excluded.
    return [
        path
        for path in sorted(source.rglob("*.py"))
        if not _is_excluded(path.relative_to(source))
    ]


def module_path(source: Path, file_path: Path) -> str:
    # [FUNCTION DOC]
    # Converts file_path (under source) into a dotted module path relative to
    # source, dropping the ".py" suffix and a trailing "__init__" component.
    # Raises ValueError if file_path is not under source.
    # KNOWN ISSUE: a top-level "__init__.py" yields an empty string.
    rel = file_path.relative_to(source).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def safe_vendor_dir_name(source: Path) -> str:
    # [FUNCTION DOC]
    # Derives a Python-identifier directory name for vendoring source: the source
    # folder name if it is already a valid identifier; otherwise the name with "-"
    # replaced by "_" plus "_<8 hex chars of sha256 of the resolved path>" if that
    # is an identifier; otherwise "source_<hash>".
    raw_name = source.name.strip()
    name = raw_name.replace("-", "_")
    if name.isidentifier() and name == raw_name:
        return name
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:8]
    if name.isidentifier():
        return f"{name}_{digest}"
    return f"source_{digest}"


def copy_source_tree(source: Path, destination: Path) -> None:
    # [FUNCTION DOC]
    # Copies source to destination with shutil.copytree, skipping entries whose
    # name alone is excluded by _is_excluded.
    # Raises: FileExistsError if destination already exists.
    # Side effects: filesystem writes.
    if destination.exists():
        raise FileExistsError(f"Vendored source path already exists: {destination}")

    def ignore(_dir: str, names: list[str]) -> set[str]:
        # [FUNCTION DOC]
        # HELPER: copytree ignore callback returning the subset of names that are
        # excluded (checked by name only, not full path).
        return {name for name in names if _is_excluded(Path(name))}

    shutil.copytree(source, destination, ignore=ignore)


def ensure_vendor_package(vendor_dir: Path) -> None:
    # [FUNCTION DOC]
    # Creates empty "__init__.py" files (if missing) in vendor_dir and its parent so
    # the vendored tree is importable as a package.
    for path in (vendor_dir.parent, vendor_dir):
        init_file = path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8", newline="\n")


def write_text(path: Path, content: str) -> None:
    # [FUNCTION DOC]
    # Writes content to path (creating parent directories) after dedenting and
    # stripping leading whitespace, as UTF-8 with LF newlines.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def render_importer_template(template_name: str, **values: str) -> str:
    # [FUNCTION DOC]
    # Reads template_name from the openenv.cli.importers.templates package resources
    # and substitutes values using string.Template "$name" placeholders.
    # Returns: the rendered text.
    # Raises: KeyError if a placeholder has no value; FileNotFoundError if the
    # template does not exist.
    """Render a Python source template used by source importers."""
    template = (
        resources.files("openenv.cli.importers.templates")
        .joinpath(template_name)
        .read_text(encoding="utf-8")
    )
    return Template(template).substitute(values)


def _dependency_name(requirement: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Normalizes a requirement to its distribution name: cut at the first
    # occurrence of any of "[", "<", ">", "=", "!", "~", ";", space; lowercase; "_"
    # replaced by "-".
    requirement = requirement.strip()
    for marker in ("[", "<", ">", "=", "!", "~", ";", " "):
        if marker in requirement:
            requirement = requirement.split(marker, 1)[0]
    return requirement.lower().replace("_", "-")


def append_dependency_files(
    env_dir: Path,
    env_name: str,
    dependencies: list[str],
) -> None:
    # [FUNCTION DOC]
    # Adds dependencies to a generated environment package.
    # If env_dir/server/requirements.txt exists, appends the dependencies whose
    # normalized names are not already listed (non-comment lines).
    # If env_dir/pyproject.toml exists, appends missing dependencies to
    # [project].dependencies and ensures [tool.setuptools.package-data][env_name]
    # contains "vendor/**/*", then rewrites the TOML file (formatting/comments of
    # the original are not preserved).
    # Returns: None. Side effects: file rewrites. Raises on invalid TOML.
    requirements = env_dir / "server" / "requirements.txt"
    if requirements.exists():
        content = requirements.read_text(encoding="utf-8")
        existing = {
            _dependency_name(line)
            for line in content.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        missing = []
        for dependency in dependencies:
            dependency_name = _dependency_name(dependency)
            if dependency_name not in existing:
                missing.append(dependency)
                existing.add(dependency_name)
        if missing:
            requirements.write_text(
                content.rstrip() + "\n" + "\n".join(missing) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    pyproject = env_dir / "pyproject.toml"
    if not pyproject.exists():
        return

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.setdefault("project", {})
    project_dependencies = list(project.get("dependencies") or [])
    existing = {_dependency_name(dependency) for dependency in project_dependencies}
    for dependency in dependencies:
        if _dependency_name(dependency) not in existing:
            project_dependencies.append(dependency)
            existing.add(_dependency_name(dependency))
    project["dependencies"] = project_dependencies

    tool = data.setdefault("tool", {})
    setuptools = tool.setdefault("setuptools", {})
    package_data = setuptools.setdefault("package-data", {})
    vendor_data = list(package_data.get(env_name) or [])
    if "vendor/**/*" not in vendor_data:
        vendor_data.append("vendor/**/*")
    package_data[env_name] = vendor_data

    pyproject.write_text(tomli_w.dumps(data), encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class DetectedEnvironment:
    # [CLASS DOC]
    # Immutable record of an environment class found in a source tree: the importer
    # source_type, the class name, its dotted module path and the file path.
    """A source environment class detected without importing user code."""

    source_type: str
    class_name: str
    module_path: str
    file_path: Path

    @property
    def qualified_name(self) -> str:
        # [FUNCTION DOC]
        # Returns "<module_path>:<class_name>".
        return f"{self.module_path}:{self.class_name}"


@dataclass(frozen=True)
class ImportPackageContext:
    # [CLASS DOC]
    # Immutable record returned by prepare_import_package: the class name prefix used
    # by the template (CamelCase env name) and the vendored directory name.
    """Shared details produced while scaffolding an imported environment package."""

    class_name_prefix: str
    vendor_dir: str


def prepare_import_package(
    source: Path, destination: Path, env_name: str
) -> ImportPackageContext:
    # [FUNCTION DOC]
    # Scaffolds an OpenEnv package at destination from the standard
    # "openenv_env" template (using the init command's template helpers and
    # replacements for env_name), then vendors source into
    # destination/vendor/<safe name> and makes it a package.
    # Returns: ImportPackageContext with the class name prefix and vendor dir name.
    # Raises: FileExistsError if the vendor path exists; template copy errors
    # propagate. Side effects: filesystem writes.
    from openenv.cli.commands.init import (
        _copy_template_directory,
        _create_template_replacements,
    )

    replacements = _create_template_replacements(env_name)
    _copy_template_directory(
        "openenv.cli.templates.openenv_env",
        "",
        destination,
        replacements,
        env_name,
    )

    vendor_dir = safe_vendor_dir_name(source)
    vendor_path = destination / "vendor" / vendor_dir
    copy_source_tree(source, vendor_path)
    ensure_vendor_package(vendor_path)

    return ImportPackageContext(
        class_name_prefix=replacements["__ENV_CLASS_NAME__"],
        vendor_dir=vendor_dir,
    )


class EnvironmentImporter(Protocol):
    # [CLASS DOC]
    # Structural protocol for an importer: a source_type string identifier and
    # detect/generate methods.
    source_type: str

    def detect(self, source: Path) -> list[DetectedEnvironment]:
        # [FUNCTION DOC]
        # Protocol method: returns environments in source supported by this importer.
        """Return environments supported by this importer."""
        ...

    def generate(
        self,
        *,
        source: Path,
        destination: Path,
        env_name: str,
        detected: DetectedEnvironment,
    ) -> None:
        # [FUNCTION DOC]
        # Protocol method: writes an OpenEnv wrapper package for detected into
        # destination named env_name.
        """Generate an OpenEnv wrapper package."""
        ...


class ImporterRegistry:
    # [CLASS DOC]
    # Holds an ordered list of importers and resolves them by source type.
    """Registry of deterministic environment importers."""

    def __init__(self, importers: list[EnvironmentImporter]):
        # [FUNCTION DOC]
        # Stores the given importers list (not copied).
        self._importers = importers

    @property
    def supported_types(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the source_type of each importer in order.
        return [importer.source_type for importer in self._importers]

    def get(self, source_type: str) -> EnvironmentImporter:
        # [FUNCTION DOC]
        # Returns the first importer whose source_type equals source_type.
        # Raises: ValueError listing supported types if none matches.
        for importer in self._importers:
            if importer.source_type == source_type:
                return importer
        supported = ", ".join(self.supported_types)
        raise ValueError(
            f"Unsupported source type {source_type!r}. Supported: {supported}"
        )

    def detect(
        self,
        source: Path,
        source_type: str | None = None,
    ) -> list[tuple[EnvironmentImporter, DetectedEnvironment]]:
        # [FUNCTION DOC]
        # Runs detect on the importer for source_type (if given, may raise ValueError)
        # or on all importers, and returns a list of (importer, detected) pairs.
        importers = [self.get(source_type)] if source_type else self._importers
        matches: list[tuple[EnvironmentImporter, DetectedEnvironment]] = []
        for importer in importers:
            for detected in importer.detect(source):
                matches.append((importer, detected))
        return matches
