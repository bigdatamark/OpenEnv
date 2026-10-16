# [MODULE DOC]
# Parses source-level metadata declarations used when building the discovery catalog:
# the optional per-environment discovery.json (DiscoveryDeclaration), pyproject.toml
# [project] tables, README YAML frontmatter, license values and the openenv dependency
# requirement. All reads go through a GitMetadataSource; malformed input raises MetadataError.
# SPDX-License-Identifier: BSD-3-Clause

"""Source declarations used by discovery, independent of runtime quality grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import tomli
import yaml
from packaging.licenses import canonicalize_license_expression, InvalidLicenseExpression
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from pydantic import Field, field_validator

from .errors import CatalogError
from .models import (
    NonEmpty,
    ProfileModel,
    RelativePath,
    RepresentativeQueries,
    SIMULATION_CONTROLS,
)
from .repository import GitMetadataSource, MetadataError
from .serialization import parse_json


class ToolDeclaration(ProfileModel):
    # [CLASS DOC]
    # Pydantic profile model (strict, frozen, no extra keys) declaring agent-facing tools:
    # protocol name, 1-256 tool names, and the repository-relative source file evidencing them.
    protocol: NonEmpty
    names: list[NonEmpty] = Field(min_length=1, max_length=256)
    source: RelativePath

    @field_validator("names")
    @classmethod
    def task_tools_only(cls, value: list[str]) -> list[str]:
        # [FUNCTION DOC]
        # Validator for names: rejects duplicates and any name whose casefolded form is a
        # simulation control (reset/step/state/get_state), raising ValueError. Returns names unchanged.
        if len(value) != len(set(value)):
            raise ValueError("tool declarations must not contain duplicate names")
        if any(name.casefold() in SIMULATION_CONTROLS for name in value):
            raise ValueError("simulation controls cannot be agent tools")
        return value


class DiscoveryDeclaration(ProfileModel):
    # [CLASS DOC]
    # Pydantic model of an environment's optional discovery.json: description, tags,
    # representative queries (0 or 2-5), license (+ optional license_source path),
    # artifact_availability (default "resolvable") and optional agent_tools ToolDeclaration.
    # All fields optional; an empty declaration is valid.
    description: NonEmpty | None = None
    tags: list[NonEmpty] = Field(default_factory=list)
    representative_queries: RepresentativeQueries = Field(default_factory=list)
    license: NonEmpty | None = None
    license_source: RelativePath | None = None
    artifact_availability: Literal["resolvable", "external", "unknown"] = "resolvable"
    agent_tools: ToolDeclaration | None = None

    @field_validator("license")
    @classmethod
    def license_expression(cls, value: str | None) -> str | None:
        # [FUNCTION DOC]
        # Validator: passes through None/"unknown"/"other"; otherwise returns the canonical
        # SPDX expression (raises on an invalid expression, surfaced as a validation error).
        if value is None or value in {"unknown", "other"}:
            return value
        return str(canonicalize_license_expression(value))


class _UniqueYamlLoader(yaml.SafeLoader):
    # [CLASS DOC]
    # HELPER: PyYAML SafeLoader variant that rejects duplicate or non-string mapping keys.
    def construct_mapping(self, node, deep=False):
        # [FUNCTION DOC]
        # HELPER: builds a dict from a YAML mapping node, raising ConstructorError if the node
        # is not a mapping, or a key is not a string or repeats. Returns the dict.
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, "Expected mapping", node.start_mark
            )
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    "Mapping keys must be unique strings",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def yaml_mapping(value: str, path: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Parses YAML text with the unique-key loader and requires a top-level mapping.
    # Inputs: value - YAML text; path - used only in error messages.
    # Returns: the dict. Raises: MetadataError("invalid_metadata") on YAML errors,
    # excessive nesting (RecursionError) or a non-mapping result.
    try:
        result = yaml.load(value, Loader=_UniqueYamlLoader)
    except (yaml.YAMLError, RecursionError) as error:
        raise MetadataError(
            "invalid_metadata", f"Invalid YAML metadata: {path}"
        ) from error
    if not isinstance(result, dict):
        raise MetadataError("invalid_metadata", f"Metadata must be a mapping: {path}")
    return result


def project_metadata(source: GitMetadataSource, path: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Reads a pyproject.toml at path from the git source and returns its [project] table
    # (empty dict if the file is missing or has no project table). Raises MetadataError on
    # invalid TOML or a non-mapping project value.
    text = source.read(path)
    if text is None:
        return {}
    try:
        parsed = tomli.loads(text)
    except tomli.TOMLDecodeError as error:
        raise MetadataError(
            "invalid_metadata", f"Invalid package metadata: {path}"
        ) from error
    project = parsed.get("project", {})
    if not isinstance(project, dict):
        raise MetadataError(
            "invalid_metadata", f"Project metadata must be a mapping: {path}"
        )
    return project


def readme_metadata(source: GitMetadataSource, path: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Reads a README at path and returns its YAML frontmatter as a dict (empty dict if the
    # file is missing or does not start with a "---" line). Raises MetadataError when the
    # frontmatter has no closing "---" line or its YAML is invalid.
    text = source.read(path)
    if text is None or not text.startswith(("---\n", "---\r")):
        return {}
    lines = text.splitlines()
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise MetadataError(
            "invalid_metadata", f"Unclosed README frontmatter: {path}"
        ) from error
    return yaml_mapping("\n".join(lines[1:end]), path)


def discovery_declaration(source: GitMetadataSource, path: str) -> DiscoveryDeclaration:
    # [FUNCTION DOC]
    # Reads discovery.json at path; returns a default DiscoveryDeclaration if absent.
    # Parses strict JSON (no duplicate keys / NaN) and validates it. Raises MetadataError on
    # bad JSON, and pydantic ValidationError if the content violates the model.
    text = source.read(path)
    if text is None:
        return DiscoveryDeclaration()
    try:
        raw = parse_json(text)
    except CatalogError as error:
        raise MetadataError(
            "invalid_metadata", f"Invalid discovery declaration: {path}"
        ) from error
    return DiscoveryDeclaration.model_validate(raw)


@dataclass(frozen=True)
class _ParsedLicense:
    # [CLASS DOC]
    # HELPER: immutable record of a parsed license: SPDX expression or "unknown"/"other",
    # plus the normalized custom license text when expression is "other".
    expression: str
    text: str | None = None


def license_declaration(value: object) -> _ParsedLicense | None:
    # [FUNCTION DOC]
    # Normalizes a pyproject/README license value.
    # Returns None for None; for a table with only "file" returns "unknown" (a file pointer
    # does not identify a license); for a table with "text" uses that text; non-string or
    # blank -> "unknown"; "unknown"/"other" kept; valid SPDX -> canonical expression;
    # otherwise "other" with the text (newlines normalized to LF).
    # Raises: MetadataError if the table has both "file" and "text".
    if value is None:
        return None
    if isinstance(value, dict):
        if "file" in value and "text" in value:
            raise MetadataError(
                "invalid_metadata",
                "License file and text declarations are mutually exclusive",
            )
        value = value.get("text")
        if value is None:
            # A file pointer alone does not identify an SPDX or custom license.
            return _ParsedLicense("unknown")
    if not isinstance(value, str) or not value.strip():
        return _ParsedLicense("unknown")
    value = value.strip()
    if value in {"unknown", "other"}:
        return _ParsedLicense(value)
    try:
        return _ParsedLicense(str(canonicalize_license_expression(value)))
    except InvalidLicenseExpression:
        return _ParsedLicense("other", value.replace("\r\n", "\n").replace("\r", "\n"))


def framework_requirement(project: dict[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # Finds the single dependency in project["dependencies"] whose canonical name is
    # "openenv" and returns the original requirement string, or None if absent.
    # Raises MetadataError if dependencies is not a list, contains a non-string or an
    # unparsable requirement, or there are multiple openenv requirements.
    requirements = []
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise MetadataError("invalid_metadata", "Package dependencies must be an array")
    for dependency in dependencies:
        if not isinstance(dependency, str):
            raise MetadataError(
                "invalid_metadata", "Dependency declarations must be strings"
            )
        try:
            parsed = Requirement(dependency)
        except InvalidRequirement as error:
            raise MetadataError(
                "invalid_metadata", "Invalid package dependency declaration"
            ) from error
        if canonicalize_name(parsed.name) == "openenv":
            requirements.append(dependency)
    if len(requirements) > 1:
        raise MetadataError(
            "invalid_metadata",
            "Multiple OpenEnv requirements need an explicit resolution",
        )
    return requirements[0] if requirements else None
