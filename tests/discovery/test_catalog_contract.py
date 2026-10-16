# [MODULE DOC]
# Contract tests for the discovery profile (0.1-draft): EnvironmentCard
# validation rules (artifact/source binding, safe repository-relative paths,
# interface descriptors, licence evidence, required wire fields), catalog
# loading with digest tamper detection, shipped JSON Schemas staying in sync
# with the Pydantic models, and representative-query count limits.
# Module constants: REVISION is a 40-char fake commit; UNSAFE_LINE_PATHS are
# paths containing control or line/paragraph separator characters.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError
from openenv.discovery import CatalogError, EnvironmentCard, load_catalog
from openenv.discovery.metadata import DiscoveryDeclaration
from openenv.discovery.models import DiscoveryEntry
from pydantic import ValidationError


REVISION = "a" * 40
UNSAFE_LINE_PATHS = [
    f"envs/control-{chr(codepoint)}" for codepoint in [*range(0x20), *range(0x7F, 0xA0)]
] + ["envs/line-\u2028separator", "envs/paragraph-\u2029separator"]


@pytest.fixture
def card() -> dict:
    # [FUNCTION DOC]
    # Fixture: a valid environment card mapping for echo_env with one git
    # artifact and orchestration + declared MCP interfaces.
    return {
        "schema_version": "0.1-draft",
        "name": "echo_env",
        "description": "Echo messages for client smoke testing.",
        "owner": {"authority": "github.com", "id": "example"},
        "source": {
            "provider": "github",
            "id": "example/environments",
            "uri": "https://github.com/example/environments.git",
            "path": "envs/echo_env",
            "revision": REVISION,
        },
        "artifact_availability": "resolvable",
        "artifacts": [
            {
                "kind": "git",
                "uri": "https://github.com/example/environments.git",
                "path": "envs/echo_env",
                "revision": REVISION,
            }
        ],
        "license": "BSD-3-Clause",
        "manifest_spec_version": 1,
        "framework_requirement": "openenv>=0.3.1",
        "interfaces": [
            {"role": "orchestration", "protocol": "openenv"},
            {
                "role": "agent-tools",
                "protocol": "mcp",
                "status": "declared",
                "source_revision": REVISION,
            },
        ],
    }


@pytest.fixture
def card_schema() -> Draft202012Validator:
    # [FUNCTION DOC]
    # Fixture: Draft 2020-12 validator for the packaged environment-card schema.
    root = files("openenv.discovery").joinpath("schemas", "0.1-draft")
    schema = json.loads(root.joinpath("environment-card.schema.json").read_text())
    return Draft202012Validator(schema)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "b" * 40),
        ("path", "envs/another"),
        ("uri", "https://github.com/x/y.git"),
    ],
)
def test_artifact_must_bind_to_the_same_selected_subject(card, field, value):
    # [FUNCTION DOC]
    # Changing artifact revision/path/uri away from the source fails validation.
    card["artifacts"][0][field] = value
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


def test_tool_declaration_cannot_borrow_another_revision(card):
    # [FUNCTION DOC]
    # A tool interface with a different source_revision fails validation.
    card["interfaces"][1]["source_revision"] = "b" * 40
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


@pytest.mark.parametrize(
    "path",
    [
        "/envs/echo",
        "../echo",
        "envs/../echo",
        "envs\\echo",
        "envs//echo",
        "envs/echo/",
        "envs/./echo",
        *UNSAFE_LINE_PATHS,
        "envs/trailing\n",
        ".\n",
        "..\n",
        "envs/\n",
        "envs/.\n",
        "envs/..\n",
    ],
)
def test_environment_locator_is_a_safe_repository_relative_path(
    card, path, card_schema
):
    # [FUNCTION DOC]
    # Absolute, traversing, backslash, double-slash, trailing-slash, dot-segment
    # and control-character paths are rejected by both model and JSON schema.
    card["source"]["path"] = path
    card["artifacts"][0]["path"] = path
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)
    with pytest.raises(SchemaValidationError):
        card_schema.validate(card)


@pytest.mark.parametrize(
    "path",
    [
        ".",
        "envs/echo",
        "envs/.hidden",
        "envs/...",
        "envs/a..b",
        "envs/with spaces",
        "envs/café",
        "envs/東京",
    ],
)
def test_schema_and_model_preserve_valid_relative_locators(card, path, card_schema):
    # [FUNCTION DOC]
    # Valid relative paths (including ".", hidden, unicode, spaces) pass both
    # and are preserved.
    card["source"]["path"] = path
    card["artifacts"][0]["path"] = path
    assert EnvironmentCard.model_validate(card).source.path == path
    card_schema.validate(card)


def test_claimed_validated_tools_are_unsupported_in_the_declaration_profile(card):
    # [FUNCTION DOC]
    # Interface status "validated" is rejected.
    card["interfaces"][1]["status"] = "validated"
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


def test_source_manifest_marker_does_not_become_a_protocol_version(card):
    # [FUNCTION DOC]
    # manifest_spec_version is kept top-level; interfaces have no version and
    # adding one is rejected.
    parsed = EnvironmentCard.model_validate(card)
    assert parsed.manifest_spec_version == 1
    assert "version" not in parsed.interfaces[0].model_dump(exclude_none=True)
    card["interfaces"][0]["version"] = "1"
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


@pytest.mark.parametrize(
    "interfaces",
    [
        [],
        [
            {
                "role": "agent-tools",
                "protocol": "mcp",
                "status": "declared",
                "source_revision": REVISION,
            }
        ],
    ],
)
def test_orchestration_descriptor_is_required(card, interfaces, card_schema):
    # [FUNCTION DOC]
    # Cards without an orchestration interface are rejected by model and schema.
    card["interfaces"] = interfaces
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)
    with pytest.raises(SchemaValidationError):
        card_schema.validate(card)


def test_orchestration_must_not_be_duplicated(card, card_schema):
    # [FUNCTION DOC]
    # Two orchestration interfaces are rejected by model and schema.
    card["interfaces"].append(copy.deepcopy(card["interfaces"][0]))
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)
    with pytest.raises(SchemaValidationError):
        card_schema.validate(card)


@pytest.mark.parametrize("availability", ["external", "unknown"])
def test_external_subject_does_not_claim_artifact_distribution(
    card, availability, card_schema
):
    # [FUNCTION DOC]
    # external/unknown availability with artifacts is rejected; without artifacts
    # it validates.
    card["artifact_availability"] = availability
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)
    with pytest.raises(SchemaValidationError):
        card_schema.validate(card)
    card["artifacts"] = []
    assert EnvironmentCard.model_validate(card).source.revision == REVISION
    card_schema.validate(card)


@pytest.mark.parametrize("missing", [False, True], ids=["empty", "missing"])
def test_resolvable_subject_requires_an_artifact(card, missing, card_schema):
    # [FUNCTION DOC]
    # Resolvable availability with empty or missing artifacts is rejected.
    if missing:
        del card["artifacts"]
    else:
        card["artifacts"] = []
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)
    with pytest.raises(SchemaValidationError):
        card_schema.validate(card)


@pytest.mark.parametrize("null_evidence", [False, True], ids=["missing", "null"])
def test_other_license_requires_evidence_and_unknown_remains_unknown(
    card, null_evidence, card_schema
):
    # [FUNCTION DOC]
    # license "other" without license_url (missing or null) is rejected; "unknown"
    # is accepted as-is.
    card["license"] = "other"
    if null_evidence:
        card["license_url"] = None
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)
    with pytest.raises(SchemaValidationError):
        card_schema.validate(card)
    card["license"] = "unknown"
    assert EnvironmentCard.model_validate(card).license == "unknown"
    card_schema.validate(card)


def test_schema_accepts_an_evidenced_custom_license(card, card_schema):
    # [FUNCTION DOC]
    # license "other" with a license_url is accepted.
    card["license"] = "other"
    card["license_url"] = "https://example.org/license"
    assert EnvironmentCard.model_validate(card).license == "other"
    card_schema.validate(card)


def test_invalid_package_requirement_is_not_presented_as_compatibility(card):
    # [FUNCTION DOC]
    # An unparsable framework_requirement is rejected.
    card["framework_requirement"] = "openenv >=> 1"
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


def test_a_publisher_policy_flag_is_not_part_of_the_card(card):
    # [FUNCTION DOC]
    # Unknown field requires_explicit_trust is rejected (extra fields forbidden).
    card["requires_explicit_trust"] = False
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


@pytest.mark.parametrize(
    "location",
    [
        ["schema_version"],
        ["source", "provider"],
        ["artifacts", 0, "kind"],
        ["interfaces", 0, "protocol"],
        ["interfaces", 1, "status"],
    ],
)
def test_required_wire_claims_are_not_invented_from_defaults(card, location):
    # [FUNCTION DOC]
    # Deleting each listed required nested key makes validation fail.
    parent = card
    for key in location[:-1]:
        parent = parent[key]
    del parent[location[-1]]
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


@pytest.mark.parametrize("field", ["name", "description"])
def test_whitespace_is_not_a_useful_required_description(card, field):
    # [FUNCTION DOC]
    # Whitespace-only name/description is rejected.
    card[field] = "   "
    with pytest.raises(ValidationError):
        EnvironmentCard.model_validate(card)


def test_loader_rejects_identifier_only_and_incomplete_records(tmp_path: Path):
    # [FUNCTION DOC]
    # load_catalog raises CatalogError for a file with an incomplete entry.
    path = tmp_path / "partial.json"
    path.write_text('{"entries":[{"identifier":"urn:air:example.org:openenv:echo"}]}')
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_loader_rejects_tampering_even_when_json_is_well_formed(tmp_path: Path, card):
    # [FUNCTION DOC]
    # Builds a catalog with a sha256 digest over canonical JSON (sorted keys,
    # compact separators); it loads, then after editing an entry without
    # re-digesting load_catalog raises CatalogError mentioning "digest".
    payload = {
        "schema_version": "0.1-draft",
        "publisher": "example.org",
        "source": {
            key: value for key, value in card["source"].items() if key != "path"
        },
        "generator": {"name": "openenv-git-catalog", "version": "1"},
        "inventory": {"root": "envs", "paths": ["envs/echo_env"]},
        "complete": True,
        "issues": [],
        "entries": [
            {
                "@context": "https://agenticresourcediscovery.org/context/v1",
                "identifier": f"urn:air:example.org:openenv:echo:{REVISION}",
                "displayName": "Echo",
                "type": "application/vnd.openenv.environment-card+json",
                "data": card,
                "description": card["description"],
                "tags": [],
                "capabilities": [],
                "representativeQueries": [],
                "metadata": {},
            }
        ],
    }
    payload["digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
    )
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    assert load_catalog(path).complete
    payload["entries"][0]["data"]["description"] = "A replaced description."
    path.write_text(json.dumps(payload))
    with pytest.raises(CatalogError, match="digest"):
        load_catalog(path)


def test_schema_files_ship_with_the_profile(card):
    # [FUNCTION DOC]
    # Packaged card/catalog/declaration schemas are valid Draft 2020-12, share
    # $schema, pin schema_version, and validate the models' dumps.
    root = files("openenv.discovery").joinpath("schemas", "0.1-draft")
    card_schema = json.loads(root.joinpath("environment-card.schema.json").read_text())
    catalog_schema = json.loads(root.joinpath("catalog.schema.json").read_text())
    assert card_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert catalog_schema["$schema"] == card_schema["$schema"]
    assert card_schema["properties"]["schema_version"]["const"] == "0.1-draft"
    assert card_schema["properties"]["source"]["$ref"].startswith("#/$defs/")
    declaration_schema = json.loads(
        root.joinpath("declaration.schema.json").read_text()
    )
    for schema in (card_schema, catalog_schema, declaration_schema):
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(card_schema).validate(
        EnvironmentCard.model_validate(card).model_dump(by_alias=True, mode="json")
    )
    Draft202012Validator(declaration_schema).validate(
        DiscoveryDeclaration().model_dump(mode="json")
    )


@pytest.mark.parametrize(
    ("count", "valid"), [(0, True), (1, False), (2, True), (5, True), (6, False)]
)
def test_query_hint_counts_agree_across_declarations_entries_and_schemas(
    card, count, valid
):
    # [FUNCTION DOC]
    # For counts 0..6, DiscoveryDeclaration, DiscoveryEntry and their JSON
    # schemas agree that 0 or 2-5 representative queries are valid and 1 or 6 are not.
    queries = [f"query {index}" for index in range(count)]
    declaration = {"representative_queries": queries}
    entry = {
        "identifier": f"urn:air:example.org:openenv:echo:{REVISION}",
        "displayName": "Echo",
        "description": card["description"],
        "data": card,
        "representativeQueries": queries,
    }
    for model, payload in (
        (DiscoveryDeclaration, declaration),
        (DiscoveryEntry, entry),
    ):
        if valid:
            assert model.model_validate(payload).representative_queries == queries
        else:
            with pytest.raises(ValidationError):
                model.model_validate(payload)

    root = files("openenv.discovery").joinpath("schemas", "0.1-draft")
    declaration_schema = json.loads(
        root.joinpath("declaration.schema.json").read_text()
    )
    catalog_schema = json.loads(root.joinpath("catalog.schema.json").read_text())
    entry_schema = {
        "$defs": catalog_schema["$defs"],
        "$ref": "#/$defs/DiscoveryEntry",
    }
    assert Draft202012Validator(declaration_schema).is_valid(declaration) is valid
    assert Draft202012Validator(entry_schema).is_valid(entry) is valid
