# [MODULE DOC]
# Tests for write_runtime_bundle: the saved collector evidence and trace must
# let runtime graders replay to identical results, preserve failure metadata,
# redact secrets (marking the evidence as modified), bound deeply nested or
# non-finite subject JSON, and distinguish missing from null schemas.
import hashlib
import json
from dataclasses import replace

import pytest
from conftest import load_fixture_manifest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime import (
    ObservationSchemaGrader,
    RewardWellFormedGrader,
    StateContractGrader,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.report import ValidationReportV2
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.contracts import RuntimeEvidence, WireExchange
from openenv.validation.types import Lane, Level, SignatureKind, Verdict
from support.runtime import evidence, exchange


def report():
    # [FUNCTION DOC]
    # HELPER: Builds a minimal v2 ValidationReport (served_min_pass manifest,
    # local lane, static+runtime levels, no results, WARN verdict).
    return ValidationReportV2(
        report_schema_version="2",
        target="subject",
        source_digest="0" * 64,
        signature=SignatureKind.OPENENV_SERVED,
        manifest=NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        ),
        policy_version="v2",
        lane=Lane.LOCAL,
        levels_run=[Level.STATIC, Level.RUNTIME],
        results=[],
        verdict=Verdict.WARN,
    )


def measured():
    # [FUNCTION DOC]
    # HELPER: Builds RuntimeEvidence for a well-formed reset/state/step/state
    # episode (reward 0.5 on step) with an integer counter observation schema.
    return evidence(
        exchange(
            "reset",
            {"type": "reset", "data": {"episode_id": "recorded", "seed": 42}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 0}, "done": False, "reward": None},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {"type": "state", "data": {"episode_id": "recorded", "step_count": 0}},
        ),
        exchange(
            "step",
            {"type": "step", "data": {"increment": 1}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 1}, "done": False, "reward": 0.5},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {"type": "state", "data": {"episode_id": "recorded", "step_count": 1}},
        ),
        observation_schema={
            "type": "object",
            "properties": {"counter": {"type": "integer"}},
            "required": ["counter"],
        },
    )


def rebuild(directory):
    # [FUNCTION DOC]
    # HELPER: Reconstructs RuntimeEvidence from a written bundle directory using
    # collector-evidence.json and the trace file it names; schemas become None
    # when marked unavailable.
    # Returns: a RuntimeEvidence.
    # Side effects: reads files.
    metadata = json.loads((directory / "collector-evidence.json").read_text())
    trace = json.loads((directory / metadata["trace_file"]).read_text())
    return RuntimeEvidence(
        exchanges=tuple(
            WireExchange(
                row["operation"],
                json.dumps(row["request_json"]),
                json.dumps(row["response_json"]),
            )
            for row in trace
        ),
        observation_schema_json=(
            json.dumps(metadata["observation_schema"])
            if metadata["schema_available"]
            else None
        ),
        reset_observation_schema_json=(
            json.dumps(metadata["reset_observation_schema"])
            if metadata["reset_schema_available"]
            else None
        ),
        failure_phase=metadata["failure_phase"],
        failure_reason=metadata["failure_reason"],
    )


def test_saved_collector_evidence_replays_all_basic_graders(tmp_path):
    # [FUNCTION DOC]
    # Reward, observation-schema and state-contract graders yield the same status
    # on original and replayed evidence; metadata records version "1", complete,
    # not redacted, and the evidence file's hash appears in SHA256SUMS.
    original = measured()
    validation_report = report()
    write_runtime_bundle(tmp_path, validation_report, evidence=original)
    replay = rebuild(tmp_path)
    subject = Subject(
        tmp_path, validation_report.manifest, None, None, tmp_path, original
    )
    for grader in (
        RewardWellFormedGrader(),
        ObservationSchemaGrader(),
        StateContractGrader(),
    ):
        assert (
            grader.run(subject).status
            == grader.run(replace(subject, runtime_evidence=replay)).status
        )
    assert json.loads(replay.observation_schema_json) == json.loads(
        original.observation_schema_json
    )
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["evidence_schema_version"] == "1"
    assert metadata["complete"] is True
    assert metadata["redacted"] is False
    sums = (tmp_path / "SHA256SUMS").read_text()
    assert (
        hashlib.sha256((tmp_path / "collector-evidence.json").read_bytes()).hexdigest()
        in sums
    )


def test_saved_schema_reproduces_observation_schema_failure(tmp_path):
    # [FUNCTION DOC]
    # A schema requiring a missing field fails identically (status and evidence)
    # on replay.
    original = replace(
        measured(), observation_schema_json='{"required":["missing-field"]}'
    )
    validation_report = report()
    write_runtime_bundle(tmp_path, validation_report, evidence=original)
    subject = Subject(
        tmp_path, validation_report.manifest, None, None, tmp_path, original
    )
    original_result = ObservationSchemaGrader().run(subject)
    replayed_result = ObservationSchemaGrader().run(
        replace(subject, runtime_evidence=rebuild(tmp_path))
    )
    assert original_result.status.value == "fail"
    assert original_result.status == replayed_result.status
    assert original_result.evidence == replayed_result.evidence


def test_truncated_trace_metadata_preserves_the_collection_failure(tmp_path):
    # [FUNCTION DOC]
    # Failure phase/reason survive the bundle and mark it incomplete.
    original = replace(
        measured(), failure_phase="step", failure_reason="step failed (TimeoutError)"
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["complete"] is False
    replayed = rebuild(tmp_path)
    assert replayed.failure_phase == original.failure_phase
    assert replayed.failure_reason == original.failure_reason


def test_schema_and_trace_redaction_marks_evidence_as_modified(tmp_path):
    # [FUNCTION DOC]
    # Token-like schema values and bearer credentials are redacted and the
    # metadata flags redacted True.
    original = replace(
        measured(),
        observation_schema_json=json.dumps({"description": "hf_notarealtoken12345"}),
        failure_reason="Authorization: Bearer not-a-real-secret",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert metadata["observation_schema"]["description"] == "[REDACTED]"
    assert "not-a-real-secret" not in (tmp_path / "collector-evidence.json").read_text()


def test_deep_subject_json_is_bounded_in_artifacts(tmp_path):
    # [FUNCTION DOC]
    # 600-level nested JSON is cut with a nesting-limit marker, flagged redacted,
    # and the trace stays under 20 KB.
    nested = '"leaf"'
    for _ in range(600):
        nested = '{"child":' + nested + "}"
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", "{}", nested),),
        observation_schema_json=nested,
        failure_phase="step",
        failure_reason="step failed (ValueError)",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert (
        "artifact nesting limit exceeded"
        in (tmp_path / "collector-trace.json").read_text()
    )
    assert (tmp_path / "collector-trace.json").stat().st_size < 20_000


def test_nonfinite_wire_values_remain_valid_artifact_json(tmp_path):
    # [FUNCTION DOC]
    # A NaN reward is written as an invalid_number object so the trace is strict
    # JSON.
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", "{}", '{"reward":NaN}'),),
        observation_schema_json="{}",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)

    def reject_constant(value):
        # [FUNCTION DOC]
        # HELPER: parse_constant hook raising ValueError for NaN/Infinity tokens.
        raise ValueError(value)

    trace = json.loads(
        (tmp_path / "collector-trace.json").read_text(), parse_constant=reject_constant
    )
    assert trace[0]["response_json"]["reward"] == {"invalid_number": "nan"}


def test_missing_schema_is_distinct_from_an_advertised_null_schema(tmp_path):
    # [FUNCTION DOC]
    # A None schema and a "null" schema each round-trip unchanged.
    for schema in (None, "null"):
        directory = tmp_path / ("missing" if schema is None else "null")
        original = replace(measured(), observation_schema_json=schema)
        write_runtime_bundle(directory, report(), evidence=original)
        assert rebuild(directory).observation_schema_json == schema


def test_malformed_wire_omission_is_visible_in_metadata(tmp_path):
    # [FUNCTION DOC]
    # A non-JSON response is omitted from the trace, recorded in
    # omitted_trace_fields, and flagged redacted.
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", "{}", "not JSON"),),
        failure_phase="step",
        failure_reason="step failed (JSONDecodeError)",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert metadata["omitted_trace_fields"] == [
        {"exchange_index": 0, "field": "response_json"}
    ]


def test_redaction_preserves_token_metadata_and_filters_secret_keys():
    # [FUNCTION DOC]
    # _redact keeps benign token-named fields (max_tokens, prompt_token_ids,
    # task_distribution) but redacts access_token values and secret-looking keys
    # and values.
    from openenv.validation.runtime.artifacts import _redact

    secret = "hf_abcdefghijk123456789"
    source = {
        "task_distribution": "task_distribution",
        "max_tokens": 100,
        "prompt_token_ids": [1, 2, 3],
        "access_token": "opaque-credential",
        secret: {"description": "public"},
        "nested": {"key": secret},
    }
    filtered = _redact(source)
    assert filtered["task_distribution"] == "task_distribution"
    assert filtered["max_tokens"] == 100
    assert filtered["prompt_token_ids"] == [1, 2, 3]
    assert filtered["access_token"] == "[REDACTED]"
    assert secret not in json.dumps(filtered)


@pytest.mark.parametrize(
    "reset_schema", [None, "null", '{"required":["missing-reset-field"]}']
)
def test_reset_schema_bundle_preserves_selection_and_grader_result(
    tmp_path, reset_schema
):
    # [FUNCTION DOC]
    # For absent, null and failing reset schemas, the reset schema round-trips and
    # the observation-schema grader result is identical on replay.
    original = replace(measured(), reset_observation_schema_json=reset_schema)
    validation_report = report()
    write_runtime_bundle(tmp_path, validation_report, evidence=original)
    replay = rebuild(tmp_path)
    if reset_schema is None:
        assert replay.reset_observation_schema_json is None
    else:
        assert json.loads(replay.reset_observation_schema_json) == json.loads(
            reset_schema
        )
    subject = Subject(
        tmp_path, validation_report.manifest, None, None, tmp_path, original
    )
    expected = ObservationSchemaGrader().run(subject)
    actual = ObservationSchemaGrader().run(replace(subject, runtime_evidence=replay))
    assert actual.status == expected.status
    assert actual.evidence == expected.evidence


@pytest.mark.parametrize(
    "schema,parse_failed",
    [("not JSON", True), ('{"description":"hf_abcdefghijk123456789"}', False)],
)
def test_reset_schema_omission_or_redaction_is_visible(tmp_path, schema, parse_failed):
    # [FUNCTION DOC]
    # An unparsable or secret-bearing reset schema is marked available, flags
    # parse failure as appropriate, sets redacted and never writes the secret.
    write_runtime_bundle(
        tmp_path,
        report(),
        evidence=replace(measured(), reset_observation_schema_json=schema),
    )
    text = (tmp_path / "collector-evidence.json").read_text()
    metadata = json.loads(text)
    assert metadata["reset_schema_available"] is True
    assert metadata["reset_schema_parse_failed"] is parse_failed
    assert metadata["redacted"] is True
    assert "hf_abcdefghijk123456789" not in text
