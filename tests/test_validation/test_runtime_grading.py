# [MODULE DOC]
# Unit tests for the basic runtime graders (RewardWellFormedGrader,
# ObservationSchemaGrader, StateContractGrader) run against synthetic wire
# evidence: strict reward typing and range, reset vs step observation schemas,
# schema safety (no external refs, killable worker with a time budget, no
# echoing of private values), envelope validation, state identity/step counts,
# and SKIP behavior for missing or truncated evidence.
import copy
import json
import subprocess
import time
from dataclasses import replace

import pytest
from conftest import load_fixture_manifest
from openenv.core.env_server.types import Observation
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime import (
    basic,
    ObservationSchemaGrader,
    RewardWellFormedGrader,
    StateContractGrader,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.runtime.contracts import RuntimeEvidence, WireExchange
from openenv.validation.types import CheckStatus
from support.runtime import evidence, exchange

GRADERS = [RewardWellFormedGrader, ObservationSchemaGrader, StateContractGrader]
OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "counter": {"type": "integer"},
        "reward": {"type": ["number", "null"]},
        "done": {"type": "boolean"},
    },
    "required": ["counter", "reward", "done"],
}


def good_rows():
    # [FUNCTION DOC]
    # HELPER: Returns a fresh list of four well-formed exchanges: reset (counter 0,
    # null reward), state (step_count 0), step (counter 1, reward 1.0), state
    # (step_count 1), all for episode "measured-episode".
    return [
        exchange(
            "reset",
            {"type": "reset", "data": {"episode_id": "measured-episode", "seed": 42}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 0}, "reward": None, "done": False},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {
                "type": "state",
                "data": {"episode_id": "measured-episode", "step_count": 0},
            },
        ),
        exchange(
            "step",
            {"type": "step", "data": {"increment": 1}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 1}, "reward": 1.0, "done": False},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {
                "type": "state",
                "data": {"episode_id": "measured-episode", "step_count": 1},
            },
        ),
    ]


def subject_with(tmp_path, rows=None, schema=OBSERVATION_SCHEMA):
    # [FUNCTION DOC]
    # HELPER: Builds a Subject using the served_min_pass manifest and runtime
    # evidence from the given rows (default good_rows) and observation schema
    # (default OBSERVATION_SCHEMA; None means unavailable).
    return Subject(
        root=tmp_path,
        manifest=NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        ),
        image_ref=None,
        running=None,
        outputs_dir=tmp_path,
        runtime_evidence=evidence(
            *(good_rows() if rows is None else rows), observation_schema=schema
        ),
    )


def mutate_response(rows, index, mutate):
    # [FUNCTION DOC]
    # HELPER: Parses rows[index]'s response JSON, applies the mutate callback to
    # it, and replaces that row with the re-encoded response.
    # Returns: the same rows list (mutated in place).
    payload = json.loads(rows[index].response_json)
    mutate(payload)
    rows[index] = replace(rows[index], response_json=json.dumps(payload))
    return rows


@pytest.mark.parametrize("grader", GRADERS)
def test_good_measured_session_passes_each_basic_contract(tmp_path, grader):
    # [FUNCTION DOC]
    # Each grader passes on the good session.
    assert grader().run(subject_with(tmp_path)).status is CheckStatus.PASS


@pytest.mark.parametrize(
    "reward",
    [
        True,
        False,
        "0.5",
        [],
        {},
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
        -0.1,
        1.1,
    ],
)
@pytest.mark.parametrize("done", [False, True])
def test_step_rewards_are_checked_without_coercion(tmp_path, reward, done):
    # [FUNCTION DOC]
    # Booleans, strings, containers, non-finite, huge and out-of-range step
    # rewards fail with reward evidence, whether done or not.
    rows = mutate_response(
        good_rows(),
        2,
        lambda response: response["data"].update(reward=reward, done=done),
    )
    result = RewardWellFormedGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert any("reward" in message for message in result.evidence)


@pytest.mark.parametrize("reward", [None, True, False])
@pytest.mark.parametrize("done", [False, True])
def test_core_reward_types_do_not_imply_level_two_certification(tmp_path, reward, done):
    # [FUNCTION DOC]
    # Rewards the core Observation model accepts (None, booleans) still fail the
    # grader at exchange 2, and the wire value is unchanged.
    # Core compatibility does not weaken the stricter certification contract.
    assert Observation(reward=reward, done=done).reward is reward
    rows = mutate_response(
        good_rows(),
        2,
        lambda response: response["data"].update(reward=reward, done=done),
    )
    result = RewardWellFormedGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert any("exchange 2: reward" in message for message in result.evidence)
    assert json.loads(rows[2].response_json)["data"]["reward"] is reward


def test_null_reset_reward_remains_unscored(tmp_path):
    # [FUNCTION DOC]
    # A null reward on reset is allowed and not modified.
    rows = good_rows()
    assert json.loads(rows[0].response_json)["data"]["reward"] is None
    assert (
        RewardWellFormedGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.PASS
    )
    assert json.loads(rows[0].response_json)["data"]["reward"] is None


@pytest.mark.parametrize(
    "invalid_index,invalid_observation",
    [(0, {"ready": "yes"}), (2, {"counter": "not-an-integer"})],
)
def test_explicit_reset_schema_is_independent_of_step_schema(
    tmp_path, invalid_index, invalid_observation
):
    # [FUNCTION DOC]
    # A reset observation not matching the step schema fails without a reset
    # schema, passes with a matching reset schema, and a later invalid reset or
    # step observation fails with evidence only for that exchange.
    rows = mutate_response(
        good_rows(),
        0,
        lambda response: response["data"].update(observation={"ready": True}),
    )
    subject = subject_with(tmp_path, rows)
    # A legacy single-schema server remains strict about reset fields.
    assert ObservationSchemaGrader().run(subject).status is CheckStatus.FAIL
    subject = replace(
        subject,
        runtime_evidence=replace(
            subject.runtime_evidence,
            reset_observation_schema_json=json.dumps(
                {
                    "type": "object",
                    "properties": {"ready": {"type": "boolean"}},
                    "required": ["ready"],
                }
            ),
        ),
    )
    assert ObservationSchemaGrader().run(subject).status is CheckStatus.PASS
    malformed = mutate_response(
        rows,
        invalid_index,
        lambda response: response["data"].update(observation=invalid_observation),
    )
    subject = replace(
        subject,
        runtime_evidence=replace(subject.runtime_evidence, exchanges=tuple(malformed)),
    )
    result = ObservationSchemaGrader().run(subject)
    assert result.status is CheckStatus.FAIL
    assert all(f"exchange {invalid_index}:" in message for message in result.evidence)


@pytest.mark.parametrize(
    "schema",
    [
        None,
        [],
        {"type": "invalid"},
        False,
        {"$ref": "https://example.invalid/reset.json"},
    ],
)
def test_explicit_invalid_or_rejecting_reset_schema_cannot_fall_back(tmp_path, schema):
    # [FUNCTION DOC]
    # An advertised reset schema that is null, invalid, rejecting or externally
    # referenced fails rather than falling back to the step schema.
    subject = subject_with(tmp_path)
    subject = replace(
        subject,
        runtime_evidence=replace(
            subject.runtime_evidence, reset_observation_schema_json=json.dumps(schema)
        ),
    )
    assert ObservationSchemaGrader().run(subject).status is CheckStatus.FAIL


def test_missing_step_reward_is_an_explicit_failure(tmp_path):
    # [FUNCTION DOC]
    # A step without a reward key fails with "exchange 2: missing reward".
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].pop("reward")
    )
    result = RewardWellFormedGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2: missing reward" in result.evidence


@pytest.mark.parametrize("reward", [0, 1, 0.5])
def test_numeric_reward_endpoints_and_fraction_are_valid(tmp_path, reward):
    # [FUNCTION DOC]
    # Rewards 0, 1 and 0.5 within the manifest range pass.
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].update(reward=reward)
    )
    assert (
        RewardWellFormedGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.PASS
    )


@pytest.mark.parametrize("grader", GRADERS)
def test_reset_only_is_incomplete_not_a_passing_step_contract(tmp_path, grader):
    # [FUNCTION DOC]
    # With only reset and state (reset done), every grader skips citing no step.
    rows = mutate_response(
        good_rows()[:2], 0, lambda response: response["data"].update(done=True)
    )
    result = grader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.SKIP
    assert any("no step" in message for message in result.evidence)


@pytest.mark.parametrize("grader", GRADERS)
def test_missing_evidence_is_a_named_skip(tmp_path, grader):
    # [FUNCTION DOC]
    # Missing runtime evidence skips with "runtime evidence is unavailable".
    subject = replace(subject_with(tmp_path), runtime_evidence=None)
    result = grader().run(subject)
    assert result.status is CheckStatus.SKIP
    assert result.evidence == ["runtime evidence is unavailable"]


@pytest.mark.parametrize("grader", GRADERS)
def test_truncated_collection_cannot_pass_from_a_valid_prefix(tmp_path, grader):
    # [FUNCTION DOC]
    # Evidence carrying a collection failure makes every grader skip.
    subject = subject_with(tmp_path)
    subject = replace(
        subject,
        runtime_evidence=replace(
            subject.runtime_evidence,
            failure_phase="step",
            failure_reason="step failed (TimeoutError)",
        ),
    )
    assert grader().run(subject).status is CheckStatus.SKIP


@pytest.mark.parametrize(
    "schema", [{"type": "not-a-json-type"}, {"required": "counter"}, None]
)
def test_invalid_or_missing_advertised_schema_fails(tmp_path, schema):
    # [FUNCTION DOC]
    # An invalid or missing observation schema fails with schema evidence.
    result = ObservationSchemaGrader().run(subject_with(tmp_path, schema=schema))
    assert result.status is CheckStatus.FAIL
    assert any("schema" in message for message in result.evidence)


@pytest.mark.parametrize(
    "reference", ["https://example.invalid/schema", "file:///etc/passwd", "other.json"]
)
@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef"])
def test_nonlocal_schema_references_are_rejected_before_starting_worker(
    tmp_path, monkeypatch, keyword, reference
):
    # [FUNCTION DOC]
    # External $ref / $dynamicRef values fail with a fixed message and never start
    # the schema worker subprocess.
    def no_worker(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fails the test if the worker subprocess is started.
        pytest.fail("untrusted external references must not reach schema evaluation")

    monkeypatch.setattr(basic.subprocess, "run", no_worker)
    result = ObservationSchemaGrader().run(
        subject_with(tmp_path, schema={keyword: reference})
    )
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["observation schema has a non-local reference"]


def test_internal_schema_references_are_supported(tmp_path):
    # [FUNCTION DOC]
    # Local "#/$defs/..." references validate and pass.
    schema = copy.deepcopy(OBSERVATION_SCHEMA)
    schema["$defs"] = {"counter": {"type": "integer"}}
    schema["properties"]["counter"] = {"$ref": "#/$defs/counter"}
    assert (
        ObservationSchemaGrader().run(subject_with(tmp_path, schema=schema)).status
        is CheckStatus.PASS
    )


def test_schema_reconstructs_reward_and_done_from_envelope(tmp_path):
    # [FUNCTION DOC]
    # Reward and done required by the schema are taken from the envelope, not the
    # nested observation.
    # These are deliberately absent from the nested observation, as on the wire.
    rows = good_rows()
    assert "reward" not in json.loads(rows[2].response_json)["data"]["observation"]
    assert (
        ObservationSchemaGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.PASS
    )


@pytest.mark.parametrize(
    "field,value", [("done", None), ("done", "false"), ("done", 0), ("observation", [])]
)
def test_invalid_raw_envelope_is_rejected_before_defaults(tmp_path, field, value):
    # [FUNCTION DOC]
    # Non-boolean done or non-object observation fails as a malformed envelope.
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].update({field: value})
    )
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2: malformed observation envelope" in result.evidence


@pytest.mark.parametrize("data", [None, [], "private-payload", 42, 0.5, False])
@pytest.mark.parametrize(
    "grader,index,envelope",
    [
        (RewardWellFormedGrader, 0, "observation"),
        (RewardWellFormedGrader, 2, "observation"),
        (ObservationSchemaGrader, 0, "observation"),
        (ObservationSchemaGrader, 2, "observation"),
        (StateContractGrader, 1, "state"),
        (StateContractGrader, 3, "state"),
    ],
)
def test_non_object_data_is_a_finding_not_a_validator_crash(
    tmp_path, grader, index, envelope, data
):
    # [FUNCTION DOC]
    # Non-object "data" in reset/step/state envelopes yields a malformed envelope
    # failure for that exchange without echoing the payload.
    rows = mutate_response(
        good_rows(), index, lambda response: response.update(data=data)
    )
    result = grader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert f"exchange {index}: malformed {envelope} envelope" in result.evidence
    assert "private-payload" not in result.model_dump_json()


def test_missing_done_is_not_filled_by_a_model_default(tmp_path):
    # [FUNCTION DOC]
    # A missing done field fails as a malformed envelope.
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].pop("done")
    )
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2: malformed observation envelope" in result.evidence


def test_schema_mismatch_evidence_does_not_echo_private_subject_values(tmp_path):
    # [FUNCTION DOC]
    # Schema mismatch evidence names the field but not the offending value.
    private_value = "not-a-number-secret-observation"
    rows = mutate_response(
        good_rows(),
        2,
        lambda response: response["data"]["observation"].update(counter=private_value),
    )
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert any("counter" in message for message in result.evidence)
    assert private_value not in result.model_dump_json()


def test_schema_worker_deadline_becomes_a_validation_finding(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # A worker TimeoutExpired (asserting the worker script, utf-8 encoding and a
    # timeout in (0, 5]) becomes a FAIL with a time-budget message.
    def timeout(command, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Validates subprocess.run arguments and raises TimeoutExpired.
        assert command[-1].endswith("schema_worker.py")
        assert kwargs["encoding"] == "utf-8"
        assert 0 < kwargs["timeout"] <= 5
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(basic.subprocess, "run", timeout)
    result = ObservationSchemaGrader().run(subject_with(tmp_path))
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["observation schema evaluation exceeded its time budget"]


def test_pathological_regex_runs_in_killable_worker(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # A catastrophic-backtracking pattern is killed by a 0.5 s worker timeout,
    # returning a time-budget FAIL in under 3 s. Runs a real subprocess.
    run = subprocess.run

    def short_deadline(command, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Forces timeout to 0.5 s and calls the real subprocess.run.
        kwargs["timeout"] = 0.5
        return run(command, **kwargs)

    monkeypatch.setattr(basic.subprocess, "run", short_deadline)
    schema = copy.deepcopy(OBSERVATION_SCHEMA)
    schema["properties"]["counter"] = {"type": "string", "pattern": "^(a+)+$"}
    rows = good_rows()
    for index in (0, 2):
        mutate_response(
            rows,
            index,
            lambda response: response["data"]["observation"].update(
                counter="a" * 100 + "!"
            ),
        )
    started = time.monotonic()
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows, schema=schema))
    assert time.monotonic() - started < 3
    assert result.status is CheckStatus.FAIL
    assert any("time budget" in message for message in result.evidence)


@pytest.mark.parametrize("episode_id", ["other-episode", None, 42])
def test_state_identity_must_match_the_requested_episode(tmp_path, episode_id):
    # [FUNCTION DOC]
    # A state episode_id differing from reset fails at exchange 3.
    rows = mutate_response(
        good_rows(), 3, lambda response: response["data"].update(episode_id=episode_id)
    )
    result = StateContractGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 3: episode_id differs from reset" in result.evidence


@pytest.mark.parametrize("count", [True, 1.0, "1", None, -1, 0, 2])
def test_state_count_is_strict_and_matches_successful_steps(tmp_path, count):
    # [FUNCTION DOC]
    # Non-integer or wrong step_count values fail at exchange 3.
    rows = mutate_response(
        good_rows(), 3, lambda response: response["data"].update(step_count=count)
    )
    result = StateContractGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 3: incorrect step_count" in result.evidence


def test_reset_state_count_must_start_at_zero(tmp_path):
    # [FUNCTION DOC]
    # A nonzero step_count after reset fails.
    rows = mutate_response(
        good_rows(), 1, lambda response: response["data"].update(step_count=5)
    )
    assert (
        StateContractGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.FAIL
    )


def test_missing_state_snapshots_do_not_pass(tmp_path):
    # [FUNCTION DOC]
    # No state exchanges fails with "no state snapshots were measured".
    rows = [row for row in good_rows() if row.operation != "state"]
    result = StateContractGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "no state snapshots were measured" in result.evidence


@pytest.mark.parametrize("grader", GRADERS)
def test_malformed_wire_is_a_finding_not_a_validator_crash(tmp_path, grader):
    # [FUNCTION DOC]
    # Non-JSON step response with a recorded collection failure makes every grader
    # skip rather than crash.
    rows = good_rows()
    rows[2] = WireExchange("step", rows[2].request_json, "{not-json")
    # Collector failure is part of the evidence even for graders that do not read
    # step response bodies (the state check only needs the step request count).
    subject = subject_with(tmp_path, rows)
    subject = replace(
        subject,
        runtime_evidence=RuntimeEvidence(
            exchanges=subject.runtime_evidence.exchanges,
            observation_schema_json=subject.runtime_evidence.observation_schema_json,
            failure_phase="step",
            failure_reason="step failed (JSONDecodeError)",
        ),
    )
    assert grader().run(subject).status is CheckStatus.SKIP


def test_schema_required_keyword_identifies_missing_property(tmp_path):
    # [FUNCTION DOC]
    # Missing required properties are reported with "(required)" and the property
    # name list.
    result = ObservationSchemaGrader().run(
        subject_with(tmp_path, schema={"required": ["tool_name"]})
    )
    assert result.status is CheckStatus.FAIL
    assert all("(required)" in message for message in result.evidence)
    assert all(
        'missing properties: ["tool_name"]' in message for message in result.evidence
    )


def test_schema_diagnostics_do_not_echo_private_property_names(tmp_path):
    # [FUNCTION DOC]
    # Secret-looking required property names are not echoed.
    secret = "hf_abcdefghijk123456789"
    result = ObservationSchemaGrader().run(
        subject_with(tmp_path, schema={"required": [secret]})
    )
    assert result.status is CheckStatus.FAIL
    assert secret not in result.model_dump_json()
