# [MODULE DOC]
# Unit tests for NormalizedManifest and VerifierBinding pydantic validation,
# using the normalized fixture manifests and targeted mutations.
import pytest
from conftest import (
    INVALID_MANIFEST_FIXTURES,
    load_fixture_manifest,
    VALID_MANIFEST_FIXTURES,
)
from openenv.validation.manifest import NormalizedManifest, VerifierBinding
from pydantic import ValidationError


@pytest.mark.parametrize("name", VALID_MANIFEST_FIXTURES)
def test_valid_fixture_manifests_round_trip(name):
    # [FUNCTION DOC]
    # Each valid fixture validates and survives a JSON dump/validate round trip
    # unchanged.
    data = load_fixture_manifest(name)
    manifest = NormalizedManifest.model_validate(data)
    assert (
        NormalizedManifest.model_validate(manifest.model_dump(mode="json")) == manifest
    )


@pytest.mark.parametrize("name", INVALID_MANIFEST_FIXTURES)
def test_invalid_fixture_manifests_are_rejected(name):
    # [FUNCTION DOC]
    # Each invalid fixture raises ValidationError.
    data = load_fixture_manifest(name)
    with pytest.raises(ValidationError):
        NormalizedManifest.model_validate(data)


def test_no_oracle_is_a_valid_manifest_not_a_parse_error():
    # [FUNCTION DOC]
    # no_oracle validates with capabilities.oracle None.
    manifest = NormalizedManifest.model_validate(load_fixture_manifest("no_oracle"))
    assert manifest.capabilities.oracle is None


def test_unpinned_judge_fails_on_the_judge_pin():
    # [FUNCTION DOC]
    # unpinned_judge fails with "judge pin is required".
    data = load_fixture_manifest("unpinned_judge")
    with pytest.raises(ValidationError, match="judge pin is required"):
        NormalizedManifest.model_validate(data)


def test_injected_state_oracle_requires_set_state():
    # [FUNCTION DOC]
    # Disabling set_state on served_min_pass fails with a set_state error.
    data = load_fixture_manifest("served_min_pass")
    data["capabilities"]["set_state"] = False
    with pytest.raises(ValidationError, match="set_state"):
        NormalizedManifest.model_validate(data)


def test_script_oracle_does_not_require_set_state():
    # [FUNCTION DOC]
    # harbor_task_min validates with set_state False and a script oracle.
    data = load_fixture_manifest("harbor_task_min")
    manifest = NormalizedManifest.model_validate(data)
    assert manifest.capabilities.set_state is False
    assert manifest.capabilities.oracle.form == "script"


def test_judge_pin_without_llm_judged_is_rejected():
    # [FUNCTION DOC]
    # Adding a judge without llm_judged fails with "llm_judged is false".
    data = load_fixture_manifest("served_min_pass")
    data["judge"] = {"model": "judge-model", "version": "2026-01-01", "params": {}}
    with pytest.raises(ValidationError, match="llm_judged is false"):
        NormalizedManifest.model_validate(data)


def test_llm_judged_requires_variance_tolerance():
    # [FUNCTION DOC]
    # llm_judged with a judge but no variance_tolerance fails.
    data = load_fixture_manifest("served_min_pass")
    data["capabilities"]["llm_judged"] = True
    data["judge"] = {"model": "judge-model", "version": "2026-01-01", "params": {}}
    with pytest.raises(ValidationError, match="variance_tolerance"):
        NormalizedManifest.model_validate(data)


def test_fully_pinned_judged_manifest_validates():
    # [FUNCTION DOC]
    # llm_judged with a judge pin and variance_tolerance validates.
    data = load_fixture_manifest("served_min_pass")
    data["capabilities"]["llm_judged"] = True
    data["judge"] = {
        "model": "judge-model",
        "version": "2026-01-01",
        "params": {"temperature": 0},
    }
    data["reward"]["variance_tolerance"] = 0.1
    manifest = NormalizedManifest.model_validate(data)
    assert manifest.judge.model == "judge-model"


def test_reward_range_must_increase():
    # [FUNCTION DOC]
    # A reversed reward range fails with "strictly increasing".
    data = load_fixture_manifest("served_min_pass")
    data["reward"]["range"] = [1.0, 0.0]
    with pytest.raises(ValidationError, match="strictly increasing"):
        NormalizedManifest.model_validate(data)


def test_script_verifier_requires_entry():
    # [FUNCTION DOC]
    # A script VerifierBinding without entry fails with "entry is required".
    with pytest.raises(ValidationError, match="entry is required"):
        VerifierBinding(kind="script")


def test_reward_channel_verifier_forbids_entry():
    # [FUNCTION DOC]
    # A reward_channel VerifierBinding with an entry fails with "only valid".
    with pytest.raises(ValidationError, match="only valid"):
        VerifierBinding(kind="reward_channel", entry="tests/test.sh")


def test_types_require_at_least_one_tag():
    # [FUNCTION DOC]
    # Empty types.tags fails with "at least 1".
    data = load_fixture_manifest("served_min_pass")
    data["types"]["tags"] = []
    with pytest.raises(ValidationError, match="at least 1"):
        NormalizedManifest.model_validate(data)


def test_harbor_manifest_signature_is_task_toml():
    # [FUNCTION DOC]
    # harbor_task_min's signature value is "task.toml".
    data = load_fixture_manifest("harbor_task_min")
    manifest = NormalizedManifest.model_validate(data)
    assert manifest.signature.value == "task.toml"
