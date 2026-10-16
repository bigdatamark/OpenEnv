# [MODULE DOC]
# Tests for severity policy loading (v1) and apply_policy verdict computation
# across lanes and check statuses.
import pytest
from conftest import EXPECTED_POLICY
from openenv.validation.policy import (
    apply_policy,
    DeclarationBounds,
    load_policy,
    PolicyEntry,
    PolicyError,
    SeverityPolicy,
)
from openenv.validation.report import CheckResult
from openenv.validation.types import CheckStatus, Lane, Level, Severity, Verdict


def result(check_id: str, status: CheckStatus) -> CheckResult:
    # [FUNCTION DOC]
    # HELPER: Builds a CheckResult with the given id and status and zero duration.
    return CheckResult(check_id=check_id, status=status, duration_s=0.0)


@pytest.fixture(scope="module")
def policy():
    # [FUNCTION DOC]
    # Module-scoped fixture returning the loaded "v1" SeverityPolicy.
    return load_policy("v1")


def test_policy_contains_every_check_id_exactly_once(policy):
    # [FUNCTION DOC]
    # Policy ids are unique and equal the EXPECTED_POLICY keys.
    ids = [e.check_id for e in policy.entries]
    assert len(ids) == len(set(ids)), "duplicate check ids in policy"
    assert set(ids) == set(EXPECTED_POLICY)


def test_policy_levels_lanes_and_severities_match_the_rfc_table(policy):
    # [FUNCTION DOC]
    # Each entry's level, lane and severity match EXPECTED_POLICY.
    for entry in policy.entries:
        level, lane, severity = EXPECTED_POLICY[entry.check_id]
        assert entry.level == level, entry.check_id
        assert entry.lane.value == lane, entry.check_id
        assert entry.severity.value == severity, entry.check_id


def test_duplicate_check_ids_are_rejected():
    # [FUNCTION DOC]
    # Constructing a SeverityPolicy with a repeated entry raises ValueError
    # mentioning duplicate check ids.
    entry = PolicyEntry(
        check_id="static.manifest",
        level=Level.STATIC,
        lane=Lane.LOCAL,
        severity=Severity.FAIL,
    )
    with pytest.raises(ValueError, match="duplicate check ids"):
        SeverityPolicy(
            policy_version="v1",
            entries=[entry, entry],
            bounds=DeclarationBounds(
                max_oracle_tolerance=0.1,
                min_floor_margin=0.1,
                max_variance_tolerance=0.2,
                max_episode_timeout_s=3600.0,
            ),
        )


def test_local_lane_never_contains_hub_or_statistical_ids(policy):
    # [FUNCTION DOC]
    # The local lane is non-empty and contains no hub./statistical. ids.
    local = policy.entries_for_lane(Lane.LOCAL)
    assert local, "local lane is empty"
    for check_id in local:
        assert not check_id.startswith(("hub.", "statistical.")), check_id


def test_hub_lane_is_a_superset_of_local(policy):
    # [FUNCTION DOC]
    # Local lane ids are a strict subset of hub lane ids.
    local = set(policy.entries_for_lane(Lane.LOCAL))
    hub = set(policy.entries_for_lane(Lane.HUB))
    assert local < hub


def test_declaration_bounds_are_positive(policy):
    # [FUNCTION DOC]
    # All policy declaration bounds are positive.
    b = policy.bounds
    assert b.max_oracle_tolerance > 0
    assert b.min_floor_margin > 0
    assert b.max_variance_tolerance > 0
    assert b.max_episode_timeout_s > 0


def test_unknown_policy_version_raises():
    # [FUNCTION DOC]
    # Loading "v999" raises PolicyError naming it.
    with pytest.raises(PolicyError, match="v999"):
        load_policy("v999")


def test_all_pass_is_pass(policy):
    # [FUNCTION DOC]
    # A single passing result yields PASS.
    results = [result("static.manifest", CheckStatus.PASS)]
    assert apply_policy(results, policy, Lane.LOCAL) is Verdict.PASS


def test_fail_severity_failure_is_fail(policy):
    # [FUNCTION DOC]
    # A failing fail-severity check yields FAIL.
    results = [result("semantic.oracle_max", CheckStatus.FAIL)]
    assert apply_policy(results, policy, Lane.LOCAL) is Verdict.FAIL


def test_warn_severity_failure_is_warn(policy):
    # [FUNCTION DOC]
    # A failing warn-severity check yields WARN.
    results = [
        result("static.manifest", CheckStatus.PASS),
        result("runtime.trajectory_record", CheckStatus.FAIL),
    ]
    assert apply_policy(results, policy, Lane.LOCAL) is Verdict.WARN


def test_error_fails_closed(policy):
    # [FUNCTION DOC]
    # An ERROR status yields FAIL even for a warn-severity check.
    results = [result("runtime.trajectory_record", CheckStatus.ERROR)]
    assert apply_policy(results, policy, Lane.LOCAL) is Verdict.FAIL


def test_skip_marks_the_run_incomplete(policy):
    # [FUNCTION DOC]
    # A SKIP status yields WARN.
    results = [result("semantic.oracle_max", CheckStatus.SKIP)]
    assert apply_policy(results, policy, Lane.LOCAL) is Verdict.WARN


def test_unknown_check_id_is_an_internal_error(policy):
    # [FUNCTION DOC]
    # An unknown check id raises PolicyError.
    results = [result("static.no_such_check", CheckStatus.PASS)]
    with pytest.raises(PolicyError, match="no_such_check"):
        apply_policy(results, policy, Lane.LOCAL)


def test_hub_id_in_a_local_run_is_an_internal_error(policy):
    # [FUNCTION DOC]
    # A hub check id in a local-lane run raises PolicyError.
    results = [result("hub.cross_host_determinism", CheckStatus.PASS)]
    with pytest.raises(PolicyError, match="cross_host"):
        apply_policy(results, policy, Lane.LOCAL)


@pytest.mark.parametrize(
    "invalid_check_id", ["static.no_such_check", "hub.cross_host_determinism"]
)
@pytest.mark.parametrize("status", [CheckStatus.ERROR, CheckStatus.FAIL])
def test_invalid_check_id_is_not_masked_by_an_earlier_terminal_result(
    policy, invalid_check_id, status
):
    # [FUNCTION DOC]
    # An invalid id still raises PolicyError when an earlier result already
    # failed or errored.
    results = [
        result("semantic.oracle_max", status),
        result(invalid_check_id, CheckStatus.PASS),
    ]
    with pytest.raises(PolicyError, match=invalid_check_id.split(".")[-1]):
        apply_policy(results, policy, Lane.LOCAL)


def test_hub_lane_grades_both_local_and_hub_ids(policy):
    # [FUNCTION DOC]
    # In the hub lane, local and hub ids are graded; a hub failure yields FAIL.
    results = [
        result("static.manifest", CheckStatus.PASS),
        result("hub.cross_host_determinism", CheckStatus.FAIL),
    ]
    assert apply_policy(results, policy, Lane.HUB) is Verdict.FAIL
