# [MODULE DOC]
# Sanity tests for validation enums and the well-known files table.
from openenv.validation.signature import WELL_KNOWN_FILES
from openenv.validation.types import (
    CheckStatus,
    Lane,
    Level,
    Severity,
    SignatureKind,
    Verdict,
)


def test_levels_are_ordered_by_cost():
    # [FUNCTION DOC]
    # Levels order static < runtime < semantic < statistical.
    assert Level.STATIC < Level.RUNTIME < Level.SEMANTIC < Level.STATISTICAL


def test_skip_and_error_are_statuses_advisory_is_a_severity():
    # [FUNCTION DOC]
    # skip/error are statuses not severities; advisory is a severity not a status.
    statuses = {s.value for s in CheckStatus}
    severities = {s.value for s in Severity}
    assert "skip" in statuses and "skip" not in severities
    assert "error" in statuses and "error" not in severities
    assert "advisory" in severities and "advisory" not in statuses


def test_verdicts():
    # [FUNCTION DOC]
    # Verdict values are pass, warn, fail.
    assert {v.value for v in Verdict} == {"pass", "warn", "fail"}


def test_lanes():
    # [FUNCTION DOC]
    # Lane values are local and hub.
    assert {lane.value for lane in Lane} == {"local", "hub"}


def test_well_known_files_lists_implemented_parsers():
    # [FUNCTION DOC]
    # WELL_KNOWN_FILES maps OPENENV_SERVED to openenv.yaml.
    assert WELL_KNOWN_FILES == {SignatureKind.OPENENV_SERVED: "openenv.yaml"}
