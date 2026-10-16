# [MODULE DOC]
# Checks that the committed JSON schemas under src/openenv/validation/schemas
# match the pydantic models' generated schemas (EXPORTS maps file name to
# model).
import json
from pathlib import Path

import pytest
from openenv.validation.manifest import NormalizedManifest, NormalizedManifestV2
from openenv.validation.report import ValidationReport, ValidationReportV2
from openenv.validation.runtime.contracts import RuntimePlan

SCHEMAS_DIR = (
    Path(__file__).parent.parent.parent / "src" / "openenv" / "validation" / "schemas"
)

EXPORTS = {
    "manifest.schema.json": NormalizedManifest,
    "report.schema.json": ValidationReport,
    "manifest-v2.schema.json": NormalizedManifestV2,
    "report-v2.schema.json": ValidationReportV2,
    "runtime-plan.schema.json": RuntimePlan,
}


@pytest.mark.parametrize("fname", sorted(EXPORTS))
def test_committed_schema_matches_model(fname):
    # [FUNCTION DOC]
    # The committed file equals the model schema rendered with indent 2, sorted
    # keys and a trailing newline; the failure message names the sync script.
    committed = (SCHEMAS_DIR / fname).read_text()
    rendered = (
        json.dumps(EXPORTS[fname].model_json_schema(), indent=2, sort_keys=True) + "\n"
    )
    assert committed == rendered, (
        f"{fname} is stale; run: PYTHONPATH=src python scripts/sync_validation_schemas.py --fix"
    )


def test_report_schema_allows_hub_and_statistical_check_ids():
    # [FUNCTION DOC]
    # The CheckResult check_id pattern accepts hub. and statistical. ids.
    schema = ValidationReport.model_json_schema()
    pattern = schema["$defs"]["CheckResult"]["properties"]["check_id"]["pattern"]
    import re

    for reserved in ("hub.cross_host_determinism", "statistical.adversarial_floor"):
        assert re.match(pattern, reserved), reserved
