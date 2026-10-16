# [MODULE DOC]
# Evidence gates and table renderers for harness qualification: checks whether an eval
# rollout was captured correctly, and validates/renders a recorded qualification report
# (cells per harness x provider) into summary, maturity-tier and detail rows for the UI.
"""Evidence gates for provider evaluation, exact capture, and optimizer qualification.

An installed adapter or reachable endpoint is not an end-to-end qualification.
Live reports are external artifacts tied to model, harness, source and task versions.
"""

from __future__ import annotations

from typing import Any

from .contract import to_trace_entries
from .models import HarborRolloutResult

PROVIDERS = ("openai", "anthropic", "hf", "vllm")
STATUSES = frozenset(
    {
        "not_run",
        "in_progress",
        "failed",
        "blocked",
        "eval_pass",
        "capture_and_reader_pass",
        "optimizer_pass",
    }
)


def evaluate_eval_capture(result: HarborRolloutResult) -> dict[str, bool]:
    # [FUNCTION DOC]
    # Returns boolean gates for an eval rollout: completed ok, verifier reward present,
    # at least one captured turn, rollout_type is "eval", training export refused (converting
    # to trace entries raises ValueError, no entries and no trainable tokens), and no
    # "[FATAL]" finding.
    rejected = False
    try:
        entries = to_trace_entries(result)
    except ValueError:
        entries = []
        rejected = True
    return {
        "rollout_completed": result.ok,
        "verifier_graded": result.reward is not None,
        "model_calls_captured": result.n_turns > 0,
        "eval_label": result.rollout_type == "eval",
        "no_training_export": rejected
        and not entries
        and result.n_trainable_tokens == 0,
        "no_capture_fatal": not any("[FATAL]" in f for f in result.findings),
    }


def qualification_rows(
    harnesses: list[str], report: dict[str, Any] | None = None
) -> list[list[str]]:
    # [FUNCTION DOC]
    # Validates report["cells"]: each (harness, provider) must be unique, provider in
    # PROVIDERS, status in STATUSES (default not_run), any *pass status needs evidence, and
    # optimizer_pass needs optimizer_validated true plus optimizer_evidence matching current
    # captures with result, inputs, scope, model, revision and a positive int rows. Raises
    # ValueError otherwise. Returns one row per harness (sorted): name followed by the status
    # for each provider in PROVIDERS order (not_run when missing).
    """Render recorded evidence without promoting legacy adapter status to provider passes."""
    records = {}
    for cell in (report or {}).get("cells", []):
        key = (cell.get("harness"), cell.get("provider"))
        if key in records:
            raise ValueError("duplicate harness/provider evidence")
        if key[1] not in PROVIDERS:
            raise ValueError(f"unknown qualification provider: {key[1]}")
        status = cell.get("status", "not_run")
        if status not in STATUSES:
            raise ValueError(f"unknown qualification status: {status}")
        if status.endswith("pass") and not cell.get("evidence"):
            raise ValueError("qualification pass requires evidence")
        if status == "optimizer_pass":
            proof = cell.get("optimizer_evidence") or {}
            if (
                cell.get("optimizer_validated") is not True
                or proof.get("matches_current_captures") is not True
                or not proof.get("result")
                or not proof.get("inputs")
                or not proof.get("scope")
                or not proof.get("model")
                or not proof.get("revision")
                or not isinstance(proof.get("rows"), int)
                or proof["rows"] <= 0
            ):
                raise ValueError(
                    "optimizer pass requires matching, scoped optimizer evidence"
                )
        records[key] = status
    return [
        [name, *(records.get((name, provider), "not_run") for provider in PROVIDERS)]
        for name in sorted(harnesses)
    ]


def harness_maturity_rows(
    harnesses: list[str], report: dict[str, Any] | None = None
) -> list[list[str]]:
    # [FUNCTION DOC]
    # Classifies each harness from qualification_rows: "stable" only if statuses are exactly
    # eval_pass for openai/anthropic/hf and optimizer_pass for vllm; "unstable" if all are
    # failed/blocked; else "experimental". Returns [name, tier, reason] rows.
    """Classify recorded profiles, not universal or production-scale reliability.

    Partial support remains experimental. Four terminal failures make an adapter
    unstable for this matrix, without claiming its vendor can never support it.
    """
    rows = []
    for name, *statuses in qualification_rows(harnesses, report):
        if statuses == ["eval_pass", "eval_pass", "eval_pass", "optimizer_pass"]:
            tier = "stable"
            reason = "All four tested profiles pass, including current-capture optimizer replay."
        elif all(status in {"failed", "blocked"} for status in statuses):
            tier = "unstable"
            reason = "No tested provider profile passes; excluded from the stable set."
        else:
            tier = "experimental"
            reason = "Partial support or validation pending; explicit opt-in only."
        rows.append([name, tier, reason])
    return rows


def qualification_details(report: dict[str, Any] | None = None) -> list[list[str]]:
    # [FUNCTION DOC]
    # Validates the report (same rules as qualification_rows), then for each cell that is not
    # not_run returns a row: harness, provider, status, model (configuration or, when current,
    # optimizer evidence), pinned/observed harness versions, tasks completed, ACP/NeMo profile,
    # optimizer scope prefixed by whether it matches current captures (or "not validated"),
    # revision, evidence paths and reason. Sorted by harness then provider.
    # Raises KeyError if a cell lacks "provider" (validation would have raised ValueError first).
    """Expose the scope and provenance behind the summary statuses.

    File references identify externally verified artifacts; this renderer does not
    re-run jobs or assert that a saved report matches the currently selected endpoint.
    """
    qualification_rows([], report)  # Apply identical validation to both UI views.
    rows = []
    for cell in (report or {}).get("cells", []):
        if cell.get("status", "not_run") == "not_run":
            continue
        config = cell.get("configuration") or {}
        proof = cell.get("optimizer_evidence") or {}
        harness = cell.get("harness", "")
        matching = proof.get("matches_current_captures") is True
        model = config.get("model") or (proof.get("model") if matching else "")
        pinned = (config.get("version_pins") or {}).get(harness)
        observed = sorted(
            {
                str(item["observed_version"])
                for item in cell.get("capture_provenance", [])
                if item.get("observed_version")
            }
        )
        version = "pinned: " + str(pinned) if pinned else "pin not recorded"
        if observed:
            version += "; observed: " + ", ".join(observed)
        rows.append(
            [
                harness,
                cell["provider"],
                cell.get("status", "not_run"),
                model or "not recorded",
                version,
                str(cell.get("tasks_completed", "not recorded")),
                str(config.get("acp_profile") or config.get("nemo_profile") or ""),
                ("current captures: " if matching else "previous captures only: ")
                + str(proof.get("scope", ""))
                if proof
                else "not validated",
                str(proof.get("revision", "")),
                "; ".join(str(path) for path in cell.get("evidence", [])),
                str(cell.get("reason") or ""),
            ]
        )
    return sorted(rows, key=lambda row: (row[0], row[1]))
