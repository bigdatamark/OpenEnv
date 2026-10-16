# [MODULE DOC]
# Runtime-level graders that inspect previously recorded wire exchanges (reset/step/
# state request and response JSON) in subject.runtime_evidence, without talking to the
# environment: reward well-formedness, observation schema conformance (evaluated in an
# isolated subprocess) and state contract (episode id / step count).
"""Pure graders over independently collected, uncoerced wire evidence."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

from ...report import CheckResult
from ...runtime.artifacts import _redact
from ...types import CheckStatus, Level


def _remote_reference(value) -> bool:
    # [FUNCTION DOC]
    # HELPER: recursively scans a JSON-schema value and returns True if any "$ref" or
    # "$dynamicRef" is not a string starting with "#" (i.e. could trigger remote/file retrieval).
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                return True
            if _remote_reference(child):
                return True
    elif isinstance(value, list):
        return any(_remote_reference(child) for child in value)
    return False


class _RuntimeGrader:
    # [CLASS DOC]
    # HELPER: base class for runtime graders: Level.RUNTIME, no capability/provider
    # requirements, depends on "runtime.startup". Subclasses set check_id and implement
    # check(subject, evidence) returning a list of problem strings.
    level = Level.RUNTIME
    requires_capabilities = frozenset()
    requires_provider = frozenset()
    depends_on = ("runtime.startup",)

    def applies_to(self, manifest) -> bool:
        # [FUNCTION DOC]
        # Always returns True.
        return True

    def run(self, subject) -> CheckResult:
        # [FUNCTION DOC]
        # Returns a CheckResult: SKIP (duration 0) if there is no runtime evidence, if evidence
        # has a failure_reason, or if no "step" exchange was recorded. Otherwise calls check();
        # ValueError/TypeError/KeyError/RecursionError/OverflowError become a single
        # "malformed runtime evidence" problem. FAIL if any problems (first 20 as evidence,
        # with remediation text) else PASS; measured holds the exchange count; duration is wall time.
        started = time.monotonic()
        evidence = subject.runtime_evidence
        if evidence is None:
            return CheckResult(
                check_id=self.check_id,
                status=CheckStatus.SKIP,
                evidence=["runtime evidence is unavailable"],
                duration_s=0,
            )
        if evidence.failure_reason:
            return CheckResult(
                check_id=self.check_id,
                status=CheckStatus.SKIP,
                evidence=["unmet dependency: runtime.startup (collection incomplete)"],
                duration_s=0,
            )
        if not any(row.operation == "step" for row in evidence.exchanges):
            return CheckResult(
                check_id=self.check_id,
                status=CheckStatus.SKIP,
                evidence=["no step was observed; the episode contract is incomplete"],
                duration_s=0,
            )
        problems = []
        try:
            problems.extend(self.check(subject, evidence))
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            problems.append("malformed runtime evidence")
        return CheckResult(
            check_id=self.check_id,
            status=CheckStatus.FAIL if problems else CheckStatus.PASS,
            measured={"exchanges": len(evidence.exchanges)},
            evidence=problems[:20] or ["observed runtime contract holds"],
            remediation=(
                "Fix the reported runtime contract violations." if problems else None
            ),
            duration_s=time.monotonic() - started,
        )


class RewardWellFormedGrader(_RuntimeGrader):
    # [CLASS DOC]
    # Checks every reset/step response carries a reward that is a finite int/float within
    # the manifest's declared reward range (reset may have reward null).
    """Require numeric step rewards for Level Two, beyond the core API contract."""

    check_id = "runtime.reward_well_formed"

    def check(self, subject, evidence) -> list[str]:
        # [FUNCTION DOC]
        # For each reset/step exchange parses response_json["data"]; reports a malformed
        # envelope if not a dict, missing reward, or reward that is not exactly int/float
        # (bools rejected), non-finite, or out of [low, high]. Adds a problem if no
        # observations were seen. Returns the problem list. KeyError/JSON errors propagate to run().
        problems = []
        observations = 0
        low, high = subject.manifest.reward.range
        for index, exchange in enumerate(evidence.exchanges):
            if exchange.operation not in {"reset", "step"}:
                continue
            observations += 1
            data = json.loads(exchange.response_json)["data"]
            if not isinstance(data, dict):
                problems.append(f"exchange {index}: malformed observation envelope")
                continue
            if "reward" not in data:
                problems.append(f"exchange {index}: missing reward")
                continue
            reward = data["reward"]
            if reward is None and exchange.operation == "reset":
                continue
            if (
                type(reward) not in (int, float)
                or (type(reward) is float and not math.isfinite(reward))
                or not low <= reward <= high
            ):
                problems.append(f"exchange {index}: reward must be finite and in range")
        if observations == 0:
            problems.append("no observations were measured")
        return problems


class ObservationSchemaGrader(_RuntimeGrader):
    # [CLASS DOC]
    # Validates observation envelopes and their observations against the environment's
    # published observation schema (and optional distinct reset schema).
    """Validate raw envelopes and reconstructed observations against the schema."""

    check_id = "runtime.observation_schema"

    def check(self, subject, evidence) -> list[str]:
        # [FUNCTION DOC]
        # Returns a problem immediately if the schema is unavailable or either schema contains
        # a non-local $ref. Checks each reset/step response envelope is type "observation" with
        # a dict data containing dict observation, bool done and a reward key; collects valid
        # raw response JSON. Then runs runtime/schema_worker.py in an isolated Python
        # subprocess (empty env, 5s timeout), passing schemas and observations as JSON on stdin;
        # non-zero exit or timeout add a budget problem, otherwise the worker's JSON list of
        # problems is redacted and appended. Returns the problems.
        # Side effects: spawns a subprocess.
        if evidence.observation_schema_json is None:
            return ["observation schema unavailable"]
        schema = json.loads(evidence.observation_schema_json)
        # Submitted schemas must not cause host-side HTTP/file retrieval.
        if _remote_reference(schema):
            return ["observation schema has a non-local reference"]
        if evidence.reset_observation_schema_json is not None:
            reset_schema = json.loads(evidence.reset_observation_schema_json)
            if _remote_reference(reset_schema):
                return ["reset observation schema has a non-local reference"]
        problems = []
        observations = []
        count = 0
        for index, exchange in enumerate(evidence.exchanges):
            if exchange.operation not in {"reset", "step"}:
                continue
            count += 1
            response = json.loads(exchange.response_json)
            data = response["data"]
            if (
                response.get("type") != "observation"
                or not isinstance(data, dict)
                or not isinstance(data.get("observation"), dict)
                or type(data.get("done")) is not bool
                or "reward" not in data
            ):
                problems.append(f"exchange {index}: malformed observation envelope")
                continue
            # Preserve the bounded wire representation across the subprocess
            # boundary: normalizing numbers such as 1e9 can inflate a valid
            # episode beyond the worker's input limit.
            observations.append(
                {
                    "index": index,
                    "operation": exchange.operation,
                    "response_json": exchange.response_json,
                }
            )
        if count == 0:
            problems.append("no observations were measured")
        # A subject-supplied regex or recursive schema can exhaust CPU. Keep all
        # schema evaluation in a disposable interpreter with a hard wall deadline.
        worker = Path(__file__).resolve().parents[2] / "runtime" / "schema_worker.py"
        try:
            checked = subprocess.run(
                [sys.executable, "-I", str(worker)],
                input=json.dumps(
                    {
                        "schema_json": evidence.observation_schema_json,
                        "reset_schema_json": evidence.reset_observation_schema_json,
                        "observations": observations,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
                env={},
            )
            if checked.returncode:
                problems.append(
                    "observation schema evaluation exceeded its resource budget"
                )
            else:
                problems.extend(_redact(json.loads(checked.stdout)))
        except subprocess.TimeoutExpired:
            problems.append("observation schema evaluation exceeded its time budget")
        return problems


class StateContractGrader(_RuntimeGrader):
    # [CLASS DOC]
    # Checks state snapshots report the episode id given at the last reset and the number
    # of steps since that reset.
    """Verify episode identity and step counts in the same replayed session."""

    check_id = "runtime.state_contract"

    def check(self, subject, evidence) -> list[str]:
        # [FUNCTION DOC]
        # Walks exchanges in order: reset records request data.episode_id and zeroes the step
        # count; step increments it; state responses must be type "state" with dict data whose
        # episode_id equals the tracked one and whose step_count is an int equal to the count.
        # Adds a problem if no state exchange was seen. Returns problems.
        problems = []
        episode_id = None
        steps = 0
        states = 0
        for index, exchange in enumerate(evidence.exchanges):
            if exchange.operation == "reset":
                episode_id = json.loads(exchange.request_json)["data"]["episode_id"]
                steps = 0
            elif exchange.operation == "step":
                steps += 1
            elif exchange.operation == "state":
                states += 1
                response = json.loads(exchange.response_json)
                data = response["data"]
                if not isinstance(data, dict):
                    problems.append(f"exchange {index}: malformed state envelope")
                    continue
                if (
                    response.get("type") != "state"
                    or data.get("episode_id") != episode_id
                ):
                    problems.append(f"exchange {index}: episode_id differs from reset")
                if (
                    type(data.get("step_count")) is not int
                    or data["step_count"] != steps
                ):
                    problems.append(f"exchange {index}: incorrect step_count")
        if states == 0:
            problems.append("no state snapshots were measured")
        return problems
