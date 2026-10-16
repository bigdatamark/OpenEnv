# [MODULE DOC]
# Pydantic models and frozen dataclasses for runtime validation: the data-only runtime plan
# (reset inputs and actions), a launch specification for providers, and the raw evidence records
# produced by the collector. Also loads and validates the plan JSON sidecar from a package.
# Limits: plan size 64 KiB, at most 100 actions, JSON depth 32, at most 10,000 JSON nodes.

"""Versioned runtime inputs and execution evidence, independent of a provider."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

from ..manifest import ExecutionDeclaration, NetworkPolicy, ResourceDeclaration

MAX_PLAN_BYTES = 65_536
MAX_PLAN_ACTIONS = 100
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 10_000


class RuntimePlanError(ValueError):
    # [CLASS DOC]
    # ValueError raised when a runtime plan cannot be read safely or fails validation.
    """A public runtime plan cannot be read safely or violates its data contract."""


def _check_json(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> None:
    # [FUNCTION DOC]
    # HELPER: Recursively verifies a value is pure JSON data: None, bool, int, finite float, str,
    # dict with str keys, or list (exact types, no subclasses). Decrements a shared node budget
    # (a single-element list, defaulting to 10,000) per node.
    # Raises ValueError for non-JSON types, non-string keys, non-finite floats, budget exhaustion, or
    # depth over 32. Returns None.
    """Reject executable objects, non-finite numbers and excessive nesting."""
    if budget is None:
        budget = [MAX_JSON_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_JSON_DEPTH:
        raise ValueError("runtime JSON exceeds the node or nesting limit")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("runtime JSON numbers must be finite")
        return
    if type(value) is str:
        return
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("runtime JSON object keys must be strings")
        children = value.values()
    elif type(value) is list:
        children = value
    else:
        raise ValueError("runtime inputs must contain only JSON data")
    for child in children:
        _check_json(child, depth=depth + 1, budget=budget)


class RuntimeReset(BaseModel):
    # [CLASS DOC]
    # Strict, frozen, extra-forbidding model for reset inputs: episode_id (1-128 chars), seed
    # (0..2^32-1), options (dict of extra JSON reset args).
    """
    Explicit reset inputs for one measured episode.

    Attributes:
        episode_id (`str`):
            Requested episode identity; verified by the state contract grader.
        seed (`int`):
            Requested seed. Acceptance alone does not establish determinism.
        options (`dict[str, Any]`, *optional*):
            Additional public reset arguments, never replacements for the seed or ID.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    episode_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=2**32 - 1)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def _public_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Validator for options: raises ValueError if it contains episode_id or seed keys, checks
        # it is pure JSON, and returns it unchanged.
        if {"episode_id", "seed"} & value.keys():
            raise ValueError("reset options cannot override episode_id or seed")
        _check_json(value)
        return value


class RuntimePlan(BaseModel):
    # [CLASS DOC]
    # Strict, frozen, extra-forbidding model: plan_schema_version must be "1", reset (RuntimeReset),
    # actions (list of 1..100 dicts).
    """
    Bounded, data-only runtime inputs. Capability declarations stay in the manifest.

    Attributes:
        plan_schema_version (`str`):
            The pinned sidecar schema version, `"1"`.
        reset ([`~openenv.validation.runtime.contracts.RuntimeReset`]):
            Inputs to the first reset on the measured WebSocket session.
        actions (`list[dict[str, Any]]`):
            Between one and 100 public actions, applied in order until termination.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    plan_schema_version: Literal["1"]
    reset: RuntimeReset
    actions: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_PLAN_ACTIONS)

    @field_validator("actions")
    @classmethod
    def _data_actions(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Validator ensuring actions are pure JSON; returns them unchanged.
        _check_json(value)
        return value

    @model_validator(mode="after")
    def _total_size(self) -> "RuntimePlan":
        # [FUNCTION DOC]
        # HELPER: After-validator raising ValueError if the model's JSON serialization exceeds 64 KiB; returns self.
        size = len(self.model_dump_json().encode("utf-8"))
        if size > MAX_PLAN_BYTES:
            raise ValueError(f"runtime plan exceeds {MAX_PLAN_BYTES} bytes")
        return self


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: JSON object_pairs_hook that builds a dict, raising ValueError on duplicate keys.
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate runtime JSON key")
        result[key] = value
    return result


def load_runtime_plan(root: Path, execution: ExecutionDeclaration) -> RuntimePlan:
    # [FUNCTION DOC]
    # Reads <root>/<execution.probe_path>, requiring it to resolve to a regular file inside the
    # resolved root; reads at most 64 KiB + 1 bytes and rejects oversize; parses JSON with duplicate-key
    # rejection, checks pure JSON, and validates into RuntimePlan.
    # Returns the RuntimePlan.
    # Raises RuntimePlanError: for pydantic validation errors (message lists up to 5 errors with
    # locations where unknown field names are replaced by "<field>", error type and message, plus a total
    # count), JSON decode errors (line/column), Unicode errors, OS errors (type name only), and other
    # ValueError/RecursionError. Side effects: filesystem reads.
    """
    Read a bounded runtime plan without importing any subject code.

    Args:
        root (`Path`):
            Package root; the resolved plan must stay inside this directory.
        execution ([`~openenv.validation.manifest.ExecutionDeclaration`]):
            Manifest-owned location of the JSON sidecar.

    Returns:
        [`~openenv.validation.runtime.contracts.RuntimePlan`]: validated public inputs.

    Raises:
        [`~openenv.validation.runtime.contracts.RuntimePlanError`]:
            Missing, escaped, oversized, ambiguous or invalid input.
    """
    try:
        package_root = Path(root).resolve(strict=True)
        path = (package_root / execution.probe_path).resolve(strict=True)
        if not path.is_relative_to(package_root) or not path.is_file():
            raise ValueError("runtime plan must be a regular file inside the package")
        with path.open("rb") as source:
            payload = source.read(MAX_PLAN_BYTES + 1)
        if len(payload) > MAX_PLAN_BYTES:
            raise ValueError(f"runtime plan exceeds {MAX_PLAN_BYTES} bytes")
        raw = json.loads(payload, object_pairs_hook=_unique_object)
        _check_json(raw)
        return RuntimePlan.model_validate(raw)
    except ValidationError as exc:
        fields = RuntimePlan.model_fields.keys() | RuntimeReset.model_fields.keys()
        errors = []
        for error in exc.errors(
            include_input=False, include_context=False, include_url=False
        )[:5]:
            location = ".".join(
                str(part) if isinstance(part, int) or part in fields else "<field>"
                for part in error["loc"]
            )
            # These messages come from the fixed schema and validators; raw inputs,
            # exception context and subject-controlled field names are omitted.
            errors.append(f"{location or '<root>'}: {error['type']} ({error['msg']})")
        if exc.error_count() > 5:
            errors.append(f"... ({exc.error_count()} errors total)")
        raise RuntimePlanError(
            "runtime plan schema validation failed: " + "; ".join(errors)
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimePlanError(
            f"invalid runtime plan JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except UnicodeError as exc:
        raise RuntimePlanError("runtime plan contains invalid text encoding") from exc
    except OSError as exc:
        raise RuntimePlanError(
            f"runtime plan could not be read ({type(exc).__name__})"
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise RuntimePlanError(f"invalid runtime plan: {exc}") from exc


class LaunchSpec(BaseModel):
    # [CLASS DOC]
    # Frozen, extra-forbidding model describing a validation-only container launch: image_ref (image
    # id or digest matching sha256 pattern), resources, network policy, run_id (lowercase DNS-like
    # label up to 63 chars), startup_timeout_s (0 < x <= 300, default 30), env_vars (at most 64).
    """
    Explicit validation-only launch request; no caller environment is inherited.

    Attributes:
        image_ref (`str`):
            Immutable local image ID or repository digest.
        resources ([`~openenv.validation.manifest.ResourceDeclaration`]):
            Subject CPU, memory, writable-storage and episode budgets.
        network ([`~openenv.validation.manifest.NetworkPolicy`]):
            Requested policy; unsupported enforcement must be refused.
        run_id (`str`):
            Unique run-owned resource label, safe for provider names.
        startup_timeout_s (`float`, *optional*, defaults to 30):
            Independent readiness deadline, at most 300 seconds.
        env_vars (`dict[str, str]`, *optional*):
            Explicit environment only; providers do not forward host credentials.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_ref: str = Field(pattern=r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
    resources: ResourceDeclaration
    network: NetworkPolicy
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    startup_timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    env_vars: dict[str, str] = Field(default_factory=dict, max_length=64)

    @field_validator("env_vars")
    @classmethod
    def _explicit_environment(cls, value: dict[str, str]) -> dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: Validator: each env var name must be a valid identifier-like name; values must not
        # contain NUL or exceed 8192 UTF-8 bytes. Raises ValueError otherwise; returns the mapping.
        for name, content in value.items():
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError("invalid environment variable name")
            if "\x00" in content or len(content.encode("utf-8")) > 8192:
                raise ValueError("environment value contains NUL or exceeds 8192 bytes")
        return value

    @field_validator("resources")
    @classmethod
    def _finite_resources(cls, value: ResourceDeclaration) -> ResourceDeclaration:
        # [FUNCTION DOC]
        # HELPER: Validator raising ValueError if resources.cpu or resources.episode_timeout_s is not finite.
        if not math.isfinite(value.cpu) or not math.isfinite(value.episode_timeout_s):
            raise ValueError("runtime resource limits must be finite")
        return value


@dataclass(frozen=True)
class WireExchange:
    # [CLASS DOC]
    # Frozen dataclass recording one raw exchange: operation (reset/step/state), request_json and
    # response_json as the original serialized strings.
    """
    One immutable raw exchange, captured before client defaults or type coercion.

    Attributes:
        operation (`str`):
            The reset, step or state operation performed on the measured session.
        request_json (`str`):
            Original serialized request; parse only when grading.
        response_json (`str`):
            Original serialized response, including malformed JSON for diagnostics.
    """

    operation: Literal["reset", "step", "state"]
    request_json: str
    response_json: str


@dataclass(frozen=True)
class RuntimeEvidence:
    # [CLASS DOC]
    # Frozen dataclass holding collector output: tuple of exchanges, step observation schema JSON,
    # optional failure phase and reason, and optional reset observation schema JSON. All default empty/None.
    """
    Collector evidence shared by graders without permitting mutation of the episode.

    Attributes:
        exchanges (`tuple[WireExchange, ...]`):
            Ordered operations on one WebSocket session.
        observation_schema_json (`str`, *optional*):
            The advertised observation schema, exactly `/schema`'s observation value.
        failure_phase (`str`, *optional*):
            Collector phase that failed; a truncated transcript cannot pass silently.
        failure_reason (`str`, *optional*):
            Credential-safe explanation of the collection failure.
        reset_observation_schema_json (`str`, *optional*):
            Explicit `/schema` reset_observation value; absent means use the step schema.
    """

    exchanges: tuple[WireExchange, ...] = ()
    observation_schema_json: str | None = None
    failure_phase: str | None = None
    failure_reason: str | None = None
    reset_observation_schema_json: str | None = None
