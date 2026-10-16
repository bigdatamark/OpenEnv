# [MODULE DOC]
# Command-line evaluation harness that runs ThinkingBox benchmark manifests through an OpenEnv ThinkingBox
# server. For each selected UID x repetition it connects a client, resets, drives either the natively
# configured ThinkingBox agent (tool calls forwarded through OpenEnv) or a custom policy, and converts the
# terminal result into a privacy-reviewed canonical DecodeResult JSONL record; operational failures go to
# a redacted errors JSONL sidecar and are retryable. Supports sharding, limits, resume, strict coverage
# validation (canary = 1 repetition, full = 20 over the canonical manifest) and native aggregation.
# Provenance (runtime, data, config, framework source hash, policy, run parameters) is embedded in every
# record and must match exactly. Module constants list safe model fields, the result schema version (3),
# native finish reasons, forbidden field names for canonical output, and framework source paths to hash.
"""Run ThinkingBox manifests through OpenEnv with native model configuration.

The harness preserves native agent, user, judge, and aggregation semantics
while emitting privacy-reviewed canonical JSONL plus operational error sidecars.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from thinkingbox.common.agent_session_base import should_end_conversation
from thinkingbox.common.agent_session_factory import get_agent_session_factory
from thinkingbox.common.chat_types import (
    DecodeResult,
    FinishReason,
    MessageT,
    ParallelToolCall,
    TestResult,
    Text,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import ConfigFile, update_tools_with_client_config
from thinkingbox.common.hydrator import get_dataset_case_by_name
from thinkingbox.common.usage_types import Usage

from thinkingbox_env.benchmark_data import load_test_uids, resolve_data_bundle
from thinkingbox_env.client import DEFAULT_MESSAGE_TIMEOUT_S, ThinkingBoxEnv
from thinkingbox_env.models import (
    CANONICAL_TEST_UIDS,
    DATA_BUNDLE_SHA256,
    DATA_COMMIT,
    DATA_MANIFEST_PATH,
    DATA_RELEASE_NAME,
    SubmittedToolCall,
    ThinkingBoxExecutionProvenance,
)
from thinkingbox_env.runtime import (
    load_thinkingbox_config as load_native_thinkingbox_config,
    load_thinkingbox_config_with_sha256 as load_native_config_with_sha256,
    load_thinkingbox_runtime_provenance,
    require_canonical_thinkingbox_runtime,
)


_SAFE_MODEL_FIELDS = (
    "type",
    "deployment",
    "model",
    "factory",
    "reasoning_effort",
    "reasoning_source",
    "is_reasoning",
    "temperature",
    "max_completion_tokens",
    "parallel_tool_calls",
)
_RESULT_SCHEMA_VERSION = 3
_BOTDESIGNER_ACTIVITY_ERROR = "BotDesignerActivityError"
_NATIVE_FINISH_REASONS: set[str] = {
    "done",
    "end_turn_tool",
    "agent_error",
    "agent_limit",
    "user_limit",
    "user_done",
    "no_user_llm",
    "skipped",
}
_FORBIDDEN_CANONICAL_FIELDS = {
    "api_key",
    "assertion",
    "assertion_source",
    "assertions",
    "authorization",
    "credential",
    "credentials",
    "effects",
    "expected_state",
    "golden_state",
    "golden_state_model",
    "headers",
    "init_result",
    "line_content",
    "private_user_context",
    "prints",
    "secret",
    "session_id",
    "tb",
    "test_code",
    "test_context",
    "test_source",
    "token",
    "traceback",
    "user_context",
    "world_state",
}
_FRAMEWORK_EXACT_PATHS = {
    "envs/thinkingbox_env/openenv.yaml",
    "envs/thinkingbox_env/pyproject.toml",
    "envs/thinkingbox_env/uv.lock",
    "examples/thinkingbox/eval_testlist.py",
    "pyproject.toml",
    "uv.lock",
}
_RUNTIME_FRAMEWORK_EXACT_PATHS = _FRAMEWORK_EXACT_PATHS - {
    "pyproject.toml",
    "uv.lock",
}
_FRAMEWORK_SOURCE_PREFIXES = (
    Path("envs/thinkingbox_env"),
    Path("src/openenv/core"),
)
_FRAMEWORK_IGNORED_PARTS = {
    ".cache",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "_build",
    "build",
    "dist",
    "generated",
    "htmlcov",
    "site-packages",
    "venv",
}


class CoverageError(ValueError):
    # [CLASS DOC]
    # ValueError subclass raised when canonical results do not exactly and validly cover the requested plan.
    """Canonical results do not exactly cover the requested evaluation plan."""


class _OperationalResultError(RuntimeError):
    # [CLASS DOC]
    # HELPER: RuntimeError for an operational (non-benchmark) failure, optionally carrying the offending
    # client result for auditing.
    def __init__(self, message: str, result: Any | None = None):
        # [FUNCTION DOC]
        # Stores the message and the optional result.
        super().__init__(message)
        self.result = result


@dataclass
class _ExecutionTrace:
    # [CLASS DOC]
    # HELPER: mutable accumulator of an episode's native messages, usage records, max steps taken, and the
    # server-reported execution provenance (plus whether it was valid and consistent).
    messages: list[MessageT] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)
    steps_taken: int = 0
    execution_provenance: dict[str, Any] | None = None
    execution_provenance_valid: bool = True

    def capture_agent(self, agent: Any) -> None:
        # [FUNCTION DOC]
        # Copies the agent conversation's messages (deep) and its valid usage entries (invalid ones skipped); no-op
        # if the agent has no conversation.
        conversation = getattr(agent, "conversation", None)
        if conversation is None:
            return
        self.messages = [
            message.model_copy(deep=True) for message in conversation.messages
        ]
        raw_usage = conversation.metadata.get("usage") or []
        self.usage = []
        for item in raw_usage:
            try:
                self.usage.append(Usage.model_validate(item))
            except (TypeError, ValueError):
                continue

    def capture_result(self, result: Any) -> None:
        # [FUNCTION DOC]
        # From a client result's observation: keeps the max non-negative steps_taken, and validates
        # metadata.execution_provenance. Marks provenance invalid if it fails validation or differs from the first
        # one seen; otherwise stores the first normalised value.
        observation = getattr(result, "observation", None)
        if observation is None:
            return
        steps_taken = getattr(observation, "steps_taken", None)
        if isinstance(steps_taken, int) and steps_taken >= 0:
            self.steps_taken = max(self.steps_taken, steps_taken)
        metadata = getattr(observation, "metadata", None)
        candidate = (
            metadata.get("execution_provenance") if isinstance(metadata, dict) else None
        )
        try:
            validated = ThinkingBoxExecutionProvenance.model_validate(candidate)
        except (TypeError, ValueError):
            self.execution_provenance_valid = False
            return
        normalized = validated.model_dump(mode="json")
        if self.execution_provenance is None:
            self.execution_provenance = normalized
        elif normalized != self.execution_provenance:
            self.execution_provenance_valid = False


@dataclass
class _EpisodeOutcome:
    # [CLASS DOC]
    # HELPER: dataclass of an episode's final client result (or None), the exception raised (or None) and its
    # trace.
    result: Any | None
    error: Exception | None
    trace: _ExecutionTrace


@dataclass(frozen=True)
class CoverageReport:
    # [CLASS DOC]
    # Frozen dataclass summarising validated coverage: tuple of canonical DecodeResults, UID count, repetition
    # indexes and total result count.
    """Summarize exact validated UID-by-repetition coverage.

    Args:
        results (`tuple[thinkingbox.common.chat_types.DecodeResult, ...]`):
            Canonical validated records.
        uid_count (`int`):
            Number of ordered manifest UIDs.
        repetitions (`tuple[int, ...]`):
            Exact validated repetition indexes.
        total_results (`int`):
            Total canonical record count.
    """

    results: tuple[DecodeResult, ...]
    uid_count: int
    repetitions: tuple[int, ...]
    total_results: int


class _EpisodeEnded(RuntimeError):
    # [CLASS DOC]
    # HELPER: exception used to unwind the native agent loop when a tool call made the episode terminal;
    # carries the terminal client result.
    def __init__(self, result: Any):
        # [FUNCTION DOC]
        # Stores the terminal result with a fixed message.
        super().__init__("ThinkingBox episode ended while executing a tool")
        self.result = result


class _OpenEnvProxy:
    # [CLASS DOC]
    # HELPER: minimal native MCP proxy context handed to the native agent; forwards tool calls through the
    # OpenEnv client. Matches individual call_tool requests to calls in the agent's latest provider batch so a
    # parallel batch is submitted to OpenEnv once, and serves each call's result from the batch response.
    """Minimal native MCP context that forwards calls through OpenEnv."""

    def __init__(
        self,
        env: ThinkingBoxEnv,
        tools: list[ToolDef],
        trace: _ExecutionTrace | None = None,
    ):
        # [FUNCTION DOC]
        # Stores the client, deep copies of the public tool definitions and optional trace, and initialises the
        # bound agent, returned call ids, per-batch submission tasks, batch results and message start index.
        self._env = env
        self._tools = [tool.model_copy(deep=True) for tool in tools]
        self._trace = trace
        self._agent: Any = None
        self._returned_call_ids: set[str] = set()
        self._batch_submissions: dict[int, asyncio.Task[None]] = {}
        self._batch_results: dict[str, str] = {}
        self._message_start = 0

    def bind(self, agent: Any) -> None:
        # [FUNCTION DOC]
        # Binds the native agent whose conversation is inspected for provider calls.
        self._agent = agent

    def begin_decode(self) -> None:
        # [FUNCTION DOC]
        # Records the current conversation length so later matching only considers messages from this decode.
        if self._agent is not None:
            self._message_start = len(self._agent.conversation.messages)

    async def list_tools(self) -> list[ToolDef]:
        # [FUNCTION DOC]
        # Returns deep copies of the tool definitions.
        return [tool.model_copy(deep=True) for tool in self._tools]

    def _provider_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[Any, ParallelToolCall] | None:
        # [FUNCTION DOC]
        # HELPER: searches batches added since begin_decode for a not-yet-returned call with the same name and
        # arguments; prefers a call without a parse error, else returns the first errored match. Returns (call,
        # batch message) or None (also when no agent is bound).
        if self._agent is None:
            return None
        errored_match: tuple[Any, ParallelToolCall] | None = None
        for message in self._agent.conversation.messages[self._message_start :]:
            if not isinstance(message, ParallelToolCall):
                continue
            for call in message.tool_calls:
                if (
                    call.id not in self._returned_call_ids
                    and call.name == name
                    and call.arguments == arguments
                ):
                    if not call.metadata.get("error"):
                        return call, message
                    errored_match = errored_match or (call, message)
        return errored_match

    async def _submit_batch_once(self, message: ParallelToolCall) -> None:
        # [FUNCTION DOC]
        # HELPER: submits a batch via the client's call_tools, records the result in the trace, raises
        # _EpisodeEnded if the episode ended, RuntimeError if no batch results came back, and stores each child's
        # content by call id.
        result = await self._env.call_tools(
            [
                SubmittedToolCall(
                    name=call.name,
                    arguments=call.arguments,
                    call_id=call.id,
                    parse_error=call.metadata.get("error"),
                )
                for call in message.tool_calls
            ]
        )
        if self._trace is not None:
            self._trace.capture_result(result)
        if result.done:
            raise _EpisodeEnded(result)
        tool_results = result.observation.tool_results
        if tool_results is None:
            raise RuntimeError("OpenEnv returned no parallel tool results")
        self._batch_results.update(
            {child.call_id: child.content for child in tool_results}
        )

    async def submit_batch(self, message: ParallelToolCall) -> None:
        # [FUNCTION DOC]
        # Submits a batch at most once (keyed by message identity, shared task) and awaits it. End-turn batches are
        # skipped (they are submitted later as terminal turns).
        if message.metadata.get("is_end_turn_tool", False):
            return
        batch_key = id(message)
        submission = self._batch_submissions.get(batch_key)
        if submission is None:
            submission = asyncio.create_task(self._submit_batch_once(message))
            self._batch_submissions[batch_key] = submission
        await submission

    async def call_tool(self, tool_name: str, /, **arguments: Any) -> str:
        # [FUNCTION DOC]
        # If the call matches a provider batch call, marks it returned, ensures the batch is submitted and returns
        # that call's stored result (RuntimeError if missing). Otherwise makes a single call_tool request, records
        # it, raises _EpisodeEnded if terminal and RuntimeError if no tool result; returns the result text.
        matched = self._provider_call(tool_name, arguments)
        if matched is None:
            result = await self._env.call_tool(tool_name, arguments)
            if self._trace is not None:
                self._trace.capture_result(result)
            if result.done:
                raise _EpisodeEnded(result)
            observation = result.observation
            if observation.tool_result is None:
                raise RuntimeError("OpenEnv returned no ThinkingBox tool result")
            return observation.tool_result

        call, message = matched
        self._returned_call_ids.add(call.id)
        await self.submit_batch(message)

        try:
            return self._batch_results.pop(call.id)
        except KeyError as exc:
            raise RuntimeError("OpenEnv omitted a parallel tool result") from exc


def load_thinkingbox_config(path: str | Path) -> ConfigFile:
    # [FUNCTION DOC]
    # Loads and validates a native ThinkingBox config from a user-expanded, resolved path.
    """Load an ordinary native ThinkingBox configuration.

    Args:
        path (`str` or `pathlib.Path`):
            Native YAML configuration path.

    Returns:
        `thinkingbox.common.config_types.ConfigFile`:
            Validated native configuration with model fields preserved.
    """
    return load_native_thinkingbox_config(Path(path).expanduser().resolve())


def load_thinkingbox_config_with_sha256(
    path: str | Path,
) -> tuple[ConfigFile, str]:
    # [FUNCTION DOC]
    # Returns (native config, SHA-256 of its bytes) for a user-expanded, resolved path.
    """Load the exact native configuration payload and fingerprint.

    Args:
        path (`str` or `pathlib.Path`):
            Native YAML configuration path.

    Returns:
        `tuple[thinkingbox.common.config_types.ConfigFile, str]`:
            Validated configuration and SHA-256 fingerprint.
    """
    return load_native_config_with_sha256(Path(path).expanduser().resolve())


def configured_agent_session_factory(config: ConfigFile) -> Any:
    # [FUNCTION DOC]
    # Returns the native agent-session factory for the config's orchestrator.
    """Construct the native agent factory from the configured orchestrator.

    Args:
        config (`thinkingbox.common.config_types.ConfigFile`):
            Validated native configuration.

    Returns:
        A native ThinkingBox agent-session factory.
    """
    return get_agent_session_factory(config.orchestrator)


def _model_provenance(model: Any | None) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: returns only the safe fields (type, deployment, model, reasoning, temperature, etc.) of a model
    # config's JSON dump, or None for no model.
    if model is None:
        return None
    raw = model.model_dump(mode="json")
    return {field: raw[field] for field in _SAFE_MODEL_FIELDS if field in raw}


def config_provenance(
    config: ConfigFile,
    path: str | Path,
    *,
    sha256: str | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Returns credential-free config provenance: SHA-256 (given or computed from the file), safe agent/user/
    # judge model fields, judge type and user-can-end flag.
    """Return credential-free provenance for the exact native configuration.

    Args:
        config (`thinkingbox.common.config_types.ConfigFile`):
            Validated native configuration.
        path (`str` or `pathlib.Path`):
            Exact configuration payload path.
        sha256 (`str`, *optional*):
            Precomputed payload fingerprint.

    Returns:
        `dict`:
            Safe agent, user, judge, and configuration fingerprints.
    """
    config_path = Path(path).expanduser().resolve()
    return {
        "sha256": sha256 or hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "agent_model": _model_provenance(config.orchestrator.agent_model),
        "user_model": _model_provenance(config.user_model),
        "judge_model": _model_provenance(config.judge_model),
        "judge_type": config.judge_type,
        "user_can_end_conversation": config.user_can_end_conversation,
    }


def _sha256_file(path: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: SHA-256 hex of a file's bytes.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: SHA-256 hex of compact, key-sorted JSON.
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision(root: Path) -> str | None:
    # [FUNCTION DOC]
    # HELPER: returns HEAD commit of the git repo at root via a git subprocess, or None on failure/empty.
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _git_root(path: Path) -> Path | None:
    # [FUNCTION DOC]
    # HELPER: returns the resolved top-level dir of the git repo containing path, or None.
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    root = completed.stdout.strip()
    return Path(root).resolve() if root else None


def _git_tracked_files(root: Path) -> list[Path] | None:
    # [FUNCTION DOC]
    # HELPER: returns the repo's tracked files as relative Paths (git ls-files -z), or None on failure.
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [Path(raw.decode("utf-8")) for raw in completed.stdout.split(b"\0") if raw]


def _is_framework_path(relative: Path) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if a relative path is part of the framework fingerprint: not under ignored/build/egg-info
    # dirs, and either one of the exact framework files or a .py file under envs/thinkingbox_env or
    # src/openenv/core.
    if any(
        part in _FRAMEWORK_IGNORED_PARTS or part.endswith(".egg-info")
        for part in relative.parts
    ):
        return False
    if relative.as_posix() in _FRAMEWORK_EXACT_PATHS:
        return True
    return relative.suffix == ".py" and any(
        relative.is_relative_to(prefix) for prefix in _FRAMEWORK_SOURCE_PREFIXES
    )


def _framework_sha256(root: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: hashes framework files of a checkout (git-tracked when available, else exact paths plus .py files
    # under the source prefixes), sorted by path, hashing path and bytes with NUL separators; missing files are
    # skipped.
    # KNOWN ISSUE: appears unused in this module; the run provenance uses _runtime_framework_sha256 instead.
    resolved_root = root.resolve()
    tracked = _git_tracked_files(resolved_root)
    if tracked is not None:
        relative_candidates = [path for path in tracked if _is_framework_path(path)]
    else:
        relative_candidates = [Path(path) for path in sorted(_FRAMEWORK_EXACT_PATHS)]
        for prefix in _FRAMEWORK_SOURCE_PREFIXES:
            directory = resolved_root / prefix
            if directory.is_dir():
                relative_candidates.extend(
                    path.relative_to(resolved_root)
                    for path in directory.rglob("*.py")
                    if _is_framework_path(path.relative_to(resolved_root))
                )

    digest = hashlib.sha256()
    for relative_path in sorted(
        set(relative_candidates),
        key=lambda path: path.as_posix(),
    ):
        path = resolved_root / relative_path
        if not path.is_file():
            continue
        relative = relative_path.as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _module_source_root(module_name: str) -> Path:
    # [FUNCTION DOC]
    # HELPER: returns the source directory of an importable module or package. Raises RuntimeError if the
    # module or its location cannot be found.
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(f"Framework module {module_name!r} is unavailable")
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    raise RuntimeError(f"Framework module {module_name!r} has no source location")


def _framework_checkout_root(package_root: Path) -> Path | None:
    # [FUNCTION DOC]
    # HELPER: returns the git checkout root only if its envs/thinkingbox_env is the imported package directory
    # (i.e. running from a source checkout), else None.
    checkout_root = _git_root(package_root)
    if checkout_root is None:
        return None
    source_package = (checkout_root / "envs/thinkingbox_env").resolve()
    return checkout_root if source_package == package_root.resolve() else None


def _runtime_framework_sha256(checkout_root: Path | None) -> str:
    # [FUNCTION DOC]
    # HELPER: hashes all non-ignored .py files of the imported thinkingbox_env and openenv.core packages
    # (labelled by package) plus, when running from a checkout, the runtime exact framework files (pyproject /
    # lock files of the env and the example script). Returns the hex digest.
    roots = {
        "thinkingbox_env": _module_source_root("thinkingbox_env"),
        "openenv.core": _module_source_root("openenv.core"),
    }
    digest = hashlib.sha256()
    for label, root in sorted(roots.items()):
        files = sorted(
            (
                path
                for path in root.rglob("*.py")
                if not any(
                    part in _FRAMEWORK_IGNORED_PARTS or part.endswith(".egg-info")
                    for part in path.relative_to(root).parts
                )
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for path in files:
            relative = f"{label}/{path.relative_to(root).as_posix()}"
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")

    if checkout_root is not None:
        for relative in sorted(_RUNTIME_FRAMEWORK_EXACT_PATHS):
            path = checkout_root / relative
            if not path.is_file():
                continue
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _policy_provenance(spec: str | None, policy: Any | None) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: returns {spec, sha256 of the policy's source} or None if no policy. Falls back to hashing the
    # whole source file; raises ValueError if no source can be found.
    if spec is None or policy is None:
        return None
    try:
        source = inspect.getsource(policy).encode("utf-8")
    except (OSError, TypeError):
        source_path = inspect.getsourcefile(policy)
        if source_path is None:
            raise ValueError("configured policy source cannot be hashed")
        source = Path(source_path).read_bytes()
    return {
        "spec": spec,
        "sha256": hashlib.sha256(source).hexdigest(),
    }


def _base_provenance(
    *,
    config: ConfigFile,
    config_path: Path,
    config_sha256: str | None = None,
    bundle: Any,
    canonical_uids: list[str],
    requested_test_list: str | None,
    policy_spec: str | None,
    policy: Any | None,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: builds the run-independent provenance mapping: schema version, ThinkingBox runtime identity/hash/
    # type, data release/revision/bundle hash, config provenance, manifest path and hashes (recomputed and
    # required to match the bundle, ValueError otherwise), task count, requested test-list hash (or manifest
    # hash), framework revision and runtime hash, policy provenance and Python version.
    package_root = _module_source_root("thinkingbox_env")
    checkout_root = _framework_checkout_root(package_root)
    runtime = load_thinkingbox_runtime_provenance()
    manifest_sha256 = _sha256_file(bundle.manifest_path)
    manifest_uids_sha256 = _sha256_json(canonical_uids)
    if (
        bundle.manifest_sha256 != manifest_sha256
        or bundle.manifest_uids_sha256 != manifest_uids_sha256
    ):
        raise ValueError("resolved data fingerprints changed while loading the run")
    requested_sha256 = (
        _sha256_file(Path(requested_test_list).expanduser().resolve())
        if requested_test_list is not None
        else manifest_sha256
    )
    manifest_path = bundle.manifest_path.relative_to(bundle.root).as_posix()
    return {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "thinkingbox_revision": runtime.identity,
        "thinkingbox_source_sha256": runtime.source_sha256,
        "thinkingbox_source_type": runtime.source_type,
        "data_release": bundle.release_name,
        "data_revision": bundle.revision,
        "data_bundle_sha256": bundle.bundle_sha256,
        "config": config_provenance(
            config,
            config_path,
            sha256=config_sha256,
        ),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest_uids_sha256": manifest_uids_sha256,
        "task_count": len(canonical_uids),
        "requested_test_list_sha256": requested_sha256,
        "framework": {
            "revision": (
                _git_revision(checkout_root) if checkout_root is not None else None
            ),
            "sha256": _runtime_framework_sha256(checkout_root),
        },
        "policy": _policy_provenance(policy_spec, policy),
        "python": platform.python_version(),
    }


def _run_provenance(
    base: dict[str, Any],
    args: argparse.Namespace,
    selected_uids: list[str],
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: returns base provenance plus a "run" mapping: agent, repeat, repetition start, shard index/count,
    # limit, message timeout, SHA-256 of the base URL and of the selected UID list.
    return {
        **base,
        "run": {
            "agent": args.agent,
            "repeat": args.repeat,
            "repetition_start": args.repetition_start,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "limit": args.limit,
            "message_timeout_s": args.message_timeout,
            "base_url_sha256": hashlib.sha256(
                args.base_url.encode("utf-8")
            ).hexdigest(),
            "selected_uids_sha256": _sha256_json(selected_uids),
        },
    }


def _native_mcp_tools(test_case: Any, reset_observation: Any) -> list[ToolDef]:
    # [FUNCTION DOC]
    # HELPER: converts the reset observation's public tools (excluding the agent's built-in tools) into native
    # ToolDefs and applies the scenario's client tool config. Returns the list.
    builtin_names = {tool.name for tool in test_case.agent.builtin_tools}
    tools = [
        ToolDef(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )
        for tool in (reset_observation.tools or [])
        if tool.name not in builtin_names
    ]
    update_tools_with_client_config(tools, test_case.scenario.tools)
    return tools


def _terminal_tail(
    messages: list[Any],
) -> tuple[ParallelToolCall, str | None] | None:
    # [FUNCTION DOC]
    # HELPER: if the messages end the conversation via an end-turn tool, returns (last end-turn batch, content of
    # the directly preceding visible non-dummy assistant text or None); otherwise None.
    if should_end_conversation(messages) != "end_turn_tool":
        return None
    terminal_index = next(
        index
        for index in range(len(messages) - 1, -1, -1)
        if isinstance(messages[index], ParallelToolCall)
        and messages[index].metadata.get("is_end_turn_tool", False)
    )
    content: str | None = None
    if terminal_index > 0:
        adjacent = messages[terminal_index - 1]
        if (
            isinstance(adjacent, Text)
            and adjacent.role == "assistant"
            and adjacent.tag == "text"
            and not adjacent.metadata.get("is_dummy", False)
        ):
            content = adjacent.content
    return messages[terminal_index], content


async def run_configured_agent(
    env: ThinkingBoxEnv,
    reset_result: Any,
    test_case: Any,
    agent_session_factory: Any,
    trace: _ExecutionTrace | None = None,
) -> Any:
    # [FUNCTION DOC]
    # Drives the native configured agent against the OpenEnv episode. Builds the forwarding proxy and the
    # native agent (with history replayed locally), then loops: finishes with "agent_limit" at the turn limit;
    # decodes a turn from the current user message, submitting non-terminal tool batches (counting agent turns,
    # plus direct responses); if the turn ended with an end-turn tool, submits it as a terminal turn; else
    # submits the last visible assistant text (finishing with agent_limit/agent_error if there is none). Returns
    # on done; raises RuntimeError if the server rejects the message; finishes with agent_error if there is no
    # user reply or agent_limit at the limit; otherwise continues with the user's reply. An _EpisodeEnded during
    # decoding returns its result; other exceptions are re-raised after capturing the trace.
    # Returns: the terminal client result. Side effects: network calls, trace mutation.
    """Drive the native configured agent while OpenEnv owns private state.

    Args:
        env ([`ThinkingBoxEnv`]):
            Connected trusted-harness client.
        reset_result (`object`):
            Public reset result returned by OpenEnv.
        test_case (`object`):
            Locally hydrated native task used only to configure the agent loop.
        agent_session_factory (`collections.abc.Callable`):
            Native configured agent factory.
        trace (`_ExecutionTrace`, *optional*):
            Private execution trace accumulator.

    Returns:
        The terminal OpenEnv client result.
    """
    trace = trace or _ExecutionTrace()
    observation = reset_result.observation
    proxy = _OpenEnvProxy(
        env,
        _native_mcp_tools(test_case, observation),
        trace,
    )
    agent = agent_session_factory(
        config=test_case.agent,
        mcp_proxy=proxy,
        mcp_tools=await proxy.list_tools(),
        bot_instructions=observation.bot_instructions,
        scenario_metadata=test_case.scenario.metadata,
    )
    proxy.bind(agent)

    def record(result: Any) -> Any:
        # [FUNCTION DOC]
        # HELPER: captures the result and agent conversation into the trace and returns the result.
        trace.capture_result(result)
        trace.capture_agent(agent)
        return result

    if test_case.history:
        agent.add_messages(
            [message.model_copy(deep=True) for message in test_case.history]
        )

    user_message: Text | None = Text(
        role="user",
        content=observation.task or "",
    )
    agent_turns = 0
    while True:
        if agent_turns >= test_case.max_agent_sim_turns:
            return record(await env.finish("agent_limit"))

        completion_messages: list[Any] = []
        limit_reached = False
        proxy.begin_decode()
        try:
            async for message in agent.decode_turn_iter(user_message):
                if agent_turns >= test_case.max_agent_sim_turns:
                    limit_reached = True
                    break
                if isinstance(message, ParallelToolCall):
                    await proxy.submit_batch(message)
                    completion_messages.append(message)
                    if not message.metadata.get("is_end_turn_tool", False):
                        agent_turns += 1
                elif (
                    isinstance(message, Text)
                    and message.role == "assistant"
                    and message.is_visible
                ):
                    completion_messages.append(message)
                    if message.tag == "direct":
                        agent_turns += 1
        except _EpisodeEnded as exc:
            return record(exc.result)
        except Exception:
            trace.capture_agent(agent)
            raise
        trace.capture_agent(agent)

        terminal_tail = _terminal_tail(completion_messages)
        if terminal_tail is not None:
            terminal_message, adjacent_content = terminal_tail
            return record(
                await env.submit_message(
                    adjacent_content,
                    terminal_tool_calls=[
                        SubmittedToolCall(
                            name=call.name,
                            arguments=call.arguments,
                            call_id=call.id,
                            parse_error=call.metadata.get("error"),
                        )
                        for call in terminal_message.tool_calls
                    ],
                    tool_calls_before_content=False,
                )
            )

        assistant_text = next(
            (
                message.content
                for message in reversed(completion_messages)
                if isinstance(message, Text)
                and message.role == "assistant"
                and message.tag == "text"
                and not message.metadata.get("is_dummy", False)
            ),
            None,
        )
        if assistant_text is None:
            if limit_reached or agent_turns >= test_case.max_agent_sim_turns:
                return record(await env.finish("agent_limit"))
            return record(await env.finish("agent_error"))

        result = await env.submit_message(assistant_text)
        agent_turns += 1
        trace.capture_result(result)
        if result.done:
            return record(result)
        if result.observation.error is not None:
            raise RuntimeError("OpenEnv rejected an assistant message")
        if result.observation.user_message is None:
            return record(await env.finish("agent_error"))
        if agent_turns >= test_case.max_agent_sim_turns:
            return record(await env.finish("agent_limit"))
        user_message = Text(
            role="user",
            content=result.observation.user_message,
            metadata={"is_user_llm": True},
        )


def _load_policy(spec: str) -> Any:
    # [FUNCTION DOC]
    # HELPER: imports "MODULE:CALLABLE" and returns the callable. Raises ValueError without the colon,
    # TypeError if the target is not callable, and import/attribute errors as raised.
    module_name, separator, attr = spec.partition(":")
    if not separator:
        raise ValueError("--policy must use MODULE:CALLABLE syntax")
    policy = getattr(importlib.import_module(module_name), attr)
    if not callable(policy):
        raise TypeError("--policy target is not callable")
    return policy


async def _invoke_policy(
    policy: Any,
    env: ThinkingBoxEnv,
    reset_result: Any,
    repetition_id: str,
) -> Any:
    # [FUNCTION DOC]
    # HELPER: calls policy(env, reset_result, repetition_id), awaiting the result if needed, and returns it.
    """Invoke the public policy contract without trusted evaluator inputs."""
    result = policy(env, reset_result, repetition_id)
    if inspect.isawaitable(result):
        result = await result
    return result


def _load_requested_uids(path: str | None, canonical: list[str]) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: returns the canonical UIDs when path is None; otherwise loads a YAML list of UIDs and raises
    # ValueError if it is not a list of strings, has duplicates or contains UIDs not in the canonical list.
    if path is None:
        return canonical
    import yaml

    raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(uid, str) for uid in raw):
        raise ValueError("test list must be a YAML list of ThinkingBox UIDs")
    if len(raw) != len(set(raw)):
        raise ValueError("test list contains duplicate ThinkingBox UIDs")
    unknown = sorted(set(raw) - set(canonical))
    if unknown:
        raise ValueError(f"test list contains {len(unknown)} unknown UIDs")
    return raw


def _repetition_id(uid: str, repetition: int) -> str:
    # [FUNCTION DOC]
    # HELPER: returns "<uid>::repeat::<repetition>".
    return f"{uid}::repeat::{repetition}"


def _expected_repetition_ids(
    uids: Sequence[str],
    repetitions: Sequence[int],
) -> dict[str, tuple[str, int]]:
    # [FUNCTION DOC]
    # HELPER: returns a mapping from repetition id to (uid, repetition) for the cross product. Raises
    # CoverageError for duplicate UIDs or repetition indexes.
    if len(uids) != len(set(uids)):
        raise CoverageError("manifest contains duplicate UIDs")
    if len(repetitions) != len(set(repetitions)):
        raise CoverageError("required repetition indexes contain duplicates")
    return {
        _repetition_id(uid, repetition): (uid, repetition)
        for uid in uids
        for repetition in repetitions
    }


def _iter_exception_chain(exc: Exception) -> Iterable[Exception]:
    # [FUNCTION DOC]
    # HELPER: yields the exception and its __cause__ chain (Exceptions only, stopping on cycles).
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _is_botdesigner_exception(exc: Exception) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if any exception in the cause chain has the BotDesignerActivityError type name.
    return any(
        type(current).__name__ == _BOTDESIGNER_ACTIVITY_ERROR
        for current in _iter_exception_chain(exc)
    )


def _contains_error_type(value: Any, error_type: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: recursively searches mappings and sequences for error_type, either as the value of a
    # type/error_type/exception_type key or as a plain value. Returns a bool.
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"type", "error_type", "exception_type"} and child == error_type:
                return True
            if _contains_error_type(child, error_type):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_error_type(child, error_type) for child in value)
    return value == error_type


def _is_botdesigner_result(result: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the result's observation error, test summary or metadata mentions the
    # BotDesignerActivityError type.
    observation = getattr(result, "observation", None)
    if observation is None:
        return False
    return _contains_error_type(
        {
            "error": getattr(observation, "error", None),
            "test_summary": getattr(observation, "test_summary", None),
            "metadata": getattr(observation, "metadata", None),
        },
        _BOTDESIGNER_ACTIVITY_ERROR,
    )


def _native_messages(outcome: _EpisodeOutcome) -> list[MessageT]:
    # [FUNCTION DOC]
    # HELPER: returns deep copies of the terminal observation's messages validated as native messages, or of the
    # trace's messages when the observation has none.
    result = outcome.result
    observation = getattr(result, "observation", None)
    raw_messages = getattr(observation, "messages", None) if observation else None
    if raw_messages:
        probe = DecodeResult(
            uid="_message_validation",
            messages=raw_messages,
            usage=[],
        )
        return [message.model_copy(deep=True) for message in probe.messages]
    return [message.model_copy(deep=True) for message in outcome.trace.messages]


def _native_usage(outcome: _EpisodeOutcome) -> list[Usage]:
    # [FUNCTION DOC]
    # HELPER: returns deep copies of trace usage if any; otherwise valid usage entries from the result's or the
    # observation's metadata "usage" list; else [].
    if outcome.trace.usage:
        return [usage.model_copy(deep=True) for usage in outcome.trace.usage]
    result = outcome.result
    observation = getattr(result, "observation", None)
    candidates = (
        getattr(result, "metadata", None),
        getattr(observation, "metadata", None) if observation else None,
    )
    for metadata in candidates:
        if isinstance(metadata, dict) and isinstance(metadata.get("usage"), list):
            usage: list[Usage] = []
            for item in metadata["usage"]:
                try:
                    usage.append(Usage.model_validate(item))
                except (TypeError, ValueError):
                    continue
            return usage
    return []


def _message_step_counts(messages: Sequence[MessageT]) -> dict[str, int]:
    # [FUNCTION DOC]
    # HELPER: returns counts of native messages, assistant texts, user texts, tool batches and tool responses.
    assistant_messages = 0
    user_messages = 0
    tool_batches = 0
    tool_responses = 0
    for message in messages:
        if isinstance(message, Text):
            if message.role == "assistant":
                assistant_messages += 1
            elif message.role == "user":
                user_messages += 1
        elif isinstance(message, ParallelToolCall):
            tool_batches += 1
        elif isinstance(message, ToolResponse):
            tool_responses += 1
    return {
        "native_messages": len(messages),
        "assistant_messages": assistant_messages,
        "user_messages": user_messages,
        "tool_batches": tool_batches,
        "tool_responses": tool_responses,
    }


def _expected_execution_provenance(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: derives the server execution provenance expected from a run provenance mapping (validated through
    # ThinkingBoxExecutionProvenance). Raises CoverageError if config or fields are invalid.
    config = provenance.get("config")
    if not isinstance(config, dict):
        raise CoverageError("run provenance has invalid config fingerprints")
    try:
        execution = ThinkingBoxExecutionProvenance(
            thinkingbox_revision=provenance.get("thinkingbox_revision"),
            thinkingbox_source_sha256=provenance.get("thinkingbox_source_sha256"),
            thinkingbox_source_type=provenance.get("thinkingbox_source_type"),
            data_release=provenance.get("data_release"),
            data_revision=provenance.get("data_revision"),
            config_sha256=config.get("sha256"),
            data_bundle_sha256=provenance.get("data_bundle_sha256"),
            manifest_path=provenance.get("manifest_path"),
            manifest_sha256=provenance.get("manifest_sha256"),
            manifest_uids_sha256=provenance.get("manifest_uids_sha256"),
            task_count=provenance.get("task_count"),
        )
    except (TypeError, ValueError) as exc:
        raise CoverageError(
            "run provenance has invalid execution fingerprints"
        ) from exc
    return execution.model_dump(mode="json")


def _require_execution_provenance(
    outcome: _EpisodeOutcome,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: captures the outcome's result into the trace and requires the server-reported execution
    # provenance to have been valid, consistent and equal to the expected one; raises _OperationalResultError
    # otherwise. Returns a copy of the expected provenance.
    if outcome.result is not None:
        outcome.trace.capture_result(outcome.result)
    expected = _expected_execution_provenance(provenance)
    if (
        not outcome.trace.execution_provenance_valid
        or outcome.trace.execution_provenance != expected
    ):
        raise _OperationalResultError(
            "ThinkingBox server execution provenance did not exactly match the run",
            outcome.result,
        )
    return dict(expected)


def _terminal_passed(result: Any, uid: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: validates a terminal client result for the UID and returns whether it passed. Raises
    # _OperationalResultError if there is no result/observation, not done, UID mismatch, kind not terminal,
    # system error, reward type not pass/fail, not graded or graded as system error, inconsistent pass status,
    # or reward not exactly 1.0/0.0.
    if result is None:
        raise _OperationalResultError("ThinkingBox returned no episode result")
    observation = getattr(result, "observation", None)
    if observation is None:
        raise _OperationalResultError(
            "ThinkingBox returned no terminal observation",
            result,
        )
    if not bool(getattr(result, "done", False)):
        raise _OperationalResultError(
            "ThinkingBox episode did not terminate",
            result,
        )
    if getattr(observation, "task_uid", None) != uid:
        raise _OperationalResultError(
            "ThinkingBox terminal UID did not match the requested UID",
            result,
        )
    if getattr(observation, "kind", None) != "terminal":
        raise _OperationalResultError(
            "ThinkingBox returned a non-terminal completion",
            result,
        )
    if bool(getattr(observation, "system_error", False)):
        raise _OperationalResultError(
            "ThinkingBox reported an operational system error",
            result,
        )

    reward_type = getattr(observation, "reward_type", None)
    if reward_type not in {"pass", "fail"}:
        raise _OperationalResultError(
            "ThinkingBox terminal reward type was not pass/fail",
            result,
        )
    summary = getattr(observation, "test_summary", None)
    if not isinstance(summary, dict) or summary.get("graded") is not True:
        raise _OperationalResultError(
            "ThinkingBox terminal result was not graded",
            result,
        )
    if summary.get("is_system_error") is True:
        raise _OperationalResultError(
            "ThinkingBox test result was a system error",
            result,
        )

    passed = reward_type == "pass"
    if summary.get("passed") is not passed:
        raise _OperationalResultError(
            "ThinkingBox pass status was internally inconsistent",
            result,
        )
    reward = getattr(result, "reward", None)
    expected_reward = 1.0 if passed else 0.0
    if not isinstance(reward, (int, float)) or float(reward) != expected_reward:
        raise _OperationalResultError(
            "ThinkingBox reward was not the expected binary value",
            result,
        )

    return passed


def _canonical_decode_result(
    *,
    uid: str,
    repetition: int,
    attempt: int,
    provenance: dict[str, Any],
    test_case: Any,
    outcome: _EpisodeOutcome,
) -> DecodeResult:
    # [FUNCTION DOC]
    # HELPER: converts an episode outcome into a canonical DecodeResult. Re-raises non-BotDesigner exceptions;
    # requires matching execution provenance. BotDesigner activity errors (exception or reported in the
    # result) become a failed agent_error result tagged with benchmark_failure_type; otherwise pass/fail comes
    # from _terminal_passed and the finish reason must be native (_OperationalResultError otherwise). Builds
    # metadata (schema version, repetition and id, attempt, steps taken, step counts, execution and run
    # provenance) and returns the DecodeResult with binary reward, test tags, usage and no private fields.
    if outcome.error is not None and not _is_botdesigner_exception(outcome.error):
        raise outcome.error
    execution_provenance = _require_execution_provenance(outcome, provenance)
    benchmark_failure_type: str | None = None
    if outcome.error is not None:
        passed = False
        finish_reason: FinishReason = "agent_error"
        benchmark_failure_type = _BOTDESIGNER_ACTIVITY_ERROR
    elif outcome.result is not None and _is_botdesigner_result(outcome.result):
        passed = False
        finish_reason = "agent_error"
        benchmark_failure_type = _BOTDESIGNER_ACTIVITY_ERROR
    else:
        passed = _terminal_passed(outcome.result, uid)
        observation = outcome.result.observation
        raw_finish_reason = getattr(observation, "finish_reason", None)
        if raw_finish_reason not in _NATIVE_FINISH_REASONS:
            raise _OperationalResultError(
                "ThinkingBox returned a non-native finish reason",
                outcome.result,
            )
        finish_reason = raw_finish_reason

    messages = _native_messages(outcome)
    steps_taken = outcome.trace.steps_taken
    if outcome.result is not None:
        observation = getattr(outcome.result, "observation", None)
        result_steps = getattr(observation, "steps_taken", None)
        if isinstance(result_steps, int) and result_steps >= 0:
            steps_taken = max(steps_taken, result_steps)
    step_counts = {
        "openenv_steps": steps_taken,
        **_message_step_counts(messages),
    }
    metadata: dict[str, Any] = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "repetition": repetition,
        "repetition_id": _repetition_id(uid, repetition),
        "attempt": attempt,
        "steps_taken": steps_taken,
        "step_counts": step_counts,
        "execution_provenance": execution_provenance,
        "provenance": provenance,
    }
    if benchmark_failure_type is not None:
        metadata["benchmark_failure_type"] = benchmark_failure_type

    reward = 1.0 if passed else 0.0
    return DecodeResult(
        uid=uid,
        messages=messages,
        test_result=TestResult(
            result=passed,
            reward=reward,
            is_system_error=False,
        ),
        test_context=None,
        test_tags=test_case.tags,
        tools=None,
        raw_messages=None,
        user_llm_history=None,
        usage=_native_usage(outcome),
        metadata=metadata,
        is_system_error=False,
        finish_reason=finish_reason,
    )


def _canonical_payload(result: DecodeResult) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: returns the canonical JSON-ready mapping for a DecodeResult (uid, messages, minimal test result,
    # tags, usage, metadata, is_system_error False, finish reason) after asserting no forbidden fields.
    # Raises ValueError without a TestResult, CoverageError on forbidden fields.
    if result.test_result is None:
        raise ValueError("canonical result requires a native TestResult")
    payload = {
        "uid": result.uid,
        "messages": [message.model_dump(mode="json") for message in result.messages],
        "test_result": {
            "result": result.test_result.result,
            "reward": result.test_result.reward,
            "is_system_error": False,
        },
        "test_tags": (
            result.test_tags.model_dump(mode="json")
            if result.test_tags is not None
            else None
        ),
        "usage": [usage.model_dump(mode="json") for usage in (result.usage or [])],
        "metadata": result.metadata,
        "is_system_error": False,
        "finish_reason": result.finish_reason,
    }
    _assert_no_forbidden_fields(payload, "payload")
    return payload


def _encode_canonical(result: DecodeResult) -> str:
    # [FUNCTION DOC]
    # HELPER: encodes the canonical payload as key-sorted JSON, round-trips it through DecodeResult and validates
    # it as a record of its own repetition id and provenance (CoverageError on problems). Returns the JSON text.
    payload = _canonical_payload(result)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    decoded = DecodeResult.model_validate_json(encoded)
    repetition = decoded.metadata.get("repetition")
    repetition_id = decoded.metadata.get("repetition_id")
    if type(repetition) is not int or not isinstance(repetition_id, str):
        raise CoverageError("canonical result omits repetition identity")
    _validate_canonical_record(
        payload,
        decoded,
        expected={repetition_id: (decoded.uid, repetition)},
        expected_provenance=decoded.metadata.get("provenance"),
    )
    return encoded


def _write_encoded_canonical(stream: Any, encoded: str) -> None:
    # [FUNCTION DOC]
    # HELPER: writes one JSONL line and flushes.
    stream.write(encoded + "\n")
    stream.flush()


def _write_canonical(stream: Any, result: DecodeResult) -> None:
    # [FUNCTION DOC]
    # HELPER: encodes and writes a canonical result line.
    _write_encoded_canonical(stream, _encode_canonical(result))


def _assert_no_forbidden_fields(
    value: Any,
    path: str,
    *,
    location: tuple[str | int, ...] = (),
    in_tool_arguments: bool = False,
) -> None:
    # [FUNCTION DOC]
    # HELPER: recursively raises CoverageError if any mapping key (case-insensitive) is a forbidden private
    # field name, except inside messages[i].tool_calls[j].arguments (agent-authored content). Errors include the
    # dotted path.
    if isinstance(value, dict):
        for key, child in value.items():
            key_name = str(key)
            if (
                not in_tool_arguments
                and key_name.casefold() in _FORBIDDEN_CANONICAL_FIELDS
            ):
                raise CoverageError(f"{path} contains forbidden field {key!r}")
            child_location = (*location, key_name)
            # Tool arguments are agent-authored, model-visible content. Their API field
            # names may overlap private evaluator fields, so only the surrounding
            # canonical envelope remains subject to the name denylist.
            child_is_tool_arguments = in_tool_arguments or (
                len(child_location) == 5
                and child_location[0] == "messages"
                and isinstance(child_location[1], int)
                and child_location[2] == "tool_calls"
                and isinstance(child_location[3], int)
                and child_location[4] == "arguments"
            )
            _assert_no_forbidden_fields(
                child,
                f"{path}.{key}",
                location=child_location,
                in_tool_arguments=child_is_tool_arguments,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_fields(
                child,
                f"{path}[{index}]",
                location=(*location, index),
                in_tool_arguments=in_tool_arguments,
            )


def _validate_provenance_shape(
    provenance: dict[str, Any],
    repetition_id: str,
) -> None:
    # [FUNCTION DOC]
    # HELPER: raises CoverageError unless the run provenance has exactly the expected keys, current schema
    # version, valid config provenance (exact keys, string hash and judge type, bool flag, models restricted to
    # safe fields), valid framework and policy provenance, a complete run mapping with sane types and ranges,
    # string identity fields and a positive task count.
    required = {
        "schema_version",
        "thinkingbox_revision",
        "thinkingbox_source_sha256",
        "thinkingbox_source_type",
        "data_release",
        "data_revision",
        "data_bundle_sha256",
        "config",
        "manifest_path",
        "manifest_sha256",
        "manifest_uids_sha256",
        "task_count",
        "requested_test_list_sha256",
        "framework",
        "policy",
        "python",
        "run",
    }
    if set(provenance) != required:
        raise CoverageError(
            f"canonical result {repetition_id!r} has non-canonical provenance shape"
        )
    if provenance.get("schema_version") != _RESULT_SCHEMA_VERSION:
        raise CoverageError(
            f"canonical result {repetition_id!r} has stale provenance schema"
        )
    config = provenance.get("config")
    framework = provenance.get("framework")
    policy = provenance.get("policy")
    run = provenance.get("run")
    required_config = {
        "sha256",
        "agent_model",
        "user_model",
        "judge_model",
        "judge_type",
        "user_can_end_conversation",
    }
    if (
        not isinstance(config, dict)
        or set(config) != required_config
        or not isinstance(config.get("sha256"), str)
        or not isinstance(config.get("judge_type"), str)
        or type(config.get("user_can_end_conversation")) is not bool
        or any(
            model is not None
            and (not isinstance(model, dict) or set(model) - set(_SAFE_MODEL_FIELDS))
            for model in (
                config.get("agent_model"),
                config.get("user_model"),
                config.get("judge_model"),
            )
        )
    ):
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid config provenance"
        )
    if (
        not isinstance(framework, dict)
        or set(framework) != {"revision", "sha256"}
        or not isinstance(framework.get("sha256"), str)
        or (
            framework.get("revision") is not None
            and not isinstance(framework.get("revision"), str)
        )
    ):
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid framework provenance"
        )
    if policy is not None and (
        not isinstance(policy, dict)
        or set(policy) != {"spec", "sha256"}
        or not isinstance(policy.get("spec"), str)
        or not isinstance(policy.get("sha256"), str)
    ):
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid policy provenance"
        )
    if not isinstance(run, dict):
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid run provenance"
        )
    required_run = {
        "agent",
        "repeat",
        "repetition_start",
        "shard_index",
        "shard_count",
        "limit",
        "message_timeout_s",
        "base_url_sha256",
        "selected_uids_sha256",
    }
    if required_run != set(run):
        raise CoverageError(
            f"canonical result {repetition_id!r} has incomplete run provenance"
        )
    if (
        not isinstance(run["agent"], str)
        or type(run["repeat"]) is not int
        or run["repeat"] < 1
        or type(run["repetition_start"]) is not int
        or run["repetition_start"] < 0
        or type(run["shard_index"]) is not int
        or type(run["shard_count"]) is not int
        or not isinstance(run["message_timeout_s"], (int, float))
        or isinstance(run["message_timeout_s"], bool)
        or run["message_timeout_s"] <= 0
        or not isinstance(run["base_url_sha256"], str)
        or not isinstance(run["selected_uids_sha256"], str)
        or (
            run["limit"] is not None
            and (type(run["limit"]) is not int or run["limit"] < 1)
        )
    ):
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid run parameters"
        )
    for key in (
        "thinkingbox_revision",
        "thinkingbox_source_sha256",
        "thinkingbox_source_type",
        "data_release",
        "data_revision",
        "data_bundle_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_uids_sha256",
        "requested_test_list_sha256",
        "python",
    ):
        if not isinstance(provenance.get(key), str):
            raise CoverageError(f"canonical result {repetition_id!r} has invalid {key}")
    if type(provenance.get("task_count")) is not int or provenance["task_count"] < 1:
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid task_count"
        )


def _validate_canonical_record(
    raw: dict[str, Any],
    result: DecodeResult,
    *,
    expected: dict[str, tuple[str, int]],
    expected_provenance: dict[str, Any] | None,
) -> str:
    # [FUNCTION DOC]
    # HELPER: strict validation of one canonical record (raw mapping plus parsed DecodeResult): no forbidden
    # fields, exact top-level fields, non-system-error, native finish reason, minimal and consistent test
    # result, tags and usage present, exact metadata shape, valid benchmark failure metadata, repetition id
    # expected and matching UID/index, schema, attempt, steps and step counts consistent with messages, valid
    # provenance shape, execution provenance consistent with run provenance, equality with expected provenance
    # when given, and raw payload identical to the re-derived canonical payload. Returns the repetition id;
    # raises CoverageError otherwise.
    _assert_no_forbidden_fields(raw, "payload")
    required_fields = {
        "uid",
        "messages",
        "test_result",
        "test_tags",
        "usage",
        "metadata",
        "is_system_error",
        "finish_reason",
    }
    missing_fields = required_fields - set(raw)
    if missing_fields:
        raise CoverageError(
            f"canonical result is missing fields: {sorted(missing_fields)}"
        )
    unexpected_fields = set(raw) - required_fields
    if unexpected_fields:
        raise CoverageError(
            f"canonical result {result.uid!r} contains unexpected fields: "
            f"{sorted(unexpected_fields)}"
        )
    if result.is_system_error:
        raise CoverageError(f"canonical result {result.uid!r} is a system error")
    if raw.get("is_system_error") is not False:
        raise CoverageError(
            f"canonical result {result.uid!r} omits explicit valid status"
        )
    if raw.get("finish_reason") not in _NATIVE_FINISH_REASONS:
        raise CoverageError(
            f"canonical result {result.uid!r} has a non-native finish reason"
        )
    if result.test_result is None or result.test_result.is_system_error:
        raise CoverageError(
            f"canonical result {result.uid!r} has no valid pass/fail TestResult"
        )
    test_result_raw = raw.get("test_result")
    if not isinstance(test_result_raw, dict) or set(test_result_raw) - {
        "result",
        "reward",
        "is_system_error",
    }:
        raise CoverageError(
            f"canonical result {result.uid!r} contains non-minimal test details"
        )
    if (
        type(test_result_raw.get("result")) is not bool
        or test_result_raw.get("is_system_error") is not False
        or type(test_result_raw.get("reward")) not in {int, float}
    ):
        raise CoverageError(
            f"canonical result {result.uid!r} has invalid TestResult types"
        )
    expected_reward = 1.0 if result.test_result.result else 0.0
    if result.test_result.reward != expected_reward:
        raise CoverageError(f"canonical result {result.uid!r} has inconsistent reward")
    if result.test_tags is None:
        raise CoverageError(f"canonical result {result.uid!r} omits test_tags")
    if not isinstance(raw.get("test_tags"), dict):
        raise CoverageError(f"canonical result {result.uid!r} has invalid test_tags")
    if result.usage is None:
        raise CoverageError(f"canonical result {result.uid!r} omits usage")
    if not isinstance(raw.get("usage"), list):
        raise CoverageError(f"canonical result {result.uid!r} has invalid usage")

    metadata = result.metadata
    if not isinstance(metadata, dict):
        raise CoverageError(f"canonical result {result.uid!r} omits metadata")
    required_metadata = {
        "schema_version",
        "repetition",
        "repetition_id",
        "attempt",
        "steps_taken",
        "step_counts",
        "execution_provenance",
        "provenance",
    }
    unexpected_metadata = set(metadata) - (
        required_metadata | {"benchmark_failure_type"}
    )
    if required_metadata - set(metadata) or unexpected_metadata:
        raise CoverageError(
            f"canonical result {result.uid!r} has non-canonical metadata shape"
        )
    benchmark_failure_type = metadata.get("benchmark_failure_type")
    if benchmark_failure_type is not None and (
        benchmark_failure_type != _BOTDESIGNER_ACTIVITY_ERROR
        or result.test_result.result
        or result.finish_reason != "agent_error"
    ):
        raise CoverageError(
            f"canonical result {result.uid!r} has invalid benchmark failure metadata"
        )
    repetition = metadata.get("repetition")
    repetition_id = metadata.get("repetition_id")
    if type(repetition) is not int or not isinstance(repetition_id, str):
        raise CoverageError(
            f"canonical result {result.uid!r} omits repetition metadata"
        )
    pair = expected.get(repetition_id)
    if pair is None:
        raise CoverageError(f"unexpected repetition ID {repetition_id!r}")
    if pair != (result.uid, repetition):
        raise CoverageError(f"repetition ID {repetition_id!r} does not match UID/index")
    if metadata.get("schema_version") != _RESULT_SCHEMA_VERSION:
        raise CoverageError(
            f"canonical result {repetition_id!r} has an unsupported schema"
        )
    attempt = metadata.get("attempt")
    if type(attempt) is not int or attempt < 1:
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid attempt metadata"
        )
    if type(metadata.get("steps_taken")) is not int:
        raise CoverageError(f"canonical result {repetition_id!r} omits steps_taken")
    if metadata["steps_taken"] < 0:
        raise CoverageError(
            f"canonical result {repetition_id!r} has negative steps_taken"
        )
    step_counts = metadata.get("step_counts")
    if not isinstance(step_counts, dict):
        raise CoverageError(f"canonical result {repetition_id!r} omits step_counts")
    expected_step_keys = {
        "openenv_steps",
        "native_messages",
        "assistant_messages",
        "user_messages",
        "tool_batches",
        "tool_responses",
    }
    if (
        set(step_counts) != expected_step_keys
        or any(type(value) is not int or value < 0 for value in step_counts.values())
        or step_counts["openenv_steps"] != metadata["steps_taken"]
        or {
            key: step_counts[key]
            for key in expected_step_keys
            if key != "openenv_steps"
        }
        != _message_step_counts(result.messages)
    ):
        raise CoverageError(
            f"canonical result {repetition_id!r} has invalid step_counts"
        )
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise CoverageError(f"canonical result {repetition_id!r} omits provenance")
    _validate_provenance_shape(provenance, repetition_id)
    execution_provenance = metadata.get("execution_provenance")
    if not isinstance(
        execution_provenance, dict
    ) or execution_provenance != _expected_execution_provenance(provenance):
        raise CoverageError(
            f"canonical result {repetition_id!r} has stale/mixed provenance: "
            "server execution fingerprint mismatch"
        )
    if expected_provenance is not None and provenance != expected_provenance:
        raise CoverageError(
            f"canonical result {repetition_id!r} has stale/mixed provenance"
        )
    if raw != _canonical_payload(result):
        raise CoverageError(
            f"canonical result {repetition_id!r} has non-canonical raw shape"
        )
    return repetition_id


def _read_canonical_results(
    paths: Path | Sequence[Path],
    *,
    expected: dict[str, tuple[str, int]],
    expected_provenance: dict[str, Any] | None,
) -> list[DecodeResult]:
    # [FUNCTION DOC]
    # HELPER: reads one or more JSONL files, parsing and strictly validating each line (CoverageError for
    # missing files, blank or malformed lines, duplicates across files). Returns the list of DecodeResults.
    source_paths = [paths] if isinstance(paths, Path) else list(paths)
    results: list[DecodeResult] = []
    seen: set[str] = set()
    for path in source_paths:
        if not path.is_file():
            raise CoverageError(f"canonical results file does not exist: {path}")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise CoverageError(f"{path}:{line_number}: blank JSONL record")
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError("record is not a JSON object")
                except Exception as exc:
                    raise CoverageError(
                        f"{path}:{line_number}: malformed canonical DecodeResult"
                    ) from exc
                _assert_no_forbidden_fields(raw, "payload")
                try:
                    result = DecodeResult.model_validate(raw)
                except Exception as exc:
                    raise CoverageError(
                        f"{path}:{line_number}: malformed canonical DecodeResult"
                    ) from exc
                repetition_id = _validate_canonical_record(
                    raw,
                    result,
                    expected=expected,
                    expected_provenance=expected_provenance,
                )
                if repetition_id in seen:
                    raise CoverageError(f"duplicate repetition ID {repetition_id!r}")
                seen.add(repetition_id)
                results.append(result)
    return results


def _require_terminal_newline(path: Path, record_type: str) -> None:
    # [FUNCTION DOC]
    # HELPER: raises CoverageError if a non-empty file does not end with a newline (a truncated last record).
    if path.stat().st_size == 0:
        return
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        if stream.read(1) != b"\n":
            raise CoverageError(
                f"{path}: {record_type} JSONL does not end with a terminal newline"
            )


def _load_resume_state(
    output: Path,
    *,
    expected: dict[str, tuple[str, int]],
    provenance: dict[str, Any],
) -> set[str]:
    # [FUNCTION DOC]
    # HELPER: returns the set of repetition ids already validly present in the output file (empty if it does not
    # exist); raises CoverageError on invalid content or missing trailing newline.
    if not output.exists():
        return set()
    results = _read_canonical_results(
        output,
        expected=expected,
        expected_provenance=provenance,
    )
    _require_terminal_newline(output, "canonical")
    return {str(result.metadata["repetition_id"]) for result in results}


def _error_attempts(
    errors_output: Path,
    *,
    expected: dict[str, tuple[str, int]],
    provenance: dict[str, Any],
) -> dict[str, int]:
    # [FUNCTION DOC]
    # HELPER: reads the errors sidecar and returns the max attempt number per repetition id (empty if absent).
    # Raises CoverageError for blank/malformed/non-object lines, unexpected repetitions, mismatched identity or
    # classification, invalid attempt, stale provenance, duplicate attempts or missing trailing newline.
    attempts: dict[str, int] = defaultdict(int)
    seen_attempts: set[tuple[str, int]] = set()
    if not errors_output.exists():
        return attempts
    with errors_output.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise CoverageError(
                    f"{errors_output}:{line_number}: blank error JSONL record"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CoverageError(
                    f"{errors_output}:{line_number}: malformed error record"
                ) from exc
            if not isinstance(record, dict):
                raise CoverageError(
                    f"{errors_output}:{line_number}: error record is not an object"
                )
            repetition_id = record.get("repetition_id")
            attempt = record.get("attempt")
            if repetition_id not in expected:
                raise CoverageError(
                    f"{errors_output}:{line_number}: unexpected error repetition"
                )
            if (
                expected[repetition_id] != (record.get("uid"), record.get("repetition"))
                or record.get("classification") != "operational_system_error"
                or record.get("retryable") is not True
            ):
                raise CoverageError(
                    f"{errors_output}:{line_number}: invalid error audit identity"
                )
            if not isinstance(attempt, int) or attempt < 1:
                raise CoverageError(
                    f"{errors_output}:{line_number}: invalid attempt count"
                )
            if record.get("provenance") != provenance:
                raise CoverageError(
                    f"{errors_output}:{line_number}: stale/mixed error provenance"
                )
            key = (repetition_id, attempt)
            if key in seen_attempts:
                raise CoverageError(
                    f"{errors_output}:{line_number}: duplicate error attempt"
                )
            seen_attempts.add(key)
            attempts[repetition_id] = max(attempts[repetition_id], attempt)
    _require_terminal_newline(errors_output, "error")
    return attempts


def _canonical_release_identity() -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: requires the pinned ThinkingBox runtime and the canonical data (UID order equal to
    # CANONICAL_TEST_UIDS and pinned release/commit/bundle/manifest path); raises CoverageError otherwise.
    # Returns the identity mapping (runtime and data fingerprints, task count). May download data.
    try:
        runtime = require_canonical_thinkingbox_runtime()
    except RuntimeError as exc:
        raise CoverageError(
            "strict coverage requires the pinned ThinkingBox VCS runtime"
        ) from exc
    bundle = resolve_data_bundle()
    manifest_uids = load_test_uids(bundle)
    if tuple(manifest_uids) != CANONICAL_TEST_UIDS:
        raise CoverageError(
            "strict coverage requires the authoritative benchmark UID order"
        )
    manifest_path = bundle.manifest_path.relative_to(bundle.root).as_posix()
    identity = {
        "thinkingbox_revision": runtime.identity,
        "thinkingbox_source_sha256": runtime.source_sha256,
        "thinkingbox_source_type": runtime.source_type,
        "data_release": bundle.release_name,
        "data_revision": bundle.revision,
        "data_bundle_sha256": bundle.bundle_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": bundle.manifest_sha256,
        "manifest_uids_sha256": bundle.manifest_uids_sha256,
        "task_count": len(manifest_uids),
    }
    expected = {
        "data_release": DATA_RELEASE_NAME,
        "data_revision": DATA_COMMIT,
        "data_bundle_sha256": DATA_BUNDLE_SHA256,
        "manifest_path": DATA_MANIFEST_PATH,
    }
    if {key: identity[key] for key in expected} != expected:
        raise CoverageError(
            "strict coverage requires the canonical executable data release"
        )
    return identity


def validate_coverage(
    paths: Path | Sequence[Path],
    manifest_uids: Sequence[str],
    repetitions: Sequence[int],
    *,
    expected_provenance: dict[str, Any] | None = None,
    expected_base_provenance: dict[str, Any] | None = None,
    profile: str | None = None,
) -> CoverageReport:
    # [FUNCTION DOC]
    # Validates exact UID x repetition coverage of canonical result files. With a profile ("canary" requires
    # repetitions (0,), "full" requires 0..19) also requires the canonical release and manifest and canonical
    # content in expected provenance. Reads and validates all records. When no exact per-run provenance is
    # given, additionally checks each record's base provenance (if expected), shard assignment, limit, repeat
    # and repetition start against the plan, rejects limited shards under a profile, and requires all records
    # to share one common provenance and shard count. Raises CoverageError for missing or extra results.
    # Returns a CoverageReport.
    """Validate exact UID-by-repetition coverage before native aggregation.

    Args:
        paths (`pathlib.Path` or `collections.abc.Sequence[pathlib.Path]`):
            Canonical result JSONL file or files.
        manifest_uids (`collections.abc.Sequence[str]`):
            Ordered manifest UIDs that results must cover.
        repetitions (`collections.abc.Sequence[int]`):
            Exact repetition indexes required for every UID.
        expected_provenance (`dict`, *optional*):
            Exact per-run provenance expected in every record.
        expected_base_provenance (`dict`, *optional*):
            Exact provenance excluding shard-specific run fields.
        profile (`str`, *optional*):
            Strict `canary` or `full` canonical coverage profile.

    Returns:
        [`CoverageReport`]:
            Validated records and exact coverage dimensions.

    Raises:
        [`CoverageError`]:
            If identity, provenance, coverage, or canonical formatting differs.
    """
    required_repetitions = tuple(repetitions)
    canonical_content: dict[str, Any] | None = None
    if profile == "canary":
        if required_repetitions != (0,):
            raise CoverageError("canary coverage requires exactly 1 repetition")
    elif profile == "full":
        if required_repetitions != tuple(range(20)):
            raise CoverageError("full coverage requires exactly 20 repetitions")
    elif profile is not None:
        raise CoverageError(f"unknown coverage profile {profile!r}")
    if profile is not None:
        canonical_content = _canonical_release_identity()
        if (
            len(manifest_uids) != canonical_content["task_count"]
            or _sha256_json(list(manifest_uids))
            != canonical_content["manifest_uids_sha256"]
        ):
            raise CoverageError(
                f"{profile} coverage requires the authoritative benchmark manifest"
            )
    if profile is not None and expected_provenance is not None:
        actual_content = {
            key: expected_provenance.get(key) for key in canonical_content or {}
        }
        if actual_content != canonical_content:
            raise CoverageError(
                f"{profile} coverage contains noncanonical benchmark content"
            )

    expected = _expected_repetition_ids(manifest_uids, required_repetitions)
    results = _read_canonical_results(
        paths,
        expected=expected,
        expected_provenance=expected_provenance,
    )
    if expected_provenance is None:
        common_provenances: set[str] = set()
        shard_counts: set[int] = set()
        expected_start = required_repetitions[0] if required_repetitions else 0
        uid_indexes = {uid: index for index, uid in enumerate(manifest_uids)}
        for result in results:
            provenance = result.metadata["provenance"]
            if (
                canonical_content is not None
                and {key: provenance.get(key) for key in canonical_content}
                != canonical_content
            ):
                raise CoverageError(
                    f"{profile} coverage contains noncanonical benchmark content"
                )
            if (
                expected_base_provenance is not None
                and {key: value for key, value in provenance.items() if key != "run"}
                != expected_base_provenance
            ):
                raise CoverageError(
                    "canonical coverage has stale config/data/runtime/framework "
                    "provenance"
                )
            run = provenance["run"]
            common_run = {
                key: value
                for key, value in run.items()
                if key not in {"shard_index", "selected_uids_sha256"}
            }
            common_provenance = {
                **{key: value for key, value in provenance.items() if key != "run"},
                "run": common_run,
            }
            common_provenances.add(_sha256_json(common_provenance))

            shard_count = run.get("shard_count")
            shard_index = run.get("shard_index")
            if (
                not isinstance(shard_count, int)
                or shard_count < 1
                or not isinstance(shard_index, int)
                or not 0 <= shard_index < shard_count
            ):
                raise CoverageError("canonical coverage has invalid shard parameters")
            shard_counts.add(shard_count)
            shard_uids = [
                uid
                for index, uid in enumerate(manifest_uids)
                if index % shard_count == shard_index
            ]
            limit = run.get("limit")
            if limit is not None:
                if not isinstance(limit, int) or limit < 1:
                    raise CoverageError("canonical coverage has an invalid run limit")
                shard_uids = shard_uids[:limit]
            if (
                run.get("repeat") != len(required_repetitions)
                or run.get("repetition_start") != expected_start
                or run.get("selected_uids_sha256") != _sha256_json(shard_uids)
            ):
                raise CoverageError(
                    "canonical coverage run parameters do not match the "
                    "required UID/repetition plan"
                )
            if (
                result.uid not in uid_indexes
                or uid_indexes[result.uid] % shard_count != shard_index
                or result.uid not in shard_uids
            ):
                raise CoverageError(
                    f"canonical result {result.uid!r} is assigned to a stale shard"
                )
            if profile is not None and limit is not None:
                raise CoverageError(f"{profile} coverage contains a limited shard")
        if len(common_provenances) > 1 or len(shard_counts) > 1:
            raise CoverageError(
                "canonical coverage contains mixed config/data/runtime/PR or "
                "run parameters"
            )
    found = {str(result.metadata["repetition_id"]) for result in results}
    missing = sorted(set(expected) - found)
    if missing:
        preview = ", ".join(missing[:3])
        raise CoverageError(
            f"canonical coverage is incomplete: {len(missing)} missing"
            + (f" ({preview})" if preview else "")
        )
    if len(results) != len(expected):
        raise CoverageError("canonical coverage has unexpected extra results")
    return CoverageReport(
        results=tuple(results),
        uid_count=len(manifest_uids),
        repetitions=required_repetitions,
        total_results=len(results),
    )


def _native_aggregation(report: CoverageReport) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: runs ThinkingBox's native per-test and overall aggregation on the report's results. Returns a
    # JSON-ready mapping with "per_test" table rows and "metrics".
    from pydantic_core import to_jsonable_python
    from thinkingbox.cli.agg_main import (
        aggregate_results,
        aggregate_results_per_test,
        make_per_test_table,
    )

    per_test = aggregate_results_per_test(report.results)
    table = make_per_test_table(per_test)
    metrics = aggregate_results(per_test)
    return {
        "per_test": [to_jsonable_python(row) for row in table],
        "metrics": to_jsonable_python(metrics),
    }


def _exception_details(exc: Exception) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: returns redacted exception details: type name, cause chain type names, with messages and
    # traceback replaced by "details redacted".
    chain = [
        {
            "type": type(current).__name__,
            "message": "details redacted",
        }
        for current in _iter_exception_chain(exc)
    ]
    return {
        "type": type(exc).__name__,
        "message": "details redacted",
        "chain": chain,
        "traceback": "details redacted",
    }


def _operational_error_record(
    *,
    uid: str,
    repetition: int,
    attempt: int,
    provenance: dict[str, Any],
    exc: Exception,
    outcome: _EpisodeOutcome | None,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: builds a redacted errors-sidecar record: schema version, uid, repetition and id, attempt, UTC
    # timestamp, classification operational_system_error (retryable, quarantined), provenance, redacted error
    # details, an audit of the terminal observation (from the exception's attached result or the outcome) and a
    # trace summary.
    result = (
        exc.result
        if isinstance(exc, _OperationalResultError) and exc.result is not None
        else (outcome.result if outcome is not None else None)
    )
    observation = getattr(result, "observation", None)
    observation_audit: dict[str, Any] | None = None
    if observation is not None:
        observation_audit = {
            "kind": getattr(observation, "kind", None),
            "task_uid": getattr(observation, "task_uid", None),
            "finish_reason": getattr(observation, "finish_reason", None),
            "reward_type": getattr(observation, "reward_type", None),
            "system_error": getattr(observation, "system_error", None),
            "error": getattr(observation, "error", None),
            "test_summary": getattr(observation, "test_summary", None),
            "steps_taken": getattr(observation, "steps_taken", None),
            "metadata": getattr(observation, "metadata", None),
        }
    return {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "uid": uid,
        "repetition": repetition,
        "repetition_id": _repetition_id(uid, repetition),
        "attempt": attempt,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "classification": "operational_system_error",
        "retryable": True,
        "quarantined": True,
        "provenance": provenance,
        "error": _exception_details(exc),
        "observation_audit": observation_audit,
        "trace_audit": (
            {
                "steps_taken": outcome.trace.steps_taken,
                "native_message_count": len(outcome.trace.messages),
                "usage_entries": len(outcome.trace.usage),
            }
            if outcome is not None
            else None
        ),
    }


def _write_error(stream: Any, record: dict[str, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: writes a key-sorted JSON line to the error stream and flushes.
    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _default_errors_output(output: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: returns "errors.jsonl" beside a "results.jsonl" output, else "<stem>.errors.jsonl" beside it.
    if output.name == "results.jsonl":
        return output.with_name("errors.jsonl")
    return output.with_name(f"{output.stem}.errors.jsonl")


def _write_aggregation(path: str, aggregation: dict[str, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: writes indented key-sorted JSON to stdout when path is "-", else to the file (creating parent dirs).
    encoded = json.dumps(aggregation, indent=2, sort_keys=True) + "\n"
    if path == "-":
        sys.stdout.write(encoded)
        return
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC]
    # Returns the argparse parser: optional positional test list; --config (required), --env-config,
    # --base-url, --message-timeout, --dataset, --env-dataset, --agent (default think), --policy, --output,
    # --coverage-input (repeatable), --errors-output, --repeat, --repetition-start, --attempts-per-repetition,
    # --shard-index, --shard-count, --limit, --resume, --coverage-profile (canary/full), --aggregate-output,
    # --validate-only.
    """Build the command-line parser for execution and coverage validation.

    Returns:
        `argparse.ArgumentParser`:
            Configured evaluator parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("test_list", nargs="?")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--env-config",
        help="Server-visible config path; defaults to --config.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--message-timeout",
        type=float,
        default=DEFAULT_MESSAGE_TIMEOUT_S,
        help="Per-operation WebSocket timeout in seconds.",
    )
    parser.add_argument("--dataset")
    parser.add_argument(
        "--env-dataset",
        help="Server-visible data path; defaults to --dataset.",
    )
    parser.add_argument("--agent", default="think")
    parser.add_argument(
        "--policy",
        help=(
            "Custom MODULE:CALLABLE invoked as "
            "(env, reset_result, repetition_id); trusted config and hydrated "
            "test-case data are never provided."
        ),
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--coverage-input",
        action="append",
        default=[],
        help="Existing canonical shard JSONL to validate/aggregate; repeat as needed.",
    )
    parser.add_argument(
        "--errors-output",
        help="Redacted operational sidecar; defaults beside --output.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repetition-start", type=int, default=0)
    parser.add_argument(
        "--attempts-per-repetition",
        type=int,
        default=1,
        help="Retry operational errors this many times in this invocation.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--coverage-profile",
        choices=("canary", "full"),
        help=(
            "Require the canonical manifest at 1 repetition (canary) or "
            "20 repetitions (full)."
        ),
    )
    parser.add_argument(
        "--aggregate-output",
        help="After strict coverage validation, write native aggregation JSON; '-' for stdout.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing canonical output without running episodes.",
    )
    return parser


async def _run_episode(
    args: argparse.Namespace,
    uid: str,
    repetition_id: str,
    *,
    test_case: Any,
    agent_session_factory: Any,
    policy: Any,
    env_config: str,
    env_dataset: str | None,
) -> _EpisodeOutcome:
    # [FUNCTION DOC]
    # HELPER: runs one episode in its own client connection: resets with the UID and server-visible dataset,
    # agent and config; returns immediately if reset was terminal; otherwise runs the configured native agent
    # or the custom policy (RuntimeError if the policy returns None). Never raises: any exception is captured in
    # the returned _EpisodeOutcome together with the last known result and trace.
    trace = _ExecutionTrace()
    episode_result: Any | None = None
    try:
        async with ThinkingBoxEnv(
            base_url=args.base_url,
            message_timeout_s=args.message_timeout,
        ) as env:
            reset_result = await env.reset(
                uid,
                dataset=env_dataset,
                agent=args.agent,
                config=env_config,
            )
            trace.capture_result(reset_result)
            if reset_result.done:
                episode_result = reset_result
                return _EpisodeOutcome(
                    result=reset_result,
                    error=None,
                    trace=trace,
                )
            if policy is None:
                result = await run_configured_agent(
                    env,
                    reset_result,
                    test_case,
                    agent_session_factory,
                    trace,
                )
            else:
                result = await _invoke_policy(
                    policy,
                    env,
                    reset_result,
                    repetition_id,
                )
                if result is None:
                    raise RuntimeError("configured policy returned no episode result")
            episode_result = result
            trace.capture_result(result)
            return _EpisodeOutcome(result=result, error=None, trace=trace)
    except Exception as exc:
        return _EpisodeOutcome(result=episode_result, error=exc, trace=trace)


async def _run(args: argparse.Namespace) -> None:
    # [FUNCTION DOC]
    # Main async flow. Validates CLI arguments (ValueError on invalid combinations), loads the config and agent
    # factory, optional policy, data bundle and canonical UIDs (checking canonical order for the pinned release),
    # the requested UIDs, then applies sharding and limit (not in coverage-input mode). Builds provenance and the
    # expected repetition plan; enforces canonical identity and unsharded full manifests for coverage profiles.
    # Coverage-input mode: validates the given shard files and optionally aggregates. Otherwise resolves the
    # output and errors files (must differ). Validate-only mode: checks the errors sidecar and the output, then
    # optionally aggregates. Run mode: optionally resumes (skipping completed repetitions and continuing attempt
    # numbers), and for each UID x repetition tries up to attempts-per-repetition times: hydrates the task
    # locally, runs the episode, builds and encodes the canonical record, appending it on success or writing a
    # redacted error record on any exception. Raises RuntimeError if any repetition stays unresolved; otherwise
    # validates full coverage of the output and optionally writes the aggregation.
    # Side effects: network episodes, file writes (append on resume, truncate otherwise).
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.repetition_start < 0:
        raise ValueError("--repetition-start must be non-negative")
    if args.attempts_per_repetition < 1:
        raise ValueError("--attempts-per-repetition must be at least 1")
    if args.message_timeout <= 0:
        raise ValueError("--message-timeout must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.aggregate_output and args.coverage_profile is None:
        raise ValueError("--aggregate-output requires --coverage-profile")
    if not args.coverage_input and args.output is None:
        raise ValueError("--output is required when running or validating one shard")
    if args.coverage_input and args.validate_only:
        raise ValueError("--coverage-input already selects validation-only mode")

    config_path = Path(args.config).expanduser().resolve()
    native_config, config_sha256 = load_thinkingbox_config_with_sha256(config_path)
    agent_session_factory = configured_agent_session_factory(native_config)
    policy = _load_policy(args.policy) if args.policy else None

    bundle = resolve_data_bundle(args.dataset)
    canonical = load_test_uids(bundle)
    if (
        bundle.release_name == DATA_RELEASE_NAME
        and bundle.revision == DATA_COMMIT
        and tuple(canonical) != CANONICAL_TEST_UIDS
    ):
        raise ValueError("resolved canonical manifest UID order is invalid")
    selected = _load_requested_uids(args.test_list, canonical)
    if args.coverage_input:
        if args.test_list is not None:
            raise ValueError("--coverage-input requires the canonical manifest")
    else:
        selected = [
            uid
            for index, uid in enumerate(selected)
            if index % args.shard_count == args.shard_index
        ]
        if args.limit is not None:
            selected = selected[: args.limit]

    base_provenance = _base_provenance(
        config=native_config,
        config_path=config_path,
        config_sha256=config_sha256,
        bundle=bundle,
        canonical_uids=canonical,
        requested_test_list=args.test_list,
        policy_spec=args.policy,
        policy=policy,
    )
    provenance = _run_provenance(base_provenance, args, selected)
    repetitions = tuple(
        range(
            args.repetition_start,
            args.repetition_start + args.repeat,
        )
    )
    expected = _expected_repetition_ids(selected, repetitions)

    if args.coverage_profile is not None:
        expected_repeat = 1 if args.coverage_profile == "canary" else 20
        canonical_identity = _canonical_release_identity()
        if {
            key: base_provenance.get(key) for key in canonical_identity
        } != canonical_identity:
            raise ValueError(
                f"{args.coverage_profile} coverage requires canonical "
                "runtime, data, and manifest provenance"
            )
        if (
            args.test_list is not None
            or args.shard_count != 1
            or args.shard_index != 0
            or args.limit is not None
            or args.repetition_start != 0
            or args.repeat != expected_repeat
            or selected != canonical
        ):
            raise ValueError(
                f"{args.coverage_profile} coverage requires the unsharded "
                f"canonical {len(canonical)}x{expected_repeat} manifest"
            )

    if args.coverage_input:
        report = validate_coverage(
            [Path(path).expanduser() for path in args.coverage_input],
            selected,
            repetitions,
            expected_base_provenance=base_provenance,
            profile=args.coverage_profile,
        )
        if args.aggregate_output:
            _write_aggregation(
                args.aggregate_output,
                _native_aggregation(report),
            )
        return

    output = Path(args.output).expanduser()
    errors_output = (
        Path(args.errors_output).expanduser()
        if args.errors_output
        else _default_errors_output(output)
    )
    if output.resolve() == errors_output.resolve():
        raise ValueError("--output and --errors-output must be different files")
    output.parent.mkdir(parents=True, exist_ok=True)
    errors_output.parent.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        _error_attempts(
            errors_output,
            expected=expected,
            provenance=provenance,
        )
        report = validate_coverage(
            output,
            selected,
            repetitions,
            expected_provenance=provenance,
            profile=args.coverage_profile,
        )
        if args.aggregate_output:
            _write_aggregation(
                args.aggregate_output,
                _native_aggregation(report),
            )
        return

    if args.resume:
        completed = _load_resume_state(
            output,
            expected=expected,
            provenance=provenance,
        )
        attempts = _error_attempts(
            errors_output,
            expected=expected,
            provenance=provenance,
        )
    else:
        completed = set()
        attempts = defaultdict(int)

    mode = "a" if args.resume else "w"
    env_config = args.env_config or str(config_path)
    env_dataset = args.env_dataset or args.dataset
    unresolved: list[str] = []

    with (
        output.open(mode, encoding="utf-8") as result_stream,
        errors_output.open(mode, encoding="utf-8") as error_stream,
    ):
        for uid in selected:
            for repetition in repetitions:
                repetition_id = _repetition_id(uid, repetition)
                if repetition_id in completed:
                    continue
                canonical_result: DecodeResult | None = None
                for _ in range(args.attempts_per_repetition):
                    attempts[repetition_id] += 1
                    attempt = attempts[repetition_id]
                    outcome: _EpisodeOutcome | None = None
                    try:
                        test_case = get_dataset_case_by_name(
                            uid,
                            base_dir=bundle.dataset_dir,
                            agent=args.agent,
                        )
                        outcome = await _run_episode(
                            args,
                            uid,
                            repetition_id,
                            test_case=test_case,
                            agent_session_factory=agent_session_factory,
                            policy=policy,
                            env_config=env_config,
                            env_dataset=env_dataset,
                        )
                        candidate_result = _canonical_decode_result(
                            uid=uid,
                            repetition=repetition,
                            attempt=attempt,
                            provenance=provenance,
                            test_case=test_case,
                            outcome=outcome,
                        )
                        encoded_result = _encode_canonical(candidate_result)
                    except Exception as exc:
                        canonical_result = None
                        _write_error(
                            error_stream,
                            _operational_error_record(
                                uid=uid,
                                repetition=repetition,
                                attempt=attempt,
                                provenance=provenance,
                                exc=exc,
                                outcome=outcome,
                            ),
                        )
                        continue
                    _write_encoded_canonical(result_stream, encoded_result)
                    canonical_result = candidate_result
                    completed.add(repetition_id)
                    break
                if canonical_result is None:
                    unresolved.append(repetition_id)

    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} repetitions remain quarantined and retryable; "
            f"see {errors_output}"
        )

    report = validate_coverage(
        output,
        selected,
        repetitions,
        expected_provenance=provenance,
        profile=args.coverage_profile,
    )
    if args.aggregate_output:
        _write_aggregation(
            args.aggregate_output,
            _native_aggregation(report),
        )


def main() -> None:
    # [FUNCTION DOC]
    # Parses command-line arguments and runs the async evaluator with asyncio.run.
    """Parse command-line arguments and execute the evaluator."""
    asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
