# [MODULE DOC]
# Small test fakes and builders for runtime validation tests: helpers that build
# WireExchange / RuntimeEvidence values, plus an inert running subject and
# runtime provider that record calls without touching Docker or the network.
"""Small reusable fakes; real protocol and Docker tests establish execution evidence."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from openenv.validation.manifest import ExecutionDeclaration
from openenv.validation.providers import ExecResult
from openenv.validation.runtime.contracts import (
    LaunchSpec,
    RuntimeEvidence,
    WireExchange,
)
from openenv.validation.types import ProviderCapability


def exchange(operation, request, response):
    # [FUNCTION DOC]
    # HELPER: Builds a WireExchange for an operation by JSON-encoding the request
    # and response values verbatim (no coercion).
    # Returns: a WireExchange(operation, request_json, response_json).
    # Raises: TypeError if a value is not JSON-serializable.
    """Build immutable evidence without hiding malformed wire values via coercion."""
    return WireExchange(operation, json.dumps(request), json.dumps(response))


def evidence(*exchanges, observation_schema=None):
    # [FUNCTION DOC]
    # HELPER: Builds a RuntimeEvidence from the given exchanges (as a tuple) and an
    # optional observation schema, JSON-encoded when provided (None otherwise).
    # Returns: a RuntimeEvidence.
    """Build one collector result from exact test exchanges and schema data."""
    schema = None if observation_schema is None else json.dumps(observation_schema)
    return RuntimeEvidence(tuple(exchanges), observation_schema_json=schema)


@dataclass
class FakeRunningSubject:
    # [CLASS DOC]
    # Dataclass standing in for a running subject: has a base_url, a stopped flag
    # and a list of executed command argv lists.
    """Inert subject recording teardown and exec, with no network or Docker calls."""

    base_url: str = "http://127.0.0.1:1"
    stopped: bool = False
    commands: list[list[str]] = field(default_factory=list)

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult:
        # [FUNCTION DOC]
        # Records a copy of argv in commands and returns a successful empty ExecResult
        # (exit 0, empty stdout/stderr, 0.0 duration). timeout_s is ignored.
        self.commands.append(list(argv))
        return ExecResult(0, "", "", 0.0)

    def inspect(self) -> dict:
        # [FUNCTION DOC]
        # Returns a fixed marker dict indicating test-only inspection data.
        return {"test_only": True}

    def logs(self, max_bytes: int = 65_536) -> str:
        # [FUNCTION DOC]
        # Returns an empty string regardless of max_bytes.
        return ""

    def stop(self) -> None:
        # [FUNCTION DOC]
        # Sets stopped to True.
        self.stopped = True


@dataclass
class FakeRuntimeProvider:
    # [CLASS DOC]
    # Dataclass fake of a validation runtime provider with a name, IMAGE_BUILD and
    # EXEC capabilities, only "public" network mode, a single shared
    # FakeRunningSubject, and lists recording launch specs and build roots.
    """Validation-only launcher fake that records explicit launch requests."""

    name: str = "fake-runtime"
    capabilities: frozenset = frozenset(
        {ProviderCapability.IMAGE_BUILD, ProviderCapability.EXEC}
    )
    supported_network_modes: frozenset[str] = frozenset({"public"})
    subject: FakeRunningSubject = field(default_factory=FakeRunningSubject)
    launches: list[LaunchSpec] = field(default_factory=list)
    builds: list[Path] = field(default_factory=list)

    def build(self, root: Path, execution: ExecutionDeclaration) -> str:
        # [FUNCTION DOC]
        # Records the build root and returns a fixed fake image digest
        # ("sha256:" followed by 64 "a" characters). execution is ignored.
        self.builds.append(root)
        return "sha256:" + "a" * 64

    def start(self, spec: LaunchSpec) -> FakeRunningSubject:
        # [FUNCTION DOC]
        # Records the launch spec and returns the shared fake subject.
        self.launches.append(spec)
        return self.subject
