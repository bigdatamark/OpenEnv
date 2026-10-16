# [PACKAGE DOC]
# Validation provider contract: error types, the ExecResult record, the RunningSubject
# protocol (a started sandbox with base URL, inspect, logs, exec, stop) and the
# ValidationProvider protocol (build an image from a package, start it from a LaunchSpec).
# Concrete implementations (e.g. Docker) live in submodules.
"""Validation provider protocol.

Validation needs more than start/stop: network-policy enforcement and in-sandbox exec
are grader dependencies. Rather than widen the core provider ABCs, validation defines
its own protocol and adapts the core providers (Docker-local, HF Sandbox) behind it.

GPU is a provider capability, not an unsupported package category: packages declaring
GPU resources validate on providers that offer GPUs; elsewhere the affected checks
SKIP with the capability named.

Universal invariants inherited from core: internal port 8000, readiness =
`GET /health` returning 200.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..manifest import ExecutionDeclaration
from ..runtime.contracts import LaunchSpec
from ..types import ProviderCapability


class ProviderError(RuntimeError):
    # [CLASS DOC]
    # Base RuntimeError for sanitized provider operation failures.
    """A bounded, sanitized provider operation failed."""


class StartupError(ProviderError):
    # [CLASS DOC]
    # ProviderError raised when a subject does not become healthy before its deadline.
    """A subject did not become healthy within its startup deadline."""


class UnsupportedCapability(ProviderError):
    # [CLASS DOC]
    # ProviderError raised when the provider cannot enforce the requested launch/execution contract.
    """The provider cannot enforce the requested execution contract."""


@dataclass(frozen=True)
class ExecResult:
    # [CLASS DOC]
    # Frozen record of an in-sandbox command: exit code, stdout, stderr and duration in seconds.
    """Result of executing a command inside the running sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


@runtime_checkable
class RunningSubject(Protocol):
    # [CLASS DOC]
    # Protocol for a started subject reachable at base_url.
    """A started validation subject: reachable, execable, stoppable."""

    base_url: str

    def inspect(self) -> dict:
        # [FUNCTION DOC]
        # Protocol method: returns provider-specific inspection data as a dict.
        ...

    def logs(self, max_bytes: int = 65536) -> str:
        # [FUNCTION DOC]
        # Protocol method: returns up to max_bytes of subject logs as text.
        ...

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult:
        # [FUNCTION DOC]
        # Protocol method: runs argv inside the sandbox with a timeout and returns an ExecResult.
        ...

    def stop(self) -> None:
        # [FUNCTION DOC]
        # Protocol method: stops and cleans up the subject.
        ...


@runtime_checkable
class ValidationProvider(Protocol):
    # [CLASS DOC]
    # Protocol for a sandbox provider: name, capability set and supported network modes.
    """
    Starts validation subjects in a sandbox with declared capabilities.

    A grader whose `requires_provider` names a capability the provider lacks is
    SKIPped with the capability named. The runner checks the manifest's declared
    network mode and GPU requirements before building. The provider must enforce
    the complete launch specification or refuse it. Enforcing `no-network` or
    `allowlist` modes requires the `NETWORK_POLICY` capability.
    """

    name: str
    capabilities: frozenset[ProviderCapability]
    supported_network_modes: frozenset[str]

    def build(self, root: Path, execution: ExecutionDeclaration) -> str:
        # [FUNCTION DOC]
        # Protocol method: builds the package at root per the execution declaration and returns an image reference.
        ...

    def start(self, spec: LaunchSpec) -> RunningSubject:
        # [FUNCTION DOC]
        # Protocol method: starts a subject from the launch spec and returns a RunningSubject;
        # must enforce the full spec or refuse.
        ...
