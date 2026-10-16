# [MODULE DOC]
# ContainerProvider implementation that runs an OpenEnv server inside an Azure
# Container Apps (ACA) sandbox using the preview azure-containerapps-sandbox SDK
# (optional dependency "openenv[aca]"). The sandbox is created from an ACA disk
# image, the server command is started in the background via a shell exec, port
# 8000 is exposed as an anonymous public HTTPS port, and readiness is polled
# on /health. Security defaults: explicit public ingress opt-in, HTTPS-only
# URLs, a warning when egress is unrestricted plus a helper to build a
# default-deny egress policy, and withholding/redacting server logs in errors.
# Defaults: port 8000, auto-suspend 900 s, suspend modes Memory or Disk.
# SPDX-License-Identifier: BSD-3-Clause

"""Azure Container Apps Sandbox provider for OpenEnv environments.

Note: the ``RFC 002 security invariant S<n>`` references in this module point to
the **proposed** "Cloud Sandbox Providers" amendment to RFC 002, which is pending
review/sign-off by the RFC authors. They describe the security properties this
provider enforces; treat them as proposed (not yet ratified) until that sign-off.
"""

from __future__ import annotations

import os
import shlex
import time
import warnings
from typing import Any, Dict, Literal, Optional

from .providers import ContainerProvider


_DEFAULT_ACA_PORT = 8000
# 15 minutes matches the conservative auto-stop default used by other cloud
# sandbox providers (e.g. Daytona). A live RL rollout holds one WebSocket for
# the whole episode, so an aggressive auto-suspend can silently drop the
# session between slow steps. See RFC 002 "Cloud Sandbox Providers".
_DEFAULT_AUTO_SUSPEND_SECONDS = 900
_VALID_SUSPEND_MODES = {"Memory", "Disk"}


def _require_secure_url(url: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns url unchanged if it is a string starting with "https://"
    # (case-insensitive). Raises RuntimeError (without echoing the URL) otherwise.
    """Enforce https/wss transport (RFC 002 security invariant S1).

    `EnvClient` derives its WebSocket URL from this base URL, so a plaintext
    URL would become a cleartext `ws://` connection. The offending URL is
    deliberately omitted from the error because an anonymous ACA port URL is a
    bearer capability (S2) that must not leak into logs.
    """
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise RuntimeError(
            "ACA sandbox returned a non-HTTPS port URL. OpenEnv requires an "
            "https/wss base_url so EnvClient traffic is encrypted. Refusing to "
            "connect over plaintext."
        )
    return url


def _validate_anonymous_port(anonymous_port: Optional[bool]) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True only if anonymous_port is truthy. Raises ValueError with
    # guidance when it is None (no explicit opt-in) or falsy (authenticated ports
    # unsupported).
    """Require explicit, public-only ingress opt-in (RFC 002 invariant S2).

    Public exposure is never implicit (`None` is rejected), and authenticated
    ACA ports are rejected because `EnvClient` has no WebSocket auth mechanism
    yet.
    """
    if anonymous_port is None:
        raise ValueError(
            "ACASandboxProvider requires explicit anonymous_port=True. "
            "ACA anonymous ports are public ingress; authenticated ACA "
            "ports are not supported by EnvClient until a client-auth RFC "
            "defines WebSocket credentials."
        )
    if not anonymous_port:
        raise ValueError(
            "authenticated ACA sandbox ports are not supported by EnvClient "
            "yet. Use anonymous_port=True only for intentionally exposed "
            "sandbox URLs, or wait for a client-auth RFC."
        )
    return True


def _resolve_required(value: str | None, env_name: str, parameter: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns value, or else the named environment variable's value.
    # Raises ValueError naming the parameter and env var if both are empty.
    resolved = value or os.environ.get(env_name)
    if not resolved:
        raise ValueError(
            f"{parameter} is required. Pass it to ACASandboxProvider or set "
            f"the {env_name} environment variable."
        )
    return resolved


def _raise_install_error(exc: ImportError) -> None:
    # [FUNCTION DOC]
    # HELPER: Always raises RuntimeError telling the user to install openenv[aca],
    # chained from the given ImportError.
    raise RuntimeError(
        "Azure Container Apps sandbox support requires optional dependencies. "
        "Install them with `pip install openenv[aca]`."
    ) from exc


class _DefaultACASandboxAdapter:
    # [CLASS DOC]
    # HELPER: Thin wrapper over the ACA sandbox SDK client exposing create, port
    # exposure, exec, delete and close operations used by the provider (tests
    # substitute a fake with the same methods).
    """Thin adapter over the preview `azure-containerapps-sandbox` SDK.

    The provider talks to this private adapter instead of spreading SDK details
    through its own logic; tests inject a duck-typed fake in its place.
    """

    def __init__(
        self,
        *,
        subscription_id: str,
        resource_group: str,
        sandbox_group: str,
        region: str | None,
        endpoint: str | None,
        credential: Any,
        sdk_kwargs: dict[str, Any],
    ):
        # [FUNCTION DOC]
        # Imports the SDK (RuntimeError via _raise_install_error if missing), defaults
        # credential to azure.identity DefaultAzureCredential, derives the endpoint from
        # region when endpoint is not given, and constructs a SandboxGroupClient for the
        # subscription/resource group/sandbox group with extra sdk_kwargs.
        # Raises: ValueError if neither endpoint nor region is given.
        try:
            from azure.containerapps.sandbox import (
                endpoint_for_region,
                SandboxGroupClient,
            )
        except ImportError as exc:  # pragma: no cover - exercised via provider tests
            _raise_install_error(exc)

        if credential is None:
            try:
                from azure.identity import DefaultAzureCredential
            except (
                ImportError
            ) as exc:  # pragma: no cover - exercised via provider tests
                _raise_install_error(exc)

            credential = DefaultAzureCredential()

        if endpoint is None:
            if region is None:
                raise ValueError(
                    "region is required when endpoint is not provided. Pass it to "
                    "ACASandboxProvider or set AZURE_REGION."
                )
            endpoint = endpoint_for_region(region)

        self._client = SandboxGroupClient(
            endpoint,
            credential,
            subscription_id=subscription_id,
            resource_group=resource_group,
            sandbox_group=sandbox_group,
            **sdk_kwargs,
        )

    def create_sandbox(
        self,
        *,
        disk: str | None,
        disk_id: str | None,
        env_vars: dict[str, str] | None,
        labels: dict[str, str] | None,
        egress_policy: Any,
        cpu: str,
        memory: str,
        disk_size: str | None,
        auto_suspend_seconds: int,
        auto_suspend_mode: str,
        create_timeout: int,
        polling_interval: int,
    ) -> Any:
        # [FUNCTION DOC]
        # Calls the SDK's begin_create_sandbox with the given source (disk or disk_id),
        # resources, auto-suspend settings, labels, env vars (as "environment"),
        # egress policy and polling settings, omitting None values, and blocks on the
        # poller result. Returns the sandbox-scoped client handle. Network I/O.
        kwargs: dict[str, Any] = {
            "disk": disk,
            "disk_id": disk_id,
            "cpu": cpu,
            "memory": memory,
            "disk_size": disk_size,
            "auto_suspend_seconds": auto_suspend_seconds,
            "auto_suspend_mode": auto_suspend_mode,
            "labels": labels,
            "environment": env_vars,
            "egress_policy": egress_policy,
            "polling_timeout": create_timeout,
            "polling_interval": polling_interval,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        # `begin_create_sandbox(...).result()` blocks until the sandbox is
        # *Running* and returns a sandbox-scoped `SandboxClient` (not a plain
        # data model). That client exposes exec / add_port / begin_delete
        # directly, which is the handle the rest of this adapter operates on.
        return self._client.begin_create_sandbox(**kwargs).result()

    def add_port(self, sandbox: Any, *, port: int, anonymous: bool) -> str:
        # [FUNCTION DOC]
        # Exposes port on the sandbox (anonymous flag passed through) and returns the
        # URL from the result's "url" attribute or dict key as a string.
        # Raises: RuntimeError if no URL is returned.
        sandbox_port = sandbox.add_port(port, anonymous=anonymous)
        url = getattr(sandbox_port, "url", None)
        if not url and isinstance(sandbox_port, dict):
            url = sandbox_port.get("url")
        if not url:
            raise RuntimeError(
                "ACA sandbox did not return a URL for the exposed port. "
                "OpenEnv requires a base_url that EnvClient can connect to directly."
            )
        return str(url)

    def exec(
        self,
        sandbox: Any,
        command: str,
        *,
        working_directory: str | None = None,
    ) -> Any:
        # [FUNCTION DOC]
        # Runs a shell command in the sandbox (optional working directory) and returns
        # the SDK result.
        return sandbox.exec(command, working_directory=working_directory)

    def delete_sandbox(self, sandbox: Any) -> None:
        # [FUNCTION DOC]
        # Deletes the sandbox, using the long-running begin_delete poller (blocking)
        # when available, otherwise a plain delete.
        if hasattr(sandbox, "begin_delete"):
            sandbox.begin_delete().result()
        else:
            sandbox.delete()

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the underlying SDK client.
        self._client.close()


def _source_from_image(image: str) -> tuple[str | None, str | None]:
    # [FUNCTION DOC]
    # HELPER: Parses an image string into (disk, disk_id): "disk:<name>" -> (name,
    # None); "disk-id:<id>" -> (None, id); a bare name -> (name, None).
    # Raises: ValueError for empty input, empty name/id after the prefix, or a bare
    # string containing "/" or ":" (looks like an OCI image reference).
    if not image or not image.strip():
        raise ValueError("image must be a non-empty ACA disk or disk-id reference")
    image = image.strip()

    if image.startswith("disk:"):
        disk = image[len("disk:") :].strip()
        if not disk:
            raise ValueError("disk: source must include a disk image name")
        return disk, None
    if image.startswith("disk-id:"):
        disk_id = image[len("disk-id:") :].strip()
        if not disk_id:
            raise ValueError("disk-id: source must include a disk image id")
        return None, disk_id

    # A bare string is treated as a public disk-image name. Guard the common
    # migration foot-gun: an ACA sandbox source is NOT a Docker/OCI registry
    # image, so a value that looks like one (a registry path with "/", or an
    # "image:tag") is almost certainly a mistake and would otherwise surface as
    # an opaque ACA SDK error. Fail fast with guidance instead.
    if "/" in image or ":" in image:
        raise ValueError(
            f"{image!r} looks like a container/OCI image reference, but "
            "ACASandboxProvider sources are Azure Container Apps sandbox disks, "
            "not Docker registry images. Use a bare public disk name, "
            "'disk:<name>' for a public disk image, or 'disk-id:<id>' for a "
            "private one (building a disk image from a container image is a "
            "separate, out-of-band ACA step)."
        )

    return image, None


def _exec_stdout(result: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: Extracts text output from an exec result: the first non-None of
    # attributes stdout, result, output; then the same keys if it is a dict; else
    # str(result).
    for attr in ("stdout", "result", "output"):
        value = getattr(result, attr, None)
        if value is not None:
            return str(value)
    if isinstance(result, dict):
        for key in ("stdout", "result", "output"):
            value = result.get(key)
            if value is not None:
                return str(value)
    return str(result)


class ACASandboxProvider(ContainerProvider):
    # [CLASS DOC]
    # ContainerProvider managing at most one ACA sandbox at a time. Stores resource
    # settings (cpu, memory, disk size, auto-suspend), labels, egress policy, the
    # default image/env vars/cmd/working directory, log surfacing flag and timeouts;
    # tracks the active sandbox handle, base URL, port, whether it started the
    # server process, and the set of injected env var values for log redaction.
    """Container provider backed by Azure Container Apps Sandboxes.

    `start_container`'s `image` is an ACA sandbox source (`disk:<name>` for a
    public disk image or `disk-id:<id>` for a private one), not a Docker
    registry image. The provider boots an OpenEnv server inside the sandbox,
    exposes it on an anonymous ACA port, and returns an `https://` URL that
    `EnvClient` connects to over `wss://`.

    The environment runs untrusted code, so the provider is secure by default
    (proposed RFC 002 security invariants): it requires explicit `anonymous_port=True`
    ingress (S2), enforces https/wss transport (S1), offers
    `deny_all_egress()` to block the cloud metadata/IMDS endpoint (S3), and
    never surfaces raw sandbox output unless `surface_server_logs=True` (S4).

    Validated end to end against a live ACA sandbox group; the underlying
    `azure-containerapps-sandbox` SDK is preview, so pin it and re-validate
    after upgrades (see `tests/test_core/test_aca_provider_integration.py`).

    `close()` stops the active sandbox *and* releases the underlying SDK client,
    so prefer using the provider as a context manager (or call `close()`
    explicitly) for deterministic cleanup. `close()`/`__enter__`/`__exit__` are
    defined on `ContainerProvider` (default no-op), so a caller holding a bare
    `ContainerProvider` reference can release the client polymorphically.

    Only one sandbox is active per provider: calling `start_container` again
    before `stop_container()`/`close()` raises `RuntimeError` rather than
    orphaning the running sandbox.

    ```python
    with ACASandboxProvider(
        image="disk:my-env",
        anonymous_port=True,
        ...,
    ) as provider:
        base_url = provider.start_container(cmd=...)
        ...
    # sandbox deleted and SDK client closed on exit
    ```
    """

    def __init__(
        self,
        *,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        sandbox_group: Optional[str] = None,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        credential: Any = None,
        cpu: str = "1000m",
        memory: str = "2048Mi",
        disk_size: Optional[str] = None,
        auto_suspend_seconds: int = _DEFAULT_AUTO_SUSPEND_SECONDS,
        auto_suspend_mode: Literal["Memory", "Disk"] = "Memory",
        labels: Optional[Dict[str, str]] = None,
        egress_policy: Any = None,
        anonymous_port: Literal[True] | None = None,
        image: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        cmd: Optional[str] = None,
        working_directory: Optional[str] = None,
        surface_server_logs: bool = False,
        create_timeout: int = 300,
        polling_interval: int = 3,
        sdk_kwargs: Optional[dict[str, Any]] = None,
        _adapter: Any = None,
    ):
        # [FUNCTION DOC]
        # Validates auto_suspend_mode (ValueError) and anonymous_port (must be True,
        # ValueError otherwise). Fills subscription_id, resource_group, sandbox_group
        # and region from AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP /
        # AZURE_SANDBOX_GROUP / AZURE_REGION when not given. Copies labels and env vars.
        # Creates a _DefaultACASandboxAdapter (requiring the first three settings,
        # ValueError if missing; owns it for closing) unless a test adapter is injected
        # (not owned). Initializes empty runtime state.
        # Side effects: may construct Azure credentials/SDK client.
        if auto_suspend_mode not in _VALID_SUSPEND_MODES:
            raise ValueError("auto_suspend_mode must be 'Memory' or 'Disk'")
        _validate_anonymous_port(anonymous_port)

        self.subscription_id = subscription_id or os.environ.get(
            "AZURE_SUBSCRIPTION_ID"
        )
        self.resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
        self.sandbox_group = sandbox_group or os.environ.get("AZURE_SANDBOX_GROUP")
        self.region = region or os.environ.get("AZURE_REGION")
        self.endpoint = endpoint

        self.cpu = cpu
        self.memory = memory
        self.disk_size = disk_size
        self.auto_suspend_seconds = auto_suspend_seconds
        self.auto_suspend_mode = auto_suspend_mode
        self.labels = dict(labels or {})
        self.egress_policy = egress_policy
        self.anonymous_port = anonymous_port
        self._image = image
        self._env_vars = dict(env_vars or {}) if env_vars is not None else None
        self.cmd = cmd
        self.working_directory = working_directory
        self.surface_server_logs = surface_server_logs
        self.create_timeout = create_timeout
        self.polling_interval = polling_interval

        if _adapter is None:
            self._adapter = _DefaultACASandboxAdapter(
                subscription_id=_resolve_required(
                    self.subscription_id, "AZURE_SUBSCRIPTION_ID", "subscription_id"
                ),
                resource_group=_resolve_required(
                    self.resource_group, "AZURE_RESOURCE_GROUP", "resource_group"
                ),
                sandbox_group=_resolve_required(
                    self.sandbox_group, "AZURE_SANDBOX_GROUP", "sandbox_group"
                ),
                region=self.region,
                endpoint=self.endpoint,
                credential=credential,
                sdk_kwargs=dict(sdk_kwargs or {}),
            )
            self._owns_adapter = True
        else:
            self._adapter = _adapter
            self._owns_adapter = False

        self._sandbox: Any = None
        self._base_url: Optional[str] = None
        self._port = _DEFAULT_ACA_PORT
        self._started_server = False
        # Injected env-var values, used to scrub captured server output before
        # it is ever surfaced in an error (RFC 002 security invariant S4).
        self._redact_values: set[str] = set()

    @staticmethod
    def deny_all_egress(allow: Optional[list[str]] = None) -> Any:
        # [FUNCTION DOC]
        # Static method returning an SDK EgressPolicy with default action Deny and an
        # Allow host rule per pattern in allow. Raises RuntimeError if the SDK is not
        # installed.
        """Build a default-deny ACA `EgressPolicy` (RFC 002 security invariant S3).

        Untrusted RL/agent code should not be able to exfiltrate data or reach
        the cloud metadata/IMDS endpoint. This returns an `EgressPolicy` whose
        default action is `Deny`, with an optional host allowlist (for example a
        model endpoint or package registry the environment legitimately needs).

        ```python
        provider = ACASandboxProvider(
            anonymous_port=True,
            egress_policy=ACASandboxProvider.deny_all_egress(
                allow=["my-model.openai.azure.com"]
            ),
        )
        ```
        """
        try:
            from azure.containerapps.sandbox import EgressHostRule, EgressPolicy
        except ImportError as exc:  # pragma: no cover - exercised via provider tests
            _raise_install_error(exc)

        host_rules = [
            EgressHostRule(pattern=pattern, action="Allow") for pattern in (allow or [])
        ]
        return EgressPolicy(default_action="Deny", host_rules=host_rules)

    def start_container(
        self,
        image: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        # [FUNCTION DOC]
        # Creates a sandbox and returns its HTTPS base URL.
        # Raises RuntimeError if a sandbox is already active; ValueError if port is
        # not 8000, no image is available, the image string is invalid, the effective
        # anonymous_port override is not True, or unknown kwargs remain (accepted:
        # labels merged over defaults, cmd, egress_policy, anonymous_port). Warns
        # (UserWarning) when egress_policy is None. Records non-empty env var values for
        # redaction. Creates the sandbox (on failure clears redaction values and
        # re-raises without deleting anything). Then, if a cmd is set, starts the
        # server in the background; exposes the port and enforces an HTTPS URL. Any
        # failure after creation calls stop_container (errors suppressed) and
        # re-raises. env_vars argument overrides constructor env vars when not None.
        # Side effects: network calls creating cloud resources.
        """Start an OpenEnv server in an ACA sandbox and return its base URL.

        `image` is an ACA sandbox *source*, not a Docker/OCI registry image: a
        bare string or `disk:<name>` for a public disk image, or `disk-id:<id>`
        for a private one. It may be omitted when supplied to the constructor.
        A value that looks like a container image (a registry path or
        `name:tag`) is rejected with guidance. Only port 8000 is supported.
        `**kwargs` accepts per-start overrides (`cmd`, `labels`,
        `egress_policy`, `anonymous_port`); unknown options raise `ValueError`
        so typos cannot silently change sandbox behavior.
        """
        if self._sandbox is not None:
            raise RuntimeError(
                "ACASandboxProvider already has an active sandbox. Call "
                "stop_container() (or close()) before starting another — a "
                "second start would orphan the running sandbox."
            )

        bind_port = port or _DEFAULT_ACA_PORT
        if bind_port != _DEFAULT_ACA_PORT:
            raise ValueError(
                f"ACASandboxProvider only supports port {_DEFAULT_ACA_PORT} "
                f"(got {bind_port})."
            )

        effective_image = image if image is not None else self._image
        if effective_image is None:
            raise ValueError(
                "ACASandboxProvider requires an image. Pass it to the constructor "
                "or start_container()."
            )
        effective_env_vars = self._env_vars if env_vars is None else env_vars

        disk, disk_id = _source_from_image(effective_image)
        labels = dict(self.labels)
        labels.update(kwargs.pop("labels", {}) or {})
        cmd = kwargs.pop("cmd", None) or self.cmd
        egress_policy = kwargs.pop("egress_policy", self.egress_policy)
        anonymous_port = kwargs.pop("anonymous_port", self.anonymous_port)
        # Re-validate in case anonymous_port was overridden at start time: public
        # exposure must stay an explicit, opt-in decision (S2).
        _validate_anonymous_port(anonymous_port)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported ACASandboxProvider start options: {unknown}")

        # Security (RFC 002 S3): the env server runs untrusted code and ACA's
        # EgressPolicy defaults to default_action="Allow", so an unset policy
        # means unrestricted egress — the workload could exfiltrate data or
        # steal the sandbox managed-identity token from the cloud metadata
        # endpoint. Warn loudly; do not silently allow it.
        if egress_policy is None:
            warnings.warn(
                "ACASandboxProvider is starting a sandbox with unrestricted "
                "egress (no egress_policy). The environment runs untrusted code; "
                "prefer a default-deny policy via "
                "ACASandboxProvider.deny_all_egress(allow=[...]) and ensure the "
                "cloud metadata/IMDS endpoint is blocked.",
                stacklevel=2,
            )

        # Record injected secret values so captured server output can be scrubbed
        # before it is ever surfaced in an error (security invariant S4).
        self._redact_values = {
            value for value in (effective_env_vars or {}).values() if value
        }

        # Two-stage cleanup. A create failure created nothing, so just drop the
        # recorded secrets and re-raise (notably it must NOT delete a sandbox
        # this call did not create — e.g. one left over on the provider). Only a
        # failure AFTER the sandbox exists runs stop_container() to delete it.
        try:
            self._sandbox = self._adapter.create_sandbox(
                disk=disk,
                disk_id=disk_id,
                env_vars=dict(effective_env_vars or {}) if effective_env_vars else None,
                labels=labels or None,
                egress_policy=egress_policy,
                cpu=self.cpu,
                memory=self.memory,
                disk_size=self.disk_size,
                auto_suspend_seconds=self.auto_suspend_seconds,
                auto_suspend_mode=self.auto_suspend_mode,
                create_timeout=self.create_timeout,
                polling_interval=self.polling_interval,
            )
        except Exception:
            self._redact_values = set()
            raise

        try:
            if cmd:
                self._start_server(cmd)
            self._base_url = _require_secure_url(
                self._adapter.add_port(
                    self._sandbox,
                    port=bind_port,
                    anonymous=anonymous_port,
                )
            )
            self._port = bind_port
        except Exception:
            # A cleanup failure here must not mask the original error: swallow
            # any exception from stop_container() so the root cause propagates.
            try:
                self.stop_container()
            except Exception:
                pass
            raise

        return self._base_url

    def _start_server(self, cmd: str) -> None:
        # `cmd` is trusted orchestrator configuration, not agent/environment
        # input — callers must not pass agent-controlled text as `cmd`, and any
        # dynamic value interpolated into it must be quoted by the caller. Given
        # that, running it through the subshell below is safe (RFC 002 amendment
        # S5 — no command injection from untrusted sources). `working_directory`
        # is still shlex-quoted defensively.
        # [FUNCTION DOC]
        # HELPER: Builds a command (prefixed with "cd <quoted working directory> &&"
        # if set), shell-quotes it, and execs in the sandbox a nohup bash invocation in
        # the background redirecting output to /tmp/openenv-server.log and writing its
        # PID to /tmp/openenv-server.pid. Marks the server as started. cmd is trusted
        # caller configuration and is not sanitized.
        if self.working_directory:
            command = f"cd {shlex.quote(self.working_directory)} && {cmd}"
        else:
            command = cmd
        escaped = shlex.quote(command)
        # The adapter's `exec` maps to the ACA SDK's `executeShellCommand`
        # endpoint, which runs its argument through a POSIX shell in the sandbox
        # (the SDK docstring says "Execute a shell command" and tells callers to
        # `shlex.quote()` interpolated input). That shell is what makes the
        # backgrounding (`&`), output redirection, and `$!` PID capture below
        # work; if a future preview SDK changes `exec` to bypass the shell, this
        # start sequence must be revisited (re-validate via the integration test).
        self._adapter.exec(
            self._sandbox,
            f"nohup bash -c {escaped} > /tmp/openenv-server.log 2>&1 & "
            "echo $! > /tmp/openenv-server.pid",
        )
        self._started_server = True

    def _redact(self, text: str, *, max_chars: int = 2000) -> str:
        # [FUNCTION DOC]
        # HELPER: Replaces each recorded injected env var value in text with "***" and,
        # if longer than max_chars (default 2000), keeps only the last max_chars
        # characters prefixed with a truncation marker. Returns the string.
        """Scrub injected secret values and bound length before surfacing output.

        Replaces any injected env-var value with `***` and keeps only the tail.
        This is best-effort (exact-match only), which is why server output is
        withheld entirely unless `surface_server_logs=True` (RFC 002 S4).
        """
        redacted = text or ""
        for value in self._redact_values:
            redacted = redacted.replace(value, "***")
        if len(redacted) > max_chars:
            redacted = "...(truncated)...\n" + redacted[-max_chars:]
        return redacted

    def _server_died_message(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the error message for a server that died during startup.
        # Without surface_server_logs (or without a sandbox) it is a fixed message
        # pointing to the sandbox log file; otherwise it reads the log file via exec
        # and appends a redacted, bounded excerpt.
        """Build the startup-crash error, secure by default (RFC 002 S4).

        Untrusted code can print secrets then force a crash to exfiltrate them
        through the exception (which lands in orchestrator/CI logs), so sandbox
        output is excluded unless `surface_server_logs=True`, in which case a
        best-effort redacted, bounded excerpt is included.
        """
        base = (
            "ACA sandbox server process died during startup. Server output is "
            "not surfaced to avoid leaking secrets injected into the sandbox; "
            "retrieve /tmp/openenv-server.log from the sandbox out of band, or "
            "construct the provider with surface_server_logs=True to include a "
            "redacted excerpt."
        )
        if not self.surface_server_logs:
            return base
        if self._sandbox is None:
            return base
        log = self._redact(
            _exec_stdout(
                self._adapter.exec(
                    self._sandbox,
                    "cat /tmp/openenv-server.log 2>/dev/null",
                )
            )
        )
        return (
            "ACA sandbox server process died during startup. The excerpt below "
            "is the sandbox server output with injected secret values redacted "
            "(best-effort); it may still contain secrets the workload printed "
            f"by other means.\nLog (redacted):\n{log}"
        )

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        # [FUNCTION DOC]
        # Polls GET <base_url>/health (5 s request timeout) about once per second until
        # it returns 200 or timeout_s elapses. If this provider started the server, each
        # iteration checks the PID file's process liveness in the sandbox and raises
        # RuntimeError with _server_died_message if it is dead.
        # Raises: TimeoutError on deadline.
        # Side effects: HTTP requests, sandbox exec calls, sleeping.
        """Wait for the ACA-hosted OpenEnv server to answer on `/health`.

        A `200` on `/health` proves HTTP reachability but not that the exposed
        ACA port proxies the `/ws` WebSocket upgrade `EnvClient` needs; the
        integration test covers the full `wss://` round-trip. If the captured
        server process dies during startup this raises `RuntimeError` (sandbox
        output is withheld unless `surface_server_logs=True`; see RFC 002 S4).
        """
        # Imported lazily inside the method that needs it, matching the other
        # providers in this package (e.g. DaytonaProvider).
        import requests

        deadline = time.time() + timeout_s
        health_url = f"{base_url}/health"

        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            if self._started_server and self._sandbox is not None:
                status = _exec_stdout(
                    self._adapter.exec(
                        self._sandbox,
                        "kill -0 $(cat /tmp/openenv-server.pid) 2>/dev/null "
                        "&& echo RUNNING || echo DEAD",
                    )
                )
                if "DEAD" in status:
                    raise RuntimeError(self._server_died_message())

            time.sleep(1.0)

        raise TimeoutError(
            f"ACA sandbox did not become ready within {timeout_s}s. "
            "If the sandbox image starts the server itself, pass cmd= to let "
            "ACASandboxProvider capture /tmp/openenv-server.log for early crash "
            "diagnostics."
        )

    def stop_container(self) -> None:
        # [FUNCTION DOC]
        # If a sandbox is active, deletes it via the adapter and always resets the
        # sandbox handle, base URL, started flag and redaction values (even if deletion
        # raises, which then propagates). With no sandbox, only clears redaction values.
        """Delete the active ACA sandbox."""
        if self._sandbox is None:
            # Still drop any injected secret values recorded by a failed start
            # (e.g. create_sandbox raised before a sandbox handle existed).
            self._redact_values = set()
            return

        try:
            self._adapter.delete_sandbox(self._sandbox)
        finally:
            self._sandbox = None
            self._base_url = None
            self._started_server = False
            # Drop the previous episode's injected secret values; a new start
            # repopulates them (avoids carrying stale values across sandboxes).
            self._redact_values = set()

    @property
    def base_url(self) -> str:
        # [FUNCTION DOC]
        # Returns the base URL of the active sandbox. Raises RuntimeError if none.
        """URL returned by the last `start_container`."""
        if self._base_url is None:
            raise RuntimeError(
                "ACASandboxProvider has no active base_url. Start the provider "
                "before reading base_url."
            )
        return self._base_url

    def close(self) -> None:
        # [FUNCTION DOC]
        # Stops the active sandbox, then closes the adapter once if this provider owns
        # it (subsequent calls do not close it again).
        """Stop the active sandbox and close the underlying SDK client.

        Overrides the base no-op so a caller can release the preview SDK client
        deterministically (also invoked on context-manager exit). The base
        `ContainerProvider` defines `close()`/`__enter__`/`__exit__`, so callers
        holding a bare `ContainerProvider` can release resources polymorphically.
        """
        self.stop_container()
        if self._owns_adapter:
            self._adapter.close()
            # don't close a second time on a subsequent close()/context exit
            self._owns_adapter = False


__all__ = ["ACASandboxProvider"]
