# [MODULE DOC]
# RuntimeProvider that launches an environment's ASGI app as a local
# subprocess with "uv run --isolated --project <path> -- uvicorn ...". The
# project may be a local directory or a "git+<url>" spec that is shallow-cloned
# into a temporary directory at start time (and deleted on stop/restart).
# Readiness is checked by polling the /health endpoint.
"""Providers for launching ASGI applications via ``uv run``."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Dict, Optional

import requests

from .providers import RuntimeProvider

GIT_URL_PREFIX = "git+"


def _check_uv_installed() -> None:
    # [FUNCTION DOC]
    # HELPER: Runs "uv --version"; raises RuntimeError if the uv executable is not
    # found. (A non-zero exit raises CalledProcessError.)
    try:
        subprocess.check_output(["uv", "--version"])
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`uv` executable not found. Install uv from https://docs.astral.sh and ensure it is on PATH."
        ) from exc


def _clone_git_project(git_url: str, timeout_s: float) -> str:
    # [FUNCTION DOC]
    # HELPER: Strips the "git+" prefix and runs "git clone --depth 1 <url>" into a
    # new temp directory with the given timeout. Returns the directory path.
    # Raises: RuntimeError if git is missing, the clone fails (includes stderr) or
    # times out; the temp directory is removed in those cases.
    """Clone a `git+<url>` spec to a temp dir and return its local path.

    `uv run --project` only discovers a project in a local directory (per
    `uv run --help`) -- it has no support for remote git sources. Callers
    that accept a `git+...` `project_path` (e.g. `EnvClient.from_env`)
    must therefore clone it themselves before handing a path to `uv run`.
    """
    repo_url = git_url[len(GIT_URL_PREFIX) :]
    clone_dir = tempfile.mkdtemp(prefix="openenv-uv-clone-")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, clone_dir],
            check=True,
            capture_output=True,
            text=True,
            # A hung/slow remote would otherwise block start() (and the
            # caller's own readiness timeout) indefinitely.
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(
            f"`git` executable not found; required to clone project_path {git_url!r}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to clone {repo_url!r}: {exc.stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(
            f"Timed out cloning {repo_url!r} after {timeout_s:.1f}s"
        ) from exc
    return clone_dir


def _find_free_port() -> int:
    # [FUNCTION DOC]
    # HELPER: Returns an OS-assigned free TCP port (released before use, so racy).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _create_uv_command(
    *,
    host: str,
    port: int,
    reload: bool,
    workers: int,
    app: str,
    project_path: str,
) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Builds the argument list "uv run --isolated --project <project_path>
    # -- uvicorn <app> --host <host> --port <port> --workers <workers>" plus
    # "--reload" when requested.
    command: list[str] = ["uv", "run", "--isolated", "--project", project_path]

    command.append("--")
    command.extend(
        [
            "uvicorn",
            app,
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            str(workers),
        ]
    )

    if reload:
        command.append("--reload")

    return command


def _poll_health(health_url: str, timeout_s: float) -> None:
    # [FUNCTION DOC]
    # HELPER: GETs health_url repeatedly (per-request timeout = min(remaining
    # time, 2 s)) with 0.5 s sleeps until a 200 response. Raises TimeoutError after
    # timeout_s.
    """Poll a health endpoint until it returns HTTP 200 or times out."""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            timeout = max(0.0001, min(deadline - time.time(), 2.0))
            response = requests.get(health_url, timeout=timeout)
            if response.status_code == 200:
                return
        except requests.RequestException:
            # Server not accepting connections yet. Fall through to the sleep
            # below rather than `continue`-ing: a refused connection returns
            # immediately, so retrying without a pause busy-spins a CPU core
            # for the whole timeout window while the server boots.
            pass

        time.sleep(0.5)

    raise TimeoutError(f"Server did not become ready within {timeout_s:.1f} seconds")


class UVProvider(RuntimeProvider):
    # [CLASS DOC]
    # RuntimeProvider managing one uv/uvicorn subprocess for a project. Stores the
    # project path (absolute for local paths, verbatim for git specs), app import
    # path, host, reload flag, default env vars and readiness timeout; tracks the
    # process, base URL and any clone directory.
    """
    RuntimeProvider implementation backed by ``uv run``.

    Args:
        project_path (`str`):
            Local path to a uv project (passed to `uv run --project`), or a
            `git+<url>` spec that is cloned to a temp directory on `start()`.
        app (`str`, *optional*, defaults to `"server.app:app"`):
            ASGI application path for uvicorn.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host interface to bind to.
        reload (`bool`, *optional*, defaults to `False`):
            Whether to enable uvicorn's reload mode.
        env_vars (`dict`, *optional*):
            Environment variables to pass through to the spawned process.
        context_timeout_s (`float`, *optional*, defaults to `60.0`):
            How long to wait for the environment to become ready.

    Examples:

        ```python
        provider = UVProvider(project_path="/path/to/env")
        base_url = provider.start()
        print(base_url)  # http://localhost:8000
        # Use the environment via base_url
        provider.stop()
        ```
    """

    def __init__(
        self,
        *,
        project_path: str,
        app: str = "server.app:app",
        host: str = "0.0.0.0",
        reload: bool = False,
        env_vars: Optional[Dict[str, str]] = None,
        context_timeout_s: float = 60.0,
    ):
        # [FUNCTION DOC]
        # Stores configuration (resolving local project paths to absolute) and checks
        # that uv is installed (RuntimeError otherwise). No process is started.
        """Initialize the UVProvider."""
        # `os.path.abspath` would mangle a `git+<url>` spec (e.g. collapsing
        # `https://` to `https:/`) and is meaningless for one anyway -- only
        # resolve it for genuine local paths. Git specs are cloned in `start()`.
        self.project_path = (
            project_path
            if project_path.startswith(GIT_URL_PREFIX)
            else os.path.abspath(project_path)
        )
        self.app = app
        self.host = host
        self.reload = reload
        self.env_vars = env_vars
        self.context_timeout_s = context_timeout_s
        _check_uv_installed()
        self._process = None
        self._base_url = None
        self._clone_dir: Optional[str] = None

    def start(
        self,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        workers: int = 1,
        **_: Dict[str, str],
    ) -> str:
        # [FUNCTION DOC]
        # Starts the server subprocess and returns its base URL.
        # Raises RuntimeError if the previous process is still alive. Uses port or a
        # free port; removes any leftover clone dir; clones git specs (RuntimeError on
        # failure); builds the uv command with workers; environment is the current
        # process env updated with constructor env vars then call env vars. Launches
        # with Popen (output inherited); on OSError removes the clone and raises
        # RuntimeError. Base URL host is 127.0.0.1 when binding to 0.0.0.0 or "::",
        # else the configured host. Extra kwargs are ignored. Does not wait for
        # readiness.
        """
        Start the environment via `uv run`.

        Args:
            port (`int`, *optional*):
                The port to bind the environment to.
            env_vars (`dict`, *optional*):
                Environment variables to pass to the environment.
            workers (`int`, *optional*, defaults to `1`):
                The number of workers to use.

        Returns:
            `str`: Base URL of the environment.

        Raises:
            RuntimeError: If the environment is already running.
        """
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("UVProvider is already running")

        bind_port = port or _find_free_port()

        # A previous start() may have left a clone dir behind if the spawned
        # process later died on its own -- the "already running" guard above
        # only checks liveness, so a restart in that state would otherwise
        # overwrite self._clone_dir and leak the old directory.
        if self._clone_dir is not None:
            shutil.rmtree(self._clone_dir, ignore_errors=True)
            self._clone_dir = None

        project_path = self.project_path
        if project_path.startswith(GIT_URL_PREFIX):
            self._clone_dir = _clone_git_project(project_path, self.context_timeout_s)
            project_path = self._clone_dir

        command = _create_uv_command(
            host=self.host,
            port=bind_port,
            reload=self.reload,
            workers=workers,
            app=self.app,
            project_path=project_path,
        )

        env = os.environ.copy()

        if self.env_vars:
            env.update(self.env_vars)
        if env_vars:
            env.update(env_vars)

        try:
            self._process = subprocess.Popen(command, env=env)
        except OSError as exc:
            if self._clone_dir is not None:
                shutil.rmtree(self._clone_dir, ignore_errors=True)
                self._clone_dir = None
            raise RuntimeError(f"Failed to launch `uv run`: {exc}") from exc

        client_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        self._base_url = f"http://{client_host}:{bind_port}"
        return self._base_url

    def wait_for_ready(self, timeout_s: float | None = None) -> None:
        # [FUNCTION DOC]
        # Uses the provider's context_timeout_s if timeout_s is None. Raises
        # RuntimeError if the process has already exited; then polls <base_url>/health
        # (TimeoutError on deadline).
        # KNOWN ISSUE: process exit is only checked once before polling, so a crash
        # during startup is reported as a timeout rather than an early error.
        """
        Wait for the environment to become ready.

        Args:
            timeout_s (`float`, *optional*):
                Maximum time in seconds to wait for the environment to become
                ready. Defaults to the provider's `context_timeout_s`.

        Raises:
            RuntimeError: If the environment is not running.
            TimeoutError: If the environment does not become ready within the timeout.
        """
        if timeout_s is None:
            timeout_s = self.context_timeout_s

        if self._process and self._process.poll() is not None:
            code = self._process.returncode
            raise RuntimeError(f"uv process exited prematurely with code {code}")

        _poll_health(f"{self._base_url}/health", timeout_s=timeout_s)

    def stop(self) -> None:
        # [FUNCTION DOC]
        # Terminates the process if alive (waits 10 s, then kills and waits 5 s),
        # clears process and base URL, and deletes any clone directory.
        """
        Stop the environment.
        """
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)

            self._process = None
            self._base_url = None

        if self._clone_dir is not None:
            shutil.rmtree(self._clone_dir, ignore_errors=True)
            self._clone_dir = None

    @property
    def base_url(self) -> str:
        # [FUNCTION DOC]
        # Returns the base URL; raises RuntimeError if not started.
        """
        The base URL of the environment.

        Returns:
            `str`: Base URL of the running environment.

        Raises:
            RuntimeError: If the environment has not been started.
        """
        if self._base_url is None:
            raise RuntimeError("UVProvider has not been started")
        return self._base_url
