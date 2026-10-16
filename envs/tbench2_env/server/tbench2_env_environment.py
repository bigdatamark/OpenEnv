# [MODULE DOC]
# Terminal-Bench 2 server environments. Tbench2Environment runs agent commands locally through CAMEL's
# TerminalToolkit in the same filesystem as the server; Tbench2DockerEnvironment runs each task in its own
# container built from the task's official image. Both score with the task's canonical tests/test.sh
# staged at the fixed paths /tests and /logs/verifier only for the verify window (pytest fallback when a
# task has no test.sh), parsing marker lines echoed into the command output.
# Task directories come from TB2_TASKS_DIR or from a downloaded and cached TB2 repo archive.
# Module globals: fixed verify paths, reward and exit-code marker strings, cached camel import error.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TB2 environment server implementation (Spaces-compatible local mode)."""

from __future__ import annotations

import io
import logging
import os
import re
import shlex
import shutil
import tarfile
import threading
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from openenv.core.env_server.interfaces import Environment


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from tbench2_env.models import Tbench2Action, Tbench2Observation, Tbench2State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import Tbench2Action, Tbench2Observation, Tbench2State

_CAMEL_IMPORT_ERROR: Exception | None = None

# Official TB2 fixed paths: tests/test.sh assumes it runs from /tests and
# writes /logs/verifier/reward.txt. Module-level only so unit tests can
# redirect them off the real root filesystem.
_VERIFY_TESTS_DIR = "/tests"
_VERIFIER_LOG_DIR = "/logs/verifier"


def _hard_remove(path: str) -> None:
    # [FUNCTION DOC]
    # HELPER: removes a file or symlink (unlinking the link itself) or recursively deletes a directory,
    # ignoring missing paths and rmtree errors.
    """Remove whatever is at *path* — file, symlink, or directory tree.

    shutil.rmtree refuses to remove a symlink and, with ignore_errors=True,
    no-ops on one. That leaves an escape hatch: a root agent that symlinks a
    fixed path (/tests, /logs/verifier) at a directory it controls would keep
    that target alive across the "wipe", so a pre-planted conftest.py survives
    into scoring. Unlink the symlink itself before falling back to rmtree.
    """
    p = Path(path)
    if p.is_symlink() or p.is_file():
        p.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _extractall(tar: tarfile.TarFile, dest: str) -> None:
    # [FUNCTION DOC]
    # HELPER: extracts a tar archive to dest using the "data" safety filter, falling back to unfiltered
    # extraction on Pythons that do not accept the filter argument.
    """extractall with the 3.12 data filter when available. The tarball is one
    this server wrote from a trusted tests/ dir, so the filter is defence in
    depth; on <3.12 (the ``filter`` kwarg predates it) fall back cleanly rather
    than raising TypeError mid-verify (the package targets Python >=3.10)."""
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def _require_terminal_toolkit() -> Any:
    # [FUNCTION DOC]
    # HELPER: imports and returns CAMEL's TerminalToolkit class. Raises RuntimeError (chained to the import
    # error) if camel-ai is unavailable; caches the failure in a module global so later calls fail fast.
    global _CAMEL_IMPORT_ERROR
    if _CAMEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "camel-ai (TerminalToolkit) is required for TB2. Install from PyPI or from the CAMEL repo."
        ) from _CAMEL_IMPORT_ERROR

    try:
        from camel.toolkits import TerminalToolkit
    except Exception as exc:  # pragma: no cover
        _CAMEL_IMPORT_ERROR = exc
        raise RuntimeError(
            "camel-ai (TerminalToolkit) is required for TB2. Install from PyPI or from the CAMEL repo."
        ) from exc

    return TerminalToolkit


def _download_tb2_repo(cache_dir: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: ensures the TB2 repository zip (TB2_REPO_URL or the GitHub main-branch archive) is downloaded into
    # cache_dir and extracted. Downloads only if the zip is absent, extracts only if the top-level directory is
    # absent. Returns the extracted repo root path. Side effects: network download and filesystem writes.
    repo_url = os.getenv(
        "TB2_REPO_URL",
        "https://github.com/laude-institute/terminal-bench-2/archive/refs/heads/main.zip",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "terminal-bench-2.zip"

    if not archive_path.exists():
        urllib.request.urlretrieve(repo_url, archive_path)

    with zipfile.ZipFile(archive_path) as zf:
        root = zf.namelist()[0].split("/")[0]
        extract_dir = cache_dir / root
        if not extract_dir.exists():
            zf.extractall(cache_dir)

    return extract_dir


def _read_instruction(task_dir: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the task's instruction.md text, or "" if missing.
    instruction_path = task_dir / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text(encoding="utf-8")
    return ""


def _task_image_workdir(task_dir: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the last WORKDIR value in the task's environment/Dockerfile (case-insensitive match),
    # or "" if no Dockerfile or no WORKDIR.
    """WORKDIR pinned by the task's own image (environment/Dockerfile).

    Empty string when the task ships no Dockerfile or no WORKDIR; multi-stage
    Dockerfiles resolve to the last WORKDIR (the final stage's).
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    workdir = ""
    if dockerfile.is_file():
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*WORKDIR\s+(\S+)", line, re.IGNORECASE)
            if m:
                workdir = m.group(1)
    return workdir


def _workdir_is_server_tree(workdir: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if this module's file lives under workdir (meaning workdir is the server's install tree,
    # not the task's); also True if path resolution raises OSError.
    """True when *workdir* contains this server's own code.

    A Dockerfile-parsed WORKDIR is only meaningful when the server runs
    inside the task image (per-task sandboxes, where the server layer lives
    elsewhere, e.g. /opt/envserver). The standard env-server image also
    installs to /app — the most common task WORKDIR — so the path merely
    existing is not enough: if the server's own code sits under it, it is
    the server tree, not the task tree.
    """
    try:
        return Path(__file__).resolve().is_relative_to(Path(workdir).resolve())
    except OSError:
        return True  # unresolvable path: fail toward the task-dir fallback


def _read_timeout(task_dir: Path, fallback: float) -> float:
    # [FUNCTION DOC]
    # HELPER: returns the verifier timeout_sec from task.toml's [verifier] table as float, or fallback when the
    # file is missing or unparsable or has no such key.
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return fallback
    try:
        data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    verifier = data.get("verifier", {})
    return float(verifier.get("timeout_sec", fallback))


# The scoring exec echoes its verdict on a marker line so the caller can parse
# it out of mixed stdout. Shared by both execution modes (local terminal
# toolkit and Docker container exec).
_REWARD_MARKER = "__TB2_REWARD__:"
_EXIT_CODE_MARKER = "__TB2_EXIT_CODE__:"


def _canonical_eval_cmd(workdir: str, timeout_s: float | None = None) -> str:
    # [FUNCTION DOC]
    # HELPER: builds the shell command that cds into workdir, runs bash /tests/test.sh (wrapped with coreutils
    # timeout when timeout_s is given and available) with output to /logs/verifier/testsh.log, prints the last
    # 20000 bytes of that log, then echoes the reward marker followed by the contents of reward.txt.
    """The official-harness scoring command, run from the agent's workdir.

    /tests/test.sh (staged by the caller) pins its own pytest toolchain (uvx:
    Python 3.13 + pytest 8.4.1 + ctrf), runs tests/test_outputs.py from
    _VERIFY_TESTS_DIR against the task workdir, and writes the binary result
    to /logs/verifier/reward.txt. test.sh stdout goes under _VERIFIER_LOG_DIR
    (wiped with the verify window) rather than a fixed /tmp path that would
    outlive scoring: its pytest -rA output can spell out expected values. Its
    tail is echoed back into the returned output — the episode is over by
    then, and callers need the pytest diagnostics on failure; the reward
    marker line stays last for parsing.

    ``timeout_s`` bounds test.sh with coreutils ``timeout`` when present in
    the image. The local mode passes None: its terminal toolkit enforces the
    budget itself. Docker exec has no server-side timeout, so the task's own
    verifier budget is enforced in-shell.
    """
    run = f"bash {_VERIFY_TESTS_DIR}/test.sh"
    if timeout_s is not None:
        run = f"if command -v timeout >/dev/null 2>&1; then timeout {int(timeout_s)} {run}; else {run}; fi"
    return (
        f"cd {shlex.quote(workdir)} && "
        f"{run} > {_VERIFIER_LOG_DIR}/testsh.log 2>&1; "
        f"tail -c 20000 {_VERIFIER_LOG_DIR}/testsh.log 2>/dev/null; "
        f"echo {_REWARD_MARKER}$(cat {_VERIFIER_LOG_DIR}/reward.txt 2>/dev/null)"
    )


def _parse_canonical_reward(output: str) -> float | None:
    # [FUNCTION DOC]
    # HELPER: scans output lines from the end for the reward marker; returns the float after it, or None if
    # there is no marker, the value is empty, or it is not numeric.
    """The verdict test.sh's verifier wrote to reward.txt, echoed on the
    marker line. None when no verdict can be recovered — no marker line, an
    empty value (reward.txt absent: test.sh crashed or was killed before its
    verifier wrote one), or a non-numeric value. A missing verdict is a
    scoring failure, not a task failure: callers raise so ``evaluate``
    reports an error (observation.error set, reward None) instead of a 0.0
    indistinguishable from tests genuinely failing — RL consumers must be
    able to drop such episodes rather than train on a false negative.
    """
    for line in output.splitlines()[::-1]:
        if _REWARD_MARKER in line:
            raw = line.split(_REWARD_MARKER, 1)[1].strip()
            if not raw:
                return None
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def _require_canonical_verdict(reward: float | None, output: str) -> float:
    # [FUNCTION DOC]
    # HELPER: returns reward unchanged; raises RuntimeError (including the last 800 chars of output) if None.
    """Raise when the canonical harness produced no verdict (see above)."""
    if reward is None:
        raise RuntimeError(
            "canonical harness produced no verdict (reward.txt missing after "
            f"tests/test.sh ran); test.sh log tail: {output[-800:]!r}"
        )
    return reward


def _fallback_eval_cmd(workdir: str) -> str:
    # [FUNCTION DOC]
    # HELPER: builds a shell command that cds into workdir, runs pytest quietly on /tests (via uvx with pinned
    # pytest when available, else python -m pytest) and echoes the exit-code marker with pytest's exit status.
    """pytest against the staged tests copy, for task dirs without the
    canonical harness (none of the 89 official TB2 tasks — they all ship
    test.sh — but custom task dirs may only have bare pytest tests). Prefer
    uvx so pytest comes with its own toolchain like the canonical harness
    does. Verify from the same directory the agent worked in.
    """
    return (
        f"cd {shlex.quote(workdir)} && "
        "if command -v uvx >/dev/null 2>&1; "
        f"then uvx --with pytest==8.4.1 pytest -q {_VERIFY_TESTS_DIR} -rA; "
        f"else python -m pytest -q {_VERIFY_TESTS_DIR} -rA; fi; "
        f"echo {_EXIT_CODE_MARKER}$?"
    )


def _parse_exit_code_marker(output: str) -> int:
    # [FUNCTION DOC]
    # HELPER: returns the integer after the last exit-code marker line in output; 1 if absent or unparsable.
    for line in output.splitlines()[::-1]:
        if _EXIT_CODE_MARKER in line:
            try:
                return int(line.split(_EXIT_CODE_MARKER, 1)[1].strip())
            except Exception:
                return 1
    return 1


class Tbench2Environment(Environment[Tbench2Action, Tbench2Observation, Tbench2State]):
    # [CLASS DOC]
    # Local-execution TB2 environment. Holds configuration, the current task dir, instruction, resolved workdir,
    # the TerminalToolkit instance and Tbench2State. Class-level shared data: an in-memory cache of withheld
    # tests/ tarballs per task dir (with a lock) and a lock serialising canonical evaluation across sessions.
    # Supports concurrent sessions.
    """OpenEnv wrapper around Terminal-Bench 2 tasks (local execution)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_dir: str | None = None,
        output_dir: str | None = None,
        command_timeout_s: float | None = None,
        safe_mode: bool = False,
        default_task_id: str | None = None,
        withhold_tests: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs (falling back to env vars): tasks_dir (TB2_TASKS_DIR), output_dir (TB2_OUTPUT_DIR, default
        # /tmp/tbench2_env_runs), command_timeout_s (TB2_COMMAND_TIMEOUT_S, default 20.0), safe_mode, default task
        # id (TB2_DEFAULT_TASK_ID, default "headless-terminal"), withhold_tests (TB2_WITHHOLD_TESTS == "1").
        # Only stores settings; no I/O.
        super().__init__()
        self.tasks_dir = tasks_dir or os.getenv("TB2_TASKS_DIR", "")
        # RL hygiene: the env server and the agent share one container
        # filesystem, so tests/ and solution/ in the task dir are readable
        # (and writable) by the agent. When enabled, reset() moves them out
        # of reach — see _withhold_verifier_assets.
        if withhold_tests is None:
            withhold_tests = os.getenv("TB2_WITHHOLD_TESTS", "0") == "1"
        self.withhold_tests = withhold_tests
        self.output_dir = Path(
            output_dir or os.getenv("TB2_OUTPUT_DIR", "/tmp/tbench2_env_runs")
        )
        # Overridable via TB2_COMMAND_TIMEOUT_S: create_app instantiates the
        # environment with no arguments, and real TB2 agent commands / the
        # canonical tests/test.sh routinely exceed the old 20s default.
        if command_timeout_s is None:
            command_timeout_s = float(os.getenv("TB2_COMMAND_TIMEOUT_S", "20.0"))
        self.command_timeout_s = command_timeout_s
        self.safe_mode = safe_mode
        self.default_task_id = default_task_id or os.getenv(
            "TB2_DEFAULT_TASK_ID", "headless-terminal"
        )

        self._state = Tbench2State()
        self._task_dir: Path | None = None
        self._terminal_toolkit = None
        self._instruction = ""
        self._workdir = ""

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        # [FUNCTION DOC]
        # Seed ignored. Requires the terminal toolkit (RuntimeError otherwise). Task id comes from kwargs task_id
        # or task_name or the default; task path from task_path or path. Resolves the task dir (may download the
        # TB2 repo; FileNotFoundError/ValueError on failure), reads the instruction, optionally withholds verifier
        # assets, creates a per-trial session log dir under output_dir. Workdir: TB2_TASK_WORKDIR, else the task
        # Dockerfile's WORKDIR unless it is the server's own tree, else (or if not an existing dir) the task dir.
        # Creates a TerminalToolkit (local backend) in that workdir and a fresh state (given or random episode id).
        # Returns: observation with the instruction, action_type "reset", reward 0.0, done False.
        del seed

        TerminalToolkit = _require_terminal_toolkit()

        task_id = (
            kwargs.get("task_id") or kwargs.get("task_name") or self.default_task_id
        )
        task_path = kwargs.get("task_path") or kwargs.get("path")

        task_dir = self._resolve_task_path(task_id, task_path)
        resolved_task_id = task_id or task_dir.name

        self._instruction = _read_instruction(task_dir)
        self._task_dir = task_dir
        if self.withhold_tests:
            self._withhold_verifier_assets(task_dir)

        trial_name = f"{resolved_task_id}.{episode_id or uuid4().hex}"
        session_logs_dir = (
            self.output_dir / trial_name / "terminal_toolkit_session_logs"
        )
        session_logs_dir.mkdir(parents=True, exist_ok=True)

        # Commands run in the task image's real WORKDIR — where agents and
        # the canonical harness expect to operate, and not always /app
        # (fix-git: /app/personal-site, prove-plus-comm: /workspace).
        # TB2_TASK_WORKDIR overrides; otherwise the task's own Dockerfile is
        # parsed, trusted only when the path isn't the server's own install
        # tree (see _workdir_is_server_tree); plain local mode falls back to
        # the task source dir.
        workdir = os.getenv("TB2_TASK_WORKDIR", "").strip()
        if not workdir:
            workdir = _task_image_workdir(task_dir)
            if workdir and _workdir_is_server_tree(workdir):
                workdir = ""
        if not (workdir and Path(workdir).is_dir()):
            workdir = str(task_dir)
        self._workdir = workdir
        # No install_dependencies: evaluation runs the task's canonical
        # tests/test.sh, which pins its own pytest toolchain via uvx (see
        # _evaluate_canonical), so the toolkit venv does not need pytest.
        self._terminal_toolkit = TerminalToolkit(
            timeout=self.command_timeout_s,
            working_directory=workdir,
            use_docker_backend=False,
            session_logs_dir=session_logs_dir,
            safe_mode=self.safe_mode,
        )

        self._state = Tbench2State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=resolved_task_id,
            task_path=str(task_dir),
            terminal_ready=True,
        )

        return Tbench2Observation(
            instruction=self._instruction,
            output="",
            success=True,
            error="",
            task_id=resolved_task_id,
            task_path=str(task_dir),
            session_id=None,
            action_type="reset",
            info={},
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: Tbench2Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        # [FUNCTION DOC]
        # Raises TypeError for a non-Tbench2Action and RuntimeError if not reset. Increments step_count and records
        # last action type/command. Dispatches on action_type: exec (shell_exec in session id, default
        # "tb2-session", with the configured timeout), write / view / wait / kill (need session_id, else error),
        # write_file (writes content to file_path), evaluate (scores; done True), close (closes; done True); other
        # types are errors. Any exception is caught and reported as success False with the message in error.
        # Records last output and session id in state.
        # Returns: observation with output, success, error, task info, session id, action type, info, reward
        # (only set by evaluate) and done.
        del timeout_s, kwargs

        if not isinstance(action, Tbench2Action):
            raise TypeError(f"Expected Tbench2Action, got {type(action)}")

        if self._terminal_toolkit is None or self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.last_command = action.command

        output = ""
        error = ""
        success = True
        reward = None
        done = False
        info: dict[str, Any] = {}
        session_id = action.session_id or "tb2-session"

        try:
            if action.action_type == "exec":
                # Pass the timeout explicitly: camel's shell_exec has its own
                # per-call default (20s) and ignores the constructor timeout,
                # which silently truncates any longer-running agent command.
                output = self._terminal_toolkit.shell_exec(
                    command=action.command,
                    block=action.block,
                    id=session_id,
                    timeout=self.command_timeout_s,
                )
            elif action.action_type == "write":
                self._ensure_session_id(action.session_id, action.action_type)
                output = self._terminal_toolkit.shell_write_to_process(
                    id=action.session_id,
                    command=action.command,
                )
            elif action.action_type == "view":
                self._ensure_session_id(action.session_id, action.action_type)
                output = self._terminal_toolkit.shell_view(id=action.session_id)
            elif action.action_type == "wait":
                self._ensure_session_id(action.session_id, action.action_type)
                wait_seconds = action.wait_seconds or 0.0
                output = self._terminal_toolkit.shell_wait(
                    id=action.session_id,
                    wait_seconds=wait_seconds,
                )
            elif action.action_type == "kill":
                self._ensure_session_id(action.session_id, action.action_type)
                self._terminal_toolkit.shell_kill_process(id=action.session_id)
                output = f"Killed session {action.session_id}"
            elif action.action_type == "write_file":
                self._terminal_toolkit.shell_write_content_to_file(
                    content=action.content,
                    file_path=action.file_path,
                )
                output = f"Wrote content to {action.file_path}"
            elif action.action_type == "evaluate":
                output, reward, info = self._evaluate_task()
                done = True
            elif action.action_type == "close":
                self.close()
                output = "Closed TB2 environment."
                done = True
            else:
                raise ValueError(f"Unsupported action_type: {action.action_type}")
        except Exception as exc:  # pragma: no cover
            success = False
            error = str(exc)

        self._state.last_output = output
        self._state.session_id = session_id or ""

        return Tbench2Observation(
            instruction=self._instruction,
            output=output,
            success=success,
            error=error,
            task_id=self._state.task_id,
            task_path=self._state.task_path,
            session_id=session_id or "",
            action_type=action.action_type,
            info=info,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> Tbench2State:
        # [FUNCTION DOC]
        # Property returning the current Tbench2State.
        return self._state

    def close(self) -> None:
        # [FUNCTION DOC]
        # Drops the toolkit, task dir and instruction (state is kept). Further steps raise RuntimeError.
        self._terminal_toolkit = None
        self._task_dir = None
        self._instruction = ""

    def _resolve_task_path(self, task_id: str | None, task_path: str | None) -> Path:
        # [FUNCTION DOC]
        # HELPER: returns an existing task directory: the expanded/resolved task_path if given, else task_id under
        # tasks_dir or under the downloaded repo (cache dir TB2_CACHE_DIR or output_dir/repo_cache).
        # Raises FileNotFoundError if the path does not exist, ValueError if neither id nor path is given.
        if task_path:
            resolved = Path(task_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Task path not found: {resolved}")
            return resolved

        if not task_id:
            raise ValueError("Provide task_id or task_path to reset TB2 environment.")

        if not self.tasks_dir:
            cache_dir = Path(
                os.getenv("TB2_CACHE_DIR", str(self.output_dir / "repo_cache"))
            )
            repo_dir = _download_tb2_repo(cache_dir)
            resolved = repo_dir / task_id
        else:
            resolved = Path(self.tasks_dir).expanduser().resolve() / task_id

        if not resolved.exists():
            raise FileNotFoundError(f"Task path not found: {resolved}")
        return resolved

    def _ensure_session_id(self, session_id: str | None, action_type: str) -> None:
        # [FUNCTION DOC]
        # HELPER: raises ValueError naming the action type if session_id is empty.
        if not session_id:
            raise ValueError(f"session_id is required for action_type='{action_type}'")

    # tests/ tarballs withheld out of agent reach at reset(), keyed by task
    # dir. Class-level: the server may construct one environment instance per
    # session, and a later reset of the same task must still find the copy
    # after the on-disk source is gone. In-process only — withhold mode
    # therefore assumes a single server process owns the task filesystem for
    # its lifetime (the intended per-task-sandbox deployment: one uvicorn
    # worker, one ephemeral sandbox per episode). It is not safe with
    # ``uvicorn --workers N`` sharing one on-disk checkout, where a sibling
    # worker would see tests/ already deleted with no cache to restore from.
    _WITHHELD_TESTS: dict[str, bytes] = {}
    _WITHHOLD_LOCK = threading.Lock()

    def _withhold_verifier_assets(self, task_dir: Path) -> None:
        # [FUNCTION DOC]
        # Under the class lock: caches tests/ (following a symlinked tests dir) as an in-memory tar keyed by the
        # task dir if not already cached, then deletes tests/ and solution/ from disk. Side effects: destructive
        # filesystem changes in the task dir; mutation of the class-level cache.
        """Move verifier assets out of the agent's reach before it can act.

        The task directory is agent-readable (same filesystem, root agent):
        tests/test_outputs.py spells out the expected outputs and solution/ is
        the literal answer — for RL training both are reward-hacking bait.
        tests/ is read into server memory (the staging source for
        _stage_tests_for_verify) and removed from disk; solution/ is simply
        removed (nothing in this env reads it — it exists only for oracle
        runs). Gated behind TB2_WITHHOLD_TESTS=1 because in plain local mode
        task_dir may be a developer's working checkout, where deleting from
        it would be hostile.
        """
        key = str(task_dir)
        with self._WITHHOLD_LOCK:
            tests_dir = task_dir / "tests"
            if tests_dir.is_dir():
                if key not in self._WITHHELD_TESTS:
                    # realpath so a symlinked tests/ caches the target's files,
                    # not just a link member that would restore empty (tar.add
                    # does not follow a top-level symlink).
                    tests_real = os.path.realpath(tests_dir)
                    buf = io.BytesIO()
                    with tarfile.open(fileobj=buf, mode="w") as tar:
                        tar.add(tests_real, arcname=".")
                    self._WITHHELD_TESTS[key] = buf.getvalue()
            _hard_remove(str(tests_dir))
            _hard_remove(str(task_dir / "solution"))

    def _stage_tests_for_verify(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: wipes and recreates /logs/verifier, wipes /tests, then fills /tests from the cached tarball if
        # present, else by copying the task's tests/ dir (symlinks preserved). Returns True if tests were staged,
        # False if the task has no tests. Caller must hold the evaluation lock.
        """Stage a pristine tests/ copy at the official fixed path for exactly
        the verify window. Caller must hold _CANONICAL_EVAL_LOCK.

        Both fixed paths are recreated from scratch (symlink included — see
        _hard_remove) so nothing an agent may have pre-planted survives into
        scoring: a /tests/conftest.py pytest would auto-load, or a symlinked
        /tests or /logs/verifier redirecting the stage at a dir the agent
        controls. When the source dir was withheld at reset, the staged copy
        comes from server memory: a copy the agent never had filesystem
        access to.
        """
        _hard_remove(_VERIFY_TESTS_DIR)
        _hard_remove(_VERIFIER_LOG_DIR)
        Path(_VERIFIER_LOG_DIR).mkdir(parents=True, exist_ok=True)

        blob = self._WITHHELD_TESTS.get(str(self._task_dir))
        if blob is not None:
            Path(_VERIFY_TESTS_DIR).mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
                _extractall(tar, _VERIFY_TESTS_DIR)
            return True
        tests_dir = Path(self._task_dir) / "tests"
        if not tests_dir.is_dir():
            return False
        shutil.copytree(tests_dir, _VERIFY_TESTS_DIR, symlinks=True)
        return True

    def _evaluate_task(self) -> tuple[str, float, dict[str, Any]]:
        # [FUNCTION DOC]
        # Raises RuntimeError if not reset. Reads the verifier timeout (default 900 s). Under the class evaluation
        # lock: stages tests; if none, returns a 0.0 result with a "missing tests" error in info; uses the canonical
        # harness when test.sh exists, else the pytest fallback. Always removes /tests and /logs/verifier afterwards.
        # Returns: (output text, reward, info mapping).
        if self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")
        if self._terminal_toolkit is None:
            raise RuntimeError("Terminal toolkit not initialized.")

        # The task's own verifier budget (task.toml [verifier].timeout_sec) —
        # heavy tests legitimately run minutes (circuit-fibsqrt declares 3600s).
        verifier_timeout_s = _read_timeout(self._task_dir, fallback=900.0)

        with self._CANONICAL_EVAL_LOCK:
            try:
                # Staging happens inside the try: a stage that errors partway
                # (I/O failure mid-extract) must be wiped like a completed
                # one — step() reports scoring errors without ending the
                # episode, so the still-live session must not find a partial
                # tests/ copy left behind.
                if not self._stage_tests_for_verify():
                    return (
                        f"no tests/ found for task {self._state.task_id}",
                        0.0,
                        {"tests_passed": False, "error": "missing tests"},
                    )
                if (Path(_VERIFY_TESTS_DIR) / "test.sh").is_file():
                    return self._evaluate_canonical(verifier_timeout_s)
                return self._evaluate_fallback(verifier_timeout_s)
            finally:
                # Verifier artifacts live on disk only for the verify window:
                # don't leave them readable to a later episode / concurrent
                # session sharing this filesystem — or to the agent itself if
                # scoring times out / errors (done stays false, so it keeps
                # its session). /logs/verifier holds reward.txt and the
                # test.sh log, whose pytest -rA output can reveal expected
                # values.
                _hard_remove(_VERIFY_TESTS_DIR)
                _hard_remove(_VERIFIER_LOG_DIR)

    # The canonical harness uses the official fixed paths (/tests,
    # /logs/verifier/reward.txt), which are per-container, not per-session.
    # Serialize staging + evaluation so concurrent sessions on one server
    # cannot overwrite each other's staged tests or read another session's
    # reward.
    _CANONICAL_EVAL_LOCK = threading.Lock()

    def _evaluate_canonical(
        self, timeout_s: float
    ) -> tuple[str, float, dict[str, Any]]:
        # [FUNCTION DOC]
        # Runs the canonical command from the agent workdir in toolkit session "tb2-tests" with the verifier
        # timeout. Returns (output, reward, info with tests_passed = reward == 1.0 and harness name).
        # Raises RuntimeError if no verdict could be parsed.
        """Score via the task's OFFICIAL harness, exactly as the TB2 verifier does.

        /tests/test.sh (staged by _stage_tests_for_verify) pins its own pytest
        toolchain (uvx: Python 3.13 + pytest 8.4.1 + ctrf), runs
        tests/test_outputs.py from /tests against the task workdir, and writes
        the binary result to /logs/verifier/reward.txt. Bare ``pytest tests/``
        (the fallback below) skips that toolchain pinning and mis-scores any
        task whose tests need it, so the canonical harness is preferred
        whenever the task ships it.
        """
        # Same working dir the agent operated in (resolved during reset).
        # The toolkit enforces the verifier budget itself, so the command
        # carries no in-shell timeout (see _canonical_eval_cmd).
        workdir = self._workdir or str(self._task_dir)
        output = self._terminal_toolkit.shell_exec(
            id="tb2-tests",
            command=_canonical_eval_cmd(workdir),
            block=True,
            timeout=timeout_s,
        )

        reward = _require_canonical_verdict(_parse_canonical_reward(output), output)
        info = {"tests_passed": reward == 1.0, "harness": "tests/test.sh"}
        return output, reward, info

    def _evaluate_fallback(self, timeout_s: float) -> tuple[str, float, dict[str, Any]]:
        # [FUNCTION DOC]
        # Runs the pytest fallback command from the agent workdir with the verifier timeout. Reward is 1.0 when the
        # parsed exit code is 0, else 0.0. Returns (output, reward, info with tests_passed and exit_code).
        """Scoring for task dirs without the canonical harness (see
        _fallback_eval_cmd)."""
        fallback_cwd = self._workdir or str(self._task_dir)
        output = self._terminal_toolkit.shell_exec(
            id="tb2-tests",
            command=_fallback_eval_cmd(fallback_cwd),
            block=True,
            timeout=timeout_s,
        )

        exit_code = _parse_exit_code_marker(output)
        reward = 1.0 if exit_code == 0 else 0.0
        info = {"tests_passed": exit_code == 0, "exit_code": exit_code}
        return output, reward, info


class Tbench2DockerEnvironment(
    Environment[Tbench2Action, Tbench2Observation, Tbench2State]
):
    # [CLASS DOC]
    # Docker-isolated TB2 environment: one container per episode from the task's official image (no host
    # fallback), with the task source copied to /task excluding tests/ and solution/, commands run in the
    # image's WORKDIR, and canonical scoring inside the container. Supports concurrent sessions (each session
    # has its own container).
    """OpenEnv wrapper around Terminal-Bench 2 tasks with Docker isolation.

    This environment runs each task in its own Docker container built from
    the task's official image (task.toml's [environment] docker_image — tasks
    without one are rejected at reset; there is no host-execution fallback).
    Agent commands run in the image's own WORKDIR, and ``evaluate`` scores
    with the task's canonical tests/test.sh staged at the official fixed
    paths for exactly the verify window — the same scoring contract as the
    local mode, just delivered over container exec instead of the in-process
    terminal toolkit.

    Requires:
    - Docker socket mounted (/var/run/docker.sock)
    - Sufficient disk space for container images
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_dir: str | None = None,
        output_dir: str | None = None,
        command_timeout_s: float = 300.0,
        safe_mode: bool = True,
        default_task_id: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores settings: tasks_dir (TB2_TASKS_DIR), output_dir (TB2_OUTPUT_DIR), command_timeout_s (default 300),
        # safe_mode (default True), default task id (TB2_DEFAULT_TASK_ID or "headless-terminal"). No I/O.
        # KNOWN ISSUE: command_timeout_s and safe_mode are stored but never used; container execs have no timeout.
        super().__init__()
        self.tasks_dir = tasks_dir or os.getenv("TB2_TASKS_DIR", "")
        self.output_dir = Path(
            output_dir or os.getenv("TB2_OUTPUT_DIR", "/tmp/tbench2_env_runs")
        )
        self.command_timeout_s = command_timeout_s
        self.safe_mode = safe_mode
        self.default_task_id = default_task_id or os.getenv(
            "TB2_DEFAULT_TASK_ID", "headless-terminal"
        )

        self._state = Tbench2State()
        self._task_dir: Path | None = None
        self._docker_client = None
        self._container = None
        self._instruction = ""
        self._task_image = ""
        self._task_config: dict[str, Any] = {}
        self._workdir = ""

    def _get_docker_client(self) -> Any:
        # [FUNCTION DOC]
        # HELPER: lazily creates and caches a docker client from the environment. Raises RuntimeError if Docker
        # is unavailable.
        """Lazy initialization of Docker client."""
        if self._docker_client is None:
            try:
                import docker

                self._docker_client = docker.from_env()
            except Exception as exc:
                raise RuntimeError(
                    f"Docker client not available. Ensure Docker socket is mounted. Error: {exc}"
                ) from exc
        return self._docker_client

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        # [FUNCTION DOC]
        # Seed ignored. Resolves the task dir like local mode, reads task.toml for [environment] docker_image and
        # the instruction. If no image is declared, closes any previous container and raises RuntimeError. Creates
        # the trial dir, closes the previous episode, stores the new task metadata and starts a container (closing
        # and re-raising on failure). Returns observation with instruction, action_type "reset", reward 0.0, done
        # False and info containing docker_image.
        # KNOWN ISSUE: episode_id is only used for the trial dir name; the state gets a fresh random id and its
        # task_id is the directory name rather than the requested task id.
        del seed

        task_id = (
            kwargs.get("task_id") or kwargs.get("task_name") or self.default_task_id
        )
        task_path = kwargs.get("task_path") or kwargs.get("path")

        task_dir = self._resolve_task_path(task_id, task_path)
        resolved_task_id = task_id or task_dir.name

        # Read task configuration including Docker image
        task_toml_path = task_dir / "task.toml"
        if task_toml_path.exists():
            task_config = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
            task_image = task_config.get("environment", {}).get("docker_image", "")
        else:
            task_image = ""
            task_config = {}

        instruction = _read_instruction(task_dir)

        # No host-execution fallback: silently running agent commands on the
        # env-server host would be a containment hole (arbitrary agent shell
        # outside any container) and a scoring-fidelity hole (no official
        # image, no canonical harness) at once. Every official TB2 task
        # declares its image; a task dir that doesn't is a bug to surface.
        if not task_image:
            # Fail closed: a rejected reset must not leave a previous
            # container usable with metadata from this new task.
            self.close()
            raise RuntimeError(
                f"task {resolved_task_id} declares no [environment] docker_image "
                "in task.toml; Docker mode runs every episode in the task's "
                "official container and does not fall back to executing on the "
                "server host. Fix the task, or use TB2_MODE=local."
            )

        # Create trial directory for logs
        trial_name = f"{resolved_task_id}.{episode_id or uuid4().hex}"
        trial_dir = self.output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        # A successful reset replaces the previous episode. Commit the new
        # metadata only after all task inputs have been validated, and leave
        # the session closed if container startup fails.
        self.close()
        self._task_config = task_config
        self._task_image = task_image
        self._instruction = instruction
        self._task_dir = task_dir
        try:
            self._start_container(task_dir, trial_dir)
        except Exception:
            self.close()
            raise

        return Tbench2Observation(
            instruction=self._instruction,
            output="",
            success=True,
            error="",
            task_id=resolved_task_id,
            task_path=str(task_dir),
            session_id=None,
            action_type="reset",
            info={"docker_image": self._task_image},
            reward=0.0,
            done=False,
        )

    def _start_container(self, task_dir: Path, trial_dir: Path) -> None:
        # [FUNCTION DOC]
        # HELPER: gets or pulls the image, resolves the workdir, runs a detached "sleep infinity" container with host
        # networking, copies the task dir to /task excluding tests/ and solution/, and resets the state.
        # Raises RuntimeError wrapping any failure. Side effects: Docker image pull and container creation.
        """Start a Docker container for the task.

        Uses file copying instead of bind mounts to support Docker-in-Docker
        scenarios where the server runs inside a container. Bind mounts reference
        host paths, which don't exist when the server is containerized.
        """
        docker = self._get_docker_client()

        try:
            # Pull image if needed
            try:
                image = docker.images.get(self._task_image)
            except Exception:
                logging.info(f"Pulling image {self._task_image}...")
                image = docker.images.pull(self._task_image)
                if isinstance(image, list):  # pull without a tag returns a list
                    image = image[0]

            self._workdir = self._resolve_workdir(image, task_dir)

            # Start container WITHOUT bind mounts (for DinD compatibility).
            # working_dir=/task only guarantees the task-source copy target
            # exists; every exec cd's into the resolved image WORKDIR.
            self._container = docker.containers.run(
                image=self._task_image,
                command="sleep infinity",
                detach=True,
                network_mode="host",
                working_dir="/task",
                remove=False,
            )

            # Copy task files into container using tar archive
            # This works in Docker-in-Docker because we read files from our
            # filesystem and stream them to the container via the Docker API.
            # Verifier assets stay out: the agent works in /task and must not
            # be able to read solution/ (the literal answer) or tests/ (the
            # expected outputs) — tests are staged only at verify time by
            # _evaluate_docker, like the official TB2 harness does.
            self._copy_dir_to_container(
                task_dir, "/task", exclude_top=("solution", "tests")
            )

            self._state = Tbench2State(
                episode_id=str(uuid4()),
                step_count=0,
                task_id=task_dir.name,
                task_path=str(task_dir),
                terminal_ready=True,
            )

        except Exception as exc:
            raise RuntimeError(f"Failed to start container: {exc}") from exc

    def _copy_dir_to_container(
        self, src_dir: Path, dest_path: str, exclude_top: tuple[str, ...] = ()
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: builds an in-memory tar of every descendant of src_dir (non-recursive per entry, skipping top-level
        # names in exclude_top) and puts it into the container at dest_path. Raises RuntimeError if no container.
        """Copy a directory into the container using tar archive.

        This method streams files via the Docker API, avoiding bind mount
        issues in Docker-in-Docker scenarios. Top-level entries named in
        *exclude_top* are skipped entirely.
        """
        if self._container is None:
            raise RuntimeError("Container not started")

        # Create tar archive in memory (recursive=False: rglob already yields
        # every descendant, so per-entry recursion would duplicate members —
        # and would re-add excluded subtrees through their parents).
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for item in src_dir.rglob("*"):
                rel = item.relative_to(src_dir)
                if rel.parts and rel.parts[0] in exclude_top:
                    continue
                tar.add(str(item), arcname=str(rel), recursive=False)

        tar_stream.seek(0)

        # Copy to container
        self._container.put_archive(dest_path, tar_stream.getvalue())

    def _resolve_workdir(self, image: Any, task_dir: Path) -> str:
        # [FUNCTION DOC]
        # HELPER: returns the image config's WorkingDir, else the Dockerfile WORKDIR, else "/task".
        """The dir agent commands and scoring run in: the task image's own
        WORKDIR — the real TB2 task state, not the /task source copy.

        Image metadata is authoritative (it sees a WORKDIR inherited from a
        base image, which Dockerfile parsing misses); fall back to parsing
        the task's Dockerfile, then to /task. The local mode's server-tree
        guard does not apply here: this server sits outside the container,
        so the image's /app is always the task tree.
        """
        try:
            workdir = (image.attrs.get("Config") or {}).get("WorkingDir") or ""
        except Exception:
            workdir = ""
        return workdir or _task_image_workdir(task_dir) or "/task"

    def _exec_in_container(
        self, command: str, workdir: str | None = None
    ) -> tuple[int, str]:
        # [FUNCTION DOC]
        # HELPER: runs "cd <quoted cwd> && <command>" via bash -c in the container (cwd = given workdir, else the
        # resolved workdir, else /task), capturing stdout and stderr. Returns (exit code, decoded output).
        # Raises RuntimeError if no container.
        """Execute a command inside the container, from the task workdir.

        The command rides as an argv element (bash -c <command>), not spliced
        into a quoted shell string — an agent command containing a single
        quote must arrive in the container byte-identical.
        """
        if self._container is None:
            raise RuntimeError("Container not started. Call reset() first.")

        cwd = workdir or self._workdir or "/task"
        exit_code, output = self._container.exec_run(
            cmd=["bash", "-c", f"cd {shlex.quote(cwd)} && {command}"],
            stdout=True,
            stderr=True,
        )
        return exit_code, output.decode("utf-8", errors="replace")

    def step(
        self,
        action: Tbench2Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        # [FUNCTION DOC]
        # Raises TypeError for a wrong action type and RuntimeError if not reset. Updates step count and last
        # action fields. Supports exec (success = exit code 0), write_file (heredoc into file_path), evaluate
        # (done True), close (done True); other types are errors. Exceptions are caught into success False / error.
        # Returns an observation shaped like local mode.
        # KNOWN ISSUE: write_file interpolates file_path unquoted into the shell and uses a fixed 'EOF' heredoc
        # delimiter, so paths with spaces or content containing a line "EOF" break the write.
        del timeout_s, kwargs

        if not isinstance(action, Tbench2Action):
            raise TypeError(f"Expected Tbench2Action, got {type(action)}")

        if self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.last_command = action.command

        output = ""
        error = ""
        success = True
        reward = None
        done = False
        info: dict[str, Any] = {}
        session_id = action.session_id or "tb2-session"

        try:
            if action.action_type == "exec":
                exit_code, output = self._exec_in_container(action.command)
                success = exit_code == 0

            elif action.action_type == "write_file":
                exit_code, _ = self._exec_in_container(
                    f"cat > {action.file_path} << 'EOF'\n{action.content}\nEOF"
                )
                success = exit_code == 0
                output = f"Wrote to {action.file_path}"

            elif action.action_type == "evaluate":
                output, reward, info = self._evaluate_docker()
                done = True

            elif action.action_type == "close":
                self.close()
                output = "Closed TB2 environment."
                done = True

            else:
                raise ValueError(
                    f"Unsupported action_type in Docker mode: {action.action_type}"
                )

        except Exception as exc:
            success = False
            error = str(exc)

        self._state.last_output = output
        self._state.session_id = session_id or ""

        return Tbench2Observation(
            instruction=self._instruction,
            output=output,
            success=success,
            error=error,
            task_id=self._state.task_id,
            task_path=self._state.task_path,
            session_id=session_id or "",
            action_type=action.action_type,
            info=info,
            reward=reward,
            done=done,
        )

    def _evaluate_docker(self) -> tuple[str, float, dict[str, Any]]:
        # [FUNCTION DOC]
        # Raises RuntimeError if no container (asserts task dir). Returns a 0.0 "missing tests" result if the task
        # has no tests/. Otherwise wipes and recreates /tests and /logs/verifier in the container (RuntimeError on
        # failure), copies tests in, and runs the canonical command with the verifier timeout (or the pytest
        # fallback). Always tries to remove the staged dirs afterwards, logging warnings on failure.
        # Returns: (output, reward, info) like local mode; RuntimeError if the canonical harness gives no verdict.
        """Score with the canonical harness inside the task container.

        Same scoring contract as the local mode's _evaluate_task: stage a
        pristine tests/ copy at the official fixed path for exactly the
        verify window, run the task's tests/test.sh from the dir the agent
        worked in (the image WORKDIR), read the binary verdict from
        /logs/verifier/reward.txt; task dirs without test.sh fall back to
        pytest. No cross-session lock is needed here, unlike local mode:
        every session drives its OWN container, so the fixed paths are
        session-private.
        """
        if self._container is None:
            raise RuntimeError("Container not started.")
        assert self._task_dir is not None, "Task directory not set"

        # Stage-at-verify, like the official TB2 harness: the initial /task
        # copy excludes tests/, and this server sits OUTSIDE the container, so
        # the copy staged here is one the agent never saw. rm -rf first so
        # nothing an agent pre-planted at the fixed paths survives into
        # scoring — a /tests/conftest.py pytest would auto-load, a symlink
        # redirecting the stage at a dir the agent controls, or a stale
        # /logs/verifier/reward.txt read back as a verdict.
        tests_src = self._task_dir / "tests"
        if not tests_src.is_dir():
            return (
                f"no tests/ found for task {self._state.task_id}",
                0.0,
                {"tests_passed": False, "error": "missing tests"},
            )

        # The task's own verifier budget (task.toml [verifier].timeout_sec) —
        # heavy tests legitimately run minutes (circuit-fibsqrt declares 3600s).
        verifier_timeout_s = _read_timeout(self._task_dir, fallback=900.0)
        workdir = self._workdir or "/task"

        wipe_ec, wipe_out = self._exec_in_container(
            f"rm -rf {_VERIFY_TESTS_DIR} {_VERIFIER_LOG_DIR} && "
            f"mkdir -p {_VERIFY_TESTS_DIR} {_VERIFIER_LOG_DIR}"
        )
        if wipe_ec != 0:
            # Fail closed: put_archive into a dir that survived the wipe would
            # merge the staged tests into whatever the agent planted there.
            raise RuntimeError(
                f"could not reset {_VERIFY_TESTS_DIR} before verify: {wipe_out.strip()}"
            )
        try:
            self._copy_dir_to_container(tests_src, _VERIFY_TESTS_DIR)

            if (tests_src / "test.sh").is_file():
                _, output = self._exec_in_container(
                    _canonical_eval_cmd(workdir, timeout_s=verifier_timeout_s)
                )
                reward = _require_canonical_verdict(
                    _parse_canonical_reward(output), output
                )
                info = {"tests_passed": reward == 1.0, "harness": "tests/test.sh"}
            else:
                _, output = self._exec_in_container(_fallback_eval_cmd(workdir))
                exit_code = _parse_exit_code_marker(output)
                reward = 1.0 if exit_code == 0 else 0.0
                info = {"tests_passed": exit_code == 0, "exit_code": exit_code}
        finally:
            # Verifier artifacts live in the container only for the verify
            # window, matching the local-mode staging: neither the staged
            # tests (expected values) nor /logs/verifier (reward + pytest -rA
            # log) may stay readable afterward — the agent keeps its session
            # if scoring errored (step() reports the failure without ending
            # the episode).
            try:
                rm_ec, rm_out = self._exec_in_container(
                    f"rm -rf {_VERIFY_TESTS_DIR} {_VERIFIER_LOG_DIR}"
                )
                if rm_ec != 0:
                    logging.warning(
                        "failed to remove staged %s after verify: %s",
                        _VERIFY_TESTS_DIR,
                        rm_out.strip(),
                    )
            except Exception:
                logging.warning(
                    "failed to remove staged %s after verify",
                    _VERIFY_TESTS_DIR,
                    exc_info=True,
                )

        return output, reward, info

    @property
    def state(self) -> Tbench2State:
        # [FUNCTION DOC]
        # Property returning the current Tbench2State.
        return self._state

    def close(self) -> None:
        # [FUNCTION DOC]
        # Stops (10 s timeout) and force-removes the container, ignoring errors, and clears task dir, instruction
        # and workdir.
        if self._container:
            try:
                self._container.stop(timeout=10)
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None
        self._task_dir = None
        self._instruction = ""
        self._workdir = ""

    def _resolve_task_path(self, task_id: str | None, task_path: str | None) -> Path:
        # [FUNCTION DOC]
        # HELPER: same task-directory resolution as the local environment (explicit path, tasks_dir, or downloaded
        # repo); raises FileNotFoundError or ValueError.
        if task_path:
            resolved = Path(task_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Task path not found: {resolved}")
            return resolved

        if not task_id:
            raise ValueError("Provide task_id or task_path to reset TB2 environment.")

        if not self.tasks_dir:
            cache_dir = Path(
                os.getenv("TB2_CACHE_DIR", str(self.output_dir / "repo_cache"))
            )
            repo_dir = _download_tb2_repo(cache_dir)
            resolved = repo_dir / task_id
        else:
            resolved = Path(self.tasks_dir).expanduser().resolve() / task_id

        if not resolved.exists():
            raise FileNotFoundError(f"Task path not found: {resolved}")
        return resolved
