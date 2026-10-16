# [MODULE DOC]
# HELPER: thin wrapper around the E2B Code Interpreter sandbox for running shell commands. The e2b import is
# optional; its failure is remembered and reported when a sandbox is created.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small E2B Code Interpreter wrapper for terminal-style environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_E2B_IMPORT_ERROR: ImportError | None = None

try:
    from e2b_code_interpreter import Sandbox
except ImportError as _e2b_import_error:  # pragma: no cover
    _E2B_IMPORT_ERROR = _e2b_import_error
    Sandbox = None  # type: ignore[assignment]


@dataclass
class ShellResult:
    # [CLASS DOC]
    # Dataclass with a command's stdout, stderr, optional error description and success flag.
    """Normalized result from a command executed in E2B."""

    stdout: str
    stderr: str
    error: str | None
    success: bool


class E2BSandbox:
    # [CLASS DOC]
    # One E2B sandbox for one episode; exposes its sandbox_id.
    """Manages one E2B sandbox for one OpenEnv episode."""

    def __init__(self, api_key: str):
        # [FUNCTION DOC]
        # Raises ImportError if e2b-code-interpreter is not installed. Creates a remote sandbox with api_key and
        # stores its id. Side effects: network call creating a billable sandbox.
        if Sandbox is None:
            raise ImportError(
                "e2b-code-interpreter is not installed. Install the "
                "terminus_env package dependencies to use E2BSandbox. "
                f"Original import error: {_E2B_IMPORT_ERROR}"
            )
        self._sbx = Sandbox.create(api_key=api_key)
        self.sandbox_id: str = self._sbx.sandbox_id

    def run_shell(self, command: str, timeout_s: int = 120) -> ShellResult:
        # [FUNCTION DOC]
        # Runs command via Python code executed in the sandbox: subprocess.run with shell=True, captured output and
        # a timeout of timeout_s seconds (default 120); prints stdout/stderr and exits with the return code when
        # non-zero. Returns the normalised ShellResult (a non-zero exit or timeout shows up as an execution error).
        shell_code = (
            "import subprocess, sys\n"
            f"_result = subprocess.run({command!r}, shell=True, capture_output=True, text=True, timeout={timeout_s})\n"
            "print(_result.stdout, end='')\n"
            "if _result.stderr: print(_result.stderr, end='', file=sys.stderr)\n"
            "if _result.returncode != 0:\n"
            "    raise SystemExit(_result.returncode)\n"
        )
        execution = self._sbx.run_code(shell_code)
        return _normalize(execution)

    def kill(self) -> None:
        # [FUNCTION DOC]
        # Kills the sandbox; on failure tries close(); all exceptions are swallowed.
        try:
            self._sbx.kill()
        except Exception:
            try:
                self._sbx.close()
            except Exception:
                pass


def _normalize(execution: Any) -> ShellResult:
    # [FUNCTION DOC]
    # HELPER: converts an E2B execution result into a ShellResult: stdout/stderr log chunks joined by newlines,
    # error formatted as "name: value" plus traceback, success when there is no error.
    stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
    stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""
    error = None
    if execution.error:
        error = (
            f"{execution.error.name}: {execution.error.value}\n"
            f"{execution.error.traceback}"
        )
    return ShellResult(
        stdout=stdout,
        stderr=stderr,
        error=error,
        success=execution.error is None,
    )
