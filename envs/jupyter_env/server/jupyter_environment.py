# [MODULE DOC]
# Server-side Jupyter MCP environment. Each episode owns one E2B sandbox
# (persistent Python kernel). Agents act only through FastMCP tools: run a code
# cell, replace and rerun the last cell, run a shell command, get a notebook
# summary, and submit a final answer. Optional per-episode setup commands run at
# reset and verify commands run on final_answer; the reward is either a number
# written by verify commands to REWARD_FILE (/home/user/logs/verifier/reward.txt)
# or the verify pass rate.
"""Jupyter MCP environment.

Exposes 4 notebook interaction tools via FastMCP:
  1. add_and_execute_code_cell    - primary execution tool
  2. edit_and_execute_current_cell - error-recovery tool
  3. execute_shell_command         - shell access inside the sandbox
  4. get_notebook_state            - compact history for agent memory

Each episode (reset → step* → [reset]) maps to exactly one E2B sandbox,
giving clean variable scope boundaries for RL training loops.
"""

import logging
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

try:
    from .e2b_sandbox import E2BSandbox
    from .notebook_tracker import NotebookTracker
except ImportError:  # pragma: no cover
    from server.e2b_sandbox import E2BSandbox
    from server.notebook_tracker import NotebookTracker

log = logging.getLogger(__name__)
REWARD_FILE = "/home/user/logs/verifier/reward.txt"
_CLEAR_REWARD_FILE = f"rm -f {REWARD_FILE} && test ! -e {REWARD_FILE}"


class JupyterEnvironment(MCPEnvironment):
    # [CLASS DOC]
    # MCPEnvironment holding the sandbox (None until reset), a NotebookTracker, the
    # model classes, a JupyterState and the submitted answer. Supports concurrent
    # sessions (one instance per session).
    """
    Stateful Jupyter notebook environment backed by an E2B Code Interpreter sandbox.

    Inherits from MCPEnvironment which auto-routes ListToolsAction and
    CallToolAction to the registered FastMCP tools. Only non-MCP actions fall
    through to ``_step_impl``.

    Concurrent sessions: each WebSocket connection gets its own instance
    (``SUPPORTS_CONCURRENT_SESSIONS = True``), so each agent has an isolated
    sandbox.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes tracker and state (new UUID), imports the model classes (package or
        # flat layout), registers the five MCP tools as closures over self, and passes
        # the FastMCP server to MCPEnvironment. No sandbox is created yet.
        self._sandbox: Optional[E2BSandbox] = None
        self._tracker = NotebookTracker()

        # Import here to avoid circular imports at module load time
        # Support both package and flat-layout execution
        try:
            from ..models import JupyterState, NotebookCell
        except ImportError:
            from models import JupyterState, NotebookCell

        self._JupyterState = JupyterState
        self._NotebookCell = NotebookCell
        self._state = JupyterState(episode_id=str(uuid4()))
        self._submitted_answer = None

        # ── Register MCP tools ──────────────────────────────────────────────
        mcp = FastMCP("jupyter_env")

        @mcp.tool
        def add_and_execute_code_cell(code: str) -> str:
            # [FUNCTION DOC]
            # MCP tool. Runs code in the sandbox, appends a tracker cell and a matching
            # NotebookCell to state, updates last_cell_success, and returns the
            # LLM-formatted output. Returns an error string if reset() has not created a
            # sandbox.
            """
            Execute Python code in the stateful Jupyter notebook.

            Variables, imports, and side-effects persist between calls within
            the same episode. Use this as the primary tool for all computation.

            Args:
                code: Python code to execute. Can span multiple lines.

            Returns:
                Stdout, expression results, and a note if images were generated.
                On error, returns the full traceback.
            """
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            result = self._sandbox.run_code(code)
            cell = self._tracker.add_code_cell(code, result)
            self._state.cells.append(self._NotebookCell(**_cell_to_model_kwargs(cell)))
            self._state.last_cell_success = result.success
            return _format_for_llm(result)

        @mcp.tool
        def edit_and_execute_current_cell(code: str) -> str:
            # [FUNCTION DOC]
            # MCP tool. Runs code, replaces the last tracker cell and the last state cell
            # with the new one, updates last_cell_success, and returns formatted output.
            # Returns an error string without a sandbox.
            # KNOWN ISSUE: the "last cell" replaced may be a shell cell, not a code cell.
            """
            Replace the last code cell with new code and re-execute it.

            Use this to fix errors in the previous cell instead of creating a
            new cell. This keeps the notebook clean.

            Args:
                code: Replacement Python code for the current cell.

            Returns:
                Same format as add_and_execute_code_cell.
            """
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            result = self._sandbox.run_code(code)
            cell = self._tracker.update_last_code_cell(code, result)
            if self._state.cells:
                self._state.cells.pop()
            self._state.cells.append(self._NotebookCell(**_cell_to_model_kwargs(cell)))
            self._state.last_cell_success = result.success
            return _format_for_llm(result)

        @mcp.tool
        def execute_shell_command(command: str) -> str:
            # [FUNCTION DOC]
            # MCP tool. Runs the shell command in the sandbox, appends a shell cell to the
            # tracker and state, and returns formatted output (last_cell_success is not
            # updated). Returns an error string without a sandbox.
            """
            Run a shell command inside the sandbox.

            Useful for package installation, file system inspection, or
            running scripts. Examples: "pip install polars", "ls -la", "cat data.csv".

            Args:
                command: Shell command string to execute.

            Returns:
                Combined stdout and stderr. On error, includes traceback.
            """
            if not self._sandbox:
                return "Error: environment not reset."
            result = self._sandbox.run_shell(command)
            cell = self._tracker.add_shell_cell(command, result)
            self._state.cells.append(self._NotebookCell(**_cell_to_model_kwargs(cell)))
            return _format_for_llm(result)

        @mcp.tool
        def get_notebook_state(include_images: bool = False) -> str:
            # [FUNCTION DOC]
            # MCP tool. Returns the tracker's summary of the last 10 cells, optionally with
            # inline base64 images.
            """
            Return a compact summary of all executed cells and their outputs.

            Useful at the start of a task (to check what has already been done)
            or when context about previous computations is needed.

            Args:
                include_images: If True, include base64-encoded PNG image data
                    inline (for multimodal models). If False (default), only
                    note that images were generated (for text-only models).

            Returns:
                Text summary of the last 10 cells with truncated outputs.
            """
            return self._tracker.get_state_summary(include_images=include_images)

        @mcp.tool
        def final_answer(answer: str) -> str:
            # [FUNCTION DOC]
            # MCP tool. Records the answer in the instance and state. With no verify
            # commands returns "Answer submitted: ..."; otherwise runs verification and
            # also reports passed/total and the reward.
            # KNOWN ISSUE: the docstring says this ends the task, but without verify
            # commands (or with no sandbox) last_reward stays None and step() does not mark
            # the episode done.
            """
            Submit your final answer to the question.

            Call this when you have computed the answer and are ready to submit.
            This ends the current task.

            Args:
                answer: Your final answer as a string.

            Returns:
                Confirmation that the answer was submitted.
            """
            self._submitted_answer = answer
            self._state.submitted_answer = answer
            if not self._state.verify_commands:
                return f"Answer submitted: {answer}"
            summary = self._run_verify_commands()
            return (
                f"Answer submitted: {answer}\n"
                f"Verification: {summary['passed']}/{summary['total']} passed; "
                f"reward={summary['reward']}"
            )

        super().__init__(mcp)

    # ── OpenEnv lifecycle ───────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        # [FUNCTION DOC]
        # Kills any existing sandbox, clears the tracker and submitted answer, and
        # creates a fresh sandbox with E2B_API_KEY. Missing key or sandbox creation
        # failure returns a done observation with error metadata (state reset, no
        # sandbox). Otherwise creates a new state with the sandbox id, reads setup
        # commands (kwarg "setup" or "setup_scripts") and verify commands ("verify" or
        # "verify_scripts"), stores the verify commands, runs setup commands (after
        # creating the verifier log dir) and returns a done error observation with all
        # setup results if any failed. If kaggle_dataset_name and files kwargs are given
        # and the dataset dir (KAGGLE_DATA_DIR or /fsx/<USER>/data/kaggle-data-10000)
        # exists, uploads each file (exact name match via recursive search, else
        # case-insensitive) to /home/user/input/ (upload errors logged). Returns a
        # "ready" observation with message, sandbox id, files loaded, setup results and
        # verify commands. seed is ignored.
        # Side effects: remote sandbox creation/destruction, file uploads.
        """
        Start a new episode.

        Kills any existing E2B sandbox and creates a fresh one.
        If setup commands are provided, they run after sandbox creation. If
        verify commands are provided, they are stored and executed when
        final_answer is called. If kaggle_dataset_name and files are provided,
        loads CSV files into /home/user/input/ in the sandbox.
        """
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

        self._tracker.reset()
        self._submitted_answer = None

        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            self._state = self._JupyterState(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
            )
            return Observation(
                done=True,
                reward=None,
                metadata={
                    "status": "error",
                    "error": (
                        "E2B_API_KEY is not set. Configure it before resetting "
                        "jupyter_env."
                    ),
                },
            )

        try:
            self._sandbox = E2BSandbox(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            self._state = self._JupyterState(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
            )
            return Observation(
                done=True,
                reward=None,
                metadata={
                    "status": "error",
                    "error": f"failed to create E2B sandbox: {type(exc).__name__}: {exc}",
                },
            )

        self._state = self._JupyterState(
            episode_id=episode_id or str(uuid4()),
            sandbox_id=self._sandbox.sandbox_id,
            step_count=0,
        )

        setup_commands = _coerce_commands(
            kwargs.get("setup", kwargs.get("setup_scripts", []))
        )
        verify_commands = _coerce_commands(
            kwargs.get("verify", kwargs.get("verify_scripts", []))
        )
        self._state.verify_commands = verify_commands

        if setup_commands:
            self._sandbox.run_shell("mkdir -p /home/user/logs/verifier")
            setup_results = self._run_shell_commands(setup_commands)
            self._state.setup_results = setup_results
            failed = [r for r in setup_results if not r.success]
            if failed:
                return Observation(
                    done=True,
                    reward=None,
                    metadata={
                        "status": "error",
                        "sandbox_id": self._state.sandbox_id,
                        "message": "Setup command failed.",
                        "setup_results": [
                            result.model_dump() for result in setup_results
                        ],
                    },
                )

        # Load Kaggle files into sandbox if provided
        kaggle_name = kwargs.get("kaggle_dataset_name", "")
        files = kwargs.get("files", [])
        files_loaded = []
        if kaggle_name and files:
            kaggle_data_dir = os.environ.get(
                "KAGGLE_DATA_DIR",
                f"/fsx/{os.environ.get('USER', '')}/data/kaggle-data-10000",
            )
            data_dir = Path(kaggle_data_dir) / kaggle_name
            if data_dir.exists():
                # Ensure /home/user/input/ exists
                self._sandbox.run_shell("mkdir -p /home/user/input")
                for filename in files:
                    candidates = list(data_dir.rglob(filename))
                    if not candidates:
                        candidates = [
                            f
                            for f in data_dir.rglob("*")
                            if f.name.lower() == filename.lower()
                        ]
                    if candidates:
                        try:
                            with open(candidates[0], "rb") as f:
                                self._sandbox.write_file(
                                    f"/home/user/input/{filename}",
                                    f.read(),
                                )
                            files_loaded.append(filename)
                        except Exception as e:
                            log.warning(f"Failed to upload {filename}: {e}")

        msg = "Jupyter environment ready. Use add_and_execute_code_cell to start."
        if files_loaded:
            msg += f" Files loaded: {', '.join(files_loaded)}"
        if setup_commands:
            msg += f" Setup commands run: {len(setup_commands)}."
        if verify_commands:
            msg += f" Verify commands registered: {len(verify_commands)}."

        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "sandbox_id": self._state.sandbox_id,
                "message": msg,
                "files_loaded": files_loaded,
                "setup_results": [
                    result.model_dump() for result in self._state.setup_results
                ],
                "verify_commands": verify_commands,
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        # [FUNCTION DOC]
        # Fallback for non-MCP actions: returns a non-terminal observation with an
        # error message pointing to ListToolsAction/CallToolAction.
        """Fallback for non-MCP actions — direct users to use MCP tools."""
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        # [FUNCTION DOC]
        # Increments step_count and delegates to MCPEnvironment.step. If an answer has
        # been submitted and a reward computed, marks the observation done and sets its
        # reward to that value. Returns the observation.
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        if self._state.submitted_answer is not None and self._state.last_reward is not None:
            obs.done = True
            obs.reward = self._state.last_reward
        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        # [FUNCTION DOC]
        # Async counterpart of step with the same step counting and final-reward logic.
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        if self._state.submitted_answer is not None and self._state.last_reward is not None:
            obs.done = True
            obs.reward = self._state.last_reward
        return obs

    @property
    def state(self):
        # [FUNCTION DOC]
        # Property returning the current JupyterState.
        return self._state

    def close(self) -> None:
        # [FUNCTION DOC]
        # Kills the sandbox if present and clears it.
        """Release the live E2B sandbox, if any."""
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    def _run_shell_commands(self, commands: Iterable[str]):
        # [FUNCTION DOC]
        # HELPER: Runs each command in the sandbox shell and returns the list of
        # CommandResult objects.
        return [
            _command_result_from_cell_result(command, self._sandbox.run_shell(command))
            for command in commands
        ]

    def _run_verify_commands(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Runs verification: returns passed/total 0 and reward None without a
        # sandbox. Otherwise creates the verifier log dir, deletes any existing reward
        # file (and confirms it is gone), runs all verify commands (stored in state),
        # reads a reward override from the reward file, and falls back to passed/total
        # when no override and at least one command ran. Stores verify results,
        # last_reward and any override-rejection reason in state.
        # Returns: dict with passed, total and reward.
        if not self._sandbox:
            return {"passed": 0, "total": 0, "reward": None}

        self._sandbox.run_shell("mkdir -p /home/user/logs/verifier")
        # The override is documented as verify-written, so drop anything the
        # policy wrote there during the episode before verification starts.
        cleared = self._sandbox.run_shell(_CLEAR_REWARD_FILE).success
        verify_results = self._run_shell_commands(self._state.verify_commands)
        self._state.verify_results = verify_results

        passed = sum(1 for result in verify_results if result.success)
        total = len(verify_results)
        reward, self._state.reward_override_ignored = _read_reward_override(
            self._sandbox, cleared
        )
        if reward is None and total:
            reward = passed / total
        self._state.last_reward = reward

        return {"passed": passed, "total": total, "reward": reward}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _cell_to_model_kwargs(cell: dict) -> dict:
    # [FUNCTION DOC]
    # HELPER: Returns the subset of a tracker cell dict whose keys are NotebookCell
    # fields.
    """Extract fields matching NotebookCell from a tracker cell dict."""
    keys = {
        "cell_id",
        "cell_type",
        "code",
        "output",
        "error",
        "execution_count",
        "has_image",
        "images",
        "success",
    }
    return {k: v for k, v in cell.items() if k in keys}


def _coerce_commands(value: Any) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Normalizes setup/verify input to a list of strings: None gives [],
    # a string gives a one-item list (or [] if blank), an iterable gives its
    # stringified non-blank items.
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _command_result_from_cell_result(command: str, result):
    # [FUNCTION DOC]
    # HELPER: Converts a CellResult into a CommandResult with the command,
    # LLM-formatted output, error and success flag.
    try:
        from ..models import CommandResult
    except ImportError:  # pragma: no cover
        from models import CommandResult

    return CommandResult(
        command=command,
        output=_format_for_llm(result),
        error=result.error,
        success=result.success,
    )


def _read_reward_override(
    sandbox, cleared: bool
) -> tuple[Optional[float], Optional[str]]:
    # [FUNCTION DOC]
    # HELPER: If the reward file could not be cleared before verification, returns
    # (None, reason). Otherwise reads the file via the sandbox shell and parses it.
    # Returns (reward or None, rejection reason or None).
    if not cleared:
        return None, "reward file could not be removed before verification"
    result = sandbox.run_shell(f"cat {REWARD_FILE} 2>/dev/null || true")
    return _parse_reward_override(result.stdout or "")


def _parse_reward_override(raw: str) -> tuple[float | None, str | None]:
    # [FUNCTION DOC]
    # HELPER: Parses reward-file text: empty gives (None, None); non-numeric or
    # non-finite gives (None, reason with the first 40 chars); otherwise (float
    # value, None). Any finite value, including negatives and values above 1, is
    # accepted.
    """Parse the reward file written by a verify command.

    The value comes from the task author rather than the agent, so any finite
    reward is accepted, including negative ones and values above 1. Only a
    value that would poison training is rejected.

    Returns `(reward, None)` for a finite value, `(None, None)` when the file
    is empty, and `(None, reason)` when the value is rejected, in which case
    the caller falls back to the verify pass rate.
    """
    raw = raw.strip()
    if not raw:
        return None, None
    try:
        value = float(raw)
    except ValueError:
        return None, f"not a number: {raw[:40]!r}"
    if not math.isfinite(value):
        return None, f"not finite: {raw[:40]!r}"
    return value, None


def _get_notebook_cell_class():
    # [FUNCTION DOC]
    # HELPER: Imports and returns the NotebookCell class (package or flat layout).
    # Currently unused.
    try:
        from ..models import NotebookCell
    except ImportError:
        from models import NotebookCell
    return NotebookCell


def _format_for_llm(result) -> str:
    # [FUNCTION DOC]
    # HELPER: Formats a CellResult for the LLM: stripped stdout, text results, an
    # image-count note and "ERROR:" plus the error text, joined by newlines; returns
    # "(no output)" if all are empty. stderr is not included.
    """Format a CellResult as a concise string for the LLM."""
    parts = []
    if result.stdout:
        parts.append(result.stdout.strip())
    if result.text_results:
        parts.extend(result.text_results)
    if result.images:
        parts.append(f"[Image output: {len(result.images)} image(s) generated]")
    if result.error:
        parts.append(f"ERROR:\n{result.error}")
    return "\n".join(parts) if parts else "(no output)"
