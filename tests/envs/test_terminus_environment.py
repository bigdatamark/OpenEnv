# [MODULE DOC]
# Tests for terminus_env's TerminusEnvironment (single "terminal" MCP tool, reset with/without E2B
# key, setup and verify commands, command execution, final-answer verification and reward-file
# override rules) using fake sandboxes.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from terminus_env.models import TerminusState
from terminus_env.server.e2b_sandbox import ShellResult
from terminus_env.server.terminus_env_environment import TerminusEnvironment


class FakeSandbox:
    # [CLASS DOC]
    # HELPER: In-memory stand-in for the E2B sandbox recording shell commands.
    sandbox_id = "fake-sandbox"

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Initialises killed False and an empty command log.
        self.killed = False
        self.shell_commands: list[str] = []

    def run_shell(self, command: str, timeout_s: int = 120) -> ShellResult:
        # [FUNCTION DOC]
        # HELPER: Records the command; fails for commands containing "exit 1", returns empty success for
        # reading the reward file, otherwise success with stdout "shell: <command>".
        self.shell_commands.append(command)
        if "exit 1" in command:
            return ShellResult(
                stdout="",
                stderr="failed",
                error="SystemExit: 1",
                success=False,
            )
        if command.startswith("cat /home/user/logs/verifier/reward.txt"):
            return ShellResult(stdout="", stderr="", error=None, success=True)
        return ShellResult(
            stdout=f"shell: {command}",
            stderr="",
            error=None,
            success=True,
        )

    def kill(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Marks the sandbox killed.
        self.killed = True


def _extract_text(result) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the first content item's text of a tool result, else str of its data, else str.
    if hasattr(result, "content") and result.content:
        return result.content[0].text
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def test_lists_single_terminal_tool_without_reset():
    # [FUNCTION DOC]
    # ListToolsAction before reset lists only "terminal".
    env = TerminusEnvironment()

    obs = env.step(ListToolsAction())

    assert [tool.name for tool in obs.tools] == ["terminal"]


def test_reset_without_e2b_key_fails_cleanly(monkeypatch):
    # [FUNCTION DOC]
    # Without E2B_API_KEY, reset returns a done error observation mentioning the key.
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    env = TerminusEnvironment()

    obs = env.reset()

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert "E2B_API_KEY" in obs.metadata["error"]


def test_reset_runs_setup_and_stores_verify_commands(monkeypatch):
    # [FUNCTION DOC]
    # reset (with E2BSandbox patched) creates the verifier log dir, runs setup, records sandbox id and
    # setup results, and stores verify commands.
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    fake_sandbox = FakeSandbox()
    monkeypatch.setattr(
        "terminus_env.server.terminus_env_environment.E2BSandbox",
        lambda api_key: fake_sandbox,
    )
    env = TerminusEnvironment()

    obs = env.reset(setup=["echo setup"], verify=["test -f answer.txt"])

    assert obs.done is False
    assert fake_sandbox.shell_commands == [
        "mkdir -p /home/user/logs/verifier",
        "echo setup",
    ]
    assert env.state.sandbox_id == "fake-sandbox"
    assert env.state.setup_results[0].success is True
    assert env.state.verify_commands == ["test -f answer.txt"]
    assert obs.metadata["verify_commands"] == ["test -f answer.txt"]


def test_reset_fails_when_setup_command_fails(monkeypatch):
    # [FUNCTION DOC]
    # A failing setup command makes reset return a done error observation.
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    monkeypatch.setattr(
        "terminus_env.server.terminus_env_environment.E2BSandbox",
        lambda api_key: FakeSandbox(),
    )
    env = TerminusEnvironment()

    obs = env.reset(setup=["exit 1"], verify=["test -f answer.txt"])

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert obs.metadata["setup_results"][0]["success"] is False


def test_terminal_command_runs_inside_existing_sandbox():
    # [FUNCTION DOC]
    # A terminal command runs in the sandbox, increments step_count and is recorded.
    env = TerminusEnvironment()
    fake_sandbox = FakeSandbox()
    env._sandbox = fake_sandbox
    env._state = TerminusState(episode_id="episode-1", sandbox_id="fake-sandbox")

    obs = env.step(
        CallToolAction(
            tool_name="terminal",
            arguments={"command": "pwd"},
        )
    )

    assert obs.error is None
    assert "shell: pwd" in _extract_text(obs.result)
    assert env.state.step_count == 1
    assert env.state.commands[0].command == "pwd"


def test_terminal_final_answer_runs_verify_commands():
    # [FUNCTION DOC]
    # A final_answer argument runs verify commands, reporting 1/2 passed with reward 0.5.
    env = TerminusEnvironment()
    fake_sandbox = FakeSandbox()
    env._sandbox = fake_sandbox
    env._state = TerminusState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=["test -f answer.txt", "exit 1"],
    )

    obs = env.step(
        CallToolAction(
            tool_name="terminal",
            arguments={"final_answer": "done"},
        )
    )

    assert obs.error is None
    assert "Verification: 1/2 passed; reward=0.5" in _extract_text(obs.result)
    assert env.state.submitted_answer == "done"
    assert env.state.last_reward == 0.5
    assert [result.command for result in env.state.verify_results] == [
        "test -f answer.txt",
        "exit 1",
    ]


REWARD_FILE = "/home/user/logs/verifier/reward.txt"


class RewardFileSandbox(FakeSandbox):
    # [CLASS DOC]
    # HELPER: FakeSandbox simulating a writable/removable verifier reward file via echo, rm -f and cat.
    """FakeSandbox whose reward file any shell command can write, as in E2B."""

    def __init__(self, *, removable: bool = True) -> None:
        # [FUNCTION DOC]
        # HELPER: Initialises with no reward file and whether it can be removed.
        super().__init__()
        self.reward_file: str | None = None
        self.removable = removable

    def run_shell(self, command: str, timeout_s: int = 120) -> ShellResult:
        # [FUNCTION DOC]
        # HELPER: Handles echo-to-reward-file, rm -f (success iff cleared) and cat; delegates others.
        if command.startswith("echo ") and command.endswith(f" > {REWARD_FILE}"):
            self.shell_commands.append(command)
            self.reward_file = command[len("echo ") : -len(f" > {REWARD_FILE}")]
            return ShellResult(stdout="", stderr="", error=None, success=True)
        if command.startswith(f"rm -f {REWARD_FILE}"):
            self.shell_commands.append(command)
            if self.removable:
                self.reward_file = None
            gone = self.reward_file is None
            return ShellResult(
                stdout="",
                stderr="",
                error=None if gone else "SystemExit: 1",
                success=gone,
            )
        if command.startswith(f"cat {REWARD_FILE}"):
            self.shell_commands.append(command)
            return ShellResult(
                stdout=self.reward_file or "", stderr="", error=None, success=True
            )
        return super().run_shell(command, timeout_s)


def _submit(sandbox, verify_commands, agent_commands=()) -> TerminusState:
    # [FUNCTION DOC]
    # HELPER: Creates an environment on the sandbox with verify commands, runs optional agent commands,
    # submits a final answer and returns the state.
    env = TerminusEnvironment()
    env._sandbox = sandbox
    env._state = TerminusState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=verify_commands,
    )
    for command in agent_commands:
        env.step(CallToolAction(tool_name="terminal", arguments={"command": command}))
    env.step(CallToolAction(tool_name="terminal", arguments={"final_answer": "done"}))
    return env.state


def test_reward_file_written_by_the_agent_is_ignored():
    # [FUNCTION DOC]
    # An agent-written reward file is cleared before verification; reward is the pass rate 0.5.
    state = _submit(
        RewardFileSandbox(),
        ["true", "exit 1"],
        agent_commands=[f"echo 1.0 > {REWARD_FILE}"],
    )

    assert state.last_reward == 0.5
    assert state.reward_override_ignored is None


def test_reward_file_is_cleared_again_before_a_second_submission():
    # [FUNCTION DOC]
    # A reward file written between two submissions is cleared again; reward stays 0.5.
    env = TerminusEnvironment()
    env._sandbox = RewardFileSandbox()
    env._state = TerminusState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=["true", "exit 1"],
    )

    env.step(CallToolAction(tool_name="terminal", arguments={"final_answer": "a"}))
    env.step(
        CallToolAction(
            tool_name="terminal",
            arguments={"command": f"echo 1.0 > {REWARD_FILE}"},
        )
    )
    env.step(CallToolAction(tool_name="terminal", arguments={"final_answer": "b"}))

    assert env.state.last_reward == 0.5


def test_reward_file_written_by_a_verify_command_overrides_the_pass_rate():
    # [FUNCTION DOC]
    # A verify-written reward file overrides the pass rate (0.8).
    state = _submit(RewardFileSandbox(), [f"echo 0.8 > {REWARD_FILE}", "exit 1"])

    assert state.last_reward == 0.8
    assert state.reward_override_ignored is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "passed"])
def test_invalid_reward_file_falls_back_to_the_pass_rate(value):
    # [FUNCTION DOC]
    # Parametrized: non-finite or non-numeric values fall back to 0.5 and are reported.
    state = _submit(RewardFileSandbox(), [f"echo {value} > {REWARD_FILE}", "exit 1"])

    assert state.last_reward == 0.5
    assert repr(value) in state.reward_override_ignored


@pytest.mark.parametrize("value", ["-0.5", "1.5"])
def test_reward_file_outside_the_unit_interval_is_kept(value):
    # [FUNCTION DOC]
    # Parametrized: finite out-of-range values are used as is.
    state = _submit(RewardFileSandbox(), [f"echo {value} > {REWARD_FILE}", "exit 1"])

    assert state.last_reward == float(value)
    assert state.reward_override_ignored is None


def test_reward_file_that_cannot_be_removed_is_ignored():
    # [FUNCTION DOC]
    # If the stale file cannot be removed, overrides are ignored ("could not be removed").
    state = _submit(
        RewardFileSandbox(removable=False),
        ["true", "exit 1"],
        agent_commands=[f"echo 1.0 > {REWARD_FILE}"],
    )

    assert state.last_reward == 0.5
    assert "could not be removed" in state.reward_override_ignored
