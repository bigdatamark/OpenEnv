# [MODULE DOC]
# Tests for tbench2_env (Terminal-Bench 2): local Tbench2Environment reset/workdir resolution,
# withholding of verifier assets (tests/ and solution/) with an in-memory tests cache, staging tests
# into fixed paths only at verify time (wiping agent-planted files and symlinks), canonical test.sh
# scoring and its failure modes, and Tbench2DockerEnvironment equivalents against fake containers
# (copy exclusions, stage/verify/cleanup ordering, workdir resolution, argv exec, image-less task
# rejection). An opt-in smoke test needs camel-ai and TB2_ENABLE_TESTS=1. Side effect at import:
# prepends the repo root and envs to sys.path; camel is imported optionally.
import io
import logging
import os
import sys
import tarfile
from pathlib import Path

import pytest

# Add the project root to the path for envs imports, and envs/ itself so the
# server module's Spaces-style absolute imports (``tbench2_env.models``)
# resolve when this file runs standalone (in the full suite another test
# happens to insert envs/ first — don't rely on collection order).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import camel  # noqa: F401
except Exception:
    camel = None

from envs.tbench2_env.models import Tbench2Action
from envs.tbench2_env.server import tbench2_env_environment
from envs.tbench2_env.server.tbench2_env_environment import (
    Tbench2DockerEnvironment,
    Tbench2Environment,
)


class _FakeTerminalToolkit:
    # [CLASS DOC]
    # HELPER: Stand-in for camel's terminal toolkit recording constructor kwargs.
    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Stores kwargs.
        self.kwargs = kwargs

    def shell_exec(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns the empty string.
        return ""


def test_tbench2_reset_uses_default_task_id(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC]
    # A parameterless reset loads the configured default task's instruction.
    """HTTP resets without kwargs should land on the default demo task."""
    task_dir = tmp_path / "headless-terminal"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Solve the terminal task.\n")

    monkeypatch.setattr(
        tbench2_env_environment,
        "_require_terminal_toolkit",
        lambda: _FakeTerminalToolkit,
    )

    env = Tbench2Environment(
        tasks_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        default_task_id="headless-terminal",
    )

    observation = env.reset()

    assert observation.success is True
    assert observation.task_id == "headless-terminal"
    assert "terminal task" in observation.instruction


def _make_env(monkeypatch, tmp_path: Path, task_id: str) -> Tbench2Environment:
    # [FUNCTION DOC]
    # HELPER: Patches _require_terminal_toolkit to the fake and returns a Tbench2Environment over
    # tmp_path with the given default task.
    monkeypatch.setattr(
        tbench2_env_environment,
        "_require_terminal_toolkit",
        lambda: _FakeTerminalToolkit,
    )
    return Tbench2Environment(
        tasks_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        default_task_id=task_id,
    )


def test_tbench2_workdir_from_task_dockerfile(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC]
    # An existing WORKDIR from the task Dockerfile becomes the working directory.
    """A parsed WORKDIR that exists and isn't the server tree becomes the cwd."""
    task_dir = tmp_path / "demo-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do it\n")
    image_workdir = tmp_path / "image-workdir"
    image_workdir.mkdir()
    (task_dir / "environment" / "Dockerfile").write_text(
        f"FROM python:3.13\nWORKDIR {image_workdir}\n"
    )

    env = _make_env(monkeypatch, tmp_path, "demo-task")
    env.reset()

    assert env._workdir == str(image_workdir)


def test_tbench2_workdir_rejects_server_tree(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC]
    # A WORKDIR pointing at the server's own package directory falls back to the task directory.
    """A parsed WORKDIR that is the server's own install tree (e.g. /app on
    the standard server image) falls back to the task source dir."""
    server_tree = Path(tbench2_env_environment.__file__).resolve().parent
    task_dir = tmp_path / "demo-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do it\n")
    (task_dir / "environment" / "Dockerfile").write_text(
        f"FROM python:3.13\nWORKDIR {server_tree}\n"
    )

    env = _make_env(monkeypatch, tmp_path, "demo-task")
    env.reset()

    assert env._workdir == str(task_dir)


# ---------------------------------------------------------------------------
# Verifier-asset withholding (TB2_WITHHOLD_TESTS) and stage-at-verify
# ---------------------------------------------------------------------------


def _make_task_dir(root: Path, name: str = "my-task") -> Path:
    # [FUNCTION DOC]
    # HELPER: Creates a task directory with tests/test.sh, tests/test_outputs.py, solution/solve.sh,
    # task.toml (docker_image debian:12) and instruction.md; returns its path.
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "tests" / "test_outputs.py").write_text("EXPECTED = 'secret42'\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("echo the-answer\n")
    (task / "task.toml").write_text('[environment]\ndocker_image = "debian:12"\n')
    (task / "instruction.md").write_text("do the thing\n")
    return task


@pytest.fixture()
def staged_paths(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC]
    # HELPER: Fixture redirecting the module's fixed verify tests dir and verifier log dir into tmp_path;
    # returns (tests_dir, logs_dir).
    """Redirect the official fixed paths (/tests, /logs/verifier) into tmp."""
    tests = tmp_path / "stage" / "tests"
    logs = tmp_path / "stage" / "logs" / "verifier"
    monkeypatch.setattr(tbench2_env_environment, "_VERIFY_TESTS_DIR", str(tests))
    monkeypatch.setattr(tbench2_env_environment, "_VERIFIER_LOG_DIR", str(logs))
    return tests, logs


@pytest.fixture(autouse=True)
def _fresh_withhold_cache(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Autouse fixture replacing the class-level withheld-tests cache with an empty dict.
    monkeypatch.setattr(Tbench2Environment, "_WITHHELD_TESTS", {})


def test_withhold_removes_assets_and_caches_tests(tmp_path: Path):
    # [FUNCTION DOC]
    # Withholding deletes tests/ and solution/, keeps task.toml and instruction.md, caches the tests, and
    # a second call keeps the cache.
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)

    env._withhold_verifier_assets(task)

    assert not (task / "solution").exists()
    assert not (task / "tests").exists()
    assert (task / "task.toml").is_file()
    assert (task / "instruction.md").is_file()
    assert str(task) in Tbench2Environment._WITHHELD_TESTS

    # Second reset of the same task (dir already gone) must keep the cache.
    env._withhold_verifier_assets(task)
    assert str(task) in Tbench2Environment._WITHHELD_TESTS


def test_stage_from_memory_and_wipe_planted_files(tmp_path: Path, staged_paths):
    # [FUNCTION DOC]
    # Staging from the cache restores the official tests, removes a planted conftest.py and reward file,
    # and leaves an empty log dir.
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._withhold_verifier_assets(task)

    # An agent pre-plants a conftest.py (pytest would auto-load it) and a
    # fake reward before verification.
    stage_tests.mkdir(parents=True)
    (stage_tests / "conftest.py").write_text("import sys; sys.exit(0)\n")
    stage_logs.mkdir(parents=True)
    (stage_logs / "reward.txt").write_text("1\n")

    assert env._stage_tests_for_verify()

    assert (stage_tests / "test.sh").is_file()
    assert (stage_tests / "test_outputs.py").read_text() == "EXPECTED = 'secret42'\n"
    assert not (stage_tests / "conftest.py").exists()
    assert not (stage_logs / "reward.txt").exists()
    assert stage_logs.is_dir() and not list(stage_logs.iterdir())


def test_stage_from_disk_without_gate(tmp_path: Path, staged_paths):
    # [FUNCTION DOC]
    # Without withholding, tests are staged from disk and the source task is untouched.
    stage_tests, _ = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task

    assert env._stage_tests_for_verify()

    assert (stage_tests / "test_outputs.py").is_file()
    # Without the gate the source checkout is never touched.
    assert (task / "tests").is_dir()
    assert (task / "solution").is_dir()


def test_stage_reports_missing_tests(tmp_path: Path, staged_paths):
    # [FUNCTION DOC]
    # Staging returns False when the task has no tests.
    task = tmp_path / "empty-task"
    task.mkdir()
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task

    assert not env._stage_tests_for_verify()


class _RecordingToolkit:
    # [CLASS DOC]
    # HELPER: Toolkit fake recording shell_exec kwargs and returning fixed output.
    def __init__(self, output: str):
        # [FUNCTION DOC]
        # HELPER: Stores the output and an empty call log.
        self.calls: list[dict] = []
        self.output = output

    def shell_exec(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records kwargs and returns the output.
        self.calls.append(kwargs)
        return self.output


def test_evaluate_canonical_from_withheld_copy(tmp_path: Path, staged_paths):
    # [FUNCTION DOC]
    # _evaluate_task runs the staged test.sh, logs under the verifier dir, reads reward 1.0, and removes
    # staged tests and logs afterwards.
    """Full evaluate flow: staging from memory + canonical test.sh scoring."""
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._workdir = str(tmp_path)
    env._terminal_toolkit = _RecordingToolkit(output="__TB2_REWARD__:1")
    env._withhold_verifier_assets(task)

    output, reward, info = env._evaluate_task()

    assert reward == 1.0
    assert info["harness"] == "tests/test.sh"
    # The verify command ran test.sh from the staged fixed path, and wrote its
    # log under the wiped verifier dir — not a fixed /tmp path that outlives it.
    command = env._terminal_toolkit.calls[0]["command"]
    assert f"{stage_tests}/test.sh" in command
    assert "/tmp/tb2_testsh.log" not in command
    assert f"{stage_logs}/testsh.log" in command
    # Both the staged tests and the verifier log/reward dir are gone once
    # scoring returns.
    assert not stage_tests.exists()
    assert not stage_logs.exists()


def test_evaluate_missing_verdict_raises_not_zero(tmp_path: Path, staged_paths):
    # [FUNCTION DOC]
    # An empty reward marker raises RuntimeError ("produced no verdict") and still cleans up.
    """test.sh ran but wrote no reward.txt (crash / kill before the verifier
    wrote one): scoring must error out — reward None on the observation — not
    return a 0.0 indistinguishable from tests genuinely failing. The staged
    assets are still wiped."""
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._workdir = str(tmp_path)
    # Marker line present but empty: reward.txt was never written.
    env._terminal_toolkit = _RecordingToolkit(output="__TB2_REWARD__:")
    env._withhold_verifier_assets(task)

    with pytest.raises(RuntimeError, match="produced no verdict"):
        env._evaluate_task()

    assert not stage_tests.exists()
    assert not stage_logs.exists()


def test_evaluate_without_tests_scores_zero(tmp_path: Path, staged_paths):
    # [FUNCTION DOC]
    # A task without tests scores 0 without running anything.
    task = tmp_path / "empty-task"
    task.mkdir()
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task
    env._terminal_toolkit = _RecordingToolkit(output="")

    output, reward, info = env._evaluate_task()

    assert reward == 0.0
    assert info["tests_passed"] is False
    assert env._terminal_toolkit.calls == []  # nothing was run


def test_evaluate_wipes_partial_stage_on_error(
    tmp_path: Path, staged_paths, monkeypatch
):
    # [FUNCTION DOC]
    # If extraction fails partway (patched _extractall), the OSError propagates and staged paths are
    # removed.
    """A stage that errors partway must be wiped like a completed one — the
    session survives a scoring error, so no partial tests/ copy may remain."""
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._workdir = str(tmp_path)
    env._terminal_toolkit = _RecordingToolkit(output="")
    env._withhold_verifier_assets(task)

    def _explode(tar, dest):
        # [FUNCTION DOC]
        # HELPER: Writes one file into the destination then raises OSError("disk full").
        (Path(dest) / "test_outputs.py").write_text("EXPECTED = 'secret42'\n")
        raise OSError("disk full")

    monkeypatch.setattr(tbench2_env_environment, "_extractall", _explode)

    with pytest.raises(OSError, match="disk full"):
        env._evaluate_task()

    assert not stage_tests.exists()
    assert not stage_logs.exists()


def test_stage_wipes_symlinked_fixed_path(tmp_path: Path, staged_paths):
    # [FUNCTION DOC]
    # A symlinked fixed tests path is replaced by a real directory; the agent's target is not written.
    """A symlinked /tests must not let the official suite merge into a dir the
    agent controls (rmtree silently no-ops on a symlink; _hard_remove doesn't)."""
    stage_tests, _ = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._withhold_verifier_assets(task)

    evil = tmp_path / "agent_controlled"
    evil.mkdir()
    (evil / "conftest.py").write_text("import sys; sys.exit(0)\n")
    stage_tests.parent.mkdir(parents=True, exist_ok=True)
    stage_tests.symlink_to(evil, target_is_directory=True)

    assert env._stage_tests_for_verify()

    assert not stage_tests.is_symlink()
    assert (stage_tests / "test.sh").is_file()
    # The official suite never leaked into the agent's directory.
    assert not (evil / "test.sh").exists()
    assert (evil / "conftest.py").exists()  # their dir left alone


def test_withhold_removes_symlinked_tests(tmp_path: Path):
    # [FUNCTION DOC]
    # A symlinked tests/ is removed (link only) and the cache contains the target's real files.
    """tests/ as a symlink is removed too — the link, not just its target."""
    task = tmp_path / "task"
    task.mkdir()
    real = tmp_path / "real_tests"
    real.mkdir()
    (real / "test_outputs.py").write_text("x = 1\n")
    (task / "tests").symlink_to(real, target_is_directory=True)

    env = Tbench2Environment(withhold_tests=True)
    env._withhold_verifier_assets(task)

    assert not (task / "tests").exists()  # symlink gone
    assert real.exists()  # its target is outside the task dir; untouched
    # The cache captured the target's real files, not a bare link member that
    # would restore an empty suite (tar.add doesn't follow a top-level link).
    import io
    import tarfile

    blob = Tbench2Environment._WITHHELD_TESTS[str(task)]
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        assert any(n.endswith("test_outputs.py") for n in tar.getnames())


# ---------------------------------------------------------------------------
# Docker mode: copy exclusions + stage-at-verify (mock container, no docker)
# ---------------------------------------------------------------------------


class _FakeContainer:
    # [CLASS DOC]
    # HELPER: Fake docker container logging put_archive and exec_run calls in order.
    """Records exec/put_archive calls in one ordered event log.

    exec_run receives argv form (["bash", "-c", payload]); the log records
    the shell payload, and raw_cmds keeps the argv for shape assertions.
    """

    def __init__(self, exec_output: bytes = b"__TB2_REWARD__:1\n"):
        # [FUNCTION DOC]
        # HELPER: Initialises event and raw command logs and the exec output.
        self.events: list[tuple] = []
        self.raw_cmds: list = []
        self.exec_output = exec_output

    def put_archive(self, dest, data):
        # [FUNCTION DOC]
        # HELPER: Logs ("put", dest, data) and returns True.
        self.events.append(("put", dest, data))
        return True

    def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
        # [FUNCTION DOC]
        # HELPER: Logs the raw command and ("exec", shell payload); returns (0, exec_output).
        self.raw_cmds.append(cmd)
        self.events.append(("exec", cmd[-1] if isinstance(cmd, list) else cmd))
        return 0, self.exec_output


def _tar_names(data: bytes) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Returns member names of a tar archive given as bytes.
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return tar.getnames()


def test_docker_copy_excludes_verifier_assets(tmp_path: Path):
    # [FUNCTION DOC]
    # _copy_dir_to_container sends one archive to /task without solution/ or tests/ and no duplicate names.
    task = _make_task_dir(tmp_path)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM debian:12\n")

    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._copy_dir_to_container(task, "/task", exclude_top=("solution", "tests"))

    ((kind, dest, data),) = container.events
    assert (kind, dest) == ("put", "/task")
    names = _tar_names(data)
    assert "task.toml" in names
    assert "environment/Dockerfile" in names
    assert not any(n.startswith(("solution", "tests")) for n in names)
    assert len(names) == len(set(names))  # recursive=False: no duplicates


def test_evaluate_docker_missing_verdict_raises_not_zero(tmp_path: Path):
    # [FUNCTION DOC]
    # An empty reward marker raises "produced no verdict" and the last exec cleans /tests and logs.
    """Docker-mode twin of the local missing-verdict case: an empty reward
    marker errors out (with cleanup) instead of scoring 0.0."""
    task = _make_task_dir(tmp_path)
    env = Tbench2DockerEnvironment()
    container = _FakeContainer(exec_output=b"__TB2_REWARD__:\n")
    env._container = container
    env._task_dir = task

    with pytest.raises(RuntimeError, match="produced no verdict"):
        env._evaluate_docker()

    assert container.events[-1][0] == "exec"
    assert "rm -rf /tests /logs/verifier" in container.events[-1][1]


def test_evaluate_docker_stages_tests_at_verify(tmp_path: Path):
    # [FUNCTION DOC]
    # Event order is wipe, put /tests, run test.sh from /task with a 900s timeout reading reward.txt,
    # then wipe; reward 1.0.
    task = _make_task_dir(tmp_path)
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._task_dir = task

    output, reward, info = env._evaluate_docker()

    assert reward == 1.0
    assert info == {"tests_passed": True, "harness": "tests/test.sh"}
    kinds = [e[0] for e in container.events]
    # rm -rf the agent-writable fixed paths BEFORE the fresh copy goes in,
    # the copy lands before test.sh runs, and both are removed again after.
    assert kinds == ["exec", "put", "exec", "exec"]
    assert "rm -rf /tests /logs/verifier" in container.events[0][1]
    assert container.events[1][1] == "/tests"
    assert "test.sh" in _tar_names(container.events[1][2])
    eval_cmd = container.events[2][1]
    # Canonical harness from the agent's workdir, bounded by the verifier
    # budget, verdict read from reward.txt — not bare pytest in /task.
    assert "bash /tests/test.sh" in eval_cmd
    assert eval_cmd.startswith("cd /task && ")  # no resolved workdir → /task
    assert "timeout 900" in eval_cmd
    assert "/logs/verifier/reward.txt" in eval_cmd
    assert "rm -rf /tests /logs/verifier" in container.events[3][1]


def test_evaluate_docker_runs_in_resolved_workdir(tmp_path: Path):
    # [FUNCTION DOC]
    # The verify command cds into the resolved workdir.
    task = _make_task_dir(tmp_path)
    env = Tbench2DockerEnvironment()
    env._container = _FakeContainer()
    env._task_dir = task
    env._workdir = "/app"

    _, reward, _ = env._evaluate_docker()

    assert reward == 1.0
    eval_cmd = env._container.events[2][1]
    assert eval_cmd.startswith("cd /app && ")


def test_evaluate_docker_fallback_without_testsh(tmp_path: Path):
    # [FUNCTION DOC]
    # Without test.sh, scoring runs pytest on /tests and uses its exit code.
    """A task dir whose tests/ ships no canonical harness scores via pytest,
    still against the staged /tests copy."""
    task = _make_task_dir(tmp_path)
    (task / "tests" / "test.sh").unlink()
    env = Tbench2DockerEnvironment()
    container = _FakeContainer(exec_output=b"__TB2_EXIT_CODE__:0\n")
    env._container = container
    env._task_dir = task

    output, reward, info = env._evaluate_docker()

    assert reward == 1.0
    assert info == {"tests_passed": True, "exit_code": 0}
    eval_cmd = container.events[2][1]
    assert "pytest -q /tests -rA" in eval_cmd
    assert "test.sh" not in eval_cmd


def test_evaluate_docker_cleans_up_when_scoring_raises(tmp_path: Path):
    # [FUNCTION DOC]
    # If running test.sh raises, the error propagates and cleanup still runs.
    """The staged /task/tests must not outlive a verify that errors out —
    step() reports the failure without ending the episode, so the agent
    keeps its session."""
    task = _make_task_dir(tmp_path)

    class _ExplodingContainer(_FakeContainer):
        # [CLASS DOC]
        # HELPER: Container whose exec raises RuntimeError for the test.sh command.
        def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
            # [FUNCTION DOC]
            # HELPER: Logs and raises for test.sh payloads, otherwise delegates.
            payload = cmd[-1] if isinstance(cmd, list) else cmd
            if "test.sh" in payload:
                self.events.append(("exec", payload))
                raise RuntimeError("docker daemon hiccup")
            return super().exec_run(cmd, workdir=workdir, stdout=stdout, stderr=stderr)

    env = Tbench2DockerEnvironment()
    container = _ExplodingContainer()
    env._container = container
    env._task_dir = task

    with pytest.raises(RuntimeError, match="hiccup"):
        env._evaluate_docker()

    assert container.events[-1][0] == "exec"
    assert "rm -rf /tests /logs/verifier" in container.events[-1][1]


def test_evaluate_docker_fails_closed_when_prestage_wipe_fails(tmp_path: Path):
    # [FUNCTION DOC]
    # A failing pre-stage wipe raises "could not reset /tests" and nothing is put.
    """If the pre-stage rm -rf fails, scoring must error out rather than
    put_archive into a dir the agent still controls."""
    task = _make_task_dir(tmp_path)

    class _StubbornContainer(_FakeContainer):
        # [CLASS DOC]
        # HELPER: Container whose pre-stage wipe (rm plus mkdir) exits 1.
        def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
            # [FUNCTION DOC]
            # HELPER: Logs the payload; returns exit 1 for the pre-stage wipe, else success.
            payload = cmd[-1] if isinstance(cmd, list) else cmd
            self.events.append(("exec", payload))
            if "rm -rf /tests" in payload and "mkdir" in payload:
                return 1, b"rm: cannot remove '/tests': busy\n"
            return 0, self.exec_output

    env = Tbench2DockerEnvironment()
    container = _StubbornContainer()
    env._container = container
    env._task_dir = task

    with pytest.raises(RuntimeError, match="could not reset /tests"):
        env._evaluate_docker()

    assert not any(e[0] == "put" for e in container.events)


def test_evaluate_docker_warns_when_cleanup_fails(tmp_path: Path, caplog):
    # [FUNCTION DOC]
    # A failing post-verify wipe logs a warning but keeps reward 1.0.
    """A post-verify rm -rf that exits nonzero must not stay silent."""
    task = _make_task_dir(tmp_path)

    class _LeakyContainer(_FakeContainer):
        # [CLASS DOC]
        # HELPER: Container whose post-verify wipe (rm without mkdir) exits 1.
        def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
            # [FUNCTION DOC]
            # HELPER: Logs the payload; returns exit 1 for the post-verify wipe, else success.
            payload = cmd[-1] if isinstance(cmd, list) else cmd
            self.events.append(("exec", payload))
            if "rm -rf /tests" in payload and "mkdir" not in payload:
                return 1, b"rm: cannot remove '/tests': busy\n"
            return 0, self.exec_output

    env = Tbench2DockerEnvironment()
    container = _LeakyContainer()
    env._container = container
    env._task_dir = task

    with caplog.at_level(logging.WARNING):
        output, reward, info = env._evaluate_docker()

    assert reward == 1.0
    assert any(
        "failed to remove staged /tests" in record.getMessage()
        for record in caplog.records
    )


def test_evaluate_docker_missing_tests_scores_zero(tmp_path: Path):
    # [FUNCTION DOC]
    # A task without tests scores 0 with no container activity.
    task = tmp_path / "no-tests-task"
    task.mkdir()
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._task_dir = task

    output, reward, info = env._evaluate_docker()

    assert reward == 0.0
    assert container.events == []


def test_docker_reset_rejects_task_without_image(tmp_path: Path):
    # [FUNCTION DOC]
    # reset on a task without docker_image raises RuntimeError.
    """No host-execution fallback: a task dir that declares no docker_image
    must fail reset loudly, not silently run agent commands on the server."""
    task = tmp_path / "imageless-task"
    task.mkdir()
    (task / "task.toml").write_text("[metadata]\n")
    (task / "instruction.md").write_text("do it\n")

    env = Tbench2DockerEnvironment(
        tasks_dir=str(tmp_path), output_dir=str(tmp_path / "runs")
    )

    with pytest.raises(RuntimeError, match="docker_image"):
        env.reset(task_id="imageless-task")


def test_docker_failed_reset_closes_previous_container(tmp_path: Path):
    # [FUNCTION DOC]
    # A rejected reset stops (timeout 10) and force-removes the previous container and clears task state.
    """A rejected reset must not leave the previous container usable with
    metadata from the new task."""
    task = tmp_path / "imageless-task"
    task.mkdir()
    (task / "task.toml").write_text("[metadata]\n")
    (task / "instruction.md").write_text("do it\n")

    class _PreviousContainer:
        # [CLASS DOC]
        # HELPER: Container fake logging stop/remove calls.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Initialises the event log.
            self.events = []

        def stop(self, timeout):
            # [FUNCTION DOC]
            # HELPER: Logs ("stop", timeout).
            self.events.append(("stop", timeout))

        def remove(self, force):
            # [FUNCTION DOC]
            # HELPER: Logs ("remove", force).
            self.events.append(("remove", force))

    env = Tbench2DockerEnvironment(
        tasks_dir=str(tmp_path), output_dir=str(tmp_path / "runs")
    )
    previous_container = _PreviousContainer()
    env._container = previous_container
    env._task_dir = tmp_path / "previous-task"
    env._instruction = "previous task"
    env._workdir = "/previous-workdir"

    with pytest.raises(RuntimeError, match="docker_image"):
        env.reset(task_id="imageless-task")

    assert previous_container.events == [("stop", 10), ("remove", True)]
    assert env._container is None
    assert env._task_dir is None
    assert env._instruction == ""
    assert env._workdir == ""


class _FakeImage:
    # [CLASS DOC]
    # HELPER: Docker image fake with Config.WorkingDir.
    def __init__(self, working_dir: str):
        # [FUNCTION DOC]
        # HELPER: Sets attrs with the given working dir.
        self.attrs = {"Config": {"WorkingDir": working_dir}}


def test_docker_workdir_prefers_image_metadata(tmp_path: Path):
    # [FUNCTION DOC]
    # _resolve_workdir prefers image WorkingDir, then the task Dockerfile WORKDIR, then /task.
    """Image Config.WorkingDir wins (it sees base-image WORKDIRs); the task
    Dockerfile is the fallback, /task the last resort."""
    task = _make_task_dir(tmp_path)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text(
        "FROM debian:12\nWORKDIR /from-dockerfile\n"
    )
    env = Tbench2DockerEnvironment()

    assert env._resolve_workdir(_FakeImage("/from-image"), task) == "/from-image"
    assert env._resolve_workdir(_FakeImage(""), task) == "/from-dockerfile"

    bare = tmp_path / "bare-task"
    bare.mkdir()
    assert env._resolve_workdir(_FakeImage(""), bare) == "/task"


def test_docker_exec_passes_command_as_argv(tmp_path: Path):
    # [FUNCTION DOC]
    # _exec_in_container runs ["bash", "-c", "cd <workdir> && <command>"] preserving quoting.
    """Agent commands ride as a bash -c argv element, so shell quoting in the
    command (a single quote, say) survives byte-identical; the exec cd's into
    the resolved image workdir."""
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._workdir = "/app"

    env._exec_in_container("echo 'hi there'")

    (raw,) = container.raw_cmds
    assert isinstance(raw, list) and raw[:2] == ["bash", "-c"]
    assert raw[2] == "cd /app && echo 'hi there'"


@pytest.mark.skipif(camel is None, reason="camel-ai not installed")
@pytest.mark.skipif(
    os.environ.get("TB2_ENABLE_TESTS", "0") != "1",
    reason="TB2_ENABLE_TESTS not enabled",
)
def test_tbench2_env_smoke():
    # [FUNCTION DOC]
    # Opt-in end-to-end: reset a real task, exec pwd successfully with output, then close. Runs real
    # commands via camel's terminal toolkit.
    env = Tbench2Environment(tasks_dir=os.environ.get("TB2_TASKS_DIR"))
    obs = env.reset(task_id=os.environ.get("TB2_TASK_ID", "headless-terminal"))
    assert obs.instruction

    result = env.step(Tbench2Action(action_type="exec", command="pwd"))
    assert result.success
    assert result.output

    env.step(Tbench2Action(action_type="close"))
    env.close()
