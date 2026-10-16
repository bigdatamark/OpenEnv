# [MODULE DOC]
# Tests for the opt-in NeMo shell agent profile (openenv.harbor.nemo_profile) and the packaged
# example shell tool. Skipped entirely when harbor's nemo_agent module is not installed.
"""The opt-in NeMo profile keeps endpoint routing and exercises real sandbox commands."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("harbor.agents.installed.nemo_agent")

from openenv.harbor.nemo_profile import NemoShellProfile


def test_nemo_react_profile_reuses_harbor_provider_configuration(tmp_path):
    # [FUNCTION DOC]
    # Builds a NemoShellProfile with an OPENAI_BASE_URL extra env and parses its generated YAML config;
    # asserts the selected LLM uses that base URL, the session id as api key, the bare model name,
    # native tool calling, and only the "shell" tool. Side effects: writes logs under tmp_path.
    agent = NemoShellProfile(
        logs_dir=tmp_path,
        model_name="openai/Qwen3.5-4B",
        llm_type="openai",
        version="1.9.0",
        extra_env={"OPENAI_BASE_URL": "https://capture.example/v1"},
    )
    config = yaml.safe_load(agent._generate_config_yaml("Qwen3.5-4B", "session-test"))
    llm = config["llms"][config["workflow"]["llm_name"]]
    assert llm["base_url"] == "https://capture.example/v1"
    assert llm["api_key"] == "session-test"
    assert llm["model_name"] == "Qwen3.5-4B"
    assert config["workflow"]["use_native_tool_calling"] is True
    assert config["workflow"]["tool_names"] == ["shell"]


def shell_module():
    # [FUNCTION DOC]
    # HELPER: Dynamically imports examples/harbor/nemo_shell_profile/src/openenv_nat_shell/shell.py
    # (resolved relative to the repo root) under the name "qualification_shell" and returns the module.
    path = (
        Path(__file__).parents[2]
        / "examples/harbor/nemo_shell_profile/src/openenv_nat_shell/shell.py"
    )
    spec = importlib.util.spec_from_file_location("qualification_shell", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_shell_executes_and_reports_failure_without_faking_success(tmp_path):
    # [FUNCTION DOC]
    # Runs a shell command that writes a file, prints it and exits 7 through the example shell's async
    # execute; asserts the JSON result reports exit_code 7 and stdout "data" and the file exists.
    shell = shell_module()
    result = json.loads(
        await shell.execute(
            "printf data > answer.txt; cat answer.txt; exit 7", cwd=str(tmp_path)
        )
    )
    assert result["exit_code"] == 7
    assert result["stdout"] == "data"
    assert (tmp_path / "answer.txt").read_text() == "data"


@pytest.mark.asyncio
async def test_shell_timeout_terminates_command_group(tmp_path):
    # [FUNCTION DOC]
    # Asserts execute raises TimeoutError for a sleeping command with a 0.1s timeout and that the
    # command's later "touch" never ran (the whole process group was killed).
    shell = shell_module()
    with pytest.raises(TimeoutError):
        await shell.execute(
            "sleep 5; touch late-output", timeout=0.1, cwd=str(tmp_path)
        )
    assert not (tmp_path / "late-output").exists()


def test_explicit_profile_uses_packaged_workflow_without_mutating_generic_seam():
    # [FUNCTION DOC]
    # Fetches the generic "nemo-agent" seam and the "shell-1.9.0" profile variant; asserts the profile
    # resolves to NemoShellProfile with version 1.9.0 and a workflow_package dir containing pyproject.toml,
    # and that the generic seam is unchanged (same object, different import path).
    from pathlib import Path

    from openenv.harbor.seams import get

    generic = get("nemo-agent")
    selected = get("nemo-agent", profile="shell-1.9.0")
    _, kwargs, _, _ = selected.resolve(
        base_url="https://proxy.example", session="session", model="model"
    )
    assert selected.import_path == "openenv.harbor.nemo_profile:NemoShellProfile"
    assert kwargs["version"] == "1.9.0"
    assert (Path(kwargs["workflow_package"]) / "pyproject.toml").is_file()
    assert get("nemo-agent") is generic
    assert generic.import_path != selected.import_path
