# [MODULE DOC]
# Tests for openenv.harbor.install_fixes, which patches upstream Harbor agent
# adapters: exporting openclaw's SQLite session transcript to JSONL, trimming
# a trailing stderr log line from openclaw's capture file so Harbor's own
# parser succeeds, OpenHands runtime preparation, Kimi process-group
# isolation, node version pinning for openclaw, positional exec_as_agent
# compatibility, and openclaw provider catalog config. Skipped without
# install_fixes or harbor's openclaw agent. Module constants hold the trim
# script, container/session paths, a realistic capture envelope and the
# trailing log line.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The upstream Harbor defect that cost `openclaw` its ATIF trajectory.

The sibling `hermes` fix went away with the seam itself: hermes-agent fails to install
(`exit 127`, 5/5 attempts), so there is nothing left to intercept.

Both failed the same way: no error, no exception, no missing file -- just an agent that quietly
produced no trace, so every rollout reported `atif=none` and the cross-check silently did not exist.
Neither is detectable from a passing rollout, which is why they are pinned here.
"""

from __future__ import annotations

import json

import pytest

install_fixes = pytest.importorskip("openenv.harbor.install_fixes")
openclaw_mod = pytest.importorskip("harbor.agents.installed.openclaw")

TRIM = install_fixes._OPENCLAW_TRIM_TRAILING_LOG
OpenClaw = openclaw_mod.OpenClaw

_CONTAINER_PATH = "/logs/agent/openclaw.txt"
_SESSION_FILE = "/root/.openclaw/agents/main/sessions/790c93f1.jsonl"


def test_sqlite_export_preserves_per_call_usage_for_harbor(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # With subprocess.run patched to emulate "openclaw sessions export-trajectory",
    # the export writes the bundle's entries as JSONL that Harbor converts to ATIF
    # steps with the original per-call completion token counts.
    import subprocess
    from types import SimpleNamespace

    meta = {"sessionId": "session-1", "sessionFile": "agent:main:main"}
    (tmp_path / "openclaw.txt").write_text(json.dumps({"meta": {"agentMeta": meta}}))
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "answer"}],
                "usage": {"input": 100 + count, "output": count},
            },
        }
        for count in [137, 124, 96, 5]
    ]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "session-branch.json").write_text(json.dumps({"entries": entries}))

    def run(command, **kwargs):
        # [FUNCTION DOC]
        # Fake subprocess.run: checks the command/flags and returns stdout naming the
        # session and bundle directory.
        assert command[:5] == [
            "openclaw",
            "sessions",
            "export-trajectory",
            "--session-key",
            "agent:main:main",
        ]
        assert kwargs["check"] and kwargs["timeout"] == 60
        return SimpleNamespace(
            stdout=json.dumps({"sessionId": "session-1", "outputDir": str(bundle)})
        )

    monkeypatch.setattr(subprocess, "run", run)
    install_fixes._export_openclaw_sqlite_transcript(str(tmp_path))
    target = tmp_path / "openclaw.session.jsonl"
    assert [json.loads(line) for line in target.read_text().splitlines()] == entries
    steps = openclaw_mod.openclaw_session_jsonl_to_atif_steps(
        target, instruction="task", model_name="test"
    )
    assert [
        step.metrics.completion_tokens for step in steps if step.source == "agent"
    ] == [137, 124, 96, 5]


def test_sqlite_export_rejects_other_session(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # An export for a different session raises ValueError and writes nothing.
    import subprocess
    from types import SimpleNamespace

    (tmp_path / "openclaw.txt").write_text(
        json.dumps(
            {
                "meta": {
                    "agentMeta": {
                        "sessionId": "expected",
                        "sessionFile": "agent:main:main",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({"sessionId": "other"})
        ),
    )
    with pytest.raises(ValueError, match="different session"):
        install_fixes._export_openclaw_sqlite_transcript(str(tmp_path))
    assert not (tmp_path / "openclaw.session.jsonl").exists()


def test_sqlite_export_preserves_existing_native_jsonl(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # An existing JSONL transcript is kept and no export subprocess runs.
    import subprocess

    target = tmp_path / "openclaw.session.jsonl"
    target.write_text("existing native transcript\n")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must preserve legacy transcript"),
    )
    install_fixes._export_openclaw_sqlite_transcript(str(tmp_path))
    assert target.read_text() == "existing native transcript\n"


# The shape openclaw actually produces: a pretty-printed envelope whose closing brace sits at
# column 0, and whose LAST nested object is the `completion` block. Both details matter below, and
# the key ORDER is taken from a real capture file (`payloads` first, `meta` last) because the
# backwards-scan trap depends on which nested object happens to be last.
_ENVELOPE = {
    "payloads": [],
    "meta": {
        "agentMeta": {"sessionId": "790c93f1", "sessionFile": _SESSION_FILE},
        "completion": {"stopReason": "stop", "finishReason": "stop"},
    },
}
# Harbor merges the agent's stderr into the same file with `2>&1`, so this lands after the JSON.
_TRAILING_LOG = (
    "[agents/agent-command] [agent] run 9e921697-bfe9-4266-ad60-6e9f65d0de5e "
    "ended with stopReason=stop"
)


def _capture_file(with_trailing_log: bool = True) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the pretty-printed envelope text, optionally followed by the
    # trailing log line.
    body = json.dumps(_ENVELOPE, indent=2)
    return f"{body}\n{_TRAILING_LOG}\n" if with_trailing_log else f"{body}\n"


def _run_trim(tmp_path) -> str:
    # [FUNCTION DOC]
    # HELPER: writes a capture file with a trailing log to tmp_path, executes the
    # production trim script with its container path replaced by that file
    # (asserting the replacement happened), and returns the resulting file text.
    # Side effects: executes code via exec and writes a file.
    """Execute the real production trim script against a temp file, not a copy of its logic."""
    target = tmp_path / "openclaw.txt"
    script = TRIM.replace(_CONTAINER_PATH, str(target))
    # If the constant is ever reworded, the substitution stops matching and this test would
    # silently exercise nothing. Fail instead.
    assert script != TRIM, f"{_CONTAINER_PATH!r} no longer appears in the trim script"
    target.write_text(_capture_file(), encoding="utf-8")
    exec(compile(script, "<trim>", "exec"), {})
    return target.read_text(encoding="utf-8")


# --- openclaw ---------------------------------------------------------------
def test_harbor_cannot_parse_its_own_capture_file_when_openclaw_logs_after_the_json():
    # [FUNCTION DOC]
    # Harbor's _load_json_object returns None when a log line follows the JSON.
    """The bug itself: one stderr line after the envelope and Harbor's parser gives up.

    `_load_json_object` requires the JSON object to consume the entire remaining suffix, but Harbor's
    own `2>&1` is what put a non-JSON line there. Returning None means `populate_context_post_run`
    returns at `if not envelope` and no `trajectory.json` is ever written.
    """
    assert OpenClaw._load_json_object(_capture_file()) is None


def test_trimming_the_trailing_log_line_makes_harbors_own_parser_succeed(tmp_path):
    # [FUNCTION DOC]
    # After trimming, Harbor parses the envelope including sessionFile.
    """The fix, stated as the only thing it is allowed to be: Harbor's parser does the parsing.

    The subclass removes the trailing lines and nothing else, so the envelope that comes back is
    Harbor's own -- including `agentMeta.sessionFile`, which is what the session copy needs.
    """
    parsed = OpenClaw._load_json_object(_run_trim(tmp_path))

    assert parsed is not None
    assert parsed["meta"]["agentMeta"]["sessionFile"] == _SESSION_FILE


def test_trim_leaves_an_already_clean_capture_file_untouched(tmp_path):
    # [FUNCTION DOC]
    # A clean capture file is unchanged by the trim script.
    """A run whose stopReason is `end_turn` logs nothing, so the file is already parseable."""
    target = tmp_path / "openclaw.txt"
    script = TRIM.replace(_CONTAINER_PATH, str(target))
    clean = _capture_file(with_trailing_log=False)
    target.write_text(clean, encoding="utf-8")

    exec(compile(script, "<trim>", "exec"), {})

    assert target.read_text(encoding="utf-8") == clean


def test_trim_survives_a_capture_file_with_no_envelope_at_all(tmp_path):
    # [FUNCTION DOC]
    # A file without JSON is left unchanged without error.
    """An agent that died before emitting JSON must not turn into a crash in our override."""
    target = tmp_path / "openclaw.txt"
    script = TRIM.replace(_CONTAINER_PATH, str(target))
    garbage = "openclaw: command not found\n"
    target.write_text(garbage, encoding="utf-8")

    exec(compile(script, "<trim>", "exec"), {})

    assert target.read_text(encoding="utf-8") == garbage


def test_a_backwards_scan_without_the_suffix_rule_latches_onto_the_wrong_object():
    # [FUNCTION DOC]
    # Demonstrates that a backwards raw-decode scan finds the nested completion
    # object rather than the envelope.
    """Why the fix trims text instead of loosening the parser -- the obvious loosening is wrong.

    Dropping Harbor's "must consume the suffix" rule looks like the one-line fix. It is not: the scan
    walks backwards, so the first thing that decodes is the LAST nested object, and `completion`
    decodes perfectly. The caller then gets a dict with no `meta` at all and builds a degenerate
    2-step trajectory from it -- which still reports `atif=match`, because `reconcile` downgrades a
    trace carrying no token counts instead of failing it. A silently wrong trace is worse than none.
    """
    text = _capture_file().strip()
    decoder = json.JSONDecoder()
    found = None
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            found = obj
            break

    assert found == {"stopReason": "stop", "finishReason": "stop"}
    assert "meta" not in found


@pytest.mark.asyncio
async def test_openhands_clean_install_still_prepares_local_runtime(monkeypatch):
    # [FUNCTION DOC]
    # Async: after a successful base install, InterceptOpenHands runs three root
    # commands creating python and poetry wrappers for the OpenHands venv.
    """A dependency-successful install must not leave LocalRuntime invoking real Poetry."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr(install_fixes.OpenHands, "install", AsyncMock())
    agent = object.__new__(install_fixes.InterceptOpenHands)
    root_exec = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_root", root_exec)
    environment = object()
    await agent.install(environment)
    commands = [call.kwargs["command"] for call in root_exec.call_args_list]
    assert len(commands) == 3
    assert all('/opt/openhands-venv/bin/python "$@"' in command for command in commands)
    assert any("/usr/local/bin/poetry" in command for command in commands)
    assert any("/opt/openhands-venv/bin/poetry" in command for command in commands)


def test_kimi_terminal_signal_does_not_kill_its_exec_transport():
    # [FUNCTION DOC]
    # A command wrapped by _isolated_process_group that runs "kill 0" exits 143
    # after printing, without killing the outer bash (runs a real subprocess).
    import subprocess

    wrapped = install_fixes._isolated_process_group("echo finished; kill 0")
    completed = subprocess.run(
        ["bash", "-c", wrapped],
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 143
    assert completed.stdout.strip() == "finished"


@pytest.mark.asyncio
async def test_openclaw_install_and_runtime_select_supported_node(monkeypatch):
    # [FUNCTION DOC]
    # Async: InterceptOpenClaw rewrites "nvm install/use 22" to 24.16.0.
    from unittest.mock import AsyncMock

    execution = AsyncMock()
    monkeypatch.setattr(install_fixes.OpenClaw, "exec_as_agent", execution)
    agent = object.__new__(install_fixes.InterceptOpenClaw)
    await agent.exec_as_agent(
        object(), command="nvm install 22 && nvm use 22 && openclaw --version"
    )
    assert (
        execution.call_args.kwargs["command"]
        == "nvm install 24.16.0 && nvm use 24.16.0 && openclaw --version"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter,parent",
    [
        (install_fixes.InterceptOpenClaw, install_fixes.OpenClaw),
        (install_fixes.InterceptKimi, install_fixes.KimiCli),
    ],
)
async def test_command_wrappers_preserve_harbor_positional_call_contract(
    monkeypatch, adapter, parent
):
    # [FUNCTION DOC]
    # Async: positional exec_as_agent arguments are forwarded as keywords to the
    # parent implementation and its result returned.
    from unittest.mock import AsyncMock

    execution = AsyncMock(return_value="executed")
    monkeypatch.setattr(parent, "exec_as_agent", execution)
    agent = object.__new__(adapter)
    environment = object()
    env = {"TEST_SETTING": "test"}
    result = await agent.exec_as_agent(environment, "echo ready", env, "/tmp", 30)
    assert result == "executed"
    execution.assert_awaited_once_with(
        environment, command="echo ready", env=env, cwd="/tmp", timeout_sec=30
    )


def test_openclaw_catalog_model_id_is_local_to_its_provider(tmp_path):
    # [FUNCTION DOC]
    # The openclaw config's provider catalog lists the provider-local model id,
    # the capture base URL plus /v1 and the session API key.
    from openenv.harbor.seams import get

    selected, kwargs, env, _ = get("openclaw").resolve(
        base_url="https://capture.example",
        session="session-test",
        model="Qwen/Qwen3.5-4B",
    )
    agent = install_fixes.InterceptOpenClaw(
        logs_dir=tmp_path, model_name=selected, extra_env=env, **kwargs
    )
    config = agent._build_full_openclaw_config()
    provider, model_id = selected.split("/", 1)
    catalog = config["models"]["providers"][provider]
    assert any(model["id"] == model_id for model in catalog["models"])
    assert catalog["baseUrl"] == "https://capture.example/v1"
    assert catalog["apiKey"] == "session-test"
