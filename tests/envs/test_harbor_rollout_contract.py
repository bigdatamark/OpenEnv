# [MODULE DOC]
# Tests for openenv.harbor.rollout (reward selection, trial config building, run_rollout failure
# paths, trace discovery) and openenv.harbor.seams (agent-facing model naming, resolve, dialects).
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward selection and the per-agent seam.

Both decide things that are wrong silently rather than loudly. A reward picked by the wrong rule
trains the policy on the wrong objective and the run looks healthy throughout; a model name formatted
the way one harness expects and another does not resolves to a different model with no error.
"""

from __future__ import annotations

import pytest

rollout = pytest.importorskip("openenv.harbor.rollout")
seams = pytest.importorskip("openenv.harbor.seams")

_pick_reward = rollout._pick_reward


@pytest.mark.parametrize(
    "harness,version",
    [
        ("opencode", "1.18.31"),
        ("claude-code", "2.1.270"),
        ("codex", "0.154.0"),
        ("mini-swe-agent", "2.4.6"),
    ],
)
def test_deployment_pin_reaches_harbor_installer(
    monkeypatch, tmp_path, harness, version
):
    # [FUNCTION DOC]
    # Parametrized per harness: a version pin in OPENENV_HARBOR_AGENT_VERSIONS (JSON) ends up in the
    # built trial config's agent kwargs "version". Skipped without harbor trial config.
    import json

    pytest.importorskip("harbor.models.trial.config")
    monkeypatch.setenv("OPENENV_HARBOR_AGENT_VERSIONS", json.dumps({harness: version}))
    config = rollout.build_trial_config(
        task_dir=tmp_path,
        harness=harness,
        sandbox="e2b",
        intercept_url="http://proxy",
        session_id="test",
        model="Qwen/Qwen3.5-2B",
        trial_name="test",
        trials_dir=tmp_path,
    )
    assert config.agent.kwargs["version"] == version


@pytest.mark.parametrize(
    "pin", ['{"codex":"latest"}', '{"codex":"0.1;echo bad"}', "[]"]
)
def test_invalid_installer_pin_rejected(monkeypatch, tmp_path, pin):
    # [FUNCTION DOC]
    # Non-numeric or malformed pins make build_trial_config raise ValueError ("exact numeric versions").
    pytest.importorskip("harbor.models.trial.config")
    monkeypatch.setenv("OPENENV_HARBOR_AGENT_VERSIONS", pin)
    with pytest.raises(ValueError, match="exact numeric versions"):
        rollout.build_trial_config(
            task_dir=tmp_path,
            harness="codex",
            sandbox="e2b",
            intercept_url="http://proxy",
            session_id="test",
            model="Qwen/Qwen3.5-2B",
            trial_name="test",
            trials_dir=tmp_path,
        )


def test_reward_preference_uses_the_existing_verifier_result():
    # [FUNCTION DOC]
    # _pick_reward with a comma-separated preference picks the first present key (an exact key match
    # such as "a,b" wins), returning (value, key).
    assert _pick_reward({"reward": 0.0}, "correctness,reward") == (0.0, "reward")
    assert _pick_reward({"correctness": 0.0, "reward": 1.0}, "correctness,reward") == (
        0.0,
        "correctness",
    )
    assert _pick_reward({"a,b": 0.25, "a": 1.0}, "a,b") == (0.25, "a,b")


# --- reward selection -------------------------------------------------------
def test_no_rewards_means_ungraded_not_zero():
    # [FUNCTION DOC]
    # Empty rewards give (None, "").
    """The verifier never ran. Scoring that as 0 makes a dead sandbox look like a wrong answer."""
    value, key = _pick_reward({})
    assert value is None and key == ""


def test_a_single_key_is_used_whatever_it_is_called():
    # [FUNCTION DOC]
    # A single reward key is used regardless of name.
    assert _pick_reward({"accuracy": 0.75}) == (0.75, "accuracy")


def test_a_key_named_reward_wins_when_several_exist():
    # [FUNCTION DOC]
    # With several keys, "reward" is chosen.
    value, key = _pick_reward({"reward": 1.0, "steps": 12.0, "latency": 0.4})
    assert (value, key) == (1.0, "reward")


def test_ambiguous_rewards_raise_rather_than_guess():
    # [FUNCTION DOC]
    # Several keys without "reward" raise ValueError mentioning reward_key.
    """Inventing a rule to combine keys is inventing reward semantics."""
    with pytest.raises(ValueError, match="reward_key"):
        _pick_reward({"correctness": 1.0, "style": 0.5})


def test_an_explicit_key_overrides_the_default_rule():
    # [FUNCTION DOC]
    # An explicit key is used even when "reward" exists.
    assert _pick_reward({"reward": 1.0, "partial": 0.25}, "partial") == (
        0.25,
        "partial",
    )


def test_an_unknown_explicit_key_raises_and_names_the_options():
    # [FUNCTION DOC]
    # An absent explicit key raises ValueError ("not in").
    with pytest.raises(ValueError, match="not in"):
        _pick_reward({"reward": 1.0}, "nope")


def test_zero_is_a_real_reward_and_is_not_treated_as_missing():
    # [FUNCTION DOC]
    # A 0.0 reward is returned, not treated as missing.
    """`0.0` is falsy; a truthiness check here would report a wrong answer as ungraded."""
    value, key = _pick_reward({"reward": 0.0})
    assert value == 0.0 and key == "reward"


def test_values_are_coerced_to_float():
    # [FUNCTION DOC]
    # Integer rewards are returned as float.
    value, _ = _pick_reward({"reward": 1})
    assert isinstance(value, float)


# --- the model name a harness is given --------------------------------------
@pytest.mark.parametrize(
    "served,expected",
    [
        ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),  # the case that caused real breakage
        ("org/team/model", "model"),
        ("plain-model", "plain-model"),
    ],
)
def test_agent_facing_model_takes_the_leaf(served, expected):
    # [FUNCTION DOC]
    # Parametrized: agent_facing_model returns the last path segment of a served model id.
    """A served id with a slash produces `provider/org/model` once a seam adds its prefix.

    gemini-cli requires exactly `provider/model_name`, cline-cli wants `provider:model`, and
    harnesses disagree about splitting on the first or the last slash, so the same string resolves
    to different models. Taking the leaf removes the ambiguity. Safe because the harness-facing name
    only has to route to the proxy: the proxy overwrites `model` with the served id upstream.
    """
    assert seams.agent_facing_model(served) == expected


def test_an_empty_served_model_raises_rather_than_producing_an_empty_name():
    # [FUNCTION DOC]
    # Empty served model raises ValueError ("empty").
    """An empty name would be formatted into the seam's prefix and sent as `openai/`.

    The harness would then fail somewhere downstream with a confusing error, so this refuses at the
    point where the cause is still obvious.
    """
    with pytest.raises(ValueError, match="empty"):
        seams.agent_facing_model("")


def test_resolve_cannot_be_bypassed_by_a_seam():
    # [FUNCTION DOC]
    # For every seam, resolve never leaks the slashed model id and returns dicts for kwargs and envs.
    """Normalisation lives inside `resolve`, so no seam can format the raw served id by mistake."""
    for name in seams.SEAMS:
        seam = seams.get(name)
        model_name, kwargs, agent_env, proc_env = seam.resolve(
            base_url="http://proxy:8100", session="sess-123", model="Qwen/Qwen3.5-9B"
        )
        assert "Qwen/Qwen3.5-9B" not in model_name, f"{name} leaked the slashed id"
        assert isinstance(kwargs, dict)
        assert isinstance(agent_env, dict)
        assert isinstance(proc_env, dict)


def test_resolve_threads_the_session_id_through_as_the_key():
    # [FUNCTION DOC]
    # The opencode seam's resolved kwargs/envs contain both the session id and the base URL.
    """The API key is the capture session id: that is the whole multiplexing scheme."""
    seam = seams.get("opencode")
    _, kwargs, agent_env, proc_env = seam.resolve(
        base_url="http://proxy:8100", session="sess-abc", model="m"
    )
    blob = f"{kwargs}{agent_env}{proc_env}"
    assert "sess-abc" in blob
    assert "http://proxy:8100" in blob


def test_every_seam_declares_a_known_dialect():
    # [FUNCTION DOC]
    # Every seam's dialect is one of openai_chat, openai_responses, anthropic, google.
    known = {"openai_chat", "openai_responses", "anthropic", "google"}
    for name in seams.SEAMS:
        assert seams.get(name).dialect in known, f"{name} has an unknown dialect"


def test_unknown_seam_raises():
    # [FUNCTION DOC]
    # get on an unknown harness raises KeyError or ValueError.
    with pytest.raises((KeyError, ValueError)):
        seams.get("definitely-not-a-harness")


# --- concurrency ------------------------------------------------------------
def test_process_env_lock_is_usable_from_several_event_loops():
    # [FUNCTION DOC]
    # Asserts rollout._PROC_ENV_LOCK is a threading lock and that 8 threads each running their own event
    # loop can acquire it without errors.
    """It guards `os.environ`, which is global to the process, not to an event loop.

    Rollouts arrive on several loops: the env server answers each request on its own, and a caller
    using `asyncio.run` per rollout creates another. An `asyncio.Lock` binds to the first loop that
    uses it and then raises "is bound to a different event loop" for every other one, which failed
    100% of concurrent rollouts while passing every sequential test.
    """
    import asyncio
    import threading

    lock = rollout._PROC_ENV_LOCK
    assert isinstance(lock, type(threading.Lock())), "must not be an asyncio primitive"

    errors: list[str] = []
    done: list[int] = []

    def worker(tag: int) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs body in a new event loop; records the tag on success or the exception text on failure.
        async def body() -> None:
            # [FUNCTION DOC]
            # HELPER: Holds the lock across a 5ms async sleep.
            with lock:
                await asyncio.sleep(0.005)

        try:
            asyncio.run(body())
            done.append(tag)
        except Exception as exc:  # noqa: BLE001 - the failure being pinned
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:2]
    assert len(done) == 8


# --- ok must mean usable ----------------------------------------------------
def test_a_fatal_capture_finding_makes_the_rollout_unusable():
    # [FUNCTION DOC]
    # run_rollout on a nonexistent task with a fake registry returns ok False, reward None and an error
    # instead of raising.
    """`ok=True` with zero model calls is the worst shape a result can take.

    A rollout that reached the verifier but produced no captured calls used to come back `ok=True`
    with zero trainable tokens, because only trace reconciliation could clear `ok` and it agreed:
    both sides were empty. One such rollout carried reward=1.0, so a trainer filtering on `ok` would
    have accepted a row containing nothing at all alongside a positive reward.
    """
    import asyncio
    from pathlib import Path

    result = asyncio.run(
        rollout.run_rollout(
            task_dir=Path("/definitely/not/a/task"),
            harness="opencode",
            sandbox="e2b",
            registry=_FakeRegistry(),
            intercept_url="http://127.0.0.1:9",
            model="m",
            trials_dir=Path("/tmp"),
        )
    )
    # The trial cannot even be built, so this must be a result rather than an exception.
    assert result.ok is False
    assert result.reward is None
    assert result.error


class _FakeSession:
    # [CLASS DOC]
    # HELPER: Minimal capture session with a fixed id and no graph.
    session_id = "sess-test"

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets graph to None.
        self.graph = None


def test_invalid_eval_policy_returns_failure_before_allocating_session(tmp_path):
    # [FUNCTION DOC]
    # An eval rollout with NaN temperature returns a ValueError failure mentioning "finite" and no
    # session id.
    import asyncio

    from openenv.core.harness.capture.sessions import SessionRegistry

    registry = SessionRegistry()
    result = asyncio.run(
        rollout.run_rollout(
            task_dir=tmp_path,
            harness="opencode",
            sandbox="e2b",
            registry=registry,
            intercept_url="http://127.0.0.1:9",
            model="m",
            trials_dir=tmp_path,
            purpose="eval",
            eval_sampling={"temperature": float("nan")},
        )
    )
    assert not result.ok
    assert result.reward is None
    assert result.exception_type == "ValueError"
    assert "finite" in result.error
    assert not result.session_id


class _FakeRegistry:
    # [CLASS DOC]
    # HELPER: Minimal SessionRegistry stand-in for run_rollout.
    """Enough of `SessionRegistry` for `run_rollout` to mint and delete a session."""

    def create(self, session_id=None, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns a new _FakeSession, ignoring arguments.
        return _FakeSession()

    def delete(self, _session_id):
        # [FUNCTION DOC]
        # HELPER: Always returns True.
        return True


def test_hf_routing_suffix_gets_a_stable_harness_alias():
    # [FUNCTION DOC]
    # A served id with a ":provider" routing suffix gets a deterministic, safe (<=63 chars, restricted
    # charset) alias distinct from similar ids; terminus-2 resolves to "hosted_vllm/" plus that alias.
    import re

    served = "Qwen/Qwen3.5-9B:together"
    alias = seams.agent_facing_model(served)
    assert re.fullmatch(r"[A-Za-z0-9._-]{1,63}", alias)
    assert alias == seams.agent_facing_model(served)
    assert alias != seams.agent_facing_model("Qwen/Qwen3.5-9B-together")
    assert alias != seams.agent_facing_model("Qwen/Qwen3.5-9B:other")
    model, _, _, _ = seams.get("terminus-2").resolve(
        base_url="https://capture.example", session="test", model=served
    )
    assert model == "hosted_vllm/" + alias


def test_long_harness_alias_is_bounded_without_colliding_on_prefix():
    # [FUNCTION DOC]
    # Long names sharing a prefix get distinct 63-character aliases.
    left = seams.agent_facing_model("a" * 100 + "x")
    right = seams.agent_facing_model("a" * 100 + "y")
    assert len(left) == len(right) == 63
    assert left != right


def test_session_observer_gets_owned_id_and_cleanup_survives_observer_failure(tmp_path):
    # [FUNCTION DOC]
    # on_session_created observer receives the rollout's own session id; when it raises, the result is a
    # failure with that error, the rollout's session is deleted and an unrelated session survives.
    import asyncio

    from openenv.core.harness.capture.sessions import SessionRegistry

    registry = SessionRegistry()
    unrelated = registry.create("other-rollout")
    observed = []

    def observe(session_id):
        # [FUNCTION DOC]
        # HELPER: Asserts the session exists and is not the unrelated one, records it, then raises
        # RuntimeError.
        assert registry.get(session_id) is not None
        assert session_id != unrelated.session_id
        observed.append(session_id)
        raise RuntimeError("observer failed")

    result = asyncio.run(
        rollout.run_rollout(
            task_dir=tmp_path,
            harness="opencode",
            sandbox="e2b",
            registry=registry,
            intercept_url="http://127.0.0.1:9",
            model="m",
            trials_dir=tmp_path,
            purpose="eval",
            capture_level="text",
            on_session_created=observe,
        )
    )
    assert observed == [result.session_id]
    assert result.ok is False and result.error == "observer failed"
    assert registry.get(result.session_id) is None
    assert registry.get(unrelated.session_id) is unrelated


def test_verifier_exclusions_survive_capture_export(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # Monkeypatches harbor's Trial and build_trial_config so the verifier returns rewards with a None
    # "aux" key; asserts reward 1.0 and a finding reporting the dropped "aux" key.
    import asyncio
    from types import SimpleNamespace

    from openenv.core.harness.capture.sessions import SessionRegistry

    trial_module = pytest.importorskip("harbor.trial.trial")

    class Trial:
        @classmethod
        # [CLASS DOC]
        # HELPER: Fake harbor Trial.
        async def create(cls, config):
            # [FUNCTION DOC]
            # HELPER: Async classmethod returning a new instance.
            return cls()

        async def run(self):
            # [FUNCTION DOC]
            # HELPER: Returns a namespace with verifier rewards reward=1.0 and aux=None.
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0, "aux": None})
            )

    monkeypatch.setattr(trial_module, "Trial", Trial)
    monkeypatch.setattr(rollout, "build_trial_config", lambda **kwargs: None)
    result = asyncio.run(
        rollout.run_rollout(
            task_dir=tmp_path,
            harness="opencode",
            sandbox="e2b",
            registry=SessionRegistry(),
            intercept_url="http://127.0.0.1:9",
            model="m",
            trials_dir=tmp_path,
        )
    )
    assert result.reward == 1.0
    assert any(
        "verifier keys dropped" in item and "aux" in item for item in result.findings
    )


@pytest.mark.parametrize("name", ["my trials", "a+b", "user@host", "c:d"])
def test_trial_uri_with_escaped_characters_still_finds_the_trace(tmp_path, name):
    # [FUNCTION DOC]
    # Parametrized on names needing percent-encoding: _trial_dir resolves a file URI back to the trial
    # directory and load_trace detects an ATIF trajectory there. Writes files under tmp_path.
    """Harbor sets `trial_uri` with `Path.as_uri()`, which percent-encodes these characters."""
    import json
    from types import SimpleNamespace

    trial_dir = tmp_path / name / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps({"steps": []}))
    trial_result = SimpleNamespace(trial_uri=trial_dir.resolve().as_uri())

    found = rollout._trial_dir(trial_result, tmp_path / name, "trial")
    assert found == trial_dir.resolve()
    assert rollout.load_trace(found)[1] == "atif"
