# [MODULE DOC]
# Tests for openenv.harbor.runner.run_batch lifecycle: the capture server must be stopped when the
# tunnel forwarder fails to start or stop, and every batch must build CaptureServer with an admin
# key (fresh random per batch, or explicit argument, or OPENENV_CAPTURE_ADMIN_KEY). All external
# dependencies are stubbed.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`openenv harbor rollout` must not strand its capture proxy when setup fails.

`run_batch` starts the capture server — which binds a port in a background thread — and only then
builds the tunnel the sandbox reaches it through. Anything that throws in between used to escape with
that thread still running, so the port stayed held and the *next* invocation died on a port conflict
that named nothing about the actual cause. Teardown had the mirror of the same gap: `forwarder.stop()`
ran before `capture.stop()` without a guard, so a tunnel that failed to shut down took the port with it.

The same entry point also publishes the proxy through a tunnel, so its session-management routes are
reachable by anyone with the URL. `_admin_ok` admits every caller while no admin key is set, which
made `run_batch` an open control plane: the tests at the bottom pin that it always hands
`CaptureServer` a key.

These tests stub every external dependency; no engine, no sandbox and no network are involved.
"""

from __future__ import annotations

import asyncio

import pytest

runner = pytest.importorskip("openenv.harbor.runner")


class FakeCaps:
    # [CLASS DOC]
    # HELPER: Stand-in for the startup capabilities object (llm info and sandbox lists).
    llm = {"model": "m", "capture_level": "tokens"}
    available_sandboxes = ("e2b",)
    sandboxes: tuple = ()


def wire(monkeypatch, tmp_path, forwarder):
    # [FUNCTION DOC]
    # HELPER: Monkeypatches runner.resolve_task_dirs to return [tmp_path], harbor startup.prepare to
    # return FakeCaps, runner.CaptureServer to a fake that records constructor kwargs and logs "capture"
    # on stop, and make_forwarder to build the given forwarder class with the stop log.
    # Returns (stop log list, list of CaptureServer kwargs).
    """Replace startup, the capture server and the dataset with local stubs.

    Returns the teardown log and the list of kwargs each `CaptureServer` was built with.
    """
    stopped: list[str] = []
    built: list[dict] = []

    monkeypatch.setattr(runner, "resolve_task_dirs", lambda _d: [tmp_path])
    monkeypatch.setattr(
        "openenv.harbor.startup.prepare", lambda **_k: FakeCaps(), raising=False
    )

    class FakeCapture:
        # [CLASS DOC]
        # HELPER: Fake CaptureServer recording kwargs and logging stops.
        registry = None
        inference = None

        def __init__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records kwargs and sets port (default 8100).
            built.append(kwargs)
            self.port = kwargs.get("port", 8100)

        def start(self):
            # [FUNCTION DOC]
            # HELPER: No-op.
            pass

        def stop(self):
            # [FUNCTION DOC]
            # HELPER: Appends "capture" to the stop log.
            stopped.append("capture")

    monkeypatch.setattr(runner, "CaptureServer", FakeCapture)
    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder",
        lambda _kind: forwarder(stopped),
        raising=False,
    )
    return stopped, built


class Quiet:
    # [CLASS DOC]
    # HELPER: Forwarder fake that starts (returning a dummy URL) and stops cleanly.
    """A forwarder that starts and stops without incident."""

    name = "gradio"

    def __init__(self, _stopped):
        # [FUNCTION DOC]
        # HELPER: Ignores the stop log.
        pass

    def start(self, _port):
        # [FUNCTION DOC]
        # HELPER: Returns "https://tunnel.invalid".
        return "https://tunnel.invalid"

    def stop(self):
        # [FUNCTION DOC]
        # HELPER: No-op.
        pass


def test_a_forwarder_that_cannot_start_releases_the_port(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # A forwarder whose start raises propagates the RuntimeError and the capture server is stopped.
    class Exploding:
        # [CLASS DOC]
        # HELPER: Forwarder whose start raises RuntimeError.
        name = "cloudflare"

        def __init__(self, _stopped):
            # [FUNCTION DOC]
            # HELPER: Ignores the stop log.
            pass

        def start(self, _port):
            # [FUNCTION DOC]
            # HELPER: Raises RuntimeError("cloudflared is not installed").
            raise RuntimeError("cloudflared is not installed")

    stopped, _ = wire(monkeypatch, tmp_path, Exploding)

    with pytest.raises(RuntimeError, match="cloudflared"):
        asyncio.run(
            runner.run_batch(llm_url="http://x/v1", dataset="d", task_indices=[0])
        )

    assert stopped == ["capture"], (
        "the capture server kept its port after a failed forwarder"
    )


def test_a_forwarder_that_cannot_stop_still_lets_the_port_go(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # A forwarder whose stop raises still lets capture.stop run afterwards; the error propagates and the
    # stop log is ["forwarder", "capture"]. run_rollout is replaced with None since no index is in range.
    class BadTeardown:
        # [CLASS DOC]
        # HELPER: Forwarder that starts fine but logs and raises on stop.
        name = "gradio"

        def __init__(self, stopped):
            # [FUNCTION DOC]
            # HELPER: Keeps the stop log.
            self._stopped = stopped

        def start(self, _port):
            # [FUNCTION DOC]
            # HELPER: Returns a dummy tunnel URL.
            return "https://tunnel.invalid"

        def stop(self):
            # [FUNCTION DOC]
            # HELPER: Appends "forwarder" then raises RuntimeError("tunnel already gone").
            self._stopped.append("forwarder")
            raise RuntimeError("tunnel already gone")

    stopped, _ = wire(monkeypatch, tmp_path, BadTeardown)
    monkeypatch.setattr(
        runner, "run_rollout", None
    )  # never reached: no indices are in range

    with pytest.raises(RuntimeError, match="tunnel already gone"):
        asyncio.run(
            runner.run_batch(llm_url="http://x/v1", dataset="d", task_indices=[99])
        )

    assert stopped == ["forwarder", "capture"], (
        "capture.stop() must run even when the forwarder's teardown raises"
    )


def run_empty_batch(**kwargs) -> None:
    # [FUNCTION DOC]
    # HELPER: Runs run_batch with an out-of-range task index (99) so only setup/teardown happen; extra
    # kwargs are forwarded.
    """Drive `run_batch` through capture + forwarder setup with no rollout in range."""
    asyncio.run(
        runner.run_batch(
            llm_url="http://x/v1", dataset="d", task_indices=[99], **kwargs
        )
    )


def test_run_batch_always_gates_the_control_plane(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # With no env key, CaptureServer receives a generated admin key of at least 32 characters.
    """The regression: the proxy went out on a public tunnel with `admin_key` unset.

    `_admin_ok` returns True for every caller while the key is unset, so `POST /sessions` minted keys
    for anyone (an open relay to the upstream) and `GET /sessions/{id}/rollout` served token-level
    training data to anyone who found the URL.
    """
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch()

    (kwargs,) = built
    assert kwargs.get("admin_key"), "CaptureServer was built without an admin key"
    assert len(kwargs["admin_key"]) >= 32


def test_run_batch_mints_a_fresh_key_per_batch(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # Two batches get different generated admin keys.
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch()
    run_empty_batch()

    first, second = (k["admin_key"] for k in built)
    assert first != second


def test_run_batch_forwards_an_explicit_admin_key(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # An explicit admin_key argument wins over the environment variable.
    monkeypatch.setenv("OPENENV_CAPTURE_ADMIN_KEY", "from-env")
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch(admin_key="explicit")

    assert built[0]["admin_key"] == "explicit", "the argument must beat the env var"


def test_run_batch_honours_the_admin_key_env_var(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # Without an argument, OPENENV_CAPTURE_ADMIN_KEY is used.
    """Same variable `HarborService` reads, so one setting covers `serve` and `rollout`."""
    monkeypatch.setenv("OPENENV_CAPTURE_ADMIN_KEY", "from-env")
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch()

    assert built[0]["admin_key"] == "from-env"
