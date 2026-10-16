# [MODULE DOC]
# Tests for hosted (Hugging Face Space) serving of the Harbor service: public
# URL derivation from SPACE_HOST/SPACE_ID, mounting the capture proxy under
# CAPTURE_MOUNT instead of forwarding, cleanup after a failed forwarder, and
# the Space entry point defaulting to the "text" capture level when the
# endpoint cannot be probed. Skipped without openenv.harbor.serving.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serving on a hosted platform, where there is one port and one URL.

Locally the capture proxy runs on its own port and is published to the sandbox. A Space exposes
exactly one port and already has a public URL, so the proxy is mounted onto the env server's app
instead and nothing is forwarded. These tests pin that split, because getting it wrong is not a
crash: it is a deployment that quietly opens a second listener it cannot publish.
"""

from __future__ import annotations

import pytest

serving = pytest.importorskip("openenv.harbor.serving")

CAPTURE_MOUNT = serving.CAPTURE_MOUNT
HarborService = serving.HarborService
space_public_url = serving.space_public_url

# Never contacted: nothing here reaches an engine.
UNUSED_LLM = "http://127.0.0.1:9/v1"


@pytest.fixture(autouse=True)
def _clear_space_env(monkeypatch):
    # [FUNCTION DOC]
    # Autouse fixture: unsets SPACE_HOST and SPACE_ID.
    """Tests must not inherit a Space identity from the developer's shell."""
    monkeypatch.delenv("SPACE_HOST", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)


def test_no_space_means_no_public_url():
    # [FUNCTION DOC]
    # Without Space env vars the public URL is "".
    assert space_public_url() == ""


def test_space_host_is_used_verbatim(monkeypatch):
    # [FUNCTION DOC]
    # SPACE_HOST becomes https://<host>.
    monkeypatch.setenv("SPACE_HOST", "owner-env.hf.space")
    assert space_public_url() == "https://owner-env.hf.space"


def test_space_host_tolerates_a_scheme_already_present(monkeypatch):
    # [FUNCTION DOC]
    # A scheme and trailing slash in SPACE_HOST are normalised.
    monkeypatch.setenv("SPACE_HOST", "https://owner-env.hf.space/")
    assert space_public_url() == "https://owner-env.hf.space"


def test_space_id_is_slugged_when_host_is_absent(monkeypatch):
    # [FUNCTION DOC]
    # SPACE_ID is lowercased and non-alphanumerics become dashes in the hf.space host.
    """`SPACE_ID` is always set; the hostname lowercases and dash-separates it."""
    monkeypatch.setenv("SPACE_ID", "AdithyaSK/harbor_data.agent-env")
    assert space_public_url() == "https://adithyask-harbor-data-agent-env.hf.space"


def test_hosted_start_mounts_and_never_forwards(monkeypatch):
    # [FUNCTION DOC]
    # On a Space, start mounts the proxy, returns the public URL plus mount path,
    # never builds a forwarder (patched to raise), and stop is safe.
    """The regression that got a Space flagged: a hosted deployment must not forward."""
    monkeypatch.setenv("SPACE_ID", "owner/env")

    def explode(*_args, **_kwargs):
        # [FUNCTION DOC]
        # Raises AssertionError if a forwarder is created.
        raise AssertionError("a hosted deployment must not create a forwarder")

    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder", explode, raising=False
    )

    service = HarborService(llm_url=UNUSED_LLM, model="m", datasets=[])
    url = service.start()

    assert service.mounted is True
    assert url == f"https://owner-env.hf.space{CAPTURE_MOUNT}"
    # No port was bound, so stop() must be safe even though start() never launched a server.
    service.stop()


def test_mounted_capture_answers_under_the_prefix():
    # [FUNCTION DOC]
    # A capture app mounted on a FastAPI host answers /health and rejects an
    # unknown session key with 401 under the prefix.
    """Mounting strips the prefix, so every dialect route keeps working unchanged."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openenv.core.harness.capture.server import create_app

    host = fastapi.FastAPI()
    host.mount(CAPTURE_MOUNT, create_app(llm_url=UNUSED_LLM, model="m"))
    client = TestClient(host)

    assert client.get(f"{CAPTURE_MOUNT}/health").json()["status"] == "ok"

    # The proxy's catch-all must match /v1/chat/completions, not /capture/v1/chat/completions.
    response = client.post(
        f"{CAPTURE_MOUNT}/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer not-a-session"},
    )
    # 401 rather than 404 proves it routed to the proxy and was rejected on identity, which is also
    # what stops a publicly mounted proxy being an open relay.
    assert response.status_code == 401
    assert "unknown API key" in response.json()["error"]["message"]


def test_a_failed_forwarder_does_not_leave_the_capture_server_running(monkeypatch):
    # [FUNCTION DOC]
    # If forwarder creation fails, start re-raises and stops the capture server,
    # leaving no public URL.
    """A half-started service poisons every later attempt.

    The capture server binds its port and starts a thread before the forwarder is built. If the
    forwarder then fails, leaving that server up means the next `start()` fails on a port conflict
    that says nothing about the real error.
    """
    stopped: list[bool] = []

    class FakeCapture:
        # [CLASS DOC]
        # Capture server stub recording stop calls.
        port = 8123

        def start(self):
            # [FUNCTION DOC]
            # No-op.
            pass

        def stop(self):
            # [FUNCTION DOC]
            # Records that stop was called.
            stopped.append(True)

    def explode(*_args, **_kwargs):
        # [FUNCTION DOC]
        # Raises RuntimeError("cloudflared is not installed").
        raise RuntimeError("cloudflared is not installed")

    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder", explode, raising=False
    )

    service = HarborService(llm_url=UNUSED_LLM, model="m", datasets=[])
    service.capture = FakeCapture()

    with pytest.raises(RuntimeError, match="cloudflared"):
        service.start()

    assert stopped == [True], "the capture server was left holding its port"
    assert service.public_url in (None, "")


def test_a_space_that_cannot_probe_does_not_claim_to_be_trainable(
    monkeypatch, tmp_path
):
    # [FUNCTION DOC]
    # With an ambiguous model list and HarborService/build_app patched, running
    # harbor_env.server.app via runpy builds the service with capture level "text".
    """The Space entry point defaulted `_CAPTURE_LEVEL` to "tokens" and only corrected it when a model
    resolved AND the probe succeeded. An ambiguous model list, an unset model or a raising probe left
    it at token level, so the proxy was built for capture and every rollout was stamped trainable —
    the exact mislabelling the capture level exists to prevent. Unknown must mean the weaker tier.
    """
    import runpy

    monkeypatch.setenv("OPENENV_LLM_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENENV_MODEL", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    # Serves two models and names neither, so nothing resolves and the probe never runs.
    #
    # Patched through the module OBJECT, not the dotted string: the package re-exports a function
    # called `validate_llm` which shadows the same-named submodule, so the string form resolves to the
    # function and monkeypatch fails with "'function' object has no attribute 'list_models'".
    import importlib

    validate_llm_mod = importlib.import_module(
        "openenv.core.harness.capture.validate_llm"
    )
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["one", "two"])
    # The service must not be started for real; capture the level it would have been built with.
    built: dict = {}

    class FakeService:
        # [CLASS DOC]
        # HarborService stub recording constructor kwargs.
        def __init__(self, **kwargs):
            # [FUNCTION DOC]
            # Records kwargs.
            built.update(kwargs)

        def start(self):
            # [FUNCTION DOC]
            # Returns "".
            return ""

        @classmethod
        def set_current(cls, _service):
            # [FUNCTION DOC]
            # Classmethod no-op.
            pass

    monkeypatch.setattr(serving, "HarborService", FakeService)
    monkeypatch.setattr(serving, "build_app", lambda **kwargs: kwargs)

    runpy.run_module("harbor_env.server.app", run_name="not_main")
    assert built.get("capture_level") == "text", (
        "an unprobed endpoint must default to the weakest tier, never to tokens"
    )
