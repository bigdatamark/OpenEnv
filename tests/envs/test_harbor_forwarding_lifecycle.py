# [MODULE DOC]
# Tests that GradioForwarder drains its tunnel child's stdout and stderr so
# the child never blocks, and stop() terminates only its own tunnel.
"""A live forwarder must not block when its child fills stdout or stderr."""

import subprocess
import sys
import time
from types import SimpleNamespace

from openenv.core.harness.capture.forwarding import GradioForwarder


def test_gradio_drains_both_pipes_and_stops_only_its_child(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # Patches gradio.networking/tunneling in sys.modules with fakes whose
    # setup_tunnel spawns a Python child writing 1 MiB to each pipe then a marker
    # file; asserts the marker appears within 5 s, stop() ends the child and keeps
    # the unrelated tunnel, and a second stop is harmless. Cleans up processes in
    # finally. Side effects: real subprocesses and file writes in tmp_path.
    finished = tmp_path / "both-pipes-written"
    children = []
    unrelated = SimpleNamespace(share_token="other", proc=None)
    tunnels = [unrelated]

    def setup_tunnel(**kwargs):
        # [FUNCTION DOC]
        # Fake: starts the child process, registers a tunnel record with a kill that
        # terminates it, and returns a fake share URL.
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, sys, time; "
                    "os.write(1, b'x' * 1048576 + b'\\n'); "
                    "os.write(2, b'y' * 1048576 + b'\\n'); "
                    "pathlib.Path(sys.argv[1]).write_text('done'); time.sleep(60)"
                ),
                str(finished),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        children.append(process)
        tunnel = SimpleNamespace(share_token=kwargs["share_token"], proc=process)

        def kill():
            # [FUNCTION DOC]
            # Terminates the child and clears the tunnel's proc.
            process.terminate()
            tunnel.proc = None

        tunnel.kill = kill
        tunnels.append(tunnel)
        return "https://test.invalid"

    monkeypatch.setitem(
        sys.modules, "gradio.networking", SimpleNamespace(setup_tunnel=setup_tunnel)
    )
    monkeypatch.setitem(
        sys.modules, "gradio.tunneling", SimpleNamespace(CURRENT_TUNNELS=tunnels)
    )
    forwarder = GradioForwarder()
    try:
        assert forwarder.start(8123) == "https://test.invalid"
        deadline = time.monotonic() + 5
        while not finished.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert finished.exists(), "forwarder blocked on a full child-process pipe"
        forwarder.stop()
        assert children[0].poll() is not None
        assert unrelated in tunnels
        forwarder.stop()
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
