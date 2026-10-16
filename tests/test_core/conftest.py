# [MODULE DOC]
# Shared pytest fixtures for openenv.core tests.
# SPDX-License-Identifier: BSD-3-Clause

import socket
import threading
import time

import pytest
import uvicorn


@pytest.fixture
def serve():
    # [FUNCTION DOC]
    # Fixture yielding a start(app) function that serves an ASGI app with uvicorn on a free
    # 127.0.0.1 port in a daemon thread and returns its base URL; on teardown asks every
    # started server to exit and joins its thread (10 s).
    """Serve an ASGI app on a free local port and return its URL."""
    servers = []

    def start(app) -> str:
        # [FUNCTION DOC]
        # HELPER: Binds a socket to port 0, runs a uvicorn Server on it in a daemon thread, waits
        # up to 15 s for startup (asserts it started) and returns "http://127.0.0.1:<port>".
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(
            target=lambda: server.run(sockets=[sock]), daemon=True
        )
        thread.start()
        servers.append((server, thread))
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        return f"http://127.0.0.1:{sock.getsockname()[1]}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)
