# [MODULE DOC]
# FastAPI entry point for the browser environment: builds the app via create_app
# with the environment class as factory and at most one concurrent session.
"""FastAPI application for the Helium browser environment."""

import os
from contextlib import suppress

from openenv.core.env_server import create_app

from ..models import BrowserAction, BrowserObservation
from .helium_browser_environment import BrowserEnvironment

app = create_app(
    BrowserEnvironment,
    BrowserAction,
    BrowserObservation,
    env_name="helium_browser_env",
    max_concurrent_envs=1,
)


def main():
    # [FUNCTION DOC]
    # Runs uvicorn on port SBX_SERVICE_PORT (default 8000). If SBX_PROXY_DIR is set,
    # creates that directory, removes any stale "<port>.sock" there and serves on
    # that Unix domain socket; otherwise serves TCP on 0.0.0.0. Blocking.
    """Run the server over TCP or the Unix socket requested by HF Sandbox."""
    import uvicorn

    port = int(os.environ.get("SBX_SERVICE_PORT", "8000"))
    if proxy_dir := os.environ.get("SBX_PROXY_DIR"):
        socket_path = os.path.join(proxy_dir, f"{port}.sock")
        os.makedirs(proxy_dir, exist_ok=True)
        with suppress(FileNotFoundError):
            os.unlink(socket_path)
        uvicorn.run(app, uds=socket_path)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
