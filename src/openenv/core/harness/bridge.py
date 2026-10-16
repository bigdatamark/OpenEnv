# [MODULE DOC]
# Loopback MCP bridge: serves an environment's FastMCP tool server over
# streamable HTTP on 127.0.0.1 from a background thread, so a harness subprocess
# can call the environment's domain tools (but never reset/step/state). Also builds
# a renamed "view" server when tool-name conflict resolution renamed tools.
# SPDX-License-Identifier: BSD-3-Clause

"""Loopback MCP bridge exposing environment tools to a harness (RFC 005)."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from .adapter import HarnessError


async def _source_tools(mcp_server: Any) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Async lookup of a FastMCP server's tools across FastMCP versions.
    # If the server has get_tools() and it returns a dict, returns that dict; else if it
    # has list_tools(), returns a mapping name -> tool built from the list; else empty dict.
    # Inputs: mcp_server - a FastMCP server object.
    # Returns: mapping from tool name to tool object.
    """Return {name: FastMCP tool}, compatible with FastMCP 2.x and 3.x.

    Async twin of [`~openenv.core.env_server.mcp_environment.get_server_tools`],
    which wraps the same calls in `run_async_safely`; awaiting directly avoids
    spawning a thread when we are already on a loop.
    """
    if hasattr(mcp_server, "get_tools"):
        result = await mcp_server.get_tools()
        if isinstance(result, dict):
            return result
    if hasattr(mcp_server, "list_tools"):
        return {tool.name: tool for tool in await mcp_server.list_tools()}
    return {}


async def build_bridge_server(mcp_server: Any, renames: dict[str, str]) -> Any:
    # [FUNCTION DOC]
    # Returns a server exposing the environment's tools under injected (renamed) names.
    # Inputs: mcp_server - the env's FastMCP server; renames - mapping injected name ->
    # source name.
    # Returns: mcp_server unchanged when renames is empty; otherwise a FastMCPProxy named
    # "<server name or 'openenv'>-harness-view" with a ToolTransform renaming each source
    # tool to its served name. The proxy uses a StatefulProxyClient so each harness MCP
    # connection gets its own source session, and a lifespan that keeps a Client open on
    # the source server for the bridge's lifetime.
    # Raises: HarnessError if any source name in renames is not exposed by mcp_server.
    # Side effects: lazily imports fastmcp submodules.
    """
    Return a FastMCP server exposing the env's tools under their injected names.

    Tool-name conflict resolution can rename an environment tool before it is
    injected into a harness (`read_file` -> `env_read_file`). The harness then
    calls the new name, so the bridge has to answer to it: serving the source
    server unchanged would advertise a name that does not resolve.

    Args:
        mcp_server (`FastMCP`):
            The environment's own tool server.
        renames (`dict[str, str]`):
            Mapping of injected name to source name. Empty means no renames.

    Returns:
        `FastMCP`: `mcp_server` itself when there is nothing to rename,
        otherwise a view of it whose tools carry the injected names.
    """
    if not renames:
        return mcp_server

    from fastmcp import Client
    from fastmcp.server.providers.proxy import FastMCPProxy, StatefulProxyClient
    from fastmcp.server.transforms import ToolTransform
    from fastmcp.tools.tool_transform import ToolTransformConfig

    source_tools = await _source_tools(mcp_server)
    missing = sorted(set(renames.values()) - set(source_tools))
    if missing:
        raise HarnessError(
            "Cannot rename tools that the MCP server does not expose: "
            + ", ".join(missing)
        )

    @asynccontextmanager
    async def lifespan(server):
        # [FUNCTION DOC]
        # HELPER: Async context manager used as the proxy server's lifespan; opens a fastmcp
        # Client on the source server for the duration and yields an empty state dict.
        # Preserve source resources across harness MCP reconnects until this
        # bridge stops, just as when serving the source server directly.
        async with Client(mcp_server):
            yield {}

    # Forward callbacks in the current request context. Each harness MCP
    # connection gets its own source session, closed on disconnect.
    client = StatefulProxyClient(mcp_server)
    view = FastMCPProxy(
        name=f"{getattr(mcp_server, 'name', 'openenv')}-harness-view",
        client_factory=client.new_stateful,
        lifespan=lifespan,
    )
    view.add_transform(
        ToolTransform(
            {
                source: ToolTransformConfig(name=served)
                for served, source in renames.items()
            }
        )
    )
    return view


class HarnessMCPBridge:
    # [CLASS DOC]
    # Runs an in-process FastMCP server as a uvicorn ASGI app on host (default loopback)
    # with an OS-assigned ephemeral port, in a daemon thread. Tracks the url, thread,
    # uvicorn server and any startup error. url is None when not running.
    """
    Serve an in-process FastMCP server over loopback HTTP for harness use.

    The bridge runs the environment's FastMCP tool server as a
    streamable-HTTP ASGI app on `127.0.0.1` with an ephemeral port, inside a
    daemon thread with its own event loop. It carries only the tool surface:
    the harness subprocess can reach the environment's domain tools but never
    OpenEnv's orchestration API (`reset`/`step`/`state`), which lives on a
    different server entirely. This makes the RFC 005 security boundary
    structural rather than filter-based.

    Args:
        mcp_server (`FastMCP`):
            The environment's FastMCP server to expose.
        host (`str`, *optional*, defaults to `"127.0.0.1"`):
            Interface to bind. Keep this loopback-only.

    Examples:

    ```python
    bridge = HarnessMCPBridge(env.mcp_server)
    url = bridge.start()
    # pass url to the harness adapter's inject_tools()
    bridge.stop()
    ```
    """

    def __init__(self, mcp_server: Any, host: str = "127.0.0.1"):
        # [FUNCTION DOC]
        # Stores the MCP server and host; initializes url, thread, uvicorn server and
        # startup error to None. No side effects.
        self._mcp_server = mcp_server
        self._host = host
        self._url: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[Any] = None
        self._startup_error: Optional[BaseException] = None

    @property
    def url(self) -> Optional[str]:
        # [FUNCTION DOC]
        # Returns the current MCP endpoint URL, or None when not running.
        """The bridge's MCP endpoint URL, or `None` when not running."""
        return self._url

    def start(self, timeout_s: float = 10.0) -> str:
        # [FUNCTION DOC]
        # Starts serving and returns the endpoint URL "http://<host>:<port>/mcp".
        # Idempotent: if the server thread is alive returns the existing URL.
        # Behavior: binds a socket to (host, 0) with SO_REUSEADDR, builds the server's
        # http_app(), creates a uvicorn Server (log level warning, lifespan on), runs it in a
        # daemon thread on that socket, then polls every 10ms until server.started.
        # Inputs: timeout_s - max seconds to wait for startup (default 10).
        # Raises: HarnessError if the thread is alive but still stopping; if the server thread
        # dies or records an exception during startup (state torn down); or on timeout (stop()
        # is called first).
        # Side effects: opens a socket, starts a thread, mutates instance state.
        """
        Start serving and return the MCP endpoint URL.

        Idempotent: returns the existing URL if already running.

        Args:
            timeout_s (`float`, *optional*, defaults to `10.0`):
                Maximum time to wait for the server to come up.

        Returns:
            `str` URL of the MCP endpoint, e.g. `"http://127.0.0.1:54321/mcp"`.

        Raises:
            [`~openenv.core.harness.adapter.HarnessError`]:
                If the server fails to start within the timeout.
        """
        if self._thread is not None and self._thread.is_alive():
            if self._uvicorn_server.should_exit:
                raise HarnessError("MCP bridge is still stopping")
            assert self._url is not None
            return self._url

        import uvicorn

        app = self._mcp_server.http_app()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, 0))
        port = sock.getsockname()[1]

        config = uvicorn.Config(app, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        self._startup_error = None

        def _serve() -> None:
            # [FUNCTION DOC]
            # HELPER: Thread target; runs the uvicorn server on the bound socket, records any
            # exception (including BaseException) in self._startup_error, and always closes the
            # socket ignoring OSError.
            try:
                server.run(sockets=[sock])
            except BaseException as exc:  # surfaced to start() below
                self._startup_error = exc
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        self._thread = thread

        deadline = time.monotonic() + timeout_s
        while not server.started:
            if self._startup_error is not None or not thread.is_alive():
                self._teardown()
                raise HarnessError(f"MCP bridge failed to start: {self._startup_error}")
            if time.monotonic() > deadline:
                self.stop()
                raise HarnessError(f"MCP bridge did not start within {timeout_s}s")
            time.sleep(0.01)

        self._url = f"http://{self._host}:{port}/mcp"
        return self._url

    def stop(self, timeout_s: float = 5.0) -> None:
        # [FUNCTION DOC]
        # Stops the bridge. Idempotent when never started.
        # Behavior: sets graceful-shutdown timeout to timeout_s/4 and should_exit; joins the
        # thread for timeout_s/2; if still alive sets force_exit and joins for the remaining
        # time to the overall deadline; then clears url/thread/server state.
        # Inputs: timeout_s - overall budget in seconds (default 5, negatives treated as 0).
        # Raises: HarnessError if the thread is still alive after the deadline; state is kept
        # so callers can retry.
        """
        Stop the bridge server.

        Idempotent: safe to call when the bridge was never started.

        Args:
            timeout_s (`float`, *optional*, defaults to `5.0`):
                Maximum time to wait for the server thread to exit.

        Raises:
            [`~openenv.core.harness.adapter.HarnessError`]:
                If shutdown times out. The server remains tracked so callers
                can retry `stop()`; `start()` rejects it while still stopping.
        """
        server = self._uvicorn_server
        thread = self._thread
        deadline = time.monotonic() + max(timeout_s, 0.0)
        if server is not None:
            # Cancel lingering HTTP streams before the thread join expires.
            # Reserve time for ASGI lifespan cleanup and forced shutdown.
            server.config.timeout_graceful_shutdown = max(timeout_s, 0.0) / 4
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(timeout_s, 0.0) / 2)
            if thread.is_alive() and server is not None:
                server.force_exit = True
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                # Keep ownership so callers can retry, and start() cannot
                # replace a server that still holds resources.
                raise HarnessError(f"MCP bridge did not stop within {timeout_s}s")
        self._teardown()

    def _teardown(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Clears url, thread and uvicorn server references (does not stop anything).
        self._url = None
        self._thread = None
        self._uvicorn_server = None


__all__ = ["build_bridge_server", "HarnessMCPBridge"]
