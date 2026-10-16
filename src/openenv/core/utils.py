# [MODULE DOC]
# HELPER: small async/URL utilities shared by OpenEnv core clients.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for OpenEnv core."""

import asyncio
import concurrent.futures


def run_async_safely(coro):
    # [FUNCTION DOC]
    # HELPER: runs a coroutine to completion from sync code and returns its result.
    # Inputs: coro - an awaitable coroutine object.
    # Returns: whatever the coroutine returns.
    # Behavior: if no event loop is running in this thread, uses asyncio.run directly;
    # otherwise spins up a one-off thread pool and runs asyncio.run in a worker thread,
    # blocking the caller until done (so the caller's loop is blocked meanwhile).
    # Raises: propagates any exception raised by the coroutine.
    """
    Run an async coroutine safely from any context.

    This handles the case where we may already be inside an async event loop
    (e.g., when called from an async framework). In that case, asyncio.run()
    would fail, so we use a ThreadPoolExecutor to run in a separate thread.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context - run in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        # No async context - use asyncio.run() directly
        return asyncio.run(coro)


def convert_to_ws_url(url: str) -> str:
    # [FUNCTION DOC]
    # HELPER: converts a base URL into a WebSocket URL.
    # Strips trailing slashes, maps http:// -> ws:// and https:// -> wss://, leaves
    # ws/wss URLs as-is, and prefixes ws:// when no scheme is present. Returns the string.
    """
    Convert an HTTP/HTTPS URL to a WS/WSS URL.

    Args:
        url: The URL to convert.

    Returns:
        The converted WebSocket URL.
    """
    ws_url = url.rstrip("/")
    if ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    elif ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
        ws_url = "ws://" + ws_url
    return ws_url
