# [MODULE DOC]
# HELPER: Script spawned as a subprocess by the agentic harness tests to impersonate an external
# CLI harness. Run with one mode name argument; the MODES mapping dispatches to a mode_* function.
# Each mode exercises one process behavior (readiness, echoing, crashing, slow start, ignoring
# SIGTERM, leaking descendants, closing stdout, unicode output, or a full MCP-calling fake harness).

# SPDX-License-Identifier: BSD-3-Clause

"""Scripted subprocess behaviors for the agentic harness tests.

This module is not imported by the tests — it is *spawned* by them
(``python scripted_harness.py <mode>``) to play the role of an external CLI
harness, so the subprocess-facing code lives in a real, lintable file instead
of inline strings.

Modes:
  mcp              - full fake harness: announces readiness, then for each
                     JSON stdin line calls the env tool over the MCP bridge
                     (from the config file at ``$HARNESS_MCP_CONFIG``) and
                     emits JSON-line events ending in ``turn_complete``.
  echo             - print ``ready``, then echo each stdin line.
  slow-start       - sleep past any startup timeout before printing ``ready``.
  exit-now         - print to stderr and exit with code 3.
  exit-with-secret - print a fake credential to stderr, then exit with code 3.
  spawn-descendant - spawn a SIGTERM-ignoring child with inherited pipes.
  descendant       - print its PID, ignore SIGTERM, and sleep.
  exit-with-output - print multiple lines, including an unterminated final line.
  close-stdout     - close stdout on request, then wait on stdin without exiting.
  crash-after-echo - ``ready``, echo one line, then exit with code 1.
  ignore-sigterm   - ignore SIGTERM, print ``ready``, then sleep.
  unicode-echo     - like echo, but the reply carries non-ASCII characters.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time


def mode_mcp() -> None:
    # [FUNCTION DOC]
    # HELPER: Full fake harness. Reads a JSON config from the file named by HARNESS_MCP_CONFIG
    # (keys mcp_url and tools), prints a "ready" JSON event listing tools, then for each JSON stdin line
    # runs one turn and prints JSON-line events. Raises KeyError if the env var is missing.
    """Fake agentic harness speaking JSON lines over stdio."""
    with open(os.environ["HARNESS_MCP_CONFIG"]) as f:
        config = json.load(f)
    bridge_url = config.get("mcp_url")
    tool_names = config.get("tools", [])

    def emit(payload: dict) -> None:
        # [FUNCTION DOC]
        # HELPER: Prints the payload as one JSON line to stdout and flushes.
        print(json.dumps(payload), flush=True)

    emit({"event": "ready", "tools": tool_names})

    async def run_turn(message: str) -> None:
        # [FUNCTION DOC]
        # HELPER: If a bridge URL is configured and "add" is among the tools, emits a tool_call event, calls
        # "add" with a=2,b=3 over MCP via a fastmcp Client, emits tool_result, and responds
        # "sum=<value> for: <message>"; otherwise responds "no tools for: <message>". Always ends with a
        # turn_complete event whose done flag is true if the message contains "finish".
        if bridge_url and "add" in tool_names:
            from fastmcp import Client

            arguments = {"a": 2, "b": 3}
            emit({"event": "tool_call", "tool_name": "add", "arguments": arguments})
            async with Client(bridge_url) as client:
                result = await client.call_tool("add", arguments)
            value = result.content[0].text
            emit({"event": "tool_result", "tool_name": "add", "result": value})
            response = f"sum={value} for: {message}"
        else:
            response = f"no tools for: {message}"
        emit(
            {
                "event": "turn_complete",
                "response": response,
                "done": "finish" in message,
            }
        )

    for line in sys.stdin:
        request = json.loads(line)
        asyncio.run(run_turn(request["message"]))


def mode_echo() -> None:
    # [FUNCTION DOC]
    # HELPER: Prints "ready" then echoes each stdin line as "echo:<stripped line>".
    print("ready", flush=True)
    for line in sys.stdin:
        print("echo:" + line.strip(), flush=True)


def mode_unicode_echo() -> None:
    # [FUNCTION DOC]
    # HELPER: Like echo mode but wraps each reply with non-ASCII characters (check mark, CJK, emoji).
    """Echo with non-ASCII output, which locale-encoded pipes would fail on."""
    print("ready", flush=True)
    for line in sys.stdin:
        print("echo:\u2713 " + line.strip() + " \u4e16\u754c \U0001f600", flush=True)


def mode_slow_start() -> None:
    # [FUNCTION DOC]
    # HELPER: Sleeps 60 seconds before printing "ready" (exceeds any startup timeout).
    time.sleep(60)
    print("ready", flush=True)


def mode_exit_now() -> None:
    # [FUNCTION DOC]
    # HELPER: Prints "dying" to stderr and exits with code 3.
    print("dying", file=sys.stderr, flush=True)
    sys.exit(3)


def mode_exit_with_secret() -> None:
    # [FUNCTION DOC]
    # HELPER: Prints a fake secret string to stderr and exits with code 3 (tests stderr redaction).
    print("FAKE_HARNESS_SECRET_FOR_TEST", file=sys.stderr, flush=True)
    sys.exit(3)


def mode_spawn_descendant() -> None:
    # [FUNCTION DOC]
    # HELPER: Prints "ready", spawns this script in "descendant" mode with inherited pipes, waits for
    # one stdin line, then exits with code 1, leaving the descendant running.
    print("ready", flush=True)
    subprocess.Popen([sys.executable, "-u", __file__, "descendant"])
    sys.stdin.readline()
    sys.exit(1)


def mode_descendant() -> None:
    # [FUNCTION DOC]
    # HELPER: Ignores SIGTERM, prints its own PID, then sleeps 60 seconds.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    time.sleep(60)


def mode_crash_after_echo() -> None:
    # [FUNCTION DOC]
    # HELPER: Prints "ready", echoes exactly one stdin line, then exits with code 1.
    print("ready", flush=True)
    line = sys.stdin.readline()
    print("echo:" + line.strip(), flush=True)
    sys.exit(1)


def mode_exit_with_output() -> None:
    # [FUNCTION DOC]
    # HELPER: Prints "first", "second", then "last" without a trailing newline, and exits normally.
    print("first")
    print("second")
    print("last", end="", flush=True)


def mode_close_stdout() -> None:
    # [FUNCTION DOC]
    # HELPER: Prints "ready", waits for a stdin line, closes the stdout file descriptor, then blocks
    # reading another stdin line.
    print("ready", flush=True)
    sys.stdin.readline()
    os.close(sys.stdout.fileno())
    sys.stdin.readline()


def mode_ignore_sigterm() -> None:
    # [FUNCTION DOC]
    # HELPER: Ignores SIGTERM, prints "ready", and sleeps 60 seconds.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("ready", flush=True)
    time.sleep(60)


MODES = {
    "mcp": mode_mcp,
    "echo": mode_echo,
    "unicode-echo": mode_unicode_echo,
    "slow-start": mode_slow_start,
    "exit-now": mode_exit_now,
    "exit-with-secret": mode_exit_with_secret,
    "spawn-descendant": mode_spawn_descendant,
    "descendant": mode_descendant,
    "exit-with-output": mode_exit_with_output,
    "close-stdout": mode_close_stdout,
    "crash-after-echo": mode_crash_after_echo,
    "ignore-sigterm": mode_ignore_sigterm,
}


if __name__ == "__main__":
    MODES[sys.argv[1]]()
