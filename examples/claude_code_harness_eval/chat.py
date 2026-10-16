# [MODULE DOC]
# Small interactive CLI client for a harness served over a WebSocket endpoint: sends user messages (from argv or
# stdin, one per line) and prints tool calls and the agent's final response for each turn.
# Side effects: WebSocket network I/O, reads stdin, prints to stdout.
# SPDX-License-Identifier: BSD-3-Clause

"""Chat with a harness served on `WS /harness`, one message per line.

    python chat.py ws://localhost:8000/harness
    python chat.py ws://localhost:8000/harness "Hi, my user id is noah_muller_9847." "What reservations do I have?"

With messages on the command line it sends them in order and exits; without,
it reads them from stdin.
"""

from __future__ import annotations

import json
import sys

from websockets.sync.client import connect


def main() -> None:
    # [FUNCTION DOC]
    # Takes the WebSocket URL and optional messages from argv (IndexError if the URL is missing). Connects, waits up to 60 s
    # for the first frame and prints it; returns unless it is "session_started". For each non-empty message, sends a
    # "message" frame and reads frames (300 s timeout each), printing tool_call frames, returning on an "error" frame, and
    # printing the response and moving on at "turn_complete". Other frame types are ignored.
    url, messages = sys.argv[1], sys.argv[2:]
    with connect(url) as websocket:
        started = json.loads(websocket.recv(timeout=60))
        print(f"[{started['type']}] {started.get('data', {})}")
        if started["type"] != "session_started":  # e.g. the server is busy
            return
        for message in messages or (line.strip() for line in sys.stdin):
            if not message:
                continue
            websocket.send(json.dumps({"type": "message", "content": message}))
            while True:
                frame = json.loads(websocket.recv(timeout=300))
                data = frame.get("data", {})
                if frame["type"] == "tool_call":
                    print(f"  -> {data['tool_name']}({data['arguments']})")
                elif frame["type"] == "error":
                    print(f"[error] {data or frame}")
                    return
                elif frame["type"] == "turn_complete":
                    print(f"agent: {data['response']}")
                    break


if __name__ == "__main__":
    main()
