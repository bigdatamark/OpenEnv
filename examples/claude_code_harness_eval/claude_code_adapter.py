# [MODULE DOC]
# RFC 005 harness adapter that drives the Claude Code CLI in headless stream-json mode as one long-lived subprocess.
# Each user message is written as a JSON line; stdout JSON events are mapped to harness events (text, tool call, tool
# result, turn complete). Environment tools are exposed via an MCP HTTP server config pointing at the harness bridge,
# Claude Code's built-in tools and settings files are disabled.
# SPDX-License-Identifier: BSD-3-Clause

"""RFC 005 adapter for Claude Code, driven through its headless stream-json mode.

One long-lived `claude -p --input-format stream-json --output-format stream-json`
process holds the conversation. Each user message is one JSON line on stdin, and
Claude Code answers with JSON-line events on stdout, ending every turn with a
`result` event. The environment's MCP tools reach it through `--mcp-config`,
pointing at the `HarnessMCPBridge` URL.

Claude Code's own tools are disabled (`--tools ""`), so the agent can only act
through the environment's tools, and it loads no settings files
(`--setting-sources ""`), so the user's hooks and CLAUDE.md stay out of the run.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    AgenticHarnessAdapter,
    HarnessConfig,
    HarnessError,
    HarnessEvent,
    HarnessEventType,
    HarnessProcess,
)

MCP_SERVER_NAME = "env"
# Claude Code exposes MCP tools as `mcp__<server>__<tool>`.
TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"


class ClaudeCodeAdapter(AgenticHarnessAdapter):
    # [CLASS DOC]
    # AgenticHarnessAdapter implementation for Claude Code. Holds the harness config, an optional appended system prompt,
    # the subprocess handle and the generated MCP config JSON.
    """
    Drive Claude Code as a turn-based harness over stdio.

    `config.command` is the executable (e.g. `["claude"]`), `config.model` is
    passed as `--model`, and `config.env_vars` reach the process (for example
    `ANTHROPIC_API_KEY`, or `ANTHROPIC_BASE_URL` for an Anthropic-compatible
    endpoint).

    Args:
        config (`HarnessConfig`):
            How to launch Claude Code.
        system_prompt (`str`, *optional*):
            Appended to Claude Code's system prompt, e.g. the policy the agent
            must follow in this environment.
    """

    def __init__(self, config: HarnessConfig, system_prompt: Optional[str] = None):
        # [FUNCTION DOC]
        # Initialises the base adapter with the config and stores the optional system prompt; no process or MCP config yet.
        super().__init__(config)
        self.system_prompt = system_prompt
        self._process: Optional[HarnessProcess] = None
        self._mcp_config: Optional[str] = None

    async def inject_tools(
        self, tools: list[Tool], bridge_url: Optional[str] = None
    ) -> None:
        # [FUNCTION DOC]
        # Stores an MCP config JSON naming a single HTTP server ("env") at the bridge URL when tools and a URL are given;
        # otherwise clears it. Must be called before start to take effect.
        if tools and bridge_url:
            self._mcp_config = json.dumps(
                {"mcpServers": {MCP_SERVER_NAME: {"type": "http", "url": bridge_url}}}
            )
        else:
            self._mcp_config = None

    async def start(self, working_directory: str) -> None:
        # [FUNCTION DOC]
        # Builds the Claude Code command line (print mode, stream-json in/out, verbose, no built-in tools, strict MCP config, no
        # setting sources, no session persistence), adding the MCP config and an allow rule for all env-server tools when set,
        # the model when configured, and the appended system prompt when given. Starts it as a HarnessProcess in the working
        # directory with configured env vars and startup timeout. Side effects: spawns a subprocess.
        command = [
            *self.config.command,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
        ]
        if self._mcp_config is not None:
            # Allow every tool of the environment's server without a permission prompt.
            command += [
                "--mcp-config",
                self._mcp_config,
                "--allowedTools",
                f"mcp__{MCP_SERVER_NAME}",
            ]
        if self.config.model:
            command += ["--model", self.config.model]
        if self.system_prompt:
            command += ["--append-system-prompt", self.system_prompt]
        self._process = HarnessProcess(
            command,
            cwd=working_directory,
            env_vars=self.config.env_vars,
            startup_timeout_s=self.config.startup_timeout_s,
        )
        # Claude Code prints nothing until it receives the first message.
        await self._process.start()

    async def stop(self) -> None:
        # [FUNCTION DOC]
        # Stops the subprocess if one was started.
        if self._process is not None:
            await self._process.stop()

    async def is_alive(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if a process exists and is still running.
        return self._process is not None and self._process.is_running()

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        # [FUNCTION DOC]
        # Async generator for one turn. Raises HarnessError if not started. Writes the user message as a stream-json line, then
        # reads lines: end of output raises HarnessError with stderr; a "result" event raises HarnessError if flagged as an error,
        # otherwise yields a TURN_COMPLETE event (response text, done False, num_turns, usage) and ends; assistant/user events
        # are mapped to text/tool events, tracking tool-use ids to names. Other event types are ignored.
        # Raises json.JSONDecodeError on non-JSON lines.
        if self._process is None:
            raise HarnessError("Claude Code is not running; call start() first")
        await self._process.write_line(
            json.dumps(
                {"type": "user", "message": {"role": "user", "content": message}}
            )
        )
        tool_names: dict[str, str] = {}
        while True:
            # HarnessEnvironment bounds the whole turn with `session_timeout_s`.
            line = await self._process.read_line()
            if line is None:
                raise HarnessError(
                    f"Claude Code exited mid-turn: {self._process.drain_stderr()}"
                )
            native = json.loads(line)
            kind = native.get("type")
            if kind == "result":
                # An API error, not an answer: end the turn as a failure of the harness.
                if native.get("is_error"):
                    raise HarnessError(
                        f"Claude Code failed: {native.get('result') or native.get('subtype')}"
                    )
                yield HarnessEvent(
                    type=HarnessEventType.TURN_COMPLETE,
                    data={
                        "response": native.get("result") or "",
                        "done": False,
                        "num_turns": native.get("num_turns"),
                        "usage": native.get("usage"),
                    },
                )
                return
            if kind in ("assistant", "user"):
                for event in _map_content(native["message"].get("content"), tool_names):
                    yield event


def _map_content(content: Any, tool_names: dict[str, str]) -> list[HarnessEvent]:
    # [FUNCTION DOC]
    # HELPER: Maps a message's content blocks to harness events: text blocks to TEXT_OUTPUT, tool_use blocks to TOOL_CALL
    # (tool name with the MCP prefix stripped, recording id -> name in the given mapping, which is mutated), tool_result blocks
    # to TOOL_RESULT (tool name looked up by id, content, error flag). Non-list content yields an empty list.
    """Map the content blocks of one assistant or user message to events."""
    if not isinstance(content, list):
        return []
    events = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            events.append(
                HarnessEvent(
                    type=HarnessEventType.TEXT_OUTPUT, data={"text": block["text"]}
                )
            )
        elif kind == "tool_use":
            name = block["name"].removeprefix(TOOL_PREFIX)
            tool_names[block["id"]] = name
            events.append(
                HarnessEvent(
                    type=HarnessEventType.TOOL_CALL,
                    data={"tool_name": name, "arguments": block.get("input", {})},
                )
            )
        elif kind == "tool_result":
            events.append(
                HarnessEvent(
                    type=HarnessEventType.TOOL_RESULT,
                    data={
                        "tool_name": tool_names.get(block["tool_use_id"], ""),
                        "result": block.get("content"),
                        "error": block.get("is_error", False),
                    },
                )
            )
    return events
