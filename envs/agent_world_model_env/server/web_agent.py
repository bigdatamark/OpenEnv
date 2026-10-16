# [MODULE DOC]
# Simple LLM agent loop used by the AWM web console. It drives the already-reset environment through the web
# interface manager's step_environment coroutine, prompting an OpenAI-compatible model that answers with
# <tool_call> JSON blocks for two meta-functions (list_tools, call_tool), and streams progress as AgentEvent objects.
"""LLM-driven agent loop for the AWM web UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from .prompts import DEFAULT_SYSTEM_PROMPT


def parse_tool_call(content: str) -> dict | None:
    # [FUNCTION DOC]
    # Extracts the first <tool_call>...</tool_call> block from model output and parses its JSON. A list uses its first
    # element. Returns the dict if it has a "name" key, otherwise None (also None for missing tags or invalid JSON).
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return data


def format_tools(tools: list[dict]) -> str:
    # [FUNCTION DOC]
    # Renders a list of tool dicts (name or tool_name, description, input_schema or inputSchema) as numbered
    # human-readable text with parameter names, types, "(required)" markers and descriptions. Returns the string.
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 60]
    for i, t in enumerate(tools, 1):
        name = t.get("name") or t.get("tool_name", "")
        desc = t.get("description", "")
        schema = t.get("input_schema") or t.get("inputSchema") or {}
        lines.append(f"{i}. {name}")
        lines.append(f"   Description: {desc}")
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                lines.append(
                    f"     - {pname}: {pinfo.get('type', 'any')}{req} — "
                    f"{pinfo.get('description', '')}"
                )
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines)


def _truncate(s: str, limit: int = 2000) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the string unchanged if within limit (default 2000), else its prefix plus a note with full length.
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated, full length {len(s)} chars)"


@dataclass
class AgentEvent:
    # [CLASS DOC]
    # Dataclass for one streamed event: kind ("info", "llm_response", "tool_call", "tool_result", "verify", "done",
    # "error"), display text and a payload dict.
    kind: str  # "info" | "llm_response" | "tool_call" | "tool_result" | "verify" | "done" | "error"
    text: str = ""
    payload: dict = field(default_factory=dict)


class AwmAgent:
    # [CLASS DOC]
    # Agent bound to a web interface manager and an AsyncOpenAI client. Configurable system prompt, iteration cap
    # (default 10), temperature (1.0) and max completion tokens (2048). Supports cooperative stop requests.
    def __init__(
        self,
        web_manager: Any,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 10,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ):
        # [FUNCTION DOC]
        # Stores configuration and creates the AsyncOpenAI client for the given base URL/key. No network I/O.
        self._web = web_manager
        self._client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._model = llm_model
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop_requested = False

    def request_stop(self) -> None:
        # [FUNCTION DOC]
        # Sets a flag that makes the run loop stop before its next LLM call.
        self._stop_requested = True

    async def _list_tools_dict(self) -> list[dict]:
        # [FUNCTION DOC]
        # HELPER: Steps the environment with a list_tools action and returns tools as dicts (non-dict tool objects are
        # converted to name/description/input_schema dicts).
        result = await self._web.step_environment({"type": "list_tools"})
        obs = result.get("observation", {}) or {}
        tools = obs.get("tools", []) or []
        out = []
        for t in tools:
            if isinstance(t, dict):
                out.append(t)
            else:
                out.append(
                    {
                        "name": getattr(t, "name", ""),
                        "description": getattr(t, "description", ""),
                        "input_schema": getattr(t, "input_schema", {}),
                    }
                )
        return out

    async def _call_tool(self, tool_name: str, args: dict) -> dict:
        # [FUNCTION DOC]
        # HELPER: Steps the environment with a call_tool action for the given name/arguments; returns the raw step result dict.
        return await self._web.step_environment(
            {"type": "call_tool", "tool_name": tool_name, "arguments": args}
        )

    async def run(
        self,
        task: str,
        verifier_mode: str | None = None,
        final_answer_fallback: str = "",
        auto_verify: bool = True,
        auto_done: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        # [FUNCTION DOC]
        # Async generator. Lists tools (error event and stop on failure), then builds messages (system prompt, task, tools
        # text) and loops up to max_iterations: calls the LLM (error event and stop on failure), yields the response, parses
        # a tool call (none means final answer and ends the loop), dispatches list_tools or call_tool (inner "arguments" may be
        # a JSON string; non-dict becomes {}), and feeds the result (or error text, truncated to 6000 chars) back as a user
        # message, yielding tool_call/tool_result events. Yields an info event when the stop flag is set or the cap is hit.
        # Afterwards, if auto_verify and verifier_mode are set, calls "verify" with the last assistant message (or the
        # fallback) as final_answer and yields a verify event; if auto_done, calls "done" with keep_session True and yields
        # a done event. Errors in these phases yield error events. Never raises.
        # Side effects: LLM network calls and environment steps (each counts as a step).
        # KNOWN ISSUE: the list_tools branch steps the environment twice (one discarded result plus the helper call).
        # KNOWN ISSUE: returning early on stop/LLM failure skips the auto verify/done phase.
        """Run the agent on the already-reset env, yielding events as it goes."""
        try:
            tools = await self._list_tools_dict()
        except Exception as e:
            yield AgentEvent(kind="error", text=f"list_tools failed: {e}")
            return

        yield AgentEvent(
            kind="info",
            text=f"Discovered {len(tools)} tools.",
            payload={"tool_names": [t.get("name") for t in tools]},
        )
        tools_text = format_tools(tools)

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
            {
                "role": "user",
                "content": f"Available tools:\n{tools_text}",
            },
        ]

        last_assistant_content = ""

        for step in range(1, self._max_iterations + 1):
            if self._stop_requested:
                yield AgentEvent(kind="info", text="Stopped by user.")
                return

            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_completion_tokens=self._max_tokens,
                )
            except Exception as e:
                yield AgentEvent(
                    kind="error", text=f"LLM call failed at step {step}: {e}"
                )
                return

            content = resp.choices[0].message.content or ""
            last_assistant_content = content
            messages.append({"role": "assistant", "content": content})

            yield AgentEvent(
                kind="llm_response",
                text=content,
                payload={"step": step},
            )

            tc = parse_tool_call(content)
            if tc is None:
                yield AgentEvent(
                    kind="info",
                    text=f"No <tool_call> in step {step}; treating as final answer.",
                )
                break

            name = tc.get("name", "")
            arguments = tc.get("arguments") or {}

            yield AgentEvent(
                kind="tool_call",
                text=f"{name} {json.dumps(arguments, ensure_ascii=False)[:300]}",
                payload={"name": name, "arguments": arguments, "step": step},
            )

            tool_response = ""
            try:
                if name == "list_tools":
                    result = await self._web.step_environment({"type": "list_tools"})
                    obs = result.get("observation", {}) or {}
                    tools = await self._list_tools_dict()
                    tool_response = format_tools(tools)
                elif name == "call_tool":
                    inner_name = arguments.get("tool_name", "")
                    inner_args = arguments.get("arguments", "{}")
                    if isinstance(inner_args, str):
                        try:
                            inner_args = json.loads(inner_args)
                        except json.JSONDecodeError:
                            inner_args = {}
                    if not isinstance(inner_args, dict):
                        inner_args = {}
                    result = await self._call_tool(inner_name, inner_args)
                    obs = result.get("observation", {}) or {}
                    if obs.get("tool_result") is not None:
                        tr = obs["tool_result"]
                        tool_response = (
                            json.dumps(tr, ensure_ascii=False)
                            if not isinstance(tr, str)
                            else tr
                        )
                    elif obs.get("error"):
                        tool_response = f"Error: {obs['error']}"
                    else:
                        tool_response = json.dumps(obs, ensure_ascii=False)
                else:
                    tool_response = (
                        f"Error: Unknown function '{name}'. "
                        "Use 'list_tools' or 'call_tool'."
                    )
            except Exception as e:
                tool_response = f"Error during tool dispatch: {e}"

            yield AgentEvent(
                kind="tool_result",
                text=_truncate(tool_response),
                payload={"step": step},
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool response:\n{_truncate(tool_response, 6000)}",
                }
            )
        else:
            yield AgentEvent(
                kind="info",
                text=f"Max iterations ({self._max_iterations}) reached.",
            )

        if auto_verify and verifier_mode:
            verify_args: dict = {"verifier_mode": verifier_mode}
            if last_assistant_content:
                verify_args["final_answer"] = last_assistant_content
            elif final_answer_fallback:
                verify_args["final_answer"] = final_answer_fallback
            try:
                verify_result = await self._call_tool("verify", verify_args)
                obs = verify_result.get("observation", {}) or {}
                yield AgentEvent(
                    kind="verify",
                    text=(
                        f"reward_type={obs.get('reward_type')}  "
                        f"reward={verify_result.get('reward')}"
                    ),
                    payload={
                        "reward_type": obs.get("reward_type"),
                        "reward": verify_result.get("reward"),
                        "verify_result": obs.get("verify_result"),
                    },
                )
            except Exception as e:
                yield AgentEvent(kind="error", text=f"verify failed: {e}")

        if auto_done:
            try:
                done_result = await self._call_tool("done", {"keep_session": True})
                obs = done_result.get("observation", {}) or {}
                yield AgentEvent(
                    kind="done",
                    text=(
                        f"Episode done. trajectory_path="
                        f"{obs.get('trajectory_path') or '(none)'}"
                    ),
                    payload={
                        "trajectory_path": obs.get("trajectory_path"),
                        "session_dir": obs.get("session_dir"),
                    },
                )
            except Exception as e:
                yield AgentEvent(kind="error", text=f"done failed: {e}")
