# [MODULE DOC]
# Adapters that let generic OpenEnv harness code (tool-calling LLM rollouts) drive a BrowserGym client. Defines a
# fixed tool list (click, fill, send_keys, scroll, noop), converts tool calls to BrowserGym action strings and back,
# formats prompts/tool results, and builds verification summaries.
"""Harness-oriented BrowserGym session adapters."""

from __future__ import annotations

import ast
import json
from typing import Any

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ResourceSessionFactory,
    StepEnvSessionAdapter,
    ToolResult,
    VerifyResult,
)
from openenv.core.llm_client import ToolCall

from .models import BrowserGymAction

_BROWSERGYM_TOOLS = [
    Tool(
        name="click",
        description="Click an element by BrowserGym bid.",
        input_schema={
            "type": "object",
            "properties": {
                "bid": {"type": "string"},
            },
            "required": ["bid"],
        },
    ),
    Tool(
        name="fill",
        description="Fill an input field by bid.",
        input_schema={
            "type": "object",
            "properties": {
                "bid": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["bid", "text"],
        },
    ),
    Tool(
        name="send_keys",
        description="Send keyboard input to the page.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="scroll",
        description="Scroll the page up or down.",
        input_schema={
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
            },
            "required": ["direction"],
        },
    ),
    Tool(
        name="noop",
        description="Take no action on the current page.",
        input_schema={"type": "object", "properties": {}},
    ),
]


def _quote(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the value as a JSON string literal (double-quoted, non-ASCII preserved).
    return json.dumps(value, ensure_ascii=False)


def build_browsergym_action_str(tool_name: str, arguments: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # Converts a tool name and arguments into a BrowserGym action string, e.g. click("bid"), fill("bid", "text"),
    # send_keys("text"), scroll("direction"), noop(). Argument values are str()-converted and JSON-quoted.
    # Raises: KeyError for an unsupported tool name or a missing required argument.
    # KNOWN ISSUE: BrowserGym's built-in scroll action takes numeric x/y deltas; a single direction string is probably
    # rejected by the BrowserGym action parser.
    """Convert a BrowserGym tool call into the action string the env expects."""

    if tool_name == "click":
        return f"click({_quote(str(arguments['bid']))})"
    if tool_name == "fill":
        return (
            f"fill({_quote(str(arguments['bid']))}, {_quote(str(arguments['text']))})"
        )
    if tool_name == "send_keys":
        return f"send_keys({_quote(str(arguments['text']))})"
    if tool_name == "scroll":
        return f"scroll({_quote(str(arguments['direction']))})"
    if tool_name == "noop":
        return "noop()"

    raise KeyError(f"Unsupported BrowserGym tool: {tool_name}")


def _format_browsergym_prompt(observation: Any, task: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: Builds the initial user prompt from an observation: goal (or task fallback), previous action error, and page
    # structure (axtree text, else text), followed by "Choose the next browser action.", joined by blank lines.
    goal = getattr(observation, "goal", "") or (task or "")
    page_text = getattr(observation, "axtree_txt", "") or getattr(
        observation, "text", ""
    )
    error = getattr(observation, "error", "")

    parts = []
    if goal:
        parts.append(f"Goal: {goal}")
    if error:
        parts.append(f"Previous action error: {error}")
    if page_text:
        parts.append(f"Page structure:\n{page_text}")
    parts.append("Choose the next browser action.")
    return "\n\n".join(parts)


def _build_browsergym_tool_result(
    task: Any,
):
    # [FUNCTION DOC]
    # HELPER: Returns a closure that turns a step result into a ToolResult for the given task.
    def builder(
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        state: Any,
    ) -> ToolResult:
        # [FUNCTION DOC]
        # HELPER: Produces a ToolResult whose data has tool name, arguments copy, goal (or task), observation text, url, error,
        # last_action_error, reward and done; done flag from the result; metadata with reward and the dumped state.
        observation = result.observation
        data = {
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "goal": getattr(observation, "goal", "") or (task or ""),
            "observation_text": getattr(observation, "axtree_txt", "")
            or getattr(observation, "text", ""),
            "url": getattr(observation, "url", ""),
            "error": getattr(observation, "error", ""),
            "last_action_error": getattr(observation, "last_action_error", False),
            "reward": result.reward,
            "done": result.done,
        }
        metadata = {
            "reward": result.reward,
            "state": state.model_dump() if hasattr(state, "model_dump") else state,
        }
        return ToolResult(data=data, done=bool(result.done), metadata=metadata)

    return builder


def _build_browsergym_verify(
    transcript: list[dict[str, Any]],
    final_state: Any | None,
    last_result: Any | None,
    state: Any,
) -> VerifyResult:
    # [FUNCTION DOC]
    # HELPER: Builds a VerifyResult from the last step result: env_reward (None if no step), done flag, metrics
    # (step_count, cum_reward, benchmark, task_name read from state) and artifacts (dumped final state, final rollout
    # state, transcript length).
    reward = None if last_result is None else last_result.reward
    done = False if last_result is None else bool(last_result.done)
    metrics = {
        "step_count": getattr(state, "step_count", 0),
        "cum_reward": getattr(state, "cum_reward", reward or 0.0),
        "benchmark": getattr(state, "benchmark", ""),
        "task_name": getattr(state, "task_name", ""),
    }
    artifacts = {
        "final_state": state.model_dump() if hasattr(state, "model_dump") else state,
        "final_rollout": final_state,
        "transcript_length": len(transcript),
    }
    return VerifyResult(
        env_reward=reward,
        done=done,
        metrics=metrics,
        artifacts=artifacts,
    )


class BrowserGymSessionFactory(ResourceSessionFactory):
    # [CLASS DOC]
    # Harness session factory: each create() call builds a fresh client from the supplied factory and wraps it in a
    # StepEnvSessionAdapter configured with the BrowserGym tools and builders above.
    """Create BrowserGym-backed resource sessions from client factories."""

    def __init__(self, client_factory, *, default_task: str | None = None):
        # [FUNCTION DOC]
        # Stores the client factory callable and an optional default task name.
        self._client_factory = client_factory
        self._default_task = default_task

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> StepEnvSessionAdapter:
        # [FUNCTION DOC]
        # Creates a session for a task (falls back to the default task). If a task is set it is passed as reset kwarg
        # "task_name". Returns a StepEnvSessionAdapter with seed, episode_id, tool specs, an action builder producing
        # BrowserGymAction from tool calls, an initial-prompt builder, tool-result builder and verify builder.
        # Side effects: calls the client factory (may open a connection).
        session_task = task if task is not None else self._default_task
        client = self._client_factory()

        reset_kwargs = {}
        if session_task is not None:
            reset_kwargs["task_name"] = session_task

        return StepEnvSessionAdapter(
            client=client,
            task=session_task,
            seed=seed,
            episode_id=episode_id,
            tool_specs=list(_BROWSERGYM_TOOLS),
            action_builder=lambda name, arguments: BrowserGymAction(
                action_str=build_browsergym_action_str(name, arguments)
            ),
            initial_messages_builder=lambda result, current_task: [
                {
                    "role": "user",
                    "content": _format_browsergym_prompt(
                        result.observation,
                        current_task,
                    ),
                }
            ],
            tool_result_builder=_build_browsergym_tool_result(session_task),
            verify_builder=_build_browsergym_verify,
            reset_kwargs=reset_kwargs,
        )


def _parse_action_call(action_text: str) -> tuple[str, list[Any]]:
    # [FUNCTION DOC]
    # HELPER: Parses action text as a single Python call expression with a bare function name and positional literal
    # arguments. Returns (function name, list of literal values). Raises: ValueError for syntax errors, non-call
    # expressions, attribute calls, keyword arguments or non-literal arguments. Does not evaluate code.
    try:
        expression = ast.parse(action_text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Unsupported BrowserGym action: {action_text}") from exc

    if not isinstance(expression, ast.Call) or not isinstance(
        expression.func, ast.Name
    ):
        raise ValueError(f"Unsupported BrowserGym action: {action_text}")
    if expression.keywords:
        raise ValueError("BrowserGym action arguments must be positional")

    args: list[Any] = []
    for arg in expression.args:
        try:
            args.append(ast.literal_eval(arg))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("BrowserGym action arguments must be literals") from exc

    return expression.func.id, args


def _expect_str(value: Any, argument_name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the value if it is a str, else raises ValueError naming the argument.
    if not isinstance(value, str):
        raise ValueError(f"BrowserGym {argument_name} argument must be a string")
    return value


def build_browsergym_action_tool_call(action_text: str) -> ToolCall:
    # [FUNCTION DOC]
    # Parses a textual BrowserGym action (click(bid), fill(bid, text), send_keys(text), scroll("up"/"down"), noop())
    # into a ToolCall with a fixed id per tool and named string args. Raises: ValueError for unsupported actions, wrong
    # argument counts, non-string arguments or an invalid scroll direction.
    """Parse a text BrowserGym action into a structured tool call."""

    name, args = _parse_action_call(action_text)
    if name == "click" and len(args) == 1:
        return ToolCall(
            id="browsergym-click",
            name="click",
            args={"bid": _expect_str(args[0], "bid")},
        )
    if name == "fill" and len(args) == 2:
        return ToolCall(
            id="browsergym-fill",
            name="fill",
            args={
                "bid": _expect_str(args[0], "bid"),
                "text": _expect_str(args[1], "text"),
            },
        )
    if name == "send_keys" and len(args) == 1:
        return ToolCall(
            id="browsergym-send_keys",
            name="send_keys",
            args={"text": _expect_str(args[0], "text")},
        )
    if name == "scroll" and len(args) == 1:
        direction = _expect_str(args[0], "direction")
        if direction not in {"up", "down"}:
            raise ValueError("BrowserGym scroll direction must be 'up' or 'down'")
        return ToolCall(
            id="browsergym-scroll",
            name="scroll",
            args={"direction": direction},
        )
    if name == "noop" and not args:
        return ToolCall(id="browsergym-noop", name="noop", args={})

    raise ValueError(f"Unsupported BrowserGym action: {action_text}")


__all__ = [
    "BrowserGymSessionFactory",
    "build_browsergym_action_str",
    "build_browsergym_action_tool_call",
]
