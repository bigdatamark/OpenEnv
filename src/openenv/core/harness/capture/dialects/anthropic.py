# [MODULE DOC]
# Anthropic Messages dialect: converts Anthropic requests into OpenAI
# chat-completions requests for the inference backend, converts chat-completions
# responses back to Anthropic message objects, and converts a streamed
# chat-completions chunk sequence into Anthropic SSE events (message_start,
# content_block_start/delta/stop, message_delta, message_stop) using a stateful
# per-request stream state. Strips Claude Code's per-request billing-header line
# from the system prompt so prompts stay prefix-stable across turns.
"""Anthropic Messages API transformer.

Transforms between Anthropic Messages API and OpenAI Chat Completions API.
Aligned with agent-harness-proxy/src/harness_proxy/transform/anthropic.py.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from ..upstream import UpstreamRequestError
from .base import BaseTransformer
from .images import (
    anthropic_content_to_openai_chat,
    openai_chat_content_to_anthropic_blocks,
)
from .reasoning import extract_reasoning_from_anthropic_content, make_signature

# Claude Code SDK leaks `x-anthropic-billing-header: ...cch=<hash>;` as the
# first line of the system prompt. The cch= hash changes per request, so
# rendered prompt tokens drift every turn and prefix_merging can't chain
# multi-turn traces. Strip the line before forwarding to SGLang.
_CLAUDE_CODE_BILLING_HEADER_RE = re.compile(
    r"^\s*x-anthropic-billing-header:[^\n]*\n?", re.IGNORECASE
)


@dataclass
class _AnthropicToolCallState:
    # [CLASS DOC]
    # HELPER: Mutable per-tool-call streaming state: tool call id, accumulated name,
    # assigned Anthropic content block index (None until started), argument text buffered
    # before the block starts, and whether the tool_use block has been opened.
    id: str
    name: str = ""
    anthropic_index: int | None = None
    buffered_arguments: str = ""
    started: bool = False


class AnthropicStreamState:
    # [CLASS DOC]
    # Tracks open Anthropic content blocks (thinking, text, tool_use) across the chunks
    # of one upstream chat-completions stream, assigns sequential block indices, records
    # stop reason and output token count, and emits Anthropic SSE event dicts.
    """Per-request Anthropic streaming state.

    Anthropic SSE blocks are stateful across chunks: content blocks must be
    explicitly started, optionally receive multiple deltas, and then be closed
    before the final message delta. This helper tracks those open blocks for a
    single upstream OpenAI/SGLang stream.
    """

    def __init__(self, model: str, finish_to_stop_reason: dict[str, str]):
        # [FUNCTION DOC]
        # Initializes state for a stream: model name, finish_reason->stop_reason map, a new
        # random "msg_<hex>" id, block counters/flags, empty thinking buffer and tool-call map,
        # stop_reason "end_turn", output_tokens 0, completed False.
        self.model = model
        self.finish_to_stop_reason = finish_to_stop_reason
        self.message_id = f"msg_{uuid.uuid4().hex}"
        self.next_block_index = 0
        self.text_block_index: int | None = None
        self.text_block_started = False
        self.thinking_block_index: int | None = None
        self.thinking_block_started = False
        self.thinking_buffer = ""
        self.tool_calls: dict[int, _AnthropicToolCallState] = {}
        self.stop_reason = "end_turn"
        self.output_tokens = 0
        self.any_block_started = False
        self.completed = False

    def process_chunk(
        self, chunk: dict[str, Any], is_first: bool = False
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Converts one OpenAI stream chunk into a list of Anthropic event dicts.
        # Inputs: chunk - parsed chat-completions chunk; is_first - emit message_start first.
        # Behavior: updates output_tokens from usage.completion_tokens; if no choices returns
        # early; maps finish_reason to stop_reason (default "end_turn"); reasoning_content
        # opens a thinking block if needed and emits thinking_delta (also buffered for the
        # signature); content closes any open thinking block, opens a text block if needed and
        # emits text_delta; each tool_calls delta (list or single dict) is passed to
        # _process_tool_call.
        # Returns: list of events. Side effects: mutates state.
        events: list[dict[str, Any]] = []

        if is_first:
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
            )

        usage = chunk.get("usage", {})
        if usage:
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)

        choices = chunk.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {}) or {}
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.stop_reason = self.finish_to_stop_reason.get(finish_reason, "end_turn")

        # Thinking blocks must precede text and tool_use per Anthropic spec.
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not self.thinking_block_started:
                events.append(self._open_thinking_block())
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.thinking_block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                }
            )
            self.thinking_buffer += reasoning

        content = delta.get("content")
        if content:
            thinking_stop = self._close_thinking_block()
            if thinking_stop:
                events.extend(thinking_stop)
            if not self.text_block_started:
                events.append(self._open_text_block())
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.text_block_index,
                    "delta": {"type": "text_delta", "text": content},
                }
            )

        tool_call_deltas = delta.get("tool_calls") or []
        if not isinstance(tool_call_deltas, list):
            tool_call_deltas = [tool_call_deltas]
        for tool_call_delta in tool_call_deltas:
            if isinstance(tool_call_delta, dict):
                events.extend(self._process_tool_call(tool_call_delta))

        return events

    def finalize(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Closes the stream. Returns [] if already completed. Otherwise emits: thinking
        # signature+stop, text block stop, stops for every started tool block (by tool index
        # order), an empty text block start/stop if no block was ever started, a message_delta
        # with stop_reason and output_tokens, and message_stop. Marks completed.
        if self.completed:
            return []

        events: list[dict[str, Any]] = []

        thinking_stop = self._close_thinking_block()
        if thinking_stop:
            events.extend(thinking_stop)

        text_stop = self._close_text_block()
        if text_stop:
            events.append(text_stop)

        for tool_index in sorted(self.tool_calls):
            tool_state = self.tool_calls[tool_index]
            if tool_state.started and tool_state.anthropic_index is not None:
                events.append(
                    {
                        "type": "content_block_stop",
                        "index": tool_state.anthropic_index,
                    }
                )

        if not self.any_block_started:
            empty_index = self.next_block_index
            events.append(
                {
                    "type": "content_block_start",
                    "index": empty_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            events.append({"type": "content_block_stop", "index": empty_index})

        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": self.output_tokens},
            }
        )
        events.append({"type": "message_stop"})

        self.completed = True
        return events

    def _open_text_block(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Opens a text block at the next index, marks it started; returns its
        # content_block_start event.
        self.text_block_started = True
        self.text_block_index = self.next_block_index
        self.next_block_index += 1
        self.any_block_started = True
        return {
            "type": "content_block_start",
            "index": self.text_block_index,
            "content_block": {"type": "text", "text": ""},
        }

    def _close_text_block(self) -> dict[str, Any] | None:
        # [FUNCTION DOC]
        # HELPER: Returns a content_block_stop event for the open text block and clears it,
        # or None if no text block is open.
        if not self.text_block_started or self.text_block_index is None:
            return None

        event = {"type": "content_block_stop", "index": self.text_block_index}
        self.text_block_started = False
        self.text_block_index = None
        return event

    def _open_thinking_block(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Opens a thinking block (empty thinking and signature) at the next index;
        # returns its content_block_start event.
        self.thinking_block_started = True
        self.thinking_block_index = self.next_block_index
        self.next_block_index += 1
        self.any_block_started = True
        return {
            "type": "content_block_start",
            "index": self.thinking_block_index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }

    def _close_thinking_block(self) -> list[dict[str, Any]] | None:
        # [FUNCTION DOC]
        # HELPER: If a thinking block is open, returns two events (a signature_delta whose
        # signature is make_signature(thinking_buffer), then content_block_stop) and clears
        # the open flag/index; otherwise None.
        # KNOWN ISSUE: thinking_buffer is never reset, so a second thinking block's signature
        # would cover the concatenation of all thinking text so far.
        if not self.thinking_block_started or self.thinking_block_index is None:
            return None
        idx = self.thinking_block_index
        events = [
            {
                "type": "content_block_delta",
                "index": idx,
                "delta": {
                    "type": "signature_delta",
                    "signature": make_signature(self.thinking_buffer),
                },
            },
            {"type": "content_block_stop", "index": idx},
        ]
        self.thinking_block_started = False
        self.thinking_block_index = None
        return events

    def _process_tool_call(
        self, tool_call_delta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Processes one streamed tool-call delta. Looks up/creates state by integer
        # index (non-int -> 0), using the delta id or a generated "toolu_<24 hex>" id; updates
        # id if a later delta carries one; appends name fragments; stringifies arguments
        # (non-string values JSON-encoded) and buffers them. Once a name exists and the block
        # is not started, closes open thinking and text blocks, opens a tool_use block at the
        # next index and flushes buffered arguments as one input_json_delta. Later argument
        # fragments are emitted directly as input_json_delta.
        # Returns: list of events. Side effects: mutates state.
        events: list[dict[str, Any]] = []

        tool_index = tool_call_delta.get("index", 0)
        if not isinstance(tool_index, int):
            tool_index = 0

        tool_state = self.tool_calls.get(tool_index)
        if tool_state is None:
            tool_state = _AnthropicToolCallState(
                # `or`, not a get() default: a present-but-null id returns None and defeats the fallback.
                id=tool_call_delta.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            )
            self.tool_calls[tool_index] = tool_state
        elif tool_call_delta.get("id"):
            tool_state.id = tool_call_delta["id"]

        function = tool_call_delta.get("function", {})
        name = function.get("name")
        if isinstance(name, str) and name:
            tool_state.name += name

        args = function.get("arguments")
        args_str = ""
        if isinstance(args, str) and args:
            args_str = args
        elif args not in (None, ""):
            args_str = json.dumps(args)

        if args_str:
            tool_state.buffered_arguments += args_str

        if tool_state.name and not tool_state.started:
            thinking_stop = self._close_thinking_block()
            if thinking_stop:
                events.extend(thinking_stop)

            text_stop = self._close_text_block()
            if text_stop:
                events.append(text_stop)

            tool_state.started = True
            tool_state.anthropic_index = self.next_block_index
            self.next_block_index += 1
            self.any_block_started = True

            events.append(
                {
                    "type": "content_block_start",
                    "index": tool_state.anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_state.id,
                        "name": tool_state.name,
                        "input": {},
                    },
                }
            )

            if tool_state.buffered_arguments:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": tool_state.anthropic_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_state.buffered_arguments,
                        },
                    }
                )
                tool_state.buffered_arguments = ""
        elif tool_state.started and args_str and tool_state.anthropic_index is not None:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": tool_state.anthropic_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args_str,
                    },
                }
            )

        return events


class AnthropicTransformer(BaseTransformer):
    # [CLASS DOC]
    # Transformer for the Anthropic Messages dialect. FINISH_TO_STOP_REASON maps OpenAI
    # finish reasons to Anthropic stop reasons (stop->end_turn, length->max_tokens,
    # tool_calls->tool_use, content_filter->refusal, stop_sequence->stop_sequence).
    """Transform between Anthropic and OpenAI API formats."""

    FINISH_TO_STOP_REASON: dict[str, str] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
        "stop_sequence": "stop_sequence",
    }

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Converts an Anthropic Messages request body into an OpenAI chat request.
        # Behavior: flattens "system" to text, strips the billing-header line, and prepends it
        # as a system message if non-empty; converts each message via _transform_message
        # (lists are spliced); sets max_tokens (default 4096), model, temperature, top_p,
        # top_k, stop (from stop_sequences), stream=True if requested; a thinking config of
        # type enabled/adaptive sets chat_template_kwargs.enable_thinking=True; tools are
        # converted and, only if the converted list is non-empty, tools and the converted
        # tool_choice (default auto) are set. Finally applies _normalize_request with
        # body["_served_model"].
        # Raises: UpstreamRequestError if messages or tools is not a list of objects.
        # Returns: new request dict.
        messages = []

        # Handle system message
        system = body.get("system")
        if system:
            system_content = self._flatten_content(system)
            # Drop Claude Code's per-request billing header line (breaks
            # prefix_merging because cch= changes every turn).
            system_content = _CLAUDE_CODE_BILLING_HEADER_RE.sub("", system_content)
            if system_content:
                messages.append({"role": "system", "content": system_content})

        # Transform messages
        request_messages = body.get("messages", [])
        if not isinstance(request_messages, list):
            raise UpstreamRequestError("Anthropic messages must be an array")
        if any(not isinstance(message, dict) for message in request_messages):
            raise UpstreamRequestError("Anthropic messages must be objects")
        for msg in request_messages:
            transformed = self._transform_message(msg)
            if transformed:
                if isinstance(transformed, list):
                    messages.extend(transformed)
                else:
                    messages.append(transformed)

        result: dict[str, Any] = {
            "messages": messages,
            "max_tokens": body.get("max_tokens", 4096),
        }
        if "model" in body:
            result["model"] = body["model"]

        if "temperature" in body:
            result["temperature"] = body["temperature"]
        if "top_p" in body:
            result["top_p"] = body["top_p"]
        if "top_k" in body:
            result["top_k"] = body["top_k"]
        if "stop_sequences" in body:
            result["stop"] = body["stop_sequences"]
        if body.get("stream", False):
            result["stream"] = True

        # Anthropic `thinking` request param → enable_thinking on chat template.
        thinking_cfg = body.get("thinking")
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") in {
            "enabled",
            "adaptive",
        }:
            chat_template_kwargs = dict(result.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = True
            result["chat_template_kwargs"] = chat_template_kwargs

        # Tools. Claude Code sometimes sends tools=[] on compaction/summary
        # turns; forwarding tool_choice without a non-empty tools list makes
        # SGLang reject with "tool_choice only allowed when tools specified".
        if "tools" in body:
            declared_tools = body["tools"]
            if not isinstance(declared_tools, list):
                raise UpstreamRequestError("Anthropic tools must be an array")
            if any(not isinstance(tool, dict) for tool in declared_tools):
                raise UpstreamRequestError("Anthropic tools must be objects")
            tools = self._transform_tools_to_openai(declared_tools)
            if tools:
                result["tools"] = tools
                result["tool_choice"] = self._transform_tool_choice_to_openai(
                    body.get("tool_choice", {"type": "auto"})
                )

        return self._normalize_request(
            result,
            body.get("_served_model"),
        )

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Converts an OpenAI chat-completions response into an Anthropic message.
        # Builds content blocks: a thinking block (with generated signature) from
        # reasoning_content, text/image blocks from message content, and tool_use blocks from
        # tool_calls (arguments parsed as JSON, {} on failure); an empty text block if none.
        # Returns: dict with id "msg_<upstream id or random>", type, role, content, model
        # (original request model, default "claude-3"), mapped stop_reason, usage converted
        # to Anthropic form; or an Anthropic error dict if there are no choices.
        # KNOWN ISSUE: tool_use id uses a get() default, so a present-but-null upstream id
        # yields a null id (the streaming path guards against this).
        choices = response.get("choices", [])
        if not choices:
            return self._error_response("No choices in response")

        choice = choices[0]
        message = choice.get("message", {})

        content = []
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning,
                    "signature": make_signature(reasoning),
                }
            )

        text = message.get("content")
        if text or (isinstance(text, list) and text):
            content.extend(openai_chat_content_to_anthropic_blocks(text))

        for tool_call in message.get("tool_calls") or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": tool_call.get("function", {}).get("name", ""),
                    "input": self._parse_json_safe(
                        tool_call.get("function", {}).get("arguments", "{}")
                    ),
                }
            )

        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = self.FINISH_TO_STOP_REASON.get(finish_reason, "end_turn")
        usage = response.get("usage", {})
        anthropic_usage = self._usage_to_anthropic(usage)

        if not content:
            content.append({"type": "text", "text": ""})

        return {
            "id": f"msg_{response.get('id', uuid.uuid4().hex)}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_request.get("model", "claude-3"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": anthropic_usage,
        }

    def create_stream_state(
        self, original_request: dict[str, Any]
    ) -> AnthropicStreamState:
        # [FUNCTION DOC]
        # Creates an AnthropicStreamState for the request's model (default "claude-3").
        return AnthropicStreamState(
            model=original_request.get("model", "claude-3"),
            finish_to_stop_reason=self.FINISH_TO_STOP_REASON,
        )

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Single-chunk fallback: creates a fresh stream state, processes the chunk, and
        # finalizes immediately if the chunk carries a finish_reason. Returns the event list.
        # Not suitable for multi-chunk streams since state is not retained.
        """Best-effort single-chunk Anthropic transform.

        The server uses `create_stream_state()` for request-scoped streaming.
        This fallback keeps direct callers working for simple single-chunk cases.
        """
        state = self.create_stream_state(original_request)
        events = state.process_chunk(chunk, is_first=is_first)
        choices = chunk.get("choices", [])
        if choices and choices[0].get("finish_reason"):
            events.extend(state.finalize())
        return events

    def _transform_message(self, msg: dict[str, Any]) -> Optional[dict | list]:
        # [FUNCTION DOC]
        # HELPER: Converts one Anthropic message to OpenAI form.
        # String content -> {role, content}; non-list content -> stringified. For assistant
        # messages with tool_use blocks: joins text blocks with newlines (None if none), adds
        # reasoning_content from thinking blocks, and converts tool_use blocks to function
        # tool_calls with JSON-encoded input. For user messages with tool_result blocks: emits
        # one "tool" message per result (flattened text, prefixed "[Tool Error]" when
        # is_error), followed by a user message with any image parts, then a trailing user
        # message with joined text blocks; returns the list or None. Otherwise converts content
        # blocks (keeping images) and adds reasoning_content for assistants.
        # Returns: a message dict, a list of messages, or None.
        """Transform a single Anthropic message to OpenAI format."""
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            return {"role": role, "content": content}

        if not isinstance(content, list):
            return {"role": role, "content": str(content)}

        # Check for mixed content: tool_result blocks + other content
        tool_results = [
            c for c in content if isinstance(c, dict) and c.get("type") == "tool_result"
        ]
        tool_uses = [
            c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"
        ]
        text_blocks = [
            c for c in content if isinstance(c, dict) and c.get("type") == "text"
        ]

        messages = []

        # Assistant `thinking` blocks → reasoning_content (kept for replay).
        reasoning_text = ""
        if role == "assistant":
            reasoning_text = extract_reasoning_from_anthropic_content(content)

        # Handle assistant messages with tool_use blocks
        if role == "assistant" and tool_uses:
            tool_calls = []
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if reasoning_text:
                msg_dict["reasoning_content"] = reasoning_text
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            return msg_dict

        # Handle user messages with tool_result blocks
        if role == "user" and tool_results:
            # Each tool_result becomes a tool message
            for tr in tool_results:
                tool_content = tr.get("content", "")
                converted_content = anthropic_content_to_openai_chat(tool_content)
                text_content = self._flatten_content(converted_content)
                # Anthropic marks failed tool results with is_error=true.
                # Surface this to the model so it can see the call failed
                # rather than treating the payload as normal output.
                if tr.get("is_error"):
                    text_content = (
                        f"[Tool Error] {text_content}"
                        if text_content
                        else "[Tool Error]"
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": text_content,
                    }
                )
                # OpenAI tool messages stay text-only; images are sent as a
                # follow-up user message, so mixed text/image order is not preserved.
                image_parts = self._image_parts(converted_content)
                if image_parts:
                    messages.append({"role": "user", "content": image_parts})

            # Any extra user text should come after the tool results.
            text_parts = [b.get("text", "") for b in text_blocks if b.get("text")]
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
            return messages if messages else None

        # Regular content blocks — keep images when present.
        result: dict[str, Any] = {
            "role": role,
            "content": anthropic_content_to_openai_chat(content),
        }
        if role == "assistant" and reasoning_text:
            result["reasoning_content"] = reasoning_text
        return result

    def _flatten_content(self, content: Any) -> str:
        # [FUNCTION DOC]
        # HELPER: Flattens Anthropic content (string, or list of strings/text blocks/nested
        # tool_result blocks) to newline-joined text; other values are stringified.
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        parts.append(self._flatten_content(block.get("content", "")))
            return "\n".join(parts)
        return str(content)

    def _image_parts(self, content: Any) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Returns the image_url parts from an OpenAI content list ([] for non-lists).
        if not isinstance(content, list):
            return []
        return [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]

    def _transform_tools_to_openai(self, tools: list[dict]) -> list[dict]:
        # [FUNCTION DOC]
        # HELPER: Converts Anthropic tool definitions to OpenAI function tools. Skips server
        # tools (a type other than custom/function and no input_schema) and tools without a
        # non-empty string name. Uses description (default "") and input_schema (default {}).
        result = []
        for tool in tools:
            # Anthropic server tools (web_search_*, code_execution_*) carry an
            # explicit `type` and have no `input_schema`. SGLang can't dispatch
            # them, so drop rather than forwarding a stub function tool.
            tool_type = tool.get("type")
            if (
                tool_type
                and tool_type not in ("custom", "function")
                and "input_schema" not in tool
            ):
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return result

    def _transform_tool_choice_to_openai(self, tool_choice: Any) -> Any:
        # [FUNCTION DOC]
        # HELPER: Maps Anthropic tool_choice to OpenAI: auto->"auto", any->"required",
        # none->"none", tool->function choice by name; anything else -> "auto".
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type")
            if tc_type == "auto":
                return "auto"
            elif tc_type == "any":
                return "required"
            elif tc_type == "none":
                return "none"
            elif tc_type == "tool":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")},
                }
        return "auto"

    def _parse_json_safe(self, s: str) -> dict:
        # [FUNCTION DOC]
        # HELPER: Parses a JSON string, returning {} on decode or type errors.
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _usage_to_anthropic(self, usage: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Converts OpenAI usage to Anthropic usage: input_tokens is prompt tokens
        # minus cached tokens (floored at 0), output_tokens is completion tokens; adds
        # cache_read_input_tokens and cache_creation_input_tokens when non-zero.
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cache_read = self._cached_prompt_tokens(usage)
        input_tokens = (
            max(prompt_tokens - cache_read, 0) if cache_read else prompt_tokens
        )

        result: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": completion_tokens,
        }
        if cache_read:
            result["cache_read_input_tokens"] = cache_read
        cache_creation = usage.get("cache_creation_input_tokens")
        if isinstance(cache_creation, int) and cache_creation:
            result["cache_creation_input_tokens"] = cache_creation
        return result

    def _cached_prompt_tokens(self, usage: dict[str, Any]) -> int:
        # [FUNCTION DOC]
        # HELPER: Returns cached prompt tokens from usage.prompt_tokens_details.cached_tokens
        # or usage.cached_tokens if an int, else 0.
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if isinstance(cached, int):
                return cached
        cached = usage.get("cached_tokens")
        return cached if isinstance(cached, int) else 0

    def _error_response(self, message: str) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Returns an Anthropic-style error object of type api_error with message.
        return {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
