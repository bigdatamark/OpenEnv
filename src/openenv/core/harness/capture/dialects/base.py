# [MODULE DOC]
# Abstract base for wire-dialect transformers (request to OpenAI chat format,
# response and stream chunks back to the source dialect) plus shared normalization:
# developer-role merging and per-model chat-template fixes.
"""Base transformer interface with inference-backend request enhancement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTransformer(ABC):
    # [CLASS DOC]
    # Abstract transformer. Subclasses implement transform_request, transform_response
    # and transform_stream_chunk; optional create_stream_state for stateful streaming.
    """Abstract base class for API transformers.

    Transforms requests from source API format to OpenAI format (for the
    inference backend), and transforms responses back to source API format.
    """

    @abstractmethod
    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract. Must convert a source-dialect request body to an OpenAI chat request.
        """Transform request body to OpenAI format for the inference backend."""
        pass

    @abstractmethod
    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract. Must convert an OpenAI response to the source dialect, given the original request.
        """Transform response back to source API format."""
        pass

    @abstractmethod
    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Abstract. Must convert one OpenAI stream chunk to one or more source-dialect events.
        """Transform a streaming chunk to source API format."""
        pass

    def is_streaming_request(self, body: dict[str, Any]) -> bool:
        # [FUNCTION DOC]
        # Returns body["stream"] (default False).
        # KNOWN ISSUE: annotated to return bool but returns the raw value, which may be non-bool.
        """Check if request is for streaming response."""
        return body.get("stream", False)

    def create_stream_state(self, original_request: dict[str, Any]) -> Any | None:
        # [FUNCTION DOC]
        # Default: returns None (no per-request stream state).
        """Create per-request stream state when chunk transforms need memory."""
        return None

    @staticmethod
    def _is_qwen35_model(model_name: str | None) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if the model name contains "qwen3.5" (case-insensitive); False for empty/None.
        if not model_name:
            return False
        return "qwen3.5" in model_name.lower()

    @staticmethod
    def _content_to_text(content: Any) -> str:
        # [FUNCTION DOC]
        # HELPER: Converts content to text: strings as-is; lists joined by newline over string
        # items and dict items with a string "text"; other truthy values stringified; falsy -> "".
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content) if content else ""

    @classmethod
    def _merge_developer_role(cls, request: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Renames "developer" role messages to "system", then merges all system
        # messages with text content into a single system message (texts joined by blank
        # lines) placed first. System messages whose content is non-empty but not reducible to
        # text are kept in place among the other messages. Non-list messages leave the
        # request unchanged.
        # Returns: the request (mutated in place). Side effects: replaces request["messages"].
        """Rename 'developer' role to 'system' and merge all system messages into one."""
        messages = request.get("messages")
        if not isinstance(messages, list):
            return request

        # Rename developer -> system
        normalized = [
            {**msg, "role": "system"}
            if isinstance(msg, dict) and msg.get("role") == "developer"
            else msg
            for msg in messages
        ]

        # Merge multiple system messages into one at the top
        system_parts: list[str] = []
        non_system: list[Any] = []
        for msg in normalized:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                text = cls._content_to_text(content)
                if text:
                    system_parts.append(text)
                elif content:
                    # A system message whose content is not reducible to text — an image part, an
                    # unfamiliar content type — used to vanish here: `_content_to_text` returned "",
                    # the falsy check skipped it, and the merged system prompt silently lost it. The
                    # prompt then differs from what the harness sent, which for a captured turn means
                    # the recorded prompt is not the one the model saw. Kept as its own message
                    # instead of merged, since it cannot be concatenated into the text block.
                    non_system.append(msg)
            else:
                non_system.append(msg)

        if system_parts:
            request["messages"] = [
                {"role": "system", "content": "\n\n".join(system_parts)},
                *non_system,
            ]
        else:
            request["messages"] = non_system
        return request

    def _normalize_request(
        self,
        request: dict[str, Any],
        model_name: str | None = None,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Removes the internal "_served_model" key, merges system roles, and for
        # Qwen3.5 models sets chat_template_kwargs.enable_thinking to False unless already set.
        # Returns: the (mutated) request.
        """Normalize the OpenAI request: drop internal keys, merge system roles,
        and apply per-model template fixes. Training-signal params (logprobs,
        token ids) are added later by the inference engine.
        """
        request.pop("_served_model", None)

        request = self._merge_developer_role(request)

        if self._is_qwen35_model(model_name):
            # Qwen3.5 outputs tool calls inside thinking; disable thinking.
            # https://www.reddit.com/r/LocalLLaMA/comments/1sccqt2/i_think_i_got_solutions_for_qwen_35_tool_call_in/
            chat_template_kwargs = dict(request.get("chat_template_kwargs") or {})
            chat_template_kwargs.setdefault("enable_thinking", False)
            request["chat_template_kwargs"] = chat_template_kwargs

        return request
