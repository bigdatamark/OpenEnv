# [MODULE DOC]
# OpenAI chat-completions dialect: a near-passthrough transformer used when the
# harness already speaks the backend's format.
"""OpenAI Chat Completions transformer with SGLang training enhancements."""

from __future__ import annotations

from typing import Any

from .base import BaseTransformer


class OpenAIChatTransformer(BaseTransformer):
    # [CLASS DOC]
    # Passthrough transformer for OpenAI chat requests; only normalizes the request and
    # rewrites the model name in responses.
    """Transform OpenAI Chat requests (passthrough + training params)."""

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Shallow-copies the body, copies max_completion_tokens into max_tokens when only the
        # former is present, then applies _normalize_request with _served_model.
        # Returns: the normalized copy.
        result = body.copy()
        if "max_tokens" not in result and "max_completion_tokens" in result:
            result["max_tokens"] = result["max_completion_tokens"]
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
        # Returns a shallow copy of the response with "model" set to the original request's
        # model when present.
        result = response.copy()
        if "model" in original_request:
            result["model"] = original_request["model"]
        return result

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a shallow copy of the chunk with "model" set to the original request's model
        # when present. is_first is ignored.
        result = chunk.copy()
        if "model" in original_request:
            result["model"] = original_request["model"]
        return result
