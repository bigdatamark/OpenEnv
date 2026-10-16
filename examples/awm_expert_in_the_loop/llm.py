# [MODULE DOC]
# HELPER: Single chat-completion helper compatible with both reasoning and non-reasoning OpenAI-style models
# (always uses max_completion_tokens; temperature optional with automatic retry when rejected).
"""A small OpenAI-compatible chat helper that tolerates reasoning models.

Reasoning models (for example the GPT-5 family) reject non-default ``temperature``
and require ``max_completion_tokens``. ``chat_complete`` sends ``temperature`` only when explicitly requested and
transparently retries without it if the server rejects the value, so the same call
site works for both reasoning and non-reasoning models. It also refuses to treat a
length-truncated, empty response as a final answer.
"""

from __future__ import annotations

from typing import Any


def _is_temperature_unsupported(exc: Exception) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if an exception message mentions "temperature" together with "does not support" or "unsupported value".
    text = str(exc).lower()
    return "temperature" in text and (
        "does not support" in text or "unsupported value" in text
    )


async def chat_complete(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float | None = None,
) -> str:
    # [FUNCTION DOC]
    # Calls client.chat.completions.create with model, messages, max_completion_tokens and (if given) temperature. If the
    # call fails with a temperature-unsupported error, retries once without temperature; other errors are re-raised.
    # Returns the first choice's content (empty string if None).
    # Raises: RuntimeError if the content is empty and finish_reason is "length"; any client exception otherwise.
    """Return the assistant text for a chat completion.

    Args:
        client:
            An `AsyncOpenAI` / `AsyncAzureOpenAI` instance.
        model (`str`):
            Deployment or model name.
        messages (`list[dict]`):
            Chat messages in OpenAI format.
        max_tokens (`int`):
            Maximum number of completion tokens.
        temperature (`float`, *optional*):
            Sampling temperature. Omitted from the request when `None`; if the model
            rejects the value (reasoning models), the call is retried without it.

    Returns:
        `str`: The assistant message content (empty string if none).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        if temperature is not None and _is_temperature_unsupported(exc):
            kwargs.pop("temperature", None)
            response = await client.chat.completions.create(**kwargs)
        else:
            raise
    choice = response.choices[0]
    content = choice.message.content or ""
    if not content and getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(
            f"Model {model!r} exhausted max_completion_tokens={max_tokens} before "
            "returning visible content."
        )
    return content
