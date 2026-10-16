# [MODULE DOC]
# Defines the agent policy abstraction (transcript -> next assistant message) used by the rollout loop, plus an
# OpenAI-compatible chat implementation.
"""Agent policies for the AWM expert-in-the-loop rollout.

A *policy* turns a chat transcript into the next assistant message. Keeping this
behind a small protocol lets the same rollout loop drive

- an OpenAI-compatible chat model (used for the benchmark and for quick local runs), and
- a trainable model sampled by a GRPO training harness, where the policy also
  records sampled tokens.

Only the OpenAI-compatible policy is needed to run the benchmark; the training
policy is constructed inside the trainer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from llm import chat_complete


@runtime_checkable
class AgentPolicy(Protocol):
    # [CLASS DOC]
    # Runtime-checkable protocol: any object with an async complete(messages) method returning the assistant text.
    """Maps a chat transcript to the next assistant message.

    Args:
        messages (`list[dict]`):
            Chat messages in OpenAI format (`{"role": ..., "content": ...}`).

    Returns:
        `str`: The assistant message content.
    """

    async def complete(self, messages: list[dict[str, Any]]) -> str: ...


class OpenAIChatPolicy:
    # [CLASS DOC]
    # Policy that delegates each turn to chat_complete with a fixed client, model, optional temperature and max tokens.
        # [FUNCTION DOC]
        # Protocol method signature only; no implementation.
    """Agent policy backed by an OpenAI-compatible chat client.

    Works with `AsyncOpenAI` and `AsyncAzureOpenAI` clients. The same class is used
    for the small agent during the benchmark and, optionally, for a strong reference
    agent when measuring an upper bound.

    Args:
        client:
            An `AsyncOpenAI` / `AsyncAzureOpenAI` instance.
        model (`str`):
            Deployment or model name to sample from.
        temperature (`float`, *optional*):
            Sampling temperature for the agent's turns. Left unset by default so
            reasoning models work; pass a value for non-reasoning models.
        max_tokens (`int`, *optional*, defaults to `2048`):
            Maximum number of tokens to generate per turn.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> None:
        # [FUNCTION DOC]
        # Stores client, model, temperature and max tokens.
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        # [FUNCTION DOC]
        # Returns the assistant text for the given messages via chat_complete. Propagates its exceptions (e.g. RuntimeError on truncated empty output).
        return await chat_complete(
            self._client,
            self._model,
            messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
