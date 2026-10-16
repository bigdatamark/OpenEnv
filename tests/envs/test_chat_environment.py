# [MODULE DOC]
# Unit test for ChatEnvironment reset with a system prompt, using a minimal
# character-code tokenizer. Prepends repo root to sys.path.
"""Unit tests for the chat environment server implementation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.chat_env.server.chat_environment import ChatEnvironment


class SimpleTokenizer:
    # [CLASS DOC]
    # HELPER: tokenizer mapping each character to its code modulo 256.
    """Minimal tokenizer for reset/step unit tests."""

    def apply_chat_template(
        self,
        conversation,
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Concatenates message contents and returns the char-code list, wrapped in an
        # outer list when return_tensors is "pt"; other options ignored.
        del tokenize, kwargs
        text = "".join(message["content"] for message in conversation)
        tokens = [ord(ch) % 256 for ch in text]
        if return_tensors == "pt":
            return [tokens]
        return tokens

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        # [FUNCTION DOC]
        # Converts token ids (calling tolist when available) back to a string of chars.
        del skip_special_tokens, kwargs
        if hasattr(token_ids, "tolist") and callable(token_ids.tolist):
            token_ids = token_ids.tolist()
        return "".join(chr(token) for token in token_ids)


def test_chat_reset_with_system_prompt_keeps_tokens_serializable() -> None:
    # [FUNCTION DOC]
    # reset returns the system message and a non-empty token sequence.
    """Reset should work when a system prompt seeds token history."""
    env = ChatEnvironment(
        tokenizer=SimpleTokenizer(),
        system_prompt="You are a helpful AI assistant.",
    )

    observation = env.reset()

    assert observation.messages == [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    assert observation.tokens
