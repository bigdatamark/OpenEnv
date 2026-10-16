# [MODULE DOC]
# Pydantic models for the chat environment: token actions, message/token history state, and
# observations carrying the full conversation and its tokens.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Chat Environment.

The Chat environment provides a chat-based interface for LLMs with support
for tokenization and message history management.
"""

from openenv.core.env_server.interfaces import Message
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


def _flatten_tokens(value) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: Recursively flattens tensors (anything with tolist), tuples and lists into a flat list of
    # ints; scalars are converted with int().
    # Raises: TypeError/ValueError for values not convertible to int.
    """Coerce nested tensor-like or sequence inputs into a flat token list."""
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_tokens(item))
        return flattened

    return [int(value)]


class ChatAction(Action):
    # [CLASS DOC]
    # Action holding a non-empty flat list of token ids (tensors or nested sequences are accepted and flattened).
    """Action for chat environments.

    Contains tokens that represent the action to be taken.
    This interfaces directly with models.
    """

    tokens: list[int] = Field(..., min_length=1)

    @field_validator("tokens", mode="before")
    @classmethod
    def _coerce_tokens(cls, value):
        # [FUNCTION DOC]
        # HELPER: Before-validator flattening list/tuple/tensor inputs.
        # Raises: TypeError for any other input type.
        """Accept either tensors or JSON arrays on the public HTTP surface."""
        if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
            return _flatten_tokens(value)
        raise TypeError("tokens must be provided as a sequence of token ids")


class ChatState(State):
    # [CLASS DOC]
    # Episode state: history_messages (role/content dicts) and history_tokens (one token list per message).
    """State of the ChatEnvironment containing message history."""

    history_messages: list[Message] = Field(default_factory=list)
    history_tokens: list[list[int]] = Field(default_factory=list)  # Same len as messages


class ChatObservation(Observation):
    # [CLASS DOC]
    # Observation with the full message history and a flat token list for the whole conversation;
    # inherits reward, done and metadata.
    """Observation returned by ChatEnvironment.

    Contains the message history in Huggingface format (list of dicts with role/content)
    and the tokenized representation of the entire conversation.

    The environment owns the tokenizer and generates the tokens from the messages.

    Example:
    messages = [
     {"role": "system", "content": "You are a helpful assistant"},
     {"role": "user", "content": "How tall is the Eiffel Tower?"},
    ]
    tokens = tensor([1, 2, 3, 4, 5, ...])  # tokenized entire conversation
    """

    messages: list[Message] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)
    # Inherited Fields from Observation ABC: reward, done, metadata
