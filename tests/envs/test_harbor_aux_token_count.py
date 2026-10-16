# [MODULE DOC]
# Tests that the capture server's approximate_token_count (about 4 characters
# per token, minimum 1) reads OpenAI, Anthropic and Google request dialects,
# including Google contents/parts and camel- or snake-case system
# instructions, and tolerates malformed bodies. Skipped without the capture
# server module. TEXT is 400 chars, EXPECTED 100 tokens.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The token-count estimator has to read every dialect the aux routes accept.

`approximate_token_count` answers the harnesses' own token-counting endpoints (Anthropic's
`/v1/messages/count_tokens`, Google's `:countTokens`). It is a character estimate on purpose — the
alternative is re-rendering the chat template locally, which is the exact drift this layer exists to
avoid — but it has to at least *find* the text. It reads the body of whichever dialect called it, so a
dialect it does not know collapses to the `max(1, ...)` floor and answers 1 for a 50k-character
conversation. An agent uses that figure to decide when to compact, so a constant 1 means it never
compacts and blows its real context window mid-rollout.

Google was that dialect: the estimator knew `messages` and `system`, and Gemini sends `contents` with
`parts`, plus `systemInstruction`.
"""

from __future__ import annotations

import pytest

server = pytest.importorskip("openenv.core.harness.capture.server")
count = server.approximate_token_count

TEXT = "x" * 400  # ~100 tokens at the estimator's 4-chars-per-token rate
EXPECTED = 100


def test_an_empty_body_is_one_token_not_zero():
    # [FUNCTION DOC]
    # An empty body counts as 1.
    """The floor exists so a caller never divides by zero; only an empty body should hit it."""
    assert count({}) == 1


def test_openai_messages():
    # [FUNCTION DOC]
    # String message content is counted.
    assert count({"messages": [{"role": "user", "content": TEXT}]}) == EXPECTED


def test_openai_content_parts():
    # [FUNCTION DOC]
    # List-of-parts content is counted.
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": TEXT}]}]}
    assert count(body) == EXPECTED


def test_anthropic_system_block():
    # [FUNCTION DOC]
    # Anthropic system blocks are counted.
    assert (
        count({"messages": [], "system": [{"type": "text", "text": TEXT}]}) == EXPECTED
    )


def test_google_contents_are_counted():
    # [FUNCTION DOC]
    # Google contents parts are counted.
    """The regression: this used to be 1 regardless of how much text `contents` held."""
    body = {"contents": [{"role": "user", "parts": [{"text": TEXT}]}]}
    assert count(body) == EXPECTED, (
        "Google's `contents` were invisible to the estimator"
    )


def test_google_system_instruction_is_counted():
    # [FUNCTION DOC]
    # systemInstruction parts are added to contents.
    body = {
        "contents": [{"role": "user", "parts": [{"text": TEXT}]}],
        "systemInstruction": {"parts": [{"text": TEXT}]},
    }
    assert count(body) == 2 * EXPECTED


def test_google_snake_case_system_instruction():
    # [FUNCTION DOC]
    # system_instruction is also recognised.
    """The REST API is camelCase, the Python SDK emits snake_case; the proxy sees both."""
    body = {"contents": [], "system_instruction": {"parts": [{"text": TEXT}]}}
    assert count(body) == EXPECTED


def test_a_long_google_conversation_scales():
    # [FUNCTION DOC]
    # Ten turns count ten times as much.
    """A multi-turn body should grow with its length — pinning that it is not a per-request constant."""
    turns = [{"role": "user", "parts": [{"text": TEXT}]} for _ in range(10)]
    assert count({"contents": turns}) == 10 * EXPECTED


def test_malformed_parts_do_not_raise():
    # [FUNCTION DOC]
    # Unexpected shapes do not raise and give at least 1.
    """Bodies arrive from a sandboxed agent, so nothing here may throw on an unexpected shape."""
    body = {
        "contents": [{"parts": ["bare string", {"text": None}, 7]}, "not a dict", None],
        "systemInstruction": "a plain string",
    }
    assert count(body) >= 1
