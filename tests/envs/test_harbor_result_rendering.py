# [MODULE DOC]
# Tests for openenv.harbor.models (conversations_from_document, turns_from_document) and the private
# HTML/contract renderers in openenv.harbor.ui: fork deduplication, both tool-call dialects, verdict
# labels, findings grouping, escaping, and the training-contract export.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turning a capture document into a result, and a result into something readable.

Two dialect families put tool calls in different places (`tool_calls` vs `tool_use` content blocks),
so a reader that knows only one shows an agent as a stream of text with no visible actions. And a
forked conversation appears once per path, which rendered as several near-identical transcripts all
claiming to be the main one.
"""

from __future__ import annotations

import pytest

models = pytest.importorskip("openenv.harbor.models")
ui = pytest.importorskip("openenv.harbor.ui")

conversations_from_document = models.conversations_from_document
turns_from_document = models.turns_from_document


def document(sequences, turns):
    # [FUNCTION DOC]
    # HELPER: Wraps sequences and turns lists into a capture-document dict.
    return {"sequences": sequences, "turns": turns}


# --- conversations ----------------------------------------------------------
def test_a_forked_root_yields_one_conversation_not_one_per_path():
    # [FUNCTION DOC]
    # Two sequences sharing one root yield a single conversation whose n_turns is the longest path (2).
    """The regression: two paths through one root rendered as two 'main conversation' blocks."""
    doc = document(
        sequences=[
            {"root_id": "r1", "role": "agent", "node_ids": ["a"], "n_turns": 1},
            {"root_id": "r1", "role": "agent", "node_ids": ["a", "b"], "n_turns": 2},
        ],
        turns=[
            {
                "node_id": "a",
                "request_messages": [{"role": "user", "content": "hi"}],
                "response_message": {"content": "one"},
            },
            {
                "node_id": "b",
                "request_messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "one"},
                ],
                "response_message": {"content": "two"},
            },
        ],
    )
    convos = conversations_from_document(doc)
    assert len(convos) == 1
    assert convos[0].n_turns == 2, "the longest path is the complete one"


def test_separate_roots_stay_separate():
    # [FUNCTION DOC]
    # Sequences with different roots produce separate conversations with their own roles.
    doc = document(
        sequences=[
            {"root_id": "r1", "role": "agent", "node_ids": ["a"], "n_turns": 1},
            {"root_id": "r2", "role": "auxiliary", "node_ids": ["b"], "n_turns": 1},
        ],
        turns=[
            {
                "node_id": "a",
                "request_messages": [{"role": "user", "content": "task"}],
                "response_message": {"content": "working"},
            },
            {
                "node_id": "b",
                "request_messages": [{"role": "user", "content": "who next?"}],
                "response_message": {"content": "agent"},
            },
        ],
    )
    convos = conversations_from_document(doc)
    assert {c.role for c in convos} == {"agent", "auxiliary"}


def test_a_conversation_keeps_the_system_prompt_and_tool_results():
    # [FUNCTION DOC]
    # The conversation messages keep system, user and tool roles and append the assistant reply.
    doc = document(
        sequences=[{"root_id": "r", "role": "agent", "node_ids": ["a"], "n_turns": 1}],
        turns=[
            {
                "node_id": "a",
                "request_messages": [
                    {"role": "system", "content": "You are an assistant."},
                    {"role": "user", "content": "count rows"},
                    {"role": "tool", "content": "42"},
                ],
                "response_message": {"content": "42 rows"},
            }
        ],
    )
    roles = [m["role"] for m in conversations_from_document(doc)[0].messages]
    assert roles == ["system", "user", "tool", "assistant"]


def test_sequences_without_nodes_or_messages_are_skipped():
    # [FUNCTION DOC]
    # A sequence with no nodes produces no conversation.
    doc = document(
        sequences=[{"root_id": "r", "role": "agent", "node_ids": [], "n_turns": 0}],
        turns=[],
    )
    assert conversations_from_document(doc) == []


# --- turns ------------------------------------------------------------------
def _agent_doc(response):
    # [FUNCTION DOC]
    # HELPER: Builds a one-turn agent document with token ids, loss mask, logprobs and the given
    # response message.
    return document(
        sequences=[
            {
                "root_id": "r",
                "role": "agent",
                "node_ids": ["a"],
                "n_turns": 1,
                "input_ids": [1, 2, 3],
                "loss_mask": [0, 1, 1],
                "logprobs": [0.0, -0.1, -0.2],
                "prompt_len": 1,
                "turn_lengths": [2],
            }
        ],
        turns=[
            {
                "node_id": "a",
                "finish_reason": "stop",
                "n_tools": 3,
                "response_message": response,
            }
        ],
    )


def test_turn_text_and_tool_calls_from_chat_completions():
    # [FUNCTION DOC]
    # OpenAI-style tool_calls are flattened into name/arguments dicts and text is kept.
    turns = turns_from_document(
        _agent_doc(
            {
                "content": "Reading the file.",
                "tool_calls": [
                    {"function": {"name": "bash", "arguments": '{"cmd":"ls"}'}}
                ],
            }
        )
    )
    assert turns[0].text == "Reading the file."
    assert turns[0].tool_calls == [{"name": "bash", "arguments": '{"cmd":"ls"}'}]


def test_turn_text_and_tool_calls_from_anthropic_blocks():
    # [FUNCTION DOC]
    # Anthropic content blocks yield the text block as text and tool_use blocks as tool calls.
    """claude-code puts tool use in content blocks; reading only `tool_calls` shows no actions."""
    turns = turns_from_document(
        _agent_doc(
            {
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ],
            }
        )
    )
    assert turns[0].text == "Checking."
    assert turns[0].tool_calls[0]["name"] == "Bash"


def test_a_turn_with_no_response_is_still_a_turn():
    # [FUNCTION DOC]
    # An empty response still produces one turn with empty text and no tool calls.
    turns = turns_from_document(_agent_doc({}))
    assert len(turns) == 1 and turns[0].text == "" and turns[0].tool_calls == []


def test_only_agent_sequences_become_turns():
    # [FUNCTION DOC]
    # An auxiliary sequence produces no turns.
    """An auxiliary call must never be credited with the reward for solving the task."""
    doc = _agent_doc({"content": "x"})
    doc["sequences"][0]["role"] = "auxiliary"
    assert turns_from_document(doc) == []


# --- rendering --------------------------------------------------------------
@pytest.mark.parametrize(
    "result,marker",
    [
        ({"ok": True, "reward": 1.0}, "Solved"),
        ({"ok": True, "reward": 0.0}, "Not solved"),
        ({"ok": True, "reward": None}, "Not graded"),
        ({"ok": False, "reward": None, "exception_type": "Boom"}, "Failed"),
    ],
)
def test_every_verdict_state_renders(result, marker):
    # [FUNCTION DOC]
    # Parametrized: _result_html shows Solved / Not solved / Not graded / Failed per reward and ok state.
    assert marker in ui._result_html({**result, "turns": []})


def test_ungraded_shows_a_dash_rather_than_a_zero():
    # [FUNCTION DOC]
    # An ungraded result's HTML does not contain "0.00".
    """A dead sandbox rendered as 0.00 reads as the model getting the answer wrong."""
    out = ui._result_html({"ok": True, "reward": None, "turns": []})
    assert "0.00" not in out


def test_findings_are_grouped_by_severity():
    # [FUNCTION DOC]
    # _findings_html includes FATAL, WARN and INFO groups with FATAL before WARN.
    out = ui._findings_html(["[FATAL] gone", "[WARN] odd", "[INFO] fyi"])
    assert "FATAL" in out and "WARN" in out and "INFO" in out
    assert out.index("FATAL") < out.index("WARN"), "worst first"


def test_no_findings_renders_nothing():
    # [FUNCTION DOC]
    # _findings_html of an empty list is the empty string.
    assert ui._findings_html([]) == ""


def test_conversation_labels_are_unambiguous_when_several_exist():
    # [FUNCTION DOC]
    # Two agent conversations are labelled "conversation 1 of 2"/"2 of 2", never "main conversation".
    convos = [
        {"role": "agent", "n_turns": 1, "messages": [{"role": "user", "content": "a"}]},
        {"role": "agent", "n_turns": 1, "messages": [{"role": "user", "content": "b"}]},
    ]
    out = ui._conversation_html({"conversations": convos})
    assert out.count("main conversation") == 0
    assert "conversation 1 of 2" in out and "conversation 2 of 2" in out


def test_turns_html_does_not_show_the_tools_offered_count():
    # [FUNCTION DOC]
    # _turns_html omits the tools-offered count but shows tool names and a confidence column.
    """That number is a property of the harness, identical on every row, and told nobody anything."""
    out = ui._turns_html(
        {
            "turns": [
                {
                    "turn": 0,
                    "completion_token_ids": [1, 2],
                    "per_token_logps": [-0.1, -0.2],
                    "tool_calls": [{"name": "bash", "arguments": "ls"}],
                    "n_tools": 24,
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    assert "24 tools" not in out
    assert "bash" in out and "confidence" in out.lower()


def test_escaping_prevents_markup_injection_from_a_model_reply():
    # [FUNCTION DOC]
    # Assistant content containing a script tag is HTML-escaped.
    out = ui._conversation_html(
        {
            "conversations": [
                {
                    "role": "agent",
                    "n_turns": 1,
                    "messages": [
                        {"role": "assistant", "content": "<script>alert(1)</script>"}
                    ],
                }
            ]
        }
    )
    assert "<script>" not in out and "&lt;script&gt;" in out


# --- the training contract --------------------------------------------------
def test_contract_keeps_full_logprobs(tmp_path):
    # [FUNCTION DOC]
    # _contract keeps per-token logprobs, prompt ids and reward.
    """The reason it is a separate artifact: these cannot be recovered by replaying the prompt."""
    data = ui._contract(
        {
            "task_name": "t",
            "reward": 1.0,
            "turns": [
                {
                    "turn": 0,
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3, 4],
                    "per_token_logps": [-0.01, -0.02],
                    "finish_reason": "stop",
                    "discarded": False,
                }
            ],
        }
    )
    assert data["turns"][0]["per_token_logps"] == [-0.01, -0.02]
    assert data["turns"][0]["prompt_token_ids"] == [1, 2]
    assert data["reward"] == 1.0


def test_contract_keeps_discarded_turns_but_flags_them():
    # [FUNCTION DOC]
    # _contract retains discarded turns with discarded True.
    """A trainer must be able to exclude them deliberately, not never learn they happened."""
    data = ui._contract(
        {
            "task_name": "t",
            "reward": 0.0,
            "turns": [
                {
                    "turn": 0,
                    "completion_token_ids": [1],
                    "per_token_logps": [-0.1],
                    "discarded": True,
                }
            ],
        }
    )
    assert data["turns"][0]["discarded"] is True


def test_no_turns_means_no_contract():
    # [FUNCTION DOC]
    # _contract returns None when there are no turns.
    assert ui._contract({"task_name": "t", "turns": []}) is None
