# [MODULE DOC]
# Tests for the capture RolloutGraph: turns link to parents by longest exact
# token prefix (or, when no token ids exist, by normalized message prefix),
# forks and abandoned branches, per-leaf training sequences with loss masks,
# turn lengths, stats, and canonical comparison of tool-call arguments.
# Skipped without the capture graph module.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rollout graph: how captured calls become training sequences.

This is the load-bearing piece of the capture layer. Turns are linked by exact token prefix and
nothing else, so every structural claim a trainer relies on (this is one conversation, this branch
was abandoned, these tokens are the model's own output) is a consequence of the linking rule. A bug
here does not crash: it silently produces training data that misattributes tokens.
"""

from __future__ import annotations

import pytest

graph_mod = pytest.importorskip("openenv.core.harness.capture.graph")

RolloutGraph = graph_mod.RolloutGraph
TurnNode = graph_mod.TurnNode
common_prefix_len = graph_mod.common_prefix_len


def node(node_id: str, prompt: list[int], sampled: list[int], **kwargs) -> TurnNode:
    # [FUNCTION DOC]
    # HELPER: builds a TurnNode with the given prompt and sampled ids and
    # logprobs of -0.1 per sampled token; extra kwargs are forwarded.
    return TurnNode(
        node_id=node_id,
        prompt_ids=prompt,
        sampled_ids=sampled,
        sampled_logprobs=[-0.1] * len(sampled),
        **kwargs,
    )


def chain(
    graph: RolloutGraph, *lengths: int, base: int = 0, context: int = 1
) -> list[TurnNode]:
    # [FUNCTION DOC]
    # HELPER: adds a linear conversation to the graph: turn i samples `length`
    # ids offset by base, and each later prompt is the previous node's end_ids plus
    # `context` interstitial tokens. Returns the added nodes.
    """Add a linear conversation.

    `context` is how many tokens the harness inserts between turns: a tool result plus the chat
    template's scaffolding. Real rollouts always have some, and turn boundaries are derived from
    those mask-0 runs, so a chain built without them is not representative.
    """
    added, prompt = [], [base]
    for i, length in enumerate(lengths):
        if i:
            prompt = prompt + [base + 900 + i] * context
        sampled = list(range(base + 100 + i * 10, base + 100 + i * 10 + length))
        current = graph.add_turn(node(f"n{base}_{i}", list(prompt), sampled))
        added.append(current)
        prompt = current.end_ids
    return added


# --- the linking rule -------------------------------------------------------
def test_common_prefix_len():
    # [FUNCTION DOC]
    # common_prefix_len counts leading equal elements.
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([1], [2]) == 0


def test_a_turn_whose_prompt_extends_another_becomes_its_child():
    # [FUNCTION DOC]
    # An extending prompt makes the second turn the child of the first.
    g = RolloutGraph()
    first, second = chain(g, 3, 4)
    assert second.parent_id == first.node_id
    assert g.children(first.node_id) == [second]


def test_an_unrelated_prompt_starts_a_new_root():
    # [FUNCTION DOC]
    # A different prefix starts a second root.
    """A different system prompt breaks the prefix, which is what makes it a separate conversation."""
    g = RolloutGraph()
    chain(g, 3)
    chain(g, 3, base=5000)
    assert len(g.roots()) == 2


def test_linking_ignores_arrival_order():
    # [FUNCTION DOC]
    # An unrelated turn arriving between parent and child does not affect linking.
    """Order of arrival must not decide structure; only the token prefix may."""
    g = RolloutGraph()
    parent = g.add_turn(node("p", [1, 2], [3, 4]))
    other = g.add_turn(node("other", [9, 9], [8]))
    child = g.add_turn(node("c", [1, 2, 3, 4], [5]))
    assert child.parent_id == parent.node_id
    assert other.parent_id is None


# --- forks and discards -----------------------------------------------------
def test_two_children_of_one_node_are_a_fork():
    # [FUNCTION DOC]
    # Two children of one node are reported as one fork.
    g = RolloutGraph()
    parent = g.add_turn(node("p", [1], [2, 3]))
    g.add_turn(node("a", [1, 2, 3], [4]))
    g.add_turn(node("b", [1, 2, 3], [5]))
    forks = g.forks()
    assert len(forks) == 1
    assert forks[0][0] == parent.node_id
    assert len(forks[0][1]) == 2


def test_an_abandoned_branch_is_discarded_and_the_continued_one_is_not():
    # [FUNCTION DOC]
    # A sibling that is never extended is discarded; the extended branch is kept.
    """A retry the agent walked away from must not be trained with the task's reward."""
    g = RolloutGraph()
    g.add_turn(node("p", [1], [2, 3]))
    g.add_turn(node("dead", [1, 2, 3], [4]))  # never extended
    g.add_turn(node("live", [1, 2, 3], [5]))
    g.add_turn(node("live2", [1, 2, 3, 5], [6]))  # extends `live`

    discarded = {n.node_id for n in g.discarded_nodes()}
    assert "dead" in discarded
    assert "live" not in discarded and "live2" not in discarded


# --- sequences, the actual training rows -------------------------------------
def test_sequence_masks_prompt_and_marks_only_sampled_tokens():
    # [FUNCTION DOC]
    # sequence_for has aligned arrays and loss mask 1 exactly on sampled tokens.
    g = RolloutGraph()
    first, second = chain(g, 2, 3)  # noqa: F841
    seq = g.sequence_for(second.node_id)

    assert len(seq.input_ids) == len(seq.loss_mask) == len(seq.logprobs)
    # Exactly the sampled tokens are trainable: 2 from the first turn, 3 from the second.
    assert sum(seq.loss_mask) == 5
    assert seq.n_trainable == 5
    trainable = [i for i, m in zip(seq.input_ids, seq.loss_mask) if m]
    assert trainable == first.sampled_ids + second.sampled_ids


def test_turn_lengths_match_what_each_turn_sampled():
    # [FUNCTION DOC]
    # turn_lengths returns each turn's sampled length.
    g = RolloutGraph()
    turns = chain(g, 2, 3, 4)
    seq = g.sequence_for(turns[-1].node_id)
    assert seq.turn_lengths() == [2, 3, 4]


def test_turn_lengths_merge_when_a_turn_adds_no_context():
    # [FUNCTION DOC]
    # With no interstitial context, adjacent turns merge into one run.
    """A property of `turn_lengths`, which no longer decides turn boundaries anywhere.

    Boundaries are runs of mask-1 tokens, so two turns with nothing between them read as one, and a
    turn with unusable logprobs contributes no run at all. `turns_from_document` used to zip
    `node_ids` against this, which dropped turns and misattributed the survivors; it now uses each
    node's own recorded prompt and sampled counts, so this limitation is confined to the helper.

    Kept because `turn_lengths` remains the join key when reconciling against an external trace,
    where a merged run would show up as a per-call count mismatch.
    """
    g = RolloutGraph()
    turns = chain(g, 2, 3, context=0)
    seq = g.sequence_for(turns[-1].node_id)
    assert seq.turn_lengths() == [5]
    assert len(seq.node_ids) == 2


def test_context_tokens_are_conditioned_on_but_never_trained():
    # [FUNCTION DOC]
    # Inserted tool-result tokens appear in the sequence with mask 0.
    """Tool results are real tokens the model saw and did not produce: mask 0, not absent."""
    g = RolloutGraph()
    parent = g.add_turn(node("p", [1, 2], [3]))
    # The harness inserted a tool result (99) between the turns.
    child = g.add_turn(node("c", [1, 2, 3, 99], [4]))

    assert child.context_ids(parent) == [99]
    seq = g.sequence_for(child.node_id)
    assert 99 in seq.input_ids
    assert seq.loss_mask[seq.input_ids.index(99)] == 0


def test_one_sequence_per_leaf():
    # [FUNCTION DOC]
    # One sequence is produced per leaf.
    g = RolloutGraph()
    g.add_turn(node("p", [1], [2]))
    g.add_turn(node("a", [1, 2], [3]))
    g.add_turn(node("b", [1, 2], [4]))
    assert len(g.sequences()) == len(g.leaves()) == 2


def test_stats_report_the_shape():
    # [FUNCTION DOC]
    # stats reports turn, root and leaf counts.
    g = RolloutGraph()
    chain(g, 2, 2)
    chain(g, 2, base=5000)
    stats = g.stats()
    assert stats["n_turns"] == 3
    assert stats["n_roots"] == 2
    assert stats["n_leaves"] == 2


def test_empty_graph_is_not_an_error():
    # [FUNCTION DOC]
    # An empty graph returns empty lists and zero turns.
    g = RolloutGraph()
    assert g.nodes() == [] and g.roots() == [] and g.sequences() == []
    assert g.stats()["n_turns"] == 0


# --- linking without token ids (an eval endpoint) ---------------------------
#
# A hosted provider returns no token ids at all, so `end_ids` is empty for every node and the token
# rule can never find a parent: `len(end) <= best_len` holds for all candidates. The graph would
# report a 20-turn conversation as 20 separate roots — not wrong exactly, but it reads as if the
# agent restarted every turn, and every root-count heuristic downstream misfires.
def eval_node(node_id: str, messages: list[dict], reply: str) -> TurnNode:
    # [FUNCTION DOC]
    # HELPER: builds a TurnNode without token ids or logprobs, carrying request
    # messages and an assistant response with the given content.
    return TurnNode(
        node_id=node_id,
        prompt_ids=[],
        sampled_ids=[],
        sampled_logprobs=None,
        request_messages=messages,
        response_message={"role": "assistant", "content": reply},
    )


def test_message_prefix_links_an_eval_conversation_into_one_root():
    # [FUNCTION DOC]
    # Message-prefix linking chains three eval turns into one root.
    graph = RolloutGraph()
    first = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    graph.add_turn(eval_node("a", first, "step 1"))
    second = [
        *first,
        {"role": "assistant", "content": "step 1"},
        {"role": "user", "content": "tool result"},
    ]
    graph.add_turn(eval_node("b", second, "step 2"))
    third = [
        *second,
        {"role": "assistant", "content": "step 2"},
        {"role": "user", "content": "tool result 2"},
    ]
    graph.add_turn(eval_node("c", third, "done"))

    assert graph.stats()["n_roots"] == 1
    assert graph.stats()["n_turns"] == 3
    assert graph.get("b").parent_id == "a"
    assert graph.get("c").parent_id == "b"


def test_an_unrelated_eval_conversation_is_still_its_own_root():
    # [FUNCTION DOC]
    # A different system prompt yields a separate root.
    """A subagent starts from a fresh system prompt and must not be grafted onto the main chain."""
    graph = RolloutGraph()
    graph.add_turn(eval_node("a", [{"role": "system", "content": "main"}], "working"))
    graph.add_turn(
        eval_node("b", [{"role": "system", "content": "subagent"}], "also working")
    )
    assert graph.stats()["n_roots"] == 2


def test_provider_noise_on_the_assistant_message_does_not_break_linking():
    # [FUNCTION DOC]
    # Extra null/empty provider fields on the assistant message do not prevent linking.
    """The message a provider returns and the one a harness echoes back are equal in meaning and
    unequal as dicts: `refusal`, `annotations` and `audio: null` get added, `content` moves between
    `null` and `""`. Comparing raw dicts would find no parent for any turn."""
    graph = RolloutGraph()
    first = [{"role": "user", "content": "go"}]
    parent = TurnNode(
        node_id="a",
        prompt_ids=[],
        sampled_ids=[],
        request_messages=first,
        response_message={
            "role": "assistant",
            "content": "step 1",
            "refusal": None,
            "annotations": [],
            "audio": None,
        },
    )
    graph.add_turn(parent)
    graph.add_turn(
        eval_node(
            "b",
            [
                *first,
                {"role": "assistant", "content": "step 1"},
                {"role": "user", "content": "next"},
            ],
            "step 2",
        )
    )
    assert graph.get("b").parent_id == "a"


def test_token_linking_still_wins_when_ids_are_present():
    # [FUNCTION DOC]
    # With token ids, token linking is used.
    """Messages are a weaker key — what the harness said it sent, not what the engine tokenised — so
    they must never be consulted while ids are available."""
    graph = RolloutGraph()
    turns = chain(graph, 3, 4)
    assert graph.get(turns[1].node_id).parent_id == turns[0].node_id
    assert graph.stats()["n_roots"] == 1


def test_tool_call_arguments_are_compared_as_json_not_as_bytes():
    # [FUNCTION DOC]
    # Tool-call arguments differing only in JSON whitespace still link.
    """The bug that made message linking inert on real data.

    An eight-turn opencode rollout against the HF router came back as eight separate roots. The
    arguments were identical; the *strings* were not, differing only in the space after the colon,
    because the harness re-serialises what the provider sent:

        {"command": "ls"}   provider
        {"command":"ls"}    echoed back
    """
    graph = RolloutGraph()
    first = [{"role": "user", "content": "go"}]
    call = {"id": "call_1", "type": "function", "function": {"name": "bash"}}
    graph.add_turn(
        TurnNode(
            node_id="a",
            prompt_ids=[],
            sampled_ids=[],
            request_messages=first,
            response_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
            },
        )
    )
    graph.add_turn(
        eval_node(
            "b",
            [
                *first,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            **call,
                            "function": {
                                **call["function"],
                                "arguments": '{"command":"ls"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "a.txt"},
            ],
            "done",
        )
    )
    assert graph.get("b").parent_id == "a"
    assert graph.stats()["n_roots"] == 1


def test_reordered_argument_keys_are_still_the_same_call():
    # [FUNCTION DOC]
    # _canonical_arguments ignores key order.
    """Any harness that round-trips arguments through a dict can reorder the keys."""
    from openenv.core.harness.capture.graph import _canonical_arguments

    assert _canonical_arguments('{"b": 2, "a": 1}') == _canonical_arguments(
        '{"a":1,"b":2}'
    )


def test_malformed_arguments_are_not_forced_to_match():
    # [FUNCTION DOC]
    # Different malformed strings stay different; surrounding whitespace is ignored.
    """Two different malformed strings are two different calls, not one."""
    from openenv.core.harness.capture.graph import _canonical_arguments

    assert _canonical_arguments("{not json") != _canonical_arguments("{also not json")
    assert _canonical_arguments("{not json") == _canonical_arguments("  {not json  ")
