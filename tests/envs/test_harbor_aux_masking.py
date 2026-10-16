# [MODULE DOC]
# Tests for openenv.harbor.rollout._mask_out_nodes: masking an auxiliary
# node's sampled span out of an exported capture sequence that also holds real
# agent turns, at any position and with varying interstitial context.
# Skipped if the capture graph/export or harbor rollout modules are missing.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Masking an auxiliary call out of a sequence that also contains real agent turns.

Auxiliary detection is per node while demotion used to be per sequence, so a sequence mixing an aux
call with genuine agent turns stayed `agent` in full and shipped the aux call as a training turn
credited with the task's reward. Masking the aux node's sampled span is the fix — and getting the span
arithmetic wrong makes the function silently do nothing, which is what happened first: an offset was
advanced as if each turn were only prompt-plus-sampled, so from the second turn on it zeroed
already-masked context and left the real completion tokens at 1.

The middle turn is the load-bearing case. Masking the FIRST turn works under either arithmetic.
"""

from __future__ import annotations

import pytest

graph_mod = pytest.importorskip("openenv.core.harness.capture.graph")
export_mod = pytest.importorskip("openenv.core.harness.capture.export")
rollout_mod = pytest.importorskip("openenv.harbor.rollout")


def chain(lengths, *, context=1):
    # [FUNCTION DOC]
    # HELPER: builds a linear RolloutGraph where turn i samples lengths[i] distinct
    # token ids (logprob -0.1, one tool) and each next prompt is the previous
    # prompt plus its sampled ids plus `context` interstitial tokens. Returns the graph.
    """A linear agent chain with `context` interstitial tokens between turns."""
    graph = graph_mod.RolloutGraph()
    prompt = [1, 2, 3]
    for index, n_sampled in enumerate(lengths):
        sampled = list(range(100 + index * 50, 100 + index * 50 + n_sampled))
        graph.add_turn(
            graph_mod.TurnNode(
                node_id=f"n{index}",
                prompt_ids=list(prompt),
                sampled_ids=sampled,
                sampled_logprobs=[-0.1] * n_sampled,
                n_tools=1,
            )
        )
        prompt = prompt + sampled + [900 + index] * context
    return graph


def document(graph):
    # [FUNCTION DOC]
    # HELPER: wraps the graph in a minimal session object and returns
    # export_session's document.
    class Session:
        # [CLASS DOC]
        # Minimal session stand-in with id "s", empty metadata and findings.
        session_id = "s"
        metadata: dict = {}
        findings: list = []

    session = Session()
    session.graph = graph
    return export_mod.export_session(session)


def span(doc, node_id):
    # [FUNCTION DOC]
    # HELPER: returns (start, end) of a node's sampled tokens in the sequence,
    # computed from its n_prompt and n_sampled.
    node = next(t for t in doc["turns"] if t["node_id"] == node_id)
    return node["n_prompt"], node["n_prompt"] + node["n_sampled"]


@pytest.mark.parametrize("aux_index", [0, 1, 2])
def test_the_aux_span_is_zeroed_wherever_it_sits(aux_index):
    # [FUNCTION DOC]
    # For aux node 0, 1 or 2, its sampled span is fully zeroed, exactly its token
    # count is removed from the loss mask, and n_trainable matches the mask sum.
    """Parametrised across positions because only the non-first cases catch the offset bug."""
    graph = chain([4, 5, 6])
    doc = document(graph)
    sequence = doc["sequences"][0]
    before = sum(sequence["loss_mask"])
    aux = f"n{aux_index}"
    start, end = span(doc, aux)

    rollout_mod._mask_out_nodes(doc, sequence, {aux})

    assert all(m == 0 for m in sequence["loss_mask"][start:end]), (
        f"the aux node's sampled span {start}:{end} must be fully masked"
    )
    expected_removed = [4, 5, 6][aux_index]
    assert sum(sequence["loss_mask"]) == before - expected_removed, (
        "exactly the aux node's tokens should stop being targets"
    )
    assert sequence["n_trainable"] == sum(sequence["loss_mask"])


def test_the_other_turns_keep_every_target():
    # [FUNCTION DOC]
    # Masking the middle node leaves the other nodes' targets intact.
    graph = chain([4, 5, 6])
    doc = document(graph)
    sequence = doc["sequences"][0]
    rollout_mod._mask_out_nodes(doc, sequence, {"n1"})
    for node_id, length in (("n0", 4), ("n2", 6)):
        start, end = span(doc, node_id)
        assert sum(sequence["loss_mask"][start:end]) == length, (
            f"{node_id} lost targets it should have kept"
        )


def test_masking_every_node_leaves_nothing_trainable():
    # [FUNCTION DOC]
    # Masking all nodes gives zero targets and trainable False.
    graph = chain([3, 3])
    doc = document(graph)
    sequence = doc["sequences"][0]
    rollout_mod._mask_out_nodes(doc, sequence, {"n0", "n1"})
    assert sum(sequence["loss_mask"]) == 0
    assert sequence["n_trainable"] == 0
    assert sequence["trainable"] is False


def test_wider_interstitial_context_does_not_shift_the_span():
    # [FUNCTION DOC]
    # With 7 context tokens between turns, masking node 1 zeroes exactly its span.
    """The offset bug scaled with the amount of context between turns, so vary it."""
    graph = chain([4, 5], context=7)
    doc = document(graph)
    sequence = doc["sequences"][0]
    start, end = span(doc, "n1")
    rollout_mod._mask_out_nodes(doc, sequence, {"n1"})
    assert all(m == 0 for m in sequence["loss_mask"][start:end])
    assert sum(sequence["loss_mask"]) == 4
