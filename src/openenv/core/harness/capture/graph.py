# [MODULE DOC]
# Rollout graph data structures: TurnNode (one model call with engine prompt ids,
# sampled ids and logprobs plus provenance), TrainingSequence (one flattened
# root-to-leaf path with ids, loss mask and logprobs) and RolloutGraph, which links
# calls by longest exact token prefix (falling back to message-list prefixes when no
# token ids exist), detects retries/forks/discards structurally, and flattens paths.
# Pure in-memory; no I/O.
"""The rollout graph: every model call a harness made, linked by token prefix.

A rollout is not a list of turns. Harnesses retry, spawn subagents, generate titles, and compact
context, and all of it arrives on one wire looking identical. A flat list forces you to guess which
turns belong together; a graph records it.

    node        one model call: the prompt the server tokenized, the tokens it sampled back
    parent      the call whose prompt+completion is a token-prefix of this call's prompt
    root        a call that extends nothing (a new conversation: the agent's, a subagent's,
                a title generator's, or the continuation after a context compaction)
    path        root -> leaf, which is exactly one training sequence

Everything downstream is a walk. `sequences()` concatenates a path into
`input_ids / loss_mask / logprobs`; branch structure tells you which paths are the agent's work and
which are discards.

WHY A GRAPH AND NOT PREFIX-MERGED CHAINS. Three things fall out of it that a chain cannot express:

  * **Retries become visible.** A retried turn is a sibling that never continued: same parent, no
    children. A proxy cannot ask the harness what it discarded, but the shape of the graph shows it.
  * **Subagents separate themselves.** A subagent has its own system prompt, so its first call
    extends nothing and starts its own root. No system-prompt keyword matching required.
  * **Compaction is representable.** A rewritten history is not a prefix extension, so it opens a
    new root instead of corrupting the chain it came from.

TOKEN FIDELITY. We never tokenize. The inference server tokenizes each prompt as a side effect of
serving it and returns `prompt_token_ids`, so turn k+1's prompt IS the canonical tokenization of
everything up to that point, including the tool results the harness inserted. Assistant bodies come
back as sampled `token_ids` with aligned logprobs. So for any path:

    concat(node.context_ids + node.sampled_ids for node in path)

reproduces, exactly, the token sequences the model actually saw and produced. Nothing is re-rendered
through a local chat template, which is the single largest source of silent train/inference skew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator


def common_prefix_len(a: list[int], b: list[int]) -> int:
    # [FUNCTION DOC]
    # Returns the number of leading equal elements of two sequences.
    """How many leading tokens `a` and `b` share."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _canonical_arguments(arguments: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: Canonicalizes tool-call arguments for comparison: None -> ""; non-strings are
    # JSON-dumped with sorted keys (str() if not serializable); JSON strings are re-dumped
    # with sorted keys; non-JSON strings are stripped.
    """Tool-call arguments in a form that survives a harness re-serialising them.

    `arguments` travels the wire as a JSON *string*, and the string an agent sends back is not the
    string the provider produced. Observed on a real opencode rollout against the HF router, the same
    `bash` call arrived as

        {"command": "python3 -c ..."}      from the provider
        {"command":"python3 -c ..."}       echoed back by the harness

    — identical arguments, different bytes, differing only in the space after the colon. Comparing
    the raw strings made the message fallback in `_find_parent_by_messages` inert on the very case it
    was written for: an eight-turn rollout came back as eight separate roots. Key ordering is the same
    hazard from any harness that round-trips through a dict.

    Falls back to the raw string when it is not JSON, which is the honest answer for a model that
    emitted malformed arguments: two different malformed strings are two different calls.
    """
    if arguments is None:
        return ""
    if not isinstance(arguments, str):
        # Some harnesses hand back an already-decoded object rather than a string.
        try:
            return json.dumps(arguments, sort_keys=True)
        except (TypeError, ValueError):
            return str(arguments)
    try:
        return json.dumps(json.loads(arguments), sort_keys=True)
    except (json.JSONDecodeError, TypeError, ValueError):
        return arguments.strip()


def _message_identity(message: Any) -> tuple:
    # [FUNCTION DOC]
    # HELPER: Returns a comparison key (role, stripped text content, tuple of (tool name,
    # canonical arguments)) for a message. List content is reduced to concatenated
    # text/input_text parts; non-dict messages become ("", str(message), ()).
    """The part of a message that decides whether two messages are the same turn.

    Compared instead of the raw dicts because the assistant message a provider returns and the one a
    harness sends back in its next request are equal in meaning and unequal as dicts: providers add
    `refusal`, `annotations` and `audio: null`, harnesses drop them, and `content` moves between
    `null` and `""`. Comparing dicts directly would find no parent for any turn and report every call
    as its own root — the exact failure the message fallback exists to avoid.

    Tool calls are reduced to name and arguments: the `id` is provider-generated and does survive a
    round trip, but keying on it would make the comparison fail for any harness that rewrites ids.
    The arguments are compared as PARSED JSON, not as the string the wire carried — see
    `_canonical_arguments`.
    """
    if not isinstance(message, dict):
        return ("", str(message), ())
    content = message.get("content")
    if isinstance(content, list):
        # Multimodal content: keep only the text parts, in order. Image bytes are large and their
        # encoding is not stable across a round trip.
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
        )
    calls = tuple(
        (
            str((call.get("function") or {}).get("name", "")),
            _canonical_arguments((call.get("function") or {}).get("arguments")),
        )
        for call in (message.get("tool_calls") or [])
        if isinstance(call, dict)
    )
    return (
        str(message.get("role") or ""),
        (str(content) if content is not None else "").strip(),
        calls,
    )


def _same_message(a: Any, b: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if both messages have equal _message_identity keys.
    """Whether two messages are the same conversational turn. See `_message_identity`."""
    return _message_identity(a) == _message_identity(b)


def _without_leading_system(messages: Any) -> list:
    # [FUNCTION DOC]
    # HELPER: Returns a new list with leading system-role messages removed ([] for non-lists).
    """The message list with any leading system messages removed.

    Used only for PARENT LOOKUP, never for what gets stored. Some harnesses re-render their system
    prompt on every step, so the same conversation carries a slightly different system message each
    turn: claude-code's differs after the first ~150 characters while the rest of the history matches
    exactly. Comparing it would fail at index 0, no parent would be found for any turn, and a 13-turn
    rollout is reported as 13 separate roots — the failure `_find_parent_by_messages` exists to
    prevent, arriving through a different door than the missing-role case it was written for.
    Measured downstream: a trace of 13 turns became 101 assistant messages once those false roots
    were concatenated.

    Leading system messages are dropped rather than compared loosely so that a conversation which
    GAINS or LOSES a system message still lines up; comparing by role alone would keep the lists
    aligned only while both have one.
    """
    if not isinstance(messages, list):
        return []
    i = 0
    while i < len(messages):
        m = messages[i]
        if isinstance(m, dict) and (m.get("role") or "") == "system":
            i += 1
            continue
        break
    return list(messages[i:])


@dataclass
class TurnNode:
    # [CLASS DOC]
    # One model call. Fields: node_id, prompt_ids (engine tokenization), sampled_ids,
    # sampled_logprobs (None if unavailable), parent_id; provenance fields index (arrival
    # order), model, finish_reason, harness_session_id, system_digest, n_tools,
    # request_messages, request_tools, sampling_params (actually submitted),
    # requested_sampling_params (as the harness asked) and response_message.
    """One model call, and where it sits relative to the call before it."""

    node_id: str
    prompt_ids: list[int]
    sampled_ids: list[int]
    sampled_logprobs: list[float] | None = None
    parent_id: str | None = None

    # Provenance, for attribution and for the per-harness notes. Never used in token math.
    index: int = 0  # arrival order within the session
    model: str | None = None
    finish_reason: str | None = None
    harness_session_id: str | None = (
        None  # the harness's OWN session id, when it sends one
    )
    system_digest: str | None = (
        None  # cheap identity for the conversation this belongs to
    )
    n_tools: int = 0
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    # Retained because TRL's `_turns_from_trace` passes `tools` to `apply_chat_template`: the tool
    # manifest is part of the rendered prompt, so a re-tokenization without it does not match what
    # the engine actually saw.
    request_tools: list[dict[str, Any]] | None = None
    # Parameters actually submitted to inference, after capture preparation and compatibility
    # edits. Missing values are engine defaults, not a verified training policy. An explicit
    # session policy pins every sampling field; the original harness request stays separate.
    sampling_params: dict[str, Any] = field(default_factory=dict)
    requested_sampling_params: dict[str, Any] = field(default_factory=dict)
    response_message: dict[str, Any] = field(default_factory=dict)

    @property
    def end_ids(self) -> list[int]:
        # [FUNCTION DOC]
        # Returns prompt_ids followed by sampled_ids (new list).
        """Cumulative token sequence after this turn: its prompt plus what it sampled."""
        return self.prompt_ids + self.sampled_ids

    def context_ids(self, parent: "TurnNode | None") -> list[int]:
        # [FUNCTION DOC]
        # Returns the tokens this node adds before generation: the whole prompt for a root, or
        # the prompt suffix after the parent's end_ids for a child.
        """Tokens this node adds to the sequence BEFORE the model starts generating.

        For a root that is the whole prompt. For a child it is the interstitial span: the tool
        results, user turns and template scaffolding the harness inserted since the parent stopped
        generating. These are real tokens the model conditioned on, but it did not produce them, so
        they are context (mask 0) rather than targets.
        """
        if parent is None:
            return list(self.prompt_ids)
        return self.prompt_ids[len(parent.end_ids) :]


@dataclass
class TrainingSequence:
    # [CLASS DOC]
    # One flattened path: input_ids, loss_mask (1 at trainable sampled tokens), logprobs
    # (0.0 at context), node_ids, prompt_len (tokens before the first sample), root_id, n_turns.
    """One path through the graph, flattened. Maps onto TRL's `TrainingSequence` fields."""

    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    node_ids: list[str]
    prompt_len: int  # tokens before the first sampled token
    root_id: str
    n_turns: int

    @property
    def n_trainable(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of 1s in loss_mask.
        return sum(self.loss_mask)

    def turn_lengths(self) -> list[int]:
        # [FUNCTION DOC]
        # Returns lengths of consecutive runs of 1s in loss_mask, in order.
        # KNOWN ISSUE: turns whose logprobs were unusable (mask 0) are omitted, and two turns
        # with no context tokens between them merge into one run, so this is not always one
        # entry per turn as documented.
        """Sampled-token count per turn, in order. The join key against a harness trace."""
        lengths, run = [], 0
        for m in self.loss_mask:
            if m:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
        return lengths


class RolloutGraph:
    # [CLASS DOC]
    # All model calls of one rollout. Stores nodes by id, arrival order and child lists.
    # Linking is independent of arrival order (late-arriving ancestors adopt orphans).
    """All model calls for one rollout, linked by prefix.

    `add_turn` is the whole ingestion path: it finds the parent by token prefix and appends. Calls
    arrive in wire order, but the graph does not depend on that order being meaningful, which matters
    because harnesses issue concurrent requests (parallel subagents, background summarisation).
    """

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes empty node map, order list and children map.
        self._nodes: dict[str, TurnNode] = {}
        self._order: list[str] = []
        self._children: dict[str, list[str]] = {}

    # --- construction --------------------------------------------------
    def add_turn(self, node: TurnNode) -> TurnNode:
        # [FUNCTION DOC]
        # Ingests a node: sets its index to the arrival position, finds its parent, stores it,
        # registers it as a child of the parent, then re-parents any existing nodes for which it
        # is a closer exact-prefix predecessor.
        # Returns: the same node (mutated). Side effects: mutates graph and node.
        node.index = len(self._order)
        node.parent_id = self._find_parent(node)
        self._nodes[node.node_id] = node
        self._order.append(node.node_id)
        self._children.setdefault(node.node_id, [])
        if node.parent_id is not None:
            self._children[node.parent_id].append(node.node_id)
        self._adopt_orphaned_roots(node)
        return node

    def _adopt_orphaned_roots(self, node: TurnNode) -> None:
        # [FUNCTION DOC]
        # HELPER: For each other node whose prompt is longer than the new node's prompt, whose
        # current parent (if any) ends before the new node's end, and whose prompt starts with
        # the new node's end_ids, moves it under the new node. No-op if end_ids is empty.
        # Side effects: mutates parent ids and child lists.
        """Re-parent descendants when a closer exact-prefix predecessor arrives late.

        `_find_parent` only looks backwards, and it skips any candidate whose `end_ids` is longer than
        the new node's prompt. So a turn that arrives BEFORE its own ancestor was permanently orphaned:
        ingesting `[1,2,3,4] -> [5]` and then `[1,2] -> [3]` produced two roots and split one genuine
        two-turn trajectory in half.

        That contradicts the guarantee this class documents — arrival order is not meaningful, because
        harnesses issue concurrent requests — and the existing arrival-order test only covered the
        ancestor-first direction. Linking is symmetric now: on insert, look forwards too.

        A,C,B must produce the same chain as A,B,C: C may already be attached to A when B arrives.
        """
        end = node.end_ids
        if not end:
            return
        for candidate_id in self._order:
            if candidate_id == node.node_id:
                continue
            candidate = self._nodes[candidate_id]
            if len(node.prompt_ids) >= len(candidate.prompt_ids):
                continue
            if candidate.parent_id is not None:
                parent = self._nodes[candidate.parent_id]
                if len(parent.end_ids) >= len(end):
                    continue
            if len(end) > len(candidate.prompt_ids):
                continue
            if common_prefix_len(end, candidate.prompt_ids) == len(end):
                if candidate.parent_id is not None:
                    self._children[candidate.parent_id].remove(candidate_id)
                candidate.parent_id = node.node_id
                self._children[node.node_id].append(candidate_id)

    def _find_parent(self, node: TurnNode) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Returns the id of the existing node whose end_ids is the longest exact prefix
        # of the new node's prompt (requiring a strictly shorter candidate prompt), or None.
        # Uses _find_parent_by_messages when the node has no prompt ids.
        """The existing node whose prompt+completion is the LONGEST exact prefix of this prompt.

        Longest wins so a deep chain attaches to its immediate predecessor rather than to an early
        ancestor that also matches. Requiring an exact prefix (not a fuzzy match) is deliberate: a
        harness that mutates its history has genuinely produced a different sequence, and quietly
        attaching it would fabricate a trajectory the model never saw. Such a call becomes a new root
        instead, which `roots()` surfaces rather than hides.

        Falls back to message prefixes when there are no token ids to compare. See `_find_parent_by_
        messages`.
        """
        if not node.prompt_ids:
            return self._find_parent_by_messages(node)

        best_id, best_len = None, 0
        for candidate_id in self._order:
            candidate = self._nodes[candidate_id]
            end = candidate.end_ids
            if len(candidate.prompt_ids) >= len(node.prompt_ids):
                continue
            if len(end) > len(node.prompt_ids) or len(end) <= best_len:
                continue
            if common_prefix_len(end, node.prompt_ids) == len(end):
                best_id, best_len = candidate_id, len(end)
        return best_id

    def _find_parent_by_messages(self, node: TurnNode) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Message-based parent lookup for captures without token ids. A candidate's span
        # is its request messages plus its response (with role defaulted to assistant); the
        # longest span that, after dropping leading system messages on both sides, matches the
        # start of the node's request messages message-by-message wins. Returns its id or None.
        """Same longest-exact-prefix rule, keyed on the message list instead of token ids.

        Only reachable on an eval endpoint, where the upstream returns no token ids at all. Without
        this the token path degenerates silently rather than wrongly: every `end_ids` is empty, so
        `len(end) <= best_len` is true for every candidate, no parent is ever found, and a 20-turn
        conversation is reported as 20 separate roots. The rollout is not wrong, but it reads as if
        the agent restarted on every turn, and `check_rollout`'s root-count heuristics all misfire.

        Messages are a weaker key than tokens — they are what the harness *said* it sent rather than
        what the engine tokenised — which is exactly why they are not used when ids are available.
        For a trace they are sufficient: the question is only which conversation a call continues.
        """
        best_id, best_len = None, 0
        for candidate_id in self._order:
            candidate = self._nodes[candidate_id]
            # The parent's own turn is its request plus the message it produced, and that whole
            # thing has to be a prefix of ours — the same "prompt + completion" span `end_ids` is.
            #
            # The role is defaulted rather than required: a response message is by definition the
            # assistant's, but not every producer spells it out (an engine may omit it, and the SSE
            # replay path reassembles the message itself), while the harness always names the role
            # when it echoes the turn back. Without the default that asymmetry alone breaks every
            # link and the whole conversation reads as one root per turn.
            reply = candidate.response_message
            end = [
                *candidate.request_messages,
                *([{"role": "assistant", **reply}] if reply else []),
            ]
            if not end or len(end) > len(node.request_messages) or len(end) <= best_len:
                continue
            # Compared with leading system messages dropped from BOTH sides: a harness that
            # re-renders its system prompt every step would otherwise fail at index 0 and orphan every
            # turn. See `_without_leading_system`.
            end_cmp = _without_leading_system(end)
            node_cmp = _without_leading_system(node.request_messages)
            if not end_cmp or len(end_cmp) > len(node_cmp):
                continue
            if all(_same_message(a, b) for a, b in zip(end_cmp, node_cmp)):
                best_id, best_len = candidate_id, len(end)
        return best_id

    # --- structure -----------------------------------------------------
    def nodes(self) -> list[TurnNode]:
        # [FUNCTION DOC]
        # Returns all nodes in arrival order.
        return [self._nodes[i] for i in self._order]

    def get(self, node_id: str) -> TurnNode:
        # [FUNCTION DOC]
        # Returns the node with the given id. Raises KeyError if absent.
        return self._nodes[node_id]

    def children(self, node_id: str) -> list[TurnNode]:
        # [FUNCTION DOC]
        # Returns the child nodes of node_id ([] if none or unknown).
        return [self._nodes[i] for i in self._children.get(node_id, [])]

    def roots(self) -> list[TurnNode]:
        # [FUNCTION DOC]
        # Returns nodes without a parent, in arrival order.
        return [self._nodes[i] for i in self._order if self._nodes[i].parent_id is None]

    def leaves(self) -> list[TurnNode]:
        # [FUNCTION DOC]
        # Returns nodes with no children, in arrival order.
        return [self._nodes[i] for i in self._order if not self._children.get(i)]

    def root_of(self, node_id: str) -> str:
        # [FUNCTION DOC]
        # Returns the id of the root reached by following parent links from node_id.
        """Which conversation this node belongs to. Walks parents to the top."""
        node = self._nodes[node_id]
        while node.parent_id is not None:
            node = self._nodes[node.parent_id]
        return node.node_id

    def path_to(self, leaf_id: str) -> list[TurnNode]:
        # [FUNCTION DOC]
        # Returns the list of nodes from the root down to leaf_id.
        path: list[TurnNode] = []
        node: TurnNode | None = self._nodes[leaf_id]
        while node is not None:
            path.append(node)
            node = self._nodes[node.parent_id] if node.parent_id else None
        return list(reversed(path))

    def paths(self) -> Iterator[list[TurnNode]]:
        # [FUNCTION DOC]
        # Yields the root-to-leaf node path for every leaf.
        """Every root-to-leaf path. One per distinct trajectory, including discarded branches."""
        for leaf in self.leaves():
            yield self.path_to(leaf.node_id)

    def forks(self) -> list[tuple[str, list[str]]]:
        # [FUNCTION DOC]
        # Returns (parent_id, child_ids) for every node with more than one child.
        """Nodes with more than one child: retries, resamples, or parallel branches."""
        return [(pid, kids) for pid, kids in self._children.items() if len(kids) > 1]

    def discarded_nodes(self) -> list[TurnNode]:
        # [FUNCTION DOC]
        # Returns childless nodes judged discarded: within each sibling group (children of a
        # fork, or roots sharing an identical prompt), if at least one sibling has children, the
        # childless siblings are discarded. Groups where all are terminal are kept.
        """Sampled turns that led nowhere.

        A sibling with no children whose parent has another child that DID continue was generated and
        thrown away: a retry after a parse failure, or a resample. Training it with the rollout's
        reward credits work that never happened. Detected purely from shape, with no harness
        cooperation, which matters because a proxy is otherwise blind to retries.

        The final turn of a real trajectory is also childless, so a sibling is only called discarded
        when at least one of its siblings continued.

        Siblings come in two shapes, and only one of them is a fork. Retrying a MID-conversation call
        gives the attempts a shared parent, so `forks()` finds them. Retrying the FIRST call — resampled
        before any tool result exists, so both attempts carry the identical prompt — gives two roots
        with `parent_id=None`, which is not a fork at all: the abandoned attempt was exported as a full
        agent sequence and trained with the rollout's reward. Roots are therefore grouped by their exact
        prompt, which is precise rather than heuristic — a subagent or an auxiliary call starts from a
        different prompt and never groups with the real first turn.
        """
        discarded: list[TurnNode] = []
        sibling_groups: list[list[str]] = [kids for _, kids in self.forks()]

        by_prompt: dict[tuple[int, ...], list[str]] = {}
        for node in self.roots():
            by_prompt.setdefault(tuple(node.prompt_ids), []).append(node.node_id)
        sibling_groups.extend(group for group in by_prompt.values() if len(group) > 1)

        for kids in sibling_groups:
            continued = [k for k in kids if self._children.get(k)]
            if not continued:
                continue  # all siblings are terminal: ambiguous, keep them all
            discarded.extend(self._nodes[k] for k in kids if not self._children.get(k))
        return discarded

    # --- flattening ----------------------------------------------------
    def sequence_for(self, leaf_id: str) -> TrainingSequence:
        # [FUNCTION DOC]
        # Flattens the path to leaf_id into a TrainingSequence: for each node, appends its
        # context ids (mask 0, logprob 0.0), then its sampled ids with mask 1 and its logprobs
        # when logprobs exist and match the sampled length, otherwise mask 0 and zero logprobs.
        # prompt_len is the root's context length.
        # Raises: KeyError for unknown ids.
        """Flatten one root-to-leaf path into token ids, mask and logprobs.

        Invariant enforced here rather than trusted: a turn whose logprobs are missing or misaligned
        contributes its tokens as CONTEXT (mask 0), never as targets. A trainable token without a
        real behaviour-policy logprob would make GRPO's importance ratio `exp(new - old)` a ratio
        against a number we invented.
        """
        path = self.path_to(leaf_id)
        input_ids: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float] = []
        prompt_len = 0

        parent: TurnNode | None = None
        for position, node in enumerate(path):
            context = node.context_ids(parent)
            input_ids.extend(context)
            loss_mask.extend([0] * len(context))
            logprobs.extend([0.0] * len(context))
            if position == 0:
                prompt_len = len(context)

            usable = node.sampled_logprobs is not None and len(
                node.sampled_logprobs
            ) == len(node.sampled_ids)
            input_ids.extend(node.sampled_ids)
            loss_mask.extend([1 if usable else 0] * len(node.sampled_ids))
            logprobs.extend(
                node.sampled_logprobs if usable else [0.0] * len(node.sampled_ids)
            )
            parent = node

        return TrainingSequence(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            node_ids=[n.node_id for n in path],
            prompt_len=prompt_len,
            root_id=path[0].node_id,
            n_turns=len(path),
        )

    def sequences(self) -> list[TrainingSequence]:
        # [FUNCTION DOC]
        # Returns one TrainingSequence per leaf.
        return [self.sequence_for(leaf.node_id) for leaf in self.leaves()]

    def stats(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns counts: n_turns, n_roots, n_leaves, n_forks, n_discarded.
        return {
            "n_turns": len(self._order),
            "n_roots": len(self.roots()),
            "n_leaves": len(self.leaves()),
            "n_forks": len(self.forks()),
            "n_discarded": len(self.discarded_nodes()),
        }
