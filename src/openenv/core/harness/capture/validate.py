# [MODULE DOC]
# Graded validation (FATAL / WARN / INFO findings) for captured data at every stage:
# check_turn per model call (token ids and logprob alignment), check_turn_eval for
# endpoints without token ids, validate_training_turn for trainer-facing turns,
# check_sequence for flattened paths, check_rollout for graph structure, and
# check_upstream_response for an endpoint's capture capability. Pure functions.
"""Token-in / token-out validation. Nothing leaves this system unchecked.

Capture failures in this stack are silent by construction. Every bug found so far returned a
perfectly well-formed payload and reported success: a missing `--return-tokens-as-token-ids` yields
text with no ids and trains on nothing; a harness that rewrites its history yields chains that stitch
into a trajectory the model never saw; an SSE client handed a JSON body yields zero tokens and no
error anywhere. None of these raise. All of them produce plausible JSON.

So validation is not a debug aid here, it is the only thing standing between a clean-looking run and
weeks of training on corrupted sequences. Checks are graded:

    FATAL    the row is not trainable. Drop it. Training on it is worse than dropping it.
    WARN     the row is trainable but something is off and should be understood.
    INFO     recorded for the per-harness notes.

`check_upstream` runs before a rollout is spent (endpoint capability), `check_turn` runs per model
call (token/logprob alignment), `check_sequence` runs on the flattened output (mask/logprob
invariants), and `check_rollout` runs on the whole graph (structure).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

FATAL, WARN, INFO = "FATAL", "WARN", "INFO"


@dataclass
class Finding:
    # [CLASS DOC]
    # One finding: level (FATAL/WARN/INFO), code and detail; str() is "[LEVEL] code: detail".
    level: str
    code: str
    detail: str

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Returns "[<level>] <code>: <detail>".
        return f"[{self.level}] {self.code}: {self.detail}"


@dataclass
class Report:
    # [CLASS DOC]
    # Collection of findings plus aux_node_ids (nodes excluded as auxiliary by a trace
    # reconciler). ok is True when there are no FATAL findings.
    findings: list[Finding] = field(default_factory=list)
    # Nodes the harness's own trace does not count as agent steps (auxiliary calls). Populated only
    # by a harness-trace reconciler. Captured correctly, but excluded from training so they cannot be credited
    # with the rollout's reward. Empty for every harness whose counts agree exactly.
    aux_node_ids: list[str] = field(default_factory=list)

    def add(self, level: str, code: str, detail: str) -> None:
        # [FUNCTION DOC]
        # Appends a Finding with the given level, code and detail.
        self.findings.append(Finding(level, code, detail))

    @property
    def fatal(self) -> list[Finding]:
        # [FUNCTION DOC]
        # Returns the FATAL findings.
        return [f for f in self.findings if f.level == FATAL]

    @property
    def ok(self) -> bool:
        # [FUNCTION DOC]
        # True if there are no FATAL findings.
        return not self.fatal

    def merge(self, other: "Report") -> "Report":
        # [FUNCTION DOC]
        # Appends other's findings (not aux_node_ids) to this report and returns self.
        self.findings.extend(other.findings)
        return self

    def text(self) -> str:
        # [FUNCTION DOC]
        # Returns "all checks passed" or newline-joined finding strings.
        if not self.findings:
            return "all checks passed"
        return "\n".join(str(f) for f in self.findings)


# --- per model call ----------------------------------------------------
def check_turn(
    prompt_ids, sampled_ids, logprobs, *, finish_reason=None, index=0
) -> Report:
    # [FUNCTION DOC]
    # Validates one captured call. FATAL: any prompt/sampled id not a non-negative int
    # (invalid_token_id); logprobs non-finite or non-numeric (invalid_logprob) or positive
    # (positive_logprob); empty prompt ids (no_prompt_ids); sampled ids without logprobs
    # (no_logprobs); logprob count differs from sampled count (logprob_misalign). WARN: no
    # sampled ids (no_sampled_ids); finish_reason "length" (truncated_turn).
    # Returns: Report. index is used only in messages.
    """Validate one captured call before it becomes a graph node."""
    report = Report()
    tag = f"turn {index}"

    for name, ids in (("prompt", prompt_ids), ("sampled", sampled_ids)):
        if any(type(token) is not int or token < 0 for token in ids):
            report.add(
                FATAL,
                "invalid_token_id",
                f"{tag}: {name} ids must be non-negative integers",
            )

    if logprobs is not None:
        if any(
            isinstance(lp, bool)
            or not isinstance(lp, (int, float))
            or not math.isfinite(lp)
            for lp in logprobs
        ):
            report.add(
                FATAL, "invalid_logprob", f"{tag}: logprobs must be finite numbers"
            )
        elif any(lp > 0 for lp in logprobs):
            report.add(
                FATAL, "positive_logprob", f"{tag}: logprobs must not be positive"
            )

    if not prompt_ids:
        report.add(
            FATAL,
            "no_prompt_ids",
            f"{tag}: server returned no prompt_token_ids. The endpoint "
            "is missing --return-tokens-as-token-ids, or the engine does not support it.",
        )
    if not sampled_ids:
        # Legitimate when the model is cut off at zero tokens, but nothing is trainable either way.
        report.add(
            WARN,
            "no_sampled_ids",
            f"{tag}: no completion token ids (finish={finish_reason})",
        )

    if logprobs is None:
        if sampled_ids:
            report.add(
                FATAL,
                "no_logprobs",
                f"{tag}: {len(sampled_ids)} sampled tokens with no "
                "logprobs. GRPO's importance ratio needs the behaviour-policy logprob for "
                "every trainable token; without them the turn can only be context.",
            )
    if finish_reason == "length":
        # WARN rather than FATAL: the tokens and their logprobs are genuine, so the turn is real
        # training data. What is not real is its ENDING — the model was cut off by the output cap
        # mid-thought, and nothing downstream distinguished that from a turn that chose to stop. A
        # trajectory made of truncated turns teaches the policy to stop early.
        report.add(
            WARN,
            "truncated_turn",
            f"{tag}: stopped on the output-token cap ({len(sampled_ids)} tokens), so this turn was "
            "cut off rather than finished. Its tokens are valid; its ending is an artefact of the cap.",
        )

    if logprobs is not None and len(logprobs) != len(sampled_ids):
        report.add(
            FATAL,
            "logprob_misalign",
            f"{tag}: {len(logprobs)} logprobs vs {len(sampled_ids)} sampled ids. An off-by-one "
            "here shifts credit onto the wrong tokens and still trains.",
        )
    return report


def validate_training_turn(prompt_ids, sampled_ids, logprobs, loss_mask) -> None:
    # [FUNCTION DOC]
    # Validates a trainer-facing turn: runs check_turn and additionally requires loss_mask
    # length == prompt + completion length, values only int 0/1, and all prompt positions 0.
    # Raises: ValueError listing the FATAL findings if any. Returns None.
    """Reject a malformed trainer-facing turn, preserving partial completion masks.

    The mask covers prompt plus completion. Masked completion tokens may retain their real
    logprobs; the mask controls supervision, while the logprobs retain capture provenance.
    """
    report = check_turn(prompt_ids, sampled_ids, logprobs)
    if len(loss_mask) != len(prompt_ids) + len(sampled_ids):
        report.add(
            FATAL, "length_mismatch", "loss_mask must cover prompt plus completion"
        )
    elif any(type(m) is not int or m not in (0, 1) for m in loss_mask):
        report.add(
            FATAL, "invalid_loss_mask", "loss_mask must contain only integer 0 or 1"
        )
    elif any(loss_mask[: len(prompt_ids)]):
        report.add(FATAL, "trainable_prompt", "prompt tokens must remain context")
    if not report.ok:
        raise ValueError(
            "invalid training turn: " + "; ".join(str(f) for f in report.fatal)
        )


def check_turn_eval(response_message, *, finish_reason=None, index=0) -> Report:
    # [FUNCTION DOC]
    # Validates one call on a non-token endpoint: WARN empty_completion if the message has
    # neither non-blank content nor tool calls; INFO truncated_turn if finish_reason is
    # "length". Returns: Report.
    """Validate one captured call on an endpoint that cannot return token ids.

    `check_turn` is the wrong instrument here: every one of its FATALs (`no_prompt_ids`,
    `no_logprobs`) is the *expected* condition for a hosted provider, so running it would fill a
    perfectly good eval rollout with fatal findings and teach everyone to ignore the findings list.

    What is still worth asserting is that the model actually said something. An empty completion with
    no tool call is a turn the agent cannot act on, and it is the shape a content filter or a
    truncated stream leaves behind.
    """
    report = Report()
    tag = f"turn {index}"
    message = response_message if isinstance(response_message, dict) else {}
    text = message.get("content") or ""
    has_output = bool(str(text).strip()) or bool(message.get("tool_calls"))

    if not has_output:
        report.add(
            WARN,
            "empty_completion",
            f"{tag}: no text and no tool call (finish={finish_reason}). The agent has nothing "
            "to act on; check for a content filter or a length cap.",
        )
    if finish_reason == "length":
        report.add(
            INFO,
            "truncated_turn",
            f"{tag}: stopped on the output-token cap, so the turn is cut mid-thought",
        )
    return report


# --- per flattened sequence -------------------------------------------
def check_sequence(seq, *, min_trainable: int = 1) -> Report:
    # [FUNCTION DOC]
    # Validates a flattened TrainingSequence. FATAL: length mismatch between input_ids,
    # loss_mask and logprobs (returns early); validate_training_turn failure treating the
    # first prompt_len tokens as prompt and the rest as completion (returns early); fewer
    # than min_trainable (default 1) trainable tokens; masked positions with non-zero
    # logprob; prompt_len covering the whole sequence; any positive logprob. INFO: count of
    # trainable tokens with logprob exactly 0.0. Returns: Report.
    """Validate a flattened path. These are the invariants the trainer assumes and never re-checks."""
    report = Report()
    n = len(seq.input_ids)

    if len(seq.loss_mask) != n or len(seq.logprobs) != n:
        report.add(
            FATAL,
            "length_mismatch",
            f"input_ids={n} loss_mask={len(seq.loss_mask)} logprobs={len(seq.logprobs)}",
        )
        return report  # every later check would be meaningless

    try:
        validate_training_turn(
            seq.input_ids[: seq.prompt_len],
            seq.input_ids[seq.prompt_len :],
            seq.logprobs[seq.prompt_len :],
            seq.loss_mask,
        )
    except ValueError as exc:
        report.add(FATAL, "invalid_training_sequence", str(exc))
        return report

    trainable = sum(seq.loss_mask)
    if trainable < min_trainable:
        report.add(
            FATAL,
            "nothing_trainable",
            f"{trainable} trainable tokens in a {n}-token sequence: this row contributes no "
            "gradient and only shrinks the effective group",
        )

    # A masked position carrying a logprob means context was scored; a trainable position without one
    # means a target was invented. Both are silent corruption, in opposite directions.
    masked_with_lp = sum(
        1 for m, lp in zip(seq.loss_mask, seq.logprobs) if m == 0 and lp != 0.0
    )
    if masked_with_lp:
        report.add(
            FATAL,
            "masked_has_logprob",
            f"{masked_with_lp} context positions carry a non-zero logprob",
        )

    if seq.prompt_len >= n:
        report.add(
            FATAL,
            "empty_response",
            f"prompt_len={seq.prompt_len} covers the whole sequence",
        )

    # Not an error: a token with logprob exactly 0.0 has probability 1.0, which is common for
    # structural tokens in a constrained tool-call grammar (`</tool_call>`, closing brackets).
    # Recorded so a *sudden* change in the rate is visible.
    zero_lp_trainable = sum(
        1 for m, lp in zip(seq.loss_mask, seq.logprobs) if m == 1 and lp == 0.0
    )
    if zero_lp_trainable:
        report.add(
            INFO,
            "certain_tokens",
            f"{zero_lp_trainable}/{trainable} trainable tokens have logprob 0.0 (p=1.0)",
        )

    positive = [lp for lp in seq.logprobs if lp > 0.0]
    if positive:
        report.add(
            FATAL,
            "positive_logprob",
            f"{len(positive)} logprobs > 0 (max {max(positive):.4f}); log-probabilities cannot "
            "be positive, so these are not logprobs",
        )
    return report


# --- per rollout -------------------------------------------------------
def check_rollout(
    graph,
    *,
    expect_single_root: bool = False,
    capture_level: str = "tokens",
    budget_stop_count: int = 0,
) -> Report:
    # [FUNCTION DOC]
    # Validates graph structure. FATAL no_turns if the graph is empty (returns early). If every
    # turn is its own root (more than one turn): WARN per_turn_capture_only at tokens level,
    # else INFO per_turn_trace_only. Else multiple roots -> WARN multiple_roots; else if
    # expect_single_root and root count != 1 -> FATAL root_count. WARN degenerate_rollout for
    # exactly one turn when budget_stop_count <= 0; WARN discarded_turns when any discarded;
    # INFO forks when any forks. Returns: Report.
    # KNOWN ISSUE: the root_count branch is effectively unreachable, since it is only
    # evaluated when there is at most one root and a non-empty graph always has one.
    """Validate graph structure. This is where harness-specific weirdness shows up first.

    `capture_level` below `tokens` changes what a root count MEANS. On the token path, one root per
    turn diagnoses a harness that re-renders its prompt instead of appending. On an eval endpoint
    there are no token ids to share a prefix with in the first place, so the same shape says nothing
    about the harness — only that message-prefix linking could not match either, which is a property
    of what the harness sends rather than a degradation of capture.
    """
    report = Report()
    stats = graph.stats()

    if stats["n_turns"] == 0:
        report.add(
            FATAL,
            "no_turns",
            "the intercept saw no model calls: the agent never reached it "
            "(wrong base URL, unresolved model, or auth rejected)",
        )
        return report

    if stats["n_roots"] == stats["n_turns"] and stats["n_turns"] > 1:
        # One root PER TURN: the harness re-renders its prompt each turn instead of appending, so no
        # token prefix is shared. terminus-2 does this (its message list grows 1,3,5..15 while the
        # rendered tokens never line up).
        #
        # WARN, not FATAL. Each turn is still an exact prompt with exact sampled tokens and real
        # logprobs, which is perfectly good SINGLE-turn training data. What is lost is cross-turn
        # structure: each row repeats its own context. Rollout-level rewards still supervise
        # these turns correctly, but repeated context increases compute and packing cost.
        if capture_level == "tokens":
            report.add(
                WARN,
                "per_turn_capture_only",
                f"every turn is its own root ({stats['n_turns']}). This harness re-renders its "
                "prompt rather than appending, so rows are single-turn. Tokens and logprobs are "
                "exact; rollout rewards can still supervise every retained turn. Repeated context "
                "increases training cost.",
            )
        else:
            report.add(
                INFO,
                "per_turn_trace_only",
                f"every turn is its own root ({stats['n_turns']}). With no token ids, turns are "
                "linked by message prefix, and this harness's messages do not extend each other "
                "exactly — so the trace is per-call rather than one thread. Nothing is lost that "
                "an eval rollout carries.",
            )
    elif stats["n_roots"] > 1:
        # Normal: aux calls (title generation), subagents, or a harness that rewrote its system
        # prompt partway (claude-code) and so continued under a second prefix family.
        report.add(
            WARN,
            "multiple_roots",
            f"{stats['n_roots']} roots across {stats['n_turns']} turns. Each root is a separate "
            "conversation (subagent, aux call, or a rewritten prompt that broke the chain).",
        )
    elif expect_single_root and stats["n_roots"] != 1:
        report.add(FATAL, "root_count", f"expected 1 root, got {stats['n_roots']}")

    if stats["n_turns"] == 1 and budget_stop_count <= 0:
        # A single call can be a valid answer or a harness that stopped early.
        # Keep the diagnostic, but let the verifier score the task outcome.
        report.add(
            WARN,
            "degenerate_rollout",
            "exactly 1 model call for the whole task; inspect the agent log if this "
            "was unexpected. Valid captured tokens remain eligible for training.",
        )

    if stats["n_discarded"]:
        report.add(
            WARN,
            "discarded_turns",
            f"{stats['n_discarded']} sampled turn(s) led nowhere (retries or resamples). They "
            "are excluded from training paths; the tokens were still generated and billed.",
        )
    if stats["n_forks"]:
        report.add(INFO, "forks", f"{stats['n_forks']} fork point(s) in the graph")
    return report


# --- endpoint capability, before spending a sandbox --------------------
def check_upstream_response(payload: dict[str, Any]) -> Report:
    # [FUNCTION DOC]
    # Checks a raw chat-completions reply for capture support. FATAL: no choices (returns
    # early); no top-level prompt_token_ids; no choices[0].token_ids; no logprobs content; or
    # logprob count != token id count. WARN token_strings when the first logprob entry's token
    # is not in "token_id:N" form (capture flags likely missing). Returns: Report.
    """Assert a raw chat-completions reply actually carries what capture needs.

    Run this against the endpoint before booting anything. The failure it catches (an endpoint served
    without the capture flags) otherwise surfaces only as empty training rows, hours later.
    """
    report = Report()
    choices = payload.get("choices") or []
    if not choices:
        report.add(FATAL, "no_choices", "response has no choices")
        return report
    choice = choices[0]

    if not payload.get("prompt_token_ids"):
        report.add(
            FATAL,
            "no_prompt_token_ids",
            "no top-level prompt_token_ids. Serve with --return-tokens-as-token-ids; without "
            "it multi-turn stitching is impossible because turn k+1's prompt is unknown.",
        )
    if not choice.get("token_ids"):
        report.add(
            FATAL,
            "no_completion_token_ids",
            "choices[0].token_ids missing. Send return_token_ids=True and serve with "
            "--return-tokens-as-token-ids.",
        )

    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        report.add(
            FATAL,
            "no_logprobs",
            "choices[0].logprobs.content missing. Send logprobs=True.",
        )
    else:
        if choice.get("token_ids") and len(content) != len(choice["token_ids"]):
            report.add(
                FATAL,
                "logprob_misalign",
                f"{len(content)} logprobs vs {len(choice['token_ids'])} token ids",
            )
        token = (content[0] or {}).get("token", "")
        if not str(token).startswith("token_id:"):
            # This says more than it looks like it says, and the mild reading of it is what makes the
            # failure silent.
            #
            # `token_id:N` in this field is what --return-tokens-as-token-ids produces. Its absence
            # means that flag was not passed, and its partner --logprobs-mode processed_logprobs
            # almost certainly was not either, since the docs present them as a pair. That second
            # flag is the one that matters here: without it vLLM returns RAW (pre-temperature)
            # logprobs instead of the sampled distribution's.
            #
            # Measured on one vLLM 0.25.1, same prompt and token at temperature 0.7:
            #     with both flags  -1.3292      without  -1.2546
            # Both are plausible, both align to the sampled ids, and `token_ids` arrives either way
            # because it comes from the REQUEST parameter rather than from either flag. So every
            # other check in this file passes and the rollout grades as fully trainable while
            # carrying a wrong importance ratio.
            report.add(
                WARN,
                "token_strings",
                f"logprob tokens are strings ({token!r}) not 'token_id:N', so "
                "--return-tokens-as-token-ids was not passed. Its partner --logprobs-mode "
                "processed_logprobs is then probably absent too, which means these logprobs are "
                "RAW (pre-temperature) rather than the sampling distribution's. Token ids arrive "
                "regardless (they come from the request parameter), so nothing else here can catch "
                "it: the rollout will look perfectly trainable and train on a wrong importance "
                "ratio. Restart the engine with both flags.",
            )
    return report
