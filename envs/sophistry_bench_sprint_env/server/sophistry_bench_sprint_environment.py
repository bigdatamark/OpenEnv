# [MODULE DOC]
# Server-side Sophistry-Bench Sprint environment. Builds an advocacy dataset from the bundled QuALITY
# split using the external sophistry_bench_sprint package, serves one row per reset, and scores the
# single submitted argument with that package's canonical reward functions (aggregate reward, claim and
# citation counts, four canary detectors, plus hidden correctness). Final reward is a weighted sum over
# eight components in fixed key order; weights come from the constructor or the SPRINT_WEIGHTS env var.
# Module globals: component key order, default weights (aggregate only), and a lazily cached canonical
# aggregate function.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from sophistry_bench_sprint import (
    alternation_canary,
    length_band_canary,
    load_environment,
    load_quality_from_json,
    packaged_quality_path,
    parse_citations,
    parse_claims,
    quality_to_advocacy_dataset,
    starts_with_canary,
    template_echo_canary,
)

try:
    from sophistry_bench_sprint import aggregate_reward as _direct_aggregate_reward
except ImportError:
    _direct_aggregate_reward = None

try:
    from ..models import AdvocacyAction, AdvocacyObservation
except ImportError:  # when imported as top-level package
    from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation


# Canonical reward-component order. The i-th SPRINT_WEIGHTS entry weights the
# i-th key here. MUST stay aligned with sophistry_bench_sprint._build_reward_funcs()
# and with the SPRINT_WEIGHTS docs in the README.
_COMPONENT_KEYS = (
    "aggregate_reward",
    "correctness_reward",
    "n_claims",
    "n_citations",
    "alternation_canary",
    "starts_with_canary",
    "length_band_canary",
    "template_echo_canary",
)

_DEFAULT_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_CANONICAL_AGGREGATE_FN = None


def _canonical_aggregate_fn():
    # [FUNCTION DOC]
    # HELPER: lazily loads the canonical sprint environment once and returns its aggregate reward function
    # (the first function of the rubric, descending into the first sub-rubric if the top-level rubric has no
    # funcs). Caches it in a module global.
    # Raises RuntimeError if that function is not named "aggregate_reward".
    # Side effects: calls load_environment() on first use; mutates the module-level cache.
    global _CANONICAL_AGGREGATE_FN
    if _CANONICAL_AGGREGATE_FN is None:
        env = load_environment()
        rubric = env.rubric
        if not getattr(rubric, "funcs", None) and getattr(rubric, "rubrics", None):
            rubric = rubric.rubrics[0]
        aggregate_fn = rubric.funcs[0]
        if aggregate_fn.__name__ != "aggregate_reward":
            raise RuntimeError(
                "expected sophistry-bench-sprint rubric.funcs[0] to be "
                f"aggregate_reward, got {aggregate_fn.__name__}"
            )
        _CANONICAL_AGGREGATE_FN = aggregate_fn
    return _CANONICAL_AGGREGATE_FN


def _aggregate_reward(
    text: str, claims: list[str], cites: list[str], passage: str
) -> float:
    # [FUNCTION DOC]
    # HELPER: computes the aggregate reward as a float. Uses the package's directly exported
    # aggregate_reward(claims, cites, passage) when available; otherwise calls the canonical rubric function
    # with a single assistant message containing text and a state holding the passage, run via asyncio.run.
    # KNOWN ISSUE: the asyncio.run fallback raises RuntimeError if called from a thread that already has a
    # running event loop.
    if _direct_aggregate_reward is not None:
        return float(_direct_aggregate_reward(claims, cites, passage))

    completion = [{"role": "assistant", "content": text}]
    state = {"info": {"passage": passage}}
    return float(
        asyncio.run(
            _canonical_aggregate_fn()(
                prompt=[],
                completion=completion,
                answer="",
                state=state,
            )
        )
    )


def _int_env(name: str, default: str) -> int:
    # [FUNCTION DOC]
    # HELPER: reads env var name (default string given) and returns it as int. Raises ValueError with the
    # var name if it is not an integer.
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _weights_from_env() -> list[float]:
    # [FUNCTION DOC]
    # HELPER: parses SPRINT_WEIGHTS as comma-separated floats (blank entries ignored). Returns a copy of the
    # default weights when unset/empty. Raises ValueError if any value is not a float or the count is not 8.
    raw = os.getenv("SPRINT_WEIGHTS")
    if not raw:
        return list(_DEFAULT_WEIGHTS)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    try:
        weights = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"SPRINT_WEIGHTS must be comma-separated floats, got {raw!r}"
        ) from exc
    if len(weights) != 8:
        raise ValueError(f"SPRINT_WEIGHTS must have 8 values, got {len(weights)}")
    return weights


class SophistryBenchSprintEnvironment(
    Environment[AdvocacyAction, AdvocacyObservation, State]
):
    # [CLASS DOC]
    # Single-step OpenEnv environment. Holds the built dataset (2 rows per source article: defend gold and
    # defend distractor), a round-robin cursor, the current passage and gold flag, a has-task flag and State.
    # Not concurrency-safe per instance (relies on one instance per session).
    """Single-step QuALITY advocacy environment (OpenEnv port).

    The dataset is built once at construction from the bundled QuALITY split,
    reusing the canonical sprint dataset builder. ``reset`` selects one advocacy
    row deterministically; ``step`` scores the completion with the canonical
    reward functions.
    """

    def __init__(
        self,
        n_items: Optional[int] = None,
        passage_chars: Optional[int] = None,
        seed: Optional[int] = None,
        weights: Optional[list[float]] = None,
        expose_correctness: Optional[bool] = None,
    ):
        # [FUNCTION DOC]
        # Inputs (each falls back to an env var when None): n_items (SPRINT_N_ITEMS, default 50 source
        # articles), passage_chars (SPRINT_PASSAGE_CHARS, default 2000), seed (SPRINT_SEED, default 0, used for
        # dataset building), weights (SPRINT_WEIGHTS), expose_correctness (SPRINT_EXPOSE_CORRECTNESS in 1/true/yes).
        # Raises ValueError if weights length is not 8, RuntimeError if the built dataset is empty.
        # Side effects: reads the packaged QuALITY JSON from disk and builds the dataset.
        super().__init__()
        # ``correctness_reward`` is the hidden ground truth (is the assigned answer
        # gold?). It is withheld from the wire observation by default so a naive
        # harness that forwards the whole observation to the policy can't leak it.
        # Trusted measurement code can opt in (SPRINT_EXPOSE_CORRECTNESS=1) to get
        # it back in ``metadata``/``components``. This flag controls only surfacing,
        # not weighting: correctness affects ``reward`` only via its SPRINT_WEIGHTS
        # entry, which is 0 by default (so it does not change the default reward).
        self.expose_correctness = (
            expose_correctness
            if expose_correctness is not None
            else os.getenv("SPRINT_EXPOSE_CORRECTNESS", "0").lower()
            in ("1", "true", "yes")
        )
        self.n_items = (
            n_items if n_items is not None else _int_env("SPRINT_N_ITEMS", "50")
        )
        self.passage_chars = (
            passage_chars
            if passage_chars is not None
            else _int_env("SPRINT_PASSAGE_CHARS", "2000")
        )
        self.build_seed = seed if seed is not None else _int_env("SPRINT_SEED", "0")
        self.weights = weights if weights is not None else _weights_from_env()
        # Guard the constructor ``weights=`` path too (the env-var path is already
        # length-checked). A wrong-length vector would otherwise be silently
        # truncated by the ``zip`` in ``step``, scrambling the reward with no error.
        if len(self.weights) != len(_COMPONENT_KEYS):
            raise ValueError(
                f"weights must have {len(_COMPONENT_KEYS)} values, "
                f"got {len(self.weights)}"
            )

        items = load_quality_from_json(packaged_quality_path())
        if len(items) > self.n_items:
            items = items[: self.n_items]
        # HuggingFace Dataset of rows: {prompt, answer, info{passage,assigned_answer,is_gold,article_id}}
        # n_items limits SOURCE QuALITY articles; the builder emits 2 advocacy
        # rows per article (defend-gold + defend-distractor), so len(dataset) == 2 * n_items.
        self.dataset = quality_to_advocacy_dataset(
            items, seed=self.build_seed, passage_chars=self.passage_chars
        )
        self._n = len(self.dataset)
        if self._n == 0:
            raise RuntimeError(
                "sprint dataset is empty; check bundled quality_dev.json"
            )

        self._cursor = 0
        # Per-episode ground truth. The base Environment defaults to
        # SUPPORTS_CONCURRENT_SESSIONS = False, so the server gives each session
        # its own instance; storing the current episode on self is therefore safe.
        # Do NOT enable concurrent sessions without making this per-session.
        self._current_passage: str = ""
        self._current_is_gold: bool = False
        self._has_task = False
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    @staticmethod
    def _system_text(row_prompt: list[dict]) -> str:
        # [FUNCTION DOC]
        # HELPER: returns the content of the first system-role message in a chat prompt list, else the first
        # message's content, else empty string.
        for msg in row_prompt:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return row_prompt[0].get("content", "") if row_prompt else ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AdvocacyObservation:
        # [FUNCTION DOC]
        # Picks a row: seed modulo dataset size if seed given, else the cursor (which then advances, wrapping).
        # Stores the row's passage and gold flag, marks a task as active, and creates a new State with the given
        # episode_id or a fresh UUID. Extra kwargs are ignored.
        # Returns: AdvocacyObservation with the system prompt text, assigned answer, article id, done False,
        # reward left as None.
        if seed is not None:
            idx = seed % self._n
        else:
            idx = self._cursor
            self._cursor = (self._cursor + 1) % self._n

        row = self.dataset[idx]
        info = row["info"]
        self._current_passage = info["passage"]
        self._current_is_gold = bool(info["is_gold"])
        self._has_task = True

        self._state = State(episode_id=episode_id or str(uuid.uuid4()), step_count=0)

        # reward left as the base default (None): no action scored on reset.
        return AdvocacyObservation(
            prompt=self._system_text(row["prompt"]),
            answer_to_defend=info["assigned_answer"],
            item_id=info["article_id"],
            done=False,
        )

    def step(self, action: AdvocacyAction, **kwargs: Any) -> AdvocacyObservation:
        # [FUNCTION DOC]
        # If no task is active, returns a terminal observation with reward 0.0 and an error message
        # ("call reset() before step()") in both error and metadata, without counting a step.
        # Otherwise increments step_count, parses claims and citations from action.text, computes the eight
        # component scores (aggregate, correctness 1/0 from the gold flag, claim/citation counts, four canaries)
        # and the weighted reward. Clears the active-task flag (one step per episode).
        # Returns: terminal AdvocacyObservation with reward, metadata and components set to the scores, with
        # correctness_reward removed unless expose_correctness is on.
        if not self._has_task:
            msg = "call reset() before step()"
            return AdvocacyObservation(
                prompt="",
                reward=0.0,
                done=True,
                error=msg,
                metadata={"error": msg},
            )

        # Count only scored steps (after the reset guard).
        self._state.step_count += 1

        text = action.text or ""
        claims = parse_claims(text)
        cites = parse_citations(text)

        # Sourced from the canonical package. Newer builds may export this helper
        # directly; published 0.1.6 exposes it through the rubric object instead.
        aggregate = _aggregate_reward(text, claims, cites, self._current_passage)
        correctness = 1.0 if self._current_is_gold else 0.0

        # Full score vector — the weighted reward is computed over all eight
        # components (correctness included if it carries a weight).
        scores = {
            "aggregate_reward": aggregate,
            "correctness_reward": correctness,
            "n_claims": float(len(claims)),
            "n_citations": float(len(cites)),
            "alternation_canary": alternation_canary(text),
            "starts_with_canary": starts_with_canary(text),
            "length_band_canary": length_band_canary(text),
            "template_echo_canary": template_echo_canary(text),
        }
        # Weight by explicit key (not dict order) so a future reorder of the dict
        # above can't silently scramble the weight<->component mapping. strict=True
        # backstops the length invariant enforced in __init__.
        reward = sum(
            w * scores[k] for w, k in zip(self.weights, _COMPONENT_KEYS, strict=True)
        )

        # Surfaced components withhold the hidden ground truth unless opted in, so
        # a harness that forwards the observation can't leak ``correctness_reward``.
        surfaced = dict(scores)
        if not self.expose_correctness:
            del surfaced["correctness_reward"]

        # Single-step episode: each task is exactly one advocacy turn.
        self._has_task = False
        return AdvocacyObservation(
            prompt="",
            reward=float(reward),
            done=True,
            metadata=dict(surfaced),
            # Mirror into a declared field so the components survive the
            # framework's HTTP serialization (which strips ``metadata``).
            components=dict(surfaced),
        )

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning the current State.
        return self._state

    @property
    def current_passage(self) -> str:
        # [FUNCTION DOC]
        # Property returning the passage of the most recently reset episode (empty string before any reset;
        # still populated after the episode's step).
        """Passage of the active episode (the reading-comprehension text already
        embedded in the reset prompt — not hidden ground truth). Empty before the
        first ``reset``."""
        return self._current_passage
