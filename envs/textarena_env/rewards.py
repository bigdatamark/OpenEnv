# [MODULE DOC]
# Auxiliary reward-signal providers for TextArena games. Defines the RewardProvider protocol, a factory
# choosing providers by env id (currently only Wordle-v0 has one), Wordle text-parsing helpers and the
# Wordle provider that mirrors GRPO Wordle heuristics.
"""Reward provider utilities for TextArena environments."""

from __future__ import annotations

import re
from typing import Dict, List, Protocol, Tuple

from .models import TextArenaAction, TextArenaObservation


class RewardProvider(Protocol):
    # [CLASS DOC]
    # Protocol for per-episode reward-signal computers with reset() and compute(action, observation).
    """Interface for computing auxiliary reward signals."""

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clears per-episode state. Protocol stub (no implementation).
        """Clear any internal state before a new episode."""

    def compute(
        self, *, action: TextArenaAction, observation: TextArenaObservation
    ) -> Dict[str, float]:
        # [FUNCTION DOC]
        # Returns a mapping from signal name to float for one step. Protocol stub (no implementation).
        """Return a mapping of reward names to float values for the step."""


def build_reward_providers(env_id: str) -> List[RewardProvider]:
    # [FUNCTION DOC]
    # Returns a list of providers for env_id: one Wordle provider for "Wordle-v0", else empty.
    """Instantiate reward providers appropriate for the given environment."""

    providers: List[RewardProvider] = []
    if env_id == "Wordle-v0":
        providers.append(_WordleRewardProvider())
    return providers


_WORDLE_GUESS_PATTERN = re.compile(r"\[[A-Za-z]{5}\]")


def extract_guess(text: str) -> str:
    # [FUNCTION DOC]
    # Returns the first bracketed 5-letter guess in text, lowercased; else the first 5 letters of text's
    # lowercase letters wrapped in brackets; else "[dunno]" when fewer than 5 letters.
    """Normalize a Wordle guess string from arbitrary text."""

    match = _WORDLE_GUESS_PATTERN.search(text)
    if match:
        return match.group(0).lower()

    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 5:
        return f"[{cleaned[:5]}]"
    return "[dunno]"


def extract_wordle_feedback(observation: TextArenaObservation) -> str:
    # [FUNCTION DOC]
    # Returns the text after "Feedback:" in the most recent message containing it (stripped), or "".
    """Pull the latest feedback text from a Wordle observation."""

    for message in reversed(observation.messages):
        content = message.content.strip()
        if "Feedback:" in content:
            return content.split("Feedback:", 1)[-1].strip()
    return ""


def extract_feedback_counts(feedback: str) -> Tuple[int, int]:
    # [FUNCTION DOC]
    # Returns (greens, yellows): from the last non-blank line consisting only of G/Y/X (spaces ignored) in the
    # feedback; (0, 0) if feedback is empty, has fewer than two non-blank lines or has no such line.
    """Return counts of green (G) and yellow (Y) markers from feedback."""

    if not feedback:
        return (0, 0)

    lines = [line.strip() for line in feedback.split("\n") if line.strip()]
    if len(lines) < 2:
        return (0, 0)

    for line in reversed(lines):
        normalized = line.replace(" ", "")
        if normalized and all(c in "GYX" for c in normalized):
            green = normalized.count("G")
            yellow = normalized.count("Y")
            return (green, yellow)

    return (0, 0)


class _WordleRewardProvider:
    # [CLASS DOC]
    # HELPER: Wordle signal provider tracking how often each guess was made this episode. Emits
    # wordle.greens, wordle.yellows (counts divided by 5), wordle.repetitions (1 minus prior occurrences of the
    # guess, so it goes negative on repeats) and wordle.correct (the observation's reward or 0).
    """Reward provider that mirrors the GRPO Wordle heuristics."""

    SIGNAL_MAP = {
        "greens": "wordle.greens",
        "yellows": "wordle.yellows",
        "repetitions": "wordle.repetitions",
        "correct": "wordle.correct",
    }

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Creates an empty guess-count mapping.
        self._guess_history: Dict[str, int] = {}

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clears the guess history.
        self._guess_history.clear()

    def compute(
        self, *, action: TextArenaAction, observation: TextArenaObservation
    ) -> Dict[str, float]:
        # [FUNCTION DOC]
        # Extracts the guess and latest feedback, computes the four signals described on the class, then records
        # the guess (unless it is "[dunno]"). Returns the signal mapping. Side effect: mutates guess history.
        guess = extract_guess(action.message)
        feedback = extract_wordle_feedback(observation)

        normalized_guess = guess if guess and guess != "[dunno]" else ""
        previous_occurrences = (
            self._guess_history.get(normalized_guess, 0) if normalized_guess else 0
        )

        green_score = 0.0
        yellow_score = 0.0
        if feedback:
            green_count, yellow_count = extract_feedback_counts(feedback)
            green_score = green_count / 5.0
            yellow_score = yellow_count / 5.0

        repetition_score = 1.0 - previous_occurrences
        correct_score = float(observation.reward or 0.0)

        if normalized_guess:
            self._guess_history[normalized_guess] = previous_occurrences + 1

        return {
            self.SIGNAL_MAP["greens"]: float(green_score),
            self.SIGNAL_MAP["yellows"]: float(yellow_score),
            self.SIGNAL_MAP["repetitions"]: float(repetition_score),
            self.SIGNAL_MAP["correct"]: float(correct_score),
        }


__all__ = [
    "RewardProvider",
    "build_reward_providers",
    "extract_feedback_counts",
    "extract_guess",
    "extract_wordle_feedback",
]
