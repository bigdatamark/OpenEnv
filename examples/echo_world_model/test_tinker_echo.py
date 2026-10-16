# [MODULE DOC]
# Pytest tests for the Tinker backend's data builder using stand-in Tinker types and a code-point tokenizer: verifies
# input/target shifting and global env-token weight normalisation, and (if tinker is installed) compatibility with real Tinker types.
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from backends.tinker_echo_demo import build_echo_data
from trajectory import ACTION, CONTEXT, ENV_OUTPUT, Segment, Trajectory


class _CharTokenizer:
    # [CLASS DOC]
    # HELPER: Tokenizer stand-in whose encode returns the code point of each character.
    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC]
        # Returns the list of character code points; add_special_tokens is ignored.
        return [ord(char) for char in text]


@dataclass
class _ModelInput:
    # [CLASS DOC]
    # HELPER: Stand-in for Tinker's ModelInput holding a token list.
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens):
        # [FUNCTION DOC]
        # Alternate constructor returning an instance with the given tokens.
        return cls(tokens)


@dataclass
class _Datum:
    # [CLASS DOC]
    # HELPER: Stand-in for Tinker's Datum holding a model input and loss inputs dict.
    model_input: _ModelInput
    loss_fn_inputs: dict


_TINKER_TYPES = SimpleNamespace(Datum=_Datum, ModelInput=_ModelInput)


def test_build_echo_data_shifts_and_normalizes_env_only_weights():
    # [FUNCTION DOC]
    # Checks inputs drop the last token, targets drop the first, and env-output target weights are 1/3 each across both trajectories.
    trajectories = [
        Trajectory(
            segments=[
                Segment(CONTEXT, "C"),
                Segment(ACTION, "AB"),
                Segment(ENV_OUTPUT, "XY"),
            ],
            reward=1.0,
        ),
        Trajectory([Segment(CONTEXT, "D"), Segment(ENV_OUTPUT, "Z")], reward=0.0),
    ]

    first, second = build_echo_data(_TINKER_TYPES, _CharTokenizer(), trajectories)

    assert first.model_input.tokens == [ord(char) for char in "CABX"]
    assert first.loss_fn_inputs["target_tokens"] == [ord(char) for char in "ABXY"]
    assert first.loss_fn_inputs["weights"] == [0.0, 0.0, 1 / 3, 1 / 3]
    assert second.loss_fn_inputs["weights"] == [1 / 3]


def test_build_echo_data_accepts_real_tinker_types():
    # [FUNCTION DOC]
    # Skips unless tinker is importable; checks a real Datum is built with the expected input tokens and weights summing to 1.
    tinker = pytest.importorskip("tinker")
    trajectory = Trajectory(
        [Segment(CONTEXT, "prompt"), Segment(ENV_OUTPUT, "result")], reward=0.0
    )

    [datum] = build_echo_data(tinker.types, _CharTokenizer(), [trajectory])

    assert isinstance(datum, tinker.types.Datum)
    assert datum.model_input.to_ints() == [ord(char) for char in "promptresul"]
    assert sum(datum.loss_fn_inputs["weights"].tolist()) == pytest.approx(1.0)
