# [MODULE DOC]
# Tests for synchronous container rubrics: Sequential (fail-fast chain returning the last score),
# Gate (threshold pass-through), WeightedSum (weight validation and combination), RubricList and
# RubricDict (collection behavior, child registration, no forward), and compositions of them.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for container rubrics: Sequential, Gate, WeightedSum, RubricList, RubricDict."""

from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import (
    Gate,
    RubricDict,
    RubricList,
    Sequential,
    WeightedSum,
)


class FixedRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric returning a fixed score (default 1.0).
    """Concrete rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0):
        # [FUNCTION DOC]
        # HELPER: Stores the score.
        super().__init__()
        self.score = score

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns the score.
        return self.score


class CountingRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric returning a fixed score and counting calls.
    """Rubric that counts how many times it's called."""

    def __init__(self, score: float = 1.0):
        # [FUNCTION DOC]
        # HELPER: Stores the score and zeroes the count.
        super().__init__()
        self.score = score
        self.call_count = 0

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Increments the count and returns the score.
        self.call_count += 1
        return self.score


class TestSequential:
    # [CLASS DOC]
    # Tests Sequential.
    """Test Sequential container."""

    def test_empty_sequential(self):
        # [FUNCTION DOC]
        # An empty Sequential returns 1.0.
        """Empty sequential returns 1.0."""
        rubric = Sequential()
        result = rubric("action", "obs")
        assert result == 1.0

    def test_single_rubric(self):
        # [FUNCTION DOC]
        # A single child's score is returned.
        """Single rubric returns its score."""
        rubric = Sequential(FixedRubric(0.8))
        result = rubric("action", "obs")
        assert result == 0.8

    def test_multiple_rubrics_all_pass(self):
        # [FUNCTION DOC]
        # With all children non-zero the last score is returned.
        """Multiple passing rubrics return last score."""
        rubric = Sequential(
            FixedRubric(1.0),
            FixedRubric(0.8),
            FixedRubric(0.9),
        )
        result = rubric("action", "obs")
        assert result == 0.9

    def test_fail_fast_on_zero(self):
        # [FUNCTION DOC]
        # A zero score stops evaluation; later children are not called.
        """Stops immediately when a rubric returns 0."""
        r1 = CountingRubric(1.0)
        r2 = CountingRubric(0.0)  # Fails
        r3 = CountingRubric(1.0)

        rubric = Sequential(r1, r2, r3)
        result = rubric("action", "obs")

        assert result == 0.0
        assert r1.call_count == 1
        assert r2.call_count == 1
        assert r3.call_count == 0  # Should not be called

    def test_children_registered(self):
        # [FUNCTION DOC]
        # Children are registered.
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = Sequential(r1, r2)

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_len_and_getitem(self):
        # [FUNCTION DOC]
        # len and indexing expose the children.
        """__len__ and __getitem__ work correctly."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = Sequential(r1, r2)

        assert len(rubric) == 2
        assert rubric[0] is r1
        assert rubric[1] is r2


class TestGate:
    # [CLASS DOC]
    # Tests Gate.
    """Test Gate container."""

    def test_gate_passes_above_threshold(self):
        # [FUNCTION DOC]
        # A score above the threshold passes through.
        """Returns child score when above threshold."""
        rubric = Gate(FixedRubric(0.8), threshold=0.5)
        result = rubric("action", "obs")
        assert result == 0.8

    def test_gate_fails_below_threshold(self):
        # [FUNCTION DOC]
        # A score below the threshold becomes 0.
        """Returns 0 when child score is below threshold."""
        rubric = Gate(FixedRubric(0.4), threshold=0.5)
        result = rubric("action", "obs")
        assert result == 0.0

    def test_gate_passes_at_threshold(self):
        # [FUNCTION DOC]
        # A score equal to the threshold passes.
        """Returns score when exactly at threshold."""
        rubric = Gate(FixedRubric(0.5), threshold=0.5)
        result = rubric("action", "obs")
        assert result == 0.5

    def test_gate_default_threshold(self):
        # [FUNCTION DOC]
        # The default threshold is 1.0.
        """Default threshold is 1.0."""
        # Passes only with perfect score
        rubric = Gate(FixedRubric(1.0))
        assert rubric("action", "obs") == 1.0

        rubric2 = Gate(FixedRubric(0.99))
        assert rubric2("action", "obs") == 0.0

    def test_gate_child_registered(self):
        # [FUNCTION DOC]
        # The wrapped rubric is registered as the only child.
        """Child rubric is auto-registered."""
        child = FixedRubric(0.5)
        rubric = Gate(child, threshold=0.3)

        children = list(rubric.children())
        assert len(children) == 1
        assert child in children


class TestWeightedSum:
    # [CLASS DOC]
    # Tests WeightedSum.
    """Test WeightedSum container."""

    def test_single_rubric_weight_one(self):
        # [FUNCTION DOC]
        # One child with weight 1 returns its score.
        """Single rubric with weight 1.0."""
        rubric = WeightedSum([FixedRubric(0.8)], [1.0])
        result = rubric("action", "obs")
        assert result == 0.8

    def test_two_rubrics_equal_weights(self):
        # [FUNCTION DOC]
        # Equal weights average scores.
        """Two rubrics with equal weights."""
        rubric = WeightedSum(
            [FixedRubric(0.6), FixedRubric(0.8)],
            [0.5, 0.5],
        )
        result = rubric("action", "obs")
        assert result == pytest.approx(0.7)

    def test_weighted_combination(self):
        # [FUNCTION DOC]
        # Unequal weights combine linearly.
        """Weighted combination with different weights."""
        rubric = WeightedSum(
            [FixedRubric(1.0), FixedRubric(0.0)],
            [0.7, 0.3],
        )
        result = rubric("action", "obs")
        assert result == pytest.approx(0.7)

    def test_weights_must_sum_to_one(self):
        # [FUNCTION DOC]
        # Weights not summing to 1 raise ValueError.
        """Raises error if weights don't sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            WeightedSum([FixedRubric(0.5), FixedRubric(0.5)], [0.5, 0.3])

    def test_lengths_must_match(self):
        # [FUNCTION DOC]
        # Mismatched rubric/weight lengths raise ValueError.
        """Raises error if lengths don't match."""
        with pytest.raises(ValueError, match="must match"):
            WeightedSum([FixedRubric(0.5), FixedRubric(0.5)], [1.0])

    def test_children_registered(self):
        # [FUNCTION DOC]
        # Children are registered.
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = WeightedSum([r1, r2], [0.5, 0.5])

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_weights_property(self):
        # [FUNCTION DOC]
        # weights returns a copy that does not affect the rubric when mutated.
        """weights property returns copy of weights."""
        rubric = WeightedSum([FixedRubric(0.5)], [1.0])

        weights = rubric.weights
        assert weights == [1.0]

        # Modifying copy shouldn't affect internal state
        weights.append(0.5)
        assert rubric.weights == [1.0]


class TestRubricList:
    # [CLASS DOC]
    # Tests RubricList.
    """Test RubricList container."""

    def test_empty_list(self):
        # [FUNCTION DOC]
        # An empty list has length 0.
        """Empty list has length 0."""
        rubric = RubricList()
        assert len(rubric) == 0

    def test_init_with_rubrics(self):
        # [FUNCTION DOC]
        # Initial rubrics are stored in order.
        """Initialize with list of rubrics."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricList([r1, r2])

        assert len(rubric) == 2
        assert rubric[0] is r1
        assert rubric[1] is r2

    def test_append(self):
        # [FUNCTION DOC]
        # append adds a rubric.
        """Append adds rubric to list."""
        rubric = RubricList()
        r1 = FixedRubric(0.5)

        rubric.append(r1)

        assert len(rubric) == 1
        assert rubric[0] is r1

    def test_extend(self):
        # [FUNCTION DOC]
        # extend adds several rubrics.
        """Extend adds multiple rubrics."""
        rubric = RubricList()
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric.extend([r1, r2])

        assert len(rubric) == 2

    def test_iteration(self):
        # [FUNCTION DOC]
        # Iteration yields rubrics in order.
        """Can iterate over rubrics."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricList([r1, r2])

        items = list(rubric)
        assert items == [r1, r2]

    def test_children_registered(self):
        # [FUNCTION DOC]
        # Items are registered as children.
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricList([r1, r2])

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_forward_not_implemented(self):
        # [FUNCTION DOC]
        # Calling the list raises NotImplementedError.
        """forward() raises NotImplementedError."""
        rubric = RubricList([FixedRubric(0.5)])

        with pytest.raises(NotImplementedError):
            rubric("action", "obs")


class TestRubricDict:
    # [CLASS DOC]
    # Tests RubricDict.
    """Test RubricDict container."""

    def test_empty_dict(self):
        # [FUNCTION DOC]
        # An empty dict has length 0.
        """Empty dict has length 0."""
        rubric = RubricDict()
        assert len(rubric) == 0

    def test_init_with_dict(self):
        # [FUNCTION DOC]
        # Initial mapping entries are accessible by key.
        """Initialize with dictionary of rubrics."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricDict({"game1": r1, "game2": r2})

        assert len(rubric) == 2
        assert rubric["game1"] is r1
        assert rubric["game2"] is r2

    def test_setitem_and_getitem(self):
        # [FUNCTION DOC]
        # Item assignment and lookup work.
        """__setitem__ and __getitem__ work."""
        rubric = RubricDict()
        r1 = FixedRubric(0.5)

        rubric["game1"] = r1

        assert rubric["game1"] is r1

    def test_contains(self):
        # [FUNCTION DOC]
        # Membership tests work by key.
        """__contains__ works."""
        rubric = RubricDict({"game1": FixedRubric(0.5)})

        assert "game1" in rubric
        assert "game2" not in rubric

    def test_keys_values_items(self):
        # [FUNCTION DOC]
        # keys, values and items work like a dict.
        """keys(), values(), items() work."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricDict({"game1": r1, "game2": r2})

        assert set(rubric.keys()) == {"game1", "game2"}
        assert set(rubric.values()) == {r1, r2}
        assert set(rubric.items()) == {("game1", r1), ("game2", r2)}

    def test_iteration(self):
        # [FUNCTION DOC]
        # Iteration yields keys.
        """Can iterate over keys."""
        rubric = RubricDict({"game1": FixedRubric(0.5), "game2": FixedRubric(0.7)})

        keys = list(rubric)
        assert set(keys) == {"game1", "game2"}

    def test_update(self):
        # [FUNCTION DOC]
        # update adds entries.
        """update() adds rubrics from dict."""
        rubric = RubricDict({"game1": FixedRubric(0.5)})
        rubric.update({"game2": FixedRubric(0.7)})

        assert len(rubric) == 2
        assert "game2" in rubric

    def test_children_registered(self):
        # [FUNCTION DOC]
        # Values are registered as children.
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricDict({"game1": r1, "game2": r2})

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_forward_not_implemented(self):
        # [FUNCTION DOC]
        # Calling the dict raises NotImplementedError.
        """forward() raises NotImplementedError."""
        rubric = RubricDict({"game1": FixedRubric(0.5)})

        with pytest.raises(NotImplementedError):
            rubric("action", "obs")


class TestContainerComposition:
    # [CLASS DOC]
    # Tests nested containers.
    """Test composing containers together."""

    def test_sequential_of_gates(self):
        # [FUNCTION DOC]
        # A Sequential of passing Gates returns the final score.
        """Sequential of Gate rubrics for hierarchical gating."""
        rubric = Sequential(
            Gate(FixedRubric(1.0)),  # Must pass completely
            Gate(FixedRubric(0.6), threshold=0.5),  # Must be >= 0.5
            FixedRubric(0.9),  # Final score
        )
        result = rubric("action", "obs")
        assert result == 0.9

    def test_sequential_fails_early(self):
        # [FUNCTION DOC]
        # A failing Gate stops the Sequential before later children.
        """Sequential stops when Gate fails."""
        r3 = CountingRubric(0.9)

        rubric = Sequential(
            Gate(FixedRubric(0.3), threshold=0.5),  # Fails
            r3,
        )
        result = rubric("action", "obs")

        assert result == 0.0
        assert r3.call_count == 0

    def test_weighted_sum_of_gates(self):
        # [FUNCTION DOC]
        # A WeightedSum of Gates weights gated scores.
        """WeightedSum with Gate rubrics."""
        rubric = WeightedSum(
            [
                Gate(FixedRubric(0.8), threshold=0.5),  # Passes: 0.8
                Gate(FixedRubric(0.3), threshold=0.5),  # Fails: 0.0
            ],
            [0.6, 0.4],
        )
        result = rubric("action", "obs")
        # 0.8 * 0.6 + 0.0 * 0.4 = 0.48
        assert result == pytest.approx(0.48)

    def test_nested_named_rubrics(self):
        # [FUNCTION DOC]
        # named_rubrics yields paths like task.rubric_0.rubric for nested containers.
        """Can traverse nested rubrics with named_rubrics()."""
        inner = Sequential(
            Gate(FixedRubric(1.0), threshold=0.5),
            FixedRubric(0.8),
        )
        outer = RubricDict({"task": inner})

        paths = dict(outer.named_rubrics())

        # Should have paths for all nested rubrics
        assert "task" in paths
        assert "task.rubric_0" in paths  # Gate
        assert "task.rubric_1" in paths  # FixedRubric
        # Gate's child
        assert "task.rubric_0.rubric" in paths
