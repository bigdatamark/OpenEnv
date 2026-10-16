# [MODULE DOC]
# Tests for the synchronous Rubric base class: abstract forward, calling and last_score tracking,
# automatic child registration and traversal (children, named_children, rubrics, named_rubrics,
# get_rubric), forward and pre-forward hooks, default reset, and state_dict defaults.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the base Rubric class."""

from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric


class SimpleRubric(Rubric):
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


class CompositeRubric(Rubric):
    # [CLASS DOC]
    # HELPER: Rubric with two SimpleRubric children (0.5 and 0.7) returning their mean.
    """Rubric with child rubrics."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Creates the children.
        super().__init__()
        self.child1 = SimpleRubric(0.5)
        self.child2 = SimpleRubric(0.7)

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # HELPER: Calls both children and returns their mean.
        return (self.child1(action, observation) + self.child2(action, observation)) / 2


class TestRubricBasics:
    # [CLASS DOC]
    # Tests basic Rubric behavior.
    """Test basic Rubric functionality."""

    def test_forward_is_abstract(self):
        # [FUNCTION DOC]
        # Rubric cannot be instantiated directly.
        """Cannot instantiate Rubric directly."""
        with pytest.raises(TypeError):
            Rubric()

    def test_simple_rubric_call(self):
        # [FUNCTION DOC]
        # Calling a rubric returns forward's result.
        """Calling a rubric invokes forward()."""
        rubric = SimpleRubric(0.8)
        result = rubric("action", "observation")
        assert result == 0.8

    def test_last_score_tracked(self):
        # [FUNCTION DOC]
        # last_score is None initially and updated after a call.
        """last_score is updated after each call."""
        rubric = SimpleRubric(0.6)
        assert rubric.last_score is None

        rubric("action", "observation")
        assert rubric.last_score == 0.6


class TestChildRegistration:
    # [CLASS DOC]
    # Tests child registration and traversal.
    """Test auto-registration of child rubrics."""

    def test_children_registered(self):
        # [FUNCTION DOC]
        # Rubric attributes are registered as children.
        """Child rubrics are registered when assigned as attributes."""
        rubric = CompositeRubric()

        children = list(rubric.children())
        assert len(children) == 2
        assert rubric.child1 in children
        assert rubric.child2 in children

    def test_named_children(self):
        # [FUNCTION DOC]
        # named_children yields attribute names with child rubrics.
        """named_children returns name-rubric pairs."""
        rubric = CompositeRubric()

        named = dict(rubric.named_children())
        assert "child1" in named
        assert "child2" in named
        assert named["child1"].score == 0.5
        assert named["child2"].score == 0.7

    def test_rubrics_recursive(self):
        # [FUNCTION DOC]
        # rubrics() yields all three descendants of a nested rubric.
        """rubrics() returns all descendants."""

        class NestedRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Rubric wrapping a CompositeRubric as attribute inner.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Creates the inner composite.
                super().__init__()
                self.inner = CompositeRubric()

            def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Returns the inner rubric's score.
                return self.inner(action, observation)

        rubric = NestedRubric()

        all_rubrics = list(rubric.rubrics())
        # inner, inner.child1, inner.child2
        assert len(all_rubrics) == 3

    def test_named_rubrics_paths(self):
        # [FUNCTION DOC]
        # named_rubrics() yields dotted paths.
        """named_rubrics() returns dot-separated paths."""

        class NestedRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Rubric wrapping a CompositeRubric as attribute inner.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Creates the inner composite.
                super().__init__()
                self.inner = CompositeRubric()

            def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Returns the inner rubric's score.
                return self.inner(action, observation)

        rubric = NestedRubric()

        paths = dict(rubric.named_rubrics())
        assert "inner" in paths
        assert "inner.child1" in paths
        assert "inner.child2" in paths

    def test_get_rubric_by_path(self):
        # [FUNCTION DOC]
        # get_rubric resolves dotted paths.
        """get_rubric() navigates dot-separated paths."""

        class NestedRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Rubric wrapping a CompositeRubric as attribute inner.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Creates the inner composite.
                super().__init__()
                self.inner = CompositeRubric()

            def forward(self, action, observation):
                # [FUNCTION DOC]
                # HELPER: Returns the inner rubric's score.
                return self.inner(action, observation)

        rubric = NestedRubric()

        assert rubric.get_rubric("inner") is rubric.inner
        assert rubric.get_rubric("inner.child1") is rubric.inner.child1

    def test_get_rubric_invalid_path(self):
        # [FUNCTION DOC]
        # get_rubric raises KeyError for unknown paths.
        """get_rubric() raises KeyError for invalid paths."""
        rubric = CompositeRubric()

        with pytest.raises(KeyError):
            rubric.get_rubric("nonexistent")


class TestHooks:
    # [CLASS DOC]
    # Tests forward hooks.
    """Test forward hook functionality."""

    def test_forward_hook_called(self):
        # [FUNCTION DOC]
        # A forward hook receives action, observation and result.
        """Forward hooks are called after forward()."""
        rubric = SimpleRubric(0.9)
        hook_calls = []

        def hook(r, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Records (action, observation, result).
            hook_calls.append((action, obs, result))

        rubric.register_forward_hook(hook)
        rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs", 0.9)

    def test_forward_pre_hook_called(self):
        # [FUNCTION DOC]
        # A pre-forward hook receives action and observation.
        """Pre-forward hooks are called before forward()."""
        rubric = SimpleRubric(0.9)
        hook_calls = []

        def pre_hook(r, action, obs):
            # [FUNCTION DOC]
            # HELPER: Records (action, observation).
            hook_calls.append((action, obs))

        rubric.register_forward_pre_hook(pre_hook)
        rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs")

    def test_multiple_hooks(self):
        # [FUNCTION DOC]
        # Multiple hooks run in registration order.
        """Multiple hooks can be registered."""
        rubric = SimpleRubric(0.5)
        results = []

        rubric.register_forward_hook(lambda r, a, o, res: results.append(1))
        rubric.register_forward_hook(lambda r, a, o, res: results.append(2))

        rubric("action", "obs")

        assert results == [1, 2]


class TestReset:
    # [CLASS DOC]
    # Tests reset.
    """Test reset functionality."""

    def test_default_reset_is_noop(self):
        # [FUNCTION DOC]
        # The default reset does not raise.
        """Default reset() does nothing (for stateless rubrics)."""
        rubric = SimpleRubric(0.5)
        rubric.reset()  # Should not raise


class TestStateDictSerialization:
    # [CLASS DOC]
    # Tests state_dict defaults.
    """Test state_dict serialization."""

    def test_default_state_dict_empty(self):
        # [FUNCTION DOC]
        # state_dict defaults to an empty dict.
        """Default state_dict returns empty dict."""
        rubric = SimpleRubric(0.5)
        assert rubric.state_dict() == {}

    def test_load_state_dict_accepts_empty(self):
        # [FUNCTION DOC]
        # load_state_dict accepts an empty dict.
        """load_state_dict accepts empty dict."""
        rubric = SimpleRubric(0.5)
        rubric.load_state_dict({})  # Should not raise
