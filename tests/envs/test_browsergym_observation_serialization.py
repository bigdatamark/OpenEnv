# [MODULE DOC]
# Tests that BrowserGymEnvironment._create_observation converts array-like
# and scalar-like values (objects with tolist/item) to plain JSON values and
# honours include_screenshot. Skipped without gymnasium.
"""Unit tests for BrowserGym observation serialization."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

pytest.importorskip("gymnasium", reason="gymnasium is not installed")

from envs.browsergym_env.models import BrowserGymState
from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment


class _ArrayLike:
    # [CLASS DOC]
    # HELPER: numpy-like object exposing tolist.
    def tolist(self):
        # [FUNCTION DOC]
        # Returns a 1x1 red pixel nested list.
        return [[[255, 0, 0]]]


class _ScalarLike:
    # [CLASS DOC]
    # HELPER: numpy-scalar-like object exposing item.
    def item(self):
        # [FUNCTION DOC]
        # Returns 3.5.
        return 3.5


def test_create_observation_serializes_array_like_fields() -> None:
    # [FUNCTION DOC]
    # With screenshots enabled, screenshot and nested obs/info scalars are
    # converted; raw obs and info land in metadata.
    env = BrowserGymEnvironment.__new__(BrowserGymEnvironment)
    env.include_screenshot = True
    env._state = BrowserGymState()

    result = BrowserGymEnvironment._create_observation(
        env,
        obs={
            "goal": "Pick the highlighted option",
            "url": "http://example.test",
            "screenshot": _ArrayLike(),
            "extra": {"score": _ScalarLike()},
        },
        info={"reward_estimate": _ScalarLike()},
        done=False,
        reward=0.0,
    )

    assert result.screenshot == [[[255, 0, 0]]]
    assert result.metadata["browsergym_obs"]["extra"]["score"] == 3.5
    assert result.metadata["browsergym_info"]["reward_estimate"] == 3.5


def test_create_observation_omits_screenshot_by_default() -> None:
    # [FUNCTION DOC]
    # With include_screenshot False, screenshot is None.
    env = BrowserGymEnvironment.__new__(BrowserGymEnvironment)
    env.include_screenshot = False
    env._state = BrowserGymState()

    result = BrowserGymEnvironment._create_observation(
        env,
        obs={"goal": "Pick the highlighted option", "screenshot": _ArrayLike()},
        info={},
        done=False,
        reward=0.0,
    )

    assert result.screenshot is None
