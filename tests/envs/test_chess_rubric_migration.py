# [MODULE DOC]
# Tests that the chess environment's ChessWinLossRubric (an exponential
# discounting trajectory rubric) is wired in, accumulates one entry per agent
# step, clears on reset, and matches the inline discounted_rewards metadata.
# Skipped without python-chess or moonfish.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for chess environment rubric migration.

Verifies that the ChessWinLossRubric produces the same discounted rewards
as the inline _compute_discounted_rewards() method, and that the rubric
integrates correctly with the environment lifecycle.
"""

import pytest

# Skip entire module if chess dependencies are not installed
pytest.importorskip("chess", reason="python-chess is not installed")
pytest.importorskip("moonfish", reason="moonfish is not installed")

from envs.chess_env import ChessAction
from envs.chess_env.server.chess_environment import ChessEnvironment
from envs.chess_env.server.rubrics import ChessWinLossRubric
from openenv.core.rubrics.trajectory import ExponentialDiscountingTrajectoryRubric


class TestRubricIsSet:
    # [CLASS DOC]
    # Rubric wiring.
    """Verify the rubric is properly wired into the environment."""

    def test_rubric_is_chess_win_loss_rubric(self):
        # [FUNCTION DOC]
        # env.rubric is a ChessWinLossRubric.
        """env.rubric is a ChessWinLossRubric instance."""
        env = ChessEnvironment(opponent=None)
        assert isinstance(env.rubric, ChessWinLossRubric)

    def test_rubric_is_exponential_discounting(self):
        # [FUNCTION DOC]
        # It subclasses ExponentialDiscountingTrajectoryRubric.
        """ChessWinLossRubric extends ExponentialDiscountingTrajectoryRubric."""
        env = ChessEnvironment(opponent=None)
        assert isinstance(env.rubric, ExponentialDiscountingTrajectoryRubric)

    def test_rubric_gamma_matches_env(self):
        # [FUNCTION DOC]
        # Rubric gamma equals the env gamma.
        """Rubric gamma matches the environment's gamma parameter."""
        env = ChessEnvironment(opponent=None, gamma=0.95)
        assert env.rubric.gamma == 0.95
        assert env.rubric.gamma == env._gamma

    def test_rubric_gamma_default(self):
        # [FUNCTION DOC]
        # Default gamma is 0.99.
        """Default gamma is 0.99."""
        env = ChessEnvironment(opponent=None)
        assert env.rubric.gamma == 0.99


class TestRubricTrajectoryAccumulation:
    # [CLASS DOC]
    # Trajectory bookkeeping.
    """Verify rubric accumulates trajectory correctly."""

    def test_trajectory_empty_after_reset(self):
        # [FUNCTION DOC]
        # Empty after reset.
        """Rubric trajectory is empty after reset."""
        env = ChessEnvironment(opponent=None, agent_color="white")
        env.reset()
        assert len(env.rubric.trajectory) == 0

    def test_trajectory_accumulates_on_step(self):
        # [FUNCTION DOC]
        # One entry after one step.
        """Rubric trajectory grows with each step."""
        env = ChessEnvironment(opponent=None, agent_color="white")
        env.reset()
        env.step(ChessAction(move="e2e4"))
        assert len(env.rubric.trajectory) == 1

    def test_trajectory_length_matches_agent_moves(self):
        # [FUNCTION DOC]
        # Four self-play steps give four entries.
        """Trajectory length equals number of step() calls."""
        env = ChessEnvironment(opponent=None, agent_color="white")
        env.reset()

        # Play fool's mate (4 moves)
        env.step(ChessAction(move="f2f3"))
        env.step(ChessAction(move="e7e5"))
        env.step(ChessAction(move="g2g4"))
        env.step(ChessAction(move="d8h4"))

        assert len(env.rubric.trajectory) == 4

    def test_trajectory_clears_on_reset(self):
        # [FUNCTION DOC]
        # reset clears the trajectory.
        """Rubric trajectory clears between episodes."""
        env = ChessEnvironment(opponent=None, agent_color="white")
        env.reset()
        env.step(ChessAction(move="e2e4"))
        assert len(env.rubric.trajectory) == 1

        env.reset()
        assert len(env.rubric.trajectory) == 0

    def test_trajectory_with_opponent(self):
        # [FUNCTION DOC]
        # Opponent replies do not add entries.
        """With an opponent, only agent step() calls feed the rubric."""
        env = ChessEnvironment(opponent="random", agent_color="white")
        env.reset()
        env.step(ChessAction(move="e2e4"))

        # Only 1 trajectory entry (the agent's move), not 2
        assert len(env.rubric.trajectory) == 1


class TestRubricMatchesInlineDiscounting:
    # [CLASS DOC]
    # Rubric step rewards equal the metadata discounted rewards.
    """Verify rubric compute_step_rewards() matches metadata discounted_rewards."""

    def test_single_move_checkmate(self):
        # [FUNCTION DOC]
        # Mate in one: lists match.
        """Rubric matches inline for single-move checkmate."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.99)
        fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
        env.reset(fen=fen)

        obs = env.step(ChessAction(move="e1e8"))
        assert obs.done is True
        assert obs.reward == 1.0

        inline_rewards = obs.metadata["discounted_rewards"]
        rubric_rewards = env.rubric.compute_step_rewards()

        assert len(rubric_rewards) == len(inline_rewards)
        for r, i in zip(rubric_rewards, inline_rewards):
            assert abs(r - i) < 1e-9

    def test_fools_mate_self_play(self):
        # [FUNCTION DOC]
        # Fool's mate with gamma 0.9: lists match.
        """Rubric matches inline for fool's mate in self-play."""
        gamma = 0.9
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=gamma)
        env.reset()

        env.step(ChessAction(move="f2f3"))
        env.step(ChessAction(move="e7e5"))
        env.step(ChessAction(move="g2g4"))
        obs = env.step(ChessAction(move="d8h4"))

        assert obs.done is True

        inline_rewards = obs.metadata["discounted_rewards"]
        rubric_rewards = env.rubric.compute_step_rewards()

        assert len(rubric_rewards) == len(inline_rewards)
        for r, i in zip(rubric_rewards, inline_rewards):
            assert abs(r - i) < 1e-9

    def test_gamma_half_single_move(self):
        # [FUNCTION DOC]
        # gamma 0.5 single move: lists match.
        """With gamma=0.5, single-move game: both should return [1.0]."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.5)
        fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
        env.reset(fen=fen)

        obs = env.step(ChessAction(move="e1e8"))
        assert obs.done is True

        inline_rewards = obs.metadata["discounted_rewards"]
        rubric_rewards = env.rubric.compute_step_rewards()

        assert rubric_rewards == pytest.approx(inline_rewards)


class TestRubricScoring:
    # [CLASS DOC]
    # score_trajectory outcomes.
    """Test the rubric's score_trajectory for different outcomes."""

    def test_win_score(self):
        # [FUNCTION DOC]
        # Win scores +1.0.
        """ChessWinLossRubric returns +1.0 on win."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.99)
        fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
        env.reset(fen=fen)

        obs = env.step(ChessAction(move="e1e8"))
        assert obs.done is True

        score = env.rubric.score_trajectory(env.rubric.trajectory)
        assert score == 1.0

    def test_loss_score(self):
        # [FUNCTION DOC]
        # Loss scores -1.0.
        """ChessWinLossRubric returns -1.0 on loss."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.99)
        env.reset()

        # Fool's mate: white loses
        env.step(ChessAction(move="f2f3"))
        env.step(ChessAction(move="e7e5"))
        env.step(ChessAction(move="g2g4"))
        obs = env.step(ChessAction(move="d8h4"))

        assert obs.done is True
        assert obs.reward == -1.0

        score = env.rubric.score_trajectory(env.rubric.trajectory)
        assert score == -1.0

    def test_draw_score(self):
        # [FUNCTION DOC]
        # Stalemate scores 0.0 with step rewards [0.0].
        """ChessWinLossRubric returns 0.0 on stalemate."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.99)
        # Set up a position where white can force stalemate in one move.
        # White queen moves to b6, creating stalemate for black king on a8.
        fen = "k7/8/K7/8/8/8/8/1Q6 w - - 0 1"
        env.reset(fen=fen)

        obs = env.step(ChessAction(move="b1b6"))
        assert obs.done is True
        assert obs.reward == 0.0

        score = env.rubric.score_trajectory(env.rubric.trajectory)
        assert score == 0.0
        assert env.rubric.compute_step_rewards() == pytest.approx([0.0])


class TestMultipleEpisodes:
    # [CLASS DOC]
    # Multi-episode behaviour.
    """Test rubric behaves correctly across multiple episodes."""

    def test_rubric_resets_between_episodes(self):
        # [FUNCTION DOC]
        # Trajectory resets between two mate-in-one episodes; second gives [1.0].
        """Rubric trajectory properly resets between episodes."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.99)

        # Episode 1
        fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
        env.reset(fen=fen)
        obs = env.step(ChessAction(move="e1e8"))
        assert obs.done is True
        assert len(env.rubric.trajectory) == 1

        # Episode 2
        env.reset(fen=fen)
        assert len(env.rubric.trajectory) == 0

        obs = env.step(ChessAction(move="e1e8"))
        assert obs.done is True
        assert len(env.rubric.trajectory) == 1
        assert env.rubric.compute_step_rewards() == pytest.approx([1.0])
