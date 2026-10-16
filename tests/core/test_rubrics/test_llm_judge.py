# [MODULE DOC]
# Tests for the LLMJudge rubric using a mock LLM client: prompt template rendering, score
# extraction from responses (default and custom regex, default score, normalization/clamping),
# hook integration, use inside WeightedSum with other rubrics, and state_dict round-tripping.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for LLMJudge rubric."""

from typing import Any

import pytest
from openenv.core.llm_client import LLMClient
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum
from openenv.core.rubrics.llm_judge import LLMJudge


class MockLLMClient(LLMClient):
    # [CLASS DOC]
    # HELPER: LLMClient returning a canned response, recording the last prompt and counting calls.
    """Mock LLM client that returns a canned response."""

    def __init__(self, response: str = "0.8"):
        # [FUNCTION DOC]
        # HELPER: Initializes with a dummy endpoint and the canned response.
        super().__init__("http://mock", 0)
        self.response = response
        self.last_prompt: str | None = None
        self.call_count = 0

    async def complete(self, prompt: str, **kwargs) -> str:
        # [FUNCTION DOC]
        # HELPER: Records the prompt, increments the count and returns the canned response.
        self.last_prompt = prompt
        self.call_count += 1
        return self.response


class TestLLMJudgePromptRendering:
    # [CLASS DOC]
    # Tests prompt rendering.
    """Test prompt template rendering."""

    @pytest.mark.asyncio
    async def test_action_and_observation_substituted(self):
        # [FUNCTION DOC]
        # Both placeholders are filled.
        """Both {action} and {observation} placeholders are filled."""
        client = MockLLMClient("0.5")
        judge = LLMJudge(
            prompt_template="Action: {action}\nObservation: {observation}\nScore:",
            client=client,
        )
        await judge("move_left", "wall_hit")

        assert client.last_prompt == "Action: move_left\nObservation: wall_hit\nScore:"

    @pytest.mark.asyncio
    async def test_action_only_template(self):
        # [FUNCTION DOC]
        # A template may omit the observation placeholder.
        """{observation} can be omitted from the template."""
        client = MockLLMClient("0.9")
        judge = LLMJudge(
            prompt_template="Rate: {action}",
            client=client,
        )
        await judge("print('hello')", "output: hello")

        assert client.last_prompt == "Rate: print('hello')"

    @pytest.mark.asyncio
    async def test_complex_objects_as_strings(self):
        # [FUNCTION DOC]
        # Non-string values are rendered with their string form.
        """Non-string action/observation are converted via str.format()."""
        client = MockLLMClient("0.7")
        judge = LLMJudge(
            prompt_template="Action={action}, Obs={observation}",
            client=client,
        )
        await judge(42, {"key": "value"})

        assert client.last_prompt == "Action=42, Obs={'key': 'value'}"


class TestLLMJudgeScoreParsing:
    # [CLASS DOC]
    # Tests score parsing.
    """Test score extraction from LLM responses."""

    @pytest.mark.asyncio
    async def test_parse_decimal(self):
        # [FUNCTION DOC]
        # A decimal in the response is extracted.
        """Extracts decimal score from response."""
        client = MockLLMClient("The score is 0.75 out of 1.")
        judge = LLMJudge(prompt_template="{action}", client=client)

        score = await judge("test", "obs")
        assert score == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_parse_integer(self):
        # [FUNCTION DOC]
        # An integer 1 yields 1.0.
        """Extracts integer score, clamped to 1.0 when normalize=True."""
        client = MockLLMClient("Score: 1")
        judge = LLMJudge(prompt_template="{action}", client=client)

        score = await judge("test", "obs")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_parse_integer_above_one_normalized(self):
        # [FUNCTION DOC]
        # With normalization, values above 1 are clamped to 1.0.
        """Integer > 1 is clamped to 1.0 with normalize=True."""
        client = MockLLMClient("Score: 8")
        judge = LLMJudge(prompt_template="{action}", client=client)

        score = await judge("test", "obs")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_parse_integer_above_one_unnormalized(self):
        # [FUNCTION DOC]
        # Without normalization, values above 1 pass through.
        """Integer > 1 passes through with normalize=False."""
        client = MockLLMClient("Score: 8")
        judge = LLMJudge(prompt_template="{action}", client=client, normalize=False)

        score = await judge("test", "obs")
        assert score == 8.0

    @pytest.mark.asyncio
    async def test_no_match_returns_default(self):
        # [FUNCTION DOC]
        # No number yields the default score 0.0.
        """Returns default_score when no number is found."""
        client = MockLLMClient("I cannot rate this.")
        judge = LLMJudge(prompt_template="{action}", client=client)

        score = await judge("test", "obs")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_custom_default_score(self):
        # [FUNCTION DOC]
        # A custom default score is used on parse failure.
        """Custom default_score is returned on parse failure."""
        client = MockLLMClient("no number here")
        judge = LLMJudge(
            prompt_template="{action}",
            client=client,
            default_score=0.5,
        )

        score = await judge("test", "obs")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_custom_score_pattern(self):
        # [FUNCTION DOC]
        # A custom regex's first group is used as the score.
        """Custom regex extracts from different response format."""
        client = MockLLMClient("Rating: [7/10]")
        judge = LLMJudge(
            prompt_template="{action}",
            client=client,
            score_pattern=r"\[(\d+)/10\]",
            normalize=False,
        )

        score = await judge("test", "obs")
        assert score == 7.0

    @pytest.mark.asyncio
    async def test_normalization_clamps_low(self):
        # [FUNCTION DOC]
        # With normalization, negative scores are clamped to 0.
        """Negative scores (from custom pattern) are clamped to 0."""
        client = MockLLMClient("Score: -0.5")
        judge = LLMJudge(
            prompt_template="{action}",
            client=client,
            score_pattern=r"(-?\d+\.?\d*)",
        )

        score = await judge("test", "obs")
        assert score == 0.0


class TestLLMJudgeHooks:
    # [CLASS DOC]
    # Tests hook integration.
    """Test integration with Rubric hook system."""

    @pytest.mark.asyncio
    async def test_pre_hook_called(self):
        # [FUNCTION DOC]
        # A pre-forward hook runs with action and observation.
        """Pre-forward hooks are called before LLM evaluation."""
        client = MockLLMClient("0.5")
        judge = LLMJudge(prompt_template="{action}", client=client)
        hook_calls = []

        def pre_hook(rubric, action, obs):
            # [FUNCTION DOC]
            # HELPER: Records a ("pre", action, observation) tuple.
            hook_calls.append(("pre", action, obs))

        judge.register_forward_pre_hook(pre_hook)
        await judge("act", "obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("pre", "act", "obs")

    @pytest.mark.asyncio
    async def test_post_hook_called(self):
        # [FUNCTION DOC]
        # A forward hook receives the parsed score.
        """Post-forward hooks receive the parsed score."""
        client = MockLLMClient("0.75")
        judge = LLMJudge(prompt_template="{action}", client=client)
        hook_calls = []

        def post_hook(rubric, action, obs, result):
            # [FUNCTION DOC]
            # HELPER: Records a ("post", result) tuple.
            hook_calls.append(("post", result))

        judge.register_forward_hook(post_hook)
        await judge("act", "obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("post", 0.75)

    @pytest.mark.asyncio
    async def test_last_score_tracked(self):
        # [FUNCTION DOC]
        # last_score is updated after evaluation.
        """last_score is updated after evaluation."""
        client = MockLLMClient("0.6")
        judge = LLMJudge(prompt_template="{action}", client=client)

        assert judge.last_score is None
        await judge("act", "obs")
        assert judge.last_score == pytest.approx(0.6)


class TestLLMJudgeWithContainers:
    # [CLASS DOC]
    # Tests LLMJudge inside containers.
    """Test LLMJudge works with container rubrics."""

    @pytest.mark.asyncio
    async def test_weighted_sum_with_llm_judges(self):
        # [FUNCTION DOC]
        # A WeightedSum of two judges returns the weighted score.
        """Multiple LLMJudges in a WeightedSum run in parallel."""
        client1 = MockLLMClient("0.8")
        client2 = MockLLMClient("0.6")

        judge1 = LLMJudge(prompt_template="{action}", client=client1)
        judge2 = LLMJudge(prompt_template="{action}", client=client2)

        combined = WeightedSum([judge1, judge2], weights=[0.7, 0.3])
        score = await combined("act", "obs")

        expected = 0.8 * 0.7 + 0.6 * 0.3
        assert score == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_mixed_sync_and_llm_judge(self):
        # [FUNCTION DOC]
        # A WeightedSum of a judge and a sync rubric returns the weighted score.
        """LLMJudge can be mixed with sync rubrics in WeightedSum."""

        class FixedRubric(Rubric):
            # [CLASS DOC]
            # HELPER: Sync rubric returning 0.5.
            def forward(self, action: Any, observation: Any) -> float:
                # [FUNCTION DOC]
                # HELPER: Returns 0.5.
                return 0.5

        client = MockLLMClient("0.9")
        judge = LLMJudge(prompt_template="{action}", client=client)
        fixed = FixedRubric()

        combined = WeightedSum([judge, fixed], weights=[0.6, 0.4])
        score = await combined("act", "obs")

        expected = 0.9 * 0.6 + 0.5 * 0.4
        assert score == pytest.approx(expected)


class TestLLMJudgeStateDictRoundtrip:
    # [CLASS DOC]
    # Tests state_dict and load_state_dict.
    """Test serialization/deserialization of LLMJudge config."""

    def test_state_dict_contents(self):
        # [FUNCTION DOC]
        # state_dict contains prompt template, score pattern, default score and normalize flag.
        """state_dict contains all configurable fields."""
        client = MockLLMClient()
        judge = LLMJudge(
            prompt_template="Rate: {action}",
            client=client,
            score_pattern=r"\[(\d+)\]",
            default_score=0.5,
            normalize=False,
        )

        state = judge.state_dict()
        assert state["prompt_template"] == "Rate: {action}"
        assert state["score_pattern"] == r"\[(\d+)\]"
        assert state["default_score"] == 0.5
        assert state["normalize"] is False

    def test_load_state_dict_restores_config(self):
        # [FUNCTION DOC]
        # load_state_dict restores all four fields (recompiling the pattern).
        """load_state_dict restores all configurable fields."""
        client = MockLLMClient()
        judge = LLMJudge(
            prompt_template="original: {action}",
            client=client,
        )

        judge.load_state_dict(
            {
                "prompt_template": "updated: {action}",
                "score_pattern": r"SCORE=(\d+)",
                "default_score": 0.99,
                "normalize": False,
            }
        )

        assert judge.prompt_template == "updated: {action}"
        assert judge._score_pattern.pattern == r"SCORE=(\d+)"
        assert judge.default_score == 0.99
        assert judge.normalize is False

    def test_load_state_dict_partial_update(self):
        # [FUNCTION DOC]
        # load_state_dict updates only the provided keys.
        """load_state_dict with partial keys only updates those fields."""
        client = MockLLMClient()
        judge = LLMJudge(
            prompt_template="original: {action}",
            client=client,
            default_score=0.0,
        )

        judge.load_state_dict({"default_score": 0.5})

        assert judge.prompt_template == "original: {action}"
        assert judge.default_score == 0.5
