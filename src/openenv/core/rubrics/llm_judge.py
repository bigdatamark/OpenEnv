# [MODULE DOC]
# LLM-as-a-judge rubric: formats a prompt from the action and observation, sends it to an
# LLMClient, and parses a numeric score from the reply.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM-as-a-judge rubric for reward computation.

Uses an LLM endpoint (via LLMClient) to evaluate agent actions/observations.

Examples:

    ```python
    client = OpenAIClient("http://localhost", 8000, model="meta-llama/...")
    judge = LLMJudge(
        prompt_template="Rate this code solution:\\n{action}\\n\\nScore (0-1):",
        client=client,
    )
    score = await judge(action, observation)
    ```

See RFC 004 for full design: rfcs/004-rubrics.md
"""

import re
from typing import Any, Dict

from openenv.core.llm_client import LLMClient
from openenv.core.rubrics.base import Rubric


class LLMJudge(Rubric):
    # [CLASS DOC]
    # Async rubric that scores (action, observation) with an LLM. Holds the prompt template
    # (with {action} and {observation} placeholders), the client, a compiled score regex
    # (default: first unsigned decimal number), a default score for parse failures, and a
    # normalize flag (clamp to [0, 1]).
    """Rubric that uses an LLM to evaluate agent actions/observations.

    The prompt template is formatted with ``{action}`` and ``{observation}``
    placeholders. The LLM response is parsed for a numeric score.

    Args:
        prompt_template (`str`):
            Template string with {action} and {observation} placeholders.
        client (`LLMClient`):
            An LLMClient instance for making LLM calls.
        score_pattern (`str`, *optional*):
            Regex to extract the score from the LLM response. Defaults to matching
            the first decimal number.
        default_score (`float`, *optional*, defaults to `0.0`):
            Score returned when parsing fails.
        normalize (`bool`, *optional*, defaults to `True`):
            If True, clamp extracted score to [0, 1].
    """

    def __init__(
        self,
        prompt_template: str,
        client: LLMClient,
        *,
        score_pattern: str | None = None,
        default_score: float = 0.0,
        normalize: bool = True,
    ):
        # [FUNCTION DOC]
        # Stores template, client, compiled score pattern, default score and normalize flag.
        super().__init__()
        self.prompt_template = prompt_template
        self._client = client
        self._score_pattern = re.compile(score_pattern or r"(\d+\.?\d*)")
        self.default_score = default_score
        self.normalize = normalize

    async def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Async. Renders the prompt, awaits client.complete(prompt), and returns the parsed score.
        # Client errors propagate.
        """Evaluate by sending a prompt to the LLM and parsing the score.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            `float`: Parsed score from the LLM response.
        """
        prompt = self._render_prompt(action, observation)
        response = await self._client.complete(prompt)
        return self._parse_score(response)

    def _render_prompt(self, action: Any, observation: Any) -> str:
        # [FUNCTION DOC]
        # Returns the template formatted with action and observation (str.format). Raises
        # KeyError/IndexError for unknown placeholders in the template.
        """Format the prompt template with action and observation.

        Override in subclasses for custom prompt construction.
        """
        return self.prompt_template.format(action=action, observation=observation)

    def _parse_score(self, response: str) -> float:
        # [FUNCTION DOC]
        # Searches the response with the score regex; uses group 1 if the pattern has groups, else
        # the whole match, converted to float. Returns default_score when there is no match or the
        # conversion fails; clamps to [0, 1] when normalize is set.
        """Extract a numeric score from the LLM response.

        Uses the configured regex pattern to find the first match.
        Returns default_score if no match is found.
        """
        match = self._score_pattern.search(response)
        if match is None:
            return self.default_score
        try:
            # Use first capture group if present, otherwise full match
            text = match.group(1) if match.lastindex else match.group(0)
            score = float(text)
        except (ValueError, IndexError):
            return self.default_score
        if self.normalize:
            score = max(0.0, min(1.0, score))
        return score

    def state_dict(self) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Returns prompt_template, score_pattern (regex source), default_score and normalize.
        """Serialize rubric configuration."""
        return {
            "prompt_template": self.prompt_template,
            "score_pattern": self._score_pattern.pattern,
            "default_score": self.default_score,
            "normalize": self.normalize,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Restores any of prompt_template, score_pattern (recompiled), default_score and normalize
        # present in state.
        """Load rubric configuration from checkpoint."""
        if "prompt_template" in state:
            self.prompt_template = state["prompt_template"]
        if "score_pattern" in state:
            self._score_pattern = re.compile(state["score_pattern"])
        if "default_score" in state:
            self.default_score = state["default_score"]
        if "normalize" in state:
            self.normalize = state["normalize"]
