# [MODULE DOC]
# Server-side WebSearch environment: each step runs one web search via WebSearchTool; there is no reward
# logic or terminal condition.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Websearch Env Environment Implementation.

A web search environment that uses the google search API (via Serper API) to search the web.
"""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Support both in-repo and standalone imports
try:
    from ..models import WebSearchAction, WebSearchObservation
except ImportError:
    from models import WebSearchAction, WebSearchObservation

from .web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class WebSearchEnvironment(Environment):
    # [CLASS DOC]
    # OpenEnv Environment holding a State, a reset counter and a WebSearchTool configured from SERPER_API_KEY
    # (top 5 results, 60 s timeout, full-page mode, no proxy).
    """
    A web search environment that uses the google search API (via Serper API) to search the web.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply returns the search results.

    Example:
        >>> env = WebSearchEnvironment()
        >>> obs = env.reset()
        >>> print(obs.web_contents)  # []
        >>>
        >>> obs = env.step(WebSearchAction(query="What is the capital of France?"))
        >>> print(obs.web_contents)  # [WebContent(title="Capital of France", content="The capital of France is Paris", url="https://en.wikipedia.org/wiki/Paris")]
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Creates the state and the search tool. Logs whether SERPER_API_KEY is set (last 3 chars only) or warns
        # if missing.
        """Initialize the searchr1_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        # Get API key from environment
        api_key = os.environ.get("SERPER_API_KEY")

        # Log API key status for debugging (without exposing the full key)
        if api_key:
            logger.info(f"SERPER_API_KEY found (ends with: {api_key[-3:]}...)")
        else:
            logger.warning("SERPER_API_KEY not found in environment variables!")
            logger.warning(
                "Please set SERPER_API_KEY in Hugging Face Spaces secrets or as an environment variable"
            )

        self._web_search_tool = WebSearchTool(
            api_key=api_key,
            top_k=5,
            timeout=60,
            snippet_only=False,
            proxy=None,
        )

    def reset(self) -> WebSearchObservation:
        # [FUNCTION DOC]
        # Starts a new episode (fresh state, increments reset count). Returns an empty observation with done False
        # and reward 0.0.
        """
        Reset the environment.

        Returns:
            WebSearchObservation with empty web contents
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return WebSearchObservation(
            content="",
            web_contents=[],
            done=False,
            reward=0.0,
        )

    def step(self, action: WebSearchAction) -> WebSearchObservation:  # type: ignore[override]
        # [FUNCTION DOC]
        # Increments step_count and returns the tool's observation for the action's query (never done, reward
        # unset).
        # KNOWN ISSUE: the docstring says it echoes the message; it performs a web search.
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: WebSearchAction containing the message to echo

        Returns:
            WebSearchObservation with the echoed message and its length
        """
        self._state.step_count += 1

        return self._web_search_tool.execute(action)

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning the current State.
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
