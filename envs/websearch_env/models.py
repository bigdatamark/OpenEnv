# [MODULE DOC]
# Pydantic models for the WebSearch environment: search action, observation and one web content result.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the WebSearch Env Environment.

The WebSearch Env environment is an environment that searches the web with Google Search API (via Serper.dev).
"""

from __future__ import annotations

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class WebSearchAction(Action):
    # [CLASS DOC]
    # Action with the required search query and an optional temporary Serper API key overriding the server's.
    # KNOWN ISSUE: the docstring says it is "a message to echo", which does not match the fields.
    """Action for the WebSearch Env environment - just a message to echo."""

    query: str = Field(..., description="The query to search the web for")
    temp_api_key: str | None = Field(
        None,
        description="The temporary API key to use for the Serper API (better to use the default API key from the environment variables)",
    )


class WebSearchObservation(Observation):
    # [CLASS DOC]
    # Observation with formatted text of the results (or an error message) and the list of WebContent results.
    # KNOWN ISSUE: the docstring says it is "the echoed message".
    """Observation from the WebSearch Env environment - the echoed message."""

    content: str = Field(
        ...,
        description="The formatted content of the search results or error message if the search failed",
    )
    web_contents: list[WebContent] = Field(
        ..., description="The web contents of the search results"
    )


class WebContent(BaseModel):
    # [CLASS DOC]
    # One search result: title, extracted content text and URL.
    """Web content of a search result."""

    title: str = Field(..., description="The title of the web content")
    content: str = Field(..., description="The content of the web content")
    url: str = Field(..., description="The URL of the web content")
