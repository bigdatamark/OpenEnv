# [MODULE DOC]
# Client for the OpenApp environment (simulated web apps driven by BrowserGym-style actions).
# Defines OpenAppEnv, which serializes OpenAppAction and parses observations/state from the server.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenApp Environment HTTP Client.

This module provides the client for connecting to an OpenApp Environment server
over HTTP.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

# Support both in-repo and standalone imports
try:
    from .models import OpenAppAction, OpenAppObservation
except ImportError:
    from openapp_env.models import OpenAppAction, OpenAppObservation


class OpenAppEnv(EnvClient[OpenAppAction, OpenAppObservation, State]):
    # [CLASS DOC]
    # EnvClient subclass for OpenApp sessions; implements payload building and response parsing.
    # Transport (WebSocket session, docker helpers) is inherited.
    """
    HTTP client for the OpenApp Environment.

    This client connects to an OpenAppEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    The OpenApp environment simulates web applications (calendar, todo, messenger, maps)
    and allows agents to interact with them using browser-based actions.

    Example:
        >>> # Connect to a running server
        >>> client = OpenAppEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.url)
        >>>
        >>> # Click on an element
        >>> result = client.step(OpenAppAction(action_type="click", bid="123"))
        >>> print(result.observation.html)
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = OpenAppEnv.from_docker_image("openapp-env:latest")
        >>> result = client.reset()
        >>> # Fill a text field
        >>> result = client.step(OpenAppAction(
        ...     action_type="fill",
        ...     bid="456",
        ...     text="Meeting with team"
        ... ))
    """

    def _step_payload(self, action: OpenAppAction) -> Dict:
        # [FUNCTION DOC]
        # HELPER: Builds the step body: always "action_type", plus bid, text, value, url, direction when
        # not None, and metadata when non-empty. Returns that mapping.
        """
        Convert OpenAppAction to JSON payload for step request.

        Args:
            action: OpenAppAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {
            "action_type": action.action_type,
        }

        # Add optional fields if present
        if action.bid is not None:
            payload["bid"] = action.bid
        if action.text is not None:
            payload["text"] = action.text
        if action.value is not None:
            payload["value"] = action.value
        if action.url is not None:
            payload["url"] = action.url
        if action.direction is not None:
            payload["direction"] = action.direction
        if action.metadata:
            payload["metadata"] = action.metadata

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[OpenAppObservation]:
        # [FUNCTION DOC]
        # HELPER: Converts a step/reset response into a StepResult wrapping an OpenAppObservation built
        # from the nested "observation" mapping (html, url, open_pages_urls, active_page_index, screenshot,
        # axtree_txt, app_state, task_info, last_action_error) with empty/zero/None defaults; done and
        # reward come from the top level, metadata prefers the top level over the nested one.
        """
        Parse server response into StepResult[OpenAppObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with OpenAppObservation
        """
        obs_data = payload.get("observation", {})
        observation = OpenAppObservation(
            html=obs_data.get("html", ""),
            url=obs_data.get("url", ""),
            open_pages_urls=obs_data.get("open_pages_urls", []),
            active_page_index=obs_data.get("active_page_index", 0),
            screenshot=obs_data.get("screenshot"),
            axtree_txt=obs_data.get("axtree_txt", ""),
            app_state=obs_data.get("app_state", {}),
            task_info=obs_data.get("task_info"),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        # [FUNCTION DOC]
        # HELPER: Converts a state response into a generic State with episode_id and step_count (default 0).
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
