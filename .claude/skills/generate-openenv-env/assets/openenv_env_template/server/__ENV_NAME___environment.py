# [MODULE DOC]
# Template server-side environment implementation: an echo environment that returns the sent message and
# a length-based reward. Imports models relatively, falling back to absolute imports when run outside the package.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
__ENV_TITLE_NAME__ Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation
except ImportError:
    from models import __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation


class __ENV_CLASS_NAME__Environment(Environment):
    # [CLASS DOC]
    # Single-session echo environment. Holds a State (episode id + step count) and a reset counter.
    # SUPPORTS_CONCURRENT_SESSIONS is False, signalling the server must not share/parallelize instances.
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = __ENV_CLASS_NAME__Environment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "__ENV_TITLE_NAME__ environment ready!"
        >>>
        >>> obs = env.step(__ENV_CLASS_NAME__Action(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Set to True only when your environment isolates state between instances
    # and max_concurrent_envs > 1 in server/app.py.
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        # [FUNCTION DOC]
        # Creates a fresh State with a random UUID episode id and step_count 0; sets reset counter to 0.
        """Initialize the __ENV_NAME__ environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(
        self, seed=None, episode_id=None, **kwargs
    ) -> __ENV_CLASS_NAME__Observation:
        # [FUNCTION DOC]
        # Starts a new episode. Inputs: seed (ignored), episode_id (used if given, else a new UUID), extra kwargs ignored.
        # Returns: observation with a fixed "ready" message, length 0, done False, reward 0.0.
        # Side effects: replaces internal state and increments the reset counter.
        """
        Reset the environment.

        Args:
            seed: Optional seed for deterministic resets
            episode_id: Optional externally-provided episode id
            **kwargs: Additional reset arguments

        Returns:
            __ENV_CLASS_NAME__Observation with a ready message
        """
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_count += 1

        return __ENV_CLASS_NAME__Observation(
            echoed_message="__ENV_TITLE_NAME__ environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: __ENV_CLASS_NAME__Action) -> __ENV_CLASS_NAME__Observation:  # type: ignore[override]
        # [FUNCTION DOC]
        # Echoes the action's message. Increments step_count.
        # Returns: observation with the message, its length, done False, reward = min(0.1 * length, 1.0),
        # and metadata containing the original message and current step number. Episodes never terminate.
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: __ENV_CLASS_NAME__Action containing the message to echo

        Returns:
            __ENV_CLASS_NAME__Observation with the echoed message and its length
        """
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # Simple bounded reward: longer messages get higher rewards up to 1.0
        reward = min(length * 0.1, 1.0)

        return __ENV_CLASS_NAME__Observation(
            echoed_message=message,
            message_length=length,
            done=False,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning the current State object (live reference, not a copy).
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
