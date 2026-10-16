# [MODULE DOC]
# Template server-side environment implementation for a generated environment:
# an echo environment that returns each action's message with a length-based
# reward. Imports models relatively, falling back to a top-level "models"
# module when run outside the package.
# SPDX-License-Identifier: BSD-3-Clause

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
    # Environment subclass with per-instance State (episode id, step count) and a
    # reset counter. Declares SUPPORTS_CONCURRENT_SESSIONS = True so the server may
    # create one instance per WebSocket session.
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

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        # [FUNCTION DOC]
        # Creates a fresh State with a random UUID episode id and step count 0; reset
        # counter starts at 0.
        """Initialize the __ENV_NAME__ environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> __ENV_CLASS_NAME__Observation:
        # [FUNCTION DOC]
        # Replaces the state with a new episode id and step count 0, increments the
        # reset counter, and returns an observation with a "<title> environment ready!"
        # message, length 0, done False, reward 0.0.
        """
        Reset the environment.

        Returns:
            __ENV_CLASS_NAME__Observation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return __ENV_CLASS_NAME__Observation(
            echoed_message="__ENV_TITLE_NAME__ environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: __ENV_CLASS_NAME__Action) -> __ENV_CLASS_NAME__Observation:  # type: ignore[override]
        # [FUNCTION DOC]
        # Increments the step count and returns an observation echoing action.message
        # with its length, done False, reward min(0.1 * length, 1.0), and metadata
        # containing the original message and current step number.
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
        # Returns the current State object (live reference).
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
