# [MODULE DOC]
# Exception hierarchy for environment server operations, rooted at
# OpenEnvError. Each subclass stores its context values as attributes and
# builds a default message when none is given.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom exceptions for environment server operations."""

from typing import Optional


class OpenEnvError(Exception):
    # [CLASS DOC]
    # Base class for all OpenEnv exceptions.
    """Base exception for all OpenEnv errors."""

    pass


class ConcurrencyConfigurationError(OpenEnvError):
    # [CLASS DOC]
    # Raised at server setup when more than one concurrent environment is requested
    # for an environment class not marked as supporting concurrent sessions.
    """
    Raised when an environment is misconfigured for concurrent sessions.

    This error is raised during server startup when max_concurrent_envs > 1
    is specified for an environment that is not marked as SUPPORTS_CONCURRENT_SESSIONS.
    """

    def __init__(
        self,
        environment_name: str,
        max_concurrent_envs: int,
        message: Optional[str] = None,
    ):
        # [FUNCTION DOC]
        # Stores environment_name and max_concurrent_envs; default message explains the
        # mismatch and the two ways to fix it.
        self.environment_name = environment_name
        self.max_concurrent_envs = max_concurrent_envs

        if message is None:
            message = (
                f"Environment '{environment_name}' is not marked as SUPPORTS_CONCURRENT_SESSIONS. "
                f"Cannot run with max_concurrent_envs={max_concurrent_envs}. "
                f"Either set max_concurrent_envs=1 or ensure the environment "
                f"properly isolates session state and set SUPPORTS_CONCURRENT_SESSIONS=True."
            )

        super().__init__(message)


class SessionCapacityError(OpenEnvError):
    # [CLASS DOC]
    # Raised when a new session is refused because the server is at its maximum
    # number of active sessions.
    """
    Raised when the server cannot accept new sessions due to capacity limits.

    This error is raised when a new WebSocket connection is attempted but
    the server has already reached max_concurrent_envs active sessions.
    """

    def __init__(
        self,
        active_sessions: int,
        max_sessions: int,
        message: Optional[str] = None,
    ):
        # [FUNCTION DOC]
        # Stores active_sessions and max_sessions; default message "Server at capacity:
        # <active>/<max> sessions active...".
        self.active_sessions = active_sessions
        self.max_sessions = max_sessions

        if message is None:
            message = (
                f"Server at capacity: {active_sessions}/{max_sessions} sessions active. "
                f"Cannot accept new connections."
            )

        super().__init__(message)


class SessionNotFoundError(OpenEnvError):
    # [CLASS DOC]
    # Raised when a session id does not exist.
    """Raised when attempting to access a session that does not exist."""

    def __init__(self, session_id: str, message: Optional[str] = None):
        # [FUNCTION DOC]
        # Stores session_id; default message "Session '<id>' not found.".
        self.session_id = session_id

        if message is None:
            message = f"Session '{session_id}' not found."

        super().__init__(message)


class SessionCreationError(OpenEnvError):
    # [CLASS DOC]
    # Raised when a session cannot be created.
    """Raised when a session cannot be created."""

    def __init__(self, reason: str, message: Optional[str] = None):
        # [FUNCTION DOC]
        # Stores reason; default message "Failed to create session: <reason>".
        self.reason = reason

        if message is None:
            message = f"Failed to create session: {reason}"

        super().__init__(message)


class EnvironmentFactoryError(OpenEnvError):
    # [CLASS DOC]
    # Raised when the environment factory fails to produce an instance.
    """Raised when the environment factory fails to create an instance."""

    def __init__(self, factory_name: str, message: Optional[str] = None):
        # [FUNCTION DOC]
        # Stores factory_name; default message naming the factory.
        self.factory_name = factory_name

        if message is None:
            message = f"Environment factory '{factory_name}' failed to create instance."

        super().__init__(message)
