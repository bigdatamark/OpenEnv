# [MODULE DOC]
# Defines the abstract turn-based adapter that wraps an external agentic harness
# (e.g. a CLI coding agent) as a long-lived process, plus the harness error hierarchy.
# Concrete adapters implement process lifecycle, tool injection and streaming turns;
# the base class provides the non-streaming send_message on top of the stream.
# SPDX-License-Identifier: BSD-3-Clause

"""Turn-based adapter interface for external agentic harnesses (RFC 005)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from ..env_server.mcp_types import Tool
from .config import HarnessConfig
from .events import HarnessEvent, HarnessEventType, HarnessResponse


class HarnessError(Exception):
    # [CLASS DOC]
    # Base exception for all harness failures. No extra behavior.
    """Base error for agentic harness failures."""


class HarnessStartupError(HarnessError):
    # [CLASS DOC]
    # Raised when the harness process fails to start or become ready in time.
    """The harness process failed to start or become ready in time."""


class HarnessNotRunningError(HarnessError):
    # [CLASS DOC]
    # Raised when an operation needs a running harness but none is alive.
    """An operation required a running harness but none was alive."""


class HarnessTurnTimeoutError(HarnessError):
    # [CLASS DOC]
    # Raised when a single conversational turn exceeds its wall-clock budget.
    """A single conversational turn exceeded its wall-clock budget."""


class AgenticHarnessAdapter(ABC):
    # [CLASS DOC]
    # Abstract base for turn-based harness adapters. Holds the HarnessConfig in
    # self.config. Subclasses override BUILTIN_TOOL_NAMES (a frozenset of the harness's
    # own tool names, empty by default) so callers can detect name conflicts with
    # injected environment tools. Abstract methods: start, stop, inject_tools,
    # send_message_streaming, is_alive. Concrete: send_message.
    """
    Abstract adapter for a turn-based external agentic harness.

    Subclass this to integrate a harness such as OpenClaw or Claude Code. The
    adapter owns the harness process lifecycle and communication; the harness
    itself is a long-lived process that maintains conversation context across
    turns. Each `send_message()` call is one conversational turn: the harness
    runs its internal ReAct loop and returns when it has a response.

    Distinct from the trainer-side rollout
    [`~openenv.core.harness.rollout.HarnessAdapter`], which drives an entire
    episode in a single call.

    Attributes:
        BUILTIN_TOOL_NAMES (`frozenset[str]`):
            Names of the harness's built-in tools, used to detect conflicts
            with injected environment tools. Concrete adapters override this.
    """

    BUILTIN_TOOL_NAMES: frozenset[str] = frozenset()

    def __init__(self, config: HarnessConfig):
        # [FUNCTION DOC]
        # Stores the given HarnessConfig on self.config. No other side effects.
        self.config = config

    @abstractmethod
    async def start(self, working_directory: str) -> None:
        # [FUNCTION DOC]
        # Abstract. Must launch the harness process in the given working directory.
        # Raises: HarnessStartupError on failure to start or become ready.
        """
        Start the harness process.

        Args:
            working_directory (`str`):
                Path where the harness should operate.

        Raises:
            [`~openenv.core.harness.adapter.HarnessStartupError`]:
                If the process fails to start or become ready in time.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        # [FUNCTION DOC]
        # Abstract. Must stop the process and release resources; must be idempotent
        # (no error if never started or already stopped).
        """
        Stop the harness process and clean up resources.

        Implementations must be idempotent: calling `stop()` on an already
        stopped (or never started) adapter must succeed silently.
        """
        ...

    @abstractmethod
    async def inject_tools(
        self, tools: list[Tool], bridge_url: Optional[str] = None
    ) -> None:
        # [FUNCTION DOC]
        # Abstract. Must make the given tool definitions (and optional bridge URL) known to
        # the harness before start(), via a harness-specific mechanism (config files,
        # CLI flags, env vars).
        """
        Inject environment MCP tool definitions into the harness configuration.

        Called before `start()` so the harness discovers the tools at startup.
        The mechanism is harness-specific (config file, CLI flags, environment
        variables).

        Args:
            tools (`list[Tool]`):
                Conflict-resolved environment tool definitions to inject.
            bridge_url (`str`, *optional*):
                URL of the MCP bridge serving these tools, when the
                environment exposes one.
        """
        ...

    @abstractmethod
    def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        # [FUNCTION DOC]
        # Abstract. Must send one user message and return an async iterator of
        # HarnessEvent objects for the turn; the last event must be TURN_COMPLETE with
        # data keys "response" and optionally "done".
        """
        Send a message and stream intermediate events for one turn.

        Yields events as the harness processes the turn (tool calls, LLM
        chunks, text output). The final event must be a
        [`~openenv.core.harness.events.HarnessEvent`] of type `TURN_COMPLETE`
        whose data carries the turn's `response` and optionally `done`.

        Args:
            message (`str`):
                The user message for this conversational turn.

        Yields:
            [`~openenv.core.harness.events.HarnessEvent`] instances.
        """
        ...

    @abstractmethod
    async def is_alive(self) -> bool:
        # [FUNCTION DOC]
        # Abstract. Must return True if the harness process is still running.
        """Check whether the harness process is still running."""
        ...

    async def send_message(self, message: str) -> HarnessResponse:
        # [FUNCTION DOC]
        # Runs one turn non-streaming: drains send_message_streaming(message) into a list.
        # Returns: HarnessResponse whose response is str(terminal.data["response"]) (default
        # ""), events is the full event list, done is bool(terminal.data["done"]) (default False).
        # Raises: HarnessError if no events were produced or the last event is not TURN_COMPLETE.
        """
        Send a message and collect the complete response for one turn.

        Drains `send_message_streaming()` and assembles a
        [`~openenv.core.harness.events.HarnessResponse`] from the terminal
        `TURN_COMPLETE` event.

        Args:
            message (`str`):
                The user message for this conversational turn.

        Returns:
            [`~openenv.core.harness.events.HarnessResponse`] with the text
            response, all turn events, and the harness's done signal.

        Raises:
            [`~openenv.core.harness.adapter.HarnessError`]:
                If the event stream ends without a `TURN_COMPLETE` event.
        """
        events: list[HarnessEvent] = []
        async for event in self.send_message_streaming(message):
            events.append(event)

        if not events or events[-1].type is not HarnessEventType.TURN_COMPLETE:
            raise HarnessError(
                "harness event stream ended without a TURN_COMPLETE event"
            )

        terminal = events[-1]
        return HarnessResponse(
            response=str(terminal.data.get("response", "")),
            events=events,
            done=bool(terminal.data.get("done", False)),
        )


__all__ = [
    "AgenticHarnessAdapter",
    "HarnessError",
    "HarnessNotRunningError",
    "HarnessStartupError",
    "HarnessTurnTimeoutError",
]
