# [MODULE DOC]
# Tests for the agentic harness foundation types: HarnessConfig validation and defaults, harness
# events and responses, events_to_metadata, resolve_tool_conflicts, and the default send_message
# implementation that aggregates a streamed turn.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RFC 005 agentic harness foundation types."""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import pytest
from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    AgenticHarnessAdapter,
    events_to_metadata,
    HarnessConfig,
    HarnessError,
    HarnessEvent,
    HarnessEventType,
    HarnessResponse,
    HarnessTransport,
    resolve_tool_conflicts,
)
from pydantic import ValidationError


def make_tool(name: str) -> Tool:
    # [FUNCTION DOC]
    # HELPER: Builds a Tool with the given name, a description and an empty object input schema.
    return Tool(
        name=name,
        description=f"tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


class TestHarnessConfig:
    # [CLASS DOC]
    # Tests HarnessConfig.
    def test_defaults(self):
        # [FUNCTION DOC]
        # Checks default working directory, env vars, transport (stdio), timeouts and optional fields.
        config = HarnessConfig(name="openclaw", command=["openclaw", "run"])
        assert config.working_directory == "/workspace"
        assert config.env_vars == {}
        assert config.transport is HarnessTransport.STDIO
        assert config.mcp_config_path is None
        assert config.startup_timeout_s == 30.0
        assert config.session_timeout_s == 600.0
        assert config.model is None
        assert config.api_key_env_var is None

    def test_extra_fields_forbidden(self):
        # [FUNCTION DOC]
        # Unknown fields raise ValidationError.
        with pytest.raises(ValidationError):
            HarnessConfig(name="x", command=["x"], unexpected="nope")

    def test_empty_command_rejected(self):
        # [FUNCTION DOC]
        # An empty command raises ValidationError.
        with pytest.raises(ValidationError):
            HarnessConfig(name="x", command=[])

    def test_transport_coerces_from_string(self):
        # [FUNCTION DOC]
        # The string "http" coerces to the STREAMABLE_HTTP transport.
        config = HarnessConfig(name="x", command=["x"], transport="http")
        assert config.transport is HarnessTransport.STREAMABLE_HTTP


class TestHarnessEvents:
    # [CLASS DOC]
    # Tests HarnessEvent, events_to_metadata and HarnessResponse.
    def test_timestamp_defaults_to_now(self):
        # [FUNCTION DOC]
        # An event's timestamp defaults to a positive value.
        event = HarnessEvent(type=HarnessEventType.TEXT_OUTPUT)
        assert event.timestamp > 0

    def test_json_round_trip(self):
        # [FUNCTION DOC]
        # An event survives JSON serialization and validation unchanged.
        event = HarnessEvent(
            type=HarnessEventType.TOOL_CALL,
            data={"tool_name": "read_file", "arguments": {"path": "a.py"}},
        )
        restored = HarnessEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_events_to_metadata_is_json_serializable(self):
        # [FUNCTION DOC]
        # events_to_metadata yields JSON-serializable dicts with string type values.
        events = [
            HarnessEvent(type=HarnessEventType.TOOL_CALL, data={"tool_name": "t"}),
            HarnessEvent(
                type=HarnessEventType.TURN_COMPLETE, data={"response": "done"}
            ),
        ]
        payload = events_to_metadata(events)
        assert json.loads(json.dumps(payload)) == payload
        assert payload[0]["type"] == "tool_call"

    def test_response_defaults(self):
        # [FUNCTION DOC]
        # HarnessResponse defaults to empty events and done False.
        response = HarnessResponse(response="hi")
        assert response.events == []
        assert response.done is False


class TestResolveToolConflicts:
    # [CLASS DOC]
    # Tests resolve_tool_conflicts.
    def test_no_conflicts_passthrough(self):
        # [FUNCTION DOC]
        # Non-conflicting tools keep their names and order.
        tools = [make_tool("query_db"), make_tool("lint")]
        resolved = resolve_tool_conflicts(tools, frozenset({"read_file"}))
        assert [t.name for t in resolved] == ["query_db", "lint"]

    def test_conflicting_names_prefixed_and_schema_preserved(self):
        # [FUNCTION DOC]
        # A tool colliding with a built-in is renamed with an env_ prefix and keeps description and schema.
        tool = Tool(
            name="read_file",
            description="env reader",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        resolved = resolve_tool_conflicts([tool], frozenset({"read_file"}))
        assert resolved[0].name == "env_read_file"
        assert resolved[0].description == "env reader"
        assert resolved[0].input_schema == tool.input_schema

    def test_reserved_name_rejected(self):
        # [FUNCTION DOC]
        # A reserved name such as reset raises ValueError.
        with pytest.raises(ValueError, match="reserved"):
            resolve_tool_conflicts([make_tool("reset")], frozenset())

    def test_duplicate_env_names_rejected(self):
        # [FUNCTION DOC]
        # Duplicate environment tool names raise ValueError.
        with pytest.raises(ValueError, match="Duplicate"):
            resolve_tool_conflicts([make_tool("lint"), make_tool("lint")], frozenset())

    def test_prefixed_name_ambiguity_rejected(self):
        # [FUNCTION DOC]
        # Renaming into a name already used by another env tool raises ValueError ("also taken").
        tools = [make_tool("read_file"), make_tool("env_read_file")]
        with pytest.raises(ValueError, match="also taken"):
            resolve_tool_conflicts(tools, frozenset({"read_file"}))

    def test_prefixed_name_collides_with_builtin_rejected(self):
        # [FUNCTION DOC]
        # Renaming into a name that is also a built-in raises ValueError.
        with pytest.raises(ValueError, match="also taken"):
            resolve_tool_conflicts(
                [make_tool("read_file")],
                frozenset({"read_file", "env_read_file"}),
            )

    def test_input_not_mutated(self):
        # [FUNCTION DOC]
        # The input tool object is not mutated.
        tool = make_tool("read_file")
        resolve_tool_conflicts([tool], frozenset({"read_file"}))
        assert tool.name == "read_file"


class StreamingAdapter(AgenticHarnessAdapter):
    # [CLASS DOC]
    # HELPER: Fake adapter whose send_message_streaming yields a fixed list of events; other lifecycle
    # methods are no-ops and it is always alive.
    """Fake adapter yielding a scripted event stream."""

    def __init__(self, config: HarnessConfig, events: list[HarnessEvent]):
        # [FUNCTION DOC]
        # HELPER: Stores the config and the scripted events.
        super().__init__(config)
        self._events = events

    async def start(self, working_directory: str) -> None:
        # [FUNCTION DOC]
        # HELPER: No-op.
        pass

    async def stop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: No-op.
        pass

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        # [FUNCTION DOC]
        # HELPER: No-op.
        pass

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        # [FUNCTION DOC]
        # HELPER: Yields each scripted event.
        for event in self._events:
            yield event

    async def is_alive(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Always returns True.
        return True


class TestSendMessageDefault:
    # [CLASS DOC]
    # Tests the default send_message aggregation.
    @staticmethod
    def _adapter(events: list[HarnessEvent]) -> StreamingAdapter:
        # [FUNCTION DOC]
        # HELPER: Builds a StreamingAdapter with a dummy config and the given events.
        return StreamingAdapter(HarnessConfig(name="fake", command=["fake"]), events)

    async def test_collects_events_and_terminal_turn_complete(self):
        # [FUNCTION DOC]
        # The response text and done flag come from TURN_COMPLETE and all events are collected.
        events = [
            HarnessEvent(type=HarnessEventType.TOOL_CALL, data={"tool_name": "t"}),
            HarnessEvent(type=HarnessEventType.TEXT_OUTPUT, data={"text": "..."}),
            HarnessEvent(
                type=HarnessEventType.TURN_COMPLETE,
                data={"response": "all fixed", "done": True},
            ),
        ]
        response = await self._adapter(events).send_message("go")
        assert response.response == "all fixed"
        assert response.done is True
        assert response.events == events

    async def test_done_defaults_false(self):
        # [FUNCTION DOC]
        # done defaults to False when TURN_COMPLETE omits it.
        events = [
            HarnessEvent(type=HarnessEventType.TURN_COMPLETE, data={"response": "wip"})
        ]
        response = await self._adapter(events).send_message("go")
        assert response.done is False

    async def test_missing_turn_complete_raises(self):
        # [FUNCTION DOC]
        # A stream without TURN_COMPLETE raises HarnessError.
        events = [HarnessEvent(type=HarnessEventType.TEXT_OUTPUT, data={"text": "..."})]
        with pytest.raises(HarnessError, match="TURN_COMPLETE"):
            await self._adapter(events).send_message("go")

    async def test_empty_stream_raises(self):
        # [FUNCTION DOC]
        # An empty stream raises HarnessError.
        with pytest.raises(HarnessError, match="TURN_COMPLETE"):
            await self._adapter([]).send_message("go")
