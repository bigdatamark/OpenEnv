# [MODULE DOC]
# Tests for MCP type definitions (Tool, ToolError, list/call actions and observations, WebSocket
# MCP envelopes, reserved names) and for how deserialize_action and
# deserialize_action_with_preprocessing route MCP payloads versus environment-specific actions.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MCP type definitions and deserialization routing."""

import pytest
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    RESERVED_TOOL_NAMES,
    Tool,
    ToolError,
    ToolErrorType,
    WSMCPMessage,
    WSMCPResponse,
)
from openenv.core.env_server.serialization import (
    deserialize_action,
    deserialize_action_with_preprocessing,
)
from openenv.core.env_server.types import Action
from pydantic import ValidationError


class TestTool:
    # [CLASS DOC]
    # Tests Tool.
    """Tests for the Tool model."""

    def test_tool_creation(self):
        # [FUNCTION DOC]
        # A Tool stores name, description and input schema.
        """Test creating a valid Tool."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "properties" in tool.input_schema

    def test_tool_requires_all_fields(self):
        # [FUNCTION DOC]
        # Missing description/input_schema raises ValidationError.
        """Test that Tool requires name, description, and input_schema."""
        with pytest.raises(ValidationError):
            Tool(name="test")  # Missing description and input_schema

    def test_tool_serialization(self):
        # [FUNCTION DOC]
        # model_dump returns the fields.
        """Test Tool can be serialized to dict."""
        tool = Tool(
            name="echo",
            description="Echo message",
            input_schema={"type": "object"},
        )
        data = tool.model_dump()
        assert data["name"] == "echo"
        assert data["description"] == "Echo message"


class TestToolError:
    # [CLASS DOC]
    # Tests ToolError.
    """Tests for the ToolError model."""

    def test_tool_error_creation(self):
        # [FUNCTION DOC]
        # A ToolError stores type and message.
        """Test creating a ToolError."""
        error = ToolError(
            error_type=ToolErrorType.EXECUTION_ERROR,
            message="Something went wrong",
        )
        assert error.error_type == ToolErrorType.EXECUTION_ERROR
        assert error.message == "Something went wrong"

    def test_all_error_types(self):
        # [FUNCTION DOC]
        # Every ToolErrorType value is accepted.
        """Test all error types can be used."""
        for error_type in ToolErrorType:
            error = ToolError(error_type=error_type, message="test")
            assert error.error_type == error_type


class TestListToolsAction:
    # [CLASS DOC]
    # Tests ListToolsAction.
    """Tests for ListToolsAction."""

    def test_list_tools_action_creation(self):
        # [FUNCTION DOC]
        # type is "list_tools".
        """Test creating a ListToolsAction."""
        action = ListToolsAction()
        assert action.type == "list_tools"

    def test_list_tools_action_metadata(self):
        # [FUNCTION DOC]
        # Metadata is supported.
        """Test ListToolsAction supports metadata."""
        action = ListToolsAction(metadata={"request_id": "123"})
        assert action.metadata["request_id"] == "123"


class TestCallToolAction:
    # [CLASS DOC]
    # Tests CallToolAction.
    """Tests for CallToolAction."""

    def test_call_tool_action_creation(self):
        # [FUNCTION DOC]
        # type is "call_tool" and tool name/arguments are stored.
        """Test creating a CallToolAction."""
        action = CallToolAction(tool_name="echo", arguments={"message": "hello"})
        assert action.type == "call_tool"
        assert action.tool_name == "echo"
        assert action.arguments["message"] == "hello"

    def test_call_tool_action_default_arguments(self):
        # [FUNCTION DOC]
        # arguments default to an empty dict.
        """Test CallToolAction has empty dict as default arguments."""
        action = CallToolAction(tool_name="list")
        assert action.arguments == {}

    def test_call_tool_requires_tool_name(self):
        # [FUNCTION DOC]
        # Missing tool_name raises ValidationError.
        """Test CallToolAction requires tool_name."""
        with pytest.raises(ValidationError):
            CallToolAction()


class TestListToolsObservation:
    # [CLASS DOC]
    # Tests ListToolsObservation.
    """Tests for ListToolsObservation."""

    def test_list_tools_observation_creation(self):
        # [FUNCTION DOC]
        # Stores the given tools.
        """Test creating a ListToolsObservation."""
        tools = [
            Tool(name="echo", description="Echo message", input_schema={}),
            Tool(name="greet", description="Greet user", input_schema={}),
        ]
        obs = ListToolsObservation(tools=tools)
        assert len(obs.tools) == 2
        assert obs.tools[0].name == "echo"
        assert obs.done is False  # Default from Observation

    def test_list_tools_observation_empty(self):
        # [FUNCTION DOC]
        # Accepts an empty tools list.
        """Test ListToolsObservation with no tools."""
        obs = ListToolsObservation(tools=[])
        assert obs.tools == []


class TestCallToolObservation:
    # [CLASS DOC]
    # Tests CallToolObservation.
    """Tests for CallToolObservation."""

    def test_call_tool_observation_success(self):
        # [FUNCTION DOC]
        # A success observation has a result and no error.
        """Test CallToolObservation for successful call."""
        obs = CallToolObservation(
            tool_name="echo",
            result={"message": "hello", "length": 5},
        )
        assert obs.tool_name == "echo"
        assert obs.result["message"] == "hello"
        assert obs.error is None

    def test_call_tool_observation_with_error(self):
        # [FUNCTION DOC]
        # An error observation stores the ToolError.
        """Test CallToolObservation with error."""
        obs = CallToolObservation(
            tool_name="broken_tool",
            result=None,
            error=ToolError(
                error_type=ToolErrorType.EXECUTION_ERROR,
                message="Tool crashed",
            ),
        )
        assert obs.tool_name == "broken_tool"
        assert obs.error is not None
        assert obs.error.error_type == ToolErrorType.EXECUTION_ERROR


class TestWSMCPMessage:
    # [CLASS DOC]
    # Tests WebSocket MCP envelopes.
    """Tests for WebSocket MCP messages."""

    def test_ws_mcp_message_creation(self):
        # [FUNCTION DOC]
        # WSMCPMessage has type "mcp" and carries the JSON-RPC data.
        """Test creating a WSMCPMessage."""
        msg = WSMCPMessage(data={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        assert msg.type == "mcp"
        assert msg.data["method"] == "tools/list"

    def test_ws_mcp_response_creation(self):
        # [FUNCTION DOC]
        # WSMCPResponse has type "mcp" and carries the result.
        """Test creating a WSMCPResponse."""
        response = WSMCPResponse(
            data={"jsonrpc": "2.0", "result": {"tools": []}, "id": 1}
        )
        assert response.type == "mcp"
        assert response.data["result"]["tools"] == []


class TestReservedToolNames:
    # [CLASS DOC]
    # Tests RESERVED_TOOL_NAMES.
    """Tests for reserved tool names."""

    def test_reserved_names_exist(self):
        # [FUNCTION DOC]
        # reset, step, state and close are reserved.
        """Test that reserved names are defined."""
        assert "reset" in RESERVED_TOOL_NAMES
        assert "step" in RESERVED_TOOL_NAMES
        assert "state" in RESERVED_TOOL_NAMES
        assert "close" in RESERVED_TOOL_NAMES

    def test_reserved_names_is_frozenset(self):
        # [FUNCTION DOC]
        # The set is a frozenset.
        """Test that reserved names cannot be modified."""
        assert isinstance(RESERVED_TOOL_NAMES, frozenset)


# Deserialization routing regression tests


class _DummyEnvAction(Action):
    # [CLASS DOC]
    # HELPER: Non-MCP Action subclass with a string field value (default "hello").
    """A non-MCP action class used to simulate env-specific action types."""

    value: str = "hello"


class TestDeserializeActionMCPRouting:
    # [CLASS DOC]
    # Tests deserialize_action MCP routing.
    """MCP action types are routed correctly when action_cls is the base Action."""

    def test_list_tools_with_base_action_cls(self):
        # [FUNCTION DOC]
        # A list_tools payload becomes ListToolsAction when action_cls is Action.
        data = {"type": "list_tools"}
        action = deserialize_action(data, Action)
        assert isinstance(action, ListToolsAction)
        assert action.type == "list_tools"

    def test_list_tools_with_call_tool_action_cls(self):
        # [FUNCTION DOC]
        # A list_tools payload becomes ListToolsAction even when action_cls is CallToolAction.
        data = {"type": "list_tools"}
        action = deserialize_action(data, CallToolAction)
        assert isinstance(action, ListToolsAction)

    def test_call_tool_with_base_action_cls(self):
        # [FUNCTION DOC]
        # A call_tool payload becomes CallToolAction with name and arguments.
        data = {"type": "call_tool", "tool_name": "echo", "arguments": {"msg": "hi"}}
        action = deserialize_action(data, Action)
        assert isinstance(action, CallToolAction)
        assert action.tool_name == "echo"
        assert action.arguments == {"msg": "hi"}

    def test_non_mcp_action_uses_action_cls(self):
        # [FUNCTION DOC]
        # Non-MCP payloads validate into the given action class.
        data = {"value": "world"}
        action = deserialize_action(data, _DummyEnvAction)
        assert isinstance(action, _DummyEnvAction)
        assert action.value == "world"

    def test_call_tool_with_string_arguments_is_rejected_without_preprocessing(self):
        # [FUNCTION DOC]
        # String-encoded arguments raise ValidationError without preprocessing.
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"}',
        }
        with pytest.raises(ValidationError):
            deserialize_action(data, Action)

    def test_invalid_non_mcp_action_raises(self):
        # [FUNCTION DOC]
        # Unknown fields raise ValidationError for a non-MCP action class.
        data = {"nonexistent_field": 123}
        with pytest.raises(ValidationError):
            deserialize_action(data, _DummyEnvAction)


class TestDeserializeActionNonMCPGuard:
    # [CLASS DOC]
    # Tests that MCP routing does not apply when action_cls is a specific non-MCP class.
    """MCP routing does NOT hijack payloads when action_cls is a specific non-MCP class."""

    def test_non_mcp_cls_with_call_tool_type_falls_through(self):
        # [FUNCTION DOC]
        # A call_tool payload validated against a non-MCP class raises ValidationError.
        data = {"type": "call_tool", "tool_name": "echo", "arguments": {}}
        with pytest.raises(ValidationError):
            deserialize_action(data, _DummyEnvAction)

    def test_non_mcp_cls_with_list_tools_type_falls_through(self):
        # [FUNCTION DOC]
        # A list_tools payload validated against a non-MCP class raises ValidationError.
        data = {"type": "list_tools"}
        with pytest.raises(ValidationError):
            deserialize_action(data, _DummyEnvAction)


class TestDeserializeWithPreprocessingMCPRouting:
    # [CLASS DOC]
    # Tests MCP routing in deserialize_action_with_preprocessing.
    """Same MCP routing works in the preprocessing variant, with string argument decoding."""

    def test_list_tools_bypasses_preprocessing(self):
        # [FUNCTION DOC]
        # list_tools becomes ListToolsAction.
        data = {"type": "list_tools"}
        action = deserialize_action_with_preprocessing(data, Action)
        assert isinstance(action, ListToolsAction)

    @pytest.mark.parametrize("action_cls", [Action, CallToolAction])
    def test_call_tool_with_dict_arguments(self, action_cls):
        # [FUNCTION DOC]
        # Parametrized action_cls: dict arguments become a CallToolAction.
        data = {"type": "call_tool", "tool_name": "solve", "arguments": {"x": 1}}
        action = deserialize_action_with_preprocessing(data, action_cls)
        assert isinstance(action, CallToolAction)
        assert action.tool_name == "solve"
        assert action.arguments == {"x": 1}

    @pytest.mark.parametrize("action_cls", [Action, CallToolAction])
    def test_call_tool_with_valid_json_string_arguments(self, action_cls):
        # [FUNCTION DOC]
        # Parametrized action_cls: a JSON object string is decoded into arguments.
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"}',
        }
        action = deserialize_action_with_preprocessing(data, action_cls)
        assert isinstance(action, CallToolAction)
        assert action.tool_name == "echo"
        assert action.arguments == {"message": "hello"}

    def test_call_tool_string_preprocessing_does_not_mutate_input(self):
        # [FUNCTION DOC]
        # Decoding does not modify the input dict.
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"}',
        }
        deserialize_action_with_preprocessing(data, Action)
        assert data["arguments"] == '{"message": "hello"}'

    def test_call_tool_with_malformed_json_string_arguments_is_rejected(self):
        # [FUNCTION DOC]
        # Malformed JSON strings raise ValidationError.
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"',
        }
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, Action)

    def test_call_tool_with_json_array_string_arguments_is_rejected(self):
        # [FUNCTION DOC]
        # A JSON array string raises ValidationError.
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '["hello"]',
        }
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, Action)

    def test_call_tool_with_scalar_json_string_arguments_is_rejected(self):
        # [FUNCTION DOC]
        # A JSON scalar string raises ValidationError.
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '"hello"',
        }
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, Action)

    def test_non_mcp_still_preprocessed(self):
        # [FUNCTION DOC]
        # Non-MCP payloads still validate into the given class.
        data = {"value": "test"}
        action = deserialize_action_with_preprocessing(data, _DummyEnvAction)
        assert isinstance(action, _DummyEnvAction)
        assert action.value == "test"


class TestDeserializeWithPreprocessingNonMCPGuard:
    # [CLASS DOC]
    # Tests the non-MCP guard in the preprocessing variant.
    """Preprocessing variant also guards against MCP hijacking."""

    def test_non_mcp_cls_with_call_tool_type_falls_through(self):
        # [FUNCTION DOC]
        # A call_tool payload against a non-MCP class raises ValidationError.
        data = {"type": "call_tool", "tool_name": "echo", "arguments": {}}
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, _DummyEnvAction)

    def test_non_mcp_cls_with_list_tools_type_falls_through(self):
        # [FUNCTION DOC]
        # A list_tools payload against a non-MCP class raises ValidationError.
        data = {"type": "list_tools"}
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, _DummyEnvAction)
