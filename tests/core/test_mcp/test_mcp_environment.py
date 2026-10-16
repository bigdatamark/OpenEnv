# [MODULE DOC]
# Basic tests for MCP types: the reserved tool name set, importability of MCPEnvironment, and
# defaults of MCP action and observation types.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MCPEnvironment base class."""

import pytest
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    RESERVED_TOOL_NAMES,
)


class TestReservedToolNames:
    # [CLASS DOC]
    # Tests RESERVED_TOOL_NAMES.
    """Tests for reserved tool name validation."""

    def test_reserved_names_prevent_reset(self):
        # [FUNCTION DOC]
        # "reset" is reserved.
        """Test that 'reset' is a reserved name."""
        assert "reset" in RESERVED_TOOL_NAMES

    def test_reserved_names_prevent_step(self):
        # [FUNCTION DOC]
        # "step" is reserved.
        """Test that 'step' is a reserved name."""
        assert "step" in RESERVED_TOOL_NAMES

    def test_reserved_names_prevent_state(self):
        # [FUNCTION DOC]
        # "state" is reserved.
        """Test that 'state' is a reserved name."""
        assert "state" in RESERVED_TOOL_NAMES

    def test_reserved_names_prevent_close(self):
        # [FUNCTION DOC]
        # "close" is reserved.
        """Test that 'close' is a reserved name."""
        assert "close" in RESERVED_TOOL_NAMES

    def test_reserved_names_immutable(self):
        # [FUNCTION DOC]
        # The set is immutable (adding raises AttributeError, i.e. it is a frozenset).
        """Test that reserved names cannot be modified."""
        with pytest.raises(AttributeError):
            RESERVED_TOOL_NAMES.add("new_name")


class TestMCPEnvironmentImports:
    # [CLASS DOC]
    # Tests MCPEnvironment imports.
    """Tests that MCPEnvironment can be imported."""

    def test_import_mcp_environment(self):
        # [FUNCTION DOC]
        # MCPEnvironment imports from its module.
        """Test that MCPEnvironment can be imported."""
        from openenv.core.env_server.mcp_environment import MCPEnvironment

        assert MCPEnvironment is not None

    def test_import_from_package(self):
        # [FUNCTION DOC]
        # MCPEnvironment is exported from openenv.core.env_server.
        """Test that MCPEnvironment is exported from the package."""
        from openenv.core.env_server import MCPEnvironment

        assert MCPEnvironment is not None


class TestMCPActions:
    # [CLASS DOC]
    # Tests MCP action types.
    """Tests for MCP action types."""

    def test_list_tools_action_default_type(self):
        # [FUNCTION DOC]
        # ListToolsAction has type "list_tools".
        """Test ListToolsAction has correct default type."""
        action = ListToolsAction()
        assert action.type == "list_tools"

    def test_call_tool_action_stores_values(self):
        # [FUNCTION DOC]
        # CallToolAction stores tool name and arguments.
        """Test CallToolAction stores tool_name and arguments."""
        action = CallToolAction(tool_name="my_tool", arguments={"key": "value"})
        assert action.tool_name == "my_tool"
        assert action.arguments == {"key": "value"}

    def test_call_tool_action_default_arguments(self):
        # [FUNCTION DOC]
        # CallToolAction arguments default to an empty dict.
        """Test CallToolAction has empty default arguments."""
        action = CallToolAction(tool_name="my_tool")
        assert action.arguments == {}


class TestMCPObservations:
    # [CLASS DOC]
    # Tests MCP observation types.
    """Tests for MCP observation types."""

    def test_list_tools_observation_empty_tools(self):
        # [FUNCTION DOC]
        # ListToolsObservation accepts an empty list and defaults done to False.
        """Test ListToolsObservation with empty tools list."""
        obs = ListToolsObservation(tools=[])
        assert obs.tools == []
        assert obs.done is False

    def test_call_tool_observation_success(self):
        # [FUNCTION DOC]
        # CallToolObservation stores tool name and result with no error.
        """Test CallToolObservation for successful call."""
        obs = CallToolObservation(tool_name="test_tool", result="success result")
        assert obs.tool_name == "test_tool"
        assert obs.result == "success result"
        assert obs.error is None
