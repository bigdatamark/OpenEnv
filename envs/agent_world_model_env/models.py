# [MODULE DOC]
# Pydantic models for AWM actions and observations. The action type is a discriminated union (by the
# "type" field) of the generic MCP ListToolsAction and CallToolAction, validated via a module-level TypeAdapter.
"""
AWM-specific Pydantic models for action and observation types.
"""

from typing import Annotated, Any

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field, field_validator, TypeAdapter


_AWMActionUnion = Annotated[
    ListToolsAction | CallToolAction,
    Field(discriminator="type"),
]
_awm_action_adapter = TypeAdapter(_AWMActionUnion)


class AWMAction(Action):
    # [CLASS DOC]
    # Facade action class: validating or asking for a JSON schema delegates to the ListTools/CallTool union, so
    # validation yields the concrete MCP action instance rather than an AWMAction.
    """Discriminated union action type for AWM.

    model_validate() returns the concrete ListToolsAction or CallToolAction
    (not an AWMAction instance), which is what AWMEnvironment.step() expects.
    """

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Action:  # type: ignore[override]
        # [FUNCTION DOC]
        # Validates a raw object against the action union. Returns: a ListToolsAction or CallToolAction.
        # Extra kwargs are accepted but ignored. Raises: pydantic ValidationError on invalid input.
        return _awm_action_adapter.validate_python(obj)

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        # [FUNCTION DOC]
        # Returns the JSON schema of the action union; kwargs are passed through to the TypeAdapter.
        return _awm_action_adapter.json_schema(**kwargs)


class AWMObservation(Observation):
    # [CLASS DOC]
    # Observation with AWM fields at the top level (reward_type, scenario, task, task_idx, has_verifier, num_tools,
    # tool_name, tool_result, error, warning, verify_result, steps_taken, scenarios, total, trajectory_path,
    # session_dir), all optional. Extra fields are forbidden. Serialization omits None fields by default so the
    # generic MCP client does not misroute payloads based on key presence.
    """
    Observation with AWM-specific fields promoted to top level.
    model_dump() excludes None-valued fields by default so that keys like
    ``tool_name=None`` do not appear in the wire payload.
    This is because the generic MCPToolClient._parse_result() routes observations based on key presence (e.g. ``"tool_name" in obs_data``). We may need to modify the MCPToolClient in the future. Currently, I try to avoid modifying any openenv code.
    """

    model_config = ConfigDict(extra="forbid")

    reward_type: str | None = Field(
        default=None,
        description="Reward classification label for this step/episode outcome",
    )
    scenario: str | None = Field(default=None, description="Current scenario name")
    task: str | None = Field(default=None, description="Current task description")
    task_idx: int | None = Field(default=None, description="Current task index")
    has_verifier: dict | bool | None = Field(
        default=None,
        description="Verifier support info: {sql: bool, code: bool} or legacy bool",
    )

    @field_validator("has_verifier", mode="before")
    @classmethod
    def _convert_bool_to_dict(cls, v: Any) -> dict | None:
        # [FUNCTION DOC]
        # HELPER: Before-validator for has_verifier. None stays None; True becomes a dict with "sql" and "code" both
        # True; False becomes None; any other value passes through unchanged.
        """Convert legacy bool format to new dict format."""
        if v is None:
            return None
        if isinstance(v, bool):
            # Legacy format: True means both modes available (conservative assumption)
            return {"sql": v, "code": v} if v else None
        return v

    num_tools: int | None = Field(
        default=None, description="Number of tools discovered"
    )
    tool_name: str | None = Field(default=None, description="Name of the tool called")
    tool_result: Any = Field(default=None, description="Result from the tool call")
    error: str | None = Field(default=None, description="Error message if any")
    warning: str | None = Field(default=None, description="Warning message if any")
    verify_result: dict | None = Field(
        default=None, description="Verifier output on episode end"
    )
    steps_taken: int | None = Field(
        default=None, description="Steps taken in this episode"
    )
    scenarios: list | None = Field(
        default=None, description="List of all scenarios (from __list_scenarios__)"
    )
    total: int | None = Field(default=None, description="Total number of scenarios")
    trajectory_path: str | None = Field(
        default=None, description="Path to saved trajectory JSON file"
    )
    session_dir: str | None = Field(
        default=None, description="Session directory path (when keep_session=True)"
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Same as the base model_dump but defaults exclude_none to True unless the caller sets it.
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class AWMListToolsObservation(ListToolsObservation):
    # [CLASS DOC]
    # ListToolsObservation (tool list) with an extra optional "error" string; unknown fields are forbidden.
    """ListToolsObservation with AWM error field promoted to top level."""

    model_config = ConfigDict(extra="forbid")

    error: str | None = Field(default=None, description="Error message if any")
