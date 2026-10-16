# [MODULE DOC]
# FastAPI app for FinQA. Reads FINQA_DATA_PATH (default /app/env/data), FINQA_MAX_STEPS (50) and FINQA_TASK
# (finqa) at import and builds the app with a per-session environment factory and a call-tool action that
# accepts JSON-string arguments.
# envs/finqa_env/server/app.py
"""
FastAPI server for the FinQA environment.

Environment Variables:
    FINQA_DATA_PATH: Path to data directory (default: /app/env/data)
    FINQA_MAX_STEPS: Maximum tool calls per episode (default: 50)
    FINQA_TASK: Task name (default: finqa)
"""

import json
import os
from typing import Any, Dict

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation
from pydantic import field_validator

from .finqa_environment import FinQAEnvironment

DATA_PATH = os.environ.get("FINQA_DATA_PATH", "/app/env/data")
MAX_STEPS = int(os.environ.get("FINQA_MAX_STEPS", "50"))
TASK = os.environ.get("FINQA_TASK", "finqa")


def _env_factory():
    # [FUNCTION DOC]
    # HELPER: Returns a new FinQAEnvironment with the configured data path, max steps and task.
    """Create a new FinQAEnvironment instance for each session."""
    return FinQAEnvironment(
        data_path=DATA_PATH,
        max_steps=MAX_STEPS,
        task=TASK,
    )


class FinQACallToolAction(CallToolAction):
    # [CLASS DOC]
    # CallToolAction whose arguments may also be given as a JSON string (as the web UI sends).
    """CallToolAction that accepts JSON strings for arguments (web UI sends strings)."""

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Before-validator: parses string arguments as JSON; other values pass through.
        # Raises: json.JSONDecodeError for invalid JSON strings.
        if isinstance(v, str):
            return json.loads(v)
        return v


app = create_app(
    _env_factory,
    FinQACallToolAction,
    CallToolObservation,
    reset_observation_cls=Observation,
    env_name="finqa_env",
)


def main():
    # [FUNCTION DOC]
    # Runs the app with uvicorn on 0.0.0.0:8000 (blocking).
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
