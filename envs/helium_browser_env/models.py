# [MODULE DOC]
# Pydantic models for the browser environment: action (with cross-field
# validation), observation and state.
"""OpenEnv models for browser actions, observations, and state."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator


class BrowserAction(Action):
    # [CLASS DOC]
    # Browser action. op is one of click, type, key, scroll, back, wait, finish.
    # x in [0, 800) and y in [0, 600) are pixel coordinates (required for click);
    # text up to 2000 chars (required non-empty for type, and must be a supported
    # key name for key); dy is the scroll amount in pixels within [-1200, 1200].
    op: Literal["click", "type", "key", "scroll", "back", "wait", "finish"]
    x: int | None = Field(default=None, ge=0, lt=800)
    y: int | None = Field(default=None, ge=0, lt=600)
    text: str = Field(default="", max_length=2000)
    dy: int = Field(default=0, ge=-1200, le=1200)

    @model_validator(mode="after")
    def validate_arguments(self):
        # [FUNCTION DOC]
        # Pydantic after-validator enforcing the per-op requirements: click needs x and
        # y, type needs non-empty text, key needs text in the supported key set (ENTER,
        # TAB, ESCAPE, BACKSPACE, CTRL+A, arrow keys). Returns self.
        # Raises: ValueError on violation.
        if self.op == "click" and (self.x is None or self.y is None):
            raise ValueError("click needs x and y")
        if self.op == "type" and not self.text:
            raise ValueError("type needs text")
        if self.op == "key" and self.text not in (
            "ENTER",
            "TAB",
            "ESCAPE",
            "BACKSPACE",
            "CTRL+A",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
        ):
            raise ValueError("unsupported key")
        return self


class BrowserObservation(Observation):
    # [CLASS DOC]
    # Observation: base64 PNG screenshot, current URL, error code string (default
    # ""), plus controls (list of dicts) and page_text fields.
    # KNOWN ISSUE: controls and page_text are never populated by the server.
    screenshot: str
    url: str
    error: str = ""
    controls: list[dict] = Field(default_factory=list)
    page_text: str = ""


class BrowserState(State):
    # [CLASS DOC]
    # State with no extra fields (episode_id and step_count from the base).
    pass
