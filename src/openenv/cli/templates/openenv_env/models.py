# [MODULE DOC]
# Template data models (pydantic) for a generated echo-style environment.
# SPDX-License-Identifier: BSD-3-Clause

"""
Data models for the __ENV_TITLE_NAME__ Environment.

The __ENV_NAME__ environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class __ENV_CLASS_NAME__Action(Action):
    # [CLASS DOC]
    # Action carrying a required message string to echo.
    """Action for the __ENV_TITLE_NAME__ environment - just a message to echo."""

    message: str = Field(..., description="Message to echo back")


class __ENV_CLASS_NAME__Observation(Observation):
    # [CLASS DOC]
    # Observation carrying the echoed message (default "") and its length (default
    # 0), plus the base Observation fields (done, reward, metadata).
    """Observation from the __ENV_TITLE_NAME__ environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
