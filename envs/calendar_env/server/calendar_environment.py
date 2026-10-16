# [MODULE DOC]
# Provides CalendarEnvironment, the conventionally named OpenEnv environment class for this env.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Calendar Environment wrapper.

This file provides the standard environment class name expected in OpenEnv
layouts while reusing the existing MCP environment implementation.
"""

from .openenv_wrapper.mcp_env_environment import MCPEnvironment


class CalendarEnvironment(MCPEnvironment):
    # [CLASS DOC]
    # Pure alias subclass of the MCP-based environment in openenv_wrapper (all behavior inherited).
    """Calendar environment backed by MCP tools."""

    pass


__all__ = ["CalendarEnvironment"]
