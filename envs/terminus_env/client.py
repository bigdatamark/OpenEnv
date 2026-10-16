# [MODULE DOC]
# Client for the Terminus environment (plain MCP tool client).
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the Terminus environment."""

from openenv.core.mcp_client import MCPToolClient


class TerminusEnv(MCPToolClient):
    # [CLASS DOC]
    # MCPToolClient subclass with no additions; use call_tool("terminal", ...) to run commands or submit.
    """MCP client for calling the Terminus single-rollout tool."""

    pass
