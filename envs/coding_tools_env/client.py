# [MODULE DOC]
# MCP client for the coding tools environment.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for coding_tools_env."""

from openenv.core.mcp_client import MCPToolClient


class CodingToolsEnv(MCPToolClient):
    # [CLASS DOC]
    # MCPToolClient subclass with no additions; tools are listed and called through the generic MCP client API.
    """MCP client for coding_tools_env."""

    pass
