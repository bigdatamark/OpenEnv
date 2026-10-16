# [MODULE DOC]
# Registers the QED Math MCP tools (get_problem, submit_proof, get_grading_guidelines) on a FastMCP
# server, each delegating to payload methods of a QEDMathEnvironment.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP tool definitions for the QED Math Environment.

Tools are registered on the environment's FastMCP instance inside
QEDMathEnvironment.__init__:
- get_problem(): Return current problem and metadata; reference solution is
    only included for answer-mode evaluation.
- submit_proof(proof): Grade proof via MathProofRubric and return score/reward.
- get_grading_guidelines(): Return the rubric for the current problem.

Reference:
    training/pipelinerl/domains/math/verifier_api.py - Proof verification logic
    envs/echo_env/server/echo_environment.py - MCP tool registration pattern
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import FastMCP

if TYPE_CHECKING:
    from .qed_math_environment import QEDMathEnvironment


def register_mcp_tools(mcp: FastMCP, env: "QEDMathEnvironment") -> None:
    # [FUNCTION DOC]
    # Defines and registers the three tool functions on mcp as closures over env. Returns None.
    # Side effect: mutates the FastMCP tool registry.
    """Register QED-Math MCP tools on a FastMCP instance.

    Args:
        mcp: FastMCP server instance to register tools on.
        env: QEDMathEnvironment instance that serves tool requests.
    """

    @mcp.tool
    def get_problem() -> dict:
        # [FUNCTION DOC]
        # Tool: returns env.get_problem_payload() (a mapping describing the current problem).
        """Get the current problem statement and associated metadata."""
        return env.get_problem_payload()

    @mcp.tool
    async def submit_proof(proof: str) -> dict:
        # [FUNCTION DOC]
        # Async tool: grades the proof via env.submit_proof_payload and returns its result mapping.
        """Submit a proof attempt and return grading output.

        Args:
            proof: The proof text to grade.
        """
        return await env.submit_proof_payload(proof)

    @mcp.tool
    def get_grading_guidelines() -> dict:
        # [FUNCTION DOC]
        # Tool: returns env.get_grading_guidelines_payload().
        """Get grading rubric text for the current problem."""
        return env.get_grading_guidelines_payload()
