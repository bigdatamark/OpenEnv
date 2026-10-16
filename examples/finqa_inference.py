#!/usr/bin/env python3
# [MODULE DOC]
# Example running an LLM (any OpenAI-compatible API; defaults to the Hugging Face router) on the FinQA environment
# started from a Docker image: discovers MCP tools, converts them to OpenAI function format, plays MAX_EPISODES
# episodes with forced tool calls until the environment reports done, and prints accuracy and average steps.
# Side effects: sys.path change, Docker container, network calls, stdout.
"""Play FinQA with an LLM via any OpenAI-compatible API.

FinQA is a financial question-answering benchmark that evaluates LLMs on their
ability to answer complex financial questions using tool calls on SEC 10-K filing data.

Prerequisites
-------------
1. Build the FinQA Docker image::

       docker build -f envs/finqa_env/server/Dockerfile -t finqa-env:latest .

2. Set your API key::

       export HF_TOKEN=your_token_here

3. Run this script::

       python examples/finqa_inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.finqa_env import CallToolAction, FinQAEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b:novita")

MAX_EPISODES = 3
MAX_TOKENS = 2048
VERBOSE = True

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC 10-K filings.

Think and reason step by step. Iteratively gather data using the available tools until you have enough information to answer the question.

When submitting your final answer:
- Provide ONLY the numerical value. No explanations, units, or LaTeX formatting.
- Always express percentages, growth rates, and percentage point differences as decimal ratios by dividing by 100 (e.g., 22% → 0.22, -8.9% → -0.089, a 4.5 percentage point difference → 0.045).
- Submit numbers exactly as they appear in the query results. Do not convert units (e.g., if the table shows values in millions, submit the number as-is, not multiplied out).
- For multi-year answers, use: year: value, year: value (e.g., 2022: 0.933, 2023: 0.930, 2024: 0.931)
- For year-over-year changes, use: year to year: value (e.g., 2022 to 2023: 0.189, 2023 to 2024: 0.025)
- For single values, just submit the number (e.g., 0.895 or -77 or 63)
- If the question is yes/no, answer Yes or No
"""


def _tools_to_openai_format(tools) -> List[dict]:
    # [FUNCTION DOC]
    # HELPER: Converts MCP tool objects into OpenAI function tool definitions, keeping only each property's type (default
    # "string") and description, plus the required list. Returns the list of definitions.
    """Convert MCP tools to OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        properties = {}
        required = []
        if tool.inputSchema and "properties" in tool.inputSchema:
            for name, schema in tool.inputSchema["properties"].items():
                properties[name] = {
                    "type": schema.get("type", "string"),
                    "description": schema.get("description", ""),
                }
            required = tool.inputSchema.get("required", [])

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return openai_tools


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------


async def play_finqa_episode(
    env: FinQAEnv,
    client: OpenAI,
    tools: List[dict],
    episode_num: int,
) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Plays one episode: resets, reads question and company from metadata, then loops until done: requests a completion with
    # tool_choice "required"; without tool calls submits the text (or "unknown") as submit_answer; otherwise uses the first
    # tool call (arguments JSON-decoded). Records the assistant tool call, replaces unknown tool names with submit_answer
    # "unknown", steps the env with a CallToolAction, and appends the tool result to the history if not done.
    # Returns: dict with episode number, company, question, reward (0.0 if None) and step count.
    # KNOWN ISSUE: the reset result is used directly as an observation (metadata/done), whereas the env client's reset elsewhere returns a step result wrapping the observation.
    # KNOWN ISSUE: invalid JSON in tool-call arguments raises JSONDecodeError and aborts the run.
    """Play a single FinQA episode."""
    tool_names = [t["function"]["name"] for t in tools]

    obs = await env.reset()
    question = obs.metadata.get("question", "")
    company = obs.metadata.get("company", "")

    if VERBOSE:
        print(f"\n{'=' * 60}")
        print(f"Episode {episode_num}")
        print(f"{'=' * 60}")
        print(f"Company: {company}")
        print(f"Question: {question}")

    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Company: {company}\nQuestion: {question}"},
    ]
    step_count = 0

    while not obs.done:
        step_count += 1
        if VERBOSE:
            print(f"\n--- Step {step_count} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=chat_history,
            tools=tools,
            tool_choice="required",
            max_completion_tokens=MAX_TOKENS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            tool_name = "submit_answer"
            tool_args = {"answer": message.content or "unknown"}
            tool_call_id = "none"
        else:
            tool_call_obj = message.tool_calls[0]
            tool_name = tool_call_obj.function.name
            tool_args = json.loads(tool_call_obj.function.arguments)
            tool_call_id = tool_call_obj.id

        chat_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                            if tool_call_id == "none"
                            else tool_call_obj.function.arguments,
                        },
                    }
                ],
            }
        )

        if VERBOSE:
            print(f"Tool: {tool_name}({json.dumps(tool_args)})"[:100])

        if tool_name not in tool_names:
            tool_name = "submit_answer"
            tool_args = {"answer": "unknown"}

        # Use step() with CallToolAction to get the full observation (done, reward)
        action = CallToolAction(tool_name=tool_name, arguments=tool_args)
        step_result = await env.step(action)
        obs = step_result.observation
        result_text = obs.result if hasattr(obs, "result") else str(obs.metadata)

        if not obs.done:
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result_text) or "No result",
                }
            )

        if VERBOSE:
            result_preview = str(result_text)[:200]
            print(f"Result: {result_preview}...")

    reward = obs.reward or 0.0

    if VERBOSE:
        outcome = "CORRECT" if reward > 0 else "INCORRECT"
        print(f"\nResult: {outcome}")
        print(f"  Reward: {reward}")

    return {
        "episode": episode_num,
        "company": company,
        "question": question,
        "reward": reward,
        "steps": step_count,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def async_main() -> None:
    # [FUNCTION DOC]
    # Requires API_KEY or HF_TOKEN (SystemExit otherwise); starts the env from the finqa-env:latest image, lists and converts
    # tools, plays the episodes and prints a summary.
    if not API_KEY:
        raise SystemExit("API_KEY (or HF_TOKEN) must be set to query the model.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with FinQAEnv.from_docker_image("finqa-env:latest") as env:
        # Discover tools via MCP and convert to OpenAI format
        mcp_tools = await env.list_tools()
        tools = _tools_to_openai_format(mcp_tools)

        if VERBOSE:
            tool_names = [t["function"]["name"] for t in tools]
            print(f"API: {API_BASE_URL}")
            print(f"Model: {MODEL}")
            print(f"Tools: {tool_names}")

        results = []
        for episode_num in range(1, MAX_EPISODES + 1):
            episode_result = await play_finqa_episode(env, client, tools, episode_num)
            results.append(episode_result)

        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        correct = sum(1 for r in results if r["reward"] > 0)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        print(f"Episodes: {len(results)}")
        print(
            f"Correct: {correct}/{len(results)} ({100 * correct / len(results):.1f}%)"
        )
        print(f"Average steps: {avg_steps:.1f}")


def main() -> None:
    # [FUNCTION DOC]
    # Synchronous entry point running async_main with asyncio.run.
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
