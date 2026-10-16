#!/usr/bin/env python3
# [MODULE DOC]
# Example script: demonstrates the RLM (recursive language model) loop of repl_env
# on a toy word-counting task, using HF InferenceClient with a Qwen3.5 model
# (overridable via REPL_LLM_MODEL env var; HF_TOKEN from env).
"""
REPL Environment with LLM Integration.

Demonstrates the RLM (Recursive Language Model) paradigm using OpenEnv's repl_env:
1. LLM generates Python code to solve a task
2. Code is executed in the sandboxed REPL
3. LLM sees the output and generates more code
4. Process repeats until FINAL() is called

Usage:
    python examples/repl_with_llm.py
"""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient
from repl_env import LocalRLMRunner, RLM_SYSTEM_PROMPT

# ============== CONFIGURATION ==============
MODEL_NAME = os.environ.get("REPL_LLM_MODEL", "Qwen/Qwen3.5-9B")
MAX_ITERATIONS = 10
# ===========================================

HF_TOKEN = os.environ.get("HF_TOKEN")


def create_chat_fn():
    # [FUNCTION DOC]
    # Builds the chat callable for the RLM runner.
    # Returns: function chat_fn(messages, model=None) -> str.
    # Side effects: creates an InferenceClient (300 s timeout).
    """Create the chat function with Qwen3.5 model card recommended params."""
    client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN, timeout=300)

    def chat_fn(messages: list[dict], model: str | None = None) -> str:
        # [FUNCTION DOC]
        # Calls HF chat completions with Qwen3.5 non-thinking sampling params (max_tokens
        # 2048, temperature 0.6, top_p 0.95, top_k 20, thinking disabled via chat template
        # kwargs) using model or MODEL_NAME; returns first choice's message content.
        response = client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=messages,
            max_tokens=2048,
            # Qwen3.5 non-thinking mode for precise coding tasks (from model card)
            temperature=0.6,
            top_p=0.95,
            presence_penalty=0.0,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                # Disable thinking mode — the RLM loop is the reasoning mechanism
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content

    return chat_fn


def main():
    # [FUNCTION DOC]
    # Entry point: runs LocalRLMRunner (max depth 3, MAX_ITERATIONS) on a short fixed
    # context with a word-count task and prints the final answer and iteration count.
    # Side effects: network calls, stdout.
    print("=" * 60)
    print("REPL Environment with LLM Integration (Qwen)")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")

    context = """
    The quick brown fox jumps over the lazy dog.
    This is a sample text for testing the REPL environment.
    It contains multiple sentences that we can analyze.
    The RLM paradigm allows models to process data programmatically.
    """
    task = "Count the total number of words in the context"

    print(f"Task: {task}")
    print(f"Context: {context[:100]}...")

    chat_fn = create_chat_fn()
    runner = LocalRLMRunner(
        chat_fn,
        system_prompt=RLM_SYSTEM_PROMPT,
        max_iterations=MAX_ITERATIONS,
        max_depth=3,
        verbose=True,
    )
    result = runner.run(context, task)

    print(f"\n{'=' * 60}")
    print(f"Final Result: {result.final_answer}")
    print(f"Iterations: {result.iterations}")
    print("=" * 60)


if __name__ == "__main__":
    main()
