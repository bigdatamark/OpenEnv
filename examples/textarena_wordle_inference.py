#!/usr/bin/env python3
# [MODULE DOC]
# Example script: an LLM (via the HF router with the OpenAI client) plays TextArena
# Wordle-v0 inside a locally started textarena-env Docker container. API key comes
# from API_KEY or HF_TOKEN; model, turn limit and verbosity are module constants.
# KNOWN ISSUE: docstring says examples/wordle_inference.py and the DeepSeek Terminus
# model, but the file name differs and MODEL is gpt-oss-120b.
"""Play TextArena Wordle with a hosted LLM via Hugging Face Inference Providers.

This script mirrors the structure of the Kuhn Poker inference sample but targets
the Wordle environment. We deploy the generic TextArena server (wrapped in
OpenEnv) inside a local Docker container and query a single hosted model using
the OpenAI-compatible API provided by Hugging Face's router.

Prerequisites
-------------
1. Build the TextArena Docker image::

       docker build -f envs/textarena_env/server/Dockerfile -t textarena-env:latest .

2. Set your Hugging Face token::

       export HF_TOKEN=your_token_here

3. Run this script::

       python examples/wordle_inference.py

By default we ask the DeepSeek Terminus model to play ``Wordle-v0``. Adjust the
``MODEL`` constant if you'd like to experiment with another provider-compatible
model.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List

from openai import OpenAI

from textarena_env import TextArenaAction, TextArenaEnv
from textarena_env.models import TextArenaMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MODEL = "openai/gpt-oss-120b:novita"
MAX_TURNS = 8
VERBOSE = True

SYSTEM_PROMPT = (
    "You are an expert Wordle solver."
    " Always respond with a single guess inside square brackets, e.g. [crane]."
    " Use lowercase letters, exactly one five-letter word per reply."
    " Reason about prior feedback before choosing the next guess."
    " Words must be 5 letters long and real English words."
    " Do not not include any other text in your response."
    " Do not repeat the same guess twice."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_history(messages: Iterable[TextArenaMessage]) -> str:
    # [FUNCTION DOC]
    # HELPER: renders messages as lines "[category] content" (category defaults to
    # MESSAGE) joined by newlines.
    """Convert TextArena message history into plain text for the model."""

    lines: List[str] = []
    for message in messages:
        tag = message.category or "MESSAGE"
        lines.append(f"[{tag}] {message.content}")
    return "\n".join(lines)


def extract_guess(text: str) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the first bracketed five-letter word in text, lowercased. Otherwise
    # strips non-letters, and returns the first 5 letters in brackets if at least 5
    # remain, else the placeholder "[dunno]".
    """Return the first Wordle-style guess enclosed in square brackets."""

    match = re.search(r"\[[A-Za-z]{5}\]", text)
    if match:
        return match.group(0).lower()
    # Fallback: remove whitespace and ensure lowercase, then wrap
    cleaned = re.sub(r"[^a-zA-Z]", "", text).lower()
    if len(cleaned) >= 5:
        return f"[{cleaned[:5]}]"
    return "[dunno]"


def make_user_prompt(prompt_text: str, messages: Iterable[TextArenaMessage]) -> str:
    # [FUNCTION DOC]
    # HELPER: builds the user prompt from the current prompt text, formatted message
    # history and an instruction to reply with a bracketed guess.
    """Combine the TextArena prompt and feedback history for the model."""

    history = format_history(messages)
    return (
        f"Current prompt:\n{prompt_text}\n\n"
        f"Conversation so far:\n{history}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------

def play_wordle(env: TextArenaEnv, client: OpenAI) -> None:
    # [FUNCTION DOC]
    # Plays one game: resets the env, then for up to MAX_TURNS turns (stopping when done)
    # asks the model for a guess (system + user prompt, temperature 0.7), parses it and
    # steps the env, printing feedback when VERBOSE; prints final reward and done.
    # Side effects: network calls, stdout.
    # KNOWN ISSUE: message content may be None, in which case calling strip raises.
    result = env.reset()
    observation = result.observation

    if VERBOSE:
        print("📜 Initial Prompt:\n" + observation.prompt)

    for turn in range(1, MAX_TURNS + 1):
        if result.done:
            break

        user_prompt = make_user_prompt(observation.prompt, observation.messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.7,
        )

        raw_output = response.choices[0].message.content.strip()
        guess = extract_guess(raw_output)

        if VERBOSE:
            print(f"\n🎯 Turn {turn}: model replied with -> {raw_output}")
            print(f"   Parsed guess: {guess}")

        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

        if VERBOSE:
            print("   Feedback messages:")
            for message in observation.messages:
                print(f"     [{message.category}] {message.content}")

    print("\n✅ Game finished")
    print(f"   Reward: {result.reward}")
    print(f"   Done: {result.done}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # [FUNCTION DOC]
    # Entry point: raises SystemExit if no API key; creates the OpenAI client, starts the
    # textarena-env:latest container configured for single-player Wordle-v0 on port 8000,
    # plays one game and always closes the env.
    if not API_KEY:
        raise SystemExit("HF_TOKEN (or API_KEY) must be set to query the model.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = TextArenaEnv.from_docker_image(
        "textarena-env:latest",
        env_vars={
            "TEXTARENA_ENV_ID": "Wordle-v0",
            "TEXTARENA_NUM_PLAYERS": "1",
        },
        ports={8000: 8000},
    )

    try:
        play_wordle(env, client)
    finally:
        env.close()


if __name__ == "__main__":
    main()


