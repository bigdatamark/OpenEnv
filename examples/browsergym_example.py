# [MODULE DOC]
# Inference example for BrowserGym MiniWoB "click-test": a vision-language model (via the Hugging Face router,
# OpenAI-compatible API, token from HF_TOKEN or API_KEY) chooses one BrowserGym action string per step from a text
# prompt (goal, URL, recent history, clickable element IDs) plus a PNG screenshot. Runs up to MAX_STEPS steps against
# a Docker-launched environment, printing rewards and errors. Side effects: Docker container, network calls, stdout.
"""BrowserGym MiniWoB example with Qwen deciding the next action.

This is an inference example for the BrowserGym environment. It uses the OpenAI
client and a vision language model to decide the next action. We use Hugging Face
Inference Providers API to access the model, but you can use any other provider that
is compatible with the OpenAI API.

Prerequisites:
- (Optional) Export the MiniWoB URL if you are hosting the tasks yourself
  (must include the `/miniwob/` suffix); the BrowserGym Docker image now
  serves the MiniWoB bundle internally on port 8888.
- Export your Hugging Face token for the router:
    `export HF_TOKEN=your_token_here`

Usage:
    python examples/browsergym_example.py
"""

import os
import re
import base64
import textwrap
from io import BytesIO
from typing import List, Optional, Dict

from openai import OpenAI
import numpy as np
from PIL import Image

from browsergym_env import BrowserGymAction, BrowserGymEnv

API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct:novita"
MAX_STEPS = 8
MAX_DOM_CHARS = 3500
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = "noop()"

DEBUG = True
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a web browser through BrowserGym.
    Reply with exactly one action string.
    The action must be a valid BrowserGym command such as:
    - noop()
    - click('<BID>')
    - type('selector', 'text to enter')
    - fill('selector', 'text to enter')
    - send_keys('Enter')
    - scroll('down')
    Use single quotes around string arguments.
    When clicking, use the BrowserGym element IDs (BIDs) listed in the user message.
    If you are unsure, respond with noop().
    Do not include explanations or additional text.
    """
).strip()


def build_history_lines(history: List[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the last 4 history lines joined by newlines, or "None" if history is empty.
    if not history:
        return "None"
    return "\n".join(history[-4:])


def extract_screenshot_uri(observation) -> Optional[str]:
    # [FUNCTION DOC]
    # HELPER: Converts the observation's screenshot pixel array to a PNG and returns it as a base64 data URI, or None if there is no screenshot.
    if observation.screenshot is None:
        return None
    screen_array = np.array(observation.screenshot, dtype=np.uint8)
    image = Image.fromarray(screen_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{data_uri}"


def extract_clickable_elements(observation) -> List[Dict[str, str]]:
    # [FUNCTION DOC]
    # Reads metadata["browsergym_obs"]["extra_element_properties"] from the observation and returns, sorted by id string,
    # a list of dicts with "bid" and a comma-joined "bbox" string (or "?") for elements flagged clickable.
    # KNOWN ISSUE: joining the bbox with a string join fails if the coordinates are numbers rather than strings.
    """Collect BrowserGym element IDs that can be clicked."""

    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}

    clickables: List[Dict[str, str]] = []
    for bid, props in extra_props.items():
        if not props.get("clickable"):
            continue

        bbox = props.get("bbox") or []
        bbox_str = ", ".join(bbox) if bbox else "?"
        clickables.append(
            {
                "bid": str(bid),
                "bbox": bbox_str,
            }
        )

    # Keep a stable ordering for readability
    clickables.sort(key=lambda item: item["bid"])
    return clickables


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: Builds the per-step user prompt text with step number, goal, URL, history, last-action-error flag and the
    # list of clickable element IDs. Returns the dedented string.
    goal = observation.goal or "(not provided)"
    url = observation.url or "(unknown)"
    error_note = "Yes" if observation.last_action_error else "No"

    clickables = extract_clickable_elements(observation)
    if clickables:
        actions_hint = "\n".join(
            f"    - {item['bid']} (bbox: {item['bbox']})" for item in clickables
        )
    else:
        actions_hint = "    (none detected)"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Current URL: {url}
        Previous steps:
        {build_history_lines(history)}
        Last action error: {error_note}

        Available clickable element IDs: {actions_hint}

        Reply with exactly one BrowserGym action string.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    # [FUNCTION DOC]
    # Extracts an action string from model text: for each non-empty line, strips an "action:"/"next action:" prefix and
    # returns the first function-call-like match with whitespace collapsed; otherwise searches the whole text; falls back
    # to "noop()" when nothing matches or the text is empty.
    # KNOWN ISSUE: an inline comment mentions falling back to the single detected element ID for natural-language clicks, but no such logic exists.
    if not response_text:
        return FALLBACK_ACTION

    # Prefer the first line that looks like an action string
    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            # Collapse internal whitespace
            action = re.sub(r"\s+", " ", action)
            # If the model tried to click by natural-language description while we
            # only exposed numeric BrowserGym IDs, fallback to the single detected ID.
            return action

    # Fall back to searching the whole response
    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        action = re.sub(r"\s+", " ", action)
        return action

    return FALLBACK_ACTION


def main() -> None:
    # [FUNCTION DOC]
    # Runs the episode: creates the OpenAI client and env, resets, and per step builds the multimodal messages, queries
    # the model (any exception uses the fallback action), parses and executes the action, records history and prints the
    # reward; stops on done or after MAX_STEPS. Always closes the env.
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = BrowserGymEnv.from_docker_image(
        image="browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": "click-test",
        },
    )

    history: List[str] = []

    try:
        result = env.reset()
        observation = result.observation
        print(f"Episode goal: {observation.goal}")

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                print("Environment signalled done. Stopping early.")
                break

            user_prompt = build_user_prompt(step, observation, history)
            user_content = [{"type": "text", "text": user_prompt}]
            screenshot_uri = extract_screenshot_uri(observation)
            if screenshot_uri:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_uri},
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            # pylint: disable=broad-except
            except Exception as exc:  # noqa: BLE001
                failure_msg = f"Model request failed ({exc}). Using fallback action."
                print(failure_msg)
                response_text = FALLBACK_ACTION

            action_str = parse_model_action(response_text)
            print(f"Step {step}: model suggested -> {action_str}")

            result = env.step(BrowserGymAction(action_str=action_str))
            observation = result.observation

            reward = result.reward or 0.0
            error_flag = " ERROR" if observation.last_action_error else ""
            history_line = (
                f"Step {step}: {action_str} -> reward {reward:+.2f}{error_flag}"
            )
            history.append(history_line)
            print(
                "  Reward: "
                f"{reward:+.2f} | Done: {result.done} | Last action error: "
                f"{observation.last_action_error}"
            )

            if result.done:
                print("Episode complete.")
                break

        else:
            print(f"Reached max steps ({MAX_STEPS}).")

    finally:
        env.close()


if __name__ == "__main__":
    main()
