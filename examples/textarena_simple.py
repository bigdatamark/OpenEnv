#!/usr/bin/env python3
# [MODULE DOC]
# Quickstart example: plays Wordle on a hosted TextArena environment Space by sending
# six fixed guesses and printing the feedback and final state.
# SPDX-License-Identifier: BSD-3-Clause

"""Quickstart example for the generic TextArena environment."""
from __future__ import annotations

from textarena_env import TextArenaEnv, TextArenaAction


def main() -> None:
    # [FUNCTION DOC]
    # Entry point: connects to the hosted Wordle Space, resets, sends each fixed guess
    # until done, prints feedback text found after "Feedback:" in messages, the reward
    # and a state snapshot. Exceptions are caught and printed with setup hints; the env
    # is always closed. Side effects: network, stdout.
    print("=" * 60)
    print("💬 TextArena Hello World - Wordle-v0")
    print("=" * 60)

    # TODO: move to openenv org
    env = TextArenaEnv(base_url="https://burtenshaw-wordle.hf.space")

    try:
        print("\n📍 Resetting environment...")
        result = env.reset()
        print(f"   Prompt:\n{result.observation.prompt}\n")

        # Wordle guesses - common starting words
        guesses = ["[crane]", "[slate]", "[audio]", "[pride]", "[money]", "[ghost]"]

        for step, guess in enumerate(guesses):
            print(f"🎯 Step {step + 1}: sending guess {guess}")
            result = env.step(TextArenaAction(message=guess))

            # Show the feedback
            for message in result.observation.messages:
                # Extract just the feedback part
                content = message.content
                if "Feedback:" in content:
                    feedback_part = content.split("Feedback:")[-1].strip()
                    print(f"   Feedback:\n{feedback_part}")

            if result.done:
                if result.reward and result.reward > 0:
                    print("\n🎉 You won!")
                break

        print("\n✅ Episode finished!")
        print(f"   Reward: {result.reward}")
        print(f"   Done: {result.done}")

        state = env.state()
        print("\n📊 Server State Snapshot:")
        print(f"   Episode ID: {state.episode_id}")
        print(f"   Step count: {state.step_count}")
        print(f"   Env ID: {state.env_id}")

    except Exception as exc:  # pragma: no cover - demonstration script
        print(f"\n❌ Error: {exc}")
        print("\nMake sure the server is running:")
        print("  cd envs/textarena_env && source .venv/bin/activate")
        print("  python -m uvicorn server.app:app --reload")

    finally:
        env.close()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()
