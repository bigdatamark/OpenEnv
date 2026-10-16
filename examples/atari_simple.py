#!/usr/bin/env python3
# [MODULE DOC]
# Minimal example running a random-action Atari episode. Starts the Atari env from a Docker image, resets, takes up to
# 100 random legal actions printing progress every 10 steps, prints total reward and the environment state, then closes.
# Side effects: prepends the repo src directory to sys.path, launches a Docker container, prints to stdout.
"""
Simple example demonstrating Atari Environment usage.

This example shows how to:
1. Connect to an Atari environment
2. Reset the environment
3. Take random actions
4. Process observations

Usage:
    # First, start the server:
    python -m envs.atari_env.server.app

    # Then run this script:
    python examples/atari_simple.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atari_env import AtariEnv, AtariAction
# import envs
# print(envs.__path__)

async def main():
    # [FUNCTION DOC]
    # Executes the episode described in the module doc and always closes the environment (stopping the container) at the end.
    # KNOWN ISSUE: the progress print formats result.reward with a float format, which fails if reward is None.
    """Run a simple Atari episode."""
    # Connect to the Atari environment server
    print("Connecting to Atari environment...")
    env = await AtariEnv.from_docker_image(
        "ghcr.io/huggingface/openenv-atari-env:latest"
    )

    try:
        # Reset the environment
        print("\nResetting environment...")
        result = await env.reset()
        print(f"Screen shape: {result.observation.screen_shape}")

        print(f"Legal actions: {result.observation.legal_actions}")
        print(f"Lives: {result.observation.lives}")

        # Run a few steps with random actions
        print("\nTaking random actions...")
        episode_reward = 0
        steps = 0

        for step in range(100):
            # Random action
            action_id = np.random.choice(result.observation.legal_actions)
            action_id = int(action_id)

            # Take action
            result = await env.step(AtariAction(action_id=action_id))

            episode_reward += result.reward or 0
            steps += 1

            # Print progress
            if step % 10 == 0:
                print(
                    f"Step {step}: reward={result.reward:.2f}, "
                    f"lives={result.observation.lives}, done={result.done}"
                )

            if result.done:
                print(f"\nEpisode finished after {steps} steps!")
                break

        print(f"\nTotal episode reward: {episode_reward:.2f}")

        # Get environment state
        state = await env.state()
        print("\nEnvironment state:")
        print(f"  Game: {state.game_name}")
        print(f"  Episode: {state.episode_id}")
        print(f"  Steps: {state.step_count}")
        print(f"  Obs type: {state.obs_type}")

    finally:
        # Cleanup
        print("\nClosing environment...")
        await env.close()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
