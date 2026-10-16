# [MODULE DOC]
# FastAPI app for the chat environment. Reads SYSTEM_PROMPT at import and TOKENIZER_NAME (default
# gpt2) when creating environments; supports in-repo (relative) and standalone imports.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Chat Environment.

This module creates an HTTP server that exposes the ChatEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Note: This server requires a tokenizer to be initialized. The tokenizer
must be specified when starting the server.

Usage:
    # Development (with auto-reload):
    uvicorn envs.chat_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.chat_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.chat_env.server.app
"""

import os

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import ChatAction, ChatObservation
    from .chat_environment import ChatEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import ChatAction, ChatObservation
    from server.chat_environment import ChatEnvironment


# Initialize tokenizer based on environment variable
def get_tokenizer():
    # [FUNCTION DOC]
    # Loads a Hugging Face AutoTokenizer by TOKENIZER_NAME and prints a message. If transformers is not
    # installed, prints a warning, prepends this directory to sys.path and returns the MockTokenizer from
    # test_chat_env.
    # Side effects: network/disk access for the tokenizer, stdout, sys.path mutation.
    """Get tokenizer from environment or use a mock for testing."""
    tokenizer_name = os.environ.get("TOKENIZER_NAME", "gpt2")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Loaded tokenizer: {tokenizer_name}")
        return tokenizer
    except ImportError:
        print(
            "Warning: transformers not installed, using mock tokenizer for testing only"
        )
        # Use mock tokenizer from tests
        import sys
        from pathlib import Path

        # Add parent directory to path to import test utilities
        test_path = Path(__file__).parent
        sys.path.insert(0, str(test_path))

        from test_chat_env import MockTokenizer

        return MockTokenizer()


# Get system prompt from environment
system_prompt = os.environ.get("SYSTEM_PROMPT", None)


# Factory function to create ChatEnvironment instances
def create_chat_environment():
    # [FUNCTION DOC]
    # Factory for create_app: returns a ChatEnvironment with a freshly loaded tokenizer and the system prompt.
    # KNOWN ISSUE: the tokenizer is reloaded for every new environment/session.
    """Factory function that creates ChatEnvironment with tokenizer."""
    tokenizer = get_tokenizer()
    return ChatEnvironment(tokenizer=tokenizer, system_prompt=system_prompt)


# Create the FastAPI app with web interface and README integration
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(
    create_chat_environment, ChatAction, ChatObservation, env_name="chat_env"
)


def main():
    # [FUNCTION DOC]
    # Runs the app with uvicorn on 0.0.0.0:8000 (blocking).
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
