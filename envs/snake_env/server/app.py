# [MODULE DOC]
# Builds the FastAPI application for the Snake environment at import time (module-level "app") and
# provides a main() entry point that serves it with uvicorn on 0.0.0.0:8000.
# Side effects: importing the module constructs the app (and possibly a SnakeEnvironment instance).
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Snake Environment.

This module creates an HTTP server that exposes the SnakeEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

import inspect
import os

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app

    from ..models import SnakeAction, SnakeObservation
    from .snake_environment import SnakeEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import SnakeAction, SnakeObservation
    from openenv.core.env_server.http_server import create_app
    from server.snake_environment import SnakeEnvironment


def _create_snake_app():
    # [FUNCTION DOC]
    # HELPER: builds the app while staying compatible with versions of create_app that take either an
    # environment factory or an environment instance. Inspects the annotation of create_app's first
    # parameter; if it mentions "Environment" but not "Callable", passes a new SnakeEnvironment instance,
    # otherwise passes the class as a factory (also the fallback if introspection fails).
    # Uses env_name "snake_env" and max_concurrent_envs from the MAX_CONCURRENT_ENVS env var (default 8).
    # Returns: the FastAPI app.
    """Build app across create_app variants that may expect a factory or an instance."""
    try:
        first_param = next(iter(inspect.signature(create_app).parameters.values()))
        annotation_text = str(first_param.annotation)
    except (StopIteration, TypeError, ValueError):
        annotation_text = "typing.Callable"

    expects_instance = (
        "Environment" in annotation_text and "Callable" not in annotation_text
    )
    env_arg = SnakeEnvironment() if expects_instance else SnakeEnvironment
    return create_app(
        env_arg,
        SnakeAction,
        SnakeObservation,
        env_name="snake_env",
        max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
    )


# Create the app with web interface and README integration.
app = _create_snake_app()


def main():
    # [FUNCTION DOC]
    # Runs the module-level app with uvicorn on host 0.0.0.0, port 8000. Blocks until the server stops.
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.snake_env.server.app
        openenv serve snake_env

    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
