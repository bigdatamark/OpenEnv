# [MODULE DOC]
# Creates the FastAPI app (module-level "app") for the Sophistry-Bench Sprint environment via
# create_app, passing the environment class as factory and env_name "sophistry_bench_sprint_env".
# Falls back to absolute package imports in the container runtime.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Sophistry-Bench Sprint Environment."""

try:
    # Installed-package context (e.g. import sophistry_bench_sprint_env.server.app)
    from openenv.core.env_server.http_server import create_app

    from ..models import AdvocacyAction, AdvocacyObservation
    from .sophistry_bench_sprint_environment import SophistryBenchSprintEnvironment
except ImportError:
    # Container runtime context (uvicorn server.app:app, PYTHONPATH=/app/env).
    # This branch resolves only because the wheel installs the package under the
    # name ``sophistry_bench_sprint_env`` via the load-bearing ``package-dir``
    # remap in pyproject.toml (source lives at the env-dir root, not a subdir).
    # If that remap is changed, container startup breaks here with a ModuleNotFound.
    from openenv.core.env_server.http_server import create_app
    from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation
    from sophistry_bench_sprint_env.server.sophistry_bench_sprint_environment import (
        SophistryBenchSprintEnvironment,
    )

app = create_app(
    SophistryBenchSprintEnvironment,
    AdvocacyAction,
    AdvocacyObservation,
    env_name="sophistry_bench_sprint_env",
)


def main():
    # [FUNCTION DOC]
    # Serves the app with uvicorn on 0.0.0.0:8000; blocks until shutdown.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
