# [PACKAGE DOC]
# Container runtime providers package. Re-exports the abstract provider bases,
# the local Docker, Docker Swarm and Kubernetes providers, and the uv-based
# provider. Cloud providers requiring extra SDKs (ACA, Daytona, Modal, Novita,
# HF sandbox) must be imported from their own modules.
# SPDX-License-Identifier: BSD-3-Clause

"""Container runtime providers."""

from .providers import (
    ContainerProvider,
    DockerSwarmProvider,
    KubernetesProvider,
    LocalDockerProvider,
    RuntimeProvider,
)
from .uv_provider import UVProvider

# Note: optional cloud providers that require extra SDKs (e.g.
# `ACASandboxProvider`, `DaytonaProvider`) are intentionally NOT re-exported
# here. Import them from their module, e.g.
# `from openenv.core.containers.runtime.aca_provider import ACASandboxProvider`.

__all__ = [
    "ContainerProvider",
    "DockerSwarmProvider",
    "LocalDockerProvider",
    "KubernetesProvider",
    "RuntimeProvider",
    "UVProvider",
]
