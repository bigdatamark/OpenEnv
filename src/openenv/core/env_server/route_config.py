# [MODULE DOC]
# Declarative registration of simple synchronous GET endpoints on a FastAPI
# app from configuration records.
# SPDX-License-Identifier: BSD-3-Clause

"""
Route configuration utilities for declarative FastAPI route registration.

This module provides utilities to reduce boilerplate in route registration
by using configuration objects instead of repeated function calls.
"""

from dataclasses import dataclass
from typing import Callable, List, Type

from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class GetEndpointConfig:
    # [CLASS DOC]
    # Dataclass describing one GET endpoint: path, zero-argument handler returning
    # a model or dict, response model, OpenAPI tag, summary and description.
    """Configuration for a simple GET endpoint."""

    path: str
    handler: Callable[[], BaseModel | dict]
    response_model: Type[BaseModel] | type[dict]
    tag: str
    summary: str
    description: str


def _make_get_endpoint(
    handler: Callable[[], BaseModel | dict],
) -> Callable[[], BaseModel | dict]:
    # [FUNCTION DOC]
    # HELPER: Wraps a sync zero-argument handler in an async endpoint function that
    # calls it (on the event loop thread, so a slow handler blocks the loop).
    """Wrap a sync GET handler in the async endpoint FastAPI expects."""

    async def endpoint() -> BaseModel | dict:
        # [FUNCTION DOC]
        # HELPER: Async endpoint returning handler().
        return handler()

    return endpoint


def register_get_endpoints(app: FastAPI, configs: List[GetEndpointConfig]) -> None:
    # [FUNCTION DOC]
    # Registers each config on app as a GET route with its response model, tag,
    # summary and description.
    """
    Register multiple GET endpoints from configuration.

    Args:
        app ([`~fastapi.FastAPI`]):
            FastAPI application instance.
        configs (`List[GetEndpointConfig]`):
            List of GET endpoint configurations.
    """
    for config in configs:
        app.get(
            config.path,
            response_model=config.response_model,
            tags=[config.tag],
            summary=config.summary,
            description=config.description,
        )(_make_get_endpoint(config.handler))
