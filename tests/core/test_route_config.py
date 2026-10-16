# [MODULE DOC]
# Tests for register_get_endpoints in the declarative route configuration helpers.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declarative route registration helpers."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.route_config import (
    GetEndpointConfig,
    register_get_endpoints,
)
from pydantic import BaseModel


class _RouteResponse(BaseModel):
    # [CLASS DOC]
    # HELPER: Pydantic response model with a single string field value.
    value: str


def test_register_get_endpoints_binds_each_handler() -> None:
    # [FUNCTION DOC]
    # Registering two GET endpoints binds each to its own handler (no late-binding mix-up).
    """Each configured endpoint keeps its own handler binding."""
    app = FastAPI()

    register_get_endpoints(
        app,
        [
            GetEndpointConfig(
                path="/first",
                handler=lambda: _RouteResponse(value="first"),
                response_model=_RouteResponse,
                tag="test",
                summary="First",
                description="First endpoint",
            ),
            GetEndpointConfig(
                path="/second",
                handler=lambda: _RouteResponse(value="second"),
                response_model=_RouteResponse,
                tag="test",
                summary="Second",
                description="Second endpoint",
            ),
        ],
    )

    client = TestClient(app)

    assert client.get("/first").json() == {"value": "first"}
    assert client.get("/second").json() == {"value": "second"}
