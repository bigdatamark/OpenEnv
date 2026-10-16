# [MODULE DOC]
# Tests for request validation at the capture proxy boundary: malformed
# Anthropic/Responses dialect payloads and non-object bodies get 400
# invalid_request_error responses, and session registration validates its
# metadata. Skipped without fastapi or the capture server module.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Request-validation behavior at the Capture Proxy boundary."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

server = pytest.importorskip("openenv.core.harness.capture.server")


@pytest.fixture
def client() -> TestClient:
    # [FUNCTION DOC]
    # Fixture: yields a TestClient for a default capture app that returns server
    # errors as responses instead of raising.
    with TestClient(server.create_app(), raise_server_exceptions=False) as client:
        yield client


def _headers(client: TestClient) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: registers a new session and returns a bearer Authorization header for it.
    session = client.post("/sessions", json={}).json()
    return {"Authorization": f"Bearer {session['session_id']}"}


@pytest.mark.parametrize(
    ("path", "body", "message"),
    [
        (
            "/v1/messages",
            {"messages": "not-an-array"},
            "Anthropic messages must be an array",
        ),
        (
            "/v1/messages",
            {"messages": ["not-an-object"]},
            "Anthropic messages must be objects",
        ),
        ("/v1/messages", {"tools": "not-an-array"}, "Anthropic tools must be an array"),
        (
            "/v1/messages",
            {"tools": ["not-an-object"]},
            "Anthropic tools must be objects",
        ),
        (
            "/v1/responses",
            {"input": {"not": "supported"}},
            "Responses input must be a string or an array of objects",
        ),
        (
            "/v1/responses",
            {"input": ["not-an-object"]},
            "Responses input items must be objects",
        ),
        (
            "/v1/responses",
            {"tools": "not-an-array"},
            "Responses tools must be an array",
        ),
        (
            "/v1/responses",
            {"tools": ["not-an-object"]},
            "Responses tools must be objects",
        ),
    ],
)
def test_invalid_dialect_payload_returns_a_client_error(
    client: TestClient, path: str, body: dict[str, object], message: str
) -> None:
    # [FUNCTION DOC]
    # Each malformed messages/tools/input shape returns 400 with the exact
    # expected error message.
    response = client.post(path, json=body, headers=_headers(client))

    assert response.status_code == 400
    assert response.json() == {
        "error": {"message": message, "type": "invalid_request_error"}
    }


def test_non_object_request_body_returns_a_client_error(client: TestClient) -> None:
    # [FUNCTION DOC]
    # A JSON array body returns 400 "body must be a JSON object".
    response = client.post("/v1/chat/completions", json=[], headers=_headers(client))

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "body must be a JSON object",
            "type": "invalid_request_error",
        }
    }


@pytest.mark.parametrize("metadata", [[], ["unexpected"], "unexpected"])
def test_session_registration_rejects_non_object_metadata(
    client: TestClient, metadata: object
) -> None:
    # [FUNCTION DOC]
    # Non-object metadata returns 400.
    response = client.post("/sessions", json={"metadata": metadata})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "metadata must be a JSON object",
            "type": "invalid_request_error",
        }
    }


@pytest.mark.parametrize("key", ["session_id", "upstream", "capture_level"])
def test_session_registration_rejects_reserved_metadata_keys(
    client: TestClient, key: str
) -> None:
    # [FUNCTION DOC]
    # session_id/upstream/capture_level in metadata return 400 naming the key.
    response = client.post("/sessions", json={"metadata": {key: "conflict"}})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": f"metadata cannot include reserved key: {key}",
            "type": "invalid_request_error",
        }
    }


def test_session_registration_accepts_non_reserved_metadata(client: TestClient) -> None:
    # [FUNCTION DOC]
    # Other metadata keys are accepted and a session id returned.
    response = client.post("/sessions", json={"metadata": {"task_id": "task-123"}})

    assert response.status_code == 200
    assert response.json()["session_id"]
