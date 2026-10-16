# [MODULE DOC]
# Tests that collect_runtime_evidence fetches the HTTP /schema response safely:
# it requests identity encoding, rejects compressed bodies before reading,
# enforces the byte budget and a hard deadline (also against a trickling raw
# socket server), and keeps reset-schema presence when later collection fails.
# Most tests stub httpx with a MockTransport and make the WebSocket connect
# fail so collection stops after the schema phase.
"""Hostile HTTP schema responses must stay bounded before JSON/schema grading."""

import json
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from openenv.validation.runtime import collector
from openenv.validation.runtime.contracts import RuntimePlan
from openenv.validation.runtime.transport import http_deadline


class TrackedStream(httpx.SyncByteStream):
    # [CLASS DOC]
    # httpx sync byte stream that yields one payload and records whether it was
    # read and closed; optionally fails if read.
    def __init__(self, payload, *, reject_read=False):
        # [FUNCTION DOC]
        # Stores payload and reject_read and initializes read/closed flags to False.
        self.payload = payload
        self.reject_read = reject_read
        self.read = False
        self.closed = False

    def __iter__(self):
        # [FUNCTION DOC]
        # Marks read, raises AssertionError if reject_read, else yields the payload.
        self.read = True
        if self.reject_read:
            raise AssertionError("Compressed bytes must never reach the decoder")
        yield self.payload

    def close(self):
        # [FUNCTION DOC]
        # Marks closed.
        self.closed = True


@pytest.fixture
def plan():
    # [FUNCTION DOC]
    # Fixture returning a one-action RuntimePlan (episode "bounded", seed 7).
    return RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "bounded", "seed": 7},
            "actions": [{"increment": 1}],
        }
    )


def schema_transport(monkeypatch, stream, headers=None):
    # [FUNCTION DOC]
    # HELPER: Patches the collector's httpx.Client to use a MockTransport that
    # returns 200 with the given headers and stream, and patches the collector's
    # WebSocket connect to raise ConnectionError.
    # Returns: the list that collects received httpx requests.
    requests = []
    original_client = httpx.Client

    def respond(request):
        # [FUNCTION DOC]
        # HELPER: Records the request and returns the canned response.
        requests.append(request)
        return httpx.Response(200, headers=headers or {}, stream=stream)

    monkeypatch.setattr(
        collector.httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(respond), **kwargs
        ),
    )

    def no_websocket(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Always raises ConnectionError to stop after schema retrieval.
        raise ConnectionError("stop after checking schema retrieval")

    monkeypatch.setattr(collector, "connect", no_websocket)
    return requests


@pytest.mark.parametrize(
    "encoding", ["gzip", "deflate", "br", "zstd", "gzip, identity", "unknown"]
)
def test_compressed_schema_is_rejected_before_body_read(monkeypatch, plan, encoding):
    # [FUNCTION DOC]
    # Any non-identity Content-Encoding fails the schema phase with ValueError
    # without reading the body (stream closed), requesting identity encoding and
    # producing no schema or exchanges.
    stream = TrackedStream(b"untrusted compressed bytes", reject_read=True)
    requests = schema_transport(monkeypatch, stream, {"Content-Encoding": encoding})
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert not stream.read
    assert stream.closed
    assert evidence.failure_phase == "schema"
    assert evidence.failure_reason == "schema failed (ValueError)"
    assert evidence.observation_schema_json is None
    assert evidence.exchanges == ()


@pytest.mark.parametrize(
    "headers",
    [{}, {"Content-Encoding": "identity"}, {"Content-Encoding": " Identity "}],
)
def test_uncompressed_schema_is_read_and_retained(monkeypatch, plan, headers):
    # [FUNCTION DOC]
    # Identity or absent encoding (case/space-insensitive) is read and its
    # observation schema retained; collection then fails at connect.
    schema = {"type": "object"}
    stream = TrackedStream(json.dumps({"observation": schema}).encode())
    requests = schema_transport(monkeypatch, stream, headers)
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert stream.read and stream.closed
    assert json.loads(evidence.observation_schema_json) == schema
    assert evidence.failure_phase == "connect"


def test_uncompressed_schema_still_obeys_total_byte_budget(monkeypatch, plan):
    # [FUNCTION DOC]
    # A body larger than MAX_MESSAGE_BYTES (patched to 64) fails the schema phase.
    stream = TrackedStream(b"x" * 65)
    schema_transport(monkeypatch, stream)
    monkeypatch.setattr(collector, "MAX_MESSAGE_BYTES", 64)
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert stream.read and stream.closed
    assert evidence.failure_phase == "schema"
    assert evidence.failure_reason == "schema failed (ValueError)"
    assert evidence.observation_schema_json is None


@pytest.mark.parametrize("slow_phase", ["headers", "body"])
def test_schema_deadline_aborts_trickling_http_server(plan, slow_phase):
    # [FUNCTION DOC]
    # Parametrized on which phase trickles. A thread serves a raw HTTP response one
    # byte every 25 ms; with a 0.15 s episode timeout the schema phase must fail
    # with TimeoutError within 0.6 s and the server thread must finish.
    body = json.dumps({"observation": {"type": "object"}}).encode()
    headers = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        port = listener.getsockname()[1]

        def serve():
            # [FUNCTION DOC]
            # HELPER: Accepts one connection, reads the request, then sends headers and/or
            # body byte by byte; swallows OSError caused by the client aborting.
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(3)
                    connection.recv(65536)
                    if slow_phase == "body":
                        connection.sendall(headers)
                    payload = headers + body if slow_phase == "headers" else body
                    for byte in payload:
                        connection.sendall(bytes([byte]))
                        time.sleep(0.025)
            except OSError:
                pass  # The deadline intentionally closes the peer transport.

        server = threading.Thread(target=serve)
        server.start()
        try:
            started = time.monotonic()
            evidence = collector.collect_runtime_evidence(
                f"http://127.0.0.1:{port}", plan, episode_timeout_s=0.15
            )
            elapsed = time.monotonic() - started
        finally:
            server.join(timeout=4)
        assert not server.is_alive()
    assert evidence.failure_phase == "schema"
    assert evidence.failure_reason == "schema failed (TimeoutError)"
    assert elapsed < 0.6, (
        f"{slow_phase} trickle escaped the 0.15-second deadline: {elapsed:.3f}s"
    )


def test_http_deadline_survives_transport_socket_detach():
    # [FUNCTION DOC]
    # http_deadline must still raise TimeoutError ("HTTP deadline") and unblock a
    # recv even after the socket captured from the connect trace hook is detached
    # into a new socket object.
    peer, connection = socket.socketpair()
    detached = None
    try:
        stream = SimpleNamespace(get_extra_info=lambda name: connection)
        with pytest.raises(TimeoutError, match="HTTP deadline"):
            with http_deadline(0.05) as extensions:
                extensions["trace"](
                    "connection.connect_tcp.complete", {"return_value": stream}
                )
                # TLS wrapping similarly detaches the socket captured at connect.
                detached = socket.socket(fileno=connection.detach())
                detached.settimeout(1)  # Bound the regression if cancellation breaks.
                assert detached.recv(1) == b""
    finally:
        peer.close()
        connection.close()
        if detached is not None:
            detached.close()


@pytest.mark.parametrize(
    "schema",
    [{}, {"reset_observation": None}, {"reset_observation": {"required": ["ready"]}}],
)
def test_reset_schema_presence_survives_later_collection_failure(
    monkeypatch, plan, schema
):
    # [FUNCTION DOC]
    # The reset_observation schema is kept (compact JSON, including null) when
    # present and None when absent, even though collection fails at connect.
    stream = TrackedStream(
        json.dumps({"observation": {"type": "object"}, **schema}).encode()
    )
    schema_transport(monkeypatch, stream)
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert evidence.failure_phase == "connect"
    assert evidence.reset_observation_schema_json == (
        json.dumps(schema["reset_observation"], separators=(",", ":"))
        if "reset_observation" in schema
        else None
    )


def test_reset_schema_shares_entire_response_byte_budget(monkeypatch, plan):
    # [FUNCTION DOC]
    # A reset schema pushing the response over the budget fails the schema phase
    # and no reset schema is kept.
    stream = TrackedStream(
        json.dumps(
            {"observation": {}, "reset_observation": {"description": "x" * 100}}
        ).encode()
    )
    schema_transport(monkeypatch, stream)
    monkeypatch.setattr(collector, "MAX_MESSAGE_BYTES", 64)
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert evidence.failure_phase == "schema"
    assert evidence.reset_observation_schema_json is None
