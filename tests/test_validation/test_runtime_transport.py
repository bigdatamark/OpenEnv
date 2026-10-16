# [MODULE DOC]
# Tests that collect_runtime_evidence keeps episode findings correct around
# WebSocket teardown noise, interrupts and blocked writes. The HTTP schema
# fetch is stubbed via httpx MockTransport and the WebSocket connection is a
# scripted fake (EpisodeConnection) or, in one case, a real blocked socket.
"""Episode findings survive teardown noise and blocked transport writes."""

import json
import socket
import threading
import time

import httpx
import pytest
from openenv.validation.runtime import collector
from openenv.validation.runtime.contracts import RuntimePlan


class AbortableTransport:
    # [CLASS DOC]
    # Fake underlying socket that sets an "aborted" threading event when shut down
    # or closed, so tests can see whether the collector aborted the transport.
    def __init__(self):
        # [FUNCTION DOC]
        # Creates the unset aborted event.
        self.aborted = threading.Event()

    def shutdown(self, how):
        # [FUNCTION DOC]
        # Sets aborted; the "how" argument is ignored.
        self.aborted.set()

    def close(self):
        # [FUNCTION DOC]
        # Sets aborted.
        self.aborted.set()


class EpisodeConnection:
    # [CLASS DOC]
    # Scripted fake WebSocket connection for a one-step "bounded" episode, with
    # optional faults: failing close-message send, failing close(), blocking sends,
    # and a failing step receive.
    def __init__(self, *, close_send=False, close_failure=False, block_send=False):
        # [FUNCTION DOC]
        # Stores fault flags, creates an AbortableTransport as socket and initializes
        # step_failure, closed, the last operation and the step counter.
        self.socket = AbortableTransport()
        self.close_send = close_send
        self.close_failure = close_failure
        self.block_send = block_send
        self.step_failure = False
        self.closed = False
        self.operation = None
        self.steps = 0

    def send(self, payload):
        # [FUNCTION DOC]
        # Records the message type as the current operation. If block_send, waits up
        # to 0.8 s for the transport abort then raises ConnectionError; if the message
        # is "close" and close_send is set, raises ConnectionError.
        self.operation = json.loads(payload)["type"]
        if self.block_send:
            # The fallback keeps the RED test finite when no watchdog exists.
            self.socket.aborted.wait(0.8)
            raise ConnectionError("transport did not accept the write")
        if self.operation == "close" and self.close_send:
            raise ConnectionError("server already closed a complete episode")

    def recv(self, timeout):
        # [FUNCTION DOC]
        # Returns a JSON reply for the last operation: a state with the current step
        # count, or an observation (incrementing steps on "step") with reward 1 and
        # done False. Raises ValueError on step when step_failure is set. timeout is
        # ignored.
        if self.operation == "state":
            return json.dumps(
                {
                    "type": "state",
                    "data": {"episode_id": "bounded", "step_count": self.steps},
                }
            )
        if self.operation == "step":
            if self.step_failure:
                raise ValueError("genuine in-episode failure")
            self.steps += 1
        return json.dumps(
            {
                "type": "observation",
                "data": {
                    "observation": {"counter": self.steps},
                    "reward": 1,
                    "done": False,
                },
            }
        )

    def close(self):
        # [FUNCTION DOC]
        # Marks closed; raises ConnectionError if close_failure is set.
        self.closed = True
        if self.close_failure:
            raise ConnectionError("websocket teardown failed")


@pytest.fixture
def collect(monkeypatch):
    # [FUNCTION DOC]
    # Fixture that stubs the collector's httpx.Client to return a simple schema
    # and builds a one-action plan.
    # Returns: a run function that injects a given fake connection as the
    # collector's connect result and calls collect_runtime_evidence.
    original_client = httpx.Client
    monkeypatch.setattr(
        collector.httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"observation": {"type": "object"}}
                )
            ),
            **kwargs,
        ),
    )
    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "bounded", "seed": 7},
            "actions": [{"increment": 1}],
        }
    )

    def run(connection, *, episode_timeout_s=2, request_timeout_s=None):
        # [FUNCTION DOC]
        # HELPER: Patches collector.connect to return the connection and runs
        # collection with the given episode/request timeouts.
        # Returns: the RuntimeEvidence.
        monkeypatch.setattr(collector, "connect", lambda *args, **kwargs: connection)
        return collector.collect_runtime_evidence(
            "http://127.0.0.1:8000",
            plan,
            episode_timeout_s=episode_timeout_s,
            request_timeout_s=request_timeout_s,
        )

    return run


@pytest.mark.parametrize("fault", ["close_send", "close_failure"])
def test_teardown_noise_does_not_fail_a_completed_episode(collect, fault):
    # [FUNCTION DOC]
    # A failing close send or close() after a complete episode leaves the full
    # trace with no failure, and the connection is closed.
    connection = EpisodeConnection(**{fault: True})
    evidence = collect(connection)
    assert [row.operation for row in evidence.exchanges] == [
        "reset",
        "state",
        "step",
        "state",
    ]
    assert evidence.failure_reason is None
    assert evidence.failure_phase is None
    assert connection.closed


@pytest.mark.parametrize("fault", ["close_send", "close"])
def test_teardown_interrupt_does_not_fail_a_completed_episode(collect, fault):
    # [FUNCTION DOC]
    # A KeyboardInterrupt during the close send or close() after completion still
    # yields a complete, failure-free trace.
    connection = EpisodeConnection()
    if fault == "close_send":
        send = connection.send

        def interrupt_close_send(payload):
            # [FUNCTION DOC]
            # HELPER: Sends normally, then raises KeyboardInterrupt if it was the close
            # message.
            send(payload)
            if connection.operation == "close":
                raise KeyboardInterrupt

        connection.send = interrupt_close_send
    else:

        def interrupt_close():
            # [FUNCTION DOC]
            # HELPER: Marks closed and raises KeyboardInterrupt.
            connection.closed = True
            raise KeyboardInterrupt

        connection.close = interrupt_close

    evidence = collect(connection)
    assert [row.operation for row in evidence.exchanges] == [
        "reset",
        "state",
        "step",
        "state",
    ]
    assert evidence.failure_reason is None
    assert evidence.failure_phase is None
    assert connection.closed


def test_teardown_failure_does_not_replace_a_genuine_operation_failure(collect):
    # [FUNCTION DOC]
    # A step failure is reported (phase step, ValueError) even when close also
    # fails; only reset and state are recorded.
    connection = EpisodeConnection(close_failure=True)
    connection.step_failure = True
    evidence = collect(connection)
    assert evidence.failure_phase == "step"
    assert evidence.failure_reason == "step failed (ValueError)"
    assert [row.operation for row in evidence.exchanges] == ["reset", "state"]
    assert connection.closed


def test_teardown_interrupt_does_not_replace_a_genuine_operation_failure(collect):
    # [FUNCTION DOC]
    # A step failure is reported even when close raises KeyboardInterrupt.
    connection = EpisodeConnection()
    connection.step_failure = True

    def interrupt_close():
        # [FUNCTION DOC]
        # HELPER: Marks closed and raises KeyboardInterrupt.
        connection.closed = True
        raise KeyboardInterrupt

    connection.close = interrupt_close
    evidence = collect(connection)
    assert evidence.failure_phase == "step"
    assert evidence.failure_reason == "step failed (ValueError)"
    assert [row.operation for row in evidence.exchanges] == ["reset", "state"]
    assert connection.closed


def test_interruption_preserves_partial_evidence_despite_teardown_failure(collect):
    # [FUNCTION DOC]
    # A KeyboardInterrupt while receiving the step raises
    # RuntimeCollectionInterrupted carrying partial evidence (reset, state; phase
    # step; KeyboardInterrupt), and the connection is closed.
    connection = EpisodeConnection(close_failure=True)
    receive = connection.recv

    def interrupt_step(timeout):
        # [FUNCTION DOC]
        # HELPER: Raises KeyboardInterrupt on the step receive; delegates otherwise.
        if connection.operation == "step":
            raise KeyboardInterrupt
        return receive(timeout)

    connection.recv = interrupt_step
    with pytest.raises(collector.RuntimeCollectionInterrupted) as interrupted:
        collect(connection)
    evidence = interrupted.value.evidence
    assert [row.operation for row in evidence.exchanges] == ["reset", "state"]
    assert evidence.failure_phase == "step"
    assert evidence.failure_reason == "step failed (KeyboardInterrupt)"
    assert connection.closed


@pytest.mark.parametrize(
    "timeouts",
    [
        {"episode_timeout_s": 0.05, "request_timeout_s": 2},
        {"episode_timeout_s": 2, "request_timeout_s": 0.05},
    ],
)
def test_blocked_send_obeys_episode_and_operation_deadlines(collect, timeouts):
    # [FUNCTION DOC]
    # With a blocking send and a short episode or request timeout, collection
    # aborts the transport within 0.5 s, closes the connection and reports a reset
    # TimeoutError with no exchanges.
    connection = EpisodeConnection(block_send=True)
    started = time.monotonic()
    evidence = collect(connection, **timeouts)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, "send exceeded the declared transport deadline"
    assert connection.socket.aborted.is_set()
    assert connection.closed
    assert evidence.exchanges == ()
    assert evidence.failure_phase == "reset"
    assert evidence.failure_reason == "reset failed (TimeoutError)"


def test_blocked_os_send_is_interrupted_and_transport_is_closed(collect):
    # [FUNCTION DOC]
    # Fills a real socketpair's send buffer so sendall blocks; the collector must
    # close the socket and report a reset TimeoutError within 1 s without needing
    # the 2 s fallback timer that shuts the socket down.
    sender, receiver = socket.socketpair()
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    sender.setblocking(False)
    try:
        while True:
            sender.send(b"x" * 4096)
    except BlockingIOError:
        pass
    sender.setblocking(True)
    connection = EpisodeConnection()
    connection.socket = sender
    connection.send = lambda payload: sender.sendall(payload.encode())
    rescued = threading.Event()

    def rescue():
        # [FUNCTION DOC]
        # HELPER: Fallback timer callback that records it fired and shuts down the
        # sender socket.
        rescued.set()
        sender.shutdown(socket.SHUT_RDWR)

    fallback = threading.Timer(2, rescue)
    fallback.daemon = True
    fallback.start()
    try:
        started = time.monotonic()
        evidence = collect(connection, episode_timeout_s=0.05)
        assert time.monotonic() - started < 1
        assert not rescued.is_set(), "collector needed the external rescue timeout"
        assert sender.fileno() == -1
        assert connection.closed
        assert evidence.failure_reason == "reset failed (TimeoutError)"
    finally:
        fallback.cancel()
        fallback.join()
        sender.close()
        receiver.close()


@pytest.mark.parametrize(
    "code",
    ["FACTORY_ERROR", "EXECUTION_ERROR", "hf_abcdefghijk12345", {"unsafe": "payload"}],
)
def test_server_error_exposes_only_known_protocol_code(collect, code):
    # [FUNCTION DOC]
    # An error reply exposes its code only if it is a string ending in "_ERROR";
    # otherwise ValueError. The server message is never included; one exchange is
    # recorded.
    connection = EpisodeConnection()
    connection.recv = lambda timeout: json.dumps(
        {"type": "error", "data": {"code": code, "message": "private server exception"}}
    )
    evidence = collect(connection)
    expected = (
        code if isinstance(code, str) and code.endswith("_ERROR") else "ValueError"
    )
    assert evidence.failure_reason == f"reset failed ({expected})"
    assert "private server exception" not in evidence.failure_reason
    assert len(evidence.exchanges) == 1


def test_default_operation_deadline_uses_remaining_episode_budget(collect):
    # [FUNCTION DOC]
    # Without a request timeout, each receive timeout is derived from the
    # remaining episode budget (between 50 and 60 s for a 60 s episode).
    connection = EpisodeConnection()
    receive = connection.recv
    timeouts = []

    def record_timeout(timeout):
        # [FUNCTION DOC]
        # HELPER: Records the timeout argument and delegates to the original recv.
        timeouts.append(timeout)
        return receive(timeout)

    connection.recv = record_timeout
    evidence = collect(connection, episode_timeout_s=60)
    assert evidence.failure_reason is None
    assert all(50 < timeout <= 60 for timeout in timeouts)
