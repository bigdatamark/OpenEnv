# [MODULE DOC]
# Tests that GenericEnvClient never loses track of a provider-allocated resource when
# setup fails and cleanup (stop) also fails: pending cleanup blocks new allocations and is
# retried, recovered cleanup stops the old resource before restarting, and factory-created
# clients require an explicit close. Runs in async and sync modes against a fake provider
# ledger (both start_container/stop_container and start/stop naming) with ws_connect mocked.
# SPDX-License-Identifier: BSD-3-Clause

"""Provider cleanup retries must not lose ownership of an allocated resource."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from openenv.core.generic_client import GenericEnvClient
from websockets.protocol import State


class ResourceLedger:
    # [CLASS DOC]
    # HELPER: Fake provider model with a single resource handle that each start overwrites;
    # records ("start"/"ready"/"stop", resource) events and the set of live resources, with
    # injectable start, readiness and stop errors. Exposes a spec'd Mock as provider.
    """Model a provider whose next start overwrites its sole resource handle."""

    def __init__(self, start_name, stop_name):
        # [FUNCTION DOC]
        # Initialises the ledger and wires the Mock provider's start/stop/wait_for_ready methods
        # (named by start_name/stop_name) to this object's methods.
        self.events = []
        self.live_resources = set()
        self.current_resource = None
        self.allocations = 0
        self.start_error = None
        self.readiness_error = None
        self.stop_error = None
        self.provider = Mock(spec=[start_name, stop_name, "wait_for_ready"])
        getattr(self.provider, start_name).side_effect = self.start
        getattr(self.provider, stop_name).side_effect = self.stop
        self.provider.wait_for_ready.side_effect = self.wait_for_ready

    def start(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Allocates the next resource id, marks it live and logs it; raises start_error if set
        # (after allocating). Returns: "http://localhost:<8000+id>".
        self.allocations += 1
        self.current_resource = self.allocations
        self.live_resources.add(self.current_resource)
        self.events.append(("start", self.current_resource))
        if self.start_error is not None:
            # A provider can acquire a resource before startup raises.
            raise self.start_error
        return f"http://localhost:{8000 + self.current_resource}"

    def wait_for_ready(self, *args):
        # [FUNCTION DOC]
        # Logs readiness for the current resource; raises readiness_error if set.
        self.events.append(("ready", self.current_resource))
        if self.readiness_error is not None:
            raise self.readiness_error

    def stop(self):
        # [FUNCTION DOC]
        # Logs a stop; raises stop_error if set; otherwise removes the current resource from the
        # live set (a repeated stop raises KeyError, modelling non-idempotent providers).
        self.events.append(("stop", self.current_resource))
        if self.stop_error is not None:
            raise self.stop_error
        # An accidental repeated stop fails, as with non-idempotent providers.
        self.live_resources.remove(self.current_resource)


@pytest.fixture(params=[("start_container", "stop_container"), ("start", "stop")])
def ledger(request):
    # [FUNCTION DOC]
    # Fixture: a ResourceLedger parametrized over the two provider method naming schemes.
    return ResourceLedger(*request.param)


@pytest.fixture(params=["async", "sync"])
def execution_mode(request):
    # [FUNCTION DOC]
    # Fixture parametrized over "async" and "sync".
    return request.param


@pytest.fixture
def client_type():
    # [FUNCTION DOC]
    # Fixture returning a GenericEnvClient subclass whose constructor raises a class-level
    # child_error when built with a base_url (simulating child construction failure).
    class SessionClient(GenericEnvClient):
        # [CLASS DOC]
        # HELPER: GenericEnvClient subclass with an injectable constructor failure.
        child_error = None

        def __init__(self, base_url=None, provider=None, **kwargs):
            # [FUNCTION DOC]
            # Raises child_error if set and base_url is given; otherwise defers to GenericEnvClient.
            if base_url is not None and self.child_error is not None:
                raise self.child_error
            super().__init__(base_url=base_url, provider=provider, **kwargs)

    return SessionClient


@pytest.fixture
def websocket_connect(monkeypatch, ledger):
    # [FUNCTION DOC]
    # Fixture patching openenv.core.env_client.ws_connect with an AsyncMock that returns a
    # fake open websocket whose recv answers with an observation naming the current resource.
    # Returns: the AsyncMock.
    async def connect(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Builds and returns the fake websocket.
        websocket = Mock(spec=["send", "recv", "close", "state"])
        websocket.state = State.OPEN
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "response",
                    "data": {
                        "observation": {"resource": ledger.current_resource},
                        "done": False,
                    },
                }
            )
        )
        websocket.close = AsyncMock()
        return websocket

    mocked_connect = AsyncMock(side_effect=connect)
    monkeypatch.setattr("openenv.core.env_client.ws_connect", mocked_connect)
    return mocked_connect


def make_client(client_type, ledger, execution_mode):
    # [FUNCTION DOC]
    # HELPER: Builds the client with the ledger's provider; returns its sync wrapper in sync mode.
    client = client_type(provider=ledger.provider)
    return client.sync() if execution_mode == "sync" else client


async def invoke(client, execution_mode, method, *args):
    # [FUNCTION DOC]
    # HELPER: Calls the named method with args, awaiting the result in async mode.
    result = getattr(client, method)(*args)
    return await result if execution_mode == "async" else result


def fail_setup(ledger, client_type, failure):
    # [FUNCTION DOC]
    # HELPER: Injects a ValueError at "start", "readiness" or child "constructor". Returns the error.
    error = ValueError(f"{failure} failed")
    if failure == "start":
        ledger.start_error = error
    elif failure == "readiness":
        ledger.readiness_error = error
    else:
        client_type.child_error = error
    return error


def allow_setup(ledger, client_type):
    # [FUNCTION DOC]
    # HELPER: Clears all injected setup errors.
    ledger.start_error = None
    ledger.readiness_error = None
    client_type.child_error = None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "readiness", "constructor"])
@pytest.mark.parametrize("retry_method", ["connect", "new_session"])
async def test_pending_cleanup_blocks_allocation_and_retries_stop_once(
    ledger, execution_mode, client_type, websocket_connect, failure, retry_method
):
    # [FUNCTION DOC]
    # After a setup failure with failing cleanup, the original error propagates and stop was
    # tried once; subsequent connect/new_session calls only retry stop (raising the cleanup
    # error), never allocate or open a websocket.
    client = make_client(client_type, ledger, execution_mode)
    original_error = fail_setup(ledger, client_type, failure)
    cleanup_error = RuntimeError("resource is still running")
    ledger.stop_error = cleanup_error
    try:
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "new_session")
        assert caught.value is original_error
        assert ledger.events.count(("stop", 1)) == 1
        allow_setup(ledger, client_type)

        # Repeated user retries may retry stop, but never allocate or connect.
        for _ in range(2):
            previous_events = list(ledger.events)
            with pytest.raises(RuntimeError) as caught:
                await invoke(client, execution_mode, retry_method)
            assert caught.value is cleanup_error
            assert ledger.events == previous_events + [("stop", 1)]
            assert ledger.live_resources == {1}
            websocket_connect.assert_not_called()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "readiness", "constructor"])
async def test_recovered_cleanup_stops_original_resource_before_restart(
    ledger, execution_mode, client_type, websocket_connect, failure
):
    # [FUNCTION DOC]
    # Once cleanup works, the next new_session stops resource 1 then starts and readies 2;
    # double close stops 2 exactly once.
    client = make_client(client_type, ledger, execution_mode)
    original_error = fail_setup(ledger, client_type, failure)
    ledger.stop_error = RuntimeError("temporary cleanup failure")
    try:
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "new_session")
        assert caught.value is original_error
        allow_setup(ledger, client_type)
        ledger.stop_error = None
        previous_events = list(ledger.events)

        child = await invoke(client, execution_mode, "new_session")

        assert ledger.events == previous_events + [
            ("stop", 1),
            ("start", 2),
            ("ready", 2),
        ]
        assert ledger.live_resources == {2}
        assert child.base_url == "http://localhost:8002"
        websocket_connect.assert_awaited_once()
        await invoke(client, execution_mode, "close")
        await invoke(client, execution_mode, "close")
        assert ledger.events.count(("stop", 2)) == 1
        assert ledger.live_resources == set()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_method", ["connect", "new_session"])
async def test_failed_explicit_close_blocks_restart_until_cleanup_succeeds(
    ledger, execution_mode, client_type, websocket_connect, retry_method
):
    # [FUNCTION DOC]
    # A failing close blocks connect/new_session (retrying stop) until stop succeeds, after
    # which the old resource is stopped and a new one started.
    client = make_client(client_type, ledger, execution_mode)
    cleanup_error = RuntimeError("close failed")
    try:
        await invoke(client, execution_mode, "connect")
        ledger.stop_error = cleanup_error
        with pytest.raises(RuntimeError) as caught:
            await invoke(client, execution_mode, "close")
        assert caught.value is cleanup_error
        previous_events = list(ledger.events)

        with pytest.raises(RuntimeError) as caught:
            await invoke(client, execution_mode, retry_method)
        assert caught.value is cleanup_error
        assert ledger.events == previous_events + [("stop", 1)]
        assert websocket_connect.await_count == 1
        assert ledger.live_resources == {1}

        ledger.stop_error = None
        previous_events = list(ledger.events)
        await invoke(client, execution_mode, retry_method)
        assert ledger.events == previous_events + [
            ("stop", 1),
            ("start", 2),
            ("ready", 2),
        ]
        await invoke(client, execution_mode, "close")
        assert ledger.live_resources == set()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "readiness"])
async def test_connect_preserves_setup_error_when_cleanup_also_fails(
    ledger, execution_mode, client_type, websocket_connect, failure
):
    # [FUNCTION DOC]
    # connect re-raises the setup error (not the cleanup error), tried stop once, never connected.
    client = make_client(client_type, ledger, execution_mode)
    original_error = fail_setup(ledger, client_type, failure)
    ledger.stop_error = RuntimeError("cleanup failed")
    try:
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "connect")
        assert caught.value is original_error
        assert ledger.events.count(("stop", 1)) == 1
        assert ledger.live_resources == {1}
        websocket_connect.assert_not_called()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
async def test_sibling_constructor_failure_keeps_live_child_usable(
    ledger, execution_mode, client_type, websocket_connect
):
    # [FUNCTION DOC]
    # A failed second child construction does not disturb a live first child, which can still step.
    client = make_client(client_type, ledger, execution_mode)
    try:
        # The provider is owned by a parent with no WebSocket of its own.
        child = await invoke(client, execution_mode, "new_session")
        original_error = fail_setup(ledger, client_type, "constructor")
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "new_session")
        assert caught.value is original_error
        assert ledger.events == [("start", 1), ("ready", 1)]
        result = await invoke(child, execution_mode, "step", {})
        assert result.observation == {"resource": 1}
        websocket_connect.assert_awaited_once()
        await invoke(client, execution_mode, "close")
        assert ledger.live_resources == set()
    finally:
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
async def test_multiple_failure_recovery_cycles_leave_no_orphaned_resources(
    ledger, execution_mode, client_type, websocket_connect
):
    # [FUNCTION DOC]
    # Three failure/recovery cycles each end with no live resources and one stop per resource.
    client = make_client(client_type, ledger, execution_mode)
    try:
        for cycle, failure in enumerate(["start", "readiness", "constructor"]):
            original_error = fail_setup(ledger, client_type, failure)
            ledger.stop_error = RuntimeError("cleanup temporarily unavailable")
            with pytest.raises(ValueError) as caught:
                await invoke(client, execution_mode, "new_session")
            assert caught.value is original_error
            allow_setup(ledger, client_type)
            ledger.stop_error = None

            await invoke(client, execution_mode, "new_session")
            current_resource = 2 * cycle + 2
            assert ledger.live_resources == {current_resource}
            await invoke(client, execution_mode, "close")
            await invoke(client, execution_mode, "close")
            assert ledger.live_resources == set()
            assert ledger.events.count(("stop", current_resource)) == 1
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_method", ["connect", "new_session"])
@pytest.mark.parametrize("cleanup_recovers", [False, True])
async def test_factory_pending_cleanup_requires_explicit_close(
    ledger, execution_mode, websocket_connect, retry_method, cleanup_recovers
):
    # [FUNCTION DOC]
    # For clients from GenericEnvClient.from_env, a failed close makes connect/new_session
    # refuse (mentioning close()) without touching the provider until close succeeds.
    bootstrap = GenericEnvClient.from_env(
        "audit/test-environment",
        provider=ledger.provider,
        use_docker=hasattr(ledger.provider, "start_container"),
    )
    client = bootstrap.sync() if execution_mode == "sync" else await bootstrap
    try:
        ledger.stop_error = RuntimeError("cleanup temporarily unavailable")
        with pytest.raises(RuntimeError, match="cleanup temporarily unavailable"):
            await invoke(client, execution_mode, "close")
        if cleanup_recovers:
            ledger.stop_error = None
        previous_events = list(ledger.events)
        websocket_connect.reset_mock()

        # A factory URL cannot be reused after an implicit cleanup retry.
        # Refuse before touching either the provider or the cached endpoint.
        for _ in range(2):
            with pytest.raises(RuntimeError, match=r"close\(\)"):
                await invoke(client, execution_mode, retry_method)
        assert ledger.events == previous_events
        assert ledger.live_resources == {1}
        websocket_connect.assert_not_called()

        ledger.stop_error = None
        await invoke(client, execution_mode, "close")
        await invoke(client, execution_mode, "close")
        assert ledger.events == previous_events + [("stop", 1)]
        assert not ledger.live_resources
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")
