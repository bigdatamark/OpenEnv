# [MODULE DOC]
# Unit tests for GenericEnvClient (dict-based actions/observations) and GenericAction:
# provider-owned startup and session cleanup, URL handling, payload/result/state parsing,
# from_docker_image/from_env factories and their sync bootstrap handles, AutoEnv/AutoAction
# integration, WebSocket reconnection and related client behaviours. Network and container
# calls are mocked throughout.
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for GenericEnvClient and GenericAction
===================================================

Tests cover:
1. GenericEnvClient instantiation and basic operations
2. Dictionary-based action/observation handling
3. from_docker_image() inheritance
4. from_env() inheritance (HuggingFace registry)
5. AutoEnv integration with skip_install parameter
6. Comparison with typed clients
7. GenericAction class
8. AutoAction with skip_install parameter
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openenv.core.client_types import StepResult
from openenv.core.env_client import _is_localhost_ws_url
from openenv.core.generic_client import GenericAction, GenericEnvClient
from openenv.core.sync_client import SyncEnvClient
from websockets.protocol import State


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_websocket():
    # [FUNCTION DOC]
    # Fixture: MagicMock websocket whose recv returns a canned observation response JSON.
    # KNOWN ISSUE: not used by any test in this module.
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.recv.return_value = '{"type": "response", "data": {"observation": {"output": "hello"}, "reward": 1.0, "done": false}}'
    return ws


@pytest.fixture
def mock_provider():
    # [FUNCTION DOC]
    # Fixture: Mock container provider whose start_container returns http://localhost:8000.
    """Create a mock container provider."""
    provider = Mock()
    provider.start_container.return_value = "http://localhost:8000"
    provider.wait_for_ready.return_value = None
    return provider


# ============================================================================
# GenericEnvClient Unit Tests
# ============================================================================


@pytest.fixture(params=[("start_container", "stop_container"), ("start", "stop")])
def session_provider(request):
    # [FUNCTION DOC]
    # Fixture parametrized over (start_container, stop_container) and (start, stop) naming.
    # Returns: (spec'd provider Mock, its start method mock, its stop method mock).
    start_name, stop_name = request.param
    provider = Mock(spec=[start_name, stop_name, "wait_for_ready"])
    start = getattr(provider, start_name)
    stop = getattr(provider, stop_name)
    start.return_value = "http://localhost:8000"
    return provider, start, stop


class FailingSessionClient(GenericEnvClient):
    # [CLASS DOC]
    # HELPER: GenericEnvClient whose constructor raises ValueError whenever a base_url is
    # given (i.e. when a child session is created).
    def __init__(self, base_url=None, provider=None, **kwargs):
        # [FUNCTION DOC]
        # Raises ValueError("child constructor failed") for a base_url; otherwise defers to super.
        if base_url is not None:
            raise ValueError("child constructor failed")
        super().__init__(base_url=base_url, provider=provider, **kwargs)


class TestSessionProviderCleanup:
    # [CLASS DOC]
    # Tests cleanup of provider-started servers when new_session fails.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["readiness", "constructor"])
    async def test_failed_session_is_cleaned_up_only_once(
        self, session_provider, failure
    ):
        # [FUNCTION DOC]
        # A readiness or child-constructor failure propagates, stops the provider once, never
        # connects, clears URLs and children; later closes do not stop again.
        provider, start, stop = session_provider
        stop.side_effect = [None, RuntimeError("already stopped")]
        if failure == "readiness":
            error = TimeoutError("not ready")
            provider.wait_for_ready.side_effect = error
            client = GenericEnvClient(provider=provider)
        else:
            client = FailingSessionClient(provider=provider)

        with patch("openenv.core.env_client.ws_connect", AsyncMock()) as connect:
            with pytest.raises((TimeoutError, ValueError)) as caught:
                await client.new_session()

        if failure == "readiness":
            assert caught.value is error
        else:
            assert str(caught.value) == "child constructor failed"
        start.assert_called_once_with()
        stop.assert_called_once_with()
        connect.assert_not_called()
        assert client.base_url is None
        assert client._ws_url is None
        assert client._child_clients == []
        await client.close()
        await client.close()
        stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_failed_cleanup_can_be_retried(self, session_provider):
        # [FUNCTION DOC]
        # If cleanup fails, the provider is kept and a later close retries stop exactly once more.
        provider, start, stop = session_provider
        error = TimeoutError("not ready")
        provider.wait_for_ready.side_effect = error
        stop.side_effect = [RuntimeError("cleanup failed"), None]
        client = GenericEnvClient(provider=provider)
        with pytest.raises(TimeoutError) as caught:
            await client.new_session()
        assert caught.value is error
        assert client._provider is provider
        stop.assert_called_once_with()
        await client.close()
        assert stop.call_count == 2
        await client.close()
        assert stop.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("connect_parent", [False, True])
    async def test_constructor_failure_preserves_running_provider(
        self, session_provider, connect_parent
    ):
        # [FUNCTION DOC]
        # A child constructor failure on an already-started parent leaves the provider and
        # parent socket running until the parent closes.
        provider, start, stop = session_provider
        parent = FailingSessionClient(provider=provider)
        ws = AsyncMock()
        with patch("openenv.core.env_client.ws_connect", AsyncMock(return_value=ws)):
            if connect_parent:
                await parent.connect()
            else:
                parent._start_provider_if_needed()
            with pytest.raises(ValueError, match="child constructor failed"):
                await parent.new_session()
            start.assert_called_once_with()
            stop.assert_not_called()
            assert parent.base_url == "http://localhost:8000"
            if connect_parent:
                assert parent._ws is ws
                ws.close.assert_not_called()
            await parent.close()
        stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_provider_can_restart_after_successful_cleanup(
        self, session_provider
    ):
        # [FUNCTION DOC]
        # After a cleaned-up readiness failure, a new session restarts the provider.
        provider, start, stop = session_provider
        provider.wait_for_ready.side_effect = TimeoutError("not ready")
        parent = GenericEnvClient(provider=provider)
        with pytest.raises(TimeoutError):
            await parent.new_session()
        stop.assert_called_once_with()
        provider.wait_for_ready.side_effect = None
        with patch(
            "openenv.core.env_client.ws_connect",
            AsyncMock(return_value=AsyncMock()),
        ):
            child = await parent.new_session()
            assert child.base_url == "http://localhost:8000"
            assert start.call_count == 2
            await parent.close()
        assert stop.call_count == 2

    @pytest.mark.parametrize("failure", ["readiness", "constructor"])
    def test_sync_session_failure_is_cleaned_up_only_once(
        self, session_provider, failure
    ):
        # [FUNCTION DOC]
        # Same as the async cleanup-once test via the sync wrapper.
        provider, start, stop = session_provider
        stop.side_effect = [None, RuntimeError("already stopped")]
        if failure == "readiness":
            provider.wait_for_ready.side_effect = TimeoutError("not ready")
            parent = GenericEnvClient(provider=provider)
        else:
            parent = FailingSessionClient(provider=provider)
        client = parent.sync()
        try:
            with pytest.raises((TimeoutError, ValueError)):
                client.new_session()
            stop.assert_called_once_with()
        finally:
            client.close()
        stop.assert_called_once_with()


class TestGenericEnvClientInstantiation:
    # [CLASS DOC]
    # Tests construction, provider-owned startup and session/child management.
    """Test GenericEnvClient instantiation."""

    def test_instantiation_with_http_url(self):
        # [FUNCTION DOC]
        # http URLs map to ws://.../ws.
        """Test that GenericEnvClient can be instantiated with HTTP URL."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        assert client._ws_url == "ws://localhost:8000/ws"

    def test_instantiation_with_https_url(self):
        # [FUNCTION DOC]
        # https URLs map to wss://.../ws.
        """Test that GenericEnvClient can be instantiated with HTTPS URL."""
        client = GenericEnvClient(base_url="https://example.com")
        assert client._ws_url == "wss://example.com/ws"

    def test_instantiation_with_ws_url(self):
        # [FUNCTION DOC]
        # ws URLs get the /ws path.
        """Test that GenericEnvClient can be instantiated with WS URL."""
        client = GenericEnvClient(base_url="ws://localhost:8000")
        assert client._ws_url == "ws://localhost:8000/ws"

    def test_instantiation_with_custom_timeouts(self):
        # [FUNCTION DOC]
        # Connect and message timeouts are stored.
        """Test custom timeout parameters."""
        client = GenericEnvClient(
            base_url="http://localhost:8000",
            connect_timeout_s=30.0,
            message_timeout_s=120.0,
        )
        assert client._connect_timeout == 30.0
        assert client._message_timeout == 120.0

    def test_instantiation_with_provider_only(self, mock_provider):
        # [FUNCTION DOC]
        # Provider-only construction has no ws URL yet.
        """Test provider-owned startup without an initial base URL."""
        client = GenericEnvClient(provider=mock_provider)
        assert client._ws_url is None

    def test_instantiation_requires_base_url_or_provider(self):
        # [FUNCTION DOC]
        # No URL and no provider raises ValueError.
        """Test that clients need either a URL or a provider."""
        with pytest.raises(ValueError, match="requires either base_url or provider"):
            GenericEnvClient()

    @pytest.mark.asyncio
    async def test_connect_starts_provider_when_base_url_omitted(self, mock_provider):
        # [FUNCTION DOC]
        # connect starts the provider with no args, waits for readiness, sets the ws URL; close
        # stops it and clears the URL.
        """The outer client can start a provider that owns its image/source."""
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        with patch("openenv.core.env_client.ws_connect", AsyncMock(return_value=ws)):
            client = GenericEnvClient(provider=mock_provider)
            await client.connect()

            mock_provider.start_container.assert_called_once_with()
            mock_provider.wait_for_ready.assert_called_once_with(
                "http://localhost:8000"
            )
            assert client._ws_url == "ws://localhost:8000/ws"

            await client.close()

        mock_provider.stop_container.assert_called_once_with()
        assert client._ws_url is None

    @pytest.mark.asyncio
    async def test_provider_owned_startup_requires_no_arg_start_container(self):
        # [FUNCTION DOC]
        # A provider whose start_container requires an image raises a guiding ValueError.
        """Providers that require an image fail with actionable guidance."""

        class _RequiresImageProvider:
            # [CLASS DOC]
            # HELPER: Provider whose start/wait must never be called.
            def start_container(self, image):
                # [FUNCTION DOC]
                # Fails the test if called.
                raise AssertionError("start_container should not be called")

            def wait_for_ready(self, base_url):
                # [FUNCTION DOC]
                # Fails the test if called.
                raise AssertionError("wait_for_ready should not be called")

            def stop_container(self):
                # [FUNCTION DOC]
                # No-op.
                pass

        client = GenericEnvClient(provider=_RequiresImageProvider())

        with pytest.raises(ValueError, match=r"start_container\(\) requires image"):
            await client.connect()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("connect_parent", [False, True])
    async def test_new_session_reuses_provider_server(
        self, mock_provider, connect_parent
    ):
        # [FUNCTION DOC]
        # Child sessions share the started server URL but do not own the provider; the provider
        # starts once and stops when the parent closes.
        """Child sessions connect to the same server without owning the provider."""
        websockets = []

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns a new AsyncMock socket and records it.
            ws = AsyncMock()
            websockets.append(ws)
            return ws

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            client = GenericEnvClient(provider=mock_provider)
            if connect_parent:
                await client.connect()
            session = await client.new_session()

            assert isinstance(session, GenericEnvClient)
            assert session is not client
            assert session._provider is None
            assert session._base_url == "http://localhost:8000"
            assert session._ws_url == "ws://localhost:8000/ws"
            assert len(websockets) == 1 + int(connect_parent)
            mock_provider.start_container.assert_called_once_with()

            await client.close()

        mock_provider.stop_container.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "start_method,stop_method",
        [("start_container", "stop_container"), ("start", "stop")],
    )
    @pytest.mark.parametrize("cleanup_fails", [False, True])
    async def test_new_session_stops_provider_when_readiness_fails(
        self, start_method, stop_method, cleanup_fails
    ):
        # [FUNCTION DOC]
        # Parametrized over method naming and cleanup failure: the readiness error propagates
        # unchanged and stop is called once with no connection.
        provider = Mock(spec=[start_method, stop_method, "wait_for_ready"])
        start = getattr(provider, start_method)
        stop = getattr(provider, stop_method)
        start.return_value = "http://localhost:8000"
        error = TimeoutError("environment never became ready")
        provider.wait_for_ready.side_effect = error
        if cleanup_fails:
            stop.side_effect = RuntimeError("cleanup failed")
        client = GenericEnvClient(provider=provider)

        with patch("openenv.core.env_client.ws_connect", AsyncMock()) as connect:
            with pytest.raises(TimeoutError) as exc_info:
                await client.new_session()

        assert exc_info.value is error
        start.assert_called_once_with()
        stop.assert_called_once_with()
        connect.assert_not_called()
        assert client.base_url is None
        assert client._child_clients == []

    @pytest.mark.asyncio
    async def test_close_stops_provider_when_child_close_raises(self, mock_provider):
        # [FUNCTION DOC]
        # A failing child close does not prevent stopping the provider.
        """Parent cleanup continues even when a child session close fails."""
        client = GenericEnvClient(provider=mock_provider)
        child = MagicMock()
        child.close = AsyncMock(side_effect=RuntimeError("child close failed"))
        client._child_clients.append(child)

        await client.close()

        child.close.assert_awaited_once_with()
        assert client._child_clients == []
        mock_provider.stop_container.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_cancelled_child_close_still_closes_all_children_and_parent(self):
        # [FUNCTION DOC]
        # Cancelling _close_async while the first child closes still waits for every child, then
        # closes the parent socket, stops the provider and clears URLs, finally re-raising
        # CancelledError.
        """Cancellation waits for every child before parent teardown."""

        class FakeRuntimeProvider:
            # [CLASS DOC]
            # HELPER: Provider with a stop method recording stopped.
            def __init__(self):
                # [FUNCTION DOC]
                # Sets stopped False.
                self.stopped = False

            def stop(self):
                # [FUNCTION DOC]
                # Sets stopped True.
                self.stopped = True

        first_close_started = asyncio.Event()
        release_first_close = asyncio.Event()
        second_close_completed = asyncio.Event()

        class FirstChild:
            # [CLASS DOC]
            # HELPER: Child whose close blocks until released.
            def __init__(self):
                # [FUNCTION DOC]
                # Sets closed False.
                self.closed = False

            async def close(self):
                # [FUNCTION DOC]
                # Signals start, waits for release, marks closed.
                first_close_started.set()
                await release_first_close.wait()
                self.closed = True

        class SecondChild:
            # [CLASS DOC]
            # HELPER: Child whose close completes immediately.
            def __init__(self):
                # [FUNCTION DOC]
                # Sets closed False.
                self.closed = False

            async def close(self):
                # [FUNCTION DOC]
                # Marks closed and signals completion.
                self.closed = True
                second_close_completed.set()

        class ParentSocket:
            # [CLASS DOC]
            # HELPER: Open socket stand-in.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # No-op.
                pass

            async def close(self):
                # [FUNCTION DOC]
                # Sets state CLOSED.
                self.state = State.CLOSED

        provider = FakeRuntimeProvider()
        client = GenericEnvClient(provider=provider)
        client._base_url = "http://localhost:8000"
        client._ws_url = "ws://localhost:8000/ws"
        parent_ws = ParentSocket()
        client._ws = parent_ws
        client._ws_loop = asyncio.get_running_loop()
        first_child = FirstChild()
        second_child = SecondChild()
        client._child_clients.extend([first_child, second_child])

        close_call = asyncio.create_task(client._close_async())
        await first_close_started.wait()
        close_call.cancel()

        await asyncio.wait_for(second_close_completed.wait(), timeout=1)
        assert not close_call.done()
        assert parent_ws.state != State.CLOSED
        assert not provider.stopped

        release_first_close.set()
        close_results = await asyncio.wait_for(
            asyncio.gather(close_call, return_exceptions=True), timeout=1
        )

        assert len(close_results) == 1
        assert isinstance(close_results[0], asyncio.CancelledError)
        assert first_child.closed
        assert second_child.closed
        assert client._child_clients == []
        assert client._ws is None
        assert parent_ws.state == State.CLOSED
        assert provider.stopped
        assert client._base_url is None
        assert client._ws_url is None

    def test_session_client_filters_constructor_kwargs(self):
        # [FUNCTION DOC]
        # _create_session_client works for subclasses whose __init__ only takes base_url.
        """Child creation respects subclasses with narrower __init__ signatures."""

        class MinimalGenericClient(GenericEnvClient):
            # [CLASS DOC]
            # HELPER: Subclass accepting only base_url.
            def __init__(self, base_url):
                # [FUNCTION DOC]
                # Passes base_url to GenericEnvClient.
                super().__init__(base_url=base_url)

        client = MinimalGenericClient(base_url="http://localhost:8000")
        child = client._create_session_client()

        assert isinstance(child, MinimalGenericClient)
        assert child._base_url == "http://localhost:8000"


class TestGenericEnvClientStepPayload:
    # [CLASS DOC]
    # Tests _step_payload.
    """Test _step_payload method."""

    def test_step_payload_passthrough(self):
        # [FUNCTION DOC]
        # The action dict is passed through unchanged.
        """Test that action dict passes through unchanged."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {"code": "print('hello')", "timeout": 30}
        payload = client._step_payload(action)

        assert payload == action
        assert payload["code"] == "print('hello')"
        assert payload["timeout"] == 30

    def test_step_payload_empty_dict(self):
        # [FUNCTION DOC]
        # Empty dict stays empty.
        """Test with empty action dict."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {}
        payload = client._step_payload(action)

        assert payload == {}

    def test_step_payload_nested_dict(self):
        # [FUNCTION DOC]
        # Nested dicts pass through.
        """Test with nested action dict."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {
            "command": "execute",
            "params": {"file": "test.py", "args": ["--verbose"]},
        }
        payload = client._step_payload(action)

        assert payload == action
        assert payload["params"]["file"] == "test.py"


class TestGenericEnvClientParseResult:
    # [CLASS DOC]
    # Tests _parse_result.
    """Test _parse_result method."""

    def test_parse_result_full_payload(self):
        # [FUNCTION DOC]
        # Observation, reward and done are mapped into a StepResult.
        """Test parsing a complete result payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "observation": {"stdout": "hello", "stderr": ""},
            "reward": 1.5,
            "done": True,
        }
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert result.observation == {"stdout": "hello", "stderr": ""}
        assert result.reward == 1.5
        assert result.done is True

    def test_parse_result_minimal_payload(self):
        # [FUNCTION DOC]
        # Defaults: empty observation, reward None, done False.
        """Test parsing a minimal payload with defaults."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {}
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert result.observation == {}
        assert result.reward is None
        assert result.done is False

    def test_parse_result_missing_reward(self):
        # [FUNCTION DOC]
        # Missing reward is None.
        """Test parsing payload without reward."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {"observation": {"data": "test"}, "done": False}
        result = client._parse_result(payload)

        assert result.observation == {"data": "test"}
        assert result.reward is None
        assert result.done is False


class TestGenericEnvClientParseState:
    # [CLASS DOC]
    # Tests _parse_state.
    """Test _parse_state method."""

    def test_parse_state_full_payload(self):
        # [FUNCTION DOC]
        # The state dict is returned as-is.
        """Test parsing a complete state payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "episode_id": "ep-123",
            "step_count": 5,
            "custom_field": "value",
        }
        state = client._parse_state(payload)

        assert state == payload
        assert state["episode_id"] == "ep-123"
        assert state["step_count"] == 5

    def test_parse_state_empty_payload(self):
        # [FUNCTION DOC]
        # Empty state stays empty.
        """Test parsing empty state payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {}
        state = client._parse_state(payload)

        assert state == {}


class TestGenericEnvClientFromDockerImage:
    # [CLASS DOC]
    # Tests from_docker_image (connect patched).
    """Test from_docker_image class method."""

    @pytest.mark.asyncio
    async def test_from_docker_image_creates_client(self, mock_provider):
        # [FUNCTION DOC]
        # Starts the image via the provider and waits for readiness.
        """Test that from_docker_image creates a connected client."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once_with("coding-env:latest")
            mock_provider.wait_for_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_docker_image_with_env_vars(self, mock_provider):
        # [FUNCTION DOC]
        # env_vars are passed to start_container.
        """Test from_docker_image with environment variables."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
                env_vars={"DEBUG": "1"},
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once_with(
                "coding-env:latest", env_vars={"DEBUG": "1"}
            )


class TestGenericEnvClientFromEnv:
    # [CLASS DOC]
    # Tests from_env (connect patched).
    """Test from_env class method (HuggingFace registry)."""

    @pytest.mark.asyncio
    async def test_from_env_with_docker(self, mock_provider):
        # [FUNCTION DOC]
        # use_docker builds a registry.hf.space image reference.
        """Test from_env with use_docker=True pulls from HF registry."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_env(
                "user/my-env",
                use_docker=True,
                provider=mock_provider,
            )

            assert isinstance(client, GenericEnvClient)
            # Should construct HF registry URL
            mock_provider.start_container.assert_called_once()
            call_args = mock_provider.start_container.call_args
            assert "registry.hf.space/user-my-env" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_from_env_uv_wait_uses_remaining_context_timeout(self):
        # [FUNCTION DOC]
        # For UV providers, readiness gets the context timeout minus time spent in start (6.5 s).
        """A slow `start()` must consume part of the UV readiness budget."""
        provider = Mock()
        provider.context_timeout_s = 10.0
        provider.start.return_value = "http://localhost:8000"
        provider.wait_for_ready.return_value = None

        with (
            patch.object(GenericEnvClient, "connect", new_callable=AsyncMock),
            patch("openenv.core.env_client.time.monotonic", side_effect=[100.0, 103.5]),
        ):
            client = await GenericEnvClient.from_env(
                "user/my-env",
                use_docker=False,
                provider=provider,
            )

        assert isinstance(client, GenericEnvClient)
        provider.wait_for_ready.assert_called_once_with(timeout_s=6.5)


class TestSyncBootstrapConstructors:
    # [CLASS DOC]
    # Tests the lazy factory handles and their .sync() bootstrap path.
    """Test the synchronous `.sync()` bootstrap chain on the client factories.

    `from_docker_image(...).sync()` / `from_env(...).sync()` let a synchronous
    consumer (e.g. a TRL GRPO rollout loop) bootstrap a container without an
    event loop and without hand-rolling provider.start_container() /
    wait_for_ready(), while `await from_docker_image(...)` keeps the async path.
    See issue #935.
    """

    @staticmethod
    def _fake_ws_connect():
        # [FUNCTION DOC]
        # HELPER: Returns (async connect function creating and recording AsyncMock sockets, list of sockets).
        """A ws_connect stand-in that records every socket it opens."""
        sockets = []

        async def _connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Creates, records and returns an AsyncMock socket.
            ws = AsyncMock()
            sockets.append(ws)
            return ws

        return _connect, sockets

    def test_from_docker_image_is_lazy_until_resolved(self, mock_provider):
        # [FUNCTION DOC]
        # Calling the factory starts nothing until resolved.
        """The factory returns a handle and starts nothing until driven/.sync()'d."""
        handle = GenericEnvClient.from_docker_image(
            image="coding-env:latest",
            provider=mock_provider,
        )
        # Nothing is started at call time — the container starts only on resolve.
        mock_provider.start_container.assert_not_called()
        # Clean up the undriven coroutine to avoid a "never awaited" warning.
        handle.close()

    def test_bootstrap_handle_is_single_use(self, mock_provider):
        # [FUNCTION DOC]
        # Resolving a handle twice raises "already been resolved"; only one start.
        """Resolving a handle twice raises instead of silently double-starting."""
        fake_connect, _ = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            handle = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            )
            client = handle.sync()
            with pytest.raises(RuntimeError, match="already been resolved"):
                handle.sync()
            client.close()

        # Only one container was ever started.
        mock_provider.start_container.assert_called_once()

    def test_sync_bootstrap_stops_provider_if_loop_setup_fails(self, mock_provider):
        # [FUNCTION DOC]
        # If _run_sync fails after start, the provider is stopped.
        """If sync setup fails after the container starts, the provider is released.

        Exercises the failure window in `.sync()` before `_connect_async` (and its
        own cleanup) runs — here `_run_sync` itself raises.
        """
        with patch.object(
            GenericEnvClient, "_run_sync", side_effect=RuntimeError("loop boom")
        ):
            with pytest.raises(RuntimeError, match="loop boom"):
                GenericEnvClient.from_docker_image(
                    image="coding-env:latest",
                    provider=mock_provider,
                ).sync()

        mock_provider.start_container.assert_called_once()
        mock_provider.stop_container.assert_called_once_with()

    def test_factory_handle_is_coroutine_compatible(self, mock_provider):
        # [FUNCTION DOC]
        # The handle is a real coroutine accepted by asyncio.run and run_async_safely.
        """The handle is a real coroutine, so asyncio.run / run_async_safely accept it.

        The previous async factories returned a coroutine; the handle must remain a
        drop-in so `asyncio.run(from_docker_image(...))` and
        `run_async_safely(from_docker_image(...))` keep working (regression guard).
        """
        import asyncio as _asyncio

        from openenv.core.utils import run_async_safely

        fake_connect, _ = self._fake_ws_connect()

        # asyncio.run(...) strictly requires a coroutine; a bare awaitable is rejected.
        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            handle = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            )
            assert _asyncio.iscoroutine(handle)
            client = _asyncio.run(handle)
            assert isinstance(client, GenericEnvClient)

        # run_async_safely (asyncio.run under the hood) must accept it too.
        fake_connect, _ = self._fake_ws_connect()
        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client2 = run_async_safely(
                GenericEnvClient.from_docker_image(
                    image="coding-env:latest",
                    provider=mock_provider,
                )
            )
            assert isinstance(client2, GenericEnvClient)

    def test_from_docker_image_sync_returns_connected_client(self, mock_provider):
        # [FUNCTION DOC]
        # .sync() returns a connected SyncEnvClient (one socket, base_url proxied); close stops the container.
        """from_docker_image(...).sync() returns a connected SyncEnvClient (no await)."""
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            ).sync()

            # A concrete sync wrapper, consistent with the instance .sync().
            assert isinstance(client, SyncEnvClient)
            mock_provider.start_container.assert_called_once_with("coding-env:latest")
            mock_provider.wait_for_ready.assert_called_once()
            # Connection was actually established on the sync background loop.
            assert len(sockets) == 1
            # base_url proxies through to the wrapped async client.
            assert client.base_url == "http://localhost:8000"

            client.close()

        mock_provider.stop_container.assert_called_once_with()

    def test_from_docker_image_sync_forwards_start_kwargs(self, mock_provider):
        # [FUNCTION DOC]
        # Extra kwargs reach start_container.
        """Extra kwargs reach provider.start_container()."""
        fake_connect, _ = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
                env_vars={"DEBUG": "1"},
            ).sync()
            mock_provider.start_container.assert_called_once_with(
                "coding-env:latest", env_vars={"DEBUG": "1"}
            )
            client.close()

    def test_from_env_sync_with_docker(self, mock_provider):
        # [FUNCTION DOC]
        # from_env(...).sync() with Docker uses the HF registry image and connects once.
        """from_env(...).sync() with use_docker=True pulls from the HF registry, no await."""
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_env(
                "user/my-env",
                use_docker=True,
                provider=mock_provider,
            ).sync()

            assert isinstance(client, SyncEnvClient)
            call_args = mock_provider.start_container.call_args
            assert "registry.hf.space/user-my-env" in call_args[0][0]
            assert len(sockets) == 1

            client.close()

    def test_from_env_sync_uv_stops_provider_on_start_failure(self):
        # [FUNCTION DOC]
        # A UV start failure stops the provider before re-raising.
        """A UV start() failure releases the provider before re-raising (sync path)."""
        provider = Mock()
        provider.context_timeout_s = None
        provider.start.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            GenericEnvClient.from_env(
                "user/my-env",
                use_docker=False,
                provider=provider,
            ).sync()

        provider.stop.assert_called_once_with()

    def test_from_env_uv_stops_provider_on_construction_failure(self, monkeypatch):
        # [FUNCTION DOC]
        # An invalid OPENENV_CLIENT_MODE after a UV start stops the provider (ValueError "Invalid mode").
        """A client-construction failure after a successful start() releases the UV provider.

        The provider start()/wait succeed, but `EnvClient.__init__` rejects an
        invalid OPENENV_CLIENT_MODE — the spawned process must still be stopped.
        """
        monkeypatch.setenv("OPENENV_CLIENT_MODE", "not-a-valid-mode")
        provider = Mock()
        provider.context_timeout_s = None
        provider.start.return_value = "http://localhost:8000"
        provider.wait_for_ready.return_value = None

        with pytest.raises(ValueError, match="Invalid mode"):
            GenericEnvClient.from_env(
                "user/my-env",
                use_docker=False,
                provider=provider,
            ).sync()

        provider.stop.assert_called_once_with()

    def test_from_env_docker_stops_container_on_construction_failure(
        self, mock_provider, monkeypatch
    ):
        # [FUNCTION DOC]
        # Same for the Docker path: the container is stopped.
        """A client-construction failure after start_container releases the container."""
        monkeypatch.setenv("OPENENV_CLIENT_MODE", "not-a-valid-mode")

        with pytest.raises(ValueError, match="Invalid mode"):
            GenericEnvClient.from_env(
                "user/my-env",
                use_docker=True,
                provider=mock_provider,
            ).sync()

        mock_provider.stop_container.assert_called_once_with()

    def test_from_docker_image_stops_container_on_construction_failure(
        self, mock_provider, monkeypatch
    ):
        # [FUNCTION DOC]
        # from_docker_image stops the container if client construction fails.
        """from_docker_image releases the container if client construction fails."""
        monkeypatch.setenv("OPENENV_CLIENT_MODE", "not-a-valid-mode")

        with pytest.raises(ValueError, match="Invalid mode"):
            GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            ).sync()

        mock_provider.stop_container.assert_called_once_with()

    _RESET_RESPONSE = (
        '{"type": "response", "data": {"observation": {"ready": true}, '
        '"reward": 0.0, "done": false}}'
    )

    def test_documented_sync_usage_with_block_and_reset(self, mock_provider):
        # [FUNCTION DOC]
        # The documented sync pattern (factory .sync(), with block, reset) returns a concrete
        # StepResult, reuses the one socket and stops the container on exit.
        """The Environment Builder docs' sync example works end-to-end (issue #1118).

        `client = MyEnv.from_docker_image(...).sync()` followed by
        `with client: client.reset()` must connect on the sync loop, return a
        concrete StepResult (not a coroutine), and release the container on exit.
        """
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_docker_image(
                image="my-env:latest",
                provider=mock_provider,
            ).sync()
            sockets[0].recv.return_value = self._RESET_RESPONSE

            with client:
                result = client.reset()

        if asyncio.iscoroutine(result):
            result.close()
            pytest.fail("client.reset() returned a coroutine in synchronous code")
        assert isinstance(result, StepResult)
        assert result.observation == {"ready": True}
        assert result.done is False
        # `with` on the eagerly connected sync handle must reuse its socket.
        assert len(sockets) == 1
        mock_provider.start_container.assert_called_once_with("my-env:latest")
        mock_provider.stop_container.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_documented_async_usage_async_with_and_reset(self, mock_provider):
        # [FUNCTION DOC]
        # The documented async pattern (await factory, async with, await reset) reuses the socket
        # and stops the container on exit.
        """The Environment Builder docs' async example works end-to-end (issue #1118).

        `client = await MyEnv.from_docker_image(...)` followed by
        `async with client: await client.reset()` must reuse the connection
        opened by the handle and release the container on exit.
        """
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = await GenericEnvClient.from_docker_image(
                image="my-env:latest",
                provider=mock_provider,
            )
            assert isinstance(client, GenericEnvClient)
            sockets[0].recv.return_value = self._RESET_RESPONSE

            async with client:
                result = await client.reset()

        assert isinstance(result, StepResult)
        assert result.observation == {"ready": True}
        # `async with` on an already-connected client must not open a second socket.
        assert len(sockets) == 1
        mock_provider.start_container.assert_called_once_with("my-env:latest")
        mock_provider.stop_container.assert_called_once_with()


class TestBaseUrlProperty:
    # [CLASS DOC]
    # Tests the read-only base_url property.
    """Test the public read-only base_url property (issue #935)."""

    def test_base_url_returns_normalized_url(self):
        # [FUNCTION DOC]
        # Trailing slashes are stripped.
        """base_url exposes the normalized URL with any trailing slash stripped."""
        client = GenericEnvClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_base_url_is_none_before_provider_start(self, mock_provider):
        # [FUNCTION DOC]
        # Provider-only clients have base_url None until started.
        """A provider-only client has no URL until it connects/starts its container."""
        client = GenericEnvClient(provider=mock_provider)
        assert client.base_url is None

    def test_base_url_is_read_only(self):
        # [FUNCTION DOC]
        # Assigning base_url raises AttributeError.
        """base_url is a read-only property; callers must not reassign it."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        with pytest.raises(AttributeError):
            client.base_url = "http://evil:9999"  # type: ignore[misc]


# ============================================================================
# AutoEnv skip_install Integration Tests
# ============================================================================


class TestAutoEnvSkipInstall:
    # [CLASS DOC]
    # Tests AutoEnv.from_env with skip_install (availability checks patched).
    """Test AutoEnv.from_env() with skip_install parameter."""

    def test_skip_install_with_base_url(self):
        # [FUNCTION DOC]
        # A reachable base_url yields a GenericEnvClient.
        """Test skip_install=True with explicit base_url."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            client = AutoEnv.from_env(
                "echo",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

    def test_skip_install_with_unavailable_server(self):
        # [FUNCTION DOC]
        # An unreachable base_url raises ConnectionError "Server not available".
        """Test skip_install=True with unavailable server raises error."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=False):
            with pytest.raises(ConnectionError) as exc_info:
                AutoEnv.from_env(
                    "echo",
                    base_url="http://localhost:8000",
                    skip_install=True,
                )

            assert "Server not available" in str(exc_info.value)

    def test_skip_install_with_hub_url_and_running_space(self):
        # [FUNCTION DOC]
        # A running Space yields a GenericEnvClient.
        """Test skip_install=True with HF Space that is running."""
        from openenv.auto.auto_env import AutoEnv

        with (
            patch.object(AutoEnv, "_check_space_availability", return_value=True),
            patch.object(
                AutoEnv,
                "_resolve_space_url",
                return_value="https://user-my-env.hf.space",
            ),
        ):
            client = AutoEnv.from_env(
                "user/my-env",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

    def test_skip_install_with_hub_url_and_docker(self, mock_provider):
        # [FUNCTION DOC]
        # A non-running Space falls back to GenericEnvClient.from_env with use_docker True.
        """Test skip_install=True with HF Space not running uses Docker."""
        from openenv.auto.auto_env import AutoEnv

        # AutoEnv resolves the from_env bootstrap handle via run_async_safely, so
        # an awaitable stand-in returning a connected client is a valid double.
        async def mock_from_env_async(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Async stand-in for from_env returning a GenericEnvClient.
            return GenericEnvClient(base_url="http://localhost:8000")

        with (
            patch.object(AutoEnv, "_check_space_availability", return_value=False),
            patch.object(
                AutoEnv,
                "_resolve_space_url",
                return_value="https://user-my-env.hf.space",
            ),
            patch.object(
                GenericEnvClient,
                "from_env",
                side_effect=mock_from_env_async,
            ) as mock_from_env,
        ):
            AutoEnv.from_env(
                "user/my-env",
                skip_install=True,
            )

            mock_from_env.assert_called_once()
            call_kwargs = mock_from_env.call_args[1]
            assert call_kwargs.get("use_docker") is True

    def test_skip_install_local_env_without_docker_image_raises(self):
        # [FUNCTION DOC]
        # A local name with neither base_url nor docker_image raises ValueError.
        """Test skip_install=True for local env without docker_image raises error."""
        from openenv.auto.auto_env import AutoEnv

        with pytest.raises(ValueError) as exc_info:
            AutoEnv.from_env(
                "echo",  # Local name, not Hub URL
                skip_install=True,
            )

        error_msg = str(exc_info.value)
        assert "skip_install=True" in error_msg
        assert "base_url" in error_msg or "docker_image" in error_msg

    def test_skip_install_local_env_with_docker_image(self, mock_provider):
        # [FUNCTION DOC]
        # A local name with docker_image starts the container via the given provider.
        """Test skip_install=True for local env with docker_image."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = AutoEnv.from_env(
                "echo",
                docker_image="echo-env:latest",
                container_provider=mock_provider,
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once()

    def test_skip_install_false_still_works(self):
        # [FUNCTION DOC]
        # Without skip_install, discovery returns the typed client class instance.
        """Test that skip_install=False (default) still works as before."""
        from openenv.auto._discovery import EnvironmentInfo, reset_discovery
        from openenv.auto.auto_env import AutoEnv

        reset_discovery()

        mock_env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo environment",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
            spec_version=1,
        )

        mock_discovery = Mock()
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_discovery.discover.return_value = {"echo": mock_env_info}

        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_env_info.get_client_class = Mock(return_value=mock_client_class)

        with (
            patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery),
            patch.object(AutoEnv, "_check_server_availability", return_value=True),
        ):
            result = AutoEnv.from_env(
                "echo",
                base_url="http://localhost:8000",
                skip_install=False,  # Explicit False
            )

            # Should return the typed client, not GenericEnvClient
            assert result is mock_client_instance
            assert not isinstance(result, GenericEnvClient)


# ============================================================================
# Comparison Tests: GenericEnvClient vs Typed Client
# ============================================================================


class TestGenericVsTypedComparison:
    # [CLASS DOC]
    # Compares GenericEnvClient with typed clients.
    """Compare behavior of GenericEnvClient vs typed clients."""

    def test_step_payload_generic_vs_typed(self):
        # [FUNCTION DOC]
        # Payload is the plain dict.
        """Compare step payload generation."""
        # GenericEnvClient - pass through
        generic_client = GenericEnvClient(base_url="http://localhost:8000")
        generic_payload = generic_client._step_payload({"message": "hello"})

        # Should be identical to what a typed client would produce
        assert generic_payload == {"message": "hello"}

    def test_parse_result_generic_returns_dict(self):
        # [FUNCTION DOC]
        # Observations are dicts accessed by key.
        """GenericEnvClient returns dict observation."""
        generic_client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "observation": {"echoed_message": "hello", "length": 5},
            "reward": 1.0,
            "done": False,
        }
        result = generic_client._parse_result(payload)

        # Observation is a dict, not a typed object
        assert isinstance(result.observation, dict)
        assert result.observation["echoed_message"] == "hello"
        # Access is via dict keys, not attributes
        assert result.observation.get("length") == 5


# ============================================================================
# Import Tests
# ============================================================================


class TestGenericEnvClientImports:
    # [CLASS DOC]
    # Tests GenericEnvClient import locations.
    """Test that GenericEnvClient can be imported from various locations."""

    def test_import_from_core(self):
        # [FUNCTION DOC]
        # openenv.core exports it.
        """Test import from openenv.core."""
        from openenv.core import GenericEnvClient as GC1

        assert GC1 is GenericEnvClient

    def test_import_from_openenv(self):
        # [FUNCTION DOC]
        # openenv exports it.
        """Test import from openenv package."""
        from openenv import GenericEnvClient as GC2

        assert GC2 is GenericEnvClient

    def test_import_from_generic_client_module(self):
        # [FUNCTION DOC]
        # The module exports it.
        """Test direct import from module."""
        from openenv.core.generic_client import GenericEnvClient as GC3

        assert GC3 is GenericEnvClient


class TestSyncEnvClientImports:
    # [CLASS DOC]
    # Tests SyncEnvClient import locations.
    """Test that SyncEnvClient can be imported from various locations."""

    def test_import_from_core(self):
        # [FUNCTION DOC]
        # openenv.core exports it.
        """Test import from openenv.core."""
        from openenv.core import SyncEnvClient as SC1

        assert SC1 is SyncEnvClient

    def test_import_from_openenv(self):
        # [FUNCTION DOC]
        # openenv exports it.
        """Test import from openenv package."""
        from openenv import SyncEnvClient as SC2

        assert SC2 is SyncEnvClient

    def test_import_from_sync_client_module(self):
        # [FUNCTION DOC]
        # The module exports it.
        """Test direct import from module."""
        from openenv.core.sync_client import SyncEnvClient as SC3

        assert SC3 is SyncEnvClient


class TestSyncEnvClientWrapper:
    # [CLASS DOC]
    # Tests the SyncEnvClient wrapper.
    """Test SyncEnvClient wrapper functionality."""

    def test_sync_method_returns_sync_client(self):
        # [FUNCTION DOC]
        # .sync() wraps the same async client.
        """Test that .sync() returns a SyncEnvClient."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = client.sync()

        assert isinstance(sync_client, SyncEnvClient)
        assert sync_client._async is client

    def test_sync_client_has_async_client_property(self):
        # [FUNCTION DOC]
        # async_client exposes the wrapped client.
        """Test that SyncEnvClient exposes async_client property."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        assert sync_client.async_client is async_client

    def test_sync_client_delegates_payload_methods(self):
        # [FUNCTION DOC]
        # _step_payload is delegated.
        """Test that SyncEnvClient delegates _step_payload to async client."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        action = {"code": "print('hello')"}
        payload = sync_client._step_payload(action)

        assert payload == action

    def test_sync_client_delegates_parse_result(self):
        # [FUNCTION DOC]
        # _parse_result is delegated.
        """Test that SyncEnvClient delegates _parse_result to async client."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        payload = {
            "observation": {"output": "hello"},
            "reward": 1.0,
            "done": False,
        }
        result = sync_client._parse_result(payload)

        assert result.observation == {"output": "hello"}
        assert result.reward == 1.0

    def test_sync_client_new_session_reuses_provider_server(self, mock_provider):
        # [FUNCTION DOC]
        # Sync new_session returns a SyncEnvClient sharing the server without owning the provider.
        """The sync wrapper exposes child sessions without another context manager."""
        websockets = []

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            ws = AsyncMock()
            websockets.append(ws)
            return ws

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            with GenericEnvClient(provider=mock_provider).sync() as client:
                session = client.new_session()

                assert isinstance(session, SyncEnvClient)
                assert isinstance(session.async_client, GenericEnvClient)
                assert session.async_client._provider is None
                assert session.async_client._base_url == "http://localhost:8000"
                assert len(websockets) == 2

        mock_provider.start_container.assert_called_once_with()
        mock_provider.stop_container.assert_called_once_with()

    def test_sync_close_stops_provider_when_child_close_raises(self, mock_provider):
        # [FUNCTION DOC]
        # A failing child close does not prevent the provider stop.
        """Sync parent cleanup continues when child close fails."""
        client = GenericEnvClient(provider=mock_provider).sync()
        child = Mock()
        child.close.side_effect = RuntimeError("child close failed")
        client._child_clients.append(child)

        client.close()

        child.close.assert_called_once_with()
        assert client._child_clients == []
        mock_provider.stop_container.assert_called_once_with()

    def test_sync_call_after_async_connect_blocks_for_result(self, monkeypatch):
        # [FUNCTION DOC]
        # After an async connect in another loop, a sync reset returns a concrete result.
        """Sync calls after async setup return concrete results, not coroutines."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        async def connect_client():
            # [FUNCTION DOC]
            # HELPER: Connects the client with _connect_async patched and checks it was awaited.
            with patch.object(
                GenericEnvClient, "_connect_async", new_callable=AsyncMock
            ) as mock_connect:
                await client.connect()
                mock_connect.assert_awaited_once()

        asyncio.run(connect_client())
        assert client._execution_mode == "async"

        expected = StepResult(
            observation={"ready": True},
            reward=0.0,
            done=False,
        )

        async def fake_reset(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Replacement _reset_async returning the expected StepResult (and recording the
            # running loop where the test checks it).
            return expected

        monkeypatch.setattr(client, "_reset_async", fake_reset)

        result = client.reset()
        if asyncio.iscoroutine(result):
            result.close()
            pytest.fail("client.reset() returned a coroutine in synchronous code")

        assert result is expected

    @pytest.mark.asyncio
    async def test_sync_locked_dispatch_inside_running_loop_uses_sync_loop(
        self, monkeypatch
    ):
        # [FUNCTION DOC]
        # A sync-locked client called inside a running loop executes on the wrapper's own loop.
        """Sync-locked clients keep work on the sync wrapper loop."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = client.sync()

        async def fake_connect():
            # [FUNCTION DOC]
            # HELPER: Replacement _connect_async that just returns the client.
            return client

        monkeypatch.setattr(client, "_connect_async", fake_connect)
        sync_client.connect()

        expected = StepResult(
            observation={"ready": True},
            reward=0.0,
            done=False,
        )
        observed = {}

        async def fake_reset(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Replacement _reset_async returning the expected StepResult (and recording the
            # running loop where the test checks it).
            observed["loop"] = asyncio.get_running_loop()
            return expected

        monkeypatch.setattr(client, "_reset_async", fake_reset)

        try:
            current_loop = asyncio.get_running_loop()
            result = client.reset()
            if asyncio.iscoroutine(result):
                result.close()
                pytest.fail("sync-locked client.reset() returned a coroutine")

            assert result is expected
            assert observed["loop"] is sync_client._loop
            assert observed["loop"] is not current_loop
        finally:
            sync_client.close()


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestGenericEnvClientContextManager:
    # [CLASS DOC]
    # Tests context manager entry/exit.
    """Test context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_enter_exit(self):
        # [FUNCTION DOC]
        # async with connects and closes once.
        """Test that async context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "connect", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "close", new_callable=AsyncMock
            ) as mock_close,
        ):
            async with GenericEnvClient(base_url="http://localhost:8000") as client:
                assert isinstance(client, GenericEnvClient)
                mock_connect.assert_called_once()

            mock_close.assert_called_once()

    def test_sync_context_manager_enter_exit(self):
        # [FUNCTION DOC]
        # Plain with on the async client connects and closes via the async internals.
        """Test that sync context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "_connect_async", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "_close_async", new_callable=AsyncMock
            ) as mock_close,
        ):
            client = GenericEnvClient(base_url="http://localhost:8000")

            with client as active_client:
                assert active_client is client
                mock_connect.assert_called_once()

            mock_close.assert_called_once()

    def test_sync_wrapper_context_manager(self):
        # [FUNCTION DOC]
        # with on the sync wrapper connects and closes.
        """Test SyncEnvClient context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "_connect_async", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "_close_async", new_callable=AsyncMock
            ) as mock_close,
        ):
            async_client = GenericEnvClient(base_url="http://localhost:8000")
            sync_client = async_client.sync()

            with sync_client as client:
                assert isinstance(client, SyncEnvClient)
                mock_connect.assert_called_once()

            mock_close.assert_called_once()


# ============================================================================
# Connection Helper Tests
# ============================================================================


class TestIsLocalhostWsUrl:
    # [CLASS DOC]
    # Tests _is_localhost_ws_url.
    """Test the loopback-host detection used to decide whether to bypass proxies."""

    @pytest.mark.parametrize(
        "ws_url",
        [
            "ws://localhost:8000/ws",
            "ws://127.0.0.1:8000/ws",
            "ws://127.5.5.5:8000/ws",  # anywhere in 127.0.0.0/8 is loopback
            "ws://[::1]:8000/ws",
            "wss://LocalHost:8000/ws",  # case-insensitive
        ],
    )
    def test_loopback_hosts_are_localhost(self, ws_url):
        # [FUNCTION DOC]
        # Parametrized loopback hosts (incl. 127/8, ::1, mixed case) are detected.
        assert _is_localhost_ws_url(ws_url) is True

    @pytest.mark.parametrize(
        "ws_url",
        [
            "ws://my-localhost-proxy.example.com:8000/ws",  # substring, not the host
            "ws://127.0.0.1.example.com:8000/ws",  # substring, not the host
            "ws://localhost.attacker.com:8000/ws",
            "wss://remote.example.com:8000/ws",
        ],
    )
    def test_remote_hosts_are_not_localhost(self, ws_url):
        # [FUNCTION DOC]
        # Parametrized hosts merely containing "localhost"/"127.0.0.1" are not.
        assert _is_localhost_ws_url(ws_url) is False


class TestGenericEnvClientConnection:
    # [CLASS DOC]
    # Tests connection setup options.
    """Test connection setup helpers."""

    @pytest.mark.asyncio
    async def test_localhost_connect_disables_proxy_per_connection(self, monkeypatch):
        # [FUNCTION DOC]
        # Localhost connects pass proxy None and never touch NO_PROXY.
        """Localhost connections pass proxy=None instead of mutating NO_PROXY."""
        monkeypatch.setenv("NO_PROXY", "example.com")
        observed = {}

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            observed["proxy"] = kwargs.get("proxy", "MISSING")
            observed["no_proxy_during"] = os.environ.get("NO_PROXY")
            return MagicMock()

        client = GenericEnvClient(base_url="http://localhost:8000")

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        # Proxy is disabled via the per-connection argument, and the global
        # NO_PROXY env var is never touched.
        assert observed["proxy"] is None
        assert observed["no_proxy_during"] == "example.com"
        assert os.environ["NO_PROXY"] == "example.com"

    @pytest.mark.asyncio
    async def test_remote_connect_uses_default_proxy(self, monkeypatch):
        # [FUNCTION DOC]
        # Remote connects pass no proxy override.
        """Non-localhost connections leave proxy handling at the default."""
        monkeypatch.setenv("NO_PROXY", "example.com")
        observed = {}

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            observed["proxy"] = kwargs.get("proxy", "MISSING")
            return MagicMock()

        client = GenericEnvClient(base_url="ws://remote.example.com:8000")

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        # No explicit proxy override: websockets uses its default (env-derived).
        assert observed["proxy"] == "MISSING"
        assert os.environ["NO_PROXY"] == "example.com"

    @pytest.mark.asyncio
    async def test_websocket_ping_options_forward_to_connect(self):
        # [FUNCTION DOC]
        # Ping interval/timeout options are forwarded.
        observed = {}

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            observed["ping_interval"] = kwargs.get("ping_interval", "MISSING")
            observed["ping_timeout"] = kwargs.get("ping_timeout", "MISSING")
            return MagicMock()

        client = GenericEnvClient(
            base_url="ws://remote.example.com:8000",
            websocket_ping_interval_s=None,
            websocket_ping_timeout_s=None,
        )

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        assert observed == {"ping_interval": None, "ping_timeout": None}

    @pytest.mark.asyncio
    async def test_concurrent_localhost_connects_do_not_leak_no_proxy(
        self, monkeypatch
    ):
        # [FUNCTION DOC]
        # Two concurrent localhost connects leave NO_PROXY unset.
        """Concurrent connects must not corrupt the process-global NO_PROXY.

        The previous implementation mutated os.environ["NO_PROXY"] around the
        connect await; interleaved gather() calls could restore a stale value
        and leak the localhost entry. Disabling the proxy per-connection makes
        this race impossible.
        """
        monkeypatch.delenv("NO_PROXY", raising=False)

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            # Yield control so the event loop interleaves the two connects.
            await asyncio.sleep(0)
            return MagicMock()

        clients = [
            GenericEnvClient(base_url="http://localhost:8000"),
            GenericEnvClient(base_url="http://127.0.0.1:8000"),
        ]

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await asyncio.gather(*(c.connect() for c in clients))

        assert "NO_PROXY" not in os.environ


class TestForeignLoopReconnect:
    # [CLASS DOC]
    # Tests reconnection rules when the socket belongs to another event loop, and socket
    # dropping/closing after receive timeouts or cancellation.
    """`connect()` must not silently no-op onto a websocket bound to a
    different (often already-closed) event loop -- e.g. a client connected
    via `await Client.from_env(...)` inside `asyncio.run(...)`, then driven
    afterwards through `.sync()`'s own dedicated background loop.
    """

    @pytest.mark.asyncio
    async def test_same_loop_reconnect_is_a_noop(self):
        # [FUNCTION DOC]
        # A second connect on the same loop does not reconnect.
        """Calling connect() again on the loop that created _ws is still a
        cheap no-op -- only a genuinely different loop should reconnect.
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            return MagicMock()

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            await client.connect()
            await client.connect()

        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_foreign_loop_triggers_reconnect_not_noop(self):
        # [FUNCTION DOC]
        # If the socket's loop differs, connect opens a new socket bound to the current loop.
        """If _ws was bound to a different loop than the one connect() is
        now running on, connect() must drop the stale reference and
        establish a fresh connection on the current loop, rather than
        returning early as if already connected.
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        first_ws = MagicMock()

        async def fake_ws_connect_first(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: ws_connect stand-in returning the first socket.
            return first_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_first
        ):
            await client.connect()

        assert client._ws is first_ws

        # Simulate "connected on a different loop": the real scenario is two
        # different asyncio event loops, which isn't practical to spin up
        # for a unit test, so we directly substitute a sentinel loop object
        # that isn't the one currently running.
        client._ws_loop = object()

        second_ws = MagicMock()

        async def fake_ws_connect_second(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: ws_connect stand-in returning the replacement socket.
            return second_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_second
        ) as mock_connect:
            await client.connect()

        mock_connect.assert_called_once()
        assert client._ws is second_ws
        assert client._ws_loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_disconnect_clears_ws_loop(self):
        # [FUNCTION DOC]
        # disconnect clears both the socket and its loop.
        """disconnect() must clear _ws_loop along with _ws so a later
        connect() on the same loop doesn't think it's still connected.
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            return MagicMock()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        await client.disconnect()

        assert client._ws is None
        assert client._ws_loop is None

    @pytest.mark.asyncio
    async def test_disconnect_does_not_reconnect_foreign_loop_websocket(self):
        # [FUNCTION DOC]
        # disconnect of a foreign-loop socket neither reconnects nor touches the old socket.
        """disconnect() must not open a replacement socket just to close it."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        first_ws = AsyncMock()

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            return first_ws

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        client._ws_loop = object()

        with patch("openenv.core.env_client.ws_connect") as mock_connect:
            await client.disconnect()

        mock_connect.assert_not_called()
        first_ws.send.assert_not_called()
        first_ws.close.assert_not_called()
        assert client._ws is None
        assert client._ws_loop is None

    @pytest.mark.asyncio
    async def test_lazy_send_reconnects_without_explicit_connect_call(self):
        # [FUNCTION DOC]
        # _ensure_connected reconnects a foreign-loop socket.
        """A caller that never explicitly calls `.connect()` -- e.g.
        `client.sync().reset()` right after `from_env()`, which goes straight
        to `_send()` -> `_ensure_connected()` -- must still reconnect on a
        foreign loop. `_ensure_connected()` previously only called `connect()`
        when `self._ws is None`, which skipped the reconnect logic entirely
        whenever `_ws` was already set (the exact foreign-loop case).
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        first_ws = MagicMock()

        async def fake_ws_connect_first(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: ws_connect stand-in returning the first socket.
            return first_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_first
        ):
            await client.connect()

        # Simulate the foreign-loop state without an explicit connect() call
        # in between, mirroring from_env() handing off to .sync().
        client._ws_loop = object()

        second_ws = AsyncMock()

        async def fake_ws_connect_second(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: ws_connect stand-in returning the replacement socket.
            return second_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_second
        ) as mock_connect:
            await client._ensure_connected()

        mock_connect.assert_called_once()
        assert client._ws is second_ws
        assert client._ws_loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_receive_does_not_reconnect_after_request_was_sent(self):
        # [FUNCTION DOC]
        # If the socket closes after sending, the receive error propagates without reconnecting.
        """A response must be read from the socket that carried its request."""

        class SocketClosedAfterSend:
            # [CLASS DOC]
            # HELPER: Socket that closes on send and fails on recv.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                self.state = State.CLOSED

            async def recv(self):
                # [FUNCTION DOC]
                # Fake socket recv: either never answers in time (sleeps 10 s) or raises ConnectionError,
                # depending on the scenario.
                raise ConnectionError("original socket closed after send")

        client = GenericEnvClient(base_url="http://localhost:8000")
        original_ws = SocketClosedAfterSend()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.side_effect = AssertionError(
            "must not receive a response on a replacement socket"
        )

        with patch(
            "openenv.core.env_client.ws_connect", return_value=replacement_ws
        ) as mock_connect:
            with pytest.raises(
                ConnectionError, match="original socket closed after send"
            ):
                await client._send_and_receive({"type": "state"})

        mock_connect.assert_not_called()
        assert client._ws is original_ws

    @pytest.mark.asyncio
    async def test_receive_timeout_drops_socket_so_next_call_reconnects(self):
        # [FUNCTION DOC]
        # A message timeout drops the socket (closed in background) and the next call reconnects.
        """A response that arrives after we time out must not be read by the
        *next* call.

        Regression test for #1143: leaving the timed-out socket open let the
        server's late response for the abandoned request sit in the buffer,
        where the next `_receive()` silently read it as its own -- and every
        response after that was shifted by one for the life of the connection.
        """

        class NeverResponds:
            # [CLASS DOC]
            # HELPER: Socket whose recv never answers in time.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                pass

            async def recv(self):
                # [FUNCTION DOC]
                # Fake socket recv: either never answers in time (sleeps 10 s) or raises ConnectionError,
                # depending on the scenario.
                await asyncio.sleep(10)

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=0.05
        )
        original_ws = NeverResponds()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await client._send_and_receive({"type": "state"})

        assert client._ws is None
        # The socket drop no longer awaits close() inline (a slow close
        # handshake would otherwise stall the caller's own deadline -- see
        # test_receive_cancelled_by_outer_deadline_...); it's scheduled as
        # a background task instead. Give the loop one tick to run it.
        await asyncio.sleep(0)
        assert original_ws.state == State.CLOSED

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.return_value = '{"type": "state", "data": {}}'

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            await client._send_and_receive({"type": "state"})

        mock_connect.assert_called_once()
        assert client._ws is replacement_ws

    @pytest.mark.asyncio
    async def test_receive_cancelled_by_outer_deadline_drops_socket_so_next_call_reconnects(
        self,
    ):
        # [FUNCTION DOC]
        # An outer wait_for cancellation also drops the socket and the next call reconnects.
        """An outer `asyncio.wait_for` around the whole call cancels
        `_receive()` with `asyncio.CancelledError`, not the client's own
        `asyncio.TimeoutError` -- a caller-imposed deadline (e.g. a step
        call wrapped in its own timeout) hits this path even when
        `message_timeout_s` itself never fires.

        Regression for the gap in the #1143 fix: only `asyncio.TimeoutError`
        was caught, so this path left the socket open and desynced exactly
        like the original bug.
        """

        class NeverResponds:
            # [CLASS DOC]
            # HELPER: Socket whose recv never answers in time.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                pass

            async def recv(self):
                # [FUNCTION DOC]
                # Fake socket recv: either never answers in time (sleeps 10 s) or raises ConnectionError,
                # depending on the scenario.
                await asyncio.sleep(10)

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=10
        )
        original_ws = NeverResponds()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client._send_and_receive({"type": "state"}), timeout=0.05
            )

        assert client._ws is None
        # Background close task, same as above -- give the loop one tick.
        await asyncio.sleep(0)
        assert original_ws.state == State.CLOSED

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.return_value = '{"type": "state", "data": {}}'

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            await client._send_and_receive({"type": "state"})

        mock_connect.assert_called_once()
        assert client._ws is replacement_ws

    @pytest.mark.asyncio
    async def test_slow_close_handshake_does_not_stall_callers_deadline(self):
        # [FUNCTION DOC]
        # A 2 s close handshake does not delay the caller's 50 ms deadline; the close still
        # completes in the background.
        """A slow close handshake on the dropped socket must not eat into
        the caller's own deadline.

        Regression for the Bugbot finding on #1149: `_receive()` used to
        `await ws.close()` inline before re-raising, so an outer
        `asyncio.wait_for(..., timeout=50ms)` didn't actually return in
        50ms if the server was slow to ack the close (websockets' default
        `close_timeout` is 10s) -- the caller was blocked on a handshake it
        never asked to wait on. The close is now fire-and-forget.
        """

        close_started = asyncio.Event()

        class NeverRespondsSlowClose:
            # [CLASS DOC]
            # HELPER: Socket that never responds and has a slow close.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                pass

            async def recv(self):
                # [FUNCTION DOC]
                # Fake socket recv: either never answers in time (sleeps 10 s) or raises ConnectionError,
                # depending on the scenario.
                await asyncio.sleep(10)

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                close_started.set()
                await asyncio.sleep(2)  # simulate a stalled close handshake
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=10
        )
        original_ws = NeverRespondsSlowClose()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client._send_and_receive({"type": "state"}), timeout=0.05
            )
        elapsed = loop.time() - start

        # The 2s close handshake must not have been awaited inline: the
        # caller's own 50ms deadline should be honored, not the socket's.
        assert elapsed < 1.0, (
            f"caller's 50ms deadline took {elapsed:.2f}s -- the close "
            "handshake was awaited inline instead of backgrounded"
        )
        assert client._ws is None

        # The close still actually happens, just not on the caller's time.
        await close_started.wait()
        await asyncio.sleep(2.1)
        assert original_ws.state == State.CLOSED

    @pytest.mark.asyncio
    async def test_close_waits_for_pending_background_close(self):
        # [FUNCTION DOC]
        # _close_async waits for pending background socket closes.
        """`close()` must not return until a backgrounded socket close (the
        fire-and-forget cleanup in `_receive()`) has actually finished.

        Regression for a second Bugbot finding on #1149:
        `_pending_close_tasks` was never awaited anywhere, so
        `SyncEnvClient.close()` could call `_stop_loop()` and tear down the
        event loop while a background close was still mid-handshake,
        abandoning it -- the server-side session could stay allocated.
        """
        close_completed = False

        class SlowBackgroundClose:
            # [CLASS DOC]
            # HELPER: Socket with a short delayed close that records completion.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                pass

            async def recv(self):
                # [FUNCTION DOC]
                # Fake socket recv: either never answers in time (sleeps 10 s) or raises ConnectionError,
                # depending on the scenario.
                await asyncio.sleep(10)

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                nonlocal close_completed
                await asyncio.sleep(0.05)
                close_completed = True
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=0.01
        )
        dropped_ws = SlowBackgroundClose()
        client._ws = dropped_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await client._send_and_receive({"type": "state"})

        assert client._pending_close_tasks  # background close was scheduled
        assert not close_completed  # ...but hasn't run yet

        await client._close_async()

        assert close_completed, (
            "close() returned before the backgrounded socket close finished"
        )

    @pytest.mark.asyncio
    async def test_reconnect_waits_for_dropped_socket_to_release_capacity(self):
        # [FUNCTION DOC]
        # Reconnect waits for the dropped socket's close to finish before connecting.
        """A replacement connection must wait for the old session to close.

        A timed-out socket is detached immediately and closed in the background.
        Reconnecting before that handshake finishes races the server's session
        accounting; with the default capacity of one, the retry is rejected.
        """
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        class SlowClose:
            # [CLASS DOC]
            # HELPER: Socket whose close blocks until released.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                pass

            async def recv(self):
                # [FUNCTION DOC]
                # Fake socket recv: either never answers in time (sleeps 10 s) or raises ConnectionError,
                # depending on the scenario.
                await asyncio.sleep(10)

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                close_started.set()
                await release_close.wait()
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=0.01
        )
        dropped_ws = SlowClose()
        client._ws = dropped_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await client._send_and_receive({"type": "state"})
        await close_started.wait()

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.return_value = '{"type": "state", "data": {}}'

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            reconnect = asyncio.create_task(client._connect_async())
            await asyncio.sleep(0)
            assert not reconnect.done()
            mock_connect.assert_not_called()

            release_close.set()
            await asyncio.wait_for(reconnect, timeout=1)

        mock_connect.assert_called_once()
        assert dropped_ws.state == State.CLOSED
        assert client._ws is replacement_ws
        await client._close_async()

    @pytest.mark.asyncio
    async def test_cancelled_close_tracks_current_socket_until_handshake_finishes(
        self,
    ):
        # [FUNCTION DOC]
        # Cancelling close keeps the socket close as a pending task, and a later connect waits for it.
        """Cancelling the current close must not abandon its server session."""
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        class SlowCurrentSocket:
            # [CLASS DOC]
            # HELPER: Current socket whose close blocks until released.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                pass

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                close_started.set()
                await release_close.wait()
                self.state = State.CLOSED

        client = GenericEnvClient(base_url="http://localhost:8000")
        current_ws = SlowCurrentSocket()
        client._ws = current_ws
        client._ws_loop = asyncio.get_running_loop()

        close_call = asyncio.create_task(client._close_async())
        await close_started.wait()
        assert client._pending_close_tasks

        close_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_call

        assert client._ws is None
        assert current_ws.state == State.OPEN
        assert any(not task.cancelled() for task in client._pending_close_tasks)

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN

        async def fake_ws_connect(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stand-in for ws_connect used by this test: records what the test inspects
            # (sockets opened, proxy/ping kwargs or environment state) and returns a mock socket.
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            reconnect = asyncio.create_task(client._connect_async())
            await asyncio.sleep(0)
            assert not reconnect.done()
            mock_connect.assert_not_called()

            release_close.set()
            await asyncio.wait_for(reconnect, timeout=1)

        assert current_ws.state == State.CLOSED
        mock_connect.assert_called_once()
        assert client._ws is replacement_ws
        await client._close_async()

    @pytest.mark.asyncio
    async def test_cancelled_close_still_closes_current_and_pending_sockets(self):
        # [FUNCTION DOC]
        # Cancelling close while draining a pending close still closes the current socket, stops
        # the provider, clears URLs and leaves the pending close running to completion.
        """Cancellation while draining an old socket must not leak either one."""

        class FakeRuntimeProvider:
            # [CLASS DOC]
            # HELPER: Provider recording whether stop was called.
            def __init__(self):
                # [FUNCTION DOC]
                # Sets stopped False.
                self.stopped = False

            def stop(self):
                # [FUNCTION DOC]
                # Sets stopped True.
                self.stopped = True

        close_started = asyncio.Event()
        release_close = asyncio.Event()

        class PendingSocket:
            # [CLASS DOC]
            # HELPER: Previously dropped socket whose close blocks until released.
            state = State.OPEN

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                close_started.set()
                await release_close.wait()
                self.state = State.CLOSED

        class CurrentSocket:
            # [CLASS DOC]
            # HELPER: Current socket closing immediately.
            state = State.OPEN

            async def send(self, _message):
                # [FUNCTION DOC]
                # Fake socket send; no-op unless the test needs it to change socket state (then it marks
                # the socket CLOSED).
                pass

            async def close(self):
                # [FUNCTION DOC]
                # Fake socket close: marks the socket CLOSED, optionally after signalling and waiting on
                # the test's events or a delay to simulate a slow close handshake.
                self.state = State.CLOSED

        provider = FakeRuntimeProvider()
        client = GenericEnvClient(provider=provider)
        client._base_url = "http://localhost:8000"
        client._ws_url = "ws://localhost:8000/ws"

        dropped_ws = PendingSocket()
        pending = asyncio.create_task(dropped_ws.close())
        client._pending_close_tasks.add(pending)
        pending.add_done_callback(client._pending_close_tasks.discard)
        await close_started.wait()

        current_ws = CurrentSocket()
        client._ws = current_ws
        client._ws_loop = asyncio.get_running_loop()

        close_task = asyncio.create_task(client._close_async())
        await asyncio.sleep(0)  # let close enter the shielded pending-close drain
        assert not close_task.done()

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert provider.stopped
        assert client._base_url is None
        assert client._ws_url is None
        assert client._ws is None
        assert current_ws.state == State.CLOSED
        assert not pending.cancelled()

        release_close.set()
        await pending
        assert dropped_ws.state == State.CLOSED


# ============================================================================
# Integration Tests (require running server)
# ============================================================================


@pytest.mark.integration
class TestGenericEnvClientIntegration:
    # [CLASS DOC]
    # Integration tests against an already running echo server on localhost:8000.
    """
    Integration tests that require a running server.

    These tests require a server to be running on localhost.

    Start a server first:
        cd envs/echo_env/server && uvicorn app:app --host 0.0.0.0 --port 8000

    Run these tests with:
        pytest -m integration tests/test_core/test_generic_client.py -v
    """

    @pytest.fixture
    def local_echo_server(self):
        # [FUNCTION DOC]
        # Fixture: returns the base URL if /health answers 200, otherwise skips.
        """Check if local echo server is running."""
        import requests

        base_url = "http://localhost:8000"
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Local echo server not healthy")
            return base_url
        except requests.RequestException:
            pytest.skip(
                "Local echo server not running. "
                "Start it with: cd envs/echo_env/server && uvicorn app:app"
            )

    def test_generic_client_with_local_server(self, local_echo_server):
        # [FUNCTION DOC]
        # Sync reset and a step echo the message.
        """Test GenericEnvClient with a real local server using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            # Reset
            result = client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step with dict action
            action = {"message": "Hello from GenericEnvClient!"}
            step_result = client.step(action)

            assert step_result is not None
            assert isinstance(step_result.observation, dict)
            assert "Hello from GenericEnvClient!" in step_result.observation.get(
                "echoed_message", ""
            )

    def test_generic_client_multiple_steps(self, local_echo_server):
        # [FUNCTION DOC]
        # Each of three steps echoes its message.
        """Test multiple steps with GenericEnvClient using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            client.reset()

            messages = ["First", "Second", "Third"]
            for msg in messages:
                result = client.step({"message": msg})
                assert msg in result.observation.get("echoed_message", "")

    def test_generic_client_state(self, local_echo_server):
        # [FUNCTION DOC]
        # state() returns a non-empty dict after steps.
        """Test getting state with GenericEnvClient using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            client.reset()

            # Execute some steps
            client.step({"message": "step 1"})
            client.step({"message": "step 2"})

            # Get state
            state = client.state()

            assert isinstance(state, dict)
            # State should have step_count
            assert "step_count" in state or len(state) > 0

    @pytest.mark.asyncio
    async def test_generic_client_async_with_local_server(self, local_echo_server):
        # [FUNCTION DOC]
        # Async reset and step echo the message.
        """Test GenericEnvClient with async API."""
        async with GenericEnvClient(base_url=local_echo_server) as client:
            # Reset
            result = await client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step with dict action
            action = {"message": "Hello from async GenericEnvClient!"}
            step_result = await client.step(action)

            assert step_result is not None
            assert isinstance(step_result.observation, dict)
            assert "Hello from async GenericEnvClient!" in step_result.observation.get(
                "echoed_message", ""
            )


@pytest.mark.integration
@pytest.mark.docker
class TestGenericEnvClientDocker:
    # [CLASS DOC]
    # Docker integration test with the echo-env:latest image.
    """
    Docker integration tests for GenericEnvClient.

    These tests require Docker to be running.

    Run these tests with:
        pytest -m "integration and docker" tests/test_core/test_generic_client.py -v
    """

    @pytest.fixture
    def check_docker_and_image(self):
        # [FUNCTION DOC]
        # Fixture: skips unless Docker is usable and echo-env:latest exists.
        """Check if Docker is available and echo-env image exists."""
        import shutil
        import subprocess

        if not shutil.which("docker"):
            pytest.skip("Docker is not installed")

        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Docker daemon is not running")
        except Exception:
            pytest.skip("Cannot access Docker")

        # Check for echo-env image
        result = subprocess.run(
            ["docker", "images", "-q", "echo-env:latest"],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            pytest.skip("Docker image 'echo-env:latest' not found")

    @pytest.mark.asyncio
    async def test_generic_client_from_docker_image(self, check_docker_and_image):
        # [FUNCTION DOC]
        # from_docker_image starts a real container; reset and step echo; always closes.
        # Side effect: prints a success line.
        """Test GenericEnvClient.from_docker_image() with real Docker."""
        client = await GenericEnvClient.from_docker_image("echo-env:latest")

        try:
            # Reset
            result = await client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step
            step_result = await client.step({"message": "Docker test!"})
            assert "Docker test!" in step_result.observation.get("echoed_message", "")

            print("GenericEnvClient.from_docker_image() works!")
        finally:
            await client.close()


# ============================================================================
# GenericAction Tests
# ============================================================================


class TestGenericAction:
    # [CLASS DOC]
    # Tests GenericAction, a dict subclass.
    """Test GenericAction class."""

    def test_create_from_kwargs(self):
        # [FUNCTION DOC]
        # Keyword arguments become dict items.
        """Test creating GenericAction from keyword arguments."""
        action = GenericAction(code="print('hello')", timeout=30)

        assert action["code"] == "print('hello')"
        assert action["timeout"] == 30

    def test_is_dict_subclass(self):
        # [FUNCTION DOC]
        # It is a dict.
        """Test that GenericAction is a dict subclass."""
        action = GenericAction(message="test")

        assert isinstance(action, dict)
        assert isinstance(action, GenericAction)

    def test_dict_methods_work(self):
        # [FUNCTION DOC]
        # get/keys/values behave like a dict.
        """Test that dict methods work on GenericAction."""
        action = GenericAction(a=1, b=2)

        assert action.get("a") == 1
        assert action.get("c", "default") == "default"
        assert list(action.keys()) == ["a", "b"]
        assert list(action.values()) == [1, 2]

    def test_empty_action(self):
        # [FUNCTION DOC]
        # No kwargs gives an empty dict.
        """Test creating empty GenericAction."""
        action = GenericAction()

        assert len(action) == 0
        assert dict(action) == {}

    def test_nested_values(self):
        # [FUNCTION DOC]
        # Nested values are kept.
        """Test GenericAction with nested values."""
        action = GenericAction(
            command="run",
            params={"file": "test.py", "args": ["--verbose"]},
        )

        assert action["command"] == "run"
        assert action["params"]["file"] == "test.py"
        assert action["params"]["args"] == ["--verbose"]

    def test_repr(self):
        # [FUNCTION DOC]
        # repr names the class and keys.
        """Test GenericAction repr."""
        action = GenericAction(code="x=1")

        repr_str = repr(action)
        assert "GenericAction" in repr_str
        assert "code=" in repr_str

    def test_can_be_used_with_generic_client(self):
        # [FUNCTION DOC]
        # It works as a payload for GenericEnvClient.
        """Test that GenericAction works with GenericEnvClient._step_payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        action = GenericAction(message="hello")

        payload = client._step_payload(action)

        assert payload == {"message": "hello"}


class TestGenericActionImports:
    # [CLASS DOC]
    # Tests GenericAction import locations.
    """Test GenericAction imports."""

    def test_import_from_core(self):
        # [FUNCTION DOC]
        # openenv.core exports it.
        """Test import from openenv.core."""
        from openenv.core import GenericAction as GA1

        assert GA1 is GenericAction

    def test_import_from_openenv(self):
        # [FUNCTION DOC]
        # openenv exports it.
        """Test import from openenv package."""
        from openenv import GenericAction as GA2

        assert GA2 is GenericAction

    def test_import_from_module(self):
        # [FUNCTION DOC]
        # The module exports it.
        """Test direct import from module."""
        from openenv.core.generic_client import GenericAction as GA3

        assert GA3 is GenericAction


# ============================================================================
# AutoAction skip_install Tests
# ============================================================================


class TestAutoActionSkipInstall:
    # [CLASS DOC]
    # Tests AutoAction.from_env with skip_install.
    """Test AutoAction.from_env() with skip_install parameter."""

    def test_skip_install_returns_generic_action(self):
        # [FUNCTION DOC]
        # Hub names return GenericAction.
        """Test skip_install=True returns GenericAction class."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("user/any-env", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_works_for_local_names(self):
        # [FUNCTION DOC]
        # Local names return GenericAction.
        """Test skip_install=True works for local environment names."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("echo", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_from_hub_alias(self):
        # [FUNCTION DOC]
        # from_hub with skip_install returns GenericAction.
        """Test skip_install works with from_hub alias."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_hub("user/my-env", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_action_can_be_instantiated(self):
        # [FUNCTION DOC]
        # The returned class builds dict actions.
        """Test that returned GenericAction can be instantiated."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("user/repo", skip_install=True)

        # Create an action
        action = ActionClass(code="print('hello')", timeout=30)

        assert action["code"] == "print('hello')"
        assert action["timeout"] == 30

    def test_skip_install_false_still_works(self):
        # [FUNCTION DOC]
        # Without skip_install, discovery returns the typed action class.
        """Test that skip_install=False (default) still works as before."""
        from openenv.auto._discovery import EnvironmentInfo, reset_discovery
        from openenv.auto.auto_action import AutoAction

        reset_discovery()

        mock_env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo environment",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
            spec_version=1,
        )

        mock_discovery = Mock()
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_discovery.discover.return_value = {"echo": mock_env_info}

        mock_action_class = Mock()
        mock_env_info.get_action_class = Mock(return_value=mock_action_class)

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            result = AutoAction.from_env("echo", skip_install=False)

            # Should return the typed action class, not GenericAction
            assert result is mock_action_class
            assert result is not GenericAction


# ============================================================================
# End-to-End: AutoEnv + AutoAction with skip_install
# ============================================================================


class TestAutoEnvAutoActionSkipInstallIntegration:
    # [CLASS DOC]
    # Tests AutoEnv and AutoAction together with skip_install.
    """Test AutoEnv and AutoAction work together with skip_install."""

    def test_both_skip_install_returns_generic_types(self):
        # [FUNCTION DOC]
        # Both return generic types that interoperate.
        """Test that both AutoEnv and AutoAction with skip_install work together."""
        from openenv.auto.auto_action import AutoAction
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            # Get client without installing package
            client = AutoEnv.from_env(
                "user/my-env",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            # Get action class without installing package
            ActionClass = AutoAction.from_env("user/my-env", skip_install=True)

            # Both should be generic types
            assert isinstance(client, GenericEnvClient)
            assert ActionClass is GenericAction

            # They should work together
            action = ActionClass(code="test")
            payload = client._step_payload(action)
            assert payload == {"code": "test"}

    def test_mixed_skip_install_raises_warning_scenario(self):
        # [FUNCTION DOC]
        # AutoAction without skip_install attempts a Hub install (patched to fail) and raises ValueError.
        # KNOWN ISSUE: the name mentions a warning, but the test checks that a ValueError is raised.
        """
        Test scenario where user forgets skip_install on AutoAction.

        This documents the expected behavior - if user uses skip_install
        on AutoEnv but not on AutoAction, AutoAction will try to install.
        """
        import os

        from openenv.auto._discovery import reset_discovery
        from openenv.auto.auto_action import AutoAction
        from openenv.auto.auto_env import AutoEnv

        reset_discovery()

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            # Get client without installing package
            client = AutoEnv.from_env(
                "user/my-env",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

            # Now if user forgets skip_install on AutoAction...
            # It will try to install from Hub and fail (no confirmation in tests)
            # Set env var to bypass confirmation, but mock installation to fail
            with (
                patch.dict(os.environ, {"OPENENV_TRUST_REMOTE_CODE": "1"}),
                patch(
                    "openenv.auto.auto_action.AutoEnv._ensure_package_from_hub"
                ) as mock_ensure,
            ):
                # Make _ensure_package_from_hub raise an error
                mock_ensure.side_effect = ValueError("Installation failed")

                # This should raise ValueError from installation attempt
                with pytest.raises(ValueError) as exc_info:
                    AutoAction.from_env("user/my-env")  # forgot skip_install=True!

                # Installation was attempted
                assert "Installation failed" in str(exc_info.value)
