# [MODULE DOC]
# Test that a server at capacity (max 1 session) refuses a second session with a
# CAPACITY_REACHED error frame that the client surfaces even if the socket closes before
# the client sends.
# SPDX-License-Identifier: BSD-3-Clause

"""A server at capacity refuses a session with an error frame the client must surface."""

from unittest.mock import AsyncMock, patch

import pytest
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.generic_client import GenericEnvClient


class _Action(Action):
    # [CLASS DOC]
    # HELPER: Trivial action with a message string.
    message: str = ""


class _Environment(Environment):
    # [CLASS DOC]
    # HELPER: Concurrent-capable no-op environment returning empty observations and state.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, **kwargs) -> Observation:
        # [FUNCTION DOC]
        # Returns an empty Observation.
        return Observation()

    def step(self, action: _Action) -> Observation:
        # [FUNCTION DOC]
        # Returns an empty Observation.
        return Observation()

    @property
    def state(self) -> State:
        # [FUNCTION DOC]
        # Property returning an empty State.
        return State()


@pytest.fixture
def server_url(serve):
    # [FUNCTION DOC]
    # Fixture serving create_app(_Environment, max_concurrent_envs=1) via the serve fixture.
    # Returns: its URL.
    return serve(create_app(_Environment, _Action, Observation, max_concurrent_envs=1))


async def test_capacity_error_survives_close_before_send(server_url):
    # [FUNCTION DOC]
    # While one client holds the only session, a second client connects, waits for the server
    # to close it, then (with the reconnect check patched out) reset raises RuntimeError with
    # CAPACITY_REACHED and "1/1 sessions active".
    async with GenericEnvClient(base_url=server_url) as holder:
        await holder.reset()

        refused = GenericEnvClient(base_url=server_url)
        await refused.connect()
        try:
            # The server sends CAPACITY_REACHED and closes. Make the close land
            # between the client's reconnect check and its send, as it does
            # under load: the send fails but the error frame is still queued.
            await refused._ws.wait_closed()
            with patch.object(refused, "_ensure_connected", AsyncMock()):
                with pytest.raises(RuntimeError, match="CAPACITY_REACHED") as exc:
                    await refused.reset()
            assert "1/1 sessions active" in str(exc.value)
        finally:
            await refused.close()
