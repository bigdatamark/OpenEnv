# [MODULE DOC]
# Tests for openenv.harbor.shared_template.enable_shared_templates, which patches Harbor's
# E2BEnvironment so every task uses one shared environment_name (from an argument or
# HARBOR_SHARED_ENV_NAME) and wraps the template-existence lookup with bounded transport retries.
"""One template per distinct environment, not one per task.

Harbor builds a template per task because it names the alias from `task.short_name`. For a suite whose
tasks share one image that is thousands of identical builds, and worse: `alias_exists()` goes true when
a build STARTS, so a GRPO group racing a task's first visit 404s on `tag 'default' does not exist`.

These tests assert on the `environment_name` Harbor's own constructor RECEIVES, because that is the
value the alias is built from — asserting on our wrapper would only prove the wrapper ran.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from openenv.harbor import shared_template


@pytest.fixture(autouse=True)
def _reset():
    # [FUNCTION DOC]
    # HELPER: Autouse fixture clearing the module's private "applied" latch before and after each test.
    """`enable_shared_templates` is idempotent by design, so tests must clear the latch."""
    shared_template._applied = False
    yield
    shared_template._applied = False


@pytest.fixture
def recorder(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Fixture that imports harbor's e2b module (skip if missing), replaces E2BEnvironment.__init__
    # with a fake recording the environment_name (keyword or second positional arg), and re-sets
    # _does_template_exist via monkeypatch so it is restored after the test. Returns (module, seen list).
    """Replace Harbor's E2B constructor with a recorder, then let the patch wrap the recorder."""
    e2b = pytest.importorskip("harbor.environments.e2b")
    seen: list[str] = []

    def fake_init(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the environment name passed to the constructor.
        seen.append(kwargs.get("environment_name") if kwargs else args[1])

    monkeypatch.setattr(e2b.E2BEnvironment, "__init__", fake_init, raising=True)
    # Restore the production method after enable_shared_templates wraps it.
    monkeypatch.setattr(
        e2b.E2BEnvironment,
        "_does_template_exist",
        e2b.E2BEnvironment._does_template_exist,
    )
    return e2b, seen


def test_the_shared_name_reaches_harbors_constructor(recorder, monkeypatch):
    # [FUNCTION DOC]
    # An explicit shared name replaces the per-task name passed to Harbor's constructor.
    e2b, seen = recorder
    monkeypatch.delenv("HARBOR_SHARED_ENV_NAME", raising=False)
    assert shared_template.enable_shared_templates("oe-dataagent") is True
    e2b.E2BEnvironment(object(), environment_name="0000_555_555434_qa_3")
    assert seen == ["oe-dataagent"], "the per-task name still reached Harbor"


def test_it_reads_the_env_var(recorder, monkeypatch):
    # [FUNCTION DOC]
    # The shared name is taken from HARBOR_SHARED_ENV_NAME when no argument is given.
    e2b, seen = recorder
    monkeypatch.setenv("HARBOR_SHARED_ENV_NAME", "from-env")
    assert shared_template.enable_shared_templates() is True
    e2b.E2BEnvironment(object(), environment_name="0000_650_650548_qa_2")
    assert seen == ["from-env"]


def test_unset_is_a_no_op(recorder, monkeypatch):
    # [FUNCTION DOC]
    # With no name, enable returns False and per-task names pass through.
    """Default behaviour must be untouched: no variable, no renaming."""
    e2b, seen = recorder
    monkeypatch.delenv("HARBOR_SHARED_ENV_NAME", raising=False)
    assert shared_template.enable_shared_templates() is False
    e2b.E2BEnvironment(object(), environment_name="0000_555_555434_qa_3")
    assert seen == ["0000_555_555434_qa_3"]


def test_a_positional_name_is_still_rewritten(recorder, monkeypatch):
    # [FUNCTION DOC]
    # A positionally passed environment name is also rewritten.
    """Harbor passes it by keyword today; a future positional call must not silently opt out."""
    e2b, seen = recorder
    monkeypatch.delenv("HARBOR_SHARED_ENV_NAME", raising=False)
    shared_template.enable_shared_templates("oe-dataagent")
    e2b.E2BEnvironment(object(), "0000_555_555434_qa_3", "session")
    assert seen == ["oe-dataagent"]


@pytest.mark.asyncio
async def test_read_only_lookup_recovers_closed_http2_connection(recorder, monkeypatch):
    # [FUNCTION DOC]
    # A LocalProtocolError on the first lookup is retried once after a 1s (mocked) sleep and returns
    # True; enabling twice does not double-wrap; _create_sandbox is untouched.
    e2b, _ = recorder
    lookup = AsyncMock(
        side_effect=[
            httpx.LocalProtocolError(
                "Invalid input ConnectionInputs.RECV_PING in state ConnectionState.CLOSED"
            ),
            True,
        ]
    )
    sleep = AsyncMock()
    create = e2b.E2BEnvironment._create_sandbox
    monkeypatch.setattr(e2b.E2BEnvironment, "_does_template_exist", lookup)
    monkeypatch.setattr(shared_template.asyncio, "sleep", sleep)
    shared_template.enable_shared_templates("test-shared")
    wrapped = e2b.E2BEnvironment._does_template_exist
    shared_template.enable_shared_templates("test-shared")
    assert e2b.E2BEnvironment._does_template_exist is wrapped
    env = e2b.E2BEnvironment(object(), environment_name="task")
    assert await env._does_template_exist() is True
    assert lookup.await_count == 2
    sleep.assert_awaited_once_with(1)
    assert e2b.E2BEnvironment._create_sandbox is create


@pytest.mark.asyncio
async def test_lookup_transport_retries_are_bounded(recorder, monkeypatch):
    # [FUNCTION DOC]
    # Persistent ReadTimeout is retried 3 times total with sleeps 1 and 2 and then re-raised as the same
    # exception object.
    e2b, _ = recorder
    failure = httpx.ReadTimeout("lookup timed out")
    lookup = AsyncMock(side_effect=failure)
    sleep = AsyncMock()
    monkeypatch.setattr(e2b.E2BEnvironment, "_does_template_exist", lookup)
    monkeypatch.setattr(shared_template.asyncio, "sleep", sleep)
    shared_template.enable_shared_templates("test-shared")
    with pytest.raises(httpx.ReadTimeout) as caught:
        await e2b.E2BEnvironment(
            object(), environment_name="task"
        )._does_template_exist()
    assert caught.value is failure
    assert lookup.await_count == 3
    assert [call.args for call in sleep.await_args_list] == [(1,), (2,)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        False,
        httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("GET", "https://example.test"),
            response=httpx.Response(401),
        ),
        asyncio.CancelledError(),
    ],
)
async def test_lookup_does_not_retry_missing_alias_permanent_error_or_cancellation(
    recorder, monkeypatch, outcome
):
    # [FUNCTION DOC]
    # Parametrized: a False result, an HTTP 401 status error, or cancellation is not retried and no
    # sleep occurs.
    e2b, _ = recorder
    lookup = (
        AsyncMock(side_effect=outcome)
        if isinstance(outcome, BaseException)
        else AsyncMock(return_value=outcome)
    )
    sleep = AsyncMock()
    monkeypatch.setattr(e2b.E2BEnvironment, "_does_template_exist", lookup)
    monkeypatch.setattr(shared_template.asyncio, "sleep", sleep)
    shared_template.enable_shared_templates("test-shared")
    env = e2b.E2BEnvironment(object(), environment_name="task")
    if isinstance(outcome, BaseException):
        with pytest.raises(type(outcome)):
            await env._does_template_exist()
    else:
        assert await env._does_template_exist() is False
    assert lookup.await_count == 1
    sleep.assert_not_awaited()
