# [MODULE DOC]
# Trusted server-side ThinkingBox environment. One instance runs one benchmark episode at a time: resolves
# and validates the pinned data bundle, hydrates the task by UID, opens a Session Proxy session for tool
# execution, records the native conversation driven by OpenEnv actions (tool calls, parallel batches,
# assistant messages, terminal turns), runs the simulated user between assistant messages, and on finish
# evaluates with the native TestScript/Judge to a binary reward. Private context (user context, tool
# side effects, settings) never leaves the server; only public observations and provenance metadata do.
# Infrastructure failures are "latched" (first stage recorded) and produce a system_error terminal result
# with reward 0. Defines a set of reserved control tool names that scenarios/agents may not use.
"""Orchestrate one isolated ThinkingBox benchmark episode through OpenEnv.

The environment keeps hydration, simulated-user context, tool side effects,
assertions, judging, and teardown within the trusted server boundary.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any
from uuid import uuid4

from openenv.core import Environment, ListToolsAction, Tool
from thinkingbox.common.agent_session import (
    make_empty_tool_response,
    MESSAGE_CONFIG,
    try_pop_direct_response,
)
from thinkingbox.common.agent_session_base import AgentSessionBase
from thinkingbox.common.agent_user_loop import replay_history_tool_calls
from thinkingbox.common.chat_types import (
    Conversation,
    Message,
    ParallelToolCall,
    TestContext,
    TestResult,
    Text,
    ToolCall,
    ToolCallResponse,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import (
    format_string_or_none,
    HydratedTestCase,
    merge_bot_instructions,
    merge_init_config,
    SessionProxyConfig,
    update_tools_with_client_config,
)
from thinkingbox.common.fixtures import build_fixtures
from thinkingbox.common.hydrator import get_dataset_case_by_name
from thinkingbox.common.judge import Judge
from thinkingbox.common.llm_session_factory import create_llm_session
from thinkingbox.common.mcp_proxy_client import MCPProxyContext
from thinkingbox.common.testrunner import TestScript
from thinkingbox.common.user_simulated_answer import UserSimulator

from .. import benchmark_data
from ..benchmark_data import DataBundle
from ..models import (
    _FinishAction,
    CallToolAction,
    DATA_COMMIT,
    DATA_RELEASE_NAME,
    SubmitMessageAction,
    ThinkingBoxAction,
    ThinkingBoxExecutionProvenance,
    ThinkingBoxObservation,
    ThinkingBoxState,
    ToolCallResult,
)
from ..runtime import load_thinkingbox_runtime_provenance
from . import config as server_config
from .config import load_runtime_settings, make_proxy_client, RuntimeSettings


logger = logging.getLogger(__name__)


def _redacted_exc_info(
    exc: Exception,
) -> tuple[type[RuntimeError], RuntimeError, Any]:
    # [FUNCTION DOC]
    # HELPER: returns an exc_info triple whose exception is a RuntimeError carrying only the original type
    # name (message redacted) and the original traceback, for safe logging.
    redacted = RuntimeError(f"{type(exc).__name__}: details redacted")
    return type(redacted), redacted, exc.__traceback__


_RESERVED_TOOL_NAMES = frozenset(
    {
        "verify",
        "geteffects",
        "get_effects",
        "__get_effects__",
        "session_create",
        "session_destroy",
        "session_info",
        "list_tasks",
        "__list_tasks__",
        "task_discovery",
        "reset",
        "step",
        "state",
        "close",
        "finish",
        "done",
    }
)


class RecordingAgentSession(AgentSessionBase):
    # [CLASS DOC]
    # Native AgentSessionBase subclass that only stores conversation state; assistant turns come from OpenEnv
    # actions instead of an internal model.
    """Store native ThinkingBox conversation state driven by OpenEnv actions."""

    def add_messages(self, messages: list[Message], add_to_llm: bool = True) -> None:
        # [FUNCTION DOC]
        # Appends messages to the conversation (the add-to-LLM flag is ignored).
        """Append native messages without invoking an internal model session."""
        del add_to_llm
        self.conversation.messages.extend(messages)

    async def decode_turn_iter(
        self, user_message: Text | None
    ) -> AsyncIterator[Message]:
        # [FUNCTION DOC]
        # Always raises NotImplementedError: native decoding is not used.
        """Reject native decoding because OpenEnv actions drive assistant turns."""
        del user_message
        raise NotImplementedError("OpenEnv actions drive decoding")
        yield  # pragma: no cover

    def can_end_conversation(self) -> bool:
        # [FUNCTION DOC]
        # Always returns True.
        """Return whether trusted OpenEnv orchestration may finalize the session."""
        return True


def _run_sync(coro: Any) -> Any:
    # [FUNCTION DOC]
    # HELPER: runs a coroutine with asyncio.run when no loop is running in this thread; raises RuntimeError
    # (telling callers to use the async methods) if called from inside a running loop.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Use reset_async(), step_async(), or aclose() in async code")


class ThinkingBoxEnvironment(
    Environment[ThinkingBoxAction, ThinkingBoxObservation, ThinkingBoxState]
):
    # [CLASS DOC]
    # OpenEnv Environment for one isolated ThinkingBox episode. Holds injectable dependencies (bundle
    # resolver, manifest loader, case hydrator, settings loader, proxy factory, user model/simulator
    # factories, evaluator), runtime provenance, the public ThinkingBoxState and private per-episode fields
    # (proxy client/context, recording session, hydrated case, settings, tool definitions and allow-lists,
    # user context/model, turn counters and limits, finish reason, latched infrastructure stage, the
    # terminal observation and the in-flight finalize task). Supports concurrent sessions.
    """Run one isolated ThinkingBox benchmark episode.

    Args:
        proxy_url (`str`, *optional*):
            Session Proxy fallback URL.
        dataset (`str`, *optional*):
            Default executable data root or `dataset/` path.
        agent (`str`, *optional*):
            Default native agent definition.
        config_path (`str`, *optional*):
            Default native ThinkingBox configuration path.
        bundle_resolver (`collections.abc.Callable`, *optional*):
            Data resolver override for tests or trusted integrations.
        manifest_loader (`collections.abc.Callable`, *optional*):
            Ordered manifest loader override.
        case_loader (`collections.abc.Callable`, *optional*):
            Native task hydrator override.
        settings_loader (`collections.abc.Callable`, *optional*):
            Private runtime-settings loader override.
        proxy_factory (`collections.abc.Callable`, *optional*):
            Session Proxy client factory override.
        user_model_factory (`collections.abc.Callable`, *optional*):
            Native simulated-user model factory override.
        user_simulator_factory (`collections.abc.Callable`, *optional*):
            Native simulated-user wrapper override.
        evaluator (`collections.abc.Callable`, *optional*):
            Native task evaluator override.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        proxy_url: str | None = None,
        dataset: str | None = None,
        agent: str | None = None,
        config_path: str | None = None,
        bundle_resolver: Callable[[str | None], DataBundle] | None = None,
        manifest_loader: Callable[[DataBundle], list[str]] | None = None,
        case_loader: Callable[[str, str, str], HydratedTestCase] | None = None,
        settings_loader: Callable[[str | None, SessionProxyConfig], RuntimeSettings]
        | None = None,
        proxy_factory: Callable[[RuntimeSettings], Any] | None = None,
        user_model_factory: Callable[[Any], Any] | None = None,
        user_simulator_factory: Callable[[Any, bool], Any] | None = None,
        evaluator: Callable[[HydratedTestCase, TestContext, RuntimeSettings], Any]
        | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores defaults (proxy URL, dataset path, agent and config path from args or server config/env) and
        # the dependency overrides (falling back to native implementations), loads installed-runtime provenance
        # (RuntimeError if ThinkingBox is not installed), and initialises empty episode state.
        super().__init__()
        self._proxy_url = proxy_url or server_config.SESSION_PROXY_URL
        self._default_dataset = (
            benchmark_data.DATASET_PATH if dataset is None else dataset
        ) or None
        self._default_agent = server_config.AGENT if agent is None else agent
        self._default_config = config_path or server_config.THINKINGBOX_CONFIG or None

        self._bundle_resolver = bundle_resolver or partial(
            benchmark_data.resolve_data_bundle,
            cache_root=benchmark_data.DATA_CACHE,
        )
        self._manifest_loader = manifest_loader or benchmark_data.load_test_uids
        self._case_loader = case_loader or get_dataset_case_by_name
        self._settings_loader = settings_loader or load_runtime_settings
        self._proxy_factory = proxy_factory or make_proxy_client
        self._user_model_factory = user_model_factory or create_llm_session
        self._user_simulator_factory = user_simulator_factory or (
            lambda model, can_end: UserSimulator(model, can_end_conversation=can_end)
        )
        self._evaluator = evaluator or self._evaluate_native
        self._runtime_provenance = load_thinkingbox_runtime_provenance()

        self._state = ThinkingBoxState()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._terminal_observation: ThinkingBoxObservation | None = None
        self._finalize_task: asyncio.Task[ThinkingBoxObservation] | None = None
        self._proxy_client: Any = None
        self._proxy: MCPProxyContext | None = None
        self._session: RecordingAgentSession | None = None
        self._tc: HydratedTestCase | None = None
        self._settings: RuntimeSettings | None = None
        self._tools: list[ToolDef] = []
        self._allowed_tool_names: set[str] = set()
        self._direct_response_tools: dict[str, str] = {}
        self._end_turn_tools: set[str] = set()
        self._user_context: str | None = None
        self._user_model: Any = None
        self._max_user_turns = 0
        self._user_turn = 0
        self._agent_turns = 0
        self._max_agent_turns = 0
        self._finish_reason: str | None = None
        self._infrastructure_stage: str | None = None
        self._infrastructure_detail: str | None = None
        self._query: str | None = None
        self._bot_instructions: str | None = None
        self._bundle: DataBundle | None = None
        self._selected_agent: str | None = None

    @property
    def state(self) -> ThinkingBoxState:
        # [FUNCTION DOC]
        # Property returning the public ThinkingBoxState.
        """Return non-sensitive lifecycle state for the active episode."""
        return self._state

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # Synchronous wrapper running reset_async; raises RuntimeError if called inside a running event loop.
        """Synchronously start the task selected by UID.

        Args:
            seed (`int`, *optional*):
                Accepted for API compatibility and intentionally unused because
                task selection is deterministic by UID.
            episode_id (`str`, *optional*):
                Infrastructure-provided episode identifier.
            kwargs (`dict`, *optional*):
                ThinkingBox reset arguments such as `test_uid`, `dataset`,
                `agent`, and `config`.

        Returns:
            [`ThinkingBoxObservation`]:
                Reset observation or privacy-reviewed error observation.
        """
        return _run_sync(self.reset_async(seed=seed, episode_id=episode_id, **kwargs))

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        test_uid: str | None = None,
        dataset: str | None = None,
        agent: str | None = None,
        config: str | None = None,
        **kwargs: Any,
    ) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # Records the running loop, waits for any in-flight finalization, tears down the previous proxy session
        # and clears private state, then resets episode fields and creates a new state (given or random episode
        # id, status idle). Returns an error observation (via _reset_failure) if the previous teardown failed or
        # test_uid is missing. Otherwise, staged with failure attribution: resolves the data bundle and checks the
        # UID is in its manifest (data), hydrates the case for the selected agent (hydration), loads runtime
        # settings and creates the proxy client (configuration), rejects reserved scenario tool names, creates the
        # proxy session with the merged world/init config (proxy_create), and starts the episode (episode_init).
        # Any exception yields the error observation. On success marks status active and returns a reset
        # observation with task text, system and bot instructions, public tools, visible messages and metadata.
        # Seed and unknown kwargs are ignored. Side effects: file I/O, network calls to the Session Proxy.
        """Asynchronously start the task selected by UID.

        Args:
            seed (`int`, *optional*):
                Accepted for API compatibility and intentionally unused.
            episode_id (`str`, *optional*):
                Infrastructure-provided episode identifier.
            test_uid (`str`, *optional*):
                UID from the resolved release manifest.
            dataset (`str`, *optional*):
                Per-reset executable data override.
            agent (`str`, *optional*):
                Per-reset native agent definition.
            config (`str`, *optional*):
                Per-reset native configuration path.
            kwargs (`dict`, *optional*):
                Additional OpenEnv reset arguments, ignored by this adapter.

        Returns:
            [`ThinkingBoxObservation`]:
                Model-visible reset or error observation.
        """
        del seed, kwargs
        self._event_loop = asyncio.get_running_loop()
        previous_teardown_error = await self._prepare_for_reset()
        self._reset_episode_fields()
        self._state = ThinkingBoxState(
            episode_id=episode_id or str(uuid4()),
            task_uid=test_uid,
            status="idle",
        )
        if previous_teardown_error is not None:
            return await self._reset_failure(
                "reset_teardown", previous_teardown_error, test_uid
            )
        if not test_uid:
            return await self._reset_failure(
                "input", ValueError("test_uid is required"), test_uid
            )

        reset_stage = "data"
        try:
            selected_dataset = dataset or self._default_dataset
            self._bundle = await asyncio.to_thread(
                self._bundle_resolver, selected_dataset
            )
            test_uids = await asyncio.to_thread(self._manifest_loader, self._bundle)
            if test_uid not in set(test_uids):
                raise ValueError("test_uid is not in the canonical manifest")

            reset_stage = "hydration"
            selected_agent = self._default_agent if agent is None else agent
            self._selected_agent = selected_agent
            self._tc = await asyncio.to_thread(
                self._case_loader,
                test_uid,
                str(self._bundle.dataset_dir),
                selected_agent,
            )
            default_proxy = SessionProxyConfig(
                endpoint_url=self._proxy_url,
                timeout=server_config.PROXY_TIMEOUT,
            )
            reset_stage = "configuration"
            self._settings = await asyncio.to_thread(
                self._settings_loader,
                config or self._default_config,
                default_proxy,
            )
            self._proxy_client = self._proxy_factory(self._settings)

            scenario_tool_names = [tool.name for tool in self._tc.scenario.tools]
            if any(name in _RESERVED_TOOL_NAMES for name in scenario_tool_names):
                raise ValueError("scenario contains a reserved control tool")
            world_config = merge_init_config(
                self._tc.scenario.world_state, self._tc.init
            )
            reset_stage = "proxy_create"
            init_result = await self._proxy_client.session_create(
                world_config, scenario_tool_names
            )
            self._proxy = MCPProxyContext(self._proxy_client, init_result)
            reset_stage = "episode_init"
            await self._start_episode(self._tc, scenario_tool_names)
        except Exception as exc:
            return await self._reset_failure(reset_stage, exc, test_uid)

        self._state.status = "active"
        self._state.task_uid = test_uid
        return ThinkingBoxObservation(
            kind="reset",
            task_uid=test_uid,
            task=self._query,
            system_instructions=self._tc.agent.system_instructions,
            bot_instructions=self._bot_instructions,
            tools=self._public_tools(),
            messages=self._visible_messages(),
            steps_taken=0,
            metadata=self._public_metadata(),
        )

    async def _start_episode(
        self, tc: HydratedTestCase, scenario_tool_names: list[str]
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: lists proxy tools and requires every scenario tool to be present (RuntimeError otherwise),
        # applies client tool config, formats query/user context/test instructions with the proxy init result when
        # the case requests it, merges bot instructions, builds the tool list (agent built-ins plus MCP tools;
        # names must be unique and not reserved, ValueError otherwise), creates the RecordingAgentSession with a
        # system prefix, replays any history tool calls through the proxy (RuntimeError on replay warnings), and
        # appends the user query. Stores allow-lists, direct-response templates, end-turn tools, limits (user turn
        # limit +1 when the user can end the conversation) and creates the user model when the case has user
        # context (RuntimeError if no user model is configured).
        assert self._proxy is not None
        assert self._settings is not None

        listed_tools = await self._proxy.list_tools()
        by_name = {tool.name: tool for tool in listed_tools}
        missing = set(scenario_tool_names) - set(by_name)
        if missing:
            raise RuntimeError("Session Proxy omitted scenario-permitted tools")
        mcp_tools = [by_name[name] for name in scenario_tool_names]
        update_tools_with_client_config(mcp_tools, tc.scenario.tools)

        query = tc.query
        user_context = tc.user_context
        test_instructions = tc.bot_instructions
        if tc.format_query:
            init_result = self._proxy.init_result
            query = format_string_or_none(query, init_result)
            user_context = format_string_or_none(user_context, init_result)
            test_instructions = format_string_or_none(test_instructions, init_result)
        bot_instructions = merge_bot_instructions(
            scenario_instructions=tc.scenario.bot_instructions,
            testcase_instructions=test_instructions,
        )

        tools = [
            tool.model_copy(deep=True) for tool in [*tc.agent.builtin_tools, *mcp_tools]
        ]
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("ThinkingBox tool names must be unique")
        if any(name in _RESERVED_TOOL_NAMES for name in names):
            raise ValueError("agent configuration contains a reserved control tool")

        self._session = RecordingAgentSession(
            config=tc.agent,
            mcp_proxy=self._proxy,
            mcp_tools=tools,
            bot_instructions=bot_instructions,
            scenario_metadata=tc.scenario.metadata,
        )
        prefix: list[Message] = [
            Text(role="system", content=tc.agent.system_instructions)
        ]
        if bot_instructions:
            prefix.append(Text(role="system", content=bot_instructions))
        self._session.conversation = Conversation(messages=prefix)
        self._session.conversation.metadata["tool_calls"] = []

        if tc.history:
            history = [message.model_copy(deep=True) for message in tc.history]
            self._session.add_messages(history)
            replay_warnings = await replay_history_tool_calls(
                history,
                self._proxy,
                builtin_tool_names={tool.name for tool in tc.agent.builtin_tools},
            )
            if replay_warnings:
                raise RuntimeError("ThinkingBox history replay failed")
        self._session.add_messages([Text(role="user", content=query or "")])

        self._tools = tools
        self._allowed_tool_names = set(names)
        self._direct_response_tools = {
            tool.name: tool.direct_response
            for tool in tools
            if tool.direct_response is not None
        }
        self._end_turn_tools = {tool.name for tool in tools if tool.is_end_turn}
        self._query = query
        self._bot_instructions = bot_instructions
        self._user_context = user_context
        self._max_user_turns = tc.max_user_sim_turns + (
            1 if self._settings.user_can_end_conversation else 0
        )
        self._max_agent_turns = tc.max_agent_sim_turns
        if user_context:
            if self._settings.user_model is None:
                raise RuntimeError(
                    "A native user_model is required for this private user context"
                )
            self._user_model = self._user_model_factory(self._settings.user_model)

    def step(
        self,
        action: ThinkingBoxAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # Synchronous wrapper running step_async; raises RuntimeError if called inside a running event loop.
        """Synchronously apply one public or trusted harness action.

        Args:
            action ([`ThinkingBoxAction`]):
                Validated ThinkingBox action.
            timeout_s (`float`, *optional*):
                Accepted for OpenEnv compatibility; client transport owns the
                effective timeout.
            kwargs (`dict`, *optional*):
                Additional OpenEnv step arguments.

        Returns:
            [`ThinkingBoxObservation`]:
                Tool, user, terminal, or error observation.
        """
        return _run_sync(self.step_async(action, timeout_s=timeout_s, **kwargs))

    async def step_async(
        self,
        action: ThinkingBoxAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # Returns the cached terminal observation if the episode ended; awaits the in-flight finalization if one
        # exists; returns an error observation if not active. ListToolsAction returns a "tools" observation without
        # counting a step. If the agent turn limit is reached, finishes with "agent_limit". Otherwise dispatches:
        # parallel CallToolAction to the batch handler, single CallToolAction to the single handler,
        # SubmitMessageAction to the terminal-turn handler (if it has terminal calls) or the message handler,
        # _FinishAction to finish with its reason; anything else gives an error observation.
        """Asynchronously apply one public or trusted harness action.

        Args:
            action ([`ThinkingBoxAction`]):
                Validated ThinkingBox action.
            timeout_s (`float`, *optional*):
                Accepted for OpenEnv compatibility.
            kwargs (`dict`, *optional*):
                Additional OpenEnv step arguments.

        Returns:
            [`ThinkingBoxObservation`]:
                Tool, user, terminal, or error observation.
        """
        del timeout_s, kwargs
        if self._terminal_observation is not None:
            return self._terminal_observation
        if self._finalize_task is not None:
            return await asyncio.shield(self._finalize_task)
        if self._state.status != "active":
            return self._error_observation("Environment is not active")

        if isinstance(action, ListToolsAction):
            return ThinkingBoxObservation(
                kind="tools",
                task_uid=self._state.task_uid,
                tools=self._public_tools(),
                messages=self._visible_messages(),
                steps_taken=self._state.step_count,
                metadata=self._public_metadata(),
            )
        if self._agent_turns >= self._max_agent_turns:
            return await self._finish("agent_limit")
        if isinstance(action, CallToolAction):
            if action.parallel_tool_calls:
                return await self._call_tool_batch(action)
            return await self._call_tool(action)
        if isinstance(action, SubmitMessageAction):
            if action.terminal_tool_calls:
                return await self._submit_terminal_turn(action)
            return await self._submit_message(action.content or "")
        if isinstance(action, _FinishAction):
            return await self._finish(action.reason)
        return self._error_observation("Unsupported action")

    async def _call_tool(self, action: CallToolAction) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: handles a single tool call. Unknown tools are recorded with a "does not exist" response and return
        # a tool observation with tool_not_found. End-turn tools are recorded with an empty response and finish the
        # episode ("end_turn_tool"). Otherwise calls the proxy (any exception latches "proxy_call" and finishes with
        # system_error), renders an optional direct response from the tool's template, records the exchange, adds a
        # direct assistant message (counting an extra agent turn) when present, and returns a tool observation.
        # Each call counts one agent turn and one step.
        assert self._session is not None
        assert self._proxy is not None
        assert action.tool_name is not None
        call = self._make_tool_call(action.tool_name, action.arguments, action.call_id)

        if action.tool_name not in self._allowed_tool_names:
            response = ToolResponse(
                name=call.name,
                content=f"Error: function '{call.name}' does not exist",
                id=call.id,
            )
            self._record_tool_exchange(call, response)
            self._agent_turns += 1
            self._increment_step()
            return ThinkingBoxObservation(
                kind="tool",
                task_uid=self._state.task_uid,
                tool_name=call.name,
                call_id=call.id,
                tool_result=response.content,
                tool_error="tool_not_found",
                messages=self._visible_messages(),
                steps_taken=self._state.step_count,
                metadata=self._public_metadata(),
            )

        self._agent_turns += 1
        self._increment_step()
        if action.tool_name in self._end_turn_tools:
            call.metadata["is_end_turn_tool"] = True
            self._session.add_messages(
                [
                    ParallelToolCall(
                        tool_calls=[call],
                        metadata={"is_end_turn_tool": True},
                    )
                ]
            )
            self._session.conversation.metadata["tool_calls"].append(
                ToolCallResponse(
                    tool_call=call,
                    tool_response=make_empty_tool_response(call),
                )
            )
            return await self._finish("end_turn_tool")

        self._session.add_messages([ParallelToolCall(tool_calls=[call])])
        try:
            result = await self._proxy.call_tool(call.name, **call.arguments)
        except Exception as exc:
            self._latch_infrastructure("proxy_call", exc)
            return await self._finish("system_error")

        content = result if isinstance(result, str) else str(result)
        template = self._direct_response_tools.get(call.name)
        if template is None:
            response = ToolResponse(name=call.name, content=content, id=call.id)
        else:
            response = try_pop_direct_response(
                call,
                content,
                template,
                error_message="Error in function execution",
            )
        direct_response = response.metadata.pop("direct_response", None)
        self._session.conversation.metadata["tool_calls"].append(
            ToolCallResponse(tool_call=call, tool_response=response)
        )
        self._session.add_messages([response])
        if direct_response is not None:
            self._session.add_messages(
                [
                    Text(
                        role="assistant",
                        content=direct_response,
                        metadata={"tag": "direct"},
                    )
                ]
            )
            self._agent_turns += 1
        return ThinkingBoxObservation(
            kind="tool",
            task_uid=self._state.task_uid,
            tool_name=call.name,
            call_id=call.id,
            tool_result=response.content,
            direct_response=direct_response,
            messages=self._visible_messages(),
            steps_taken=self._state.step_count,
            metadata=self._public_metadata(),
        )

    async def _call_tool_batch(self, action: CallToolAction) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: handles a parallel batch as one agent turn and one step. If any call is an end-turn tool, records
        # the batch and empty responses for end-turn calls and finishes ("end_turn_tool"). Otherwise processes calls
        # in order: parse errors give invalid_args responses, unknown tools give tool_not_found, valid tools are
        # executed through the proxy (exception latches and finishes with system_error) with optional direct
        # responses. Records all exchanges and direct assistant messages (each counting an agent turn). Returns a
        # tool_batch observation with ordered ToolCallResults.
        assert self._session is not None
        assert self._proxy is not None
        calls = [
            self._make_tool_call(
                call.name,
                call.arguments,
                call.call_id,
                parse_error=call.parse_error,
            )
            for call in action.parallel_tool_calls
        ]
        parallel = ParallelToolCall(tool_calls=calls)
        self._agent_turns += 1
        self._increment_step()

        end_calls = [call for call in calls if call.name in self._end_turn_tools]
        if end_calls:
            parallel.metadata["is_end_turn_tool"] = True
            for call in end_calls:
                call.metadata["is_end_turn_tool"] = True
                self._session.conversation.metadata["tool_calls"].append(
                    ToolCallResponse(
                        tool_call=call,
                        tool_response=make_empty_tool_response(call),
                    )
                )
            self._session.add_messages([parallel])
            return await self._finish("end_turn_tool")

        self._session.add_messages([parallel])
        responses: list[ToolResponse] = []
        results: list[ToolCallResult] = []
        direct_responses: list[str] = []
        for call in calls:
            tool_error: str | None = None
            parse_error = call.metadata.get("error")
            if parse_error:
                response = ToolResponse(
                    name=call.name,
                    content=parse_error,
                    id=call.id,
                )
                direct_response = None
                tool_error = "invalid_args"
            elif call.name not in self._allowed_tool_names:
                response = ToolResponse(
                    name=call.name,
                    content=f"Error: function '{call.name}' does not exist",
                    id=call.id,
                )
                direct_response = None
                tool_error = "tool_not_found"
            else:
                try:
                    result = await self._proxy.call_tool(call.name, **call.arguments)
                except Exception as exc:
                    self._latch_infrastructure("proxy_call", exc)
                    return await self._finish("system_error")
                else:
                    content = result if isinstance(result, str) else str(result)
                    template = self._direct_response_tools.get(call.name)
                    if template is None:
                        response = ToolResponse(
                            name=call.name,
                            content=content,
                            id=call.id,
                        )
                    else:
                        response = try_pop_direct_response(
                            call,
                            content,
                            template,
                            error_message="Error in function execution",
                        )
                    direct_response = response.metadata.pop("direct_response", None)

            responses.append(response)
            results.append(
                ToolCallResult(
                    name=call.name,
                    call_id=call.id,
                    content=response.content,
                    tool_error=tool_error,
                    direct_response=direct_response,
                )
            )
            if direct_response is not None:
                direct_responses.append(direct_response)

        for call, response in zip(calls, responses, strict=True):
            self._session.conversation.metadata["tool_calls"].append(
                ToolCallResponse(tool_call=call, tool_response=response)
            )
        self._session.add_messages(responses)
        if direct_responses:
            self._session.add_messages(
                [
                    Text(
                        role="assistant",
                        content=direct_response,
                        metadata={"tag": "direct"},
                    )
                    for direct_response in direct_responses
                ]
            )
            self._agent_turns += len(direct_responses)

        return ThinkingBoxObservation(
            kind="tool_batch",
            task_uid=self._state.task_uid,
            tool_results=results,
            messages=self._visible_messages(),
            steps_taken=self._state.step_count,
            metadata=self._public_metadata(),
        )

    async def _submit_terminal_turn(
        self, action: SubmitMessageAction
    ) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: records a provider turn that ended with terminal tool calls (never executed). Returns an error
        # observation if any call names an unavailable tool or none is an end-turn tool. Otherwise records empty
        # responses for end-turn calls, adds the batch and the optional assistant text (marked done if it matches
        # the done convention) in provider order, counts agent turns per added message and one step, and finishes
        # with "end_turn_tool".
        assert self._session is not None
        calls = [
            self._make_tool_call(
                call.name,
                call.arguments,
                call.call_id,
                parse_error=call.parse_error,
            )
            for call in action.terminal_tool_calls
        ]
        if any(call.name not in self._allowed_tool_names for call in calls):
            return self._error_observation(
                "Terminal turn contains a tool unavailable to this task"
            )
        end_calls = [call for call in calls if call.name in self._end_turn_tools]
        if not end_calls:
            return self._error_observation(
                "Terminal turn requires a configured end-turn tool"
            )

        parallel = ParallelToolCall(
            tool_calls=calls,
            metadata={"is_end_turn_tool": True},
        )
        for call in end_calls:
            call.metadata["is_end_turn_tool"] = True
            self._session.conversation.metadata["tool_calls"].append(
                ToolCallResponse(
                    tool_call=call,
                    tool_response=make_empty_tool_response(call),
                )
            )
        messages: list[Message] = [parallel]
        if action.content is not None:
            text = Text(role="assistant", content=action.content)
            if MESSAGE_CONFIG.is_done(action.content):
                text.metadata["is_done"] = True
            messages = (
                [parallel, text]
                if action.tool_calls_before_content
                else [text, parallel]
            )
        self._session.add_messages(messages)
        self._agent_turns += sum(
            isinstance(message, (ParallelToolCall, Text)) for message in messages
        )
        self._increment_step()
        return await self._finish("end_turn_tool")

    async def _submit_message(self, content: str) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: records an assistant text message (done-marked when applicable), counting one agent turn and one
        # step. Finishes when the session says the conversation should end, the agent limit is reached, there is no
        # user model ("no_user_llm") or the user turn limit is exceeded. Otherwise generates the simulated user's
        # reply with the private user context (exception latches "user_simulator" and finishes with system_error),
        # records it, finishes with "user_done" if the user may end and did, else returns a "user" observation with
        # the assistant response and the user's message.
        assert self._session is not None
        assert self._settings is not None
        message = Text(role="assistant", content=content)
        if MESSAGE_CONFIG.is_done(content):
            message.metadata["is_done"] = True
        self._session.add_messages([message])
        self._agent_turns += 1
        self._increment_step()

        finish_reason = self._session.should_end_conversation()
        if finish_reason is not None:
            return await self._finish(finish_reason)
        if self._agent_turns >= self._max_agent_turns:
            return await self._finish("agent_limit")

        self._user_turn += 1
        if self._user_model is None:
            return await self._finish("no_user_llm")
        if self._user_turn > self._max_user_turns:
            return await self._finish("user_limit")

        try:
            simulator = self._user_simulator_factory(
                self._user_model,
                self._settings.user_can_end_conversation,
            )
            user_message = await simulator.generate(
                chat_history=self._session.conversation.messages,
                user_context=self._user_context or "",
            )
            user_message = user_message.model_copy(deep=True)
            user_message.metadata["is_user_llm"] = True
            self._session.add_messages([user_message])
        except Exception as exc:
            self._latch_infrastructure("user_simulator", exc)
            return await self._finish("system_error")

        if self._settings.user_can_end_conversation and user_message.metadata.get(
            "is_done", False
        ):
            return await self._finish("user_done")
        return ThinkingBoxObservation(
            kind="user",
            task_uid=self._state.task_uid,
            response=content,
            user_message=user_message.content,
            messages=self._visible_messages(),
            steps_taken=self._state.step_count,
            metadata=self._public_metadata(),
        )

    async def _finish(self, reason: str) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: returns the cached terminal observation if present; otherwise, on first call records the reason,
        # sets status finalizing and starts a single finalize task. Awaits the (shielded) task and returns its
        # observation, so concurrent callers share one finalization.
        if self._terminal_observation is not None:
            return self._terminal_observation
        if self._finalize_task is None:
            self._finish_reason = reason
            self._state.status = "finalizing"
            self._finalize_task = asyncio.create_task(self._finalize_once())
        return await asyncio.shield(self._finalize_task)

    async def _finalize_once(self) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: produces the terminal observation exactly once. If no failure is latched, builds the native test
        # context with UID, case metadata and provenance, runs the evaluator (awaiting if needed; TypeError if it
        # does not return a TestResult), and derives pass/graded/system-error (a native system error is latched as
        # "evaluator"). Exceptions latch "effects_or_evaluator". Always tears down the proxy (failure latched as
        # "teardown"). Reward is 1.0 only if passed with no latched failure. Returns a terminal observation with
        # response, messages captured before teardown, finish reason, reward type, system error flag, test summary
        # (including the failure stage), error text, reward and done; caches it, updates state status (done or
        # error) and clears private state.
        response = ""
        passed = False
        graded = False
        test_system_error = False
        messages = self._visible_messages()
        teardown_error: Exception | None = None
        try:
            if self._infrastructure_stage is None:
                if self._session is None or self._tc is None or self._settings is None:
                    raise RuntimeError("ThinkingBox episode state is incomplete")
                context = await self._session.make_test_context()
                context.metadata.update(
                    {
                        "uid": self._tc.uid,
                        **self._tc.metadata,
                        "thinkingbox_revision": self._runtime_provenance.identity,
                        "thinkingbox_source_sha256": (
                            self._runtime_provenance.source_sha256
                        ),
                        "thinkingbox_source_type": (
                            self._runtime_provenance.source_type
                        ),
                        "data_release": (
                            self._bundle.release_name
                            if self._bundle is not None
                            else DATA_RELEASE_NAME
                        ),
                        "data_revision": (
                            self._bundle.revision
                            if self._bundle is not None
                            else DATA_COMMIT
                        ),
                    }
                )
                response = context.response
                evaluation = self._evaluator(self._tc, context, self._settings)
                if inspect.isawaitable(evaluation):
                    evaluation = await evaluation
                if not isinstance(evaluation, TestResult):
                    raise TypeError("ThinkingBox evaluator returned an invalid result")
                graded = True
                test_system_error = evaluation.is_system_error
                passed = bool(evaluation.result) and not test_system_error
                if test_system_error:
                    self._latch_infrastructure(
                        "evaluator",
                        RuntimeError("Native TestScript reported a system error"),
                    )
        except Exception as exc:
            self._latch_infrastructure("effects_or_evaluator", exc)
        finally:
            teardown_error = await self._teardown_proxy()
        if teardown_error is not None:
            self._latch_infrastructure("teardown", teardown_error)

        system_error = self._infrastructure_stage is not None
        if system_error:
            passed = False
        reward = 1.0 if passed else 0.0
        reward_type = "system_error" if system_error else ("pass" if passed else "fail")
        error = (
            f"ThinkingBox infrastructure failure during {self._infrastructure_stage}."
            if system_error
            else None
        )
        test_summary: dict[str, Any] = {
            "passed": passed,
            "graded": graded,
            "is_system_error": system_error or test_system_error,
        }
        if system_error:
            test_summary["infrastructure_stage"] = self._infrastructure_stage
        observation = ThinkingBoxObservation(
            kind="terminal",
            task_uid=self._state.task_uid,
            response=response or self._last_assistant_response(),
            messages=messages,
            finish_reason=self._finish_reason,
            reward_type=reward_type,
            system_error=system_error,
            test_summary=test_summary,
            error=error,
            reward=reward,
            done=True,
            steps_taken=self._state.step_count,
            metadata=self._public_metadata(),
        )
        self._terminal_observation = observation
        self._state.status = "error" if system_error else "done"
        self._state.system_error = system_error
        self._clear_sensitive()
        return observation

    async def _evaluate_native(
        self,
        tc: HydratedTestCase,
        context: TestContext,
        settings: RuntimeSettings,
    ) -> TestResult:
        # [FUNCTION DOC]
        # HELPER: default evaluator: builds a Judge (with an LLM session when a judge model is configured), the
        # case fixtures and a TestScript for the case's test code, and returns the awaited evaluation TestResult.
        judge_llm = (
            create_llm_session(settings.judge_model)
            if settings.judge_model is not None
            else None
        )
        judge = Judge(judge_llm, judge_type=settings.judge_type)
        fixtures = build_fixtures(tc.fixtures)
        script = TestScript(
            tc.test_code,
            judge,
            fixtures=fixtures,
            test_uid=tc.uid,
        )
        return await script.evaluate(context)

    async def _prepare_for_reset(self) -> Exception | None:
        # [FUNCTION DOC]
        # HELPER: waits for any in-flight finalization, tears down the proxy session and clears private state.
        # Returns the teardown exception, or None.
        if self._finalize_task is not None and not self._finalize_task.done():
            await asyncio.shield(self._finalize_task)
        error = await self._teardown_proxy()
        self._clear_sensitive()
        return error

    def _reset_episode_fields(self) -> None:
        # [FUNCTION DOC]
        # HELPER: clears closed flag, terminal observation, finalize task, latched failure, finish reason and turn
        # counters.
        self._closed = False
        self._terminal_observation = None
        self._finalize_task = None
        self._infrastructure_stage = None
        self._infrastructure_detail = None
        self._finish_reason = None
        self._user_turn = 0
        self._agent_turns = 0

    async def _reset_failure(
        self, stage: str, exc: Exception, test_uid: str | None
    ) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: latches the failure stage, tears down the proxy (latching teardown failures), builds and caches a
        # done "error" observation with reward 0.0, reward type system_error and the failing stage in the error
        # text, sets state status error with system_error True, clears private state and returns the observation.
        self._latch_infrastructure(stage, exc)
        teardown_error = await self._teardown_proxy()
        if teardown_error is not None:
            self._latch_infrastructure("teardown", teardown_error)
        observation = ThinkingBoxObservation(
            kind="error",
            task_uid=test_uid,
            finish_reason="reset_error",
            reward_type="system_error",
            system_error=True,
            error=f"ThinkingBox reset failed during {self._infrastructure_stage}.",
            reward=0.0,
            done=True,
            steps_taken=0,
            metadata=self._public_metadata(),
        )
        self._terminal_observation = observation
        self._state.status = "error"
        self._state.system_error = True
        self._clear_sensitive()
        return observation

    async def _teardown_proxy(self) -> Exception | None:
        # [FUNCTION DOC]
        # HELPER: detaches the proxy client and context and, if a client existed, destroys its session. Returns the
        # exception raised by session_destroy, or None.
        client, self._proxy_client = self._proxy_client, None
        self._proxy = None
        if client is None:
            return None
        try:
            await client.session_destroy()
        except Exception as exc:
            return exc
        return None

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Idempotent async close: waits for in-flight finalization, tears down the proxy (latching failures),
        # clears private state and sets status closed with system_error reflecting any latched failure.
        """Asynchronously destroy the proxy session and clear private state."""
        if self._closed:
            return
        self._closed = True
        if self._finalize_task is not None and not self._finalize_task.done():
            await asyncio.shield(self._finalize_task)
        teardown_error = await self._teardown_proxy()
        if teardown_error is not None:
            self._latch_infrastructure("teardown", teardown_error)
        self._clear_sensitive()
        self._state.status = "closed"
        self._state.system_error = self._infrastructure_stage is not None

    def close(self) -> Any:
        # [FUNCTION DOC]
        # Synchronous close that works from any context: inside the episode's own (or no recorded) running loop it
        # schedules aclose as a task and returns the task; from another thread while the episode loop runs, it
        # submits aclose to that loop and returns an awaitable wrapper (if the caller has a loop) or blocks for the
        # result; with no running loops it runs aclose with asyncio.run.
        """Synchronously destroy the proxy session and clear private state."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is not None and (
            self._event_loop is None or current_loop is self._event_loop
        ):
            return current_loop.create_task(self.aclose())
        if self._event_loop is not None and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.aclose(), self._event_loop)
            if current_loop is not None:
                return asyncio.wrap_future(future, loop=current_loop)
            return future.result()
        return asyncio.run(self.aclose())

    def _record_tool_exchange(self, call: ToolCall, response: ToolResponse) -> None:
        # [FUNCTION DOC]
        # HELPER: records a single call as a one-call batch message, its response in the conversation's tool_calls
        # metadata, and the response message.
        assert self._session is not None
        self._session.add_messages([ParallelToolCall(tool_calls=[call])])
        self._session.conversation.metadata["tool_calls"].append(
            ToolCallResponse(tool_call=call, tool_response=response)
        )
        self._session.add_messages([response])

    @staticmethod
    def _make_tool_call(
        name: str,
        arguments: dict[str, Any],
        call_id: str | None,
        *,
        parse_error: str | None = None,
    ) -> ToolCall:
        # [FUNCTION DOC]
        # HELPER: builds a native ToolCall with name and arguments, setting the id only when given and storing a
        # parse error in metadata under "error".
        values: dict[str, Any] = {"name": name, "arguments": arguments}
        if call_id is not None:
            values["id"] = call_id
        if parse_error is not None:
            values["metadata"] = {"error": parse_error}
        return ToolCall(**values)

    def _increment_step(self) -> None:
        # [FUNCTION DOC]
        # HELPER: increments state step_count.
        self._state.step_count += 1

    def _public_tools(self) -> list[Tool]:
        # [FUNCTION DOC]
        # HELPER: returns public OpenEnv Tool descriptors (name, description or "", input schema or {}) for the
        # episode's tools.
        return [
            Tool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.input_schema or {},
            )
            for tool in self._tools
        ]

    def _visible_messages(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: returns the conversation messages dumped in JSON mode, or [] without a session.
        if self._session is None:
            return []
        return [
            message.model_dump(mode="json")
            for message in self._session.conversation.messages
        ]

    def _last_assistant_response(self) -> str:
        # [FUNCTION DOC]
        # HELPER: returns the content of the most recent assistant Text message tagged "text", or "".
        if self._session is None:
            return ""
        for message in reversed(self._session.conversation.messages):
            if (
                isinstance(message, Text)
                and message.role == "assistant"
                and message.tag == "text"
            ):
                return message.content
        return ""

    def _public_metadata(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: returns public metadata: benchmark name, ThinkingBox revision, data release/revision (bundle
        # values or pinned defaults), selected agent, full execution provenance when bundle and settings exist, and
        # the case's scenario and tags when a case is loaded.
        metadata: dict[str, Any] = {
            "benchmark": benchmark_data.DATASET_NAME,
            "thinkingbox_revision": self._runtime_provenance.identity,
            "data_release": (
                self._bundle.release_name
                if self._bundle is not None
                else DATA_RELEASE_NAME
            ),
            "data_revision": (
                self._bundle.revision if self._bundle is not None else DATA_COMMIT
            ),
        }
        if self._selected_agent is not None:
            metadata["agent"] = self._selected_agent
        if self._bundle is not None and self._settings is not None:
            metadata["execution_provenance"] = ThinkingBoxExecutionProvenance(
                thinkingbox_revision=self._runtime_provenance.identity,
                thinkingbox_source_sha256=self._runtime_provenance.source_sha256,
                thinkingbox_source_type=self._runtime_provenance.source_type,
                data_release=self._bundle.release_name,
                data_revision=self._bundle.revision,
                config_sha256=self._settings.config_sha256,
                data_bundle_sha256=self._bundle.bundle_sha256,
                manifest_path=self._bundle.manifest_path.relative_to(
                    self._bundle.root
                ).as_posix(),
                manifest_sha256=self._bundle.manifest_sha256,
                manifest_uids_sha256=self._bundle.manifest_uids_sha256,
                task_count=self._bundle.task_count,
            ).model_dump(mode="json")
        if self._tc is not None:
            scenario = self._tc.metadata.get("scenario")
            if scenario:
                metadata["scenario"] = scenario
            metadata["tags"] = self._tc.tags.model_dump(mode="json")
        return metadata

    def _error_observation(self, message: str) -> ThinkingBoxObservation:
        # [FUNCTION DOC]
        # HELPER: returns a non-terminal "error" observation with the message, visible messages, step count and
        # public metadata.
        return ThinkingBoxObservation(
            kind="error",
            task_uid=self._state.task_uid,
            error=message,
            messages=self._visible_messages(),
            steps_taken=self._state.step_count,
            metadata=self._public_metadata(),
        )

    def _latch_infrastructure(self, stage: str, exc: Exception) -> None:
        # [FUNCTION DOC]
        # HELPER: logs the failure (redacted exception, stage, whether primary, task and episode ids) and records
        # the stage and exception type name only for the first failure of the episode.
        primary = self._infrastructure_stage is None
        logger.error(
            "ThinkingBox infrastructure failure",
            extra={
                "event_name": "thinkingbox.infrastructure_failure",
                "tb_stage": stage,
                "tb_primary": primary,
                "tb_task_uid": self._state.task_uid,
                "tb_episode_id": self._state.episode_id,
                "exception_type": type(exc).__name__,
            },
            exc_info=_redacted_exc_info(exc),
        )
        if primary:
            self._infrastructure_stage = stage
            self._infrastructure_detail = type(exc).__name__

    def _clear_sensitive(self) -> None:
        # [FUNCTION DOC]
        # HELPER: drops all private per-episode data (session, case, settings, tools and allow-lists, user context
        # and model, query, bot instructions, bundle, selected agent).
        self._session = None
        self._tc = None
        self._settings = None
        self._tools = []
        self._allowed_tool_names = set()
        self._direct_response_tools = {}
        self._end_turn_tools = set()
        self._user_context = None
        self._user_model = None
        self._query = None
        self._bot_instructions = None
        self._bundle = None
        self._selected_agent = None
