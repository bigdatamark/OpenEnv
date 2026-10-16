# [MODULE DOC]
# Synchronous typed client for a deployed harbor_env server: Task API discovery over
# plain HTTP (httpx) and rollout execution / capability queries via MCP tool calls over
# the OpenEnv WebSocket client.
"""Typed client for a deployed harbor_env server.

Two surfaces, matching the server: the Task API for discovery (plain HTTP, cheap, side-effect free)
and one MCP tool for execution (a single long call).

```python
from openenv.harbor.client import HarborEnv

with HarborEnv(base_url="http://localhost:8000") as env:
    split = env.splits()[0]["name"]
    print(env.num_tasks(split))
    result = env.run_rollout(split=split, task_index=0, harness="opencode", sandbox="e2b")
    print(result.reward, result.turns[0].completion_token_ids[:8])
```
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import httpx
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.mcp_client import MCPToolClient
from openenv.core.utils import run_async_safely

from .models import HarborRolloutResult, HarborTaskRef

# A rollout is minutes. The base client defaults to 60s, which fires mid-run.
_DEFAULT_MESSAGE_TIMEOUT_S = 1800.0


class HarborEnv(MCPToolClient):
    # [CLASS DOC]
    # MCPToolClient subclass holding an extra httpx client for the Task API endpoints, with
    # a 1800s default WebSocket message timeout.
    """Client for `harbor_env`.

    Args:
        base_url (`str`):
            Server root, e.g. `http://localhost:8000`.
        message_timeout_s (`float`, *optional*, defaults to `1800.0`):
            Websocket message timeout. Raised well above the base client's 60s because a single
            rollout runs for minutes and the default would time out mid-call.
    """

    def __init__(
        self,
        base_url: str,
        *,
        message_timeout_s: float = _DEFAULT_MESSAGE_TIMEOUT_S,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC]
        # Initializes the MCP client with base_url and message timeout (extra kwargs passed
        # through) and creates an httpx client rooted at base_url with a 120s timeout.
        super().__init__(
            base_url=base_url, message_timeout_s=message_timeout_s, **kwargs
        )
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0)

    # --- discovery (Task API) --------------------------------------------
    def splits(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # GET /harbor_env/splits; returns the JSON list of split descriptors. Raises httpx.HTTPStatusError on error status.
        """Datasets this server offers, with task counts."""
        return self._http.get("/harbor_env/splits").raise_for_status().json()

    def num_tasks(self, split: str = "") -> int:
        # [FUNCTION DOC]
        # POST /harbor_env/num_tasks with the split; returns the int num_tasks (0 if absent).
        payload = self._post("/harbor_env/num_tasks", {"split": split})
        return int(payload.get("num_tasks", 0))

    def get_task(self, split: str, index: int) -> HarborTaskRef:
        # [FUNCTION DOC]
        # POST /harbor_env/task with split and index; validates payload["task"] (or the whole
        # payload) as a HarborTaskRef and returns it.
        payload = self._post("/harbor_env/task", {"split": split, "index": index})
        return HarborTaskRef.model_validate(payload.get("task", payload))

    def get_task_range(
        self, split: str, start: int = 0, stop: int = 20
    ) -> list[HarborTaskRef]:
        # [FUNCTION DOC]
        # POST /harbor_env/task_range with split/start/stop; returns a list of HarborTaskRef from payload["tasks"].
        payload = self._post(
            "/harbor_env/task_range", {"split": split, "start": start, "stop": stop}
        )
        return [HarborTaskRef.model_validate(t) for t in payload.get("tasks", [])]

    # --- execution (MCP) ---------------------------------------------------
    def run_rollout(
        self,
        *,
        split: str = "",
        task_index: int = 0,
        harness: str = "opencode",
        sandbox: str = "e2b",
        reward_key: str = "",
        keep_sandbox: bool = False,
        force_build: bool = False,
        llm_url: str = "",
        model: str = "",
        api_key: str = "",
        auth_header: str = "",
        provider: str = "openai",
        purpose: str = "auto",
        eval_sampling: dict[str, Any] | None = None,
        agent_timeout_sec: float = 0.0,
        agent_step_limit: int = 0,
        sampling: dict[str, Any] | None = None,
    ) -> HarborRolloutResult:
        # [FUNCTION DOC]
        # Calls the "run_rollout" MCP tool with all arguments; provider, purpose, eval_sampling
        # and sampling are only sent when non-default (for compatibility with older servers).
        # Parses the text result as a HarborRolloutResult and returns it. Raises RuntimeError if
        # the tool reports an error; pydantic ValidationError on malformed results.
        # KNOWN ISSUE: the docstring omits keep_sandbox, force_build, provider, purpose and eval_sampling.
        """Run one rollout and return its result.

        `harness` and `sandbox` are per-call, so consecutive rollouts can use different agents and
        different backends against the same server.

        Args:
            split (`str`, *optional*):
                Dataset spec. Defaults to the server's first.
            task_index (`int`, *optional*, defaults to `0`):
                Index into the split.
            harness (`str`, *optional*, defaults to `"opencode"`):
                A validated seam name, or a `module:Class` import path for your own agent.
            sandbox (`str`, *optional*, defaults to `"e2b"`):
                Harbor environment type, e.g. `e2b` or `modal`.
            reward_key (`str`, *optional*):
                Which reward key is the training signal, for multi-reward tasks.
            llm_url (`str`, *optional*):
                Engine for THIS rollout. Probed on first use (cached per engine) and the measured
                tier decides `rollout_type`: token ids plus processed logprobs give `train`, anything
                less gives `eval`. Omit to use whatever engine the server was booted with, if any.
            model (`str`, *optional*):
                Served model id at `llm_url`. Resolved automatically when that endpoint serves
                exactly one model.
            api_key (`str`, *optional*):
                Credential for `llm_url`, when it is token-gated.
            auth_header (`str`, *optional*):
                Header to send the credential under, when not `Authorization`.
            agent_step_limit (`int`, *optional*):
                Cap model calls at the proxy, including auxiliary calls, and set a native step
                limit where supported. `0` leaves model calls uncapped.
            agent_timeout_sec (`float`, *optional*):
                Override the agent's execution timeout; `0` uses the task's configured value.
                Sandbox setup retains its separate build and healthcheck timeouts.
            sampling (`dict`, *optional*):
                Explicit full-vocabulary training policy, e.g. `{"temperature": 0.8}`. Use the
                trainer's recompute temperature. Requested and submitted policies remain separate.

        Returns:
            [`HarborRolloutResult`]: Reward, per-turn token ids and logprobs, and findings.
        """
        raw = self._call(
            "run_rollout",
            split=split,
            task_index=task_index,
            harness=harness,
            sandbox=sandbox,
            reward_key=reward_key,
            keep_sandbox=keep_sandbox,
            force_build=force_build,
            llm_url=llm_url,
            model=model,
            api_key=api_key,
            auth_header=auth_header,
            # Preserve the established OpenAI/auto wire contract with older
            # servers. Explicit new semantics must still be sent (and rejected
            # by a server that cannot implement them).
            **({"provider": provider} if provider != "openai" else {}),
            **({"purpose": purpose} if purpose != "auto" else {}),
            **({"eval_sampling": eval_sampling} if eval_sampling is not None else {}),
            agent_timeout_sec=agent_timeout_sec,
            agent_step_limit=agent_step_limit,
            **({"sampling": sampling} if sampling is not None else {}),
        )
        return HarborRolloutResult.model_validate_json(_as_text(raw))

    def capabilities(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Calls the "capabilities" MCP tool and returns its JSON-decoded dict.
        """Harnesses, sandboxes, datasets and LLM status for this server."""
        return json.loads(_as_text(self._call("capabilities")))

    # --- internals ---------------------------------------------------------
    def _call(self, name: str, **kwargs: Any) -> Any:
        # [FUNCTION DOC]
        # HELPER: sends a CallToolAction via step() (awaiting it through run_async_safely in
        # async mode). Returns observation.result for CallToolObservation, raising RuntimeError
        # with the error message/type if the tool failed; otherwise returns the raw observation.
        """Call an MCP tool from synchronous code.

        `MCPToolClient.call_tool` cannot be used here. It is a coroutine that internally does
        `await self.step(action)`, but `EnvClient.step` dispatches on execution mode and returns a
        concrete `StepResult` in sync mode, so awaiting it raises `TypeError: object StepResult
        can't be used in 'await' expression`. Driving `step` directly works in both modes.
        """
        result = self.step(CallToolAction(tool_name=name, arguments=kwargs))
        if inspect.isawaitable(result):  # async mode returns an awaitable instead
            result = run_async_safely(result)

        observation = result.observation
        if isinstance(observation, CallToolObservation):
            if observation.error is not None:
                raise RuntimeError(
                    f"tool {name!r} failed: {observation.error.message} "
                    f"({observation.error.error_type.value})"
                )
            return observation.result
        return observation

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: POSTs JSON body to path, raises on error status, returns the decoded JSON.
        return self._http.post(path, json=body).raise_for_status().json()

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the httpx client, then always closes the underlying MCP/WebSocket client.
        try:
            self._http.close()
        finally:
            super().close()


def _as_text(raw: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: extracts text from an MCP result: strings as-is; dicts use content[0]["text"]
    # if present, otherwise JSON-dump the dict; non-empty lists use the first element's
    # text attribute (or str of it); anything else str().
    """MCP tool results arrive as text content; unwrap whatever shape the transport used."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        content = raw.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
        return json.dumps(raw)
    if isinstance(raw, list) and raw:
        first = raw[0]
        return str(getattr(first, "text", first))
    return str(raw)
