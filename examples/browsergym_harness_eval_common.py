# [MODULE DOC]
# HELPER: Shared helpers for the BrowserGym harness examples: defaults (image, benchmark, task, steps, model, sampling),
# action extraction from free-form model text into BrowserGym tool calls, chat-transcript conversion, an OpenAI-backed
# white-box model step, Docker runtime start/attach, session factory creation, white-box and black-box episode runners
# that verify and score each rollout, summary helpers, and a tiny threaded HTTP JSON-RPC server exposing a
# SessionMCPBridge at /mcp for CLI agents.
"""Shared helpers for BrowserGym harness inference and evaluation examples."""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from browsergym_env import BrowserGymEnv
from browsergym_env.harness import (
    BrowserGymSessionFactory,
    build_browsergym_action_tool_call,
)
from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider
from openenv.core.env_server.mcp_types import JsonRpcErrorCode, JsonRpcResponse
from openenv.core.harness import (
    _resolve_env_reward,
    CLIHarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
    SessionMCPBridge,
    VerifyResult,
)
from openenv.core.llm_client import LLMResponse

DEFAULT_BROWSERGYM_IMAGE = "browsergym-env:latest"
DEFAULT_BENCHMARK = "miniwob"
DEFAULT_TASK_NAME = "click-test"
DEFAULT_MAX_STEPS = 8
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 200
FALLBACK_ACTION = "noop()"

SYSTEM_PROMPT = """You control a web browser through BrowserGym.

Reply with exactly one BrowserGym action string, such as:
- click('13')
- fill('42', 'hello world')
- send_keys('Enter')
- scroll('down')
- noop()

Use only one action. Do not explain your reasoning.
If you are unsure, reply with noop().
"""

ACTION_PREFIX_RE = re.compile(r"^(action|next action)\s*[:-]\s*", re.IGNORECASE)
ACTION_PATTERN = re.compile(
    r"(click|fill|send_keys|scroll|noop)\s*\(.*?\)",
    re.IGNORECASE | re.DOTALL,
)


def _serialize_json(value: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: Serialises a value to text: None becomes "", strings pass through, others become sorted-key JSON (non-serialisable values via str).
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def extract_browsergym_action(
    response_text: str,
    fallback_action: str = FALLBACK_ACTION,
) -> str:
    # [FUNCTION DOC]
    # Searches each non-empty line (stripped of backticks and an "action:"/"next action:" prefix) for a click/fill/
    # send_keys/scroll/noop call (case-insensitive), then the whole text. The match has whitespace collapsed, the function
    # name lowercased and trailing semicolons removed. Returns it, or fallback_action if nothing matches or text is empty.
    """Extract a BrowserGym action from free-form model output."""

    if not response_text:
        return fallback_action

    for raw_line in response_text.splitlines():
        line = raw_line.strip().strip("`")
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = re.sub(r"\s+", " ", match.group(0).strip())
            open_paren = action.find("(")
            if open_paren > 0:
                action = f"{action[:open_paren].lower()}{action[open_paren:]}"
            return action.rstrip(";")

    match = ACTION_PATTERN.search(response_text)
    if match:
        action = re.sub(r"\s+", " ", match.group(0).strip())
        open_paren = action.find("(")
        if open_paren > 0:
            action = f"{action[:open_paren].lower()}{action[open_paren:]}"
        return action.rstrip(";")

    return fallback_action


def build_browsergym_tool_call_from_output(response_text: str):
    # [FUNCTION DOC]
    # Extracts an action and builds a BrowserGym action tool call from it; if building raises ValueError, builds the call for "noop()" instead.
    """Convert free-form model output into a BrowserGym tool call."""

    action = extract_browsergym_action(response_text)
    try:
        return build_browsergym_action_tool_call(action)
    except ValueError:
        return build_browsergym_action_tool_call(FALLBACK_ACTION)


def build_browsergym_chat_messages(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Returns a chat transcript starting with the stripped system prompt; tool messages become user messages prefixed
    # "Observation from <name>:", unknown roles become "user", and content is stringified (None/empty becomes "").
    """Convert harness messages into a plain chat-completions transcript."""

    chat_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""

        if role == "tool":
            tool_name = message.get("name", "tool")
            chat_messages.append(
                {
                    "role": "user",
                    "content": f"Observation from {tool_name}:\n{content}",
                }
            )
            continue

        if role not in {"user", "assistant", "system"}:
            role = "user"
        chat_messages.append({"role": role, "content": str(content)})

    return chat_messages


def build_openai_model_step(
    client: OpenAI,
    model: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    # [FUNCTION DOC]
    # Returns a model_step(messages, tools, sampling) closure for the white-box harness. It ignores tools, calls the
    # chat-completions API with the converted transcript (temperature/max_tokens taken from sampling when present, else
    # the defaults), converts the reply into exactly one BrowserGym tool call, and returns a ModelStepResult wrapping an
    # LLMResponse. Side effects: network call per step.
    """Build a white-box model step for BrowserGym inference/evaluation."""

    def model_step(messages, tools, sampling):
        del tools
        completion = client.chat.completions.create(
            model=model,
            messages=build_browsergym_chat_messages(
                messages,
                system_prompt=system_prompt,
            ),
            temperature=sampling.get("temperature", temperature),
            max_tokens=sampling.get("max_tokens", max_tokens),
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        tool_call = build_browsergym_tool_call_from_output(response_text)
        return ModelStepResult(
            response=LLMResponse(content=response_text, tool_calls=[tool_call]),
        )

    return model_step


def create_openai_client(
    *,
    api_base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    # [FUNCTION DOC]
    # Creates a synchronous OpenAI client with optional base URL and an API key from the argument, OPENAI_API_KEY or
    # API_KEY (omitted if none, leaving OpenAI's own resolution). Returns the client.
    """Create a sync OpenAI-compatible client for the examples."""

    client_kwargs: dict[str, Any] = {}
    if api_base_url:
        client_kwargs["base_url"] = api_base_url

    resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if resolved_key:
        client_kwargs["api_key"] = resolved_key

    return OpenAI(**client_kwargs)


@dataclass
class BrowserGymRuntime(AbstractContextManager["BrowserGymRuntime"]):
    # [CLASS DOC]
    # Dataclass context manager holding a BrowserGym server base URL and, when it started the container, the Docker
    # provider. Closing (or exiting the context) stops the container once.
    """Runtime information for a BrowserGym server used by the examples."""

    base_url: str
    provider: LocalDockerProvider | None = None

    def close(self) -> None:
        # [FUNCTION DOC]
        # Stops the container if this runtime owns a provider and clears the provider reference; otherwise does nothing.
        if self.provider is not None:
            self.provider.stop_container()
            self.provider = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # [FUNCTION DOC]
        # Context-manager exit: calls close; does not suppress exceptions.
        self.close()


def start_browsergym_runtime(
    *,
    base_url: str | None = None,
    image: str = DEFAULT_BROWSERGYM_IMAGE,
    benchmark: str = DEFAULT_BENCHMARK,
    task_name: str = DEFAULT_TASK_NAME,
) -> BrowserGymRuntime:
    # [FUNCTION DOC]
    # If base_url is given, returns a runtime that just points at it. Otherwise starts a Docker container from the image
    # with BROWSERGYM_BENCHMARK and BROWSERGYM_TASK_NAME env vars, waits until ready, and returns a runtime that owns it.
    # Side effects: Docker container start; may raise provider errors or timeouts.
    """Start or attach to a BrowserGym server for evaluation examples."""

    if base_url is not None:
        return BrowserGymRuntime(base_url=base_url)

    provider = LocalDockerProvider()
    env_vars = {
        "BROWSERGYM_BENCHMARK": benchmark,
        "BROWSERGYM_TASK_NAME": task_name,
    }
    runtime_base_url = provider.start_container(image=image, env_vars=env_vars)
    provider.wait_for_ready(runtime_base_url)
    return BrowserGymRuntime(base_url=runtime_base_url, provider=provider)


def build_browsergym_session_factory(
    *,
    base_url: str,
    task_name: str = DEFAULT_TASK_NAME,
) -> BrowserGymSessionFactory:
    # [FUNCTION DOC]
    # Returns a BrowserGymSessionFactory that creates a new BrowserGymEnv client per session for the given base URL and default task.
    """Create a BrowserGym session factory for a running environment server."""

    return BrowserGymSessionFactory(
        client_factory=lambda: BrowserGymEnv(base_url=base_url),
        default_task=task_name,
    )


def _rollout_final_state(rollout: HarnessRolloutResult) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Converts a rollout into a plain final-state dict: done flag, metrics copy, events (type/payload) and tool
    # trace entries (tool name, arguments, result data/done/metadata/error).
    return {
        "done": rollout.done,
        "metrics": dict(rollout.metrics),
        "events": [
            {
                "type": event.type,
                "payload": dict(event.payload),
            }
            for event in rollout.events
        ],
        "tool_trace": [
            {
                "tool_name": entry.tool_name,
                "arguments": dict(entry.arguments),
                "result": {
                    "data": entry.result.data,
                    "done": entry.result.done,
                    "metadata": dict(entry.result.metadata),
                    "error": entry.result.error,
                },
            }
            for entry in rollout.tool_trace
        ],
    }


@dataclass
class EpisodeEvaluation:
    # [CLASS DOC]
    # Dataclass with one episode's rollout, verify result, resolved reward, done flag, success flag (reward > 0) and step count.
    """Normalized result from one inference/evaluation episode."""

    rollout: HarnessRolloutResult
    verify: VerifyResult
    reward: float
    done: bool
    success: bool
    step_count: int


def _finalize_episode(
    session,
    rollout: HarnessRolloutResult,
) -> EpisodeEvaluation:
    # [FUNCTION DOC]
    # HELPER: Calls session.verify with the rollout transcript and final state, resolves the reward from rollout/verify,
    # sets done if either reports done, takes step_count from verify metrics (default: tool trace length), and returns an EpisodeEvaluation.
    verify = session.verify(
        transcript=rollout.messages,
        final_state=_rollout_final_state(rollout),
    )
    reward = _resolve_env_reward(rollout, verify)
    done = bool(verify.done or rollout.done)
    step_count = int(verify.metrics.get("step_count", len(rollout.tool_trace)))
    success = reward > 0.0
    return EpisodeEvaluation(
        rollout=rollout,
        verify=verify,
        reward=reward,
        done=done,
        success=success,
        step_count=step_count,
    )


def run_white_box_episode(
    *,
    session_factory: BrowserGymSessionFactory,
    model_step,
    limits: HarnessRunLimits,
    task: Any = None,
    seed: int | None = None,
    episode_id: str | None = None,
) -> EpisodeEvaluation:
    # [FUNCTION DOC]
    # Creates a session from the factory, runs MCPHarnessAdapter white-box with the model step and limits, finalises and
    # returns the EpisodeEvaluation. Always closes the session.
    """Run one BrowserGym episode through the white-box harness path."""

    session = session_factory.create(task=task, seed=seed, episode_id=episode_id)
    try:
        rollout = MCPHarnessAdapter().run_white_box(
            model_step=model_step,
            session=session,
            limits=limits,
        )
        return _finalize_episode(session, rollout)
    finally:
        session.close()


def run_black_box_episode(
    *,
    session_factory: BrowserGymSessionFactory,
    harness_adapter: CLIHarnessAdapter,
    limits: HarnessRunLimits,
    task: Any = None,
    seed: int | None = None,
    episode_id: str | None = None,
) -> EpisodeEvaluation:
    # [FUNCTION DOC]
    # Creates a session from the factory, runs the given CLI harness adapter in black-box mode, finalises and returns the
    # EpisodeEvaluation. Always closes the session.
    """Run one BrowserGym episode through the black-box harness path."""

    session = session_factory.create(task=task, seed=seed, episode_id=episode_id)
    try:
        rollout = harness_adapter.run_black_box(
            session=session,
            limits=limits,
        )
        return _finalize_episode(session, rollout)
    finally:
        session.close()


def summarize_episodes(episodes: list[EpisodeEvaluation]) -> dict[str, float]:
    # [FUNCTION DOC]
    # Returns a dict of float metrics across episodes: count, average reward, success rate and average steps (all zeros if empty).
    """Aggregate reward and completion metrics across episodes."""

    if not episodes:
        return {
            "episodes": 0.0,
            "avg_reward": 0.0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
        }

    episode_count = len(episodes)
    total_reward = sum(episode.reward for episode in episodes)
    total_success = sum(1 for episode in episodes if episode.success)
    total_steps = sum(episode.step_count for episode in episodes)
    return {
        "episodes": float(episode_count),
        "avg_reward": total_reward / episode_count,
        "success_rate": total_success / episode_count,
        "avg_steps": total_steps / episode_count,
    }


def format_episode_summary(index: int, episode: EpisodeEvaluation) -> str:
    # [FUNCTION DOC]
    # Returns a one-line text summary of an episode (index, reward, success, done, steps).
    """Render one episode summary line for CLI output."""

    return (
        f"Episode {index}: reward={episode.reward:.2f} "
        f"success={episode.success} done={episode.done} steps={episode.step_count}"
    )


def _normalize_mcp_call_result(result: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Converts a bridge tools/call result into MCP-style output: a text content item (error text or serialised data),
    # structuredContent with data/done/metadata/error, raw data, and isError true when an error is present.
    text_payload = result.get("error") or _serialize_json(result.get("data"))
    return {
        "content": [{"type": "text", "text": text_payload}],
        "structuredContent": {
            "data": result.get("data"),
            "done": result.get("done", False),
            "metadata": result.get("metadata", {}),
            "error": result.get("error"),
        },
        "data": result.get("data"),
        "isError": bool(result.get("error")),
    }


class SessionMCPHttpServer(AbstractContextManager["SessionMCPHttpServer"]):
    # [CLASS DOC]
    # Context manager running a threaded HTTP server (default 127.0.0.1, ephemeral port) in a daemon thread that forwards
    # JSON-RPC POST requests on /mcp to a SessionMCPBridge and answers GET /mcp with a status payload.
    """Tiny HTTP wrapper that exposes SessionMCPBridge on /mcp."""

    def __init__(
        self,
        bridge: SessionMCPBridge,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        # [FUNCTION DOC]
        # Stores the bridge, host and port; no server or thread yet.
        self._bridge = bridge
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        # [FUNCTION DOC]
        # Returns the http://host:port/mcp URL of the running server. Raises RuntimeError if not started.
        if self._server is None:
            raise RuntimeError("HTTP bridge is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/mcp"

    def start(self) -> "SessionMCPHttpServer":
        # [FUNCTION DOC]
        # Starts the server if not already running (idempotent) and returns self. Defines a request handler class closed over
        # the bridge, binds the server and launches serve_forever in a daemon thread. Side effects: opens a listening socket.
        if self._server is not None:
            return self

        bridge = self._bridge

        class Handler(BaseHTTPRequestHandler):
            # [CLASS DOC]
            # HELPER: Request handler for the bridge server: JSON responses, silent logging, GET health on /mcp, POST JSON-RPC on /mcp, 404 elsewhere.
            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                # [FUNCTION DOC]
                # Writes a JSON body with the given status, Content-Type and Content-Length headers.
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                # [FUNCTION DOC]
                # Suppresses the default per-request logging.
                del format, args

            def do_GET(self) -> None:  # noqa: N802
                # [FUNCTION DOC]
                # Responds 404 for paths other than /mcp; otherwise returns a status ok / transport jsonrpc JSON payload.
                if self.path != "/mcp":
                    self.send_error(404)
                    return
                self._send_json({"status": "ok", "transport": "jsonrpc"})

            def do_POST(self) -> None:  # noqa: N802
                # [FUNCTION DOC]
                # Responds 404 for paths other than /mcp. Reads the body by Content-Length and parses JSON (parse errors return a
                # JSON-RPC parse-error response). Passes the request to bridge.handle_request, mapping KeyError to method-not-found,
                # ValueError to invalid-params and other exceptions to internal-error responses. For tools/call results, normalises the
                # result into MCP content format. Always replies with HTTP 200 and the JSON-RPC payload.
                # KNOWN ISSUE: if the body parses to a non-object (e.g. a JSON list), the later request.get calls raise AttributeError outside the error handling.
                if self.path != "/mcp":
                    self.send_error(404)
                    return

                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                try:
                    request = json.loads(raw_body or b"{}")
                except json.JSONDecodeError:
                    self._send_json(
                        JsonRpcResponse.error_response(
                            JsonRpcErrorCode.PARSE_ERROR,
                            message="Parse error",
                        ).model_dump()
                    )
                    return

                try:
                    response = bridge.handle_request(request)
                except KeyError as exc:
                    response = JsonRpcResponse.error_response(
                        JsonRpcErrorCode.METHOD_NOT_FOUND,
                        message=str(exc.args[0]) if exc.args else "Method not found",
                        request_id=request.get("id"),
                    ).model_dump()
                except ValueError as exc:
                    response = JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        message=str(exc),
                        request_id=request.get("id"),
                    ).model_dump()
                except Exception as exc:
                    response = JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INTERNAL_ERROR,
                        message=str(exc),
                        request_id=request.get("id"),
                    ).model_dump()

                if (
                    request.get("method") == "tools/call"
                    and "result" in response
                    and isinstance(response["result"], dict)
                ):
                    response["result"] = _normalize_mcp_call_result(response["result"])

                self._send_json(response)

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="browsergym-session-mcp-http",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        # [FUNCTION DOC]
        # Shuts down and closes the server if running, then joins the thread (5 s timeout) and clears references.
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # [FUNCTION DOC]
        # Context-manager exit: calls close; does not suppress exceptions.
        self.close()


__all__ = [
    "BrowserGymRuntime",
    "DEFAULT_BENCHMARK",
    "DEFAULT_BROWSERGYM_IMAGE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MODEL",
    "DEFAULT_TASK_NAME",
    "EpisodeEvaluation",
    "FALLBACK_ACTION",
    "SYSTEM_PROMPT",
    "SessionMCPHttpServer",
    "build_browsergym_chat_messages",
    "build_browsergym_session_factory",
    "build_browsergym_tool_call_from_output",
    "build_openai_model_step",
    "create_openai_client",
    "extract_browsergym_action",
    "format_episode_summary",
    "run_white_box_episode",
    "run_black_box_episode",
    "start_browsergym_runtime",
    "summarize_episodes",
]
