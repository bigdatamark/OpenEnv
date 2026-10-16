# [MODULE DOC]
# Builds the FastAPI app with the Gradio web interface (used when
# ENABLE_WEB_INTERFACE is set). It wraps the standard app from
# create_fastapi_app and adds: redirects from "/" and "/web" to "/web/",
# GET /web/metadata, a /ws/ui WebSocket broadcasting episode state updates,
# POST /web/reset, POST /web/step, GET /web/state, and the Gradio Blocks mounted
# at /web (default playground, optionally combined with an env-provided custom
# UI in tabs). The web UI drives a single environment instance owned by a
# WebInterfaceManager, shared by all visitors and separate from /ws sessions.
# Also generates the "Quick Start" Python snippet by statically parsing the
# environment package's client.py and __init__.py, loads README content from
# the filesystem, and keeps legacy action-field metadata helpers.
# KNOWN ISSUE: the module docstring mentions "openenv push --enable-interface",
# but the push flag is "--interface".
# SPDX-License-Identifier: BSD-3-Clause

"""
Web interface for OpenEnv environments.

When ENABLE_WEB_INTERFACE is set, the server exposes a Gradio UI at /web for
reset, step, and state observation. Controlled by the CLI enable_interface
option (e.g. openenv push --enable-interface) or ENABLE_WEB_INTERFACE env var.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, get_origin, List, Optional, Tuple, Type

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from ._utils import overrides_method
from .gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME
from .gradio_ui import build_gradio_app, get_gradio_display_title
from .interfaces import Environment
from .serialization import deserialize_action_with_preprocessing, serialize_observation
from .types import Action, EnvironmentMetadata, Observation, State


_CLIENT_BASES = {"EnvClient", "MCPClientBase", "MCPToolClient"}


def _read_package(package: str) -> Tuple[Optional[str], set]:
    # [FUNCTION DOC]
    # HELPER: Locates package via importlib (without importing it) and parses its
    # client.py to find the first top-level class whose base (ignoring generics
    # and module prefixes) is EnvClient, MCPClientBase or MCPToolClient, and its
    # __init__.py to collect imported names and all string constants (as a proxy
    # for exports). Returns (client class name or None, set of names); (None,
    # empty set) if the package is not found. Raises SyntaxError on unparsable
    # files.
    """
    The client class defined in the package's `client.py` and the names its
    `__init__.py` exports. The files are parsed, not imported, so the server
    never loads client code.
    """
    import ast
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None, set()
    root = Path(next(iter(spec.submodule_search_locations)))

    def parse(name: str) -> Optional[ast.Module]:
        # [FUNCTION DOC]
        # HELPER: Parses the named file under the package root into an AST, or returns
        # None if it does not exist.
        path = root / name
        return ast.parse(path.read_text()) if path.exists() else None

    client = None
    tree = parse("client.py")
    for node in tree.body if tree else []:
        if isinstance(node, ast.ClassDef):
            bases = {ast.unparse(b).split("[")[0].split(".")[-1] for b in node.bases}
            if bases & _CLIENT_BASES:
                client = node.name
                break

    exports = set()
    tree = parse("__init__.py")
    for node in ast.walk(tree) if tree else []:
        if isinstance(node, ast.ImportFrom):
            exports.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            exports.add(node.value)
    return client, exports


def _example_value(annotation: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns a placeholder literal string for a type annotation (by its
    # origin): False, 0, 0.0, "...", [], {} for bool/int/float/str/list/dict, else
    # "...".
    """A placeholder literal for a field of the given type."""
    origin = get_origin(annotation) or annotation
    literals = {
        bool: "False",
        int: "0",
        float: "0.0",
        str: '"..."',
        list: "[]",
        dict: "{}",
    }
    return literals.get(origin, "...")


def get_quick_start_markdown(
    metadata: Optional[EnvironmentMetadata],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
) -> str:
    # [FUNCTION DOC]
    # Returns Markdown showing how to connect with Python. Package name is the
    # metadata name lowercased with spaces as underscores (default "env"); if no
    # client class is found there, retries with the action class's top-level module
    # name (unless it is the same, openenv, models or server). Base URL is
    # https://<SPACE_HOST> when set, else http://localhost:8000. Without a client
    # class, returns a short pointer to the README. Otherwise includes a pip
    # install line when SPACE_ID is set, and a sync client snippet: for
    # CallToolAction-based envs, listing tools (plus a call_tool hint); else a
    # reset and a step with the action built from its required fields with
    # placeholder values, importing the action from the package if exported, else
    # from its module.
    """
    Build the Quick Start markdown shown next to the web interface.

    The client class is the one the environment package exports, the URL is the
    Space's own when running on Hugging Face Spaces, and the example action uses
    the action's real fields.
    """
    import os

    from .mcp_types import CallToolAction

    package = (metadata.name if metadata else "env").replace(" ", "_").lower()
    client, exports = _read_package(package)
    if client is None:
        # The env name can differ from its package (reasoning_gym vs reasoning_gym_env).
        module_root = action_cls.__module__.split(".")[0]
        if module_root not in (package, "openenv", "models", "server"):
            package = module_root
            client, exports = _read_package(package)
    space_id = os.environ.get("SPACE_ID")
    space_host = os.environ.get("SPACE_HOST")
    base_url = f"https://{space_host}" if space_host else "http://localhost:8000"

    if client is None:
        return (
            "### Connect to this environment\n\n"
            f"The server is at `{base_url}`. See the environment's README for how to connect."
        )

    lines = ["### Connect to this environment", ""]
    if space_id:
        lines += [
            "```bash",
            f"pip install git+https://huggingface.co/spaces/{space_id}",
            "```",
            "",
        ]

    if issubclass(action_cls, CallToolAction):
        code = [
            f"from {package} import {client}",
            "",
            f'with {client}(base_url="{base_url}").sync() as env:',
            "    env.reset()",
            "    print([tool.name for tool in env.list_tools()])",
        ]
        after = "Then call a tool with `env.call_tool(name, **arguments)`."
    else:
        action = action_cls.__name__
        exported = action in exports
        action_module = package if exported else action_cls.__module__
        fields = ", ".join(
            f"{name}={_example_value(field.annotation)}"
            for name, field in action_cls.model_fields.items()
            if field.is_required() and name != "metadata"
        )
        code = [
            *(
                [f"from {package} import {action}, {client}"]
                if exported
                else [
                    f"from {action_module} import {action}",
                    f"from {package} import {client}",
                ]
            ),
            "",
            f'with {client}(base_url="{base_url}").sync() as env:',
            "    result = env.reset()",
            f"    result = env.step({action}({fields}))",
            "    print(result.observation, result.reward)",
        ]
        after = ""

    lines += ["```python", *code, "```"]
    if after:
        lines += ["", after]
    lines += [
        "",
        "More in the [OpenEnv documentation](https://huggingface.co/docs/openenv).",
    ]
    return "\n".join(lines)


def load_environment_metadata(
    env: Environment, env_name: Optional[str] = None
) -> EnvironmentMetadata:
    # [FUNCTION DOC]
    # Returns get_metadata() for an Environment instance. For a factory (partials
    # unwrapped) builds default metadata: name = env_name or the class/function
    # name, description "<class or env name> environment", version "1.0.0", plus
    # README content found by _load_readme_from_filesystem. Does not instantiate
    # factories.
    """
    Load environment metadata including README content.

    Args:
        env: The environment instance or factory (class, function or
            `functools.partial`). A factory is not instantiated. If an instance,
            `get_metadata()` is called.
        env_name (`str`, *optional*):
            Optional environment name for README file lookup.

    Returns:
        `EnvironmentMetadata` with loaded information.
    """
    # An Environment instance provides its own metadata. Anything else is a factory:
    # a class (e.g. PythonCodeActEnv), a factory function (e.g.
    # create_chat_environment) or a `functools.partial` of either.
    if isinstance(env, Environment):
        return env.get_metadata()

    if isinstance(env, functools.partial):
        env = env.func

    # Determine the class name for default metadata
    if inspect.isclass(env):
        class_name = env.__name__
    else:
        class_name = env_name or env.__name__

    # Default metadata
    metadata = EnvironmentMetadata(
        name=env_name or class_name,
        description=f"{class_name} environment",
        version="1.0.0",
    )

    # Try to load README from file system
    readme_content = _load_readme_from_filesystem(env_name)
    if readme_content:
        metadata.readme_content = readme_content

    return metadata


def _load_readme_from_filesystem(env_name: Optional[str]) -> Optional[str]:
    # [FUNCTION DOC]
    # HELPER: Returns the text of the first readable README among: the
    # ENV_README_PATH env var path, "envs/<env_name>/README.md" (relative to cwd,
    # when env_name given), "/app/env/README.md", "/app/README.md"; None if none.
    # Read errors are ignored.
    """
    Load README content from the filesystem.

    Tries multiple locations in order: the path given by the `ENV_README_PATH`
    environment variable, the repository path `envs/{env_name}/README.md`, and the
    container paths `/app/env/README.md` (where Space images copy the environment)
    and `/app/README.md`.
    """
    import os
    from pathlib import Path

    # Try environment variable path
    custom_path = os.environ.get("ENV_README_PATH")
    if custom_path and Path(custom_path).exists():
        try:
            return Path(custom_path).read_text(encoding="utf-8")
        except Exception:
            pass

    candidates = [Path(f"envs/{env_name}/README.md")] if env_name else []
    candidates += [Path("/app/env/README.md"), Path("/app/README.md")]
    for readme in candidates:
        if readme.exists():
            try:
                return readme.read_text(encoding="utf-8")
            except Exception:
                pass

    return None


class ActionLog(BaseModel):
    # [CLASS DOC]
    # Record of one web UI step: ISO timestamp, action dict, observation dict,
    # optional reward, done flag and step count.
    """Log entry for an action taken."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    timestamp: str = Field(description="Timestamp when action was taken")
    action: Dict[str, Any] = Field(description="Action that was taken")
    observation: Dict[str, Any] = Field(description="Observation returned from action")
    reward: Optional[float] = Field(
        default=None, description="Reward received from action"
    )
    done: bool = Field(description="Whether the episode is done after this action")
    step_count: int = Field(description="Step count when this action was taken")


class EpisodeState(BaseModel):
    # [CLASS DOC]
    # Web UI episode state: episode id, step count, current observation, action
    # log list and whether the episode was just reset.
    """Current episode state for the web interface."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: Optional[str] = Field(default=None, description="Current episode ID")
    step_count: int = Field(description="Current step count in episode")
    current_observation: Optional[Dict[str, Any]] = Field(
        default=None, description="Current observation"
    )
    action_logs: List[ActionLog] = Field(
        default_factory=list, description="List of action logs"
    )
    is_reset: bool = Field(
        default=True, description="Whether the episode has been reset"
    )


class WebInterfaceManager:
    # [CLASS DOC]
    # Holds the single environment instance used by the web UI (instantiating a
    # factory if needed), the action/observation classes, metadata, the episode
    # state, connected /ws/ui sockets, and a one-thread executor for sync env
    # calls. Keeps at most 1000 action log entries.
    """Manages the web interface for an environment."""

    MAX_ACTION_LOGS = 1000

    def __init__(
        self,
        env: Environment,
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        metadata: Optional[EnvironmentMetadata] = None,
    ):
        # Anything that is not an Environment instance is a factory, instantiate it
        # [FUNCTION DOC]
        # Instantiates env if it is not an Environment instance, stores classes and
        # metadata (default built from env's class name), and initializes an empty
        # episode state, client list and executor.
        # KNOWN ISSUE: the default metadata uses the class of the env argument, which is
        # "type" when a factory class was passed rather than the environment class.
        self.env = env if isinstance(env, Environment) else env()
        self.action_cls = action_cls
        self.observation_cls = observation_cls
        self.metadata = metadata or EnvironmentMetadata(
            name=env.__class__.__name__,
            description=f"{env.__class__.__name__} environment",
        )
        self.episode_state = EpisodeState(
            episode_id=None,
            step_count=0,
            current_observation=None,
            action_logs=[],
        )
        self.connected_clients: List[WebSocket] = []
        # Thread pool for running sync code (e.g., Playwright sync API) in async context
        self._executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _get_valid_kwargs(
        sig: inspect.Signature,
        kwargs: Dict[str, Any],
        skip_params: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Returns kwargs accepted by sig (all if it has **kwargs), minus
        # skip_params.
        """Filter kwargs to only those accepted by the target function."""
        skip_params = skip_params or set()
        valid_kwargs: Dict[str, Any] = {}
        has_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

        for key, value in kwargs.items():
            if key in skip_params:
                continue
            if key in sig.parameters or has_var_kwargs:
                valid_kwargs[key] = value

        return valid_kwargs

    async def _run_sync_in_thread_pool(self, func, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Runs func(*args, **kwargs) in the manager's single-thread executor and
        # returns the result.
        """Run a synchronous function in the thread pool executor.

        This is needed for environments using sync libraries (e.g., Playwright sync API)
        that cannot be called directly from an async context.
        """
        loop = asyncio.get_event_loop()
        # Use default arguments to capture values at lambda definition time
        # to avoid closure issues with late binding
        return await loop.run_in_executor(
            self._executor, lambda f=func, a=args, kw=kwargs: f(*a, **kw)
        )

    async def connect_websocket(self, websocket: WebSocket):
        # [FUNCTION DOC]
        # Accepts the socket, registers it and broadcasts the current state.
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.connected_clients.append(websocket)

        # Send current state to the new client
        await self._send_state_update()

    async def disconnect_websocket(self, websocket: WebSocket):
        # [FUNCTION DOC]
        # Unregisters the socket if present.
        """Disconnect a WebSocket client."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)

    async def _send_state_update(self):
        # [FUNCTION DOC]
        # HELPER: Sends {"type": "state_update", "episode_state": ...} as JSON to every
        # connected client, dropping clients whose send fails. No-op without clients.
        """Send current state to all connected clients."""
        if not self.connected_clients:
            return

        state_data = {
            "type": "state_update",
            "episode_state": self.episode_state.model_dump(),
        }

        # Send to all connected clients
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(state_data))
            except Exception:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)

    async def reset_environment(
        self, reset_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Resets the env with accepted reset kwargs (async override if present, else
        # sync in the executor), reads its state, serializes the observation, resets
        # the episode state (episode id from state, step 0, current observation, logs
        # cleared, is_reset True), broadcasts it, and returns the serialized
        # observation dict.
        """Reset the environment and update state."""
        reset_kwargs = reset_kwargs or {}

        is_async = overrides_method(self.env.reset_async, Environment.reset_async)
        sig = inspect.signature(self.env.reset_async if is_async else self.env.reset)
        valid_kwargs = self._get_valid_kwargs(sig, reset_kwargs)

        if is_async:
            observation = await self.env.reset_async(**valid_kwargs)
        else:
            # Run sync reset in thread pool to avoid blocking event loop
            # and to support environments using sync libraries (e.g., Playwright)
            observation = await self._run_sync_in_thread_pool(
                self.env.reset, **valid_kwargs
            )
        state: State = self.env.state

        # Serialize observation once using shared utility
        serialized = serialize_observation(observation)

        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = 0
        self.episode_state.current_observation = serialized["observation"]
        self.episode_state.action_logs = []
        self.episode_state.is_reset = True

        # Send state update
        await self._send_state_update()

        return serialized

    async def step_environment(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Deserializes action_data with web preprocessing (ValidationError on invalid
        # input), runs the env's sync step in the executor, reads state, serializes the
        # observation, appends an ActionLog (trimmed to the last 1000), updates the
        # episode state (is_reset False), broadcasts it, and returns the serialized
        # observation dict.
        # KNOWN ISSUE: unlike reset_environment, an env's async step override is never
        # used here; the sync step always runs in the executor.
        """Execute a step in the environment and update state."""
        # Deserialize action with preprocessing for web interface special cases
        action: Action = deserialize_action_with_preprocessing(
            action_data, self.action_cls
        )

        # Run sync step in thread pool to avoid blocking event loop
        # and to support environments using sync libraries (e.g., Playwright)
        observation: Observation = await self._run_sync_in_thread_pool(
            self.env.step, action
        )
        state: State = self.env.state

        # Serialize observation once using shared utility
        serialized = serialize_observation(observation)

        # Create action log
        action_log = ActionLog(
            timestamp=datetime.now().isoformat(),
            action=action.model_dump(exclude={"metadata"}),
            observation=serialized["observation"],
            reward=observation.reward,
            done=observation.done,
            step_count=state.step_count,
        )

        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = state.step_count
        self.episode_state.current_observation = serialized["observation"]
        self.episode_state.action_logs.append(action_log)
        if len(self.episode_state.action_logs) > self.MAX_ACTION_LOGS:
            self.episode_state.action_logs = self.episode_state.action_logs[
                -self.MAX_ACTION_LOGS :
            ]
        self.episode_state.is_reset = False

        # Send state update
        await self._send_state_update()

        return serialized

    def get_state(self) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Returns the env's current state as a dict.
        """Get current environment state."""
        state: State = self.env.state
        return state.model_dump()


def create_web_interface_app(
    env: Environment,
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[Any] = None,
    gradio_builder: Optional[Callable[..., Any]] = None,
    custom_tab_name: str = "Custom",
    custom_tab_primary: bool = False,
    show_default_tab: bool = True,
    title_override: Optional[str] = None,
    state_cls: Type[State] = State,
    *,
    mode: Optional[Any] = None,
    reset_observation_cls: Optional[Type[Observation]] = None,
) -> FastAPI:
    # [FUNCTION DOC]
    # Creates the base app via create_fastapi_app (same env, classes, concurrency,
    # name, state class, mode), loads metadata, creates the WebInterfaceManager
    # (instantiating the env), registers the web routes, computes legacy action
    # fields, chat detection and quick start Markdown, builds the Gradio blocks
    # (title_override or the default display title), mounts them at /web with the
    # OpenEnv theme and CSS, and returns the app. Errors from server construction,
    # env instantiation or a gradio_builder propagate.
    """
    Create a FastAPI application with web interface for the given environment.

    Args:
        env: The Environment instance to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass returned by step
        reset_observation_cls: The reset observation model published in /schema.
            Defaults to observation_cls.
        env_name: Optional environment name for README loading
        max_concurrent_envs: Maximum concurrent WebSocket sessions
        concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings
        gradio_builder: Optional callable (web_manager, action_fields, metadata,
            is_chat_env, title, quick_start_md) -> gr.Blocks to use instead of the
            default Gradio UI. Lets envs replace or customize the /web interface.
        custom_tab_name: Label shown on the env-specific tab when ``gradio_builder``
            is provided. Defaults to ``"Custom"`` for backwards compatibility; envs
            that ship a rich custom UI should pass a descriptive name
            (e.g. ``"REPL"``). Ignored when ``show_default_tab=False`` (no tab chrome
            is rendered).
        custom_tab_primary: When True, the env-specific tab is rendered first and
            selected by default; the auto-generated Playground becomes secondary.
            Use this for envs whose custom tab is the real interaction surface
            (so visitors don't land on a less informative schema form). Ignored
            when ``show_default_tab=False``.
        show_default_tab: When False, the auto-generated Playground tab is not
            rendered and the env's ``gradio_builder`` output is mounted directly
            (single-view UI, no tab chrome). Only meaningful when
            ``gradio_builder`` is provided.
        title_override: If set, used verbatim as the Gradio app/browser-tab
            title instead of the default ``"OpenEnv Agentic Environment: {name}"``.
        state_cls: The State subclass this environment reports. Used for the /state
            response model and the state entry of /schema. Defaults to State.
        mode: Server mode (``ServerMode`` or string). When ``None``, resolved
            from the ``OPENENV_MODE`` environment variable, defaulting to
            simulation.

    Returns:
        FastAPI application instance with web interface
    """
    from .http_server import create_fastapi_app

    # Create the base environment app
    app = create_fastapi_app(
        env,
        action_cls,
        observation_cls,
        max_concurrent_envs,
        concurrency_config,
        env_name=env_name,
        state_cls=state_cls,
        reset_observation_cls=reset_observation_cls,
        mode=mode,
    )

    # Load environment metadata
    metadata = load_environment_metadata(env, env_name)

    # Create web interface manager
    web_manager = WebInterfaceManager(env, action_cls, observation_cls, metadata)

    # Web API routes first (so they take precedence over Gradio mount at /web)
    @app.get("/", include_in_schema=False)
    async def web_root():
        # [FUNCTION DOC]
        # GET /: redirects to /web/.
        """Redirect the app root to the Gradio interface."""
        return RedirectResponse(url="/web/")

    @app.get("/web", include_in_schema=False)
    async def web_root_no_slash():
        # [FUNCTION DOC]
        # GET /web: redirects to /web/.
        """Redirect /web to /web/ for mounted Gradio deployments behind proxies."""
        return RedirectResponse(url="/web/")

    @app.get("/web/metadata")
    async def web_metadata():
        # [FUNCTION DOC]
        # GET /web/metadata: returns the metadata as a dict.
        """Get environment metadata."""
        return web_manager.metadata.model_dump()

    @app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        # [FUNCTION DOC]
        # WebSocket /ws/ui: registers the client (sending current state) and keeps
        # reading frames until disconnect, then unregisters it.
        """WebSocket endpoint for web UI real-time updates.

        Uses `/ws/ui` to avoid conflict with `/ws` in `http_server.py`,
        which is used for concurrent environment sessions.
        """
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset(request: Optional[Dict[str, Any]] = Body(default=None)):
        # [FUNCTION DOC]
        # POST /web/reset: resets with the optional JSON body as reset kwargs and
        # returns the serialized observation.
        """Reset endpoint for web interface."""
        return await web_manager.reset_environment(request)

    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        # [FUNCTION DOC]
        # POST /web/step: if the body has "message", converts it with the env's
        # message_to_action when available (tokens as a list, or the action fields
        # without metadata) or sends {"message": ...}; otherwise uses body["action"]
        # (default {}). Steps and returns the serialized observation. Errors propagate
        # as HTTP 500.
        """Step endpoint for web interface."""
        # Check if this is a message-based request (chat environment)
        if "message" in request:
            message = request["message"]
            if hasattr(web_manager.env, "message_to_action"):
                action = web_manager.env.message_to_action(message)
                if hasattr(action, "tokens"):
                    action_data = {"tokens": action.tokens.tolist()}
                else:
                    action_data = action.model_dump(exclude={"metadata"})
            else:
                action_data = {"message": message}
        else:
            action_data = request.get("action", {})

        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        # [FUNCTION DOC]
        # GET /web/state: returns the env state; a RuntimeError becomes HTTP 409.
        """State endpoint for web interface."""
        try:
            return web_manager.get_state()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    action_fields = _extract_action_fields(action_cls)
    is_chat_env = _is_chat_env(action_cls)
    quick_start_md = get_quick_start_markdown(metadata, action_cls, observation_cls)

    display_title = title_override or get_gradio_display_title(metadata)
    gradio_blocks = _build_gradio_blocks(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        quick_start_md,
        display_title=display_title,
        gradio_builder=gradio_builder,
        custom_tab_name=custom_tab_name,
        custom_tab_primary=custom_tab_primary,
        show_default_tab=show_default_tab,
    )
    app = gr.mount_gradio_app(
        app,
        gradio_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )

    return app


def _build_gradio_blocks(
    web_manager: WebInterfaceManager,
    action_fields: List[Dict[str, Any]],
    metadata: EnvironmentMetadata,
    is_chat_env: bool,
    quick_start_md: str,
    *,
    display_title: str,
    gradio_builder: Optional[Callable[..., Any]],
    custom_tab_name: str,
    custom_tab_primary: bool,
    show_default_tab: bool,
) -> gr.Blocks:
    # [FUNCTION DOC]
    # HELPER: Builds the default playground via build_gradio_app and sets its
    # title. Without a gradio_builder returns it. Otherwise calls the builder
    # (web_manager, action_fields, metadata, is_chat_env, display_title,
    # quick_start_md), requiring a gr.Blocks (TypeError otherwise); returns the
    # custom blocks alone when show_default_tab is False, else a TabbedInterface
    # with "Playground" and the custom tab in the order chosen by
    # custom_tab_primary, titled display_title.
    """Build the Gradio block tree mounted at `/web`."""
    default_blocks = build_gradio_app(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        title=metadata.name,
        quick_start_md=quick_start_md,
    )
    default_blocks.title = display_title

    if gradio_builder is None:
        return default_blocks

    custom_blocks = gradio_builder(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        display_title,
        quick_start_md,
    )
    if not isinstance(custom_blocks, gr.Blocks):
        raise TypeError(
            f"gradio_builder must return a gr.Blocks instance, "
            f"got {type(custom_blocks).__name__}"
        )

    if not show_default_tab:
        # No TabbedInterface wrapper to carry the app title.
        custom_blocks.title = display_title
        return custom_blocks

    if custom_tab_primary:
        tab_blocks = [custom_blocks, default_blocks]
        tab_labels = [custom_tab_name, "Playground"]
    else:
        tab_blocks = [default_blocks, custom_blocks]
        tab_labels = ["Playground", custom_tab_name]

    # The playground has its own header, so the title only names the browser tab.
    tabbed = gr.TabbedInterface(tab_blocks, tab_names=tab_labels)
    tabbed.title = display_title
    return tabbed


def _is_chat_env(action_cls: Type[Action]) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if the action class has a "tokens" field whose
    # annotation has a __name__ and whose string form contains "Tensor".
    """Return True if the action class is a chat-style env (tokens field)."""
    if hasattr(action_cls, "model_fields"):
        for field_name, field_info in action_cls.model_fields.items():
            if (
                field_name == "tokens"
                and hasattr(field_info.annotation, "__name__")
                and "Tensor" in str(field_info.annotation)
            ):
                return True
    return False


def _extract_action_fields(action_cls: Type[Action]) -> List[Dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: From the action's JSON schema (empty list on AttributeError),
    # returns one dict per property except metadata with name, input type,
    # required flag, description, default, enum choices, numeric and length bounds,
    # pattern, placeholder and help text.
    """Extract enhanced field metadata from Action class for form generation."""
    # Use Pydantic's JSON schema generation for robust metadata extraction
    try:
        schema = action_cls.model_json_schema()
    except AttributeError:
        # Fallback for non-Pydantic v2 models or if something goes wrong
        return []

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    action_fields = []

    for field_name, field_info in properties.items():
        if field_name == "metadata":
            continue

        # JSON schema "type" can be a string or list/undefined
        # Determine our internal input type
        input_type = _determine_input_type_from_schema(field_info, field_name)

        is_required = field_name in required_fields

        action_fields.append(
            {
                "name": field_name,
                "type": input_type,
                "required": is_required,
                "description": field_info.get("description", ""),
                "default_value": field_info.get("default"),
                "choices": field_info.get("enum"),
                "min_value": field_info.get("minimum"),
                "max_value": field_info.get("maximum"),
                "min_length": field_info.get("minLength"),
                "max_length": field_info.get("maxLength"),
                "pattern": field_info.get("pattern"),
                "placeholder": _generate_placeholder(field_name, field_info),
                "help_text": _generate_help_text(field_name, field_info),
            }
        )

    return action_fields


def _determine_input_type_from_schema(
    field_info: Dict[str, Any], field_name: str
) -> str:
    # [FUNCTION DOC]
    # HELPER: Maps a field to a UI input type: "tensor" if the name contains
    # "tokens", "select" for enums, "checkbox" for booleans, "number" for
    # integer/number, "textarea" for strings with maxLength > 100 or names
    # containing message/code, otherwise "text".
    """Determine input type from JSON schema for form generation (Gradio UI)."""
    schema_type = field_info.get("type")

    # Check for specific tensor field convention
    if "tokens" in field_name.lower():
        return "tensor"

    if "enum" in field_info:
        return "select"

    if schema_type == "boolean":
        return "checkbox"

    if schema_type == "integer" or schema_type == "number":
        return "number"

    if schema_type == "string":
        # Check if it should be a textarea
        if (
            field_info.get("maxLength", 0) > 100
            or "message" in field_name.lower()
            or "code" in field_name.lower()
        ):
            return "textarea"
        return "text"

    # Default fallback
    return "text"


def _generate_placeholder(field_name: str, field_info: Dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns placeholder text based on the field name (code and tokens get
    # specific hints; otherwise "Enter <name with spaces>...").
    """Generate placeholder text."""
    if "message" in field_name.lower():
        return f"Enter {field_name.replace('_', ' ')}..."
    elif "code" in field_name.lower():
        return "Enter Python code here..."
    elif "tokens" in field_name.lower():
        return "Enter comma-separated token IDs (e.g., 1,2,3,4,5)"
    else:
        return f"Enter {field_name.replace('_', ' ')}..."


def _generate_help_text(field_name: str, field_info: Dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the schema description if present, else a canned hint for
    # names containing action_id, game_name, tokens, code or message, else "".
    """Generate help text."""
    description = field_info.get("description", "")
    if description:
        return description

    if "action_id" in field_name.lower():
        return "The action ID to execute in environment"
    elif "game_name" in field_name.lower():
        return "Name of game or environment"
    elif "tokens" in field_name.lower():
        return "Token IDs as a comma-separated list of integers"
    elif "code" in field_name.lower():
        return "Python code to execute in environment"
    elif "message" in field_name.lower():
        return "Text message to send"

    return ""
