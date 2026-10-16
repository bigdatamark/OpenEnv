# [MODULE DOC]
# Implements "openenv collect": runs a rollout collection job against a deployed environment
# server with the harness CollectRunner, using a scripted teacher (first legal action) or an LLM
# teacher (hosted OpenAI/Anthropic or a self-hosted OpenAI-compatible endpoint), writes
# results.jsonl and metadata.json, and can push the output to a Hub dataset. Env specs are
# "openspiel:<game>" or "reasoning_gym:<dataset>"; their client and session factory classes are
# imported optionally at module import (None if unavailable).
# KNOWN ISSUE: the module docstring says only openspiel_env is supported, but reasoning_gym is too.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a rollout collection job against a deployed OpenEnv environment.

Currently ships with a built-in registry for ``openspiel_env``. Adding
other envs is a matter of writing a thin session factory in the env
package and registering it here.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from openenv.core.harness import HarnessRunLimits, MCPHarnessAdapter, ModelStepResult
from openenv.core.harness.collect import (
    build_model_step,
    CollectRunner,
    push_to_hf_hub,
    RolloutSerializer,
)
from openenv.core.llm_client import create_llm_client, LLMClient, LLMResponse, ToolCall

from .._cli_utils import console

# Imported eagerly so tests can monkeypatch these names on this module.
# The real imports only happen when the selected --env is openspiel/reasoning_gym.
try:
    from openspiel_env.client import OpenSpielEnv  # type: ignore[import-not-found]
    from openspiel_env.harness import (  # type: ignore[import-not-found]
        OpenSpielSessionFactory,
    )
except ImportError:  # pragma: no cover - openspiel env optional at import time
    OpenSpielEnv = None  # type: ignore[assignment]
    OpenSpielSessionFactory = None  # type: ignore[assignment]

try:
    from reasoning_gym_env.client import (  # type: ignore[import-not-found]
        ReasoningGymEnv,
    )
    from reasoning_gym_env.harness import (  # type: ignore[import-not-found]
        ReasoningGymSessionFactory,
    )
except ImportError:  # pragma: no cover - reasoning_gym env optional at import time
    ReasoningGymEnv = None  # type: ignore[assignment]
    ReasoningGymSessionFactory = None  # type: ignore[assignment]

app = typer.Typer(help="Collect rollouts from a deployed OpenEnv environment.")

_PROVIDER_API_KEY_ENVS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY", "API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY", "API_KEY"],
}

_SYSTEM_PROMPT = """You are playing a game through a tool-calling interface.
For each turn, inspect the observation and call the provided tool with a
valid argument. Only call one tool per turn."""


def _uses_llm_teacher(provider: str, llm_endpoint: str | None) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if an LLM endpoint is given or the provider is not "scripted".
    """Return whether this collect job should use an LLM-backed teacher."""
    return llm_endpoint is not None or provider != "scripted"


def _teacher_provider_label(provider: str, llm_endpoint: str | None) -> str:
    # [FUNCTION DOC]
    # HELPER: returns "openai-compatible" when an LLM endpoint is given, else the provider name.
    """Normalize the provider label shown in metadata and CLI output."""
    if llm_endpoint is not None:
        return "openai-compatible"
    return provider


def _resolve_api_key(provider: str) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the first non-empty env var among the provider's key names (provider-specific,
    # then API_KEY). Raises: typer.BadParameter if none is set or the provider is unknown.
    for name in _PROVIDER_API_KEY_ENVS.get(provider, []):
        value = os.getenv(name)
        if value:
            return value
    raise typer.BadParameter(
        f"No API key found for provider={provider!r}. "
        f"Set one of: {_PROVIDER_API_KEY_ENVS.get(provider, [])}",
        param_hint="--provider",
    )


def _extract_legal_actions(messages: list[dict[str, Any]]) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: scans messages newest first for legal actions, either a JSON-object content with a
    # legal_actions list or text "Legal actions: [..]". Returns a list of ints (empty if not found or
    # the textual list is empty). Raises: ValueError for non-integer entries.
    for message in reversed(messages):
        content = message.get("content") or ""
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict) and "legal_actions" in payload:
            legal = payload["legal_actions"]
            if isinstance(legal, list):
                return [int(a) for a in legal]
        match = re.search(r"Legal actions:\s*\[([^\]]*)\]", content)
        if match:
            raw = match.group(1).strip()
            if not raw:
                return []
            return [int(x) for x in raw.split(",")]
    return []


def _build_scripted_model_step():
    # [FUNCTION DOC]
    # HELPER: returns a scripted model-step function for smoke tests without API keys.
    """Default teacher: pick the first legal action from the latest observation.

    Matches the scripted teacher in ``examples/ttt_collect_demo.py``. Lets
    users smoke-test the CLI without any API key configured.
    """

    def model_step(messages, tools, sampling):
        # [FUNCTION DOC]
        # Picks the first legal action from the messages (0 if none) and returns a ModelStepResult calling
        # the first offered tool (or "play_move") with that action_id. Sampling is ignored.
        del sampling
        legal = _extract_legal_actions(messages)
        action_id = legal[0] if legal else 0
        tool_name = tools[0].name if tools else "play_move"
        return ModelStepResult(
            response=LLMResponse(
                content=f"Playing action_id={action_id} (first legal).",
                tool_calls=[
                    ToolCall(
                        id=f"scripted-{action_id}",
                        name=tool_name,
                        args={"action_id": action_id},
                    )
                ],
            ),
        )

    return model_step


def _build_llm_model_step(
    provider: str,
    model: str,
    *,
    llm_endpoint: str | None,
    llm_port: int | None,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
):
    # [FUNCTION DOC]
    # HELPER: builds an LLM model-step function. With llm_endpoint, creates an OpenAIClient for that
    # endpoint/port (API key from OPENAI_API_KEY or a placeholder) and prints its base URL; otherwise
    # creates a hosted client for the provider with its API key. Uses the given system prompt or the
    # default one. Raises: typer.BadParameter for an invalid endpoint or missing API key.
    effective_system_prompt = system_prompt or _SYSTEM_PROMPT

    if llm_endpoint:
        # Self-hosted OpenAI-compatible endpoint (vLLM, TGI, Ollama, ...).
        from openenv.core.llm_client import OpenAIClient

        try:
            client: LLMClient = OpenAIClient(
                endpoint=llm_endpoint,
                port=llm_port,
                model=model,
                api_key=os.getenv("OPENAI_API_KEY") or "not-needed",
                system_prompt=effective_system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--llm-endpoint") from exc
        console.print(f"[cyan]LLM endpoint:[/cyan] {client.base_url}")
    else:
        client = create_llm_client(
            provider=provider,
            model=model,
            api_key=_resolve_api_key(provider),
            system_prompt=effective_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return build_model_step(client, system_prompt=effective_system_prompt)


def _build_session_factory(
    env_spec: str,
    base_url: str,
    dataset_config: dict[str, Any] | None = None,
):
    # [FUNCTION DOC]
    # HELPER: splits the env spec on ":" and returns an OpenSpielSessionFactory (game variant) or
    # ReasoningGymSessionFactory (dataset variant with optional config), each creating a client for
    # base_url. Raises: typer.BadParameter for a missing variant, an unavailable env package or an
    # unknown family.
    """Dispatch an env spec to a session factory.

    Supported specs:
    - ``openspiel:<game>``      e.g. ``openspiel:tic_tac_toe``
    - ``reasoning_gym:<dataset>`` e.g. ``reasoning_gym:chain_sum``
    """
    env_name, _, variant = env_spec.partition(":")
    if not variant:
        raise typer.BadParameter(
            f"Missing variant. Use e.g. {env_name}:chain_sum",
            param_hint="ENV",
        )

    if env_name == "openspiel":
        if OpenSpielEnv is None or OpenSpielSessionFactory is None:
            raise typer.BadParameter(
                "openspiel_env is not importable. Ensure envs/ is on PYTHONPATH.",
                param_hint="ENV",
            )
        return OpenSpielSessionFactory(
            lambda: OpenSpielEnv(base_url=base_url),
            game_name=variant,
        )

    if env_name == "reasoning_gym":
        if ReasoningGymEnv is None or ReasoningGymSessionFactory is None:
            raise typer.BadParameter(
                "reasoning_gym_env is not importable. Ensure envs/ is on PYTHONPATH.",
                param_hint="ENV",
            )
        return ReasoningGymSessionFactory(
            lambda: ReasoningGymEnv(base_url=base_url),
            dataset_name=variant,
            dataset_config=dataset_config,
        )

    raise typer.BadParameter(
        f"Unknown env {env_spec!r}. Supported: openspiel:<game>, reasoning_gym:<dataset>",
        param_hint="ENV",
    )


@app.command()
def collect(
    env: Annotated[
        str,
        typer.Argument(
            help="Env spec in 'family:variant' form (e.g. openspiel:tic_tac_toe).",
        ),
    ],
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Env server URL (local Docker or Hugging Face Space).",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write results.jsonl + metadata.json.",
        ),
    ],
    num_episodes: Annotated[
        int,
        typer.Option("--num-episodes", "-n", help="Number of episodes to collect."),
    ] = 10,
    max_turns: Annotated[
        int, typer.Option("--max-turns", help="Max tool/model turns per episode.")
    ] = 9,
    episode_id_prefix: Annotated[
        str,
        typer.Option("--episode-id-prefix", help="Prefix for serialized episode ids."),
    ] = "ep",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Skip episodes already present in results.jsonl.",
        ),
    ] = True,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help="Teacher provider: scripted | openai | anthropic.",
        ),
    ] = "scripted",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model id (required when provider != scripted).",
        ),
    ] = None,
    llm_endpoint: Annotated[
        str | None,
        typer.Option(
            "--llm-endpoint",
            help=(
                "Base URL of a self-hosted OpenAI-compatible server "
                "(vLLM/TGI/Ollama), e.g. http://localhost:8000. /v1 is appended "
                "when the URL has no path; a URL with a path is used as-is."
            ),
        ),
    ] = None,
    llm_port: Annotated[
        int | None,
        typer.Option(
            "--llm-port",
            help=(
                "Port appended to --llm-endpoint when the URL does not include one. "
                "No default: earlier releases assumed 8000."
            ),
        ),
    ] = None,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 0.2,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Max completion tokens.")
    ] = 200,
    keep_losses: Annotated[
        bool,
        typer.Option(
            "--keep-losses",
            help="Keep losing rollouts (default: filter rollouts with reward < 0).",
        ),
    ] = False,
    push_to_hub: Annotated[
        str | None,
        typer.Option(
            "--push-to-hub",
            "-H",
            help="Destination dataset repo id ('user/name'). Uploads after collect.",
        ),
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Create the Hub dataset repo as private.",
        ),
    ] = False,
    commit_message: Annotated[
        str | None,
        typer.Option(
            "--commit-message",
            help="Commit message for the Hub upload.",
        ),
    ] = None,
    dataset_config: Annotated[
        str | None,
        typer.Option(
            "--dataset-config",
            help=(
                "JSON string of dataset config for envs that support it "
                "(e.g. reasoning_gym). Example: "
                '\'{"min_terms": 2, "max_terms": 3}\''
            ),
        ),
    ] = None,
    system_prompt: Annotated[
        str | None,
        typer.Option(
            "--system-prompt",
            help="Custom system prompt for the teacher model.",
        ),
    ] = None,
) -> None:
    # [FUNCTION DOC]
    # CLI command: validates that --model is set for LLM teachers, parses --dataset-config JSON
    # (BadParameter if invalid), builds the session factory and model step, writes metadata, runs
    # CollectRunner for num_episodes with max_turns, resume and (unless --keep-losses) a filter
    # keeping rewards >= 0, prints counts and averages, and optionally pushes the output directory
    # to a Hub dataset repo. Side effects: network calls, file writes, Hub upload.
    """Collect rollouts from a deployed OpenEnv environment."""
    uses_llm_teacher = _uses_llm_teacher(provider, llm_endpoint)
    teacher_provider = _teacher_provider_label(provider, llm_endpoint)

    if uses_llm_teacher and not model:
        raise typer.BadParameter(
            "--model is required when using a hosted provider or --llm-endpoint.",
            param_hint="--model",
        )

    parsed_dataset_config: dict[str, Any] | None = None
    if dataset_config is not None:
        try:
            parsed_dataset_config = json.loads(dataset_config)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(
                f"--dataset-config must be valid JSON: {exc}",
                param_hint="--dataset-config",
            )

    factory = _build_session_factory(
        env, base_url, dataset_config=parsed_dataset_config
    )

    if uses_llm_teacher:
        model_step = _build_llm_model_step(
            provider=provider,
            model=model,  # type: ignore[arg-type]  # validated above
            llm_endpoint=llm_endpoint,
            llm_port=llm_port,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    else:
        model_step = _build_scripted_model_step()

    serializer = RolloutSerializer(output_dir)
    serializer.write_metadata(
        {
            "env": env,
            "env_base_url": base_url,
            "provider": teacher_provider,
            "model": model,
            "llm_endpoint": llm_endpoint,
            "llm_port": llm_port if llm_endpoint else None,
            "num_episodes_requested": num_episodes,
            "temperature": temperature,
            "keep_losses": keep_losses,
        }
    )

    should_keep = None if keep_losses else (lambda record: record.reward >= 0.0)

    collect_runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=serializer,
        limits=HarnessRunLimits(max_turns=max_turns),
    )

    console.print(f"[cyan]Env server:[/cyan] {base_url}")
    console.print(
        f"[cyan]Teacher:[/cyan] {teacher_provider}" + (f"/{model}" if model else "")
    )
    console.print(f"[cyan]Output:[/cyan] {output_dir}")

    result = collect_runner.run(
        model_step=model_step,
        num_episodes=num_episodes,
        episode_id_prefix=episode_id_prefix,
        resume=resume,
        should_keep=should_keep,
    )

    console.print(
        f"[green]Collected[/green]={result.num_collected} "
        f"[yellow]skipped[/yellow]={result.num_skipped} "
        f"[red]dropped[/red]={result.num_dropped} "
        f"[red]failed[/red]={result.num_failed}"
    )
    console.print(
        f"avg_reward={result.avg_reward:.3f} success_rate={result.success_rate:.0%}"
    )

    if push_to_hub:
        console.print(f"[cyan]Pushing to Hub:[/cyan] {push_to_hub}")
        url = push_to_hf_hub(
            output_dir=output_dir,
            repo_id=push_to_hub,
            private=private,
            commit_message=commit_message,
        )
        console.print(f"[green]Dataset at:[/green] {url}")


def _invoke_via_entry_point() -> None:  # pragma: no cover - wired in __main__
    # [FUNCTION DOC]
    # HELPER: runs this module's Typer app and exits with its return value.
    sys.exit(app())
