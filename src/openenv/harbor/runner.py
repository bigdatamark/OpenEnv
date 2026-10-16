# [MODULE DOC]
# Serverless batch driver behind `openenv harbor rollout`: runs startup checks, boots an
# in-process capture proxy and a public forwarder, runs selected tasks of one dataset
# sequentially through Harbor, prints per-rollout lines and a summary, and tears down.
# Re-exports CaptureServer.
"""Drive rollouts without a server: boot capture, run tasks, tear down.

`openenv harbor rollout` uses this. It exists so the whole path (LLM, capture proxy, forwarding,
seam, Harbor trial, sandbox, verifier, reconciliation) can be exercised with no env server in
the way. When something breaks, that halves the search space immediately: if this works and `serve` does
not, the problem is the serving layer and nothing below it.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from openenv.core.harness.capture import CaptureServer

from .models import HarborRolloutResult
from .rollout import run_rollout
from .tasks import resolve_task_dirs


# Re-exported, not defined here. `CaptureServer` moved to `openenv.core.harness.capture` when a
# second environment needed an in-process proxy: it never touched a Harbor type, so living under
# `harbor/` forced unrelated callers to import Harbor for a class that uses none of it.
__all__ = ["CaptureServer", "run_batch"]


async def run_batch(
    *,
    llm_url: str,
    dataset: str,
    task_indices: list[int],
    harness: str = "opencode",
    sandbox: str = "e2b",
    model: str | None = None,
    port: int = 8100,
    expose: str = "gradio",
    trials_dir: Path | None = None,
    reward_key: str = "",
    keep_sandbox: bool = False,
    force_build: bool = False,
    env_file: str | None = None,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    provider: str = "openai",
    admin_key: str | None = None,
) -> list[HarborRolloutResult]:
    # [FUNCTION DOC]
    # Async. Calls prepare() (LLM must be reachable; prints capabilities), takes model and
    # capture level (default "tokens") from it, resolves api_key, and raises RuntimeError if
    # the sandbox backend is not available. Resolves dataset task dirs, creates trials_dir
    # (default /tmp/openenv-harbor-trials), resolves admin_key (argument, env, or random),
    # starts a CaptureServer on port and a forwarder of type expose (stopping capture if the
    # forwarder fails). For each index: out-of-range indices are skipped with a message;
    # otherwise awaits run_rollout with the capture registry/public URL and prints a summary
    # and up to 3 findings. Always stops forwarder then capture. Prints the batch report and
    # returns results in order (skipped indices omitted).
    # KNOWN ISSUE: docstring says "One per index", but out-of-range indices produce no result.
    # Side effects: network listeners/tunnels, filesystem (trials dir), stdout, sandboxes.
    """Run `task_indices` from `dataset` and print a per-rollout report.

    Args:
        llm_url (`str`):
            OpenAI-spec inference endpoint.
        dataset (`str`):
            Dataset spec (HF repo id, local dir, or Harbor `name@version`).
        task_indices (`list[int]`):
            Which tasks to run, by index into the resolved dataset.
        harness (`str`, *optional*, defaults to `"opencode"`):
            Seam name or `module:Class`.
        sandbox (`str`, *optional*, defaults to `"e2b"`):
            Harbor environment type.
        expose (`str`, *optional*, defaults to `"gradio"`):
            How the sandbox reaches the capture proxy: `gradio`, `cloudflare` or `direct`.
        admin_key (`str`, *optional*):
            Key the capture proxy's session-management routes require. Defaults to
            `$OPENENV_CAPTURE_ADMIN_KEY`, else a random one minted for this batch. The proxy is
            published on a public URL, so these routes are never left open.

    Returns:
        `list[HarborRolloutResult]`: One per index, in order.
    """
    # Imported here, not at module scope: a hosted deployment mounts the capture proxy on its own
    # app and never forwards, so `forwarding` is not shipped there.
    from openenv.core.harness.capture.forwarding import make_forwarder

    from .startup import prepare

    caps = prepare(
        llm_url=llm_url,
        model=model,
        datasets=[dataset],
        env_file=env_file,
        require_llm=True,
        quiet=False,
        api_key=api_key,
        auth_header=auth_header,
        provider=provider,
    )
    model = caps.llm.get("model") or model or ""
    capture_level = caps.llm.get("capture_level") or "tokens"
    # `prepare` already read the dotenv, so a key that lives only in --env-file is visible now.
    api_key = api_key or os.environ.get("OPENENV_LLM_API_KEY") or None

    if sandbox not in caps.available_sandboxes:
        detail = next(
            (s.detail for s in caps.sandboxes if s.name == sandbox), "not checked"
        )
        raise RuntimeError(f"sandbox {sandbox!r} is not usable here: {detail}")

    task_dirs = resolve_task_dirs(dataset)
    trials_dir = trials_dir or Path("/tmp/openenv-harbor-trials")
    trials_dir.mkdir(parents=True, exist_ok=True)

    # The forwarder below puts the proxy on a public URL, and `_admin_ok` waves every caller through
    # when no key is set — so an unset key here is an open control plane: anyone with the URL could
    # list rollouts, read their tokens, delete them, or mint a session key the proxy then honours.
    # Rollouts never go through those routes (they use the in-process registry), so the key is only
    # for an operator, and a random one costs nothing. Same resolution as `HarborService`.
    admin_key = (
        admin_key
        or os.environ.get("OPENENV_CAPTURE_ADMIN_KEY")
        or secrets.token_urlsafe(32)
    )

    capture = CaptureServer(
        llm_url=llm_url,
        model=model,
        port=port,
        api_key=api_key,
        auth_header=auth_header,
        provider=provider,
        capture_level=capture_level,
        admin_key=admin_key,
    )
    capture.start()
    # The capture proxy is already listening on a bound port in a background thread, so an exception
    # between here and the `try` below would leave that thread up and the port held — and the next
    # attempt would then die on "port already in use" rather than on the real error. Forwarder setup
    # is the risky part (cloudflared spawns a binary, gradio opens a tunnel), so it goes under its own
    # guard. `HarborService.start` guards the same pair the same way.
    try:
        forwarder = make_forwarder(expose)
        public_url = forwarder.start(port)
    except BaseException:
        capture.stop()
        raise
    print(f"\ncapture  :{port} -> {public_url}  ({forwarder.name})")
    print(
        "         session routes are gated; set OPENENV_CAPTURE_ADMIN_KEY to call them yourself"
    )
    print(f"trials   {trials_dir}\n")

    results: list[HarborRolloutResult] = []
    try:
        for i in task_indices:
            if not 0 <= i < len(task_dirs):
                print(f"  skip index {i}: out of range (dataset has {len(task_dirs)})")
                continue
            task_dir = task_dirs[i]
            print(f"[{harness} / {sandbox}] task {i}: {task_dir.name} ...", flush=True)
            result = await run_rollout(
                task_dir=task_dir,
                harness=harness,
                sandbox=sandbox,
                registry=capture.registry,
                intercept_url=public_url,
                model=model,
                trials_dir=trials_dir,
                dataset=dataset,
                reward_key=reward_key,
                keep_sandbox=keep_sandbox,
                force_build=force_build,
                capture_level=capture_level,
                inference=capture.inference,
            )
            results.append(result)
            print("   " + _summarise(result))
            for finding in result.findings[:3]:
                print(f"      {finding[:150]}")
    finally:
        # `capture.stop()` releases the port, so it must run even if the forwarder's own teardown
        # throws — otherwise a failing tunnel shutdown strands the proxy for the rest of the process.
        try:
            forwarder.stop()
        finally:
            capture.stop()

    print("\n" + _report(results))
    return results


def _summarise(r: HarborRolloutResult) -> str:
    # [FUNCTION DOC]
    # HELPER: one-line status for a result: ok/FAILED(type), reward (2dp or None), turns,
    # roots, multi-turn vs per-turn, trainable tokens (train) or EVAL/<capture level>, ATIF
    # status, wall seconds, plus an indented truncated error line when present.
    reward = "None" if r.reward is None else f"{r.reward:.2f}"
    mode = "multi-turn" if r.multi_turn else "per-turn"
    status = "ok" if r.ok else f"FAILED ({r.exception_type or 'error'})"
    # An eval rollout has no trainable tokens by construction, so printing `tokens=0` next to a
    # healthy reward invites the reading that capture broke. Name the rollout type instead.
    detail = (
        f"tokens={r.n_trainable_tokens:<6}"
        if r.rollout_type == "train"
        else f"EVAL/{r.capture_level:<7}"
    )
    return (
        f"{status:<26} reward={reward:<6} turns={r.n_turns:<3} roots={r.n_roots:<3} "
        f"{mode:<11} {detail} atif={r.atif:<9} {r.wall_s:.0f}s"
        + (f"\n      {r.error[:180]}" if r.error else "")
    )


def _report(results: list[HarborRolloutResult]) -> str:
    # [FUNCTION DOC]
    # HELPER: multi-line batch summary: usable count, solved (reward > 0) of graded, ungraded
    # count, trainable token/turn totals or an EVAL note when all are eval, and a note that
    # capture and reward are independent. Returns "no rollouts ran" for an empty list.
    if not results:
        return "no rollouts ran"
    ok = sum(1 for r in results if r.ok)
    graded = [r for r in results if r.reward is not None]
    solved = sum(1 for r in graded if r.reward and r.reward > 0)
    lines = [
        "=" * 78,
        f"capture   {ok}/{len(results)} usable",
        f"solved    {solved}/{len(graded)} graded"
        + (
            f"   ({len(results) - len(graded)} ungraded — the verifier never ran)"
            if len(graded) != len(results)
            else ""
        ),
    ]
    if all(r.rollout_type == "eval" for r in results):
        lines.append(
            f"tokens    none — these are EVAL rollouts ({results[0].capture_level}); "
            f"{sum(r.n_turns for r in results)} turns captured as trace only"
        )
    else:
        lines.append(
            f"tokens    {sum(r.n_trainable_tokens for r in results)} trainable across "
            f"{sum(r.n_turns for r in results)} turns"
        )
    # Capture quality and task success are independent, and conflating them has burned us before:
    # a perfectly captured rollout can score 0 because the model was wrong.
    lines.append("NOTE: capture and reward are independent measurements.")
    return "\n".join(lines)
