# [MODULE DOC]
# Task selection utilities for AWM benchmarking/training: identifies usable tasks (having a code verifier and not
# passing trivially with no actions) and builds a deterministic shuffled train/val split.
"""Task selection for honest AWM benchmarking and training.

Some AWM tasks pass with no actions (a no-op ``reset -> verify`` already returns
``reward_type == "complete"``) and the ``sql`` verifier needs an external judge. To
keep baseline-vs-expert numbers and training rewards meaningful we:

- use only the deterministic ``code`` verifier, and
- exclude tasks with no code verifier or a trivial no-op pass.

``build_split`` discovers a non-trivial task set and partitions it into train / val.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import awm_data
from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction


async def noop_verify_complete(env, scenario: str, task_idx: int) -> bool:
    # [FUNCTION DOC]
    # Resets into the task, immediately runs the code verifier with an empty final answer, ends the session, and returns
    # True if the reward_type was "complete". Side effects: network I/O. Propagates env exceptions.
    """Return `True` if ``reset -> verify`` passes with no actions (a trivial task)."""
    await env.reset(scenario=scenario, task_idx=task_idx)
    result = await env.step(
        CallToolAction(
            tool_name="verify", arguments={"verifier_mode": "code", "final_answer": ""}
        )
    )
    await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
    return result.observation.reward_type == "complete"


async def is_usable_task(env, scenario: str, task_idx: int) -> bool:
    # [FUNCTION DOC]
    # Returns False if the task has no code verifier in the dataset; otherwise returns the negation of the no-op verify
    # check, treating any exception as unusable (False).
    """A task is usable if it has a code verifier and is not a trivial no-op pass."""
    if awm_data.get_verifier_code(scenario, task_idx) is None:
        return False
    try:
        return not await noop_verify_complete(env, scenario, task_idx)
    except Exception:
        return False


@dataclass
class TaskSplit:
    # [CLASS DOC]
    # Dataclass holding lists of (scenario, task_idx) pairs for train and val.
    """A train/val partition of `(scenario, task_idx)` pairs."""

    train: list[tuple[str, int]] = field(default_factory=list)
    val: list[tuple[str, int]] = field(default_factory=list)


async def build_split(
    base_url: str,
    *,
    scenario_prefixes: tuple[str, ...] | None = ("workflow_automation",),
    max_tasks: int = 200,
    val_fraction: float = 0.35,
    seed: int = 0,
) -> TaskSplit:
    # [FUNCTION DOC]
    # Collects verifier tasks for the prefixes, shuffles them with the seed, probes each (fresh AWMEnv per task) until
    # max_tasks usable ones are found, then sorts, reshuffles with a new RNG of the same seed, and assigns the first
    # floor(len * val_fraction) to val and the rest to train (each sorted). Returns the TaskSplit. Side effects: network I/O.
    """Discover usable tasks and split them into train / val.

    Args:
        base_url (`str`):
            AWM server base URL.
        scenario_prefixes (`tuple[str, ...]`, *optional*, defaults to `("workflow_automation",)`):
            Restrict discovery to scenarios whose normalised name starts with one of
            these prefixes. Pass `None` to consider every scenario.
        max_tasks (`int`, *optional*, defaults to `200`):
            Stop after this many usable tasks are found.
        val_fraction (`float`, *optional*, defaults to `0.35`):
            Fraction of usable tasks assigned to the validation set.
        seed (`int`, *optional*, defaults to `0`):
            Shuffle seed for a deterministic split.

    Returns:
        [`TaskSplit`]: The discovered train / val partition.
    """
    candidates = list(awm_data.iter_verifier_tasks(scenario_prefixes))
    random.Random(seed).shuffle(candidates)

    usable: list[tuple[str, int]] = []
    for scenario, task_idx in candidates:
        if len(usable) >= max_tasks:
            break
        async with AWMEnv(base_url=base_url) as env:
            if await is_usable_task(env, scenario, task_idx):
                usable.append((scenario, task_idx))

    usable.sort()
    rng = random.Random(seed)
    rng.shuffle(usable)
    n_val = int(len(usable) * val_fraction)
    return TaskSplit(train=sorted(usable[n_val:]), val=sorted(usable[:n_val]))
