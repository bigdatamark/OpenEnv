# [MODULE DOC]
# Backends that implement the REPL's LLM helper functions: a direct backend sending single-message
# chats (with threaded batching and optional result truncation) and a recursive backend that spawns
# child RLM runners for recursive queries, bounded by depth, a tree-wide child count, per-batch caps
# and a cooperative per-child timeout. Records a trace for each child run.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Recursive backend abstractions for repl_env.

This module keeps direct LM calls and recursive child spawning out of the
runner and environment. The runner owns the iterative loop; the backend owns
query/query_batched/child-recursion behavior.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol


ChatFn = Callable[..., str]


class RecursiveBackend(Protocol):
    # [CLASS DOC]
    # Structural protocol for backends: max_depth, depth, child_traces and the four query methods.
    max_depth: int
    depth: int
    child_traces: list["ChildTrace"]

    def query(self, prompt: str, model: str | None = None) -> str:
        # [FUNCTION DOC]
        # Protocol: single LLM query returning text.
        ...

    def query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        # [FUNCTION DOC]
        # Protocol: batched LLM queries returning texts in input order.
        ...

    def recursive_query(self, prompt: str, model: str | None = None) -> str:
        # [FUNCTION DOC]
        # Protocol: recursive (child RLM) query returning text.
        ...

    def recursive_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        # [FUNCTION DOC]
        # Protocol: batched recursive queries returning texts in input order.
        ...


@dataclass
class BackendLimits:
    # [CLASS DOC]
    # Mutable limits shared by a recursion tree: max depth, batch thread workers, optional total children,
    # per-batch children, result truncation length and per-child timeout, plus an internal spawned-children
    # counter guarded by a thread lock.
    max_depth: int = 1
    max_batch_workers: int = 8
    max_children_total: int | None = None
    max_children_per_batch: int | None = None
    result_truncation_limit: int | None = None
    # Cooperative timeout: checked between iterations, not during LLM calls.
    # A slow LLM call within an iteration will not be interrupted — the timeout
    # fires at the next iteration boundary. For mid-call cancellation, use
    # process-based isolation instead.
    per_child_timeout_s: float | None = None
    # Tree-global child counter shared across all recursion depths
    _children_spawned: int = field(default=0, init=False, repr=False)
    _children_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )


@dataclass
class ChildTrace:
    # [CLASS DOC]
    # Mutable record of one child run: depth, duration in seconds, first 80 characters of prompt and result
    # (or None) and error text.
    depth: int
    duration_s: float
    prompt_preview: str
    result_preview: str | None
    error: str | None


class DirectLMBackend:
    # [CLASS DOC]
    # Backend calling the chat function directly; recursive queries fall back to plain queries.
    """Direct LM backend with no child recursion beyond fallback to itself."""

    def __init__(
        self,
        llm_chat_fn: ChatFn,
        *,
        depth: int = 0,
        limits: BackendLimits | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the chat function, depth and limits (default BackendLimits) and an empty trace list.
        self.llm_chat_fn = llm_chat_fn
        self.depth = depth
        self.limits = limits or BackendLimits()
        self.max_depth = self.limits.max_depth
        self.child_traces: list[ChildTrace] = []

    def query(self, prompt: str, model: str | None = None) -> str:
        # [FUNCTION DOC]
        # Sends one user message (calling the chat function with the model, retrying without it on TypeError)
        # and returns the possibly truncated reply.
        # KNOWN ISSUE: a TypeError raised inside the chat call itself also triggers the retry, so a failing
        # call may be made twice.
        try:
            result = self.llm_chat_fn([{"role": "user", "content": prompt}], model)
        except TypeError:
            result = self.llm_chat_fn([{"role": "user", "content": prompt}])
        return self._truncate(result)

    def query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        # [FUNCTION DOC]
        # Runs query for each prompt in a thread pool (up to max_batch_workers) and returns replies in input
        # order; a failed prompt yields "Error: <message>". Empty input returns an empty list.
        if not prompts:
            return []
        max_workers = min(len(prompts), self.limits.max_batch_workers)
        results: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.query, prompt, model): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = f"Error: {exc}"
        return results

    def recursive_query(self, prompt: str, model: str | None = None) -> str:
        # [FUNCTION DOC]
        # Same as query.
        return self.query(prompt, model)

    def recursive_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        # [FUNCTION DOC]
        # Same as query_batched.
        return self.query_batched(prompts, model)

    def _truncate(self, result: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Truncates result to result_truncation_limit characters when set.
        limit = self.limits.result_truncation_limit
        if limit is not None and len(result) > limit:
            return result[:limit]
        return result


class LocalChildRLMBackend(DirectLMBackend):
    # [CLASS DOC]
    # Backend that runs recursive queries in child runners created by runner_factory, sharing limits
    # across the tree and optionally reporting subcall start/completion to callbacks.
    """Recursive backend that spawns child LocalRLMRunner instances."""

    def __init__(
        self,
        llm_chat_fn: ChatFn,
        *,
        runner_factory: Callable[..., object],
        system_prompt: str,
        max_iterations: int,
        env_max_iterations_multiplier: int,
        depth: int = 0,
        limits: BackendLimits | None = None,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None]
        | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Initialises the direct backend part and stores the runner factory, system prompt, iteration settings
        # and callbacks.
        super().__init__(llm_chat_fn, depth=depth, limits=limits)
        self.runner_factory = runner_factory
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.env_max_iterations_multiplier = env_max_iterations_multiplier
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete

    def recursive_query(self, prompt: str, model: str | None = None) -> str:
        # [FUNCTION DOC]
        # Runs a child RLM on the prompt (used as both context and task). Falls back to a plain query when the
        # next depth reaches max_depth. Returns "Error: max_children_total exceeded" when the shared cap is
        # hit; otherwise increments the shared counter, calls the start callback (errors ignored), runs the
        # child with the per-child timeout and returns its truncated final answer (empty if none). Child
        # exceptions are re-raised. Always appends a ChildTrace and calls the completion callback (errors ignored).
        next_depth = self.depth + 1
        if next_depth >= self.max_depth:
            return self.query(prompt, model)
        with self.limits._children_lock:
            if self.limits.max_children_total is not None:
                if self.limits._children_spawned >= self.limits.max_children_total:
                    return "Error: max_children_total exceeded"
            self.limits._children_spawned += 1
        start = time.perf_counter()
        error: str | None = None
        result_text = ""
        resolved_model = model or "default"
        if self.on_subcall_start is not None:
            try:
                self.on_subcall_start(next_depth, str(resolved_model), prompt[:80])
            except Exception:
                pass
        try:
            child = self.runner_factory(
                self.llm_chat_fn,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations,
                max_depth=self.max_depth,
                depth=next_depth,
                env_max_iterations_multiplier=self.env_max_iterations_multiplier,
                max_batch_workers=self.limits.max_batch_workers,
                backend_factory=self._child_backend_factory,
                on_subcall_start=self.on_subcall_start,
                on_subcall_complete=self.on_subcall_complete,
            )
            result = child.run(
                prompt, prompt, model=model, timeout_s=self.limits.per_child_timeout_s
            )
            result_text = self._truncate(result.final_answer or "")
            return result_text
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration = time.perf_counter() - start
            self.child_traces.append(
                ChildTrace(
                    depth=next_depth,
                    duration_s=duration,
                    prompt_preview=prompt[:80],
                    result_preview=(result_text[:80] if result_text else None),
                    error=error,
                )
            )
            if self.on_subcall_complete is not None:
                try:
                    self.on_subcall_complete(
                        next_depth,
                        str(resolved_model),
                        duration,
                        error,
                    )
                except Exception:
                    pass

    def recursive_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        # [FUNCTION DOC]
        # Runs recursive_query for each prompt (first max_children_per_batch only, if set) in a thread pool and
        # returns results in order, with "Error: <message>" for failures. Prompts beyond the cap are dropped
        # silently, so the result can be shorter than the input.
        if not prompts:
            return []
        batch_limit = self.limits.max_children_per_batch
        if batch_limit is not None:
            prompts = prompts[:batch_limit]
        max_workers = min(len(prompts), self.limits.max_batch_workers)
        results: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.recursive_query, prompt, model): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = f"Error: {exc}"
        return results

    def _child_backend_factory(
        self, llm_chat_fn: ChatFn, **kwargs
    ) -> "LocalChildRLMBackend":
        # [FUNCTION DOC]
        # HELPER: Builds a child LocalChildRLMBackend at the given depth sharing this backend's limits,
        # factory and callbacks. Reads system_prompt, max_iterations, env_max_iterations_multiplier and depth
        # from kwargs (KeyError if missing).
        return LocalChildRLMBackend(
            llm_chat_fn,
            runner_factory=self.runner_factory,
            system_prompt=kwargs["system_prompt"],
            max_iterations=kwargs["max_iterations"],
            env_max_iterations_multiplier=kwargs["env_max_iterations_multiplier"],
            depth=kwargs["depth"],
            limits=self.limits,
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
        )
