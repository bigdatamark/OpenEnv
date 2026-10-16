# [MODULE DOC]
# Replaces the process-global os.environ with a mapping whose reads consult a
# context-local (contextvars) overlay first, so concurrent rollouts in one process can
# each see their own credentials (e.g. API key = session id) without a lock. Writes go
# to the real environment. Can be disabled via OPENENV_CONCURRENT_PROC_ENV=0.
# KNOWN ISSUE: the module docstring says subprocess builds the child environment from
# os.environ; on POSIX, subprocess with env=None inherits the C-level environment, so the
# overlay only reaches children when callers pass env explicitly (e.g. os.environ.copy()).
"""Per-rollout `os.environ` reads, so credential-by-env harnesses stop serialising.

Three validated harnesses — claude-code, gemini-cli and goose — read `os.environ` inside their
`run()` to assemble the env dict they hand to the sandbox:

    api_key = os.environ.get("OPENAI_API_KEY")          # goose.py:653
    "ANTHROPIC_BASE_URL": os.environ.get(...)           # claude_code.py:1393
    for var in auth_vars: env[var] = os.environ[var]    # gemini_cli.py:824

And in this system **the API key IS the rollout's session id**, which is how one capture proxy
multiplexes N concurrent rollouts. So each concurrent rollout needs a DIFFERENT value of the same
variable at the same instant, in one process. `os.environ` is process-global, so the only correct
answer used to be `_PROC_ENV_LOCK` — serialise them, and accept that three harnesses run one rollout
at a time while the other twelve run in parallel.

The observation that removes the lock: those wrappers only ever READ, and only to build a dict. They
do not need the value to be globally visible — they need it to be visible *to them, now*. That is a
context-local read, and `contextvars` are exactly that, propagating into asyncio tasks and into
`asyncio.to_thread` (which copies the context).

So `os.environ` is replaced by a mapping that consults a context-local overlay first and the real
environment second. Each rollout sets its own overlay; concurrent rollouts see different values from
the same expression; nothing global is mutated, and no lock is needed.

Two deliberate properties:

  * **Iteration and `copy()` return the MERGED view.** `subprocess` builds a child's environment from
    `os.environ` unless told otherwise, so a proxy that hid the overlay from iteration would silently
    launch subprocesses without the credentials — the failure would look like a bad key, not a bad
    proxy.
  * **Writes go to the real environment.** Only reads are context-local. Code that sets a variable
    expecting it to persist keeps working, which matters because this replaces a global object used by
    every library in the process.

`OPENENV_CONCURRENT_PROC_ENV=0` disables it and restores the lock, because this swaps out a global that
the whole process reads and a single bad interaction is worth being able to switch off without a
rollback.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator, MutableMapping
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_overlay: ContextVar[dict[str, str] | None] = ContextVar(
    "openenv_env_overlay", default=None
)
_installed = False
_real_environ: MutableMapping[str, str] | None = None


class _ContextEnviron(MutableMapping):
    # [CLASS DOC]
    # MutableMapping proxy over the real os.environ: lookups, membership, iteration, length
    # and copy() see the base merged with the current context's overlay (overlay wins);
    # set/delete affect only the base.
    """`os.environ` whose reads consult a context-local overlay first."""

    def __init__(self, base: MutableMapping[str, str]) -> None:
        # [FUNCTION DOC]
        # Stores the wrapped base mapping.
        self._base = base

    # --- reads: overlay wins ---------------------------------------------
    def __getitem__(self, key: str) -> str:
        # [FUNCTION DOC]
        # Returns the overlay value if the key is in the current overlay, else the base value
        # (KeyError if absent in both).
        over = _overlay.get()
        if over is not None and key in over:
            return over[key]
        return self._base[key]

    def __iter__(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # Yields all base keys (snapshot), then overlay keys not already present in the base.
        over = _overlay.get() or {}
        seen = set()
        for key in list(self._base):
            seen.add(key)
            yield key
        for key in over:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of distinct keys across base and overlay.
        over = _overlay.get() or {}
        return len(set(self._base) | set(over))

    def __contains__(self, key: object) -> bool:
        # [FUNCTION DOC]
        # True if the key is in the overlay or the base.
        over = _overlay.get()
        return (over is not None and key in over) or key in self._base

    def copy(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Returns a plain dict of base values updated with overlay values.
        """Merged, because `subprocess` uses this to build a child's environment."""
        merged = dict(self._base)
        merged.update(_overlay.get() or {})
        return merged

    # --- writes: real environment ----------------------------------------
    def __setitem__(self, key: str, value: str) -> None:
        # [FUNCTION DOC]
        # Writes to the real environment (which also calls putenv).
        self._base[key] = value

    def __delitem__(self, key: str) -> None:
        # [FUNCTION DOC]
        # Deletes from the real environment; KeyError if absent there.
        # KNOWN ISSUE: a key also set in the current overlay stays visible after deletion.
        del self._base[key]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        # [FUNCTION DOC]
        # Debug representation with the merged variable count and whether an overlay is active.
        return f"_ContextEnviron({len(self)} vars, overlay={bool(_overlay.get())})"


def enabled() -> bool:
    # [FUNCTION DOC]
    # Returns False only when OPENENV_CONCURRENT_PROC_ENV is "0" (default enabled).
    return os.environ.get("OPENENV_CONCURRENT_PROC_ENV", "1") != "0"


def install() -> bool:
    # [FUNCTION DOC]
    # If already installed returns True; if disabled returns False; otherwise saves the real
    # os.environ and replaces os.environ with a _ContextEnviron wrapping it, logs, returns True.
    # Side effects: mutates the global os module attribute for the whole process.
    """Swap `os.environ` for the context-aware proxy. Idempotent; returns whether it is active."""
    global _installed, _real_environ
    if _installed:
        return True
    if not enabled():
        return False
    _real_environ = os.environ
    os.environ = _ContextEnviron(_real_environ)  # type: ignore[assignment]
    _installed = True
    logger.info(
        "os.environ reads are context-local; credential-by-env harnesses no longer serialise"
    )
    return True


@contextlib.contextmanager
def overlay(values: dict[str, str]):
    # [FUNCTION DOC]
    # Context manager that sets the current context's overlay to the parent overlay merged
    # with values for the duration of the block, then restores the previous overlay. Only
    # affects reads through the installed proxy (no effect if install() was not called).
    """Make `values` visible to `os.environ` reads in THIS context only."""
    parent = _overlay.get() or {}
    token = _overlay.set({**parent, **values})
    try:
        yield
    finally:
        _overlay.reset(token)
