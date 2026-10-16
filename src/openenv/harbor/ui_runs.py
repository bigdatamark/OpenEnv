# [MODULE DOC]
# Run management for rollouts started from the Harbor UI: each rollout runs in a daemon
# worker thread (its own asyncio loop) independent of the browser request; live runs are
# tracked in memory, recently finished ones kept in memory, and finished records
# optionally persisted as one JSON file per run. Supports per-visitor ownership filtering,
# global and per-owner concurrency caps, and reference-counted caching of visitor endpoints
# in the capture proxy. Never stores API keys.
"""Rollouts started from the UI: running ones held in memory, finished ones saved to disk.

The first UI tied a rollout to the browser request that started it. Closing the tab lost the result,
and nothing that had run before could be looked at again or compared. Here a rollout is started once
and runs in a worker thread whatever the page does; when it finishes its result is written to disk, so
the Runs tab can list, reopen and compare everything this server has run.

What is saved is the rollout result and where it came from (dataset, task, harness, sandbox, model),
and a digest of the browser that started it, so a deployment can show each visitor only their own.
Never an API key: a visitor's endpoint is recorded by its kind and model name, nothing more.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LIVE = ("starting", "running")


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Returns a compact summary of a run record: the metadata fields plus ok, reward, rewards,
    # turn/root counts, total tool calls and generated tokens over turns, wall time, rollout
    # type, capture level, ATIF status and exception type (ok is None when no result).
    """The fields a run list and a comparison read, without the token arrays."""
    r = record.get("result") or {}
    turns = r.get("turns") or []
    return {
        **{k: record.get(k) for k in _META},
        "ok": r.get("ok") if r else None,
        "reward": r.get("reward"),
        "rewards": r.get("rewards") or {},
        "n_turns": r.get("n_turns", 0),
        "n_roots": r.get("n_roots", 0),
        "tool_calls": sum(len(t.get("tool_calls") or []) for t in turns),
        "generated": sum(len(t.get("completion_token_ids") or []) for t in turns),
        "wall_s": r.get("wall_s") or record.get("wall_s"),
        "rollout_type": r.get("rollout_type"),
        "capture_level": r.get("capture_level"),
        "atif": r.get("atif"),
        "exception_type": r.get("exception_type"),
    }


_META = (
    "id",
    "status",
    "created",
    "finished",
    "dataset",
    "task_index",
    "task_name",
    "task_title",
    "harness",
    "sandbox",
    "model",
    "endpoint",
    "purpose",
    "error",
    "owner",
)


class RunStore:
    # [CLASS DOC]
    # On-disk store of finished runs (one <id>.json per run) with a lazily built, lock-protected
    # in-memory index of summaries.
    """Finished rollouts as one JSON file each, with an in-memory index of their summaries.

    Args:
        root (`Path`):
            Directory to keep them in. Created on first write.
    """

    def __init__(self, root: Path) -> None:
        # [FUNCTION DOC]
        # Stores the root directory; the index is built on first use.
        self.root = root
        self._index: dict[str, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: builds (once) and returns the index from non-hidden *.json files in root,
        # skipping unreadable or id-less files. Caller should hold the lock when mutating.
        if self._index is None:
            index: dict[str, dict[str, Any]] = {}
            for path in sorted(self.root.glob("*.json")) if self.root.is_dir() else []:
                if path.name.startswith("."):
                    continue  # the folder's own bookkeeping, not runs
                try:
                    record = json.loads(path.read_text())
                    if isinstance(record, dict) and record.get("id"):
                        index[record["id"]] = summarize(record)
                except (OSError, ValueError, TypeError, AttributeError):
                    continue  # a file this version cannot read is skipped, never fatal
            self._index = index
        return self._index

    def save(self, record: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Writes the record atomically to <root>/<id>.json (creating root; non-JSON values
        # stringified) and updates the index. Raises OSError on write failure; KeyError without id.
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{record['id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, default=str))
        tmp.replace(
            path
        )  # atomic, so a crash mid-write never leaves half a record in the list
        with self._lock:
            self._load()[record["id"]] = summarize(record)

    def get(self, run_id: str) -> dict[str, Any] | None:
        # [FUNCTION DOC]
        # Returns the parsed record for run_id, or None for ids containing "/" or "..", starting
        # with ".", empty, missing or unreadable files.
        if not run_id or "/" in run_id or ".." in run_id or run_id.startswith("."):
            return None
        try:
            return json.loads((self.root / f"{run_id}.json").read_text())
        except (OSError, ValueError):
            return None

    def list(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Returns all summaries sorted by created time, newest first.
        with self._lock:
            return sorted(
                self._load().values(), key=lambda s: s.get("created") or 0, reverse=True
            )


@dataclass
class LiveRun:
    # [CLASS DOC]
    # Mutable dataclass for an in-flight rollout: identity/metadata, owner digest, quota key
    # (memory only), timestamps, status (starting/running/done/failed), capture session id,
    # result dict and error.
    """A rollout in flight: what it is, and the capture session its live view follows."""

    id: str
    dataset: str
    task_index: int
    task_name: str
    harness: str
    sandbox: str
    model: str
    endpoint: str
    purpose: str
    owner: str = ""
    # What `per_owner` counts against; in memory only (not in `_META`), so never written to disk.
    quota: str = ""
    task_title: str = ""
    created: float = field(default_factory=time.time)
    status: str = "starting"
    session_id: str | None = None
    finished: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def record(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a dict of the persisted metadata fields, wall_s (to finish or now) and result.
        return {
            **{k: getattr(self, k) for k in _META},
            "wall_s": (self.finished or time.time()) - self.created,
            "result": self.result,
        }


class RunManager:
    # [CLASS DOC]
    # Starts UI rollouts and tracks live, recently finished and stored runs.
    """Starts UI rollouts, tracks the running ones, and hands finished ones to the store.

    Args:
        store (`RunStore`, *optional*):
            Where finished rollouts go. Without one they stay in memory until the process exits.
        max_live (`int`, *optional*, defaults to `4`):
            Rollouts the UI may run at once. Each one holds a sandbox and spends the endpoint's credit,
            and on a public deployment anyone with the URL can press Run.
    """

    def __init__(
        self, store: RunStore | None = None, max_live: int | None = None
    ) -> None:
        # [FUNCTION DOC]
        # Stores the optional store and the live cap (default 4), and initializes live, finished,
        # engine refcount maps and a lock.
        self.store = store
        self.max_live = max_live or 4
        self._live: dict[str, LiveRun] = {}
        self._engines: dict[
            tuple, int
        ] = {}  # visitor endpoints in use, by proxy cache key
        self._done: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        engine: dict[str, Any],
        spec: str,
        index: int,
        harness: str,
        sandbox: str,
        service: Any,
        owner: str = "",
        title: str = "",
        per_owner: int | None = None,
        private_urls: bool = True,
        quota: str = "",
    ) -> str:
        # [FUNCTION DOC]
        # Resolves the task directory (errors propagate), creates a LiveRun with a timestamped
        # random id, model (engine or service default) and endpoint label ("server default" or
        # the engine kind). Under the lock, raises RuntimeError if max_live runs are live or the
        # quota key already has per_owner live runs; otherwise registers it. Increments the
        # visitor-engine refcount and starts a daemon thread that runs _rollout via asyncio.run;
        # the worker records the result (done if ok else failed) or a failure message, releases
        # the engine, atomically moves the run from live to finished (keeping the last 50, or 200
        # without a store), and saves it to the store (OSError ignored). Returns the run id immediately.
        """Start a rollout in a worker thread and return its run id at once.

        `owner` is the digest `ui_settings.owner_of` makes of the visitor's browser id; listings and
        lookups made for that visitor pass the same digest. `per_owner` caps how many of the running
        rollouts one owner may hold, counted by `quota` when one is given: the signed-in Hugging Face
        account, since a browser id is free to replace. `private_urls=False` checks a visitor's endpoint URL again right
        before the rollout calls it (see `ui_settings.url_problem`).

        Raises:
            `RuntimeError`: when `max_live` rollouts are already running, or `per_owner` of the owner's.
        """
        from .tasks import HarborTaskProvider

        task_dir = HarborTaskProvider([spec]).task_dir(spec, int(index))
        custom = not engine.get("server_default")
        run = LiveRun(
            id=time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3),
            dataset=spec,
            task_index=int(index),
            task_name=task_dir.name,
            task_title=title or task_dir.name,
            harness=harness,
            sandbox=sandbox,
            model=str(engine.get("model") or service.model or ""),
            endpoint=str(engine.get("kind") or "custom endpoint")
            if custom
            else "server default",
            purpose=str(engine.get("purpose") or "eval"),
            owner=owner,
            quota=quota or owner,
        )
        with self._lock:
            running = [r for r in self._live.values() if r.status in LIVE]
            if len(running) >= self.max_live:
                raise RuntimeError(
                    f"{self.max_live} rollouts are already running on this server. "
                    "Wait for one to finish."
                )
            if (
                per_owner
                and sum(1 for r in running if r.quota == run.quota) >= per_owner
            ):
                raise RuntimeError(
                    f"You already have {per_owner} rollout{'s' if per_owner != 1 else ''} running. "
                    "Wait for one to finish."
                )
            self._live[run.id] = run

        def created(session_id: str) -> None:
            # The live view follows exactly this session. Finding "the newest session" in the registry
            # instead would show one visitor another's run as soon as two ran at once.
            # [FUNCTION DOC]
            # HELPER: session-created callback setting the run's session id and status "running".
            run.session_id = session_id
            run.status = "running"

        def worker() -> None:
            # [FUNCTION DOC]
            # HELPER: thread body running the rollout and finalizing the run as described above.
            import asyncio

            outcome: dict[str, Any] = {
                "status": "failed",
                "error": "The rollout worker was interrupted.",
            }
            try:
                result = asyncio.run(
                    _rollout(
                        engine,
                        task_dir,
                        spec,
                        harness,
                        sandbox,
                        service,
                        created,
                        private_urls,
                    )
                )
                data = result.model_dump(mode="json")
                outcome = {
                    "result": data,
                    "status": "done" if data.get("ok") else "failed",
                    "error": data.get("error"),
                }
            except asyncio.CancelledError as exc:
                outcome = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: cancelled",
                }
            except Exception as exc:  # noqa: BLE001 - a failed rollout is a result, never a crash
                outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                try:
                    self._release_engine(engine, service)
                finally:
                    with self._lock:
                        # All at once, under the lock that lists runs: a tick between "status done"
                        # and "no longer live" would draw a finished run as running forever.
                        run.result = outcome.get("result")
                        run.status, run.error = outcome["status"], outcome.get("error")
                        run.finished = time.time()
                        record = run.record()
                        self._done[run.id] = record
                        self._live.pop(run.id, None)
                        # Saved runs are read back from disk; without a store, memory is the only
                        # history, so a longer one is kept.
                        keep = _KEEP_DONE if self.store is not None else _KEEP_DONE * 4
                        for old in list(self._done)[:-keep]:
                            del self._done[old]
                    if self.store is not None:
                        try:
                            self.store.save(record)
                        except OSError:
                            pass  # result remains in memory and on screen

        self._hold_engine(engine)
        threading.Thread(target=worker, daemon=True, name=f"harbor-ui-{run.id}").start()
        return run.id

    def _hold_engine(self, engine: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: increments the refcount of a visitor endpoint's upstream cache key (no-op for the server default).
        upstream = _visitor_upstream(engine)
        if upstream is not None:
            with self._lock:
                key = upstream.cache_key
                self._engines[key] = self._engines.get(key, 0) + 1

    def _release_engine(self, engine: dict[str, Any], service: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: decrements that refcount; when it reaches zero removes it and makes the proxy
        # forget the cached client, all under the lock.
        """Forget a visitor's endpoint in the proxy once no running rollout uses it.

        The proxy resolves the engine on every model call, so forgetting it while another rollout on
        the same endpoint is still running would make that rollout probe it again, for real.
        """
        upstream = _visitor_upstream(engine)
        if upstream is None:
            return
        with self._lock:
            key = upstream.cache_key
            left = self._engines.get(key, 1) - 1
            if left > 0:
                self._engines[key] = left
                return
            self._engines.pop(key, None)
            # Keep zero-count and eviction atomic with `_hold_engine`. Otherwise another rollout can
            # acquire this key after the pop and have its freshly resolved client dropped here.
            _forget_visitor_engine(upstream, service)

    def live(self, run_id: str, owner: str | None = None) -> LiveRun | None:
        # [FUNCTION DOC]
        # Returns the live run if present and visible to owner (None = no filter), else None.
        with self._lock:
            run = self._live.get(run_id)
        return run if run is not None and _mine(run.owner, owner) else None

    def get(self, run_id: str, owner: str | None = None) -> dict[str, Any] | None:
        # [FUNCTION DOC]
        # Returns the full record from live runs, in-memory finished runs, or the store, filtered
        # by owner; None if absent or not visible.
        """A run's full record: live, finished in this process, or saved earlier.

        With `owner`, only a run that visitor started. `None` means no filter, which is the
        `all` visibility; an empty string matches nothing, so a visitor with no id sees no one's runs.
        """
        with self._lock:
            live = self._live.get(run_id)
            record = live.record() if live is not None else self._done.get(run_id)
        if record is None and self.store is not None:
            record = self.store.get(run_id)
        return (
            record if record is not None and _mine(record.get("owner"), owner) else None
        )

    def list(self, owner: str | None = None) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Returns summaries of live, in-memory finished and stored runs (deduplicated by id),
        # filtered by owner, newest first.
        """Every run this server knows about, newest first, as summaries; with `owner`, that
        visitor's only."""
        with self._lock:
            live = [summarize(r.record()) for r in self._live.values()]
            done = [summarize(r) for r in self._done.values()]
        seen = {s["id"] for s in live + done}
        stored = [
            s for s in (self.store.list() if self.store else []) if s["id"] not in seen
        ]
        return sorted(
            (s for s in live + done + stored if _mine(s.get("owner"), owner)),
            key=lambda s: s.get("created") or 0,
            reverse=True,
        )


def _mine(recorded: Any, owner: str | None) -> bool:
    # [FUNCTION DOC]
    # HELPER: visibility test: True when owner is None (no filter), else only when owner is
    # non-empty and equals the recorded owner.
    return owner is None or (bool(owner) and recorded == owner)


_KEEP_DONE = 50


def _visitor_upstream(engine: dict[str, Any]) -> Any:
    # [FUNCTION DOC]
    # HELPER: returns a capture Upstream for a visitor-supplied endpoint (url, model, api key,
    # provider), or None for an empty URL or the server default.
    """The `Upstream` a visitor's endpoint maps to, or `None` for the server's own."""
    url = str(engine.get("url") or "").strip()
    if not url or engine.get("server_default"):
        return None
    from openenv.core.harness.capture.sessions import Upstream

    return Upstream(
        llm_url=url,
        model=str(engine.get("model") or ""),
        api_key=str(engine.get("api_key") or "") or None,
        provider=str(engine.get("provider") or "openai"),
    )


def _forget_visitor_engine(upstream: Any, service: Any) -> None:
    # [FUNCTION DOC]
    # HELPER: removes the upstream's cached client from the capture app's upstream pool.
    """Drop the proxy's cached client for a visitor's endpoint.

    The pool keeps one client per engine and credential so concurrent rollouts share a probe. For a
    visitor's key that cache would otherwise outlive the page the key was typed into.
    """
    service.capture.app.state.upstreams.forget(upstream)


async def _rollout(
    engine: dict[str, Any],
    task_dir: Path,
    spec: str,
    harness: str,
    sandbox: str,
    service: Any,
    on_session_created: Any,
    private_urls: bool = True,
) -> Any:
    # [FUNCTION DOC]
    # HELPER: async. For a visitor endpoint re-checks the URL (ValueError if now disallowed)
    # and resolves it through the proxy pool (probing it); otherwise uses the pool default with
    # the service's capture level and model. Runs rollout.run_rollout with the harness profile
    # from the engine, the service's registry and public URL, trials dir from
    # OPENENV_HARBOR_TRIALS_DIR (default /tmp/openenv-harbor-trials), the configured reward
    # key, require_reward False, purpose (default eval) and the session callback. Returns the result.
    """Resolve the endpoint this rollout uses, then run it.

    The server's own endpoint is the default and needs nothing from the browser: its key stays in the
    capture proxy. A custom endpoint validated in the page goes through the proxy's pool, so its tier
    comes from a real probe of that endpoint, cached after the first run.
    """
    from . import rollout as _rollout_module
    from .ui_settings import load as load_settings, url_problem

    pool = service.capture.app.state.upstreams
    upstream = _visitor_upstream(engine)
    if upstream is not None:
        # Checked when it was connected, and again now: a name can resolve elsewhere since then.
        problem = url_problem(upstream.llm_url, private_ok=private_urls)
        if problem:
            raise ValueError(problem)
        client, level = await pool.resolve(upstream)
        served = client.served_model or upstream.model
    else:
        upstream, (client, _) = None, pool.default
        level = service.capture_level
        served = service.model
    return await _rollout_module.run_rollout(
        task_dir=task_dir,
        harness=harness,
        harness_profile=(engine.get("harness_profiles") or {}).get(harness),
        sandbox=sandbox,
        registry=service.capture.registry,
        intercept_url=service.public_url,
        model=served,
        trials_dir=Path(
            os.environ.get("OPENENV_HARBOR_TRIALS_DIR", "/tmp/openenv-harbor-trials")
        ),
        dataset=spec,
        # A page's run is for reading: several rewards with none chosen are shown, not a failure.
        reward_key=load_settings().reward_key,
        require_reward=False,
        capture_level=level,
        purpose=str(engine.get("purpose") or "eval"),
        upstream=upstream,
        inference=client,
        on_session_created=on_session_created,
    )


_MANAGER: RunManager | None = None
_MANAGER_LOCK = threading.Lock()


def manager() -> RunManager:
    # [FUNCTION DOC]
    # Returns the process-wide RunManager, creating it on first call from UI settings (store in
    # the runs dir when history is on, max_runs cap). Thread-safe.
    """The process-wide run manager, created on first use with the configured history directory."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            from .ui_settings import load

            settings = load()
            root = settings.runs_dir
            _MANAGER = RunManager(
                RunStore(root) if root is not None else None, settings.max_runs
            )
        return _MANAGER
