# [MODULE DOC]
# Process-wide registry of live AWMEnvironment sessions with last-activity timestamps, plus a daemon thread that
# periodically tears down idle sessions. Exposes the singleton "registry".
"""
Module-level session registry for AWM environments.

Tracks active AWMEnvironment instances with idle timestamps, and runs a
background daemon thread that periodically kills idle sessions.

All config values are read from config.py:
    MAX_IDLE_TIME, CLEANUP_INTERVAL, ALLOWED_IDLE_SESSIONS
"""

import logging
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any

from .config import ALLOWED_IDLE_SESSIONS, CLEANUP_INTERVAL, MAX_IDLE_TIME

if TYPE_CHECKING:
    from .awm_environment import AWMEnvironment

logger = logging.getLogger(__name__)


class SessionRegistry:
    # [CLASS DOC]
    # Lock-protected mapping from session id to an entry holding a weak reference to the environment, monotonic
    # last-active time, scenario name and wall-clock registration time. Starts its cleanup daemon on first register.
    """Thread-safe singleton tracking active AWMEnvironment instances."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes the lock, empty session map and daemon-started flag.
        self._lock = threading.Lock()
        # session_id -> {env: weakref, last_active: float, scenario: str|None}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._daemon_started = False

    def register(
        self, session_id: str, env: "AWMEnvironment", scenario: str | None = None
    ) -> None:
        # [FUNCTION DOC]
        # Adds or replaces the entry for session_id (weakref to env, now as last-active, scenario) and starts the cleanup
        # daemon if not yet running.
        with self._lock:
            self._sessions[session_id] = {
                "env": weakref.ref(env),
                "last_active": time.monotonic(),
                "scenario": scenario,
                "registered_at": time.time(),
            }
        self._ensure_daemon()

    def unregister(self, session_id: str) -> None:
        # [FUNCTION DOC]
        # Removes the session entry if present.
        with self._lock:
            self._sessions.pop(session_id, None)

    def touch(self, session_id: str) -> None:
        # [FUNCTION DOC]
        # Sets the session's last-active time to now; no-op for unknown ids.
        """Update last_active timestamp for a session."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry["last_active"] = time.monotonic()

    @property
    def active_count(self) -> int:
        # [FUNCTION DOC]
        # Property: number of registered sessions.
        with self._lock:
            return len(self._sessions)

    def get_stats(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a dict with total_sessions, the three idle-config values, max_idle_s (largest idle seconds rounded to 0.1,
        # 0 if none) and a count of sessions per scenario ("unknown" when unset).
        """Return registry stats for the /stats endpoint."""
        now = time.monotonic()
        with self._lock:
            idle_times = []
            scenarios: dict[str, int] = {}
            for entry in self._sessions.values():
                idle = now - entry["last_active"]
                idle_times.append(idle)
                sc = entry.get("scenario") or "unknown"
                scenarios[sc] = scenarios.get(sc, 0) + 1

            return {
                "total_sessions": len(self._sessions),
                "max_idle_time_config": MAX_IDLE_TIME,
                "cleanup_interval_config": CLEANUP_INTERVAL,
                "allowed_idle_sessions_config": ALLOWED_IDLE_SESSIONS,
                "max_idle_s": round(max(idle_times), 1) if idle_times else 0,
                "scenarios": scenarios,
            }

    def cleanup_idle(self) -> int:
        # [FUNCTION DOC]
        # Only when the number of sessions exceeds ALLOWED_IDLE_SESSIONS: removes every session idle longer than
        # MAX_IDLE_TIME and calls the live environment's _cleanup_session(). Errors during cleanup are logged.
        # Returns: number of sessions whose environment was successfully cleaned (removed dead refs are not counted).
        """Kill sessions that have been idle longer than MAX_IDLE_TIME.

        Only triggers when total sessions exceed ALLOWED_IDLE_SESSIONS.
        Returns the number of sessions cleaned up.
        """
        now = time.monotonic()
        to_remove: list[str] = []

        with self._lock:
            if len(self._sessions) <= ALLOWED_IDLE_SESSIONS:
                return 0

            for sid, entry in self._sessions.items():
                idle = now - entry["last_active"]
                if idle > MAX_IDLE_TIME:
                    to_remove.append(sid)

        cleaned = 0
        for sid in to_remove:
            with self._lock:
                entry = self._sessions.pop(sid, None)
            if entry is None:
                continue

            env_ref = entry["env"]
            env = env_ref()
            if env is not None:
                try:
                    env._cleanup_session()
                    logger.info(
                        f"[cleanup] Killed idle session {sid} "
                        f"(scenario={entry.get('scenario')})"
                    )
                    cleaned += 1
                except Exception as e:
                    logger.warning(f"[cleanup] Error cleaning session {sid}: {e}")

        if cleaned > 0:
            logger.info(f"[cleanup] Cleaned {cleaned} idle sessions")

        return cleaned

    def _ensure_daemon(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Starts the cleanup daemon thread once (flag-guarded).
        # KNOWN ISSUE: the flag check is not under the lock, so concurrent first registrations could start two daemons.
        """Start the cleanup daemon thread if not already running."""
        if self._daemon_started:
            return
        self._daemon_started = True
        t = threading.Thread(target=self._daemon_loop, daemon=True, name="awm-cleanup")
        t.start()
        logger.info("[cleanup] Daemon thread started")

    def _daemon_loop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Infinite loop: sleeps CLEANUP_INTERVAL, runs cleanup_idle (errors logged), then drops entries whose
        # environment has been garbage-collected.
        """Background loop that periodically cleans idle sessions."""
        while True:
            time.sleep(CLEANUP_INTERVAL)
            try:
                self.cleanup_idle()
            except Exception as e:
                logger.error(f"[cleanup] Daemon error: {e}")

            # Prune dead weakrefs
            with self._lock:
                dead = [
                    sid
                    for sid, entry in self._sessions.items()
                    if entry["env"]() is None
                ]
                for sid in dead:
                    del self._sessions[sid]


# Module-level singleton
registry = SessionRegistry()
