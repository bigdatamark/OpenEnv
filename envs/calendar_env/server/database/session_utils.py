# [MODULE DOC]
# HELPER: thin module-level facade over a single global CalendarSessionManager instance,
# created at import time, so callers can get sessions / init databases by database_id.
"""
Database session utilities - Simplified session management
"""

from database.session_manager import CalendarSessionManager
from sqlalchemy.orm import Session

# Global session manager instance
_session_manager = CalendarSessionManager()


def get_session(database_id: str) -> Session:
    # [FUNCTION DOC]
    # HELPER: returns a SQLAlchemy Session for the given database_id by delegating to the
    # global session manager. Side effects: may create the per-database engine/file lazily.
    """Get database session for the specified database ID"""
    return _session_manager.get_session(database_id)


def init_database(database_id: str):
    # [FUNCTION DOC]
    # HELPER: delegates to the global session manager's init_database with create_tables left
    # False, so it only creates/caches the engine (and the ./mcp_databases dir); returns None.
    """Initialize database for the specified database ID"""
    return _session_manager.init_database(database_id)
