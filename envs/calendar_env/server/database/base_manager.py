# [MODULE DOC]
# Generic raw-sqlite3 CRUD helper base class. Each operation opens a fresh sqlite3
# connection to db_path, runs one statement, commits where needed and closes.
"""
Base Manager - Common CRUD operations for all Calendar services
"""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseManager:
    # [CLASS DOC]
    # Base class holding a SQLite file path and offering query/insert/update/delete plus
    # id-based convenience methods. Table/column/order names are interpolated directly into
    # SQL (not parameterized), so callers must pass trusted identifiers.
    """Base manager for common database operations"""

    def __init__(self, db_path: str):
        # [FUNCTION DOC]
        # Stores db_path. No I/O.
        self.db_path = db_path

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        # [FUNCTION DOC]
        # Runs a SELECT with positional params and returns all rows as a list of dicts (column -> value).
        # Side effects: opens/closes a sqlite3 connection. Propagates sqlite3 errors.
        """Execute a SELECT query and return results"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        # [FUNCTION DOC]
        # Runs an INSERT, commits and returns cursor.lastrowid.
        # Raises: ValueError wrapping sqlite3.IntegrityError (logged); other sqlite errors propagate.
        """Execute an INSERT query and return the last row ID"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error: {e}")
            raise ValueError(f"Database constraint violation: {e}")
        finally:
            conn.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        # [FUNCTION DOC]
        # Runs an UPDATE, commits and returns the affected row count.
        """Execute an UPDATE query and return the number of affected rows"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def execute_delete(self, query: str, params: tuple = ()) -> int:
        # [FUNCTION DOC]
        # Runs a DELETE, commits and returns the affected row count.
        """Execute a DELETE query and return the number of affected rows"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_by_id(self, table: str, record_id: int) -> Optional[Dict]:
        # [FUNCTION DOC]
        # Returns the row of the given table whose "id" equals record_id as a dict, or None.
        """Get a record by ID from any table"""
        query = f"SELECT * FROM {table} WHERE id = ?"
        results = self.execute_query(query, (record_id,))
        return results[0] if results else None

    def get_all(
        self, table: str, limit: int = 100, offset: int = 0, order_by: str = "id DESC"
    ) -> List[Dict]:
        # [FUNCTION DOC]
        # Returns up to limit rows of a table ordered by order_by (default "id DESC") starting at offset.
        """Get all records from a table with pagination"""
        query = f"SELECT * FROM {table} ORDER BY {order_by} LIMIT ? OFFSET ?"
        return self.execute_query(query, (limit, offset))

    def count_records(
        self, table: str, where_clause: str = "", params: tuple = ()
    ) -> int:
        # [FUNCTION DOC]
        # Returns COUNT(*) of a table, optionally filtered by a raw where_clause with params; 0 if no result.
        """Count records in a table"""
        query = f"SELECT COUNT(*) as count FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        result = self.execute_query(query, params)
        return result[0]["count"] if result else 0

    def update_record(
        self, table: str, record_id: int, updates: Dict[str, Any]
    ) -> bool:
        # [FUNCTION DOC]
        # Updates the row with the given id using the field->value mapping; returns False if
        # updates is empty, else True if any row changed.
        # Side effects: mutates the caller's updates dict by adding "updated_at" (local-time ISO string).
        # KNOWN ISSUE: always sets updated_at even though the comment says "if the table has this column"; tables without that column will raise an sqlite error.
        """Update a record with given field values"""
        if not updates:
            return False

        # Add updated_at timestamp if the table has this column
        updates["updated_at"] = datetime.now().isoformat()

        set_clauses = []
        params = []

        for field, value in updates.items():
            set_clauses.append(f"{field} = ?")
            params.append(value)

        params.append(record_id)

        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE id = ?"
        affected_rows = self.execute_update(query, tuple(params))
        return affected_rows > 0

    def delete_record(self, table: str, record_id: int) -> bool:
        # [FUNCTION DOC]
        # Deletes the row with the given id; returns True if a row was removed.
        """Delete a record by ID"""
        query = f"DELETE FROM {table} WHERE id = ?"
        affected_rows = self.execute_delete(query, (record_id,))
        return affected_rows > 0

    def table_exists(self, table_name: str) -> bool:
        # [FUNCTION DOC]
        # Returns True if sqlite_master lists a table with the given name.
        """Check if a table exists"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
