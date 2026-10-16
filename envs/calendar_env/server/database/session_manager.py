# [MODULE DOC]
# Per-database_id SQLAlchemy engine/session management for calendar SQLite files stored
# under ./mcp_databases/calendar_<id>.sqlite, plus schema introspection of the ORM models.
"""
Calendar Session Manager - Database session handling for Calendar services using SQLAlchemy
"""

import logging
import os
from typing import Dict

from database.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class CalendarSessionManager:
    # [CLASS DOC]
    # Caches one engine and one sessionmaker per database id in instance dicts.
    """Calendar session manager with SQLAlchemy support for calendar services"""

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes empty engine and sessionmaker caches.
        self._engines = {}  # Cache engines per database
        self._session_makers = {}  # Cache session makers per database

    def get_db_path(self, db_id: str) -> str:
        # [FUNCTION DOC]
        # Returns "./mcp_databases/calendar_<db_id>.sqlite". Side effect: creates ./mcp_databases if missing.
        """Get database path for a specific session ID"""
        databases_dir = "./mcp_databases"
        os.makedirs(databases_dir, exist_ok=True)
        return os.path.join(databases_dir, f"calendar_{db_id}.sqlite")

    def get_engine(self, db_id: str):
        # [FUNCTION DOC]
        # Returns the cached engine for db_id, creating a SQLite engine (check_same_thread off,
        # pre-ping, 1h recycle) on first use. Does not create tables.
        """Get SQLAlchemy engine for the specified database"""
        if db_id not in self._engines:
            db_path = self.get_db_path(db_id)
            database_url = f"sqlite:///{db_path}"
            self._engines[db_id] = create_engine(
                database_url,
                echo=False,  # Set to True for SQL logging
                connect_args={"check_same_thread": False},  # Needed for SQLite
                pool_pre_ping=True,  # Verify connections before using them
                pool_recycle=3600,  # Recycle connections after 1 hour
            )
        return self._engines[db_id]

    def get_session_maker(self, db_id: str):
        # [FUNCTION DOC]
        # Returns the cached sessionmaker bound to db_id's engine, creating it on first use.
        """Get SQLAlchemy session maker for the specified database"""
        if db_id not in self._session_makers:
            engine = self.get_engine(db_id)
            self._session_makers[db_id] = sessionmaker(bind=engine)
        return self._session_makers[db_id]

    def get_session(self, db_id: str):
        # [FUNCTION DOC]
        # Returns a new Session for db_id; caller is responsible for closing it.
        """Get SQLAlchemy session for the specified database"""
        session_maker = self.get_session_maker(db_id)
        return session_maker()

    def close_session(self, db_id: str):
        # [FUNCTION DOC]
        # Drops the cached sessionmaker and disposes/removes the cached engine for db_id; logs. Does not delete the file.
        """Close and remove session for the specified database"""
        if db_id in self._session_makers:
            del self._session_makers[db_id]
        if db_id in self._engines:
            self._engines[db_id].dispose()
            del self._engines[db_id]
        logger.info(f"Closed session for database id '{db_id}'")

    def init_database(self, db_id: str, create_tables: bool = False):
        # [FUNCTION DOC]
        # Ensures an engine exists for db_id and, only if create_tables is True, creates all ORM
        # tables. Returns None. Re-raises any error after logging.
        """Initialize the database with SQLAlchemy models"""
        try:
            engine = self.get_engine(db_id)

            # Create all tables
            if create_tables:
                Base.metadata.create_all(engine)

            logger.info(
                f"Calendar database {db_id} initialized successfully with SQLAlchemy"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database {db_id}: {e}")
            raise

    def get_database_schema(self) -> Dict:
        # [FUNCTION DOC]
        # Builds a description of every table in Base.metadata: returns a dict table_name ->
        # {"table_name", "columns": name -> SQL-like definition string (mapped type such as
        # INTEGER/TEXT/BOOLEAN/REAL/DATETIME, PRIMARY KEY, AUTOINCREMENT, UNIQUE, NOT NULL,
        # DEFAULT), "foreign_keys": list of FOREIGN KEY strings, "indexes": list of index strings}.
        # No DB access.
        # KNOWN ISSUE: python_type raises NotImplementedError for some column types rather than being absent, so the hasattr branch may raise; also "AUTOINCREMENT" is appended whenever autoincrement is truthy, including its default "auto" value, and unique indexes render as "UNIQUE INDEX" while others render as "INDEX INDEX".
        """Get the database schema definition dynamically from SQLAlchemy models"""
        from sqlalchemy import inspect

        schema = {}

        # Get all mapped classes (models) from the Base metadata
        for table_name, table in Base.metadata.tables.items():
            table_info = {
                "table_name": table_name,
                "columns": {},
                "foreign_keys": [],
                "indexes": [],
            }

            # Process columns
            for column in table.columns:
                column_def = []

                # Column type
                column_type = str(column.type)
                if hasattr(column.type, "python_type"):
                    if column.type.python_type == int:
                        column_def.append("INTEGER")
                    elif column.type.python_type == str:
                        column_def.append("TEXT")
                    elif column.type.python_type == bool:
                        column_def.append("BOOLEAN")
                    elif column.type.python_type == float:
                        column_def.append("REAL")
                    else:
                        column_def.append(column_type.upper())
                else:
                    # Handle special SQLAlchemy types
                    type_str = str(column.type).upper()
                    if "VARCHAR" in type_str or "STRING" in type_str:
                        column_def.append("TEXT")
                    elif "INTEGER" in type_str:
                        column_def.append("INTEGER")
                    elif "TEXT" in type_str:
                        column_def.append("TEXT")
                    elif "BOOLEAN" in type_str:
                        column_def.append("BOOLEAN")
                    elif "DATETIME" in type_str:
                        column_def.append("DATETIME")
                    else:
                        column_def.append(type_str)

                # Primary key
                if column.primary_key:
                    column_def.append("PRIMARY KEY")
                    if column.autoincrement:
                        column_def.append("AUTOINCREMENT")

                # Unique constraint
                if column.unique:
                    column_def.append("UNIQUE")

                # Not null constraint
                if not column.nullable:
                    column_def.append("NOT NULL")

                # Default value
                if column.default is not None:
                    if hasattr(column.default, "arg"):
                        if callable(column.default.arg):
                            column_def.append("DEFAULT (function)")
                        else:
                            column_def.append(f"DEFAULT '{column.default.arg}'")
                    else:
                        column_def.append(f"DEFAULT '{column.default}'")

                table_info["columns"][column.name] = " ".join(column_def)

            # Process foreign keys
            for fk in table.foreign_keys:
                ref_table = fk.column.table.name
                ref_column = fk.column.name
                local_column = fk.parent.name
                table_info["foreign_keys"].append(
                    f"FOREIGN KEY ({local_column}) REFERENCES {ref_table}({ref_column})"
                )

            # Process indexes
            for index in table.indexes:
                index_columns = [col.name for col in index.columns]
                index_type = "UNIQUE" if index.unique else "INDEX"
                table_info["indexes"].append(
                    f"{index_type} INDEX {index.name} ON {table_name} ({', '.join(index_columns)})"
                )

            schema[table_name] = table_info

        return schema
