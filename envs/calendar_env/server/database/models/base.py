# [MODULE DOC]
# Defines Base, the single SQLAlchemy declarative base shared by all calendar ORM models.
# Importing it creates the declarative registry (in-memory metadata only, no DB I/O).
"""
Base database model
"""

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
