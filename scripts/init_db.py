#!/usr/bin/env python
"""Create all database tables (idempotent)."""
from database.models import Base
from database.session import engine
from loguru import logger


def init_db() -> None:
    Base.metadata.create_all(engine)
    logger.success("Tabelle database create (o già esistenti)")


if __name__ == "__main__":
    init_db()
