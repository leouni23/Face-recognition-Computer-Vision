from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings

_settings = get_settings()

_kwargs = {"pool_pre_ping": True}
if _settings.database_url.startswith("sqlite"):
    _kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(_settings.database_url, **_kwargs)

if _settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def migrate_schema() -> None:
    """Idempotent lightweight migrations for the existing SQLite DB (no Alembic).
    `Base.metadata.create_all` adds new TABLES but never new COLUMNS, so add them here."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "face_templates" not in insp.get_table_names():
        return  # fresh DB → create_all already built the full schema
    cols = {c["name"] for c in insp.get_columns("face_templates")}
    if "model_pack" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE face_templates ADD COLUMN model_pack VARCHAR(32)"))


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
