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
