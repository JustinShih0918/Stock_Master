"""Database engine and idempotent writes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache

from sqlalchemy import Engine, Table, create_engine

from stockmaster.settings import get_settings

CHUNK = 2000


def sqlalchemy_url(url: str | None = None) -> str:
    """DATABASE_URL with the psycopg 3 driver (``postgresql://`` -> ``postgresql+psycopg://``)."""
    url = url or get_settings().database_url
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    return create_engine(sqlalchemy_url(url), pool_pre_ping=True, future=True)


def _insert(engine: Engine, table: Table):
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # pragma: no cover - only postgres (production) and sqlite (tests) are supported
        raise NotImplementedError(engine.dialect.name)
    return insert(table)


def upsert(engine: Engine, table: Table, rows: Iterable[Mapping], update: bool = True,
           index_elements: Sequence[str] | None = None, keep: Sequence[str] = ()) -> int:
    """Insert rows; on primary-key conflict update the non-key columns (or skip, ``update=False``).

    Columns missing from a row, and columns named in ``keep`` (e.g. first-seen timestamps), are left
    untouched on update. Every row must have the same keys. Returns the number of rows sent."""
    rows = [dict(r) for r in rows]
    if not rows:
        return 0
    keys = list(index_elements or [c.name for c in table.primary_key.columns])
    n = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i:i + CHUNK]
            stmt = _insert(engine, table).values(chunk)
            cols = [c for c in chunk[0] if c not in keys and c not in keep]
            if update and cols:
                stmt = stmt.on_conflict_do_update(index_elements=keys,
                                                  set_={c: stmt.excluded[c] for c in cols})
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=keys)
            conn.execute(stmt)
            n += len(chunk)
    return n
