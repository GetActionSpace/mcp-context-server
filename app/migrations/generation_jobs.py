"""Generation jobs migration for mcp-context-server.

``generation_jobs`` holds one row per context entry whose generated outputs are still
outstanding (``pending``, or ``running`` under a worker's lease) or whose generation gave
up (``failed``); an entry without a row has every output it needs. The entry id is the
primary key and a foreign key onto ``context_entries`` with ``ON DELETE CASCADE``, so an
entry holds at most one job and its job goes with it. ``attempts`` counts the claims of the
job, ``last_error`` keeps the text of its last failure, ``claim_token`` identifies the
claim of a running job, ``created_at`` orders the queue and ``available_at`` is the time
from which a pending job may be claimed or a running job's lease ends. The queue index
``idx_generation_jobs_queue (status, available_at, created_at)`` serves the scan for the
oldest claimable job and the counts per status. Both backends take every timestamp from
the database clock.

Both base schema files declare the table and the queue index with the statements below,
and the startup preparation runs the base schema before any migration, so a fresh
database, an upgraded one and a UUID migration CLI target all hold them, and on those
this migration finds both and writes nothing. It creates whichever of the two a database
lacks when it runs, with the same statements, and runs after every other provisioning
migration and before the request role's grants and the row-level security migration, so
those cover a table it creates on the same start.

Both backends first read, without a lock, whether the table exists and whether the index
exists on it, under the names the bare identifiers resolve to: ``sqlite_master`` on
SQLite, and on PostgreSQL ``to_regclass`` through ``search_path``, the resolution the bare
names of every later statement use. A start that finds both writes nothing. Otherwise one
write reads them again, so a start that waited for a concurrent one builds nothing that
one built, and creates what is missing with ``IF NOT EXISTS``: on SQLite through the write
queue, on PostgreSQL under the migration budget and the shared advisory lock.
"""

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations._pg_ddl import fetch_migration
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GENERATION_JOBS_TABLE = 'generation_jobs'
GENERATION_JOBS_QUEUE_INDEX = 'idx_generation_jobs_queue'

# The statements both base schema files declare; a test keeps the declarations identical to them.
_CREATE_TABLE_SQLITE = f'''
CREATE TABLE IF NOT EXISTS {GENERATION_JOBS_TABLE} (
    context_entry_id TEXT PRIMARY KEY NOT NULL REFERENCES context_entries(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claim_token TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    available_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
'''

_CREATE_TABLE_POSTGRESQL = f'''
CREATE TABLE IF NOT EXISTS {GENERATION_JOBS_TABLE} (
    context_entry_id UUID PRIMARY KEY REFERENCES context_entries(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claim_token TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    available_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
'''

_CREATE_QUEUE_INDEX = (
    f'CREATE INDEX IF NOT EXISTS {GENERATION_JOBS_QUEUE_INDEX} ON {GENERATION_JOBS_TABLE} '
    '(status, available_at, created_at)'
)

# Whether the table exists, and whether the queue index exists on it.
_STATE_SQL_SQLITE = '''
    SELECT
        EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?) AS table_present,
        EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ? AND tbl_name = ?) AS index_present
'''
_STATE_ARGS_SQLITE = (GENERATION_JOBS_TABLE, GENERATION_JOBS_QUEUE_INDEX, GENERATION_JOBS_TABLE)

# Whether the table the bare name resolves to exists, and whether the queue index exists on it.
_STATE_SQL_POSTGRESQL = '''
    SELECT
        to_regclass($1) IS NOT NULL AS table_present,
        EXISTS (
            SELECT 1
            FROM pg_index
            JOIN pg_class AS index_relation ON index_relation.oid = pg_index.indexrelid
            WHERE pg_index.indrelid = to_regclass($1)
              AND index_relation.relname = $2
        ) AS index_present
'''
_STATE_ARGS_POSTGRESQL = (GENERATION_JOBS_TABLE, GENERATION_JOBS_QUEUE_INDEX)


class _StateRow(Protocol):
    """A row read by column name, such as the one :data:`_STATE_SQL_POSTGRESQL` returns."""

    def __getitem__(self, key: str, /) -> object: ...


@dataclass(frozen=True, slots=True)
class _GenerationJobsState:
    """What exists of the generation jobs objects.

    Attributes:
        table_present: Whether the jobs table exists.
        index_present: Whether the queue index exists on the jobs table.
    """

    table_present: bool
    index_present: bool

    @classmethod
    def from_row(cls, row: _StateRow) -> '_GenerationJobsState':
        """Read the state from the one row :data:`_STATE_SQL_POSTGRESQL` returns."""
        return cls(table_present=bool(row['table_present']), index_present=bool(row['index_present']))

    @property
    def complete(self) -> bool:
        """Whether the table and the queue index both exist."""
        return self.table_present and self.index_present

    def missing(self, create_table: str) -> list[tuple[str, str]]:
        """Name every missing object and the statement that creates it, the table before its index.

        Args:
            create_table: The backend's ``CREATE TABLE`` statement for the jobs table.

        Returns:
            The description and the statement of each missing object.
        """
        missing: list[tuple[str, str]] = []
        if not self.table_present:
            missing.append((f'table {GENERATION_JOBS_TABLE}', create_table))
        if not self.index_present:
            missing.append((f'index {GENERATION_JOBS_QUEUE_INDEX}', _CREATE_QUEUE_INDEX))
        return missing


def _read_sqlite_state(conn: sqlite3.Connection) -> _GenerationJobsState:
    """Read the state of the generation jobs objects on a SQLite connection."""
    row = conn.execute(_STATE_SQL_SQLITE, _STATE_ARGS_SQLITE).fetchone()
    return _GenerationJobsState(table_present=bool(row[0]), index_present=bool(row[1]))


async def _probe(backend: StorageBackend) -> _GenerationJobsState:
    """Read the state of the generation jobs objects without a lock.

    Args:
        backend: The backend the migration runs on.

    Returns:
        The state.
    """
    if backend.backend_type == 'sqlite':
        return await backend.execute_read(_read_sqlite_state, access_scope=MAINTENANCE)

    async def _read_postgresql(conn: asyncpg.Connection) -> _GenerationJobsState:
        rows = await conn.fetch(_STATE_SQL_POSTGRESQL, *_STATE_ARGS_POSTGRESQL)
        return _GenerationJobsState.from_row(rows[0])

    return await backend.execute_read(cast(Any, _read_postgresql), access_scope=MAINTENANCE)


async def _build_missing_sqlite(backend: StorageBackend) -> list[str]:
    """Create the objects a probe on the writer connection finds missing.

    Args:
        backend: The SQLite backend.

    Returns:
        The description of each created object; empty when another process created them first.
    """

    def _build(conn: sqlite3.Connection) -> list[str]:
        missing = _read_sqlite_state(conn).missing(_CREATE_TABLE_SQLITE)
        for _description, statement in missing:
            conn.execute(statement)
        return [description for description, _statement in missing]

    return await backend.execute_write(_build, access_scope=MAINTENANCE)


async def _build_missing_postgresql(backend: StorageBackend) -> list[str]:
    """Create the objects a probe under the shared lock finds missing, under the migration budget.

    Args:
        backend: The PostgreSQL maintenance backend.

    Returns:
        The description of each created object; empty when a concurrent start created them
        while this one waited for the lock.
    """
    migration_timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _build(conn: asyncpg.Connection) -> list[str]:
        await begin_migration(conn, migration_timeout_s)
        rows = await fetch_migration(conn, _STATE_SQL_POSTGRESQL, migration_timeout_s, *_STATE_ARGS_POSTGRESQL)
        missing = _GenerationJobsState.from_row(rows[0]).missing(_CREATE_TABLE_POSTGRESQL)
        for _description, statement in missing:
            await execute_migration_ddl(conn, statement, migration_timeout_s)
        return [description for description, _statement in missing]

    return await backend.execute_write(cast(Any, _build), access_scope=MAINTENANCE)


async def apply_generation_jobs_migration(backend: StorageBackend) -> None:
    """Create the generation jobs table and its queue index where they are missing.

    Idempotent on both backends: a start whose probe finds both objects writes nothing and,
    on PostgreSQL, takes no lock.

    Args:
        backend: The backend to migrate: on PostgreSQL the maintenance backend, logged in
            as the table owner.
    """
    if (await _probe(backend)).complete:
        logger.debug('Generation jobs table and index %s present, skipping', GENERATION_JOBS_QUEUE_INDEX)
        return

    if backend.backend_type == 'sqlite':
        created = await _build_missing_sqlite(backend)
    else:
        created = await _build_missing_postgresql(backend)
    if not created:
        logger.debug('Generation jobs table and index built by a concurrent start, nothing to do')
        return
    logger.info('Generation jobs migration applied: created %s', ', '.join(created))
