"""Tests for the generation jobs migration.

The migration provisions ``generation_jobs``, one row per context entry whose generation
is outstanding or has failed, and its queue index. No base schema declares either, so the
startup preparation creates both on a fresh database and on an upgraded one. The SQLite
tests run the preparation and the migration on real databases and pin the table's
constraints; the PostgreSQL tests pin the statements and drive every branch through a
recording backend, and ``tests/integration/postgresql/test_generation_jobs_migration_postgresql.py``
runs the migration on a live server.
"""

import asyncio
import contextlib
import logging
import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import pytest_asyncio

import app.migrations.generation_jobs as generation_jobs_module
from app.backends import StorageBackend
from app.backends import create_backend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.migrations import apply_generation_jobs_migration
from app.migrations.generation_jobs import GENERATION_JOBS_QUEUE_INDEX
from app.migrations.generation_jobs import GENERATION_JOBS_TABLE
from app.schemas import load_schema
from app.startup import init_database
from tests.repositories._access_scope_layouts import sqlite_scoped_db

_MIGRATION_LOGGER = generation_jobs_module.__name__
_APPLIED = 'Generation jobs migration applied'
_CREATED_BOTH = f'{_APPLIED}: created table {GENERATION_JOBS_TABLE}, index {GENERATION_JOBS_QUEUE_INDEX}'

# PRAGMA table_info rows: (name, type, notnull, default, primary key position).
_SQLITE_COLUMNS = [
    ('context_entry_id', 'TEXT', 1, None, 1),
    ('status', 'TEXT', 1, "'pending'", 0),
    ('attempts', 'INTEGER', 1, '0', 0),
    ('last_error', 'TEXT', 0, None, 0),
    ('claim_token', 'TEXT', 0, None, 0),
    ('created_at', 'TIMESTAMP', 1, 'CURRENT_TIMESTAMP', 0),
    ('available_at', 'TIMESTAMP', 1, 'CURRENT_TIMESTAMP', 0),
]
_ENTRY_ID = '0190abcdef1234567890abcdef123456'
_INSERT_ENTRY = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
    "VALUES (?, 'generation-jobs', 'agent', 'text', 'generation jobs probe', 'alice')"
)


def _normalized(statement: str) -> str:
    return ' '.join(statement.split())


async def _schema_objects(backend: StorageBackend) -> list[tuple[str, str, str | None]]:
    """Return the type, name and stored definition of every schema object of the jobs table."""

    def _read(conn: sqlite3.Connection) -> list[tuple[str, str, str | None]]:
        cursor = conn.execute(
            'SELECT type, name, sql FROM sqlite_master WHERE tbl_name = ? ORDER BY type, name', (GENERATION_JOBS_TABLE,),
        )
        return [(str(row[0]), str(row[1]), None if row[2] is None else str(row[2])) for row in cursor.fetchall()]

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


async def _schema_version(backend: StorageBackend) -> int:
    """Return the schema version, which SQLite advances on every change of the schema."""

    def _read(conn: sqlite3.Connection) -> int:
        return int(conn.execute('PRAGMA schema_version').fetchone()[0])

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


async def _table_shape(backend: StorageBackend) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], list[str]]:
    """Return the columns, the foreign keys and the queue index columns of the jobs table."""

    def _read(conn: sqlite3.Connection) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], list[str]]:
        columns: list[tuple[object, ...]] = [
            (row[1], row[2], row[3], row[4], row[5])
            for row in conn.execute(f'PRAGMA table_info({GENERATION_JOBS_TABLE})').fetchall()
        ]
        foreign_keys: list[tuple[object, ...]] = [
            (row[2], row[3], row[4], row[6])
            for row in conn.execute(f'PRAGMA foreign_key_list({GENERATION_JOBS_TABLE})').fetchall()
        ]
        index_columns = [str(row[2]) for row in conn.execute(f'PRAGMA index_info({GENERATION_JOBS_QUEUE_INDEX})').fetchall()]
        return columns, foreign_keys, index_columns

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


def _applied_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith(_APPLIED)]


@pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
def test_no_base_schema_declares_the_generation_jobs_table(backend_type: str) -> None:
    schema = load_schema(backend_type)

    assert GENERATION_JOBS_TABLE not in schema
    assert GENERATION_JOBS_QUEUE_INDEX not in schema


@pytest.mark.asyncio
async def test_init_database_alone_creates_no_generation_jobs_table(tmp_path: Path) -> None:
    backend = create_backend(backend_type='sqlite', db_path=str(tmp_path / 'base_schema_only.db'))
    await backend.initialize()
    try:
        await init_database(backend=backend)

        assert await _schema_objects(backend) == []
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_prepare_database_creates_the_table_and_the_queue_index(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        columns, foreign_keys, index_columns = await _table_shape(db.backend)

    assert columns == _SQLITE_COLUMNS
    assert foreign_keys == [('context_entries', 'context_entry_id', 'id', 'CASCADE')]
    assert index_columns == ['status', 'available_at', 'created_at']


@pytest.mark.asyncio
async def test_a_database_predating_the_access_columns_boots_with_the_table(
    backend_pre_migration: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with sqlite_scoped_db(backend_pre_migration, 'fp32', monkeypatch) as db:
        columns, _foreign_keys, index_columns = await _table_shape(db.backend)

    assert columns == _SQLITE_COLUMNS
    assert index_columns == ['status', 'available_at', 'created_at']


@pytest.mark.asyncio
async def test_a_first_application_creates_both_objects_and_reports_them(
    backend: StorageBackend, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
        await apply_generation_jobs_migration(backend)

    assert [(kind, name) for kind, name, _sql in await _schema_objects(backend)] == [
        ('index', GENERATION_JOBS_QUEUE_INDEX),
        ('index', f'sqlite_autoindex_{GENERATION_JOBS_TABLE}_1'),
        ('table', GENERATION_JOBS_TABLE),
    ]
    assert _applied_messages(caplog) == [_CREATED_BOTH]


@pytest.mark.asyncio
async def test_a_second_application_writes_nothing_and_reports_nothing(
    backend: StorageBackend, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await apply_generation_jobs_migration(backend)
    objects = await _schema_objects(backend)
    version = await _schema_version(backend)
    writes: list[tuple[object, ...]] = []

    async def _record_write(*args: object, **_kwargs: object) -> None:
        writes.append(args)

    monkeypatch.setattr(backend, 'execute_write', _record_write)
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger=_MIGRATION_LOGGER):
        await apply_generation_jobs_migration(backend)

    assert writes == []
    assert await _schema_objects(backend) == objects
    assert await _schema_version(backend) == version
    assert _applied_messages(caplog) == []


@pytest.mark.asyncio
async def test_a_dropped_queue_index_is_built_again_alone(backend: StorageBackend, caplog: pytest.LogCaptureFixture) -> None:
    await apply_generation_jobs_migration(backend)
    objects = await _schema_objects(backend)

    def _drop_index(conn: sqlite3.Connection) -> None:
        conn.execute(f'DROP INDEX {GENERATION_JOBS_QUEUE_INDEX}')

    await backend.execute_write(_drop_index, access_scope=MAINTENANCE)
    caplog.clear()

    with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
        await apply_generation_jobs_migration(backend)

    assert await _schema_objects(backend) == objects
    assert _applied_messages(caplog) == [f'{_APPLIED}: created index {GENERATION_JOBS_QUEUE_INDEX}']


class TestSqliteTable:
    """The constraints of the jobs table on a migrated SQLite database, read through a plain connection."""

    @pytest_asyncio.fixture
    async def migrated_db(self, tmp_path: Path) -> Path:
        """A database file holding the base schema, one entry and the migrated jobs table.

        Returns:
            The database file, with no backend open on it.
        """
        db_path = tmp_path / 'generation_jobs.db'
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.executescript(load_schema('sqlite'))
            conn.execute(_INSERT_ENTRY, (_ENTRY_ID,))
            conn.commit()
        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            await apply_generation_jobs_migration(backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=5.0)
        return db_path

    @staticmethod
    def _connect(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def test_a_job_named_by_its_entry_alone_starts_pending_and_claimable(self, migrated_db: Path) -> None:
        with contextlib.closing(self._connect(migrated_db)) as conn:
            conn.execute(f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))
            row = conn.execute(
                'SELECT status, attempts, last_error, claim_token, created_at IS NOT NULL, available_at <= CURRENT_TIMESTAMP '
                f'FROM {GENERATION_JOBS_TABLE}',
            ).fetchone()

        assert row == ('pending', 0, None, None, 1, 1)

    @pytest.mark.parametrize('status', ['pending', 'running', 'failed'])
    def test_each_job_status_is_accepted(self, migrated_db: Path, status: str) -> None:
        with contextlib.closing(self._connect(migrated_db)) as conn:
            conn.execute(f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id, status) VALUES (?, ?)', (_ENTRY_ID, status))

            assert conn.execute(f'SELECT status FROM {GENERATION_JOBS_TABLE}').fetchall() == [(status,)]

    @pytest.mark.parametrize('status', ['ready', 'done', 'PENDING'])
    def test_any_other_status_is_refused(self, migrated_db: Path, status: str) -> None:
        with contextlib.closing(self._connect(migrated_db)) as conn, pytest.raises(sqlite3.IntegrityError, match='CHECK'):
            conn.execute(f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id, status) VALUES (?, ?)', (_ENTRY_ID, status))

    def test_an_entry_holds_at_most_one_job(self, migrated_db: Path) -> None:
        with contextlib.closing(self._connect(migrated_db)) as conn:
            conn.execute(f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))

            with pytest.raises(sqlite3.IntegrityError, match='UNIQUE'):
                conn.execute(f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))

    def test_a_job_for_a_missing_entry_is_refused(self, migrated_db: Path) -> None:
        with (
            contextlib.closing(self._connect(migrated_db)) as conn,
            pytest.raises(sqlite3.IntegrityError, match='FOREIGN KEY'),
        ):
            conn.execute(f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id) VALUES (?)', ('f' * 32,))

    def test_deleting_the_entry_deletes_its_job(self, migrated_db: Path) -> None:
        with contextlib.closing(self._connect(migrated_db)) as conn:
            conn.execute(f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))

            conn.execute('DELETE FROM context_entries WHERE id = ?', (_ENTRY_ID,))

            assert conn.execute(f'SELECT count(*) FROM {GENERATION_JOBS_TABLE}').fetchone() == (0,)


_POSTGRESQL_TABLE_DDL = (
    'CREATE TABLE IF NOT EXISTS generation_jobs ( '
    'context_entry_id UUID PRIMARY KEY REFERENCES context_entries(id) ON DELETE CASCADE, '
    "status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'failed')), "
    'attempts INTEGER NOT NULL DEFAULT 0, '
    'last_error TEXT, '
    'claim_token TEXT, '
    'created_at TIMESTAMPTZ NOT NULL DEFAULT now(), '
    'available_at TIMESTAMPTZ NOT NULL DEFAULT now() )'
)
_INDEX_DDL = 'CREATE INDEX IF NOT EXISTS idx_generation_jobs_queue ON generation_jobs (status, available_at, created_at)'
_STATE_ARGS = (GENERATION_JOBS_TABLE, GENERATION_JOBS_QUEUE_INDEX)
_COMPLETE = {'table_present': True, 'index_present': True}
_ABSENT = {'table_present': False, 'index_present': False}


def test_the_postgresql_table_keys_one_job_per_entry_on_the_entry_with_a_cascading_foreign_key() -> None:
    assert _normalized(generation_jobs_module._CREATE_TABLE_POSTGRESQL) == _POSTGRESQL_TABLE_DDL


def test_the_queue_index_orders_by_status_then_availability_then_age_on_both_backends() -> None:
    assert generation_jobs_module._CREATE_QUEUE_INDEX == _INDEX_DDL


def test_the_postgresql_probe_reads_the_table_and_its_index_through_the_search_path() -> None:
    probe = _normalized(generation_jobs_module._STATE_SQL_POSTGRESQL)

    assert 'to_regclass($1) IS NOT NULL AS table_present' in probe
    assert 'WHERE pg_index.indrelid = to_regclass($1) AND index_relation.relname = $2' in probe


class _RecordingPostgreSQLBackend:
    """A PostgreSQL backend whose lock-free probe answers a fixed state and whose writes run on a placeholder connection."""

    backend_type = 'postgresql'

    def __init__(self, state: Mapping[str, object]) -> None:
        self._state = state
        self.probe_args: list[tuple[object, ...]] = []
        self.writes = 0

    async def execute_read(self, operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        assert access_scope is MAINTENANCE
        return await operation(SimpleNamespace(fetch=self._fetch))

    async def execute_write(self, operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        assert access_scope is MAINTENANCE
        self.writes += 1
        return await operation(object())

    async def _fetch(self, sql: str, *args: object) -> list[Mapping[str, object]]:
        assert sql == generation_jobs_module._STATE_SQL_POSTGRESQL
        self.probe_args.append(args)
        return [self._state]


def _record_migration_steps(monkeypatch: pytest.MonkeyPatch, *, under_lock: Mapping[str, object]) -> list[tuple[str, ...]]:
    """Replace the lock, DDL and read helpers of the migration with recorders, under a 42-second budget.

    Args:
        monkeypatch: Restores the helpers after the test.
        under_lock: The state the probe repeated under the lock finds.

    Returns:
        The recorded steps, filled in as the migration runs.
    """
    steps: list[tuple[str, ...]] = []

    async def _begin(_conn: object, timeout_s: float) -> None:
        steps.append(('begin_migration', str(timeout_s)))

    async def _ddl(_conn: object, statement: str, timeout_s: float) -> None:
        steps.append(('ddl', _normalized(statement), str(timeout_s)))

    async def _fetch(_conn: object, statement: str, timeout_s: float, *args: object) -> list[Mapping[str, object]]:
        assert statement == generation_jobs_module._STATE_SQL_POSTGRESQL
        assert args == _STATE_ARGS
        steps.append(('probe under the lock', str(timeout_s)))
        return [under_lock]

    monkeypatch.setattr(generation_jobs_module, 'begin_migration', _begin)
    monkeypatch.setattr(generation_jobs_module, 'execute_migration_ddl', _ddl)
    monkeypatch.setattr(generation_jobs_module, 'fetch_migration', _fetch)
    monkeypatch.setattr(
        generation_jobs_module, 'settings', SimpleNamespace(storage=SimpleNamespace(postgresql_migration_timeout_s=42.0)),
    )
    return steps


async def _migrate(
    monkeypatch: pytest.MonkeyPatch, *, before: Mapping[str, object], under_lock: Mapping[str, object] | None = None,
) -> tuple[_RecordingPostgreSQLBackend, list[tuple[str, ...]]]:
    """Apply the migration to a recording backend in the given state and return the backend and the recorded steps."""
    steps = _record_migration_steps(monkeypatch, under_lock=before if under_lock is None else under_lock)
    backend = _RecordingPostgreSQLBackend(before)

    await apply_generation_jobs_migration(cast(StorageBackend, backend))

    assert backend.probe_args == [_STATE_ARGS]
    return backend, steps


@pytest.mark.asyncio
async def test_postgresql_builds_the_table_and_the_index_under_one_lock_on_a_database_without_them(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
        backend, steps = await _migrate(monkeypatch, before=_ABSENT)

    assert backend.writes == 1
    assert steps == [
        ('begin_migration', '42.0'),
        ('probe under the lock', '42.0'),
        ('ddl', _POSTGRESQL_TABLE_DDL, '42.0'),
        ('ddl', _INDEX_DDL, '42.0'),
    ]
    assert _applied_messages(caplog) == [_CREATED_BOTH]


@pytest.mark.asyncio
async def test_postgresql_builds_only_a_missing_index(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    before = {'table_present': True, 'index_present': False}

    with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
        backend, steps = await _migrate(monkeypatch, before=before)

    assert backend.writes == 1
    assert steps == [('begin_migration', '42.0'), ('probe under the lock', '42.0'), ('ddl', _INDEX_DDL, '42.0')]
    assert _applied_messages(caplog) == [f'{_APPLIED}: created index {GENERATION_JOBS_QUEUE_INDEX}']


@pytest.mark.asyncio
async def test_postgresql_takes_no_lock_and_writes_nothing_when_both_exist(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
        backend, steps = await _migrate(monkeypatch, before=_COMPLETE)

    assert backend.writes == 0
    assert steps == []
    assert _applied_messages(caplog) == []


@pytest.mark.asyncio
async def test_postgresql_builds_nothing_a_concurrent_start_built_while_it_waited(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
        backend, steps = await _migrate(monkeypatch, before=_ABSENT, under_lock=_COMPLETE)

    assert backend.writes == 1
    assert steps == [('begin_migration', '42.0'), ('probe under the lock', '42.0')]
    assert _applied_messages(caplog) == []
