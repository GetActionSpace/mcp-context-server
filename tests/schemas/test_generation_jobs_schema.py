"""Tests for the generation jobs table both base schema files declare.

``generation_jobs`` holds one row per context entry whose generation is outstanding or has
failed, and its queue index serves the claim scan and the counts. The base schema is the
only provisioner of the two: the startup preparation runs it at every start, so its
``IF NOT EXISTS`` statements create them on a fresh database and on one a release without
them prepared, and leave them as they are on every later start. The tests pin the two
declarations against each other, run the base schema and the startup preparation on real
SQLite databases, and pin the table's constraints;
``tests/integration/postgresql/test_generation_jobs_schema_postgresql.py`` proves the same
on a live PostgreSQL server.
"""

import contextlib
import sqlite3
from pathlib import Path

import pytest

from app.backends import StorageBackend
from app.backends import create_backend
from app.backends.base import MAINTENANCE
from app.schemas import load_schema
from app.settings import get_settings
from app.startup import init_database
from app.startup.database_setup import prepare_database
from tests.repositories._access_scope_layouts import sqlite_scoped_db

_TABLE = 'generation_jobs'
_QUEUE_INDEX = 'idx_generation_jobs_queue'

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
# PRAGMA foreign_key_list rows: (parent table, column, parent column, on delete).
_SQLITE_FOREIGN_KEYS = [('context_entries', 'context_entry_id', 'id', 'CASCADE')]
_QUEUE_INDEX_COLUMNS = ['status', 'available_at', 'created_at']
_SQLITE_SHAPE = (_SQLITE_COLUMNS, _SQLITE_FOREIGN_KEYS, _QUEUE_INDEX_COLUMNS)
# The schema objects of the jobs table as (type, name): the queue index, the primary key's index, the table.
_SQLITE_OBJECTS = [
    ('index', _QUEUE_INDEX),
    ('index', f'sqlite_autoindex_{_TABLE}_1'),
    ('table', _TABLE),
]
_QUEUE_INDEX_DDL = f'CREATE INDEX IF NOT EXISTS {_QUEUE_INDEX} ON {_TABLE} (status, available_at, created_at)'
# How the PostgreSQL declaration spells what the SQLite one spells otherwise: the entry id
# type, whose primary key implies NOT NULL there, and the timestamp type and clock.
_SQLITE_SPELLING = {
    'UUID PRIMARY KEY': 'TEXT PRIMARY KEY NOT NULL',
    'TIMESTAMPTZ': 'TIMESTAMP',
    'now()': 'CURRENT_TIMESTAMP',
}
_ENTRY_ID = '0190abcdef1234567890abcdef123456'
_INSERT_ENTRY = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
    "VALUES (?, 'generation-jobs', 'agent', 'text', 'generation jobs probe', 'alice')"
)
_ENTRY_ROW_SQL = 'SELECT id, thread_id, owner_id, version, updated_at FROM context_entries WHERE id = ?'


def _compact(statement: str) -> str:
    """Return ``statement`` with every whitespace character removed, so layout never decides a comparison."""
    return ''.join(statement.split())


def _declared(schema: str, head: str) -> list[str]:
    """Return every statement of ``schema`` that starts with ``head``, comment lines removed and whitespace normalized.

    Args:
        schema: The text of a base schema file.
        head: The opening words of the statements to return.

    Returns:
        The matching statements, each with its whitespace runs collapsed to single spaces,
        in the order the file declares them.
    """
    code = '\n'.join(line for line in schema.splitlines() if not line.strip().startswith('--'))
    return [' '.join(statement.split()) for statement in code.split(';') if _compact(statement).startswith(_compact(head))]


def _in_sqlite_spelling(statement: str) -> str:
    """Return the PostgreSQL ``statement`` with each dialect spelling replaced by the SQLite one."""
    for postgresql, sqlite in _SQLITE_SPELLING.items():
        statement = statement.replace(postgresql, sqlite)
    return statement


async def _catalog_rows(backend: StorageBackend) -> list[tuple[object, ...]]:
    """Return the type, name, root page and SQL of every schema object of the jobs table."""

    def _read(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
        cursor = conn.execute(
            'SELECT type, name, rootpage, sql FROM sqlite_master WHERE tbl_name = ? ORDER BY type, name', (_TABLE,),
        )
        return [tuple(row) for row in cursor.fetchall()]

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


async def _schema_objects(backend: StorageBackend) -> list[tuple[object, ...]]:
    """Return the type and name of every schema object of the jobs table."""
    return [(row[0], row[1]) for row in await _catalog_rows(backend)]


async def _table_shape(backend: StorageBackend) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], list[str]]:
    """Return the columns, the foreign keys and the queue index columns of the jobs table."""

    def _read(conn: sqlite3.Connection) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], list[str]]:
        columns: list[tuple[object, ...]] = [
            (row[1], row[2], row[3], row[4], row[5]) for row in conn.execute(f'PRAGMA table_info({_TABLE})').fetchall()
        ]
        foreign_keys: list[tuple[object, ...]] = [
            (row[2], row[3], row[4], row[6]) for row in conn.execute(f'PRAGMA foreign_key_list({_TABLE})').fetchall()
        ]
        index_columns = [str(row[2]) for row in conn.execute(f'PRAGMA index_info({_QUEUE_INDEX})').fetchall()]
        return columns, foreign_keys, index_columns

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


async def _rows(backend: StorageBackend, sql: str, *args: object) -> list[tuple[object, ...]]:
    """Return every row ``sql`` reads with ``args`` bound."""

    def _read(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
        return [tuple(row) for row in conn.execute(sql, args).fetchall()]

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


def test_both_base_schemas_declare_one_jobs_table_identical_apart_from_the_dialect_spellings() -> None:
    sqlite_tables = _declared(load_schema('sqlite'), f'CREATE TABLE IF NOT EXISTS {_TABLE} (')
    postgresql_tables = _declared(load_schema('postgresql'), f'CREATE TABLE IF NOT EXISTS {_TABLE} (')

    assert len(sqlite_tables) == 1
    assert len(postgresql_tables) == 1
    assert _compact(_in_sqlite_spelling(postgresql_tables[0])) == _compact(sqlite_tables[0])


@pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
def test_each_base_schema_declares_the_queue_index_by_status_then_availability_then_age(backend_type: str) -> None:
    indexes = _declared(load_schema(backend_type), f'CREATE INDEX IF NOT EXISTS {_QUEUE_INDEX} ON')

    assert [_compact(index) for index in indexes] == [_compact(_QUEUE_INDEX_DDL)]


@pytest.mark.asyncio
async def test_the_base_schema_alone_creates_the_table_and_the_queue_index_on_a_fresh_database(tmp_path: Path) -> None:
    backend = create_backend(backend_type='sqlite', db_path=str(tmp_path / 'base_schema_only.db'))
    await backend.initialize()
    try:
        await init_database(backend=backend)

        assert await _table_shape(backend) == _SQLITE_SHAPE
        assert await _schema_objects(backend) == _SQLITE_OBJECTS
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_a_fresh_database_prepared_as_the_server_does_holds_the_table_and_the_queue_index(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        assert await _table_shape(db.backend) == _SQLITE_SHAPE
        assert await _schema_objects(db.backend) == _SQLITE_OBJECTS


@pytest.mark.asyncio
async def test_a_second_start_leaves_the_table_its_queue_index_and_its_jobs_as_they_were(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        entry_id = next(iter(db.ids.values()))

        def _store_job(conn: sqlite3.Connection) -> None:
            conn.execute(f"INSERT INTO {_TABLE} (context_entry_id, status, attempts) VALUES (?, 'failed', 2)", (entry_id,))

        await db.backend.execute_write(_store_job, access_scope=MAINTENANCE)
        catalog = await _catalog_rows(db.backend)
        jobs = await _rows(db.backend, f'SELECT * FROM {_TABLE}')

        await prepare_database(db.backend, get_settings(), request_role=None)

        assert await _catalog_rows(db.backend) == catalog
        assert await _rows(db.backend, f'SELECT * FROM {_TABLE}') == jobs
        assert [row[:3] for row in jobs] == [(entry_id, 'failed', 2)]


@pytest.mark.asyncio
async def test_a_database_prepared_by_a_release_without_the_table_gets_it_at_its_next_start_and_keeps_its_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / 'prepared_without_the_table.db'
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(load_schema('sqlite'))
        conn.execute(f'DROP TABLE {_TABLE}')
        conn.execute(_INSERT_ENTRY, (_ENTRY_ID,))
        conn.commit()
        entry = [tuple(row) for row in conn.execute(_ENTRY_ROW_SQL, (_ENTRY_ID,)).fetchall()]
    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    try:
        assert await _schema_objects(backend) == []

        async with sqlite_scoped_db(backend, 'fp32', monkeypatch) as db:
            assert await _table_shape(db.backend) == _SQLITE_SHAPE
            assert await _schema_objects(db.backend) == _SQLITE_OBJECTS
            assert await _rows(db.backend, f'SELECT count(*) FROM {_TABLE}') == [(0,)]
            assert await _rows(db.backend, _ENTRY_ROW_SQL, _ENTRY_ID) == entry
    finally:
        await backend.shutdown()


class TestSqliteTable:
    """The constraints of the jobs table the base schema declares, read through a plain connection."""

    @pytest.fixture
    def prepared_db(self, tmp_path: Path) -> Path:
        """A database file holding the base schema and one entry.

        Returns:
            The database file, with no connection open on it.
        """
        db_path = tmp_path / 'generation_jobs.db'
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.executescript(load_schema('sqlite'))
            conn.execute(_INSERT_ENTRY, (_ENTRY_ID,))
            conn.commit()
        return db_path

    @staticmethod
    def _connect(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def test_a_job_named_by_its_entry_alone_starts_pending_and_claimable(self, prepared_db: Path) -> None:
        with contextlib.closing(self._connect(prepared_db)) as conn:
            conn.execute(f'INSERT INTO {_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))
            row = conn.execute(
                'SELECT status, attempts, last_error, claim_token, created_at IS NOT NULL, available_at <= CURRENT_TIMESTAMP '
                f'FROM {_TABLE}',
            ).fetchone()

        assert row == ('pending', 0, None, None, 1, 1)

    @pytest.mark.parametrize('status', ['pending', 'running', 'failed'])
    def test_each_job_status_is_accepted(self, prepared_db: Path, status: str) -> None:
        with contextlib.closing(self._connect(prepared_db)) as conn:
            conn.execute(f'INSERT INTO {_TABLE} (context_entry_id, status) VALUES (?, ?)', (_ENTRY_ID, status))

            assert conn.execute(f'SELECT status FROM {_TABLE}').fetchall() == [(status,)]

    @pytest.mark.parametrize('status', ['ready', 'done', 'PENDING'])
    def test_any_other_status_is_refused(self, prepared_db: Path, status: str) -> None:
        with contextlib.closing(self._connect(prepared_db)) as conn, pytest.raises(sqlite3.IntegrityError, match='CHECK'):
            conn.execute(f'INSERT INTO {_TABLE} (context_entry_id, status) VALUES (?, ?)', (_ENTRY_ID, status))

    def test_an_entry_holds_at_most_one_job(self, prepared_db: Path) -> None:
        with contextlib.closing(self._connect(prepared_db)) as conn:
            conn.execute(f'INSERT INTO {_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))

            with pytest.raises(sqlite3.IntegrityError, match='UNIQUE'):
                conn.execute(f'INSERT INTO {_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))

    def test_a_job_for_a_missing_entry_is_refused(self, prepared_db: Path) -> None:
        with (
            contextlib.closing(self._connect(prepared_db)) as conn,
            pytest.raises(sqlite3.IntegrityError, match='FOREIGN KEY'),
        ):
            conn.execute(f'INSERT INTO {_TABLE} (context_entry_id) VALUES (?)', ('f' * 32,))

    def test_deleting_the_entry_deletes_its_job(self, prepared_db: Path) -> None:
        with contextlib.closing(self._connect(prepared_db)) as conn:
            conn.execute(f'INSERT INTO {_TABLE} (context_entry_id) VALUES (?)', (_ENTRY_ID,))

            conn.execute('DELETE FROM context_entries WHERE id = ?', (_ENTRY_ID,))

            assert conn.execute(f'SELECT count(*) FROM {_TABLE}').fetchone() == (0,)
