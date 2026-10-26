"""The generation jobs table and its migration against a real PostgreSQL server.

Every database here is ``pg_rls_db``, owned by the non-superuser ``mcp_owner``, and every
boot prepares it through ``prepare_database`` on a maintenance backend logged in as that
owner, with row-level security off or on. The base schema declares the table and its
queue index, so the tests prove that a fresh database ends with the jobs table, owned by
``mcp_owner``, its constraints and its queue index, with no work left for the migration;
that a database prepared before the table existed gets both at its next boot, its entries
left as they were and none of them holding a job; that a current database takes no lock
and issues no DDL in the migration, while the migration builds a dropped table with its
index, and a dropped index alone; and that the table holds at most one job per stored
entry, accepts the three statuses alone, starts a job pending and claimable, and loses
the job with its entry. Every assertion reads the catalog or the rows through a superuser
or owner connection.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Final

import asyncpg
import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.migrations import apply_generation_jobs_migration
from app.migrations.generation_jobs import GENERATION_JOBS_QUEUE_INDEX
from app.migrations.generation_jobs import GENERATION_JOBS_TABLE
from app.startup.database_setup import prepare_database
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_LOCK_SQL
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_UNLOCK_SQL
from tests.integration.postgresql._rls_catalog import configure_rls_layout
from tests.integration.postgresql._rls_catalog import ddl_log
from tests.integration.postgresql._rls_catalog import install_ddl_log
from tests.integration.postgresql._rls_catalog import maintenance_backend
from tests.integration.postgresql._rls_matrix import REQUEST_ROLE
from tests.integration.postgresql.conftest import RlsUrls

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

_MIGRATION_LOGGER: Final = 'app.migrations.generation_jobs'
_APPLIED: Final = 'Generation jobs migration applied'
_SKIPPED: Final = f'Generation jobs table and index {GENERATION_JOBS_QUEUE_INDEX} present, skipping'

_COLUMNS_SQL: Final = '''
    SELECT attribute.attname, format_type(attribute.atttypid, attribute.atttypmod), attribute.attnotnull,
           pg_get_expr(column_default.adbin, column_default.adrelid)
    FROM pg_attribute AS attribute
    LEFT JOIN pg_attrdef AS column_default
      ON column_default.adrelid = attribute.attrelid AND column_default.adnum = attribute.attnum
    WHERE attribute.attrelid = to_regclass('public.generation_jobs') AND attribute.attnum > 0 AND NOT attribute.attisdropped
    ORDER BY attribute.attnum
'''
# The key, foreign key and check constraints; the NOT NULL flags are read with the columns,
# since only PostgreSQL 18 and later also record them in pg_constraint.
_CONSTRAINTS_SQL: Final = (
    'SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint '
    "WHERE conrelid = to_regclass('public.generation_jobs') AND contype IN ('p', 'f', 'c') ORDER BY conname"
)
_TABLE_OWNER_SQL: Final = "SELECT relowner::regrole::text FROM pg_class WHERE oid = to_regclass('public.generation_jobs')"

# What every prepared database holds: the columns as (name, type, NOT NULL, default), the
# constraints, the queue index and the table owner.
_PREPARED: Final = (
    [
        ('context_entry_id', 'uuid', True, None),
        ('status', 'text', True, "'pending'::text"),
        ('attempts', 'integer', True, '0'),
        ('last_error', 'text', False, None),
        ('claim_token', 'text', False, None),
        ('created_at', 'timestamp with time zone', True, 'now()'),
        ('available_at', 'timestamp with time zone', True, 'now()'),
    ],
    [
        (
            'generation_jobs_context_entry_id_fkey',
            'FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE',
        ),
        ('generation_jobs_pkey', 'PRIMARY KEY (context_entry_id)'),
        (
            'generation_jobs_status_check',
            "CHECK ((status = ANY (ARRAY['pending'::text, 'running'::text, 'failed'::text])))",
        ),
    ],
    (
        f'CREATE INDEX {GENERATION_JOBS_QUEUE_INDEX} ON public.{GENERATION_JOBS_TABLE} '
        'USING btree (status, available_at, created_at)'
    ),
    'mcp_owner',
)

_INSERT_ENTRY: Final = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
    "VALUES ($1, 'generation-jobs', 'agent', 'text', 'generation jobs probe', $2)"
)
_INSERT_JOB: Final = f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id) VALUES ($1)'
_INSERT_JOB_WITH_STATUS: Final = f'INSERT INTO {GENERATION_JOBS_TABLE} (context_entry_id, status) VALUES ($1, $2)'

_ALICE_ENTRY: Final = uuid.UUID('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
_BOB_ENTRY: Final = uuid.UUID('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
_ENTRIES: Final = ((_ALICE_ENTRY, 'alice'), (_BOB_ENTRY, 'bob'))


@pytest_asyncio.fixture
async def admin(pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A superuser connection to the database.

    Yields:
        The open connection, closed afterwards.
    """
    conn = await asyncpg.connect(pg_rls_db.admin)
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def owner(pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A connection of the table owner ``mcp_owner``, for changes made outside the server.

    Yields:
        The open connection, closed afterwards.
    """
    conn = await asyncpg.connect(pg_rls_db.maintenance)
    try:
        yield conn
    finally:
        await conn.close()


async def _boot(urls: RlsUrls, monkeypatch: pytest.MonkeyPatch, *, rls: bool) -> None:
    """Prepare the database once as the server does, with row-level security on or off."""
    settings = configure_rls_layout(urls, monkeypatch, layout='compressed', overrides={}, rls=rls)
    async with maintenance_backend(urls.maintenance) as backend:
        await prepare_database(backend, settings, request_role=REQUEST_ROLE if rls else None)


async def _jobs_objects(conn: asyncpg.Connection) -> tuple[object, ...]:
    """Return the jobs table's columns and constraints, the queue index definition and the table owner."""
    columns = [
        (str(row[0]), str(row[1]), bool(row[2]), None if row[3] is None else str(row[3]))
        for row in await conn.fetch(_COLUMNS_SQL)
    ]
    constraints = [(str(row[0]), str(row[1])) for row in await conn.fetch(_CONSTRAINTS_SQL)]
    index = await conn.fetchval('SELECT pg_get_indexdef(to_regclass($1))', f'public.{GENERATION_JOBS_QUEUE_INDEX}')
    table_owner = await conn.fetchval(_TABLE_OWNER_SQL)
    return columns, constraints, index, table_owner


async def _seed_entries(conn: asyncpg.Connection) -> None:
    """Store the entries of alice and bob through raw SQL."""
    for entry, owner_id in _ENTRIES:
        await conn.execute(_INSERT_ENTRY, entry, owner_id)


def _applied_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith(_APPLIED)]


class TestPreparedDatabases:
    """A fresh database and one prepared before the table existed both end with the table and its index.

    The base schema creates both at every boot, so the migration that follows it finds them and writes nothing.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('rls', [False, True], ids=['rls-off', 'rls-on'])
    async def test_a_fresh_database_gets_the_table_and_its_queue_index_from_the_base_schema(
        self,
        pg_rls_db: RlsUrls,
        monkeypatch: pytest.MonkeyPatch,
        admin: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
        rls: bool,
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger=_MIGRATION_LOGGER):
            await _boot(pg_rls_db, monkeypatch, rls=rls)

        assert await _jobs_objects(admin) == _PREPARED
        assert _applied_messages(caplog) == []
        assert _SKIPPED in caplog.messages

    @pytest.mark.asyncio
    @pytest.mark.parametrize('rls', [False, True], ids=['rls-off', 'rls-on'])
    async def test_a_database_prepared_before_the_table_existed_gets_it_at_its_next_boot_and_keeps_its_entries(
        self,
        pg_rls_db: RlsUrls,
        monkeypatch: pytest.MonkeyPatch,
        admin: asyncpg.Connection,
        owner: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
        rls: bool,
    ) -> None:
        await _boot(pg_rls_db, monkeypatch, rls=rls)
        await owner.execute(f'DROP TABLE {GENERATION_JOBS_TABLE}')
        await _seed_entries(owner)
        entries = await admin.fetch('SELECT id, owner_id, version, updated_at FROM context_entries ORDER BY id')
        caplog.clear()

        with caplog.at_level(logging.DEBUG, logger=_MIGRATION_LOGGER):
            await _boot(pg_rls_db, monkeypatch, rls=rls)

        assert await _jobs_objects(admin) == _PREPARED
        assert _applied_messages(caplog) == []
        assert _SKIPPED in caplog.messages
        assert await admin.fetch('SELECT id, owner_id, version, updated_at FROM context_entries ORDER BY id') == entries
        assert await admin.fetchval(f'SELECT count(*) FROM {GENERATION_JOBS_TABLE}') == 0


class TestLaterBoots:
    """The migration changes nothing on a current database and rebuilds what is missing under the lock."""

    @pytest_asyncio.fixture
    async def booted(self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[StorageBackend]:
        """The database booted once with row-level security off.

        Yields:
            A maintenance backend, logged in as ``mcp_owner``.
        """
        await _boot(pg_rls_db, monkeypatch, rls=False)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            yield backend

    @pytest.mark.asyncio
    async def test_a_current_database_takes_no_lock_and_issues_no_ddl(
        self,
        booted: StorageBackend,
        pg_rls_db: RlsUrls,
        admin: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await install_ddl_log(admin)
        holder = await asyncpg.connect(pg_rls_db.admin)
        await holder.execute(SCHEMA_INIT_LOCK_SQL)
        caplog.clear()
        try:
            with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
                await asyncio.wait_for(apply_generation_jobs_migration(booted), timeout=10)
        finally:
            await holder.execute(SCHEMA_INIT_UNLOCK_SQL)
            await holder.close()

        assert await ddl_log(admin) == []
        assert _applied_messages(caplog) == []

    @pytest.mark.asyncio
    async def test_a_dropped_table_is_built_again_with_its_queue_index(
        self,
        booted: StorageBackend,
        admin: asyncpg.Connection,
        owner: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute(f'DROP TABLE {GENERATION_JOBS_TABLE}')
        await install_ddl_log(admin)
        caplog.clear()

        with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
            await apply_generation_jobs_migration(booted)

        assert await _jobs_objects(admin) == _PREPARED
        assert await ddl_log(admin) == ['CREATE TABLE', 'CREATE INDEX']
        assert _applied_messages(caplog) == [
            f'{_APPLIED}: created table {GENERATION_JOBS_TABLE}, index {GENERATION_JOBS_QUEUE_INDEX}',
        ]

    @pytest.mark.asyncio
    async def test_a_dropped_queue_index_is_built_again_alone(
        self,
        booted: StorageBackend,
        admin: asyncpg.Connection,
        owner: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute(f'DROP INDEX {GENERATION_JOBS_QUEUE_INDEX}')
        assert (await _jobs_objects(admin))[2] is None
        await install_ddl_log(admin)
        caplog.clear()

        with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
            await apply_generation_jobs_migration(booted)

        assert await _jobs_objects(admin) == _PREPARED
        assert await ddl_log(admin) == ['CREATE INDEX']
        assert _applied_messages(caplog) == [f'{_APPLIED}: created index {GENERATION_JOBS_QUEUE_INDEX}']


class TestTable:
    """The constraints of the jobs table on a prepared database, exercised as the table owner."""

    @pytest_asyncio.fixture
    async def seeded(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch, owner: asyncpg.Connection,
    ) -> asyncpg.Connection:
        """Prepare the database with row-level security off and store the entries of alice and bob.

        Returns:
            The table owner's connection.
        """
        await _boot(pg_rls_db, monkeypatch, rls=False)
        await _seed_entries(owner)
        return owner

    @pytest.mark.asyncio
    async def test_a_job_named_by_its_entry_alone_starts_pending_and_claimable(self, seeded: asyncpg.Connection) -> None:
        await seeded.execute(_INSERT_JOB, _ALICE_ENTRY)

        rows = await seeded.fetch(
            'SELECT status, attempts, last_error, claim_token, created_at = available_at, available_at <= now() '
            f'FROM {GENERATION_JOBS_TABLE}',
        )

        assert [tuple(row) for row in rows] == [('pending', 0, None, None, True, True)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', ['pending', 'running', 'failed'])
    async def test_each_job_status_is_accepted(self, seeded: asyncpg.Connection, status: str) -> None:
        await seeded.execute(_INSERT_JOB_WITH_STATUS, _ALICE_ENTRY, status)

        assert await seeded.fetchval(f'SELECT status FROM {GENERATION_JOBS_TABLE}') == status

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', ['ready', 'done', 'PENDING'])
    async def test_any_other_status_is_refused(self, seeded: asyncpg.Connection, status: str) -> None:
        with pytest.raises(asyncpg.CheckViolationError):
            await seeded.execute(_INSERT_JOB_WITH_STATUS, _ALICE_ENTRY, status)

    @pytest.mark.asyncio
    async def test_an_entry_holds_at_most_one_job(self, seeded: asyncpg.Connection) -> None:
        await seeded.execute(_INSERT_JOB, _ALICE_ENTRY)

        with pytest.raises(asyncpg.UniqueViolationError):
            await seeded.execute(_INSERT_JOB, _ALICE_ENTRY)

    @pytest.mark.asyncio
    async def test_a_job_for_a_missing_entry_is_refused(self, seeded: asyncpg.Connection) -> None:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await seeded.execute(_INSERT_JOB, uuid.UUID('ffffffffffffffffffffffffffffffff'))

    @pytest.mark.asyncio
    async def test_deleting_the_entry_deletes_its_job_alone(self, seeded: asyncpg.Connection) -> None:
        await seeded.execute(_INSERT_JOB, _ALICE_ENTRY)
        await seeded.execute(_INSERT_JOB, _BOB_ENTRY)

        await seeded.execute('DELETE FROM context_entries WHERE id = $1', _ALICE_ENTRY)

        assert [row[0] for row in await seeded.fetch(f'SELECT context_entry_id FROM {GENERATION_JOBS_TABLE}')] == [_BOB_ENTRY]
