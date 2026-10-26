"""The grant owner migration against a real PostgreSQL server.

Every database here is ``pg_rls_db``, owned by the non-superuser ``mcp_owner``, and every
boot prepares it through ``prepare_database`` on a maintenance backend logged in as that
owner, with row-level security off or on. The tests prove that a fresh database, one that
predates the access-control columns and one whose grants predate the owner column all
end with the owner column NOT NULL, the unique index on ``context_entries (id, owner_id)``
and the composite foreign key in place of the single-column one; that the grants stored
before the column get their entry's owner and the boot reports how many; that a current
database takes no lock and issues no DDL, while a missing foreign key is rebuilt; and that
reassigning an owner with ``mcp-context-server-migrate --reassign-owner`` cascades into
the grants of the moved entries and leaves their grantor as it was. Every assertion reads
the catalog or the rows through a superuser or owner connection.
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
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.cli.migrate_reassign_owner import run_reassign_owner
from app.migrations.grant_owner import ENTRY_OWNER_INDEX
from app.migrations.grant_owner import GRANT_OWNER_COLUMN
from app.migrations.grant_owner import GRANT_OWNER_FOREIGN_KEY
from app.migrations.grant_owner import apply_grant_owner_migration
from app.schemas import load_schema
from app.startup.database_setup import prepare_database
from tests.helpers import insert_grant
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_LOCK_SQL
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_UNLOCK_SQL
from tests.integration.postgresql._rls_catalog import configure_rls_layout
from tests.integration.postgresql._rls_catalog import ddl_log
from tests.integration.postgresql._rls_catalog import install_ddl_log
from tests.integration.postgresql._rls_catalog import maintenance_backend
from tests.integration.postgresql._rls_matrix import REQUEST_ROLE
from tests.integration.postgresql._rls_urls import RlsUrls

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

_MIGRATION_LOGGER: Final = 'app.migrations.grant_owner'
_FOREIGN_KEY_DEFINITION: Final = (
    f'FOREIGN KEY (context_entry_id, {GRANT_OWNER_COLUMN}) REFERENCES context_entries(id, owner_id) '
    'ON UPDATE CASCADE ON DELETE CASCADE'
)
_INDEX_DEFINITION: Final = f'CREATE UNIQUE INDEX {ENTRY_OWNER_INDEX} ON public.context_entries USING btree (id, owner_id)'
# What every prepared database holds: the NOT NULL text column, the unique index, and the
# composite foreign key as the only foreign key of the grants table.
_PREPARED: Final = (('text', True), _INDEX_DEFINITION, [(GRANT_OWNER_FOREIGN_KEY, _FOREIGN_KEY_DEFINITION)])

# The PostgreSQL context_entries columns without owner_id and visibility: the shape of a
# database created before the access-control columns joined the base schema.
_PRE_ACCESS_CONTROL_CONTEXT_ENTRIES_DDL: Final = '''
    CREATE TABLE context_entries (
        id UUID NOT NULL PRIMARY KEY,
        thread_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
        content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
        text_content TEXT,
        metadata JSONB,
        summary TEXT,
        content_hash TEXT,
        version BIGINT NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
'''

_INSERT_ENTRY: Final = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
    "VALUES ($1, 'grant-owner', 'agent', 'text', 'grant owner probe', $2)"
)
# The grant row as it is written before the owner column exists.
_INSERT_GRANT_WITHOUT_OWNER: Final = (
    'INSERT INTO context_entry_grants (context_entry_id, principal_type, principal_id, permission, granted_by) '
    'VALUES ($1, $2, $3, $4, $5)'
)
_GRANT_ROWS_SQL: Final = (
    f'SELECT context_entry_id, principal_id, permission, granted_by, {GRANT_OWNER_COLUMN} FROM context_entry_grants '
    'ORDER BY context_entry_id, principal_id, permission'
)

_ALICE_ENTRY: Final = uuid.UUID('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
_ALICE_OTHER_ENTRY: Final = uuid.UUID('abababababababababababababababab')
_BOB_ENTRY: Final = uuid.UUID('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
_ENTRIES: Final = ((_ALICE_ENTRY, 'alice'), (_ALICE_OTHER_ENTRY, 'alice'), (_BOB_ENTRY, 'bob'))
# (entry, grantee, permission, grantor)
_GRANTS: Final = (
    (_ALICE_ENTRY, 'bob', 'read', 'alice'),
    (_ALICE_ENTRY, 'carol', 'write', 'alice'),
    (_ALICE_OTHER_ENTRY, 'dave', 'read', 'alice'),
    (_BOB_ENTRY, 'alice', 'read', 'bob'),
)


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


async def _grant_owner_objects(conn: asyncpg.Connection) -> tuple[object, ...]:
    """Return the owner column's type and NOT NULL flag, the index definition and every foreign key of the grants table."""
    column = await conn.fetchrow(
        'SELECT format_type(atttypid, atttypmod), attnotnull FROM pg_attribute '
        "WHERE attrelid = to_regclass('public.context_entry_grants') AND attname = $1 AND NOT attisdropped",
        GRANT_OWNER_COLUMN,
    )
    index = await conn.fetchval('SELECT pg_get_indexdef(to_regclass($1))', f'public.{ENTRY_OWNER_INDEX}')
    foreign_keys = await conn.fetch(
        'SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint '
        "WHERE conrelid = to_regclass('public.context_entry_grants') AND contype = 'f' ORDER BY conname",
    )
    return (
        None if column is None else (str(column[0]), bool(column[1])),
        index,
        [(str(row[0]), str(row[1])) for row in foreign_keys],
    )


async def _seed_entries(conn: asyncpg.Connection) -> None:
    """Store the entries of alice and bob through raw SQL."""
    for entry, owner_id in _ENTRIES:
        await conn.execute(_INSERT_ENTRY, entry, owner_id)


async def _grant_rows(conn: asyncpg.Connection) -> list[tuple[str, str, str, str, str]]:
    """Return every grant as (entry hex, grantee, permission, grantor, entry owner)."""
    rows = await conn.fetch(_GRANT_ROWS_SQL)
    return [(row[0].hex, str(row[1]), str(row[2]), str(row[3]), str(row[4])) for row in rows]


def _expected_rows(owners: dict[uuid.UUID, str]) -> list[tuple[str, str, str, str, str]]:
    """The seeded grants, each recording ``owners``' owner of its entry and its original grantor."""
    return sorted((entry.hex, grantee, permission, grantor, owners[entry]) for entry, grantee, permission, grantor in _GRANTS)


class TestPreparedDatabases:
    """Every path to a prepared database ends with the same column, index and foreign key."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('rls', [False, True], ids=['rls-off', 'rls-on'])
    async def test_a_fresh_database_gets_the_owner_column_index_and_foreign_key(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch, admin: asyncpg.Connection, rls: bool,
    ) -> None:
        await _boot(pg_rls_db, monkeypatch, rls=rls)

        assert await _grant_owner_objects(admin) == _PREPARED

    @pytest.mark.asyncio
    @pytest.mark.parametrize('rls', [False, True], ids=['rls-off', 'rls-on'])
    async def test_a_database_predating_the_access_columns_boots_with_them(
        self,
        pg_rls_db: RlsUrls,
        monkeypatch: pytest.MonkeyPatch,
        admin: asyncpg.Connection,
        owner: asyncpg.Connection,
        rls: bool,
    ) -> None:
        """The base schema runs over a table without owner_id, so only the migration may name it."""
        await owner.execute(_PRE_ACCESS_CONTROL_CONTEXT_ENTRIES_DDL)

        await _boot(pg_rls_db, monkeypatch, rls=rls)

        assert await _grant_owner_objects(admin) == _PREPARED

    @pytest.mark.asyncio
    async def test_grants_stored_before_the_column_get_their_entry_owner(
        self,
        pg_rls_db: RlsUrls,
        monkeypatch: pytest.MonkeyPatch,
        admin: asyncpg.Connection,
        owner: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute(load_schema('postgresql').replace('{SCHEMA}', quote_pg_identifier('public')))
        await _seed_entries(owner)
        for entry, grantee, permission, grantor in _GRANTS:
            await owner.execute(_INSERT_GRANT_WITHOUT_OWNER, entry, 'user', grantee, permission, grantor)
        assert (await _grant_owner_objects(admin))[0] is None

        with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
            await _boot(pg_rls_db, monkeypatch, rls=False)

        assert f'Grant owner migration applied: {len(_GRANTS)} grants backfilled' in caplog.messages
        assert await _grant_rows(admin) == _expected_rows(dict(_ENTRIES))
        assert await _grant_owner_objects(admin) == _PREPARED


class TestLaterBoots:
    """A boot over a current database changes nothing; a missing object is rebuilt under the lock."""

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
        try:
            with caplog.at_level(logging.INFO, logger=_MIGRATION_LOGGER):
                await asyncio.wait_for(apply_grant_owner_migration(booted), timeout=10)
        finally:
            await holder.execute(SCHEMA_INIT_UNLOCK_SQL)
            await holder.close()

        assert await ddl_log(admin) == []
        assert not [message for message in caplog.messages if message.startswith('Grant owner migration applied')]

    @pytest.mark.asyncio
    async def test_a_dropped_foreign_key_is_rebuilt(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection,
    ) -> None:
        await owner.execute(f'ALTER TABLE context_entry_grants DROP CONSTRAINT {GRANT_OWNER_FOREIGN_KEY}')
        assert (await _grant_owner_objects(admin))[2] == []

        await apply_grant_owner_migration(booted)

        assert await _grant_owner_objects(admin) == _PREPARED


class TestReassignedOwner:
    """The foreign key carries an ownership change into the grants of every moved entry."""

    @pytest_asyncio.fixture
    async def seeded_url(self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch, owner: asyncpg.Connection) -> str:
        """Prepare the database, then store the entries and their grants as the table owner.

        Returns:
            The table owner's connection string, which the CLI is given.
        """
        await _boot(pg_rls_db, monkeypatch, rls=False)
        await _seed_entries(owner)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            for entry, grantee, permission, grantor in _GRANTS:
                await insert_grant(backend, entry.hex, 'user', grantee, permission, grantor)
        return pg_rls_db.maintenance

    def test_reassign_owner_cascades_into_the_grants_and_keeps_their_grantor(self, seeded_url: str) -> None:
        assert asyncio.run(self._grant_rows(seeded_url)) == _expected_rows(dict(_ENTRIES))

        assert run_reassign_owner(seeded_url, 'alice', 'erin', dry_run=False) == 0

        moved = {entry: ('erin' if owner_id == 'alice' else owner_id) for entry, owner_id in _ENTRIES}
        assert asyncio.run(self._grant_rows(seeded_url)) == _expected_rows(moved)

    @staticmethod
    async def _grant_rows(url: str) -> list[tuple[str, str, str, str, str]]:
        conn = await asyncpg.connect(url)
        try:
            return await _grant_rows(conn)
        finally:
            await conn.close()
