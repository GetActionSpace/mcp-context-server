"""Removing the row-level security objects, against a real PostgreSQL server.

The database is ``pg_rls_db``: booted once with row-level security off, then ``tags``
gets row-level security enabled outside the server, then a boot with
``ENABLE_POSTGRESQL_RLS`` installs the objects. Removal, as the table owner
``mcp_owner``, must report the same thing in a dry run and in a real run, change nothing
in the dry run, and in the real run leave none of the objects behind, disable row-level
security on every table the migration enabled and keep it on ``tags``, reporting it.
Afterwards a boot with the flag off starts. The superuser oracle connection stays open
throughout and does not refuse a removal, since no superuser session counts as a server
instance; ``tests/integration/postgresql/test_rls_cli_postgresql.py`` covers the refusal
while a member is connected.
"""

from collections.abc import AsyncGenerator
from typing import Final

import asyncpg
import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.migrations import RlsRemovalReport
from app.migrations import apply_rls
from app.migrations import guard_rls_disable_over_enabled
from app.migrations import remove_rls
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import parse_fingerprint
from app.settings import get_settings
from app.startup.database_setup import prepare_database
from tests.integration.postgresql._rls_catalog import configure_rls_layout
from tests.integration.postgresql._rls_catalog import maintenance_backend
from tests.integration.postgresql._rls_catalog import object_oids
from tests.integration.postgresql._rls_catalog import policies
from tests.integration.postgresql._rls_catalog import row_security
from tests.integration.postgresql._rls_matrix import rls_settings
from tests.integration.postgresql._rls_urls import RlsUrls

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

SCHEMA_NAME: Final = 'public'
OUR_FUNCTIONS: Final = (
    'mcp_context_principal_id', 'mcp_context_groups', 'mcp_context_may_publish', 'mcp_context_may_write',
    'mcp_context_guard_access_columns', 'mcp_context_reject_view_write',
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
async def installed(
    pg_rls_db: RlsUrls, admin: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[StorageBackend]:
    """The fp32 layout booted with the flag off, ``tags`` enabled out of band, then the objects installed.

    Yields:
        The maintenance backend, logged in as ``mcp_owner``.
    """
    flag_off = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides={}, rls=False)
    async with maintenance_backend(pg_rls_db.maintenance) as backend:
        await prepare_database(backend, flag_off, request_role=None)
        owner = await asyncpg.connect(pg_rls_db.maintenance)
        try:
            await owner.execute('ALTER TABLE tags ENABLE ROW LEVEL SECURITY')
        finally:
            await owner.close()
        await apply_rls(backend, rls_settings())
        assert await admin.fetchval('SELECT to_regclass($1) IS NOT NULL', 'public.context_entries_readable')
        yield backend


async def _preenabled(conn: asyncpg.Connection) -> dict[str, bool | None]:
    """Return what each table's read policy recorded about row-level security before the migration."""
    recorded: dict[str, bool | None] = {}
    for (table, name), policy in (await policies(conn, SCHEMA_NAME)).items():
        if name == f'{table}_read':
            fingerprint = parse_fingerprint(policy.comment)
            recorded[table] = None if fingerprint is None else fingerprint.preenabled
    return recorded


async def _our_functions(conn: asyncpg.Connection) -> list[str]:
    rows = await conn.fetch(
        'SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace '
        'WHERE n.nspname = $1 AND p.proname = ANY($2::text[]) ORDER BY 1',
        SCHEMA_NAME,
        list(OUR_FUNCTIONS),
    )
    return [str(row[0]) for row in rows]


class TestRemoveRls:
    """Removal undoes the migration and nothing else, and a dry run only reports it."""

    @pytest.mark.asyncio
    async def test_the_migration_recorded_which_tables_it_enabled(
        self, installed: StorageBackend, admin: asyncpg.Connection,
    ) -> None:
        _ = installed
        recorded = await _preenabled(admin)

        assert recorded.pop('tags') is True
        assert set(recorded.values()) == {False}

    @pytest.mark.asyncio
    async def test_a_dry_run_reports_what_a_real_run_does_and_changes_nothing(
        self, installed: StorageBackend, admin: asyncpg.Connection,
    ) -> None:
        tables = [table for table in MANAGED_TABLES if table in await row_security(admin, SCHEMA_NAME)]
        before = await object_oids(admin, SCHEMA_NAME)
        assert ('context_entry_grants', 'context_entry_grants_delete') in await policies(admin, SCHEMA_NAME)
        assert ('generation_jobs', 'generation_jobs_read') in await policies(admin, SCHEMA_NAME)
        assert await _our_functions(admin) == sorted(OUR_FUNCTIONS)

        dry = await remove_rls(installed, rls_settings(), dry_run=True, force=False)

        assert await object_oids(admin, SCHEMA_NAME) == before
        assert dry.disabled == [table for table in tables if table != 'tags']
        assert 'generation_jobs' in dry.disabled
        assert dry.left_enabled == {'tags': 'preenabled=true'}
        assert any('DROP VIEW' in statement for statement in dry.statements)

        real = await remove_rls(installed, rls_settings(), dry_run=False, force=False)

        assert real == dry
        assert await policies(admin, SCHEMA_NAME) == {}
        assert await _our_functions(admin) == []
        assert await admin.fetchval('SELECT to_regclass($1)', 'public.context_entries_readable') is None
        assert await admin.fetchval(
            "SELECT count(*) FROM pg_trigger WHERE tgname = 'guard_context_entry_access_columns'",
        ) == 0
        enabled = await row_security(admin, SCHEMA_NAME)
        assert {table: enabled[table][0] for table in tables} == {table: table == 'tags' for table in tables}

    @pytest.mark.asyncio
    async def test_after_removal_nothing_is_left_and_the_flag_off_starts(
        self, installed: StorageBackend, admin: asyncpg.Connection,
    ) -> None:
        await remove_rls(installed, rls_settings(), dry_run=False, force=False)

        again = await remove_rls(installed, rls_settings(), dry_run=False, force=False)

        assert again == RlsRemovalReport(statements=[], disabled=[], left_enabled={'tags': 'no readable comment'})
        flag_off = get_settings()
        assert not flag_off.access_control.postgresql_rls_enabled
        await guard_rls_disable_over_enabled(installed, flag_off)
        await prepare_database(installed, flag_off, request_role=None)
        assert await _our_functions(admin) == []
