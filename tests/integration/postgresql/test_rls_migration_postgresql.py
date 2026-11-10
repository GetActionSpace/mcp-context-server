"""The row-level security migration across boots, against a real PostgreSQL server.

The database is ``pg_rls_db``, owned by the non-superuser ``mcp_owner``, which every boot
here prepares through ``prepare_database`` on a maintenance backend, as the server does
with ``ENABLE_POSTGRESQL_RLS``. These tests prove that a boot over a current database
issues no DDL and takes no lock, while a difference makes the migration wait for the
schema-init lock and repair only that; that the fp32 and compressed layouts, a database
without embedding storage and a node-summary table switched on later each end with
policies on exactly the tables that exist; that a new policy role re-creates every
policy and the read view while a new member login role changes nothing; that an object
altered outside the server, a column added to ``context_entries`` and an extra grantee
on the view are repaired and named; that the policies and helper an earlier release
renders differently or lacks are brought up to date without a WARNING, and that the
generation jobs table an earlier release lacked gets its policies and grants at the boot
that creates it; that the FTS language rebuild re-creates the view in its own transaction;
and that a boot with the flag off over installed objects is refused.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Final

import asyncpg
import pytest
import pytest_asyncio

from app.access_scope import AccessMode
from app.access_scope import build_policy_predicate
from app.access_scope import policy_operands
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.errors import ConfigurationError
from app.migrations import apply_request_role_grants
from app.migrations import apply_rls
from app.migrations import guard_rls_disable_over_enabled
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import parse_fingerprint
from app.migrations.rls_ddl import policy_names
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import records_preenabled
from app.repositories import RepositoryContainer
from app.settings import get_settings
from app.startup.database_setup import prepare_database
from tests.integration.postgresql._rls_catalog import ADVISORY_WAITERS_SQL
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_LOCK_SQL
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_UNLOCK_SQL
from tests.integration.postgresql._rls_catalog import configure_rls_layout
from tests.integration.postgresql._rls_catalog import ddl_log
from tests.integration.postgresql._rls_catalog import install_ddl_log
from tests.integration.postgresql._rls_catalog import maintenance_backend
from tests.integration.postgresql._rls_catalog import object_oids
from tests.integration.postgresql._rls_catalog import policies
from tests.integration.postgresql._rls_catalog import row_security
from tests.integration.postgresql._rls_catalog import view_acl
from tests.integration.postgresql._rls_catalog import view_columns
from tests.integration.postgresql._rls_matrix import POLICY_ROLE
from tests.integration.postgresql._rls_matrix import REQUEST_ROLE
from tests.integration.postgresql._rls_matrix import rls_settings
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import Layout

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

SCHEMA_NAME: Final = 'public'
SCHEMA: Final = '"public"'
VIEW: Final = f'{SCHEMA}.context_entries_readable'
ALT_POLICY_ROLE: Final = 'mcp_rls_alt_policy'
ALT_LOGIN_ROLE: Final = 'mcp_rls_alt_login'
JOBS_TABLE: Final = f'{SCHEMA}.generation_jobs'
JOB_PRIVILEGES: Final = ('SELECT', 'INSERT', 'UPDATE', 'DELETE')
INSERT_PUBLIC_ENTRY: Final = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
    "VALUES ($1, 'rls-migration', 'agent', 'text', 'migration probe', 'alice', 'public')"
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


@pytest_asyncio.fixture
async def booted(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[StorageBackend]:
    """The fp32 layout booted once with row-level security on, the policy role granted what requests need.

    Yields:
        The maintenance backend, logged in as ``mcp_owner``.
    """
    settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides={}, rls=True)
    async with maintenance_backend(pg_rls_db.maintenance) as backend:
        await prepare_database(backend, settings, request_role=None)
        await apply_request_role_grants(backend, settings, request_role=REQUEST_ROLE)
        yield backend


async def _job_privileges(conn: asyncpg.Connection) -> dict[str, bool]:
    """Return whether the policy role holds each privilege requests need on the jobs table."""
    return {
        privilege: bool(await conn.fetchval('SELECT has_table_privilege($1, $2, $3)', POLICY_ROLE, JOBS_TABLE, privilege))
        for privilege in JOB_PRIVILEGES
    }


async def _managed_tables(conn: asyncpg.Connection) -> set[str]:
    return set(await row_security(conn, SCHEMA_NAME)) & set(MANAGED_TABLES)


async def _assert_installed(conn: asyncpg.Connection, *, policy_role: str = POLICY_ROLE) -> None:
    """Assert every managed table that exists carries exactly our policies, for the policy role, enabled without FORCE."""
    tables = await _managed_tables(conn)
    installed = await policies(conn, SCHEMA_NAME)
    assert set(installed) == {(table, name) for table in tables for name in policy_names(table)}
    assert {policy.roles for policy in installed.values()} == {(policy_role,)}
    assert all(parse_fingerprint(policy.comment) is not None for policy in installed.values())
    enabled = await row_security(conn, SCHEMA_NAME)
    assert {table: enabled[table] for table in tables} == dict.fromkeys(tables, (True, False))
    view_owner = await conn.fetchval('SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid = to_regclass($1)', VIEW)
    assert view_owner == 'mcp_owner'
    assert await view_acl(conn, VIEW) == [(policy_role, 'SELECT')]


async def _apply_holding_the_lock(
    backend: StorageBackend, admin: asyncpg.Connection, urls: RlsUrls, *, policy_role: str = POLICY_ROLE,
) -> int:
    """Apply the migration while another session holds the schema-init lock.

    A migration that asked for the lock would wait for it, so the run must finish while
    the lock is still held.

    Returns:
        The most sessions seen waiting for an advisory lock during the run.
    """
    holder = await asyncpg.connect(urls.admin)
    await holder.execute(SCHEMA_INIT_LOCK_SQL)
    waiters = 0
    task = asyncio.create_task(apply_rls(backend, rls_settings(policy_role=policy_role)))
    try:
        for _ in range(3000):
            if task.done():
                break
            waiters = max(waiters, int(await admin.fetchval(ADVISORY_WAITERS_SQL)))
            await asyncio.sleep(0.01)
        assert task.done(), 'the migration waited for the schema-init lock'
    finally:
        await holder.execute(SCHEMA_INIT_UNLOCK_SQL)
        await holder.close()
        await asyncio.wait_for(task, timeout=30)
    return waiters


class TestNoDdlBoot:
    """A boot over a current database issues no DDL and takes no lock; a difference waits for the lock."""

    @pytest.mark.parametrize('layout', ['fp32', 'compressed'])
    @pytest.mark.asyncio
    async def test_a_second_boot_issues_no_ddl_and_takes_no_lock(
        self, layout: Layout, pg_rls_db: RlsUrls, admin: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = configure_rls_layout(pg_rls_db, monkeypatch, layout=layout, overrides={}, rls=True)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            await prepare_database(backend, settings, request_role=None)
            await _assert_installed(admin)
            vector_table = 'vec_context_embeddings' if layout == 'fp32' else 'vec_context_embeddings_compressed'
            assert vector_table in await _managed_tables(admin)
            before = await object_oids(admin, SCHEMA_NAME)
            await install_ddl_log(admin)

            waiters = await _apply_holding_the_lock(backend, admin, pg_rls_db)

            assert waiters == 0
            assert await ddl_log(admin) == []
            assert await object_oids(admin, SCHEMA_NAME) == before

    @pytest.mark.asyncio
    async def test_a_difference_waits_for_the_lock_and_repairs_only_itself(
        self, booted: StorageBackend, pg_rls_db: RlsUrls, admin: asyncpg.Connection, owner: asyncpg.Connection,
    ) -> None:
        await owner.execute('DROP POLICY tags_insert ON tags')
        before = await object_oids(admin, SCHEMA_NAME)
        holder = await asyncpg.connect(pg_rls_db.admin)
        await holder.execute(SCHEMA_INIT_LOCK_SQL)
        try:
            task = asyncio.create_task(apply_rls(booted, rls_settings()))
            for _ in range(1500):
                if int(await admin.fetchval(ADVISORY_WAITERS_SQL)) == 1:
                    break
                await asyncio.sleep(0.01)
            assert int(await admin.fetchval(ADVISORY_WAITERS_SQL)) == 1
            assert not task.done()
        finally:
            await holder.execute(SCHEMA_INIT_UNLOCK_SQL)
            await holder.close()
        await asyncio.wait_for(task, timeout=30)

        after = await object_oids(admin, SCHEMA_NAME)
        assert set(after) - set(before) == {'policy tags.tags_insert'}
        assert {key: oid for key, oid in after.items() if key in before} == before
        await _assert_installed(admin)


class TestTablesThatExist:
    """Each boot covers the managed tables that exist, and a table created later is covered by the next boot."""

    @pytest.mark.asyncio
    async def test_a_database_without_embedding_storage_gets_policies_on_its_tables_alone(
        self, pg_rls_db: RlsUrls, admin: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        overrides = {'ENABLE_EMBEDDING_GENERATION': 'false'}
        settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides=overrides, rls=True)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            await prepare_database(backend, settings, request_role=None)

        assert await _managed_tables(admin) == {
            'context_entries', 'context_entry_grants', 'tags', 'image_attachments', 'context_index_nodes', 'generation_jobs',
        }
        await _assert_installed(admin)

    @pytest.mark.asyncio
    async def test_the_node_summary_table_switched_on_later_gets_its_policies_at_the_next_boot(
        self, pg_rls_db: RlsUrls, admin: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        off = {'ENABLE_INDEX_TREE_NODE_SUMMARIES': 'false'}
        settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides=off, rls=True)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            await prepare_database(backend, settings, request_role=None)
            assert 'context_index_nodes' not in await _managed_tables(admin)
            await _assert_installed(admin)
            before = await object_oids(admin, SCHEMA_NAME)

            on = {'ENABLE_INDEX_TREE_NODE_SUMMARIES': 'true'}
            settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides=on, rls=True)
            await prepare_database(backend, settings, request_role=None)

        await _assert_installed(admin)
        after = await object_oids(admin, SCHEMA_NAME)
        assert {key: oid for key, oid in after.items() if key in before} == before
        late = {f'policy context_index_nodes.{name}' for name in policy_names('context_index_nodes')}
        assert set(after) - set(before) == late
        read = (await policies(admin, SCHEMA_NAME))[('context_index_nodes', 'context_index_nodes_read')]
        fingerprint = parse_fingerprint(read.comment)
        assert fingerprint is not None
        assert fingerprint.preenabled is False


class TestRoles:
    """The policy role is part of every policy and of the view; a login role is part of nothing."""

    @pytest_asyncio.fixture
    async def alt_policy_role(self, admin: asyncpg.Connection) -> AsyncGenerator[str]:
        """A second NOLOGIN policy role, dropped with whatever names it afterwards.

        Yields:
            Its name.
        """
        await admin.execute(f'CREATE ROLE {ALT_POLICY_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS')
        try:
            yield ALT_POLICY_ROLE
        finally:
            await admin.execute(f'DROP OWNED BY {ALT_POLICY_ROLE}')
            await admin.execute(f'DROP ROLE {ALT_POLICY_ROLE}')

    @pytest_asyncio.fixture
    async def alt_login_role(self, admin: asyncpg.Connection) -> AsyncGenerator[str]:
        """A second request login role, an inheriting member of the policy role.

        Yields:
            Its name, which is also its password.
        """
        await admin.execute(
            f"CREATE ROLE {ALT_LOGIN_ROLE} LOGIN PASSWORD '{ALT_LOGIN_ROLE}' NOSUPERUSER NOBYPASSRLS INHERIT",
        )
        await admin.execute(f'GRANT {POLICY_ROLE} TO {ALT_LOGIN_ROLE}')
        try:
            yield ALT_LOGIN_ROLE
        finally:
            await admin.execute(f'DROP ROLE {ALT_LOGIN_ROLE}')

    @pytest.mark.asyncio
    async def test_a_new_policy_role_re_creates_every_policy_and_the_view(
        self, booted: StorageBackend, admin: asyncpg.Connection, alt_policy_role: str,
    ) -> None:
        before = await policies(admin, SCHEMA_NAME)
        view_oid = await admin.fetchval('SELECT to_regclass($1)::oid', VIEW)

        await apply_rls(booted, rls_settings(policy_role=alt_policy_role))

        after = await policies(admin, SCHEMA_NAME)
        assert set(after) == set(before)
        assert all(after[key].oid != before[key].oid for key in before)
        await _assert_installed(admin, policy_role=alt_policy_role)
        assert await admin.fetchval('SELECT to_regclass($1)::oid', VIEW) != view_oid
        assert f"'{alt_policy_role}'" in await admin.fetchval('SELECT pg_get_viewdef(to_regclass($1), true)', VIEW)

        await install_ddl_log(admin)
        await apply_rls(booted, rls_settings(policy_role=alt_policy_role))
        assert await ddl_log(admin) == []

    @pytest.mark.asyncio
    async def test_a_new_member_login_role_changes_nothing_and_reads_through_the_view(
        self, booted: StorageBackend, pg_rls_db: RlsUrls, admin: asyncpg.Connection, owner: asyncpg.Connection,
        alt_login_role: str,
    ) -> None:
        before = await object_oids(admin, SCHEMA_NAME)
        await install_ddl_log(admin)

        await apply_rls(booted, rls_settings())
        report = await apply_request_role_grants(booted, rls_settings(), request_role=alt_login_role)

        assert report.statements == []
        assert not report.locked
        assert await ddl_log(admin) == []
        assert await object_oids(admin, SCHEMA_NAME) == before
        await owner.execute(INSERT_PUBLIC_ENTRY, uuid.uuid4())
        login_url = pg_rls_db.app.replace(f'{REQUEST_ROLE}:{REQUEST_ROLE}@', f'{alt_login_role}:{alt_login_role}@')
        login = await asyncpg.connect(login_url)
        try:
            assert await login.fetchval('SELECT current_user') == alt_login_role
            assert await login.fetchval(f'SELECT count(*) FROM {VIEW}') == 1
        finally:
            await login.close()


class TestRepairs:
    """What changed outside the server is put back at the next boot and named in a WARNING."""

    @pytest.mark.asyncio
    async def test_an_altered_policy_is_re_created(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        before = await object_oids(admin, SCHEMA_NAME)
        await owner.execute('ALTER POLICY tags_read ON tags USING (true)')

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        after = await object_oids(admin, SCHEMA_NAME)
        assert after['policy tags.tags_read'] != before['policy tags.tags_read']
        assert {key: oid for key, oid in after.items() if key != 'policy tags.tags_read'} == {
            key: oid for key, oid in before.items() if key != 'policy tags.tags_read'
        }
        assert await admin.fetchval(
            "SELECT pg_get_expr(polqual, polrelid) FROM pg_policy WHERE polname = 'tags_read'",
        ) != 'true'
        fingerprint = parse_fingerprint((await policies(admin, SCHEMA_NAME))[('tags', 'tags_read')].comment)
        assert fingerprint is not None
        assert fingerprint.preenabled is False
        assert any(f'POLICY tags_read ON {SCHEMA}.tags' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_column_added_re_creates_the_view_with_its_acl_and_trigger_naming_the_grants_it_drops(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute('ALTER TABLE context_entries ADD COLUMN extra_probe text')
        await owner.execute(f'GRANT SELECT ON {VIEW} TO mcp_outsider')
        policies_before = await policies(admin, SCHEMA_NAME)

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        columns = await view_columns(admin, VIEW)
        assert columns == await view_columns(admin, f'{SCHEMA}.context_entries')
        assert columns[-1] == 'extra_probe'
        assert await view_acl(admin, VIEW) == [(POLICY_ROLE, 'SELECT')]
        assert await admin.fetchval(
            "SELECT count(*) FROM pg_trigger WHERE tgrelid = to_regclass($1) AND tgname = 'reject_writes' "
            "AND tgenabled <> 'D'",
            VIEW,
        ) == 1
        view = read_view([f'"{column}"' for column in columns], schema=SCHEMA, role_name=POLICY_ROLE)
        fingerprint = parse_fingerprint(await admin.fetchval("SELECT obj_description(to_regclass($1), 'pg_class')", VIEW))
        assert fingerprint is not None
        assert fingerprint.sha == view.sha
        assert fingerprint.definition == digest(await read_definition(admin, view))
        assert {key: policy.oid for key, policy in (await policies(admin, SCHEMA_NAME)).items()} == {
            key: policy.oid for key, policy in policies_before.items()
        }
        assert any('mcp_outsider (SELECT)' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_an_altered_grants_delete_policy_is_re_created_and_named(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        key = 'policy context_entry_grants.context_entry_grants_delete'
        before = await object_oids(admin, SCHEMA_NAME)
        await owner.execute('ALTER POLICY context_entry_grants_delete ON context_entry_grants USING (true)')

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        after = await object_oids(admin, SCHEMA_NAME)
        assert after[key] != before[key]
        assert {name: oid for name, oid in after.items() if name != key} == {
            name: oid for name, oid in before.items() if name != key
        }
        assert await admin.fetchval(
            "SELECT pg_get_expr(polqual, polrelid) FROM pg_policy WHERE polname = 'context_entry_grants_delete'",
        ) != 'true'
        [warning] = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert f'POLICY context_entry_grants_delete ON {SCHEMA}.context_entry_grants' in warning

    @pytest.mark.asyncio
    async def test_an_extra_grantee_on_the_unchanged_view_is_revoked_and_named(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute(f'GRANT SELECT ON {VIEW} TO mcp_outsider')
        await owner.execute(f'GRANT INSERT ON {VIEW} TO {POLICY_ROLE}')
        before = await object_oids(admin, SCHEMA_NAME)

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        assert await object_oids(admin, SCHEMA_NAME) == before
        assert await view_acl(admin, VIEW) == [(POLICY_ROLE, 'SELECT')]
        [warning] = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert 'mcp_outsider (SELECT)' in warning
        assert f'{POLICY_ROLE} (INSERT)' in warning


def _earlier_policies(schema: str, role: str) -> list[RlsObject]:
    """Return the policies an earlier release renders differently, as that release creates them.

    That release has no write-right helper, so its ``context_entries`` insert check names
    the owner and the publish right alone, and its grants table has no owner column, so its
    grants read policy admits the grantee's own rows only, its grants insert check reads the
    entry's owner through ``context_entries``, and it has no grants delete policy.

    Returns:
        The three policies in that release's rendering.
    """
    owner = build_policy_predicate(AccessMode.OWNER, outer='context_entries', schema=schema)
    owner_of_parent = build_policy_predicate(AccessMode.OWNER, outer='ce', schema=schema)
    operands = policy_operands(schema)
    grants = f'{schema}.context_entry_grants'
    statements = {
        ('context_entries_insert', f'{schema}.context_entries'): (
            f'CREATE POLICY context_entries_insert ON {schema}.context_entries AS PERMISSIVE FOR INSERT TO {role} '
            f"WITH CHECK ({owner} AND (context_entries.visibility <> 'public' OR {operands.may_publish}))"
        ),
        ('context_entry_grants_read', grants): (
            f'CREATE POLICY context_entry_grants_read ON {grants} AS PERMISSIVE FOR SELECT TO {role} USING '
            f"((principal_type = 'user' AND principal_id = {schema}.mcp_context_principal_id()) "
            f"OR (principal_type = 'group' AND principal_id = ANY({schema}.mcp_context_groups())))"
        ),
        ('context_entry_grants_insert', grants): (
            f'CREATE POLICY context_entry_grants_insert ON {grants} AS PERMISSIVE FOR INSERT TO {role} '
            f'WITH CHECK (granted_by = {operands.principal} AND EXISTS (SELECT 1 FROM {schema}.context_entries ce '
            f'WHERE ce.id = context_entry_grants.context_entry_id AND {owner_of_parent}))'
        ),
    }
    return [
        RlsObject(kind=RlsObjectKind.POLICY, name=name, anchor=anchor, statement=statement)
        for (name, anchor), statement in statements.items()
    ]


class TestUpgrade:
    """A database an earlier release secured is brought to the current policies at the next boot, silently."""

    @pytest.mark.asyncio
    async def test_an_earlier_releases_policies_are_repaired_without_a_warning(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        role = f'"{POLICY_ROLE}"'
        earlier = _earlier_policies(SCHEMA, role)
        await owner.execute(f'DROP POLICY context_entry_grants_delete ON {SCHEMA}.context_entry_grants')
        for policy in earlier:
            await owner.execute(policy.drop_statement)
            await owner.execute(policy.statement)
            preenabled = False if records_preenabled(policy) else None
            definition = digest(await read_definition(owner, policy))
            await owner.execute(comment_statement(policy, definition=definition, preenabled=preenabled))
        await owner.execute(f'DROP FUNCTION {SCHEMA}.mcp_context_may_write()')
        before = await object_oids(admin, SCHEMA_NAME)
        assert 'function mcp_context_may_write' not in before
        assert 'policy context_entry_grants.context_entry_grants_delete' not in before

        with caplog.at_level(logging.INFO, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        after = await object_oids(admin, SCHEMA_NAME)
        changed = {
            'function mcp_context_may_write',
            'policy context_entries.context_entries_insert',
            'policy context_entry_grants.context_entry_grants_read',
            'policy context_entry_grants.context_entry_grants_insert',
            'policy context_entry_grants.context_entry_grants_delete',
        }
        assert {name for name in after if after[name] != before.get(name)} == changed
        assert set(before) - set(after) == set()
        await _assert_installed(admin)
        records = [record for record in caplog.records if record.name == 'app.migrations.rls']
        assert [record.levelno for record in records] == [logging.INFO]
        assert records[0].getMessage().startswith('Applied row-level security')

        await install_ddl_log(admin)
        await apply_rls(booted, rls_settings())
        assert await ddl_log(admin) == []

    @pytest.mark.asyncio
    async def test_the_jobs_table_an_earlier_release_lacked_gets_its_policies_and_grants_at_the_next_boot(
        self, booted: StorageBackend, pg_rls_db: RlsUrls, admin: asyncpg.Connection, owner: asyncpg.Connection,
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute('DROP TABLE generation_jobs')
        assert 'generation_jobs' not in await _managed_tables(admin)
        await _assert_installed(admin)
        before = await object_oids(admin, SCHEMA_NAME)

        settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides={}, rls=True)
        with caplog.at_level(logging.INFO, logger='app.migrations.rls'):
            await prepare_database(booted, settings, request_role=None)

        await _assert_installed(admin)
        after = await object_oids(admin, SCHEMA_NAME)
        assert {key: oid for key, oid in after.items() if key in before} == before
        assert set(after) - set(before) == {f'policy generation_jobs.{name}' for name in policy_names('generation_jobs')}
        read = (await policies(admin, SCHEMA_NAME))[('generation_jobs', 'generation_jobs_read')]
        fingerprint = parse_fingerprint(read.comment)
        assert fingerprint is not None
        assert fingerprint.preenabled is False
        records = [record for record in caplog.records if record.name == 'app.migrations.rls']
        assert [record.levelno for record in records] == [logging.INFO]
        assert records[0].getMessage().startswith('Applied row-level security')

        assert await _job_privileges(admin) == dict.fromkeys(JOB_PRIVILEGES, False)
        await apply_request_role_grants(booted, settings, request_role=REQUEST_ROLE)
        assert await _job_privileges(admin) == dict.fromkeys(JOB_PRIVILEGES, True)


class TestFtsLanguageRebuild:
    """The FTS language rebuild re-creates the read view in its own transaction, as the migration would render it.

    Like the migration, it names every grant on the view its drop takes away beyond the
    policy role's SELECT.
    """

    @pytest.mark.asyncio
    async def test_the_view_survives_a_failed_rebuild_and_is_re_created_by_a_successful_one(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fts = RepositoryContainer(booted).fts
        view_oid = await admin.fetchval('SELECT to_regclass($1)::oid', VIEW)

        with pytest.raises(Exception, match='no_such_language'):
            await fts.migrate_language('no_such_language', scope=MAINTENANCE)
        assert await admin.fetchval('SELECT to_regclass($1)::oid', VIEW) == view_oid

        await owner.execute(f'GRANT SELECT ON {VIEW} TO mcp_outsider')
        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await fts.migrate_language('german', scope=MAINTENANCE)

        assert any('mcp_outsider (SELECT)' in record.getMessage() for record in caplog.records)
        assert await admin.fetchval('SELECT to_regclass($1)::oid', VIEW) != view_oid
        columns = await view_columns(admin, VIEW)
        assert columns == await view_columns(admin, f'{SCHEMA}.context_entries')
        assert columns[-1] == 'text_search_vector'
        assert await fts.get_current_language(scope=MAINTENANCE) == 'german'
        assert await view_acl(admin, VIEW) == [(POLICY_ROLE, 'SELECT')]
        await install_ddl_log(admin)
        await apply_rls(booted, rls_settings())
        assert await ddl_log(admin) == []


class TestFlagOffGuard:
    """A boot with ENABLE_POSTGRESQL_RLS off over the installed objects is refused before any provisioning."""

    @pytest.mark.asyncio
    async def test_starting_with_the_flag_off_is_refused_naming_every_table(self, booted: StorageBackend) -> None:
        flag_off = get_settings()
        assert not flag_off.access_control.postgresql_rls_enabled

        with pytest.raises(ConfigurationError) as raised:
            await guard_rls_disable_over_enabled(booted, flag_off)

        tables = [table for table in MANAGED_TABLES if table != 'vec_context_embeddings_compressed']
        assert f'(tables: {", ".join(tables)}, context_entries_readable)' in str(raised.value)
        assert 'mcp-context-server-migrate --disable-rls' in str(raised.value)
        with pytest.raises(ConfigurationError, match='ENABLE_POSTGRESQL_RLS=false'):
            await prepare_database(booted, flag_off, request_role=None)
