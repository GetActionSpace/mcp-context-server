"""The row-level security read view of ``context_entries`` against a real PostgreSQL server.

The request login role ``mcp_app`` reads ``context_entries`` through
``context_entries_readable``, a view owned by the table owner ``mcp_owner`` whose rows are
those the bound identity may read, and only for members of the policy role. These tests
prove that the view returns exactly the readable rows of every identity of the policy
matrix through the repositories with no application predicate, that it refuses every
write and locking read, that a role outside the policy role reads nothing through it
whatever it is granted, public rows included, that a dropped policy role fails every
read, that privileges default privileges hand out on the view are revoked when it is
created, that the fingerprint of the view detects a definition replaced without the
gate, and that a view an earlier release created with the gate as a condition of its
own is replaced at the next boot, after which a boot issues no DDL.
"""

import logging
import uuid
from collections.abc import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_gated_read_predicate
from app.access_scope import build_policy_predicate
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.migrations import apply_rls
from app.migrations import create_read_view
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import fingerprint
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_ddl import reject_trigger
from tests.integration.postgresql._rls_catalog import ddl_log
from tests.integration.postgresql._rls_catalog import install_ddl_log
from tests.integration.postgresql._rls_matrix import MATRIX_SCOPES
from tests.integration.postgresql._rls_matrix import MATRIX_VISIBILITY
from tests.integration.postgresql._rls_matrix import POLICY_ROLE
from tests.integration.postgresql._rls_matrix import labels
from tests.integration.postgresql._rls_matrix import rls_settings
from tests.integration.postgresql._rls_matrix import scoped_transaction
from tests.integration.postgresql._rls_profiles import pinned_repositories
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import in_seed_order

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

SCHEMA = '"public"'
VIEW = f'{SCHEMA}.context_entries_readable'
GATE = f"pg_catalog.pg_has_role(CURRENT_USER, '{POLICY_ROLE}', 'USAGE')"
ALICE = AccessScope('alice', frozenset())
REFUSED = asyncpg.exceptions.InsufficientPrivilegeError
MIGRATION_TIMEOUT_S = 300.0

INSERT_ENTRY = (
    'INSERT INTO {relation} (id, thread_id, source, content_type, text_content, owner_id, visibility) '
    "VALUES ($1, 'rls-view', 'agent', 'text', 'view probe', 'alice', 'private')"
)
# The grantees and privileges the view's ACL holds besides its owner.
VIEW_ACL = (
    'SELECT COALESCE(r.rolname, $2), a.privilege_type FROM pg_class c '
    'CROSS JOIN LATERAL aclexplode(c.relacl) AS a LEFT JOIN pg_roles r ON r.oid = a.grantee '
    'WHERE c.oid = to_regclass($1) AND a.grantee <> c.relowner ORDER BY 1, 2'
)
CONTEXT_COLUMNS = (
    'SELECT attname FROM pg_attribute WHERE attrelid = to_regclass($1) AND attnum > 0 AND NOT attisdropped ORDER BY attnum'
)


@pytest_asyncio.fixture
async def admin_conn(matrix_db: ScopedDb, pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A raw superuser connection to the matrix database.

    Yields:
        The open connection, closed afterwards.
    """
    _ = matrix_db
    conn = await asyncpg.connect(pg_rls_db.admin)
    try:
        yield conn
    finally:
        await conn.close()


async def _rendered_view(conn: asyncpg.Connection) -> RlsObject:
    """Render the view the migration creates for the current columns of context_entries."""
    columns = [quote_pg_identifier(str(row[0])) for row in await conn.fetch(CONTEXT_COLUMNS, f'{SCHEMA}.context_entries')]
    return read_view(columns, schema=SCHEMA, role_name=POLICY_ROLE)


async def _view_acl(conn: asyncpg.Connection) -> list[tuple[str, str]]:
    return [(str(row[0]), str(row[1])) for row in await conn.fetch(VIEW_ACL, VIEW, 'PUBLIC')]


async def _comment(conn: asyncpg.Connection, obj: RlsObject) -> object:
    """Return the comment on ``obj``: a view, a function or a trigger."""
    if obj.kind is RlsObjectKind.TRIGGER:
        return await conn.fetchval(
            "SELECT obj_description(t.oid, 'pg_trigger') FROM pg_trigger t "
            'WHERE t.tgrelid = to_regclass($1) AND t.tgname = $2',
            obj.anchor,
            obj.name,
        )
    if obj.kind is RlsObjectKind.VIEW:
        return await conn.fetchval("SELECT obj_description(to_regclass($1), 'pg_class')", obj.anchor)
    return await conn.fetchval("SELECT obj_description(to_regprocedure($1), 'pg_proc')", obj.anchor)


class TestRows:
    """The view returns exactly the rows the bound identity may read."""

    @pytest.mark.asyncio
    async def test_view_rows_are_the_readable_rows_of_every_identity(self, matrix_db: ScopedDb, pg_rls_db: RlsUrls) -> None:
        """With no application predicate, a repository read through the view returns the readable rows alone."""
        observed: dict[str, tuple[str, ...]] = {}
        for name, scope in MATRIX_SCOPES.items():
            async with pinned_repositories(pg_rls_db.app, scope) as repos:
                rows, stats = await repos.context.search_contexts(limit=100, scope=SYSTEM_SCOPE)
            assert 'error' not in stats, stats
            observed[name] = in_seed_order(matrix_db.labels_of(str(row['id']) for row in rows))

        assert observed == MATRIX_VISIBILITY[AccessMode.READ]

    @pytest.mark.asyncio
    async def test_raw_reads_of_the_view_match_the_read_policy(self, matrix_db: ScopedDb, pg_rls_db: RlsUrls) -> None:
        """The view and the base table under its read policy admit the same rows for every identity."""
        conn = await asyncpg.connect(pg_rls_db.app)
        try:
            for name, scope in MATRIX_SCOPES.items():
                async with scoped_transaction(conn, scope):
                    through_view = labels(matrix_db, await conn.fetch('SELECT id FROM context_entries_readable'))
                    through_policy = labels(matrix_db, await conn.fetch('SELECT id FROM context_entries'))
                assert through_view == through_policy == MATRIX_VISIBILITY[AccessMode.READ][name], name
        finally:
            await conn.close()


class TestWrites:
    """The policy role holds SELECT alone on the view, and the view rejects writes even when granted."""

    @pytest.mark.parametrize(
        ('statement', 'new_row'),
        [
            pytest.param(INSERT_ENTRY.format(relation='context_entries_readable'), True, id='insert'),
            pytest.param("UPDATE context_entries_readable SET text_content = 'changed' WHERE id = $1", False, id='update'),
            pytest.param('DELETE FROM context_entries_readable WHERE id = $1', False, id='delete'),
            pytest.param('SELECT id FROM context_entries_readable WHERE id = $1 FOR KEY SHARE', False, id='key-share'),
        ],
    )
    @pytest.mark.asyncio
    async def test_writes_and_locking_reads_through_the_view_are_refused(
        self, statement: str, new_row: bool, matrix_db: ScopedDb, pg_rls_db: RlsUrls,
    ) -> None:
        """Every write and every locking read of the request role raises 42501, even on its own row."""
        target = uuid.uuid4() if new_row else uuid.UUID(hex=matrix_db.ids['alice_private'])
        conn = await asyncpg.connect(pg_rls_db.app)
        try:
            async with scoped_transaction(conn, ALICE):
                with pytest.raises(REFUSED):
                    await conn.execute(statement, target)
        finally:
            await conn.close()

        rows = await matrix_db.fetch_all('SELECT count(*) FROM context_entries WHERE id = ?', (target.hex,))
        assert rows == [(0 if new_row else 1,)]


class TestOutsiders:
    """A role outside the policy role reads nothing through the view and writes nothing, whatever it holds."""

    @pytest.mark.asyncio
    async def test_a_view_grant_gives_an_outsider_no_row(
        self, matrix_db: ScopedDb, admin_conn: asyncpg.Connection, pg_rls_db: RlsUrls,
    ) -> None:
        """SELECT on the view, granted out of band, reads no row under any identity of the matrix, public rows included."""
        public_count = 'SELECT count(*) FROM context_entries_readable WHERE visibility = $1'
        member = await asyncpg.connect(pg_rls_db.app)
        try:
            async with scoped_transaction(member, None):
                public_rows = labels(matrix_db, await member.fetch('SELECT id FROM context_entries_readable'))
                assert await member.fetchval(public_count, 'public') == len(public_rows)
        finally:
            await member.close()
        assert public_rows
        assert public_rows == MATRIX_VISIBILITY[AccessMode.READ]['unset']

        await admin_conn.execute(f'GRANT SELECT ON {VIEW} TO mcp_outsider')
        conn = await asyncpg.connect(pg_rls_db.outsider)
        try:
            for name, scope in MATRIX_SCOPES.items():
                async with scoped_transaction(conn, scope):
                    assert await conn.fetchval('SELECT count(*) FROM context_entries_readable') == 0, name
                    assert await conn.fetchval(public_count, 'public') == 0, name
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_schema_wide_grants_give_an_outsider_no_row_and_no_write(
        self, admin_conn: asyncpg.Connection, pg_rls_db: RlsUrls,
    ) -> None:
        """ALL on every table of the schema still reads nothing and inserts nothing, through either relation."""
        await admin_conn.execute(f'GRANT ALL ON ALL TABLES IN SCHEMA {SCHEMA} TO mcp_outsider')
        conn = await asyncpg.connect(pg_rls_db.outsider)
        try:
            async with scoped_transaction(conn, ALICE):
                assert await conn.fetchval('SELECT count(*) FROM context_entries') == 0
                assert await conn.fetchval('SELECT count(*) FROM context_entries_readable') == 0
            for relation in ('context_entries', 'context_entries_readable'):
                async with scoped_transaction(conn, ALICE):
                    with pytest.raises(REFUSED):
                        await conn.execute(INSERT_ENTRY.format(relation=relation), uuid.uuid4())
        finally:
            await conn.close()


class TestPolicyRole:
    """The gate names the policy role, so the view fails closed when that role is gone."""

    @pytest.mark.asyncio
    async def test_dropping_the_policy_role_fails_every_view_read(
        self, admin_conn: asyncpg.Connection, pg_rls_db: RlsUrls,
    ) -> None:
        """Once the role the gate names is dropped, a read through the view raises instead of returning rows."""
        probe_role = f'mcp_view_gate_{uuid.uuid4().hex[:8]}'
        await admin_conn.execute(f'CREATE ROLE {probe_role} NOLOGIN NOSUPERUSER NOBYPASSRLS')
        try:
            await admin_conn.execute(f'GRANT {probe_role} TO mcp_app')
            owner = await asyncpg.connect(pg_rls_db.maintenance)
            try:
                async with owner.transaction():
                    await owner.execute(f'DROP VIEW {VIEW}')
                    await create_read_view(owner, schema='public', policy_role=probe_role, timeout_s=MIGRATION_TIMEOUT_S)
                # A privilege that does not depend on the policy role, so only the gate stands between mcp_app and the rows.
                await admin_conn.execute(f'GRANT SELECT ON {VIEW} TO mcp_app')
                app_conn = await asyncpg.connect(pg_rls_db.app)
                try:
                    async with scoped_transaction(app_conn, ALICE):
                        assert await app_conn.fetchval('SELECT count(*) FROM context_entries_readable') > 0

                    await admin_conn.execute(f'DROP OWNED BY {probe_role}')
                    await admin_conn.execute(f'DROP ROLE {probe_role}')

                    async with scoped_transaction(app_conn, ALICE):
                        with pytest.raises(asyncpg.exceptions.UndefinedObjectError, match=probe_role):
                            await app_conn.fetchval('SELECT count(*) FROM context_entries_readable')
                finally:
                    await app_conn.close()
                with pytest.raises(asyncpg.exceptions.UndefinedObjectError, match=probe_role):
                    await owner.fetchval('SELECT count(*) FROM context_entries_readable')
            finally:
                await owner.close()
        finally:
            if await admin_conn.fetchval('SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = $1)', probe_role):
                await admin_conn.execute(f'DROP OWNED BY {probe_role}')
                await admin_conn.execute(f'DROP ROLE {probe_role}')


class TestInstalledView:
    """What the migration leaves in the catalog for the view."""

    @pytest.mark.asyncio
    async def test_the_view_is_owned_by_the_table_owner_without_options_and_rejects_writes(
        self, admin_conn: asyncpg.Connection,
    ) -> None:
        """The owner reads past the policies, no option demotes the query's conditions and the trigger is in place."""
        row = await admin_conn.fetchrow(
            'SELECT pg_get_userbyid(c.relowner), c.relkind::text, c.reloptions FROM pg_class c WHERE c.oid = to_regclass($1)',
            VIEW,
        )
        triggers = await admin_conn.fetch(
            'SELECT tgname, pg_get_triggerdef(oid) FROM pg_trigger WHERE tgrelid = to_regclass($1)', VIEW,
        )

        assert row is not None
        assert (row[0], row[1], row[2]) == ('mcp_owner', 'v', None)
        assert [str(trigger[0]) for trigger in triggers] == ['reject_writes']
        assert 'INSTEAD OF INSERT OR DELETE OR UPDATE' in str(triggers[0][1])
        assert await _view_acl(admin_conn) == [(POLICY_ROLE, 'SELECT')]

    @pytest.mark.asyncio
    async def test_the_view_and_its_write_rejection_carry_their_fingerprints(self, admin_conn: asyncpg.Connection) -> None:
        """Each comment holds the sha of the rendered statement and the digest of the server's own definition."""
        for obj in (await _rendered_view(admin_conn), reject_function(SCHEMA), reject_trigger(SCHEMA)):
            comment = await _comment(admin_conn, obj)
            definition = digest(await read_definition(admin_conn, obj))
            assert comment == fingerprint(sha=obj.sha, definition=definition, preenabled=None), obj.target

    @pytest.mark.asyncio
    async def test_the_definition_digest_changes_when_the_view_loses_its_gate(
        self, admin_conn: asyncpg.Connection, pg_rls_db: RlsUrls,
    ) -> None:
        """A view replaced out of band keeps its comment, whose definition digest differs from that of the new definition."""
        view = await _rendered_view(admin_conn)
        before = digest(await read_definition(admin_conn, view))
        gated = build_gated_read_predicate(outer='ce', schema=SCHEMA, gate=GATE)
        ungated = build_policy_predicate(AccessMode.READ, outer='ce', schema=SCHEMA)
        assert view.statement.endswith(f' WHERE {gated}')
        owner = await asyncpg.connect(pg_rls_db.maintenance)
        try:
            await owner.execute(view.statement.replace('CREATE VIEW', 'CREATE OR REPLACE VIEW').replace(gated, ungated))
        finally:
            await owner.close()

        comment = await _comment(admin_conn, view)
        after = digest(await read_definition(admin_conn, view))
        assert comment == fingerprint(sha=view.sha, definition=before, preenabled=None)
        assert after != before

    @pytest.mark.asyncio
    async def test_default_privileges_on_a_new_view_are_revoked_when_it_is_created(
        self, matrix_db: ScopedDb, admin_conn: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Whatever default privileges grant on a new view, only the policy role keeps SELECT, and the revocation is logged."""
        await admin_conn.execute(
            f'ALTER DEFAULT PRIVILEGES FOR ROLE mcp_owner IN SCHEMA {SCHEMA} GRANT ALL ON TABLES TO mcp_outsider, PUBLIC',
        )
        await admin_conn.execute(
            f'ALTER DEFAULT PRIVILEGES FOR ROLE mcp_owner IN SCHEMA {SCHEMA} GRANT ALL ON TABLES TO {POLICY_ROLE}',
        )
        # A current view is left alone, so the migration creates a new one only once this one is gone.
        await admin_conn.execute(f'DROP VIEW {VIEW}')

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(matrix_db.seed_backend, rls_settings())

        assert await _view_acl(admin_conn) == [(POLICY_ROLE, 'SELECT')]
        [warning] = [record for record in caplog.records if record.name == 'app.migrations.rls']
        message = warning.getMessage()
        assert 'PUBLIC (' in message
        assert 'mcp_outsider (' in message
        assert f'{POLICY_ROLE} (' in message
        assert 'INSERT' in message

    @pytest.mark.asyncio
    async def test_the_request_role_reads_the_view_after_the_migration_reruns(
        self, matrix_db: ScopedDb, pg_rls_db: RlsUrls,
    ) -> None:
        """Another apply leaves the view and its privileges in place, so the request role reads it right after."""
        await apply_rls(matrix_db.seed_backend, rls_settings())

        conn = await asyncpg.connect(pg_rls_db.app)
        try:
            async with scoped_transaction(conn, ALICE):
                rows = labels(matrix_db, await conn.fetch('SELECT id FROM context_entries_readable'))
        finally:
            await conn.close()
        assert rows == MATRIX_VISIBILITY[AccessMode.READ]['alice']


class TestUpgrade:
    """A view an earlier release created with the gate as a condition of its own is replaced once."""

    @pytest.mark.asyncio
    async def test_a_view_gated_by_a_condition_of_its_own_is_re_created_once(
        self, matrix_db: ScopedDb, admin_conn: asyncpg.Connection, pg_rls_db: RlsUrls, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The next boot drops and re-creates the view with the gate in its operands; the boot after it issues no DDL."""
        view = await _rendered_view(admin_conn)
        gated = build_gated_read_predicate(outer='ce', schema=SCHEMA, gate=GATE)
        ungated = build_policy_predicate(AccessMode.READ, outer='ce', schema=SCHEMA)
        statement = view.statement.replace(f' WHERE {gated}', f' WHERE {GATE} AND {ungated}')
        assert statement != view.statement
        earlier = RlsObject(kind=RlsObjectKind.VIEW, name=view.name, anchor=view.anchor, statement=statement)
        owner = await asyncpg.connect(pg_rls_db.maintenance)
        try:
            await owner.execute(statement.replace('CREATE VIEW', 'CREATE OR REPLACE VIEW'))
            definition = digest(await read_definition(owner, earlier))
            await owner.execute(comment_statement(earlier, definition=definition, preenabled=None))
        finally:
            await owner.close()
        await install_ddl_log(admin_conn)

        with caplog.at_level(logging.INFO, logger='app.migrations.rls'):
            await apply_rls(matrix_db.seed_backend, rls_settings())

        assert (await ddl_log(admin_conn))[:2] == ['DROP VIEW', 'CREATE VIEW']
        definition = digest(await read_definition(admin_conn, view))
        assert await _comment(admin_conn, view) == fingerprint(sha=view.sha, definition=definition, preenabled=None)
        assert await _view_acl(admin_conn) == [(POLICY_ROLE, 'SELECT')]
        assert [record.levelno for record in caplog.records if record.name == 'app.migrations.rls'] == [logging.INFO]
        conn = await asyncpg.connect(pg_rls_db.app)
        try:
            async with scoped_transaction(conn, ALICE):
                rows = labels(matrix_db, await conn.fetch('SELECT id FROM context_entries_readable'))
        finally:
            await conn.close()
        assert rows == MATRIX_VISIBILITY[AccessMode.READ]['alice']

        await admin_conn.execute('TRUNCATE public.ddl_log')
        await apply_rls(matrix_db.seed_backend, rls_settings())
        assert await ddl_log(admin_conn) == []
