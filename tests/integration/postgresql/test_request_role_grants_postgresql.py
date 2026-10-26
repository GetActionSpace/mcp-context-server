"""The policy role's privileges against a real PostgreSQL server.

On the ``pg_rls_db`` database, prepared and seeded by its owner ``mcp_owner``, the owner
also creates a table, a serial column's sequence and a standalone sequence the server
knows nothing about. ``apply_request_role_grants`` then gives the policy role exactly the
privileges the request role needs: USAGE on the schema, SELECT, INSERT, UPDATE and DELETE
on every managed table of the layout, SELECT on the compression provenance, USAGE and
SELECT on the sequences the managed tables own and EXECUTE on the merge-patch function,
and nothing on the unrelated objects. A second run finds nothing missing, takes no lock
and leaves every access control list as it was.
"""

from collections.abc import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio

from app.errors import ConfigurationError
from app.migrations import GrantReport
from app.migrations import apply_request_role_grants
from app.migrations.rls_ddl import MANAGED_TABLES
from tests.integration.postgresql._rls_matrix import POLICY_ROLE
from tests.integration.postgresql._rls_matrix import REQUEST_ROLE
from tests.integration.postgresql._rls_matrix import rls_settings
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_owner_seeded_db

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

TABLE_PRIVILEGES = ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER')
SEQUENCE_PRIVILEGES = ('USAGE', 'SELECT', 'UPDATE')
DML = frozenset({'SELECT', 'INSERT', 'UPDATE', 'DELETE'})

# Every privilege the policy role effectively holds on every table and sequence of the schema.
RELATION_PRIVILEGES_SQL = f'''
SELECT c.relname, c.relkind::text,
       ARRAY(SELECT p FROM unnest(CASE WHEN c.relkind = 'S' THEN ARRAY{list(SEQUENCE_PRIVILEGES)}
                                       ELSE ARRAY{list(TABLE_PRIVILEGES)} END) AS p
             WHERE CASE WHEN c.relkind = 'S' THEN has_sequence_privilege($1, c.oid, p)
                        ELSE has_table_privilege($1, c.oid, p) END
             ORDER BY p)
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'S')
'''
# The access control list of the schema, every relation in it and the merge-patch function.
ACLS_SQL = '''
SELECT 'schema', nspacl::text FROM pg_namespace WHERE nspname = 'public'
UNION ALL
SELECT c.relname, c.relacl::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public'
UNION ALL
SELECT 'jsonb_merge_patch', proacl::text FROM pg_proc WHERE oid = 'public.jsonb_merge_patch(jsonb, jsonb)'::regprocedure
ORDER BY 1
'''


@pytest_asyncio.fixture(params=['fp32', 'compressed'])
async def owner_db(
    request: pytest.FixtureRequest, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """The ``pg_rls_db`` database prepared and seeded by its owner, with three objects unknown to the server.

    Yields:
        The seeded database in the fp32 and in the compressed layout.
    """
    async with postgresql_owner_seeded_db(
        request_url=pg_rls_db.app, maintenance_url=pg_rls_db.maintenance, layout=request.param, monkeypatch=monkeypatch,
    ) as db:
        await db.execute('CREATE TABLE unrelated_notes (id BIGSERIAL PRIMARY KEY, body TEXT)')
        await db.execute('CREATE SEQUENCE unrelated_counter')
        yield db


@pytest_asyncio.fixture
async def admin_conn(owner_db: ScopedDb, pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A superuser connection to the database, which reads every catalog row.

    Yields:
        The open connection, closed afterwards.
    """
    _ = owner_db
    conn = await asyncpg.connect(pg_rls_db.admin)
    try:
        yield conn
    finally:
        await conn.close()


async def _held(conn: asyncpg.Connection) -> dict[str, frozenset[str]]:
    rows = await conn.fetch(RELATION_PRIVILEGES_SQL, POLICY_ROLE)
    return {str(row[0]): frozenset(row[2]) for row in rows}


async def _expected(conn: asyncpg.Connection) -> dict[str, frozenset[str]]:
    """Return the privileges the policy role must hold, from an independent read of the layout."""
    tables = {
        str(row[0]): str(row[1])
        for row in await conn.fetch(
            "SELECT c.relname, c.relkind::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'S')",
        )
    }
    expected = dict.fromkeys(tables, frozenset[str]())
    for table in MANAGED_TABLES:
        if table not in tables:
            continue
        expected[table] = DML
        has_id = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' "
            "AND table_name = $1 AND column_name = 'id')",
            table,
        )
        sequence = await conn.fetchval("SELECT pg_get_serial_sequence($1, 'id')", f'public.{table}') if has_id else None
        if sequence is not None:
            expected[sequence.removeprefix('public.')] = frozenset({'USAGE', 'SELECT'})
    if 'compression_metadata' in tables:
        expected['compression_metadata'] = frozenset({'SELECT'})
    return expected


@pytest.mark.asyncio
async def test_the_policy_role_receives_exactly_the_needed_privileges(
    owner_db: ScopedDb, admin_conn: asyncpg.Connection,
) -> None:
    """Managed tables, owned sequences, the provenance table, the schema and the function; nothing else."""
    report = await apply_request_role_grants(owner_db.seed_backend, rls_settings(), request_role=REQUEST_ROLE)

    held = await _held(admin_conn)
    expected = await _expected(admin_conn)
    assert report.locked
    assert report.statements
    assert held == expected
    assert held['generation_jobs'] == DML
    assert held['unrelated_notes'] == held['unrelated_notes_id_seq'] == held['unrelated_counter'] == frozenset()
    assert ('compression_metadata' in held) is (owner_db.layout == 'compressed')
    assert await admin_conn.fetchval("SELECT has_schema_privilege($1, 'public', 'USAGE')", POLICY_ROLE)
    assert await admin_conn.fetchval(
        "SELECT has_function_privilege($1, 'public.jsonb_merge_patch(jsonb, jsonb)', 'EXECUTE')", POLICY_ROLE,
    )


@pytest.mark.asyncio
async def test_a_second_run_issues_nothing_and_takes_no_lock(owner_db: ScopedDb, admin_conn: asyncpg.Connection) -> None:
    """The grants are computed missing-only, so an unchanged boot writes no catalog row."""
    await apply_request_role_grants(owner_db.seed_backend, rls_settings(), request_role=REQUEST_ROLE)
    before = await admin_conn.fetch(ACLS_SQL)

    report = await apply_request_role_grants(owner_db.seed_backend, rls_settings(), request_role=REQUEST_ROLE)

    assert report == GrantReport(statements=[], locked=False)
    assert await admin_conn.fetch(ACLS_SQL) == before


@pytest.mark.asyncio
async def test_identical_roles_grant_nothing(owner_db: ScopedDb, admin_conn: asyncpg.Connection) -> None:
    """When requests log in as the maintenance role itself, there is no split to grant across."""
    before = await admin_conn.fetch(ACLS_SQL)

    report = await apply_request_role_grants(owner_db.seed_backend, rls_settings(), request_role='mcp_owner')

    assert report == GrantReport(statements=[], locked=False)
    assert await admin_conn.fetch(ACLS_SQL) == before


@pytest.mark.asyncio
async def test_a_missing_policy_role_is_refused_before_any_grant(owner_db: ScopedDb, admin_conn: asyncpg.Connection) -> None:
    """The refusal prints the statements that create the role and its membership, and changes nothing."""
    before = await admin_conn.fetch(ACLS_SQL)

    statements = 'CREATE ROLE "mcp_missing_policy" NOLOGIN; GRANT "mcp_missing_policy" TO "mcp_app";'
    with pytest.raises(ConfigurationError, match=statements):
        await apply_request_role_grants(
            owner_db.seed_backend, rls_settings(policy_role='mcp_missing_policy'), request_role=REQUEST_ROLE,
        )

    assert await admin_conn.fetch(ACLS_SQL) == before
