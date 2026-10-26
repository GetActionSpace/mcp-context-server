"""The row-level security helper functions against a real PostgreSQL server.

The four helpers hand the transaction's bound identity to the policies and the read
view. Their SQL-standard bodies carry no SET clause, so the planner inlines every call
into the expression around it instead of calling a function once per row, and the
names in the bodies were bound when the migration created them under a search path of
pg_catalog alone, so no search path a caller sets can change what they read. These
tests read the helpers' catalog entries, plan a routed read as the request login role
``mcp_app`` to prove the grants policy evaluates the inlined bodies, its owner arm
included, shadow a function and an operator the bodies use under a caller's search path,
and replace helpers created in the earlier ``SET search_path`` form, after which the next
boot issues no DDL.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Final
from typing import cast

import asyncpg
import pytest
import pytest_asyncio

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.migrations import apply_rls
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import parse_fingerprint
from tests.integration.postgresql._rls_catalog import ddl_log
from tests.integration.postgresql._rls_catalog import install_ddl_log
from tests.integration.postgresql._rls_matrix import MATRIX_VISIBILITY
from tests.integration.postgresql._rls_matrix import labels
from tests.integration.postgresql._rls_matrix import rls_matrix_db
from tests.integration.postgresql._rls_matrix import rls_settings
from tests.integration.postgresql._rls_matrix import scoped_transaction
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import ScopedDb

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

SCHEMA: Final = '"public"'
VIEW: Final = f'{SCHEMA}.context_entries_readable'
CAROL: Final = AccessScope('carol', frozenset({'team-x'}))
SHADOW_PATH: Final = f'SET LOCAL search_path = shadow, {SCHEMA}, pg_catalog'
# What the catalog holds for a helper as the migration creates it: an invoker's function with no
# SET clause and a SQL-standard body, STABLE and PARALLEL SAFE.
CURRENT_FACTS: Final = (False, None, True, 's', 's')
# The statements an earlier release created the first three helpers with: string bodies under a SET clause.
EARLIER_HELPERS: Final = (
    (
        'mcp_context_principal_id', 'text',
        "SELECT NULLIF(pg_catalog.current_setting('app.principal_id', true), '')",
    ),
    (
        'mcp_context_groups', 'text[]',
        "SELECT COALESCE(NULLIF(pg_catalog.current_setting('app.groups', true), '')::text[], '{}'::text[])",
    ),
    (
        'mcp_context_may_publish', 'boolean',
        "SELECT COALESCE(NULLIF(pg_catalog.current_setting('app.may_publish', true), '') = 'on', false)",
    ),
)
_HELPER_FACTS_SQL: Final = (
    'SELECT p.proname, p.prosecdef, p.proconfig, p.prosqlbody IS NOT NULL, p.provolatile::text, p.proparallel::text '
    "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = 'public' AND p.proname = ANY($1::text[])"
)
_GRANTS_TABLE: Final = 'context_entry_grants'
_CONDITIONS: Final = ('Filter', 'Recheck Cond', 'Index Cond', 'Join Filter', 'Hash Cond')


@pytest_asyncio.fixture
async def matrix_db(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """The policy matrix database in the fp32 layout, with the helpers, policies and read view installed.

    Yields:
        The seeded database.
    """
    async with rls_matrix_db(pg_rls_db, layout='fp32', monkeypatch=monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def app_conn(matrix_db: ScopedDb, pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A raw connection of the request login role ``mcp_app`` to the matrix database.

    Yields:
        The open connection, closed afterwards.
    """
    _ = matrix_db
    conn = await asyncpg.connect(pg_rls_db.app)
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def owner(matrix_db: ScopedDb, pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A connection of the table owner ``mcp_owner`` to the matrix database.

    Yields:
        The open connection, closed afterwards.
    """
    _ = matrix_db
    conn = await asyncpg.connect(pg_rls_db.maintenance)
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def shadow(owner: asyncpg.Connection) -> None:
    """Create, as the table owner, a schema ``mcp_app`` may use that shadows ``current_setting`` and text equality.

    The shadow ``current_setting`` answers ``'alice'`` for every setting, and the shadow
    ``=`` on two texts is never true.
    """
    await owner.execute('CREATE SCHEMA shadow')
    await owner.execute(
        "CREATE FUNCTION shadow.current_setting(text, boolean) RETURNS text LANGUAGE sql IMMUTABLE RETURN 'alice'",
    )
    await owner.execute('CREATE FUNCTION shadow.never_equal(text, text) RETURNS boolean LANGUAGE sql IMMUTABLE RETURN false')
    await owner.execute('CREATE OPERATOR shadow.= (LEFTARG = text, RIGHTARG = text, FUNCTION = shadow.never_equal)')
    await owner.execute('GRANT USAGE ON SCHEMA shadow TO mcp_app')


def _helper_names() -> list[str]:
    return [helper.name for helper in helper_functions(SCHEMA)]


async def _helper_facts(conn: asyncpg.Connection) -> dict[str, tuple[object, ...]]:
    """Read each helper's catalog entry.

    Returns:
        Per helper: whether it is SECURITY DEFINER, its SET clauses, whether it has a
        SQL-standard body, its volatility and its parallel safety.
    """
    return {str(row[0]): tuple(row[1:]) for row in await conn.fetch(_HELPER_FACTS_SQL, _helper_names())}


def _grants_conditions(node: object, *, under_grants: bool = False) -> list[str]:
    """Collect the conditions of the plan nodes that scan ``context_entry_grants``.

    Returns:
        Every condition of those nodes and of the index scans beneath them.
    """
    found: list[str] = []
    if isinstance(node, dict):
        plan = cast('dict[str, object]', node)
        scans_grants = under_grants or plan.get('Relation Name') == _GRANTS_TABLE
        if scans_grants:
            found.extend(str(plan[key]) for key in _CONDITIONS if key in plan)
        for key, child in plan.items():
            found.extend(_grants_conditions(child, under_grants=scans_grants and key == 'Plans'))
    elif isinstance(node, list):
        for item in cast('list[object]', node):
            found.extend(_grants_conditions(item, under_grants=under_grants))
    return found


class TestCatalog:
    """Each helper is an invoker's STABLE, PARALLEL SAFE SQL function with a SQL-standard body and no SET clause."""

    @pytest.mark.asyncio
    async def test_helpers_carry_a_sql_standard_body_and_no_configuration(
        self, matrix_db: ScopedDb, pg_rls_db: RlsUrls,
    ) -> None:
        _ = matrix_db
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            facts = await _helper_facts(conn)
            acls = [
                await conn.fetchval('SELECT proacl::text FROM pg_proc WHERE oid = to_regprocedure($1)', helper.anchor)
                for helper in helper_functions(SCHEMA)
            ]
        finally:
            await conn.close()

        assert sorted(facts) == sorted(
            ['mcp_context_principal_id', 'mcp_context_groups', 'mcp_context_may_publish', 'mcp_context_may_write'],
        )
        assert facts == dict.fromkeys(_helper_names(), CURRENT_FACTS)
        assert all('mcp_context_request=X/' in str(acl) for acl in acls), acls


class TestInlining:
    """The grants read policy evaluates the inlined bodies; no helper is called by name in a routed read."""

    @pytest.mark.asyncio
    async def test_a_routed_read_evaluates_the_inlined_identity_in_the_grants_policy(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        predicate = build_access_predicate(CAROL, mode=AccessMode.READ, backend_type='postgresql', outer='ce')
        statement = f'SELECT ce.id FROM {VIEW} AS ce WHERE {predicate.sql} ORDER BY ce.id DESC LIMIT 20'

        async with scoped_transaction(app_conn, CAROL):
            rows = await app_conn.fetch(statement, *predicate.params)
            document = str(await app_conn.fetchval(f'EXPLAIN (VERBOSE, FORMAT JSON) {statement}', *predicate.params))

        assert labels(matrix_db, rows) == MATRIX_VISIBILITY[AccessMode.READ]['carol']
        grants_conditions = _grants_conditions(json.loads(document))
        assert any("current_setting('app.principal_id'" in condition for condition in grants_conditions), document
        assert any('entry_owner_id' in condition for condition in grants_conditions), document
        assert 'mcp_context_principal_id(' not in document
        assert 'mcp_context_groups(' not in document


class TestResolutionBoundAtCreation:
    """A caller's search path that shadows what the bodies name changes nothing the helpers read."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('shadow')
    async def test_a_shadow_current_setting_leaves_an_unbound_caller_with_the_public_rows(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        async with scoped_transaction(app_conn, None):
            await app_conn.execute(SHADOW_PATH)
            shadowed = await app_conn.fetchval("SELECT current_setting('app.principal_id', true)")
            principal = await app_conn.fetchval(f'SELECT {SCHEMA}.mcp_context_principal_id()')
            through_view = labels(matrix_db, await app_conn.fetch(f'SELECT id FROM {VIEW}'))
            through_table = labels(matrix_db, await app_conn.fetch(f'SELECT id FROM {SCHEMA}.context_entries'))

        assert shadowed == 'alice'
        assert principal is None
        assert through_view == through_table == MATRIX_VISIBILITY[AccessMode.READ]['unset']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('shadow')
    async def test_a_shadow_equality_leaves_a_cleared_identity_empty(self, app_conn: asyncpg.Connection) -> None:
        # A bound transaction that ends leaves each setting reading '' for the rest of the session.
        async with scoped_transaction(app_conn, CAROL):
            pass

        async with scoped_transaction(app_conn, None):
            await app_conn.execute(SHADOW_PATH)
            shadowed = await app_conn.fetchval("SELECT ''::text = ''::text")
            cleared = await app_conn.fetchval("SELECT pg_catalog.current_setting('app.principal_id', true)")
            principal = await app_conn.fetchval(f'SELECT {SCHEMA}.mcp_context_principal_id()')
            groups = await app_conn.fetchval(f'SELECT {SCHEMA}.mcp_context_groups()')
            may_publish = await app_conn.fetchval(f'SELECT {SCHEMA}.mcp_context_may_publish()')
            may_write = await app_conn.fetchval(f'SELECT {SCHEMA}.mcp_context_may_write()')

        assert shadowed is False
        assert cleared == ''
        assert principal is None
        assert groups == []
        assert may_publish is False
        assert may_write is False


class TestUpgrade:
    """Helpers an earlier release created with a SET clause are replaced at the next boot; the boot after it issues no DDL."""

    @pytest.mark.asyncio
    async def test_helpers_in_the_earlier_form_are_replaced_once(
        self, matrix_db: ScopedDb, owner: asyncpg.Connection, pg_rls_db: RlsUrls, caplog: pytest.LogCaptureFixture,
    ) -> None:
        for name, returns, body in EARLIER_HELPERS:
            statement = (
                f'CREATE OR REPLACE FUNCTION {SCHEMA}.{name}() RETURNS {returns} '
                f'LANGUAGE sql STABLE PARALLEL SAFE SET search_path = pg_catalog, pg_temp AS $$ {body} $$'
            )
            await owner.execute(statement)
            helper = next(helper for helper in helper_functions(SCHEMA) if helper.name == name)
            earlier = comment_statement(helper, definition=digest(await read_definition(owner, helper)), preenabled=None)
            await owner.execute(earlier.replace(f'sha={helper.sha}', f'sha={digest(statement)}'))
        admin = await asyncpg.connect(pg_rls_db.admin)
        try:
            earlier_facts = (False, ['search_path=pg_catalog, pg_temp'], False, 's', 's')
            earlier_names = [name for name, _, _ in EARLIER_HELPERS]
            assert await _helper_facts(admin) == {
                name: earlier_facts if name in earlier_names else CURRENT_FACTS for name in _helper_names()
            }
            await install_ddl_log(admin)

            with caplog.at_level(logging.INFO, logger='app.migrations.rls'):
                await apply_rls(matrix_db.seed_backend, rls_settings())

            assert await ddl_log(admin) == ['CREATE FUNCTION'] * 3 + ['COMMENT'] * 3
            assert await _helper_facts(admin) == dict.fromkeys(_helper_names(), CURRENT_FACTS)
            for helper in helper_functions(SCHEMA):
                comment = await admin.fetchval("SELECT obj_description(to_regprocedure($1), 'pg_proc')", helper.anchor)
                fingerprint = parse_fingerprint(comment)
                assert fingerprint is not None
                assert fingerprint.sha == helper.sha
                assert fingerprint.definition == digest(await read_definition(admin, helper))
            [record] = [record for record in caplog.records if record.name == 'app.migrations.rls']
            assert record.levelno == logging.INFO
            assert record.getMessage().startswith('Applied row-level security')

            await admin.execute('TRUNCATE public.ddl_log')
            await apply_rls(matrix_db.seed_backend, rls_settings())
            assert await ddl_log(admin) == []
        finally:
            await admin.close()
