"""Plans of the routed read statements under row-level security, planned as the request login role.

Policies are security conditions, and PostgreSQL turns a query's own condition into an index
condition beside them only when every function in it is leakproof; the full-text match and the
JSON accessors are not. The repositories of a row-level security request backend therefore read
``context_entries`` through its read view, which the table owner reads without policies and which
the planner pulls up into the query, so the view's conditions and the query's compete by cost
alone. The view evaluates its policy-role gate inside the identity operands and the public arm,
each a scalar subquery the planner runs once as an InitPlan, so no condition of the view names no
column: such a condition would sit above every join with the view as a One-Time Filter and keep
the planner from memoizing the view side of that join.

These tests capture the SQL the repositories send as ``mcp_app`` and plan it. The browse with a
string equality and with a string IN, the full-text page, the hybrid full-text leg, and the
vector search with a metadata filter in both embedding layouts, each planned with sequential
scans disabled, scan the index their own filter can use and evaluate the gate in InitPlans alone,
while the same statement naming the base table scans no such index under the policies. The
unique-tags count and the top tags of the statistics, which join the tags to the view, carry no
One-Time Filter and no condition naming the gate. Each runs with the application predicate, as the
request profile sends it, and without it, as the repositories send it for the system scope.

A caller with grants reads the grants table under its read policy, whose arms are inlined and hold
no subquery: in the plans of a routed read and of a child-table read, nothing beneath a grants scan
runs a subquery or reads ``context_entries``. With function statistics on, the same reads call no
helper function, while they and a direct call are counted once the principal helper takes a form
the planner cannot inline.
"""

import json
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

import asyncpg
import pytest
import pytest_asyncio

import app.backends.postgresql_backend as pg_package
from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.access_scope import Scope
from app.backends import StorageBackend
from app.backends.base import BackendAccess
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.backends.postgresql_backend.session_profiles import SessionProfile
from app.repositories import RepositoryContainer
from app.settings import get_settings
from app.tools.search.hybrid import _prepare_hybrid_fts_query
from tests.helpers import rebind_package_settings
from tests.integration.postgresql._rls_matrix import REQUEST_ROLE
from tests.integration.postgresql._rls_matrix import rls_matrix_db
from tests.integration.postgresql._rls_profiles import PinnedIdentityProfile
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import Visibility
from tests.repositories._access_scope_cases import vector_at

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ALICE = AccessScope('alice', frozenset())
# A caller that reads through grants: a user grant and a grant to its group.
CAROL = AccessScope('carol', frozenset({'team-x'}))
CAROL_GRANTED_LABEL = 'alice_private_team_read'
GRANTS_TABLE = 'context_entry_grants'
SCHEMA = '"public"'
HELPERS = ('mcp_context_principal_id', 'mcp_context_groups', 'mcp_context_may_publish', 'mcp_context_may_write')
PRINCIPAL_HELPER = f'{SCHEMA}.{HELPERS[0]}()'
FIELD = 'status'
TARGET = 'plan-target'
FTS_TERM = 'zircon'
HYBRID_QUERY = 'zircon basalt quartz feldspar'
FILLER_ROWS = 400
VIEW_RELATION = 'context_entries_readable AS '
BASE_RELATION = 'context_entries AS '
FTS_INDEX = 'idx_text_search_gin'
METADATA_INDEX = f'idx_metadata_{FIELD}'
GATE_FUNCTION = 'pg_has_role'
TAG_COLUMN = 't.tag'
# The plan-node keys that hold a condition the node applies.
CONDITION_KEYS = ('Index Cond', 'Recheck Cond', 'Filter', 'Join Filter', 'Hash Cond', 'Merge Cond', 'One-Time Filter')
# Target rows: the filters below select exactly these among the filler rows.
TARGETS: tuple[tuple[str, str, Visibility], ...] = (
    ('plan_alice_private', 'alice', 'private'),
    ('plan_alice_public', 'alice', 'public'),
    ('plan_bob_private', 'bob', 'private'),
)


@dataclass(frozen=True, slots=True)
class Captured:
    """One statement a repository sent, with its arguments as the repository bound them."""

    sql: str
    args: tuple[object, ...]


type AnyConnection = asyncpg.Connection[asyncpg.Record] | asyncpg.pool.PoolConnectionProxy[asyncpg.Record]


class _RecordingConnection:
    """A connection that records every statement before running it on the real connection."""

    def __init__(self, conn: AnyConnection, statements: list[Captured]) -> None:
        self._conn = conn
        self._statements = statements

    async def fetch(self, sql: str, *args: object) -> list[asyncpg.Record]:
        self._statements.append(Captured(sql, args))
        return await self._conn.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: object) -> asyncpg.Record | None:
        self._statements.append(Captured(sql, args))
        return await self._conn.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: object) -> object:
        self._statements.append(Captured(sql, args))
        return await self._conn.fetchval(sql, *args)

    async def execute(self, sql: str, *args: object) -> str:
        self._statements.append(Captured(sql, args))
        return await self._conn.execute(sql, *args)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


class _CapturingBackend:
    """A request backend whose reads record the statements the repositories send."""

    def __init__(self, backend: PostgreSQLBackend) -> None:
        self.backend = backend
        self.statements: list[Captured] = []

    @property
    def backend_type(self) -> str:
        return self.backend.backend_type

    def readable_relation(self, table: str, *, alias: str) -> str:
        return self.backend.readable_relation(table, alias=alias)

    async def execute_read[T](
        self, operation: Callable[..., Awaitable[T]], *args: object, access_scope: BackendAccess, **kwargs: object,
    ) -> T:
        async def _recorded(conn: AnyConnection, *op_args: object, **op_kwargs: object) -> T:
            return await operation(_RecordingConnection(conn, self.statements), *op_args, **op_kwargs)

        return await self.backend.execute_read(_recorded, *args, access_scope=access_scope, **kwargs)


@asynccontextmanager
async def capturing_repositories(
    url: str, profile: SessionProfile,
) -> AsyncGenerator[tuple[RepositoryContainer, _CapturingBackend]]:
    """Open repositories over a request backend with ``profile`` whose reads are recorded.

    Args:
        url: Connection string of the request login role.
        profile: The session profile of the backend.

    Yields:
        The repositories and the capturing backend; the backend is shut down afterwards.
    """
    backend = PostgreSQLBackend(connection_string=url, profile=profile, credentials=NO_CREDENTIAL_OVERRIDE)
    await backend.initialize()
    try:
        capturing = _CapturingBackend(backend)
        yield RepositoryContainer(cast(StorageBackend, capturing)), capturing
    finally:
        await backend.shutdown()


@pytest_asyncio.fixture(params=['fp32'])
async def plan_db(
    request: pytest.FixtureRequest, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """The policy matrix plus a few target rows among many filler rows, with fresh statistics.

    Yields:
        The seeded database in the requested layout (fp32 unless a test parametrizes it).
    """
    async with rls_matrix_db(pg_rls_db, layout=request.param, monkeypatch=monkeypatch) as db:
        for label, owner, visibility in TARGETS:
            await db.add_entry(
                label,
                owner=owner,
                visibility=visibility,
                text=f'# {label}\n\nA plan target that names the {FTS_TERM} term.\n',
                metadata={FIELD: TARGET, 'label': label},
                angle=0.2,
            )
        await db.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility, metadata) '
            "SELECT gen_random_uuid(), 'plan-filler', 'agent', 'text', 'filler row ' || g, "
            "CASE WHEN g % 2 = 0 THEN 'alice' ELSE 'bob' END, 'private', "
            f"jsonb_build_object('{FIELD}', 'filler-' || (g % 40)) FROM generate_series(1, {FILLER_ROWS}) AS g",
        )
        await db.execute('ANALYZE context_entries')
        yield db


type Shape = Callable[[RepositoryContainer, Scope], Awaitable[object]]


async def _search_eq(repos: RepositoryContainer, scope: Scope) -> object:
    return await repos.context.search_contexts(metadata={FIELD: TARGET}, limit=50, scope=scope)


async def _search_in(repos: RepositoryContainer, scope: Scope) -> object:
    filters = [{'key': FIELD, 'operator': 'in', 'value': [TARGET, 'plan-absent'], 'case_sensitive': True}]
    return await repos.context.search_contexts(metadata_filters=filters, limit=50, scope=scope)


async def _fts_page(repos: RepositoryContainer, scope: Scope) -> object:
    return await repos.fts.search(query=FTS_TERM, mode='match', limit=50, language=get_settings().fts.language, scope=scope)


async def _hybrid_fts_leg(repos: RepositoryContainer, scope: Scope) -> object:
    settings = get_settings()
    query, mode = _prepare_hybrid_fts_query(
        HYBRID_QUERY, settings.hybrid_search.fts_or_threshold, 'postgresql', settings.fts.language,
    )
    assert mode == 'boolean'
    return await repos.fts.search(
        query=query, mode=mode, limit=100, highlight=True, language=settings.fts.language, scope=scope,
    )


async def _vector_metadata(repos: RepositoryContainer, scope: Scope) -> object:
    return await repos.embeddings.search(query_embedding=vector_at(0.0), limit=20, metadata={FIELD: TARGET}, scope=scope)


# Each shape, the index its own filter can use and a fragment only the planned statement carries.
SHAPES: dict[str, tuple[Shape, str, str]] = {
    'search-eq': (_search_eq, METADATA_INDEX, f"'{FIELD}'"),
    'search-in': (_search_in, METADATA_INDEX, f"'{FIELD}'"),
    'fts-page': (_fts_page, FTS_INDEX, '@@'),
    'hybrid-fts-leg': (_hybrid_fts_leg, FTS_INDEX, '@@'),
}

# How each variant reaches the view: the request profile with the caller's predicate, or the
# identity pinned under the system scope, whose statements carry no predicate.
VARIANTS: dict[str, tuple[Callable[[], SessionProfile], Scope]] = {
    'app-predicate': (lambda: RlsRequestProfile(read_views=RLS_READ_VIEWS), ALICE),
    'view-alone': (lambda: PinnedIdentityProfile(ALICE), SYSTEM_SCOPE),
}


def _walk(node: object) -> list[dict[str, object]]:
    """Return every plan node of an EXPLAIN (FORMAT JSON) document."""
    found: list[dict[str, object]] = []
    if isinstance(node, dict):
        plan = cast('dict[str, object]', node)
        if 'Node Type' in plan:
            found.append(plan)
        for child in plan.values():
            found.extend(_walk(child))
    elif isinstance(node, list):
        for item in cast('list[object]', node):
            found.extend(_walk(item))
    return found


@dataclass(frozen=True, slots=True)
class PlanFacts:
    """Which indexes a plan searches with a condition, and where it evaluates the policy-role gate.

    An index scanned without a condition does not count: with sequential scans disabled the
    planner may read a whole partial index whose predicate the query implies, and apply the
    query's condition to every row it returns.

    Attributes:
        searched_indexes: The indexes scanned with an index condition.
        gate_initplans: How many InitPlans compute a value from the gate.
        gated_conditions: Every node condition, One-Time Filters included, that names the gate.
        one_time_filters: Every One-Time Filter of the plan.
        text: The statement and one line per plan node, for failure messages.
    """

    searched_indexes: frozenset[str]
    gate_initplans: int
    gated_conditions: tuple[str, ...]
    one_time_filters: tuple[str, ...]
    text: str


async def _plan(backend: PostgreSQLBackend, statement: Captured, *, disable_seqscan: bool) -> PlanFacts:
    """Plan ``statement`` verbosely as the request role bound to alice."""

    async def _explain(conn: asyncpg.Connection) -> str:
        if disable_seqscan:
            await conn.execute('SET LOCAL enable_seqscan = off')
        return str(await conn.fetchval(f'EXPLAIN (VERBOSE, FORMAT JSON) {statement.sql}', *statement.args))

    document = await backend.execute_read(_explain, access_scope=ALICE)
    nodes = _walk(json.loads(document))
    conditions = [str(node[key]) for node in nodes for key in CONDITION_KEYS if key in node]
    return PlanFacts(
        searched_indexes=frozenset(str(node['Index Name']) for node in nodes if 'Index Name' in node and 'Index Cond' in node),
        gate_initplans=sum(
            1 for node in nodes
            if node.get('Parent Relationship') == 'InitPlan' and GATE_FUNCTION in str(node.get('Output', ''))
        ),
        gated_conditions=tuple(condition for condition in conditions if GATE_FUNCTION in condition),
        one_time_filters=tuple(str(node['One-Time Filter']) for node in nodes if 'One-Time Filter' in node),
        text='\n'.join([statement.sql, *(_describe(node) for node in nodes)]),
    )


def _describe(node: dict[str, object]) -> str:
    """Return one line naming a plan node, what it scans, what it computes for its parent and its conditions."""
    keys = ('Parent Relationship', 'Relation Name', 'Alias', 'Index Name', 'Output', *CONDITION_KEYS)
    return ' | '.join([str(node['Node Type']), *(f'{key}: {node[key]}' for key in keys if key in node)])


async def _captured_plans(
    url: str, variant: str, shape: Shape, marker: str,
) -> list[tuple[PlanFacts, PlanFacts]]:
    """Run ``shape`` in ``variant``, then plan each captured search statement through the view and the base table."""
    profile_factory, scope = VARIANTS[variant]
    async with capturing_repositories(url, profile_factory()) as (repos, capturing):
        await shape(repos, scope)
        searches = [
            statement for statement in capturing.statements
            if VIEW_RELATION in statement.sql and marker in statement.sql
        ]
        assert searches, [statement.sql for statement in capturing.statements]
        plans: list[tuple[PlanFacts, PlanFacts]] = []
        for statement in searches:
            assert (variant == 'app-predicate') is ('owner_id' in statement.sql), statement.sql
            base = Captured(statement.sql.replace(VIEW_RELATION, BASE_RELATION), statement.args)
            plans.append((
                await _plan(capturing.backend, statement, disable_seqscan=True),
                await _plan(capturing.backend, base, disable_seqscan=True),
            ))
        return plans


def _assert_gate_in_initplans_alone(facts: PlanFacts) -> None:
    assert facts.gate_initplans > 0, facts.text
    assert facts.gated_conditions == (), facts.text
    assert facts.one_time_filters == (), facts.text


def _assert_index_through_the_view_only(plans: list[tuple[PlanFacts, PlanFacts]], index: str) -> None:
    for through_view, through_base in plans:
        assert index in through_view.searched_indexes, through_view.text
        _assert_gate_in_initplans_alone(through_view)
        assert index not in through_base.searched_indexes, through_base.text
        assert through_base.gate_initplans == 0, through_base.text
        assert through_base.gated_conditions == (), through_base.text


@pytest.mark.parametrize('variant', list(VARIANTS))
@pytest.mark.parametrize('shape', list(SHAPES))
@pytest.mark.asyncio
async def test_search_statements_keep_their_index_through_the_view(
    shape: str, variant: str, plan_db: ScopedDb, pg_rls_db: RlsUrls,
) -> None:
    """The browse and full-text statements scan their index through the view and lose it on the base table."""
    _ = plan_db
    run, index, marker = SHAPES[shape]

    _assert_index_through_the_view_only(await _captured_plans(pg_rls_db.app, variant, run, marker), index)


@pytest.mark.parametrize('variant', list(VARIANTS))
@pytest.mark.parametrize('plan_db', ['fp32', 'compressed'], indirect=True)
@pytest.mark.asyncio
async def test_vector_search_with_a_metadata_filter_keeps_the_metadata_index_through_the_view(
    variant: str, plan_db: ScopedDb, pg_rls_db: RlsUrls,
) -> None:
    """The fp32 search and the compressed candidate query filter entries through the metadata index."""
    _ = plan_db

    plans = await _captured_plans(pg_rls_db.app, variant, _vector_metadata, f"'{FIELD}'")

    _assert_index_through_the_view_only(plans, METADATA_INDEX)


@pytest.mark.parametrize('variant', list(VARIANTS))
@pytest.mark.asyncio
async def test_tag_aggregates_join_the_view_without_a_one_time_filter(
    variant: str, plan_db: ScopedDb, pg_rls_db: RlsUrls,
) -> None:
    """The unique-tags count and the top tags evaluate the gate in InitPlans, never as a condition over the join."""
    _ = plan_db
    profile_factory, scope = VARIANTS[variant]
    async with capturing_repositories(pg_rls_db.app, profile_factory()) as (repos, capturing):
        stats = await repos.statistics.get_database_statistics(scope=scope)
        aggregates = [statement for statement in capturing.statements if TAG_COLUMN in statement.sql]
        assert len(aggregates) == 2, [statement.sql for statement in aggregates]
        plans = [await _plan(capturing.backend, statement, disable_seqscan=False) for statement in aggregates]

    assert stats['unique_tags'] > 0
    for statement, facts in zip(aggregates, plans, strict=True):
        assert VIEW_RELATION in statement.sql, statement.sql
        _assert_gate_in_initplans_alone(facts)


@pytest.mark.asyncio
async def test_targets_are_what_the_filters_select(plan_db: ScopedDb, pg_rls_db: RlsUrls) -> None:
    """The planned filters select the alice-readable targets, so the plans are of statements that return rows."""
    async with capturing_repositories(pg_rls_db.app, RlsRequestProfile(read_views=RLS_READ_VIEWS)) as (repos, _):
        rows, _stats = await repos.context.search_contexts(metadata={FIELD: TARGET}, limit=50, scope=ALICE)
        fts_rows, _fts_stats = await repos.fts.search(
            query=FTS_TERM, mode='match', limit=50, language=get_settings().fts.language, scope=ALICE,
        )

    expected = {'plan_alice_private', 'plan_alice_public'}
    assert set(plan_db.labels_of(str(row['id']) for row in rows)) == expected
    assert set(plan_db.labels_of(str(row['id']) for row in fts_rows)) == expected


async def _plan_nodes(backend: PostgreSQLBackend, statement: Captured, scope: AccessScope) -> list[dict[str, object]]:
    """Plan ``statement`` verbosely as the request role bound to ``scope`` and return its nodes."""

    async def _explain(conn: asyncpg.Connection) -> str:
        return str(await conn.fetchval(f'EXPLAIN (VERBOSE, FORMAT JSON) {statement.sql}', *statement.args))

    return _walk(json.loads(await backend.execute_read(_explain, access_scope=scope)))


def _conditions(node: dict[str, object]) -> list[str]:
    return [str(node[key]) for key in CONDITION_KEYS if key in node]


async def _grant_reads_as_carol(plan_db: ScopedDb, url: str) -> list[tuple[Captured, list[dict[str, object]]]]:
    """Run a routed read and a child-table read as carol, then plan each statement that reads grants.

    Returns:
        Each captured statement with the nodes of its plan.
    """
    async with capturing_repositories(url, RlsRequestProfile(read_views=RLS_READ_VIEWS)) as (repos, capturing):
        await repos.context.search_contexts(limit=50, scope=CAROL)
        await repos.tags.get_tags_for_context(plan_db.ids[CAROL_GRANTED_LABEL], scope=CAROL)
        reads = [statement for statement in capturing.statements if 'context_entry_grants' in statement.sql]
        reads.extend(statement for statement in capturing.statements if statement.sql.startswith('SELECT tag FROM tags'))
        assert len(reads) == 2, [statement.sql for statement in capturing.statements]
        return [(statement, await _plan_nodes(capturing.backend, statement, CAROL)) for statement in reads]


@pytest.mark.asyncio
async def test_the_grants_scan_holds_no_subplan_and_reads_no_entries(plan_db: ScopedDb, pg_rls_db: RlsUrls) -> None:
    """Under the grants read policy, the routed read and the child-table read scan the grants table alone.

    The policy's owner arm compares the grant's recorded entry owner with the inlined
    identity, so nothing beneath a scan of ``context_entry_grants`` runs a subquery or
    reads ``context_entries``, and the scan applies the owner arm itself.
    """
    owner_arms = 0
    for statement, nodes in await _grant_reads_as_carol(plan_db, pg_rls_db.app):
        grant_scans = [node for node in nodes if node.get('Relation Name') == GRANTS_TABLE]
        assert grant_scans, statement.sql
        for scan in grant_scans:
            beneath = [node for node in _walk(scan.get('Plans', [])) if node is not scan]
            assert all(node.get('Parent Relationship') != 'SubPlan' for node in beneath), statement.sql
            assert all(node.get('Relation Name') != 'context_entries' for node in beneath), statement.sql
            for condition in (*_conditions(scan), *(text for node in beneath for text in _conditions(node))):
                assert 'SubPlan' not in condition, condition
                assert 'context_entries' not in condition, condition
            owner_arms += any('entry_owner_id' in condition for condition in _conditions(scan))
    assert owner_arms > 0


async def _helper_calls(admin: asyncpg.Connection) -> dict[str, int]:
    """Read how often each helper was called, in a fresh statistics snapshot."""
    await admin.execute('SELECT pg_stat_clear_snapshot()')
    rows = await admin.fetch(
        'SELECT p.proname, COALESCE(s.calls, 0) FROM pg_proc p '
        'LEFT JOIN pg_stat_user_functions s ON s.funcid = p.oid '
        "WHERE p.pronamespace = 'public'::regnamespace AND p.proname = ANY($1::text[])",
        list(HELPERS),
    )
    return {str(row[0]): int(row[1]) for row in rows}


async def _flush(conn: asyncpg.Connection) -> None:
    """Ask the session to flush its statistics when its transaction commits."""
    await conn.execute('SELECT pg_stat_force_next_flush()')


async def _call_principal_then_flush(conn: asyncpg.Connection) -> None:
    """Call the principal helper by name, then ask for the flush at commit."""
    await conn.fetchval(f'SELECT {PRINCIPAL_HELPER}')
    await _flush(conn)


async def _track_functions(conn: asyncpg.Connection) -> str:
    return str(await conn.fetchval('SHOW track_functions'))


async def _read_then_flush(repos: RepositoryContainer, backend: PostgreSQLBackend, plan_db: ScopedDb) -> None:
    """Run the routed read and the child-table read as carol, then flush the session's statistics."""
    await repos.context.search_contexts(limit=50, scope=CAROL)
    await repos.tags.get_tags_for_context(plan_db.ids[CAROL_GRANTED_LABEL], scope=CAROL)
    await backend.execute_write(_flush, access_scope=CAROL)


@pytest.mark.asyncio
async def test_routed_and_child_reads_call_no_helper(
    plan_db: ScopedDb, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helpers are inlined into every policy and view condition, so a read calls none of them by name.

    With ``track_functions`` at ``all`` every function call is counted except a SQL function
    the planner inlines, and a direct call of an inlinable helper is inlined as well. So the
    control first gives the principal helper a SET clause, which keeps the planner from
    inlining it: then a direct call and the same reads are counted, which shows the counters
    work in this session and that the reads reach the helper. With the SET clause reset, the
    same reads add no call. The request backend holds one connection, so every call, read
    and flush shares one session, and each flush is forced at its commit, since the
    counters otherwise reach the shared statistics up to a second later.
    """
    settings = get_settings()
    storage = settings.storage.model_copy(update={'postgresql_pool_min': 1, 'postgresql_pool_max': 1})
    rebind_package_settings(monkeypatch, pg_package, settings.model_copy(update={'storage': storage}))
    admin = await asyncpg.connect(pg_rls_db.admin)
    owner = await asyncpg.connect(pg_rls_db.maintenance)
    database = quote_pg_identifier(str(await admin.fetchval('SELECT current_database()')))
    await admin.execute(f"ALTER DATABASE {database} SET track_functions = 'all'")
    await admin.execute(f'GRANT EXECUTE ON FUNCTION pg_catalog.pg_stat_force_next_flush() TO {REQUEST_ROLE}')
    try:
        async with capturing_repositories(pg_rls_db.app, RlsRequestProfile(read_views=RLS_READ_VIEWS)) as (repos, capturing):
            backend = capturing.backend
            assert await backend.execute_read(_track_functions, access_scope=CAROL) == 'all'
            before = await _helper_calls(admin)

            await owner.execute(f'ALTER FUNCTION {PRINCIPAL_HELPER} SET search_path = pg_catalog, pg_temp')
            await backend.execute_write(_call_principal_then_flush, access_scope=CAROL)
            direct = await _helper_calls(admin)
            await _read_then_flush(repos, backend, plan_db)
            uninlined = await _helper_calls(admin)

            await owner.execute(f'ALTER FUNCTION {PRINCIPAL_HELPER} RESET search_path')
            await _read_then_flush(repos, backend, plan_db)
            after = await _helper_calls(admin)
    finally:
        await owner.close()
        await admin.execute(f'ALTER DATABASE {database} RESET track_functions')
        await admin.close()

    principal = HELPERS[0]
    assert direct[principal] == before[principal] + 1, (before, direct)
    assert uninlined[principal] > direct[principal], (direct, uninlined)
    assert after == uninlined
