"""Plans of the PostgreSQL reads of ingest-managed entries, with row-level security on and off.

The range read bounds the source id; the owner page reads the first page of the caller's
members, or continues after a ``(source_id, id)`` cursor, in ``(source_id, id)`` order. All
of them compare and order the source id under the ``C`` collation, the collation the ingest
index keys it on. The range read and the page after a cursor must search the index on the
thread and the bound. The owner pages must walk the index in its own order, an ordered
index scan with no sort anywhere in the plan, so a page reads about as many rows as it
returns however large the corpus is. With row-level security on, the request role reads through the read view, which
the planner pulls up into the query, so the index survives beside the view's own
conditions; with it off, the reads name the table itself.

Each read is planned twice: the custom plan of the statement with its bound arguments, and
the generic plan a prepared statement may settle on after its first five executions, which
the driver's statement cache can make the plan of most requests. Both compare the block's
scope with the thread the statement binds, which the planner rewrites to a comparison with
that thread, so both depend on the statistics the ingest migration keeps on the scope
expression.

The tests seed a skewed corpus: one thread of ten thousand members, a tenth of them owned
by another principal and a twentieth public, beside many small threads. They refresh the
statistics, capture the statement each repository read sends and plan it as the request
backend bound to alice. The plan shows the bound on the collated key without the
collation, which the index column carries.
"""

import json
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal
from typing import cast

import asyncpg
import pytest

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import BackendAccess
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.ingest_sql import INGEST_INDEX_NAME
from app.repositories import RepositoryContainer
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_rls_scoped_db
from tests.repositories._access_scope_layouts import postgresql_scoped_db

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ALICE = AccessScope('alice', frozenset({'team-x'}))
CORPUS = 'kb:plans'
MEMBERS = 10000
FILLER_THREADS = 50
FILLER_ROWS = 2500
POLICY_ROLE = 'mcp_context_request'
VIEW_RELATION = 'context_entries_readable AS ce'
BASE_RELATION = 'context_entries ce'
COLLATED_KEY = '''((ce.metadata #>> '{ingest,source_id}') COLLATE "C")'''
PLANNED_KEY = "metadata #>> '{ingest,source_id}'"
SORT_NODES = frozenset({'Sort', 'Incremental Sort'})
PREPARED_NAME = 'ingest_plan_shape'

# Members of the corpus: bob owns every tenth, alice the rest, and every twentieth of
# alice's is public.
_SEED_MEMBERS_SQL = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility, metadata) '
    f"SELECT gen_random_uuid(), '{CORPUS}', 'agent', 'text', 'Document ' || g, "
    "CASE WHEN g % 10 = 0 THEN 'bob' ELSE 'alice' END, CASE WHEN g % 20 = 5 THEN 'public' ELSE 'private' END, "
    f"jsonb_build_object('ingest', jsonb_build_object('scope', '{CORPUS}', 'source_id', 'doc-' || lpad(g::text, 5, '0'), "
    "'meta_digest', repeat('0', 64), 'synced_at', '2026-10-18T12:00:00Z')) "
    f'FROM generate_series(1, {MEMBERS}) AS g'
)
# Rows of many small threads, as a deployment's session threads sit beside a corpus.
_SEED_FILLER_SQL = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility, metadata) '
    f"SELECT gen_random_uuid(), 'plan-filler-' || (g % {FILLER_THREADS}), 'agent', 'text', 'Filler ' || g, "
    "CASE WHEN g % 2 = 0 THEN 'alice' ELSE 'bob' END, 'private', jsonb_build_object('label', 'filler') "
    f'FROM generate_series(1, {FILLER_ROWS}) AS g'
)

type PlanMode = Literal['custom', 'generic']
PLAN_MODES: tuple[PlanMode, ...] = ('custom', 'generic')


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


class _CapturingBackend:
    """A request backend whose reads record the statements the repositories send."""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend
        self.statements: list[Captured] = []

    @property
    def backend_type(self) -> str:
        return self.backend.backend_type

    def readable_relation(self, table: str, *, alias: str) -> str:
        return self.backend.readable_relation(table, alias=alias)

    async def execute_read[T](self, operation: Callable[..., Awaitable[T]], *, access_scope: BackendAccess) -> T:
        async def _recorded(conn: AnyConnection) -> T:
            return await operation(_RecordingConnection(conn, self.statements))

        return await self.backend.execute_read(_recorded, access_scope=access_scope)


@asynccontextmanager
async def _rls_on_db(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """The database with row-level security on, its request channel logged in as ``mcp_app``."""
    async with postgresql_rls_scoped_db(
        request_url=pg_rls_db.app,
        maintenance_url=pg_rls_db.maintenance,
        policy_role=POLICY_ROLE,
        request_profile=RlsRequestProfile(read_views=RLS_READ_VIEWS),
        layout='fp32',
        monkeypatch=monkeypatch,
    ) as db:
        yield db


@asynccontextmanager
async def _rls_off_db(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """An isolated database with row-level security off, one role serving both channels."""
    async with postgresql_scoped_db(pg_test_url, 'fp32', monkeypatch) as db:
        yield db


async def _seed_corpus(db: ScopedDb) -> None:
    """Store the corpus members and the filler rows through the seed channel and refresh the statistics."""
    await db.execute(_SEED_MEMBERS_SQL)
    await db.execute(_SEED_FILLER_SQL)
    await db.execute('ANALYZE context_entries')


type Read = Callable[[RepositoryContainer], Awaitable[object]]


async def _range(repos: RepositoryContainer) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=ALICE, thread_id=CORPUS, source_id_after='doc-01000', source_id_through='doc-01500',
    )


async def _first_page(repos: RepositoryContainer) -> object:
    return await repos.context.owned_ingest_page(scope=ALICE, thread_id=CORPUS, after=None, limit=21)


async def _next_page(repos: RepositoryContainer) -> object:
    return await repos.context.owned_ingest_page(scope=ALICE, thread_id=CORPUS, after=('doc-05000', 'f' * 32), limit=21)


READS: dict[str, Read] = {
    'range': _range,
    'first-page': _first_page,
    'next-page': _next_page,
}
OWNER_PAGES = frozenset({'first-page', 'next-page'})
BOUNDED_READS = frozenset({'range', 'next-page'})


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


def _sql_literal(value: object) -> str:
    """Render a bound argument of the captured statement as a SQL literal for ``EXECUTE``."""
    if isinstance(value, bool):
        raise TypeError(f'no boolean argument is expected: {value!r}')
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, list):
        elements = ', '.join(_sql_literal(element) for element in cast('list[object]', value))
        return f'ARRAY[{elements}]::text[]'
    raise TypeError(f'unexpected argument type {type(value).__name__}: {value!r}')


async def _explain_custom(conn: AnyConnection, statement: Captured) -> str:
    """Plan the statement with its arguments bound, which the planner folds in as constants."""
    return str(await conn.fetchval(f'EXPLAIN (VERBOSE, FORMAT JSON) {statement.sql}', *statement.args))


async def _explain_generic(conn: AnyConnection, statement: Captured) -> str:
    """Plan the statement as a prepared statement held to its generic plan, which no argument value shapes."""
    arguments = ', '.join(_sql_literal(value) for value in statement.args)
    try:
        async with conn.transaction():
            await conn.execute(f'PREPARE {PREPARED_NAME} AS {statement.sql}')
            await conn.execute('SET LOCAL plan_cache_mode = force_generic_plan')
            return str(await conn.fetchval(f'EXPLAIN (VERBOSE, FORMAT JSON) EXECUTE {PREPARED_NAME}({arguments})'))
    finally:
        await conn.execute(f'DEALLOCATE {PREPARED_NAME}')


async def _planned_read(db: ScopedDb, read: Read, *, plan_mode: PlanMode) -> tuple[Captured, list[dict[str, object]]]:
    """Run ``read`` on the request channel, then plan the statement it sent as the request backend bound to alice."""
    capturing = _CapturingBackend(db.backend)
    rows = await read(RepositoryContainer(cast(StorageBackend, capturing)))
    assert rows, 'the read returned no rows, so its plan would not be the plan of a read that finds members'
    assert len(capturing.statements) == 1, capturing.statements
    statement = capturing.statements[0]

    async def _explain(conn: AnyConnection) -> str:
        if plan_mode == 'custom':
            return await _explain_custom(conn, statement)
        return await _explain_generic(conn, statement)

    document = await db.backend.execute_read(_explain, access_scope=ALICE)
    return statement, _walk(json.loads(document))


def _plan_text(statement: Captured, nodes: list[dict[str, object]]) -> str:
    """Render the statement and a summary of every plan node for an assertion message."""
    summaries = [
        json.dumps({key: node[key] for key in ('Node Type', 'Index Name', 'Index Cond', 'Filter') if key in node}, default=str)
        for node in nodes
    ]
    return '\n'.join([statement.sql, *summaries])


def _ingest_index_nodes(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the plan nodes that read the ingest index."""
    return [node for node in nodes if node.get('Index Name') == INGEST_INDEX_NAME]


def _assert_searches_the_ingest_index(statement: Captured, nodes: list[dict[str, object]]) -> None:
    """Require a bound on the collated key that the plan answers as a condition of the ingest index."""
    text = _plan_text(statement, nodes)
    assert f'{COLLATED_KEY} >' in statement.sql, text
    searches = [str(node['Index Cond']) for node in _ingest_index_nodes(nodes) if 'Index Cond' in node]
    assert len(searches) == 1, text
    assert 'thread_id' in searches[0], text
    assert PLANNED_KEY in searches[0], text


def _assert_walks_the_ingest_index_in_order(statement: Captured, nodes: list[dict[str, object]]) -> None:
    """Require an ordered scan of the ingest index on the thread and no sort anywhere in the plan."""
    text = _plan_text(statement, nodes)
    assert f'ORDER BY {COLLATED_KEY}, ce.id' in statement.sql, text
    scans = _ingest_index_nodes(nodes)
    assert len(scans) == 1, text
    assert scans[0]['Node Type'] == 'Index Scan', text
    assert 'thread_id' in str(scans[0].get('Index Cond', '')), text
    assert not [node for node in nodes if node['Node Type'] in SORT_NODES], text


def _assert_plan(read: str, statement: Captured, nodes: list[dict[str, object]], *, relation: str) -> None:
    """Hold the plan of ``read`` to the shape its kind of read requires."""
    assert f'FROM {relation} ' in statement.sql, statement.sql
    if read in OWNER_PAGES:
        _assert_walks_the_ingest_index_in_order(statement, nodes)
    if read in BOUNDED_READS:
        _assert_searches_the_ingest_index(statement, nodes)


@pytest.mark.parametrize('plan_mode', PLAN_MODES)
@pytest.mark.parametrize('read', list(READS))
@pytest.mark.asyncio
async def test_reads_use_the_ingest_index_through_the_read_view(
    read: str, plan_mode: PlanMode, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _rls_on_db(pg_rls_db, monkeypatch) as db:
        await _seed_corpus(db)

        statement, nodes = await _planned_read(db, READS[read], plan_mode=plan_mode)

    _assert_plan(read, statement, nodes, relation=VIEW_RELATION)


@pytest.mark.parametrize('plan_mode', PLAN_MODES)
@pytest.mark.parametrize('read', list(READS))
@pytest.mark.asyncio
async def test_reads_use_the_ingest_index_on_the_table_without_row_level_security(
    read: str, plan_mode: PlanMode, pg_test_url: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _rls_off_db(pg_test_url, monkeypatch) as db:
        await _seed_corpus(db)

        statement, nodes = await _planned_read(db, READS[read], plan_mode=plan_mode)

    _assert_plan(read, statement, nodes, relation=BASE_RELATION)
