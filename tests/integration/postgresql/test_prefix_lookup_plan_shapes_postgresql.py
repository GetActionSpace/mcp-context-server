"""Plans and results of the PostgreSQL short-id lookup, with row-level security on and off.

``find_ids_by_prefix`` bounds the native ``uuid`` id by the prefix padded with ``0`` and
with ``f`` (:func:`app.ids.id_prefix_bounds`). A ``uuid`` compares as its 16 bytes, the
order of its lowercase hex digits, so the ids sharing the prefix form one range of every
index whose first column is the id. The plan reads that range with the two bounds as its
index condition and returns it in the ``ORDER BY id`` order, with no sequential scan and
no sort. Which such index the planner picks is its own choice: the custom plan tends to
the unique ``(id, owner_id)`` index, the generic plan to the primary key, so the pins
accept any index of ``context_entries`` whose first column is ``id``, read from the
catalog. With row-level security on, the request role reads through the read view, which
the planner pulls up into the query, so the index survives beside the view's own
conditions; with it off, the read names the table.

Each statement is planned twice: the custom plan with its bound arguments, and the
generic plan a prepared statement may settle on after its first five executions. The
table holds ten thousand entries of two owners after ``ANALYZE``. The result test compares
the lookup with the readable ids that start with each prefix, computed from the stored rows
and grants.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from contextlib import asynccontextmanager
from typing import Literal
from typing import cast

import pytest

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.repositories import RepositoryContainer
from tests.helpers import Captured
from tests.helpers import CapturingBackend
from tests.integration.postgresql._plans import explain_custom
from tests.integration.postgresql._plans import explain_generic
from tests.integration.postgresql._plans import plan_nodes
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_rls_scoped_db
from tests.repositories._access_scope_layouts import postgresql_scoped_db
from tests.repositories.context_repository._prefix_lookup import assert_lookup_matches_readable_ids
from tests.repositories.context_repository._prefix_lookup import prefixes_of

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ALICE = AccessScope('alice', frozenset({'team-x'}))
ROWS = 10000
POLICY_ROLE = 'mcp_context_request'
VIEW_RELATION = 'context_entries_readable AS context_entries'
BASE_RELATION = 'context_entries'
PRIMARY_KEY = 'context_entries_pkey'
INDEX_SCANS = frozenset({'Index Scan', 'Index Only Scan'})
SORT_NODES = frozenset({'Sort', 'Incremental Sort'})
PREFIX = 'a5a5a5a5'

# Bob owns every tenth entry, alice the rest; every twentieth of alice's is public.
_SEED_SQL = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
    "SELECT gen_random_uuid(), 'thread-' || (g % 50), 'agent', 'text', 'Entry ' || g, "
    "CASE WHEN g % 10 = 0 THEN 'bob' ELSE 'alice' END, CASE WHEN g % 20 = 5 THEN 'public' ELSE 'private' END "
    f'FROM generate_series(1, {ROWS}) AS g'
)
# Two of alice's rows share the prefix with one of bob's, so the two-row lookup is ambiguous.
_PREFIXED = (
    (PREFIX + '1' * 24, 'alice', 'private'),
    (PREFIX + '2' * 24, 'alice', 'private'),
    (PREFIX + '3' * 24, 'bob', 'private'),
)
# Prefixes the result test adds to those cut from the stored ids: the prefixed rows, an
# odd length among them, one only bob's row carries, an absent one and both extremes.
_EXTRA_PREFIXES = frozenset({PREFIX, PREFIX + '1', PREFIX + '3', 'c7c7c7c7', 'ffffffff', '00000000'})

type PlanMode = Literal['custom', 'generic']
PLAN_MODES: tuple[PlanMode, ...] = ('custom', 'generic')

_ID_LEADING_INDEXES_SQL = (
    'SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid '
    'JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0] '
    "WHERE i.indrelid = 'context_entries'::regclass AND a.attname = 'id'"
)


@asynccontextmanager
async def _rls_on_db(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """The seeded database with row-level security on, its request channel logged in as ``mcp_app``."""
    async with postgresql_rls_scoped_db(
        request_url=pg_rls_db.app,
        maintenance_url=pg_rls_db.maintenance,
        policy_role=POLICY_ROLE,
        request_profile=RlsRequestProfile(read_views=RLS_READ_VIEWS),
        layout='fp32',
        monkeypatch=monkeypatch,
    ) as db:
        await _seed(db)
        yield db


@asynccontextmanager
async def _rls_off_db(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """The seeded database with row-level security off, one role serving both channels."""
    async with postgresql_scoped_db(pg_test_url, 'fp32', monkeypatch) as db:
        await _seed(db)
        yield db


async def _seed(db: ScopedDb) -> None:
    """Store the random ids and the prefixed rows through the seed channel and refresh the statistics."""
    await db.execute(_SEED_SQL)
    for context_id, owner, visibility in _PREFIXED:
        await db.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
            "VALUES (?, 'prefixed', 'agent', 'text', 'Prefixed', ?, ?)",
            (context_id, owner, visibility),
        )
    await db.execute('ANALYZE context_entries')


@pytest.fixture(params=['rls_on', 'rls_off'])
def seeded_db(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> AbstractAsyncContextManager[ScopedDb]:
    """The seeded database of the parameter's setting, to be entered by the test.

    The fixture of the setting's database is requested during setup, outside the test's
    event loop, so only the setting a test runs under creates a database.

    Returns:
        The context manager that prepares, seeds and yields the database.
    """
    if request.param == 'rls_on':
        return _rls_on_db(cast(RlsUrls, request.getfixturevalue('pg_rls_db')), monkeypatch)
    return _rls_off_db(cast(str, request.getfixturevalue('pg_test_url')), monkeypatch)


async def _planned_lookup(db: ScopedDb, plan_mode: PlanMode) -> tuple[list[str], Captured, list[dict[str, object]]]:
    """Run the two-row lookup of :data:`PREFIX` as alice, then plan the statement it sent as the request backend."""
    capturing = CapturingBackend(db.backend)
    found = await RepositoryContainer(cast(StorageBackend, capturing)).context.find_ids_by_prefix(PREFIX, limit=2, scope=ALICE)
    assert len(capturing.statements) == 1, capturing.statements
    statement = capturing.statements[0]

    explain = explain_custom if plan_mode == 'custom' else explain_generic
    document = await db.backend.execute_read(explain, statement, access_scope=ALICE)
    return found, statement, plan_nodes(document)


async def _id_leading_indexes(db: ScopedDb) -> frozenset[str]:
    """Return the indexes of ``context_entries`` whose first column is the id."""
    names = frozenset(str(row[0]) for row in await db.fetch_all(_ID_LEADING_INDEXES_SQL))
    assert PRIMARY_KEY in names, names
    return names


def _summary(nodes: list[dict[str, object]]) -> str:
    """Render the identifying fields of every plan node for an assertion message."""
    keys = ('Node Type', 'Relation Name', 'Index Name', 'Index Cond', 'Filter')
    return '\n'.join(json.dumps({key: node[key] for key in keys if key in node}, default=str) for node in nodes)


def _assert_id_index_range(
    statement: Captured, nodes: list[dict[str, object]], *, relation: str, id_indexes: frozenset[str],
) -> None:
    """Assert the one read of ``context_entries`` is an index scan of an id-leading index bounded on both sides."""
    text = statement.sql + '\n' + _summary(nodes)
    assert f'FROM {relation}\n' in statement.sql, text
    scans = [node for node in nodes if node.get('Relation Name') == 'context_entries']
    assert len(scans) == 1, text
    assert scans[0]['Node Type'] in INDEX_SCANS, text
    assert scans[0].get('Index Name') in id_indexes, text
    condition = str(scans[0].get('Index Cond', ''))
    assert 'id >=' in condition, text
    assert 'id <=' in condition, text
    assert not [node for node in nodes if node['Node Type'] in SORT_NODES], text


@pytest.mark.parametrize('plan_mode', PLAN_MODES)
@pytest.mark.asyncio
async def test_prefix_lookup_scans_an_id_index_range_through_the_read_view(
    plan_mode: PlanMode, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _rls_on_db(pg_rls_db, monkeypatch) as db:
        found, statement, nodes = await _planned_lookup(db, plan_mode)
        id_indexes = await _id_leading_indexes(db)

    assert len(found) == 2, found
    _assert_id_index_range(statement, nodes, relation=VIEW_RELATION, id_indexes=id_indexes)


@pytest.mark.parametrize('plan_mode', PLAN_MODES)
@pytest.mark.asyncio
async def test_prefix_lookup_scans_an_id_index_range_on_the_table_without_row_level_security(
    plan_mode: PlanMode, pg_test_url: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _rls_off_db(pg_test_url, monkeypatch) as db:
        found, statement, nodes = await _planned_lookup(db, plan_mode)
        id_indexes = await _id_leading_indexes(db)

    assert len(found) == 2, found
    _assert_id_index_range(statement, nodes, relation=BASE_RELATION, id_indexes=id_indexes)


@pytest.mark.asyncio
async def test_prefix_lookup_matches_the_readable_ids_with_the_prefix(
    seeded_db: AbstractAsyncContextManager[ScopedDb],
) -> None:
    async with seeded_db as db:
        lowest = [str(row[0]) for row in await db.fetch_all('SELECT id FROM context_entries ORDER BY id LIMIT 60')]

        await assert_lookup_matches_readable_ids(db, ALICE, prefixes_of(lowest) | _EXTRA_PREFIXES)
