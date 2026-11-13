"""Query-plan pins for the SQLite reads of ingest-managed entries.

Both reads select the source id, the block, the content hash and the access columns of
the members of one thread. The ingest index carries every one of them, so SQLite answers
the reads from index pages and never walks a row's text overflow pages to reach
``metadata``: the range read searches the index on the thread and the source-id bounds,
and the owner page reads it in ``(source_id, id)`` order, so neither sorts.

The pins run on databases that the server's startup preparation (``prepare_database``)
builds, so every index a server has competes with the ingest index, and each read runs
through the repository as a request principal with and without groups, its statement
planned with the parameters it bound. Two databases are planned. One holds the members of
a corpus thread beside the rows of one other thread and carries no ``sqlite_stat1``
table, as a database is before any ``ANALYZE``. The other holds a corpus thread far larger
than the database's average thread, beside many small threads, after ``ANALYZE``, the
statistics the server's own ``PRAGMA optimize`` writes too: their averages make
``idx_context_owner_thread`` and ``idx_thread_id`` look more selective than the ingest
index, so the plans hold there only because the reads name the ingest index.
"""

import re
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast

import pytest
import pytest_asyncio

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.ingest_sql import INGEST_INDEX_NAME
from app.repositories import RepositoryContainer
from tests.helpers import Captured
from tests.helpers import CapturingBackend
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import sqlite_scoped_db

CORPUS = 'kb:plans'
MEMBERS = 200
FILLER_ROWS = 200
# The analyzed database: a corpus thread of thousands of members beside many threads of a
# few rows each, so the average thread holds a few rows and the corpus thread thousands.
SKEWED_MEMBERS = 4000
SMALL_THREADS = 400
ROWS_PER_SMALL_THREAD = 10
# The databases the reads are planned on: before any ANALYZE, and analyzed on the skewed corpus.
_DATABASES = ['before-analyze', 'analyzed-skewed']
# A plan step that reads context_entries under the alias both reads use.
_ENTRY_STEP = re.compile(r'^(?:SCAN|SEARCH) ce\b')
# A search of the ingest index that starts at the corpus thread, never a scan of every managed row.
_INGEST_INDEX_STEP = re.compile(rf'^SEARCH ce USING (?:COVERING )?INDEX {INGEST_INDEX_NAME} \(thread_id=\?')
_SORT_STEP = 'USE TEMP B-TREE FOR ORDER BY'

_SCOPES = [
    pytest.param(AccessScope('alice', frozenset()), id='principal-only'),
    pytest.param(AccessScope('alice', frozenset({'team-x'})), id='principal-with-groups'),
]

# Members and filler rows: alice owns the even members, bob the odd ones; every fifth is public.
_SEED_SQL = (
    'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) '
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id, visibility) '
    "SELECT lower(hex(randomblob(16))), ?, 'agent', 'text', 'Document ' || i, "
    "json_object('ingest', json_object('scope', ?, 'source_id', printf('doc-%04d', i), "
    "'meta_digest', ?, 'synced_at', '2026-10-18T12:00:00Z')), "
    "CASE WHEN i % 2 = 0 THEN 'alice' ELSE 'bob' END, "
    "CASE WHEN i % 5 = 0 THEN 'public' ELSE 'private' END FROM n"
)
_FILLER_SQL = (
    'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) '
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id, visibility) '
    "SELECT lower(hex(randomblob(16))), 'plan-filler', 'agent', 'text', 'Filler ' || i, "
    "json_object('label', 'filler'), 'alice', 'private' FROM n"
)
# Rows spread round-robin over the small threads, two of every three owned by alice.
_SMALL_THREADS_SQL = (
    'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) '
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id, visibility) '
    "SELECT lower(hex(randomblob(16))), 'session-' || (i % ?), 'agent', 'text', 'Session row ' || i, "
    "json_object('status', 'done'), CASE WHEN i % 3 = 0 THEN 'bob' ELSE 'alice' END, 'private' FROM n"
)


@pytest_asyncio.fixture(params=_DATABASES)
async def plan_db(
    request: pytest.FixtureRequest, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """A database prepared by server startup, holding the members of :data:`CORPUS` and rows of other threads.

    ``before-analyze`` holds :data:`MEMBERS` members beside one thread of filler rows and
    no statistics; ``analyzed-skewed`` holds :data:`SKEWED_MEMBERS` members beside
    :data:`SMALL_THREADS` small threads, and ``ANALYZE`` has run on it.

    Yields:
        The seeded database.
    """
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        if request.param == 'before-analyze':
            await db.execute(_SEED_SQL, (MEMBERS, CORPUS, CORPUS, '0' * 64))
            await db.execute(_FILLER_SQL, (FILLER_ROWS,))
        else:
            await db.execute(_SEED_SQL, (SKEWED_MEMBERS, CORPUS, CORPUS, '0' * 64))
            await db.execute(_SMALL_THREADS_SQL, (SMALL_THREADS * ROWS_PER_SMALL_THREAD, SMALL_THREADS))
            await db.execute('ANALYZE')
        yield db


type Read = Callable[[RepositoryContainer, AccessScope], Awaitable[object]]


async def _range_unbounded(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after=None, source_id_through=None,
    )


async def _range_bounded(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after='doc-0050', source_id_through='doc-0100',
    )


async def _range_after(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after='doc-0150', source_id_through=None,
    )


async def _range_through(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after=None, source_id_through='doc-0050',
    )


async def _first_page(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.owned_ingest_page(scope=scope, thread_id=CORPUS, after=None, limit=21)


async def _next_page(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.owned_ingest_page(scope=scope, thread_id=CORPUS, after=('doc-0100', 'f' * 32), limit=21)


# Each read and the constraints its index search must carry beyond the thread equality.
READS: dict[str, tuple[Read, tuple[str, ...]]] = {
    'range-unbounded': (_range_unbounded, ()),
    'range-bounded': (_range_bounded, ('<expr>>?', '<expr><?')),
    'range-after': (_range_after, ('<expr>>?',)),
    'range-through': (_range_through, ('<expr><?',)),
    'first-page': (_first_page, ()),
    'next-page': (_next_page, ('<expr>>?',)),
}


async def _captured_statement(db: ScopedDb, read: Read, scope: AccessScope) -> Captured:
    """Run ``read`` as ``scope`` and return the one statement it sent."""
    capturing = CapturingBackend(db.backend)
    rows = await read(RepositoryContainer(cast(StorageBackend, capturing)), scope)
    assert rows, 'the read returned no rows, so its plan would not be the plan of a read that finds members'
    assert len(capturing.statements) == 1, capturing.statements
    return capturing.statements[0]


async def _plan(backend: StorageBackend, statement: Captured) -> list[str]:
    """Return the detail column of every ``EXPLAIN QUERY PLAN`` row of ``statement``."""

    def _explain(conn: sqlite3.Connection) -> list[str]:
        return [str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {statement.sql}', statement.params).fetchall()]

    return await backend.execute_read(_explain, access_scope=MAINTENANCE)


@pytest.mark.parametrize('scope', _SCOPES)
@pytest.mark.parametrize('read', list(READS))
@pytest.mark.asyncio
async def test_ingest_reads_search_the_ingest_index_without_sorting(
    read: str, scope: AccessScope, plan_db: ScopedDb,
) -> None:
    run, constraints = READS[read]
    statement = await _captured_statement(plan_db, run, scope)

    plan = await _plan(plan_db.backend, statement)

    entry_steps = [step for step in plan if _ENTRY_STEP.match(step)]
    assert len(entry_steps) == 1, f'{statement.sql!r} reads context_entries more than once or never: {plan}'
    step = entry_steps[0]
    assert _INGEST_INDEX_STEP.match(step), f'{statement.sql!r} reads context_entries outside the ingest index: {plan}'
    for constraint in constraints:
        assert constraint in step, f'{statement.sql!r} does not search the index on {constraint}: {plan}'
    assert not any(_SORT_STEP in planned for planned in plan), f'{statement.sql!r} sorts: {plan}'


@pytest.mark.parametrize('plan_db', ['analyzed-skewed'], indirect=True)
@pytest.mark.asyncio
async def test_the_analyzed_statistics_favor_the_owner_thread_index(plan_db: ScopedDb) -> None:
    """``sqlite_stat1`` describes the ingest index, and its averages rank the owner-thread index ahead of it.

    The first average of the ingest index counts the rows per thread, which on the skewed
    corpus is the whole corpus thread; the owner-thread index's second average counts the
    rows per owner and thread, a few rows. Planned by those averages alone, the owner pages
    would read the owner-thread index and sort.
    """
    rows = await plan_db.fetch_all("SELECT idx, stat FROM sqlite_stat1 WHERE tbl = 'context_entries'")
    stats = {str(name): [int(value) for value in str(stat).split()] for name, stat in rows}

    assert INGEST_INDEX_NAME in stats, sorted(stats)
    rows_per_thread = stats[INGEST_INDEX_NAME][1]
    rows_per_owner_and_thread = stats['idx_context_owner_thread'][2]
    assert rows_per_thread > 10 * rows_per_owner_and_thread, stats
