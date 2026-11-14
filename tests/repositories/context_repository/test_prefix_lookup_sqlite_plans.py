"""Query plans and results of the SQLite short-id lookup.

``find_ids_by_prefix`` bounds the id by the prefix padded with ``0`` and with ``f``
(:func:`app.ids.id_prefix_bounds`), so the ids sharing the prefix form one range of an
index whose first column is the id, and SQLite answers the lookup with a search of that
range instead of visiting every entry. The pins run on databases that the server's startup
preparation builds, so every index a server has competes for the read: one before any
``ANALYZE`` and one after it, the statistics the server's own ``PRAGMA optimize`` writes
too. The result test compares the lookup with the readable ids that start with each
prefix, computed from the stored rows and grants.
"""

import re
from collections.abc import AsyncGenerator
from typing import cast

import pytest
import pytest_asyncio

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.repositories import RepositoryContainer
from tests.helpers import Captured
from tests.helpers import CapturingBackend
from tests.helpers import sqlite_query_plan
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import sqlite_scoped_db
from tests.repositories.context_repository._prefix_lookup import assert_lookup_matches_readable_ids
from tests.repositories.context_repository._prefix_lookup import prefixes_of

ROWS = 4000
_DATABASES = ['before-analyze', 'analyzed']
_ENTRY_STEP = re.compile(r'^(?:SCAN|SEARCH) context_entries\b')
# SQLite prints the ``<=`` bound as ``id<?``.
_RANGE_STEP = re.compile(r'^SEARCH context_entries USING (?:COVERING )?INDEX (\S+) \(id>\? AND id<\?\)$')
_SORT_STEP = 'USE TEMP B-TREE FOR ORDER BY'

ALICE = AccessScope('alice', frozenset())
ALICE_WITH_GROUPS = AccessScope('alice', frozenset({'team-x'}))
_SCOPES = [pytest.param(ALICE, id='principal-only'), pytest.param(ALICE_WITH_GROUPS, id='principal-with-groups')]

# Alice owns two of every three rows; every fifth row is public.
_SEED_SQL = (
    'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) '
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
    "SELECT lower(hex(randomblob(16))), 'thread-' || (i % 50), 'agent', 'text', 'Entry ' || i, "
    "CASE WHEN i % 3 = 0 THEN 'bob' ELSE 'alice' END, CASE WHEN i % 5 = 0 THEN 'public' ELSE 'private' END FROM n"
)
# Two readable rows share one prefix, one more row under it is bob's and private.
_PREFIXED = (
    ('a5a5a5a5' + '1' * 24, 'alice', 'private'),
    ('a5a5a5a5' + '2' * 24, 'alice', 'private'),
    ('a5a5a5a5' + '3' * 24, 'bob', 'private'),
    ('b6b6b6b6' + '4' * 24, 'bob', 'private'),
)
# Ambiguous, odd length, the longest prefix, a prefix only bob's private row carries, absent.
PREFIXES = ['a5a5a5a5', 'a5a5a5a51', 'a5a5a5a5' + '2' * 23, 'b6b6b6b6', 'c7c7c7c7']


@pytest_asyncio.fixture(params=_DATABASES)
async def plan_db(
    request: pytest.FixtureRequest, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """A prepared database of random ids beside the prefixed rows, with or without statistics."""
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        await db.execute(_SEED_SQL, (ROWS,))
        for context_id, owner, visibility in _PREFIXED:
            await db.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
                "VALUES (?, 'prefixed', 'agent', 'text', 'Prefixed', ?, ?)",
                (context_id, owner, visibility),
            )
        if request.param == 'analyzed':
            await db.execute('ANALYZE')
        yield db


async def _captured_lookup(db: ScopedDb, prefix: str, scope: AccessScope) -> Captured:
    """Run the two-row lookup and return the one statement it sent."""
    capturing = CapturingBackend(db.backend)
    await RepositoryContainer(cast(StorageBackend, capturing)).context.find_ids_by_prefix(prefix, limit=2, scope=scope)
    assert len(capturing.statements) == 1, capturing.statements
    return capturing.statements[0]


async def _leading_column(db: ScopedDb, index: str) -> str:
    """Return the first column of an index."""
    rows = await db.fetch_all(f"SELECT name FROM pragma_index_info('{index}') ORDER BY seqno LIMIT 1")
    return str(rows[0][0])


@pytest.mark.parametrize('scope', _SCOPES)
@pytest.mark.parametrize('prefix', PREFIXES)
@pytest.mark.asyncio
async def test_prefix_lookup_searches_an_id_index_range(prefix: str, scope: AccessScope, plan_db: ScopedDb) -> None:
    statement = await _captured_lookup(plan_db, prefix, scope)

    plan = await sqlite_query_plan(plan_db.backend, statement.sql, statement.params)

    entry_steps = [step for step in plan if _ENTRY_STEP.match(step)]
    assert len(entry_steps) == 1, plan
    match = _RANGE_STEP.match(entry_steps[0])
    assert match, plan
    assert await _leading_column(plan_db, match.group(1)) == 'id', plan
    assert not any(_SORT_STEP in step for step in plan), plan


@pytest.mark.parametrize('scope', _SCOPES)
@pytest.mark.asyncio
async def test_prefix_lookup_matches_the_readable_ids_with_the_prefix(scope: AccessScope, plan_db: ScopedDb) -> None:
    lowest = [str(row[0]) for row in await plan_db.fetch_all('SELECT id FROM context_entries ORDER BY id LIMIT 60')]

    await assert_lookup_matches_readable_ids(plan_db, scope, prefixes_of(lowest) | set(PREFIXES))
