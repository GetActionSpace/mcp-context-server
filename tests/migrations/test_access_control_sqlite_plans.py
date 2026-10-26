"""Query-plan pins for scoped SQLite statements over ``context_entries``.

The READ predicate ORs an owner match, a public match and a correlated EXISTS over
``context_entry_grants``. SQLite cannot answer that OR through separate index lookups,
so a scoped statement without an indexable client filter visits every
``context_entries`` row, and ``owner_id``/``visibility`` sit after ``text_content`` in
the row, so reading them from the table walks each row's text overflow pages. The
SQLite covering access indexes the access-control migration provisions let these
statements read index pages only.

The pins run on a database that the server's startup preparation (``prepare_database``)
builds with embedding compression on, as it is by default, and that holds the
access-scope seed rows. The database carries no ``sqlite_stat1`` table, so the plans are
the ones a database gets before any ``ANALYZE`` has run.

- ``get_statistics``: the tool runs with every block whose figures derive from stored
  entries enabled, and the backend records every statement it executes. The plan of
  each statement that names ``context_entries`` reads it, under its table name or the
  ``ce`` alias, only through a covering access index; the one exception is the summary
  count, which tests the ``summary`` column that no index carries. No statement that a
  virtual table runs internally reads ``context_entries``: a query on the
  external-content FTS table without ``MATCH`` passes through to ``context_entries`` and
  reads each row's text, which ``EXPLAIN QUERY PLAN`` does not show.
- The semantic-search candidate statement, with and without client filters, reads
  ``context_entries`` only through a covering access index.
"""

import re
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

import app.startup
import app.tools
from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.repositories import RepositoryContainer
from tests.helpers import as_principal
from tests.repositories._access_scope_layouts import sqlite_scoped_db

# A plan step that reads context_entries, under its table name or the statistics alias.
_ENTRY_STEP = re.compile(r'^(?:SCAN|SEARCH) (?:context_entries|ce)\b')
_COVERING_ACCESS_INDEX = 'USING COVERING INDEX idx_context_access_'
_NAMES_CONTEXT_ENTRIES = re.compile(r'\bcontext_entries\b')
# The SQLite trace callback prefixes every statement a virtual table runs internally.
_INTERNAL_STATEMENT = '-- '
# The summary count tests the summary column, which no access index carries.
_SUMMARY_COUNT = "summary IS NOT NULL AND summary != ''"
# One marker per entry-derived figure of get_statistics: each must name a traced statement.
_SCOPED_FIGURE_MARKERS = (
    'SELECT COUNT(*) AS count FROM context_entries WHERE',
    'GROUP BY source',
    'GROUP BY content_type',
    'GROUP BY thread_id',
    'FROM image_attachments i',
    'FROM tags t',
    'FROM embedding_metadata em',
    'FROM context_entries_fts',
    'FROM context_index_nodes n',
    'FROM generation_jobs gj',
    _SUMMARY_COUNT,
)

_SCOPES = [
    pytest.param(AccessScope('local', frozenset()), id='principal-only'),
    pytest.param(AccessScope('alice', frozenset({'team-a', 'team-b'})), id='principal-with-groups'),
]


@pytest_asyncio.fixture
async def startup_backend(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend prepared by server startup with compression on and seeded with the access rows."""
    async with sqlite_scoped_db(async_db_initialized, 'compressed', monkeypatch) as db:
        yield db.backend


async def _plan(backend: StorageBackend, sql: str, params: list[Any]) -> list[str]:
    """Return the detail column of every ``EXPLAIN QUERY PLAN`` row for one statement."""

    def _explain(conn: sqlite3.Connection) -> list[str]:
        return [str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params).fetchall()]

    return await backend.execute_read(_explain, access_scope=MAINTENANCE)


def _assert_entries_read_from_covering_index(sql: str, plan: list[str]) -> None:
    """Require at least one context_entries step and a covering access index on every one."""
    entry_steps = [step for step in plan if _ENTRY_STEP.match(step)]
    assert entry_steps, f'no context_entries step in the plan of {sql!r}: {plan}'
    uncovered = [step for step in entry_steps if _COVERING_ACCESS_INDEX not in step]
    assert not uncovered, f'{sql!r} reads context_entries outside a covering access index: {plan}'


def _statistics_settings() -> MagicMock:
    """Tool settings with every block whose figures derive from stored entries enabled."""
    settings = MagicMock()
    settings.embedding.generation_enabled = True
    settings.compression.enabled = True
    settings.semantic_search.enabled = True
    settings.fts.enabled = True
    settings.summary.generation_enabled = True
    settings.index_tree.node_summaries_enabled = True
    settings.reranking.enabled = False
    return settings


async def _get_statistics_statements(
    backend: StorageBackend, scope: AccessScope, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Run the ``get_statistics`` tool as the scope's principal and capture every statement it executed.

    Returns:
        The executed statements in order. The SQLite trace callback reports each one with
        its bound values inlined, so each text is the exact statement that ran; a statement
        a virtual table ran internally carries the ``-- `` prefix.
    """
    statements: list[str] = []
    execute_read = backend.execute_read

    async def _tracing_execute_read(operation: Callable[..., object], *args: Any, **kwargs: Any) -> object:
        def _traced(conn: sqlite3.Connection, *inner_args: Any, **inner_kwargs: Any) -> object:
            conn.set_trace_callback(statements.append)
            try:
                return operation(conn, *inner_args, **inner_kwargs)
            finally:
                conn.set_trace_callback(None)

        return await execute_read(_traced, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(app.startup, '_backend', backend)
        patch.setattr(app.startup, '_repositories', RepositoryContainer(backend))
        patch.setattr(app.startup, '_summary_provider', MagicMock())
        patch.setattr('app.tools.discovery.settings', _statistics_settings())
        patch.setattr('app.tools.discovery.get_embedding_provider', MagicMock)
        patch.setattr('app.tools.discovery.DB_PATH', db_path)
        patch.setattr(backend, 'execute_read', _tracing_execute_read)
        with as_principal(scope.principal_id, groups=scope.groups):
            await app.tools.get_statistics()
    return statements


class TestScopedStatementPlans:
    """Scoped statements over context_entries read only a covering access index."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('scope', _SCOPES)
    async def test_get_statistics_statements_use_covering_access_indexes(
        self,
        startup_backend: StorageBackend,
        scope: AccessScope,
        temp_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every statement get_statistics runs reads context_entries only through a covering
        access index, except the summary count, and no virtual table reads it internally."""
        statements = await _get_statistics_statements(startup_backend, scope, temp_db_path, monkeypatch)
        direct = [statement for statement in statements if not statement.startswith(_INTERNAL_STATEMENT)]
        internal = [statement for statement in statements if statement.startswith(_INTERNAL_STATEMENT)]

        for marker in _SCOPED_FIGURE_MARKERS:
            assert any(marker in statement for statement in direct), f'no statement carries {marker!r}: {direct}'
        assert [statement for statement in internal if _NAMES_CONTEXT_ENTRIES.search(statement)] == []
        for statement in direct:
            if _NAMES_CONTEXT_ENTRIES.search(statement) and _SUMMARY_COUNT not in statement:
                _assert_entries_read_from_covering_index(statement, await _plan(startup_backend, statement, []))

    @pytest.mark.asyncio
    @pytest.mark.parametrize('scope', _SCOPES)
    @pytest.mark.parametrize(
        ('client_filter', 'client_params'),
        [
            pytest.param('', [], id='unfiltered'),
            pytest.param('source = ? AND ', ['agent'], id='source'),
            pytest.param('thread_id = ? AND ', ['thread-a'], id='thread'),
        ],
    )
    async def test_candidate_statement_uses_covering_access_index(
        self,
        startup_backend: StorageBackend,
        scope: AccessScope,
        client_filter: str,
        client_params: list[Any],
    ) -> None:
        """The semantic-search candidate statement -- the client filters, then the READ
        predicate -- reads context_entries from a covering access index."""
        read = build_access_predicate(scope, mode=AccessMode.READ, backend_type='sqlite', outer='context_entries')
        sql = f'SELECT id FROM context_entries WHERE {client_filter}{read.sql}'

        plan = await _plan(startup_backend, sql, [*client_params, *read.params])

        _assert_entries_read_from_covering_index(sql, plan)
