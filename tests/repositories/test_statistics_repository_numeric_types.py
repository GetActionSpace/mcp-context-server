"""Tests that statistics aggregates are native floats.

Covers the _to_float helper, PostgreSQL Decimal aggregates and database size through a stub backend, and SQLite averages.
"""

import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any
from typing import TypeVar
from typing import cast

import pytest

from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from app.backends.base import base_relation
from app.repositories.statistics_repository import StatisticsRepository
from app.repositories.statistics_repository import _to_float
from tests.helpers import LOCAL_SCOPE

T = TypeVar('T')


class _PgStubConnection:
    """Async stub of an asyncpg connection for the PostgreSQL statistics paths.

    Dispatches on the SQL text so each query in ``get_database_statistics``
    receives a realistic value. The numeric aggregate (``AVG(...)``) returns
    ``decimal.Decimal`` exactly as asyncpg maps PostgreSQL ``NUMERIC`` results;
    the count aggregates and ``pg_database_size`` return native ``int``.
    ``fetchrow``/``fetch`` return mappings, matching the ``row['column']`` access
    the production closures perform.
    """

    def __init__(self, avg_value: Decimal | None) -> None:
        self._avg_value = avg_value

    async def fetchval(self, query: str, *_args: object) -> object:
        normalized = ' '.join(query.split())
        if 'AVG(entry_count)' in normalized:
            return self._avg_value
        if 'COUNT(DISTINCT thread_id)' in normalized:
            return 2
        if 'SELECT DISTINCT t.tag' in normalized:
            return 3
        if 'COUNT(*)' in normalized:
            return 20
        raise AssertionError(f'Unexpected fetchval query: {normalized}')

    async def fetchrow(self, query: str, *_args: object) -> dict[str, object] | None:
        normalized = ' '.join(query.split())
        if 'pg_database_size' in normalized:
            return {'db_size': 8192}
        raise AssertionError(f'Unexpected fetchrow query: {normalized}')

    async def fetch(self, query: str, *_args: object) -> list[dict[str, object]]:
        normalized = ' '.join(query.split())
        if 'GROUP BY source' in normalized:
            return [{'source': 'user', 'count': 12}, {'source': 'agent', 'count': 8}]
        if 'GROUP BY content_type' in normalized:
            return [{'content_type': 'text', 'count': 20}]
        if 'GROUP BY thread_id' in normalized:
            return [{'thread_id': 't1', 'count': 12}, {'thread_id': 't2', 'count': 8}]
        if 'GROUP BY t.tag' in normalized:
            return [{'tag': 'python', 'count': 5}, {'tag': 'sqlite', 'count': 3}]
        if 'GROUP BY gj.status' in normalized:
            return [{'status': 'pending', 'count': 2}, {'status': 'failed', 'count': 1}]
        raise AssertionError(f'Unexpected fetch query: {normalized}')


class _PgStubBackend:
    """Minimal ``StorageBackend`` stub routing through the PostgreSQL code path.

    ``backend_type == 'postgresql'`` selects the PostgreSQL branch of every
    statistics method, and ``execute_read`` awaits the production async closure
    with a :class:`_PgStubConnection`. This exercises the real ``_to_float``
    coercion on a ``decimal.Decimal`` aggregate without a live PostgreSQL
    instance, covering the asyncpg ``Decimal`` aggregate type that a
    SQLite-only test cannot reach (SQLite computes ``AVG`` as a native float).
    """

    backend_type = 'postgresql'

    def __init__(self, avg_value: Decimal | None) -> None:
        self._conn = _PgStubConnection(avg_value)

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name every table by its own name, as a backend without access views does."""
        return base_relation(table, alias=alias)

    async def execute_read(self, operation: Callable[[Any], Awaitable[T]], *, access_scope: BackendAccess) -> T:
        _ = access_scope
        return await operation(self._conn)


class TestToFloatHelper:
    """Unit tests for the ``_to_float`` aggregate-normalization helper."""

    def test_decimal_converts_to_native_float(self) -> None:
        result = _to_float(Decimal('10.00'))
        assert type(result) is float
        assert result == 10.0

    def test_zero_decimal_preserved_as_float(self) -> None:
        result = _to_float(Decimal(0))
        assert type(result) is float
        assert result == 0.0

    def test_none_converts_to_default_zero(self) -> None:
        result = _to_float(None)
        assert type(result) is float
        assert result == 0.0

    def test_float_passes_through_rounded(self) -> None:
        # round(float, 2) uses Python's round-half-to-even with binary float
        # representation; 1.235 is stored as 1.2349999... so it rounds to 1.23.
        result = _to_float(1.235)
        assert type(result) is float
        assert result == round(1.235, 2)

    def test_float_two_decimal_places_preserved(self) -> None:
        result = _to_float(5.5)
        assert type(result) is float
        assert result == 5.5

    def test_int_converts_to_float(self) -> None:
        result = _to_float(7)
        assert type(result) is float
        assert result == 7.0


@pytest.mark.asyncio
class TestPostgresqlDecimalAggregates:
    """PostgreSQL ``AVG()`` aggregates arrive as ``decimal.Decimal`` and leave as native floats.

    asyncpg maps PostgreSQL ``AVG()`` to ``decimal.Decimal``, which serializes to
    a JSON string (e.g. ``"10.00"``) and fails the MCP ``number`` output schema.
    Driving the PostgreSQL branch through a stub backend that returns ``Decimal``
    asserts the repository emits a native ``float`` for ``avg_entries_per_thread``.
    A SQLite-only test cannot reach this path (SQLite computes ``AVG`` as a native
    float).
    """

    async def test_avg_entries_per_thread_is_native_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(Decimal('10.00'))))
        stats = await repo.get_database_statistics(scope=LOCAL_SCOPE)
        avg = stats['avg_entries_per_thread']
        assert not isinstance(avg, Decimal)
        assert not isinstance(avg, str)
        assert type(avg) is float
        assert avg == 10.0

    async def test_null_avg_entries_returns_zero_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(None)))
        stats = await repo.get_database_statistics(scope=LOCAL_SCOPE)
        avg = stats['avg_entries_per_thread']
        assert type(avg) is float
        assert avg == 0.0


@pytest.mark.asyncio
class TestPostgresqlDatabaseSize:
    """Guard that PostgreSQL ``database_size_mb`` is a native float from pg_database_size.

    The PostgreSQL branch queries ``pg_database_size(current_database())`` and
    must convert the ``bigint`` byte count to a native ``float`` MB value. The
    local ``db_path`` argument is irrelevant to a PostgreSQL database and must
    not be file-stat'd.
    """

    async def test_database_size_mb_is_native_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(Decimal('10.00'))))
        stats = await repo.get_database_statistics(scope=LOCAL_SCOPE)
        assert 'database_size_mb' in stats
        size = stats['database_size_mb']
        assert type(size) is float
        # 8192 bytes / (1024 * 1024) rounded to 2 dp.
        assert size == round(8192 / (1024 * 1024), 2)

    async def test_local_db_path_ignored_on_postgresql(self, tmp_path: Path) -> None:
        # A bogus local path must not influence the PostgreSQL size, which comes
        # solely from pg_database_size.
        bogus = tmp_path / 'not_a_pg_database.db'
        bogus.write_bytes(b'x' * (5 * 1024 * 1024))
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(Decimal('10.00'))))
        stats = await repo.get_database_statistics(scope=LOCAL_SCOPE, db_path=bogus)
        assert stats['database_size_mb'] == round(8192 / (1024 * 1024), 2)


@pytest.mark.asyncio
class TestSqliteTruthinessAlignment:
    """Guard that the SQLite avg aggregate is a native float, including zero.

    SQLite computes ``AVG`` in Python as a native float, so the ``_to_float``
    wrap is idempotent. These tests confirm a float ``0.0``, not an int ``0``, on
    an empty database and a native float on a populated database.
    """

    async def test_empty_avg_entries_per_thread_is_zero_float(
        self, stats_repo: StatisticsRepository,
    ) -> None:
        stats = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)
        avg = stats['avg_entries_per_thread']
        assert type(avg) is float
        assert avg == 0.0

    async def test_populated_avg_entries_per_thread_is_native_float(
        self, stats_test_db: StorageBackend, stats_repo: StatisticsRepository,
    ) -> None:
        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd0000aa01', 't1', 'user', 'text', 'A', 'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd0000aa02', 't1', 'agent', 'text', 'B', 'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd0000aa03', 't2', 'user', 'text', 'C', 'local')",
            )

        await stats_test_db.execute_write(_insert_data, access_scope=MAINTENANCE)

        stats = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)
        avg = stats['avg_entries_per_thread']
        assert type(avg) is float
        # 3 entries across 2 threads -> average 1.5.
        assert avg == 1.5
