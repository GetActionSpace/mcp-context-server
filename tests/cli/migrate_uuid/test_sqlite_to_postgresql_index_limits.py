"""Regression tests for the SQLite->PostgreSQL btree index-tuple pre-check.

A legacy corpus predating the write-path length caps can hold a thread_id, a tag, or
an indexed metadata value whose UTF-8 bytes exceed the pre-check's budget for its
column. The pre-check skips such a row and records an error that names the row and
the column, so the rest of the run completes and the run exits 1.

The full-run tests drive the real ``run_migration_mixed_sqlite_to_postgresql`` against
a fake asyncpg target connection (the PostgreSQL probe helpers are patched out), so no
live PostgreSQL is required; the per-row copy loops -- the code under test -- run
unchanged.
"""

import json
import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.pg_prechecks import PG_MAX_INDEXED_THREAD_ID_BYTES
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.sqlite_to_postgresql import run_migration_mixed_sqlite_to_postgresql
from tests.cli._owner_connection import AnswersAsOwner

# Integer-keyed source schema (the shape the CLI accepts as input) with only the
# context_entries and tags tables these tests seed.
_INTEGER_KEYED_SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    summary TEXT,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);
'''


def _seed_source(
    path: Path,
    entries: list[dict[str, object]],
    tags: list[tuple[int, str]] | None = None,
) -> None:
    """Create an integer-keyed source DB at ``path`` and seed it.

    Each entry dict needs ``id``, ``thread_id``, ``source``, ``content_type``,
    ``text_content``, ``metadata`` (a dict serialized to JSON, or None), and
    ``created_at``. Tags are ``(context_entry_id, tag)`` pairs.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_INTEGER_KEYED_SCHEMA_SQL)
        for entry in entries:
            metadata = entry.get('metadata')
            if isinstance(metadata, (dict, list)):
                metadata = json.dumps(metadata)
            created_at = entry['created_at']
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, metadata, '
                'summary, content_hash, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    entry['id'],
                    entry['thread_id'],
                    entry['source'],
                    entry['content_type'],
                    entry.get('text_content'),
                    metadata,
                    entry.get('summary'),
                    entry.get('content_hash'),
                    created_at,
                    created_at,
                ),
            )
        for context_entry_id, tag in tags or []:
            conn.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (context_entry_id, tag),
            )
        conn.commit()
    finally:
        conn.close()


class _FakeTargetConn(AnswersAsOwner):
    """Minimal async stand-in for the asyncpg target connection."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def execute(self, query: str, *args: object) -> str:
        """Record the SQL and its bound parameters, returning a status string."""
        self.executed.append((query, args))
        return 'OK'

    async def close(self) -> None:
        """Mark the connection closed."""
        self.closed = True

    def inserts(self, table: str) -> list[tuple[object, ...]]:
        """Return the bound-parameter tuples of every ``INSERT INTO <table>`` call."""
        prefix = f'INSERT INTO {table}'
        return [args for query, args in self.executed if query.startswith(prefix)]


async def _run_with_fake_target(
    source: Path,
    *,
    dry_run: bool = False,
) -> tuple[MigrationStats, _FakeTargetConn]:
    """Run the SQLite->PostgreSQL migration against a fake target connection.

    Patches ``asyncpg.connect`` and the PostgreSQL probe helpers so the real per-row
    copy loops run without a live PostgreSQL server.

    Returns:
        The populated migration stats and the fake connection that recorded every
        executed statement.
    """
    fake_conn = _FakeTargetConn()

    async def _fake_connect(*_args: object, **_kwargs: object) -> _FakeTargetConn:
        return fake_conn

    async def _has_data(*_args: object, **_kwargs: object) -> bool:
        return False

    async def _table_exists(*_args: object, **_kwargs: object) -> bool:
        return True

    async def _ensure_fts(*_args: object, **_kwargs: object) -> None:
        return None

    options = MigrationOptions(
        source_url=f'sqlite:///{source.as_posix()}',
        target_url='postgresql://user:pass@localhost:5432/db',
        dry_run=dry_run,
        report_path=None,
    )
    with (
        mock.patch('asyncpg.connect', _fake_connect),
        mock.patch('app.cli.migrate_uuid.sqlite_to_postgresql.target_pg_has_data', _has_data),
        mock.patch('app.cli.migrate_uuid.sqlite_to_postgresql.pg_table_exists', _table_exists),
        mock.patch('app.cli.migrate_uuid.sqlite_to_postgresql.ensure_target_pg_fts', _ensure_fts),
    ):
        stats = await run_migration_mixed_sqlite_to_postgresql(options)
    return stats, fake_conn


class TestSqliteToPostgresqlIndexPrecheck:
    """The real cross-backend copy loops skip a row whose indexed value is over its budget."""

    @pytest.mark.asyncio
    async def test_oversized_indexed_metadata_row_is_skipped(self, tmp_path: Path) -> None:
        """A row whose indexed metadata value is over its budget is skipped; the clean entry and its tag migrate."""
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': {'task_name': 'audit'},
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'legacy entry', 'metadata': {'task_name': 'a' * 3000},
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
            tags=[(1, 'good-tag'), (2, 'child-of-skipped-parent')],
        )

        stats, fake = await _run_with_fake_target(source)

        assert stats.rows_migrated == 1
        assert len(fake.inserts('context_entries')) == 1
        skip_errors = [error for error in stats.errors if 'id=2' in error]
        assert len(skip_errors) == 1
        assert "'metadata.task_name'" in skip_errors[0]
        # The tag of the skipped parent is not copied.
        assert [args[1] for args in fake.inserts('tags')] == ['good-tag']
        assert stats.tags_migrated == 1

    @pytest.mark.asyncio
    async def test_oversized_thread_id_row_is_skipped(self, tmp_path: Path) -> None:
        """A thread_id one byte over PG_MAX_INDEXED_THREAD_ID_BYTES, within the larger budget of a tag, is skipped."""
        oversized = 'a' * (PG_MAX_INDEXED_THREAD_ID_BYTES + 1)
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': oversized, 'source': 'user', 'content_type': 'text',
                    'text_content': 'legacy entry', 'metadata': None,
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': None,
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, fake = await _run_with_fake_target(source)

        assert stats.rows_migrated == 1
        assert len(fake.inserts('context_entries')) == 1
        assert any("column 'thread_id' skipped" in error for error in stats.errors)

    @pytest.mark.asyncio
    async def test_skipped_row_does_not_count_its_reference_rewrites(self, tmp_path: Path) -> None:
        """Values inside a skipped row never reach the target, so none of them is counted."""
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': None,
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'legacy entry',
                    'metadata': {'task_name': 'a' * 3000, 'references': {'context_ids': [1, 1, 1]}},
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, _ = await _run_with_fake_target(source)

        assert stats.rows_migrated == 1
        assert stats.id_paths == {DEFAULT_ID_PATH: IdPathCounts()}

    @pytest.mark.asyncio
    async def test_dry_run_surfaces_the_skip_without_inserting(self, tmp_path: Path) -> None:
        """--dry-run reports the over-budget row and inserts nothing."""
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'legacy entry', 'metadata': {'project': 'a' * 4000},
                    'created_at': '2025-01-01 12:00:00',
                },
            ],
        )

        stats, fake = await _run_with_fake_target(source, dry_run=True)

        assert stats.rows_migrated == 0
        assert fake.inserts('context_entries') == []
        assert any("'metadata.project'" in error for error in stats.errors)
