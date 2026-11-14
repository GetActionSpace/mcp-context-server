"""Regression tests for the SQLite->PostgreSQL unstorable-value pre-check.

A NUL (U+0000) or unpaired UTF-16 surrogate is legal in a SQLite TEXT value but
fatal on PostgreSQL: PostgreSQL refuses a NUL in a text value, and its jsonb parser
rejects the ``\\u0000`` escape the store path emits for a metadata NUL (SQLSTATE
22P05). A jsonb column has a third
divergence class: a non-finite JSON number. ``json.loads`` accepts the non-standard
tokens ``NaN``/``Infinity``/``-Infinity`` and silently turns a standard but overflowing
literal such as ``1e400`` into ``inf``. The metadata rewriter cannot re-encode such a
value, so it keeps the source text verbatim, and attachment metadata is bound as the
source holds it. A verbatim ``NaN`` or ``Infinity`` token fails the ``::jsonb`` cast and,
unscreened, aborts the whole cross-backend migration mid-transaction with only a raw
driver error and no row identification. A verbatim ``1e400`` passes the cast, since jsonb
stores it as a JSON number, but the server refuses a non-finite number at its store
boundary, and the pre-check skips it too. These tests prove each such row is identified
and skipped with an error and the run does not crash; the ``--dry-run`` tests show the
same rows reported and nothing written.

The full-run tests drive the real ``run_migration_mixed_sqlite_to_postgresql``
against a fake asyncpg target connection (the PostgreSQL probe helpers are patched
out), so no live PostgreSQL is required; the per-row copy loops -- the code under
test -- run unchanged.
"""

import json
import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.sqlite_to_postgresql import run_migration_mixed_sqlite_to_postgresql
from tests.cli._owner_connection import AnswersAsOwner
from tests.cli.migrate_uuid._sources import INTEGER_KEYED_SCHEMA_SQL


def _seed_source(
    path: Path,
    entries: list[dict[str, object]],
    tags: list[tuple[int, str]] | None = None,
    images: list[tuple[int, str, str | None]] | None = None,
) -> None:
    """Create an integer-keyed source DB at ``path`` and seed it.

    Each entry dict needs ``id``, ``thread_id``, ``source``, ``content_type``,
    ``text_content``, ``metadata`` (a dict serialized to JSON, or None), and
    ``created_at``. Tags are ``(context_entry_id, tag)`` pairs; images are
    ``(context_entry_id, mime_type, image_metadata_json)`` triples.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(INTEGER_KEYED_SCHEMA_SQL)
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
        for context_entry_id, mime_type, image_metadata in images or []:
            conn.execute(
                'INSERT INTO image_attachments '
                '(context_entry_id, image_data, mime_type, image_metadata, position) '
                'VALUES (?, ?, ?, ?, ?)',
                (context_entry_id, b'\x89PNG\r\n', mime_type, image_metadata, 0),
            )
        conn.commit()
    finally:
        conn.close()


class _FakeTargetConn(AnswersAsOwner):
    """Minimal async stand-in for the asyncpg target connection.

    Records every ``execute`` call so a test can assert which rows were inserted
    (or, under ``--dry-run``, that none were). The PostgreSQL probe helpers are
    patched out, so this only needs the ``execute`` / ``close`` surface the copy
    loops touch.
    """

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

    def statements(self) -> list[str]:
        """Return every executed SQL statement in call order."""
        return [query for query, _ in self.executed]


async def _run_with_fake_target(
    source: Path,
    *,
    dry_run: bool,
) -> tuple[MigrationStats, _FakeTargetConn]:
    """Run the SQLite->PostgreSQL migration against a fake target connection.

    Patches ``asyncpg.connect`` and the PostgreSQL probe helpers so the real
    per-row copy loops run without a live PostgreSQL server.

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
        # Report the target as already initialized so auto-init is skipped and the
        # FTS backstop (also patched out) is taken instead.
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


def _mixed_source_with_nul_rows(path: Path) -> None:
    """Seed a source with a NUL in text_content, metadata, a tag and an image_metadata, plus children of a skipped parent."""
    _seed_source(
        path,
        entries=[
            {
                'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'clean entry', 'metadata': {'a': 'b'},
                'created_at': '2025-01-01 12:00:00',
            },
            {
                'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                'text_content': 'poisoned\x00text', 'metadata': None,
                'created_at': '2025-01-02 12:00:00',
            },
            {
                'id': 3, 'thread_id': 't3', 'source': 'user', 'content_type': 'text',
                'text_content': 'ok text', 'metadata': {'note': 'has\x00nul'},
                'created_at': '2025-01-03 12:00:00',
            },
        ],
        tags=[
            (1, 'good-tag'),
            (1, 'bad\x00tag'),
            (2, 'child-of-skipped-parent'),
        ],
        images=[
            (1, 'image/png', json.dumps({'caption': 'fine'})),
            (1, 'image/png', json.dumps({'caption': 'has\x00nul'})),
            (2, 'image/png', json.dumps({'caption': 'fine'})),
        ],
    )


class TestSqliteToPostgresqlNulPrecheck:
    """The real cross-backend copy loops skip unstorable rows instead of aborting."""

    @pytest.mark.asyncio
    async def test_nul_rows_skipped_and_run_completes(self, tmp_path: Path) -> None:
        """Every unstorable row is identified and skipped; the clean data still migrates
        and the transaction commits rather than rolling back."""
        source = tmp_path / 'nul_source.db'
        _mixed_source_with_nul_rows(source)

        stats, conn = await _run_with_fake_target(source, dry_run=False)

        # Only the single clean context row is copied.
        assert stats.rows_migrated == 1
        assert len(conn.inserts('context_entries')) == 1
        assert stats.tags_migrated == 1
        assert len(conn.inserts('tags')) == 1
        assert stats.images_migrated == 1
        assert len(conn.inserts('image_attachments')) == 1

        # Each unstorable row is reported with source id and offending column.
        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "column 'text_content'" in errors
        assert 'context_entries row id=3' in errors
        assert "column 'metadata'" in errors
        assert 'tags row context_entry_id=1' in errors
        assert "column 'tag'" in errors
        assert 'image_attachments row context_entry_id=1' in errors
        assert "column 'image_metadata'" in errors

        # Children of the skipped parent (id=2) are skipped, not orphaned onto a
        # never-inserted parent (which would raise an FK violation on PostgreSQL).
        warnings = '\n'.join(stats.warnings)
        assert warnings.count('parent context_entries row was skipped') == 2

        # The run committed; it neither crashed nor rolled back.
        assert 'COMMIT' in conn.statements()
        assert 'ROLLBACK' not in conn.statements()

    @pytest.mark.asyncio
    async def test_dry_run_surfaces_nul_rows_without_inserting(self, tmp_path: Path) -> None:
        """--dry-run reports every unstorable row and writes nothing to the target."""
        source = tmp_path / 'nul_source_dry.db'
        _mixed_source_with_nul_rows(source)

        stats, conn = await _run_with_fake_target(source, dry_run=True)

        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert 'context_entries row id=3' in errors
        assert "tags row context_entry_id=1 column 'tag'" in errors
        assert "image_attachments row context_entry_id=1 column 'image_metadata'" in errors

        # No write of any kind reaches the target under a dry run.
        assert all(not statement.startswith('INSERT') for statement in conn.statements())
        assert 'BEGIN' not in conn.statements()
        assert 'COMMIT' not in conn.statements()
        # The clean row is still counted (matching existing dry-run counting semantics).
        assert stats.rows_migrated == 1

    @pytest.mark.asyncio
    async def test_malformed_metadata_json_skipped_not_bound_to_jsonb(self, tmp_path: Path) -> None:
        """A row whose metadata is unparseable JSON is identified and skipped, never bound into
        the target's ``metadata::jsonb`` cast (which would reject it mid-transaction and abort
        the whole run). The clean row still migrates and the run commits.
        """
        source = tmp_path / 'malformed_metadata_source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': {'a': 'b'},
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'ok text', 'metadata': '{not valid json',
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, conn = await _run_with_fake_target(source, dry_run=False)

        # Only the clean row is copied; the malformed-metadata row is skipped.
        assert stats.rows_migrated == 1
        context_inserts = conn.inserts('context_entries')
        assert len(context_inserts) == 1

        # The malformed row is reported with its source id and the offending column.
        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "column 'metadata'" in errors

        # The verbatim malformed string never reaches an INSERT bind: had it been bound to the
        # $n::jsonb cast, PostgreSQL would have aborted the transaction mid-run.
        assert all('{not valid json' not in args for args in context_inserts)

        # The run committed rather than rolling back on a mid-transaction jsonb parse error.
        assert 'COMMIT' in conn.statements()
        assert 'ROLLBACK' not in conn.statements()

    @pytest.mark.asyncio
    async def test_content_hash_nul_skipped_not_aborting(self, tmp_path: Path) -> None:
        """A row whose content_hash carries a NUL is identified and skipped, not aborted.

        content_hash is a plain TEXT column bound raw at the INSERT (no CHECK
        constraint on either backend). Left out of the unstorable-string pre-check
        candidates, a NUL there would pass the pre-check, and PostgreSQL, which
        refuses a NUL in a text value, would reject the INSERT mid-transaction,
        rolling back the whole cross-backend migration with only the raw driver
        error and no row identification. The pre-check inspects content_hash, so
        the offending row is skipped and reported while the clean data migrates
        and the run commits.
        """
        source = tmp_path / 'content_hash_nul_source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': {'a': 'b'},
                    'content_hash': 'cleanhash', 'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'ok text', 'metadata': None,
                    'content_hash': 'poisoned\x00hash', 'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, conn = await _run_with_fake_target(source, dry_run=False)

        # Only the clean row is copied; the content_hash-NUL row is skipped.
        assert stats.rows_migrated == 1
        assert len(conn.inserts('context_entries')) == 1

        # The offending row is reported with its source id and the content_hash column.
        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "column 'content_hash'" in errors

        # The poisoned content_hash never reaches an INSERT bind: had it been bound raw,
        # asyncpg would have aborted the transaction mid-run.
        assert all('poisoned\x00hash' not in args for args in conn.inserts('context_entries'))

        # The run committed rather than rolling back on the mid-transaction NUL bind.
        assert 'COMMIT' in conn.statements()
        assert 'ROLLBACK' not in conn.statements()

    @pytest.mark.asyncio
    async def test_dry_run_surfaces_content_hash_nul_row(self, tmp_path: Path) -> None:
        """--dry-run identifies the content_hash-NUL row without inserting."""
        source = tmp_path / 'content_hash_nul_source_dry.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': {'a': 'b'},
                    'content_hash': 'cleanhash', 'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'ok text', 'metadata': None,
                    'content_hash': 'poisoned\x00hash', 'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, conn = await _run_with_fake_target(source, dry_run=True)

        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "column 'content_hash'" in errors

        # No write of any kind reaches the target under a dry run.
        assert all(not statement.startswith('INSERT') for statement in conn.statements())
        assert 'BEGIN' not in conn.statements()
        assert 'COMMIT' not in conn.statements()


def _source_with_non_finite_metadata(path: Path) -> None:
    """Seed a source whose metadata carries each non-finite spelling.

    The values are passed as raw JSON strings so SQLite stores the exact literal a
    legacy database would hold (``json.dumps`` would already have normalized them).
    """
    _seed_source(
        path,
        entries=[
            {
                'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'clean entry', 'metadata': '{"score": 1.5}',
                'created_at': '2025-01-01 12:00:00',
            },
            {
                'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                'text_content': 'overflowing literal', 'metadata': '{"score": 1e400}',
                'created_at': '2025-01-02 12:00:00',
            },
            {
                'id': 3, 'thread_id': 't3', 'source': 'user', 'content_type': 'text',
                'text_content': 'infinity token', 'metadata': '{"score": Infinity}',
                'created_at': '2025-01-03 12:00:00',
            },
            {
                'id': 4, 'thread_id': 't4', 'source': 'agent', 'content_type': 'text',
                'text_content': 'nan token', 'metadata': '{"a": {"b": NaN}}',
                'created_at': '2025-01-04 12:00:00',
            },
        ],
        images=[
            (1, 'image/png', '{"iso": 100}'),
            (1, 'image/png', '{"iso": NaN}'),
        ],
    )


class TestSqliteToPostgresqlNonFinitePrecheck:
    """Rows whose metadata decodes to a non-finite float are skipped, not fatal."""

    @pytest.mark.asyncio
    async def test_non_finite_rows_skipped_and_run_completes(self, tmp_path: Path) -> None:
        """Each non-finite spelling is identified and skipped; the clean data still migrates.

        Without the guard the metadata text the rewriter keeps verbatim reaches the
        ``$6::jsonb`` bind: its ``Infinity`` or ``NaN`` token fails the cast, the handler
        issues ROLLBACK, and every other row in the run is discarded with a raw driver error
        naming no source row. The ``1e400`` text passes the cast, so in a run without those
        tokens its row would be stored with a number the server refuses.
        """
        source = tmp_path / 'non_finite_source.db'
        _source_with_non_finite_metadata(source)

        stats, conn = await _run_with_fake_target(source, dry_run=False)

        assert stats.rows_migrated == 1
        context_inserts = conn.inserts('context_entries')
        assert len(context_inserts) == 1

        errors = '\n'.join(stats.errors)
        for row_id in (2, 3, 4):
            assert f'context_entries row id={row_id}' in errors
        assert "column 'metadata'" in errors
        assert 'image_attachments row context_entry_id=1' in errors
        assert "column 'image_metadata'" in errors

        # Only the clean image survives.
        assert stats.images_migrated == 1

        # No invalid JSON token ever reaches an INSERT bind.
        bound = [str(arg) for args in context_inserts for arg in args]
        assert all('Infinity' not in value and 'NaN' not in value for value in bound)

        # The run committed rather than rolling back on a mid-transaction jsonb parse error.
        assert 'COMMIT' in conn.statements()
        assert 'ROLLBACK' not in conn.statements()

    @pytest.mark.asyncio
    async def test_dry_run_surfaces_non_finite_rows(self, tmp_path: Path) -> None:
        """--dry-run names every offending row so the operator can repair the source first."""
        source = tmp_path / 'non_finite_source_dry.db'
        _source_with_non_finite_metadata(source)

        stats, conn = await _run_with_fake_target(source, dry_run=True)

        errors = '\n'.join(stats.errors)
        for row_id in (2, 3, 4):
            assert f'context_entries row id={row_id}' in errors
        assert "image_attachments row context_entry_id=1 column 'image_metadata'" in errors

        assert all(not statement.startswith('INSERT') for statement in conn.statements())
        assert 'BEGIN' not in conn.statements()
        assert 'COMMIT' not in conn.statements()
        assert stats.rows_migrated == 1


def _source_with_oversized_indexed_values(path: Path) -> None:
    """Seed a source carrying a thread_id and a tag over their PostgreSQL index budgets."""
    _seed_source(
        path,
        entries=[
            {
                'id': 1,
                'thread_id': 'clean-thread',
                'source': 'user',
                'content_type': 'text',
                'text_content': 'clean row',
                'metadata': None,
                'created_at': '2026-01-01 00:00:00',
            },
            {
                'id': 2,
                'thread_id': 'T' * 4000,
                'source': 'user',
                'content_type': 'text',
                'text_content': 'row whose thread_id cannot be indexed on PostgreSQL',
                'metadata': None,
                'created_at': '2026-01-01 00:00:01',
            },
        ],
        tags=[(1, 'ok'), (1, 'g' * 4000)],
    )


class TestSqliteToPostgresqlIndexSizePrecheck:
    """A legacy value over its PostgreSQL index budget is skipped, not fatal.

    A corpus predating the write-path length caps can hold a thread_id or tag larger
    than the PostgreSQL target's btree index-tuple budget, so it is screened per row,
    like the NUL/surrogate class.
    """

    @pytest.mark.asyncio
    async def test_oversized_rows_skipped_and_run_completes(self, tmp_path: Path) -> None:
        """The offending rows are named and skipped; the clean data still migrates."""
        source = tmp_path / 'oversized_index_source.db'
        _source_with_oversized_indexed_values(source)

        stats, conn = await _run_with_fake_target(source, dry_run=False)

        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "column 'thread_id'" in errors
        assert 'index-tuple budget' in errors
        assert 'tags row context_entry_id=1' in errors

        # Only the clean entry and the clean tag are copied.
        assert stats.rows_migrated == 1
        assert stats.tags_migrated == 1
        assert len(conn.inserts('context_entries')) == 1

        # The run committed rather than rolling back on a mid-transaction index error.
        assert 'COMMIT' in conn.statements()
        assert 'ROLLBACK' not in conn.statements()

    @pytest.mark.asyncio
    async def test_dry_run_surfaces_oversized_rows(self, tmp_path: Path) -> None:
        """--dry-run names the offending rows before any real run touches the target."""
        source = tmp_path / 'oversized_index_source_dry.db'
        _source_with_oversized_indexed_values(source)

        stats, conn = await _run_with_fake_target(source, dry_run=True)

        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "tags row context_entry_id=1 column 'tag'" in errors
        assert all(not statement.startswith('INSERT') for statement in conn.statements())
        assert stats.rows_migrated == 1
