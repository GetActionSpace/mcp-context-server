"""Tests for the SQLite copy helpers: free-form text, embeddings, chunk internals and vec0 rowids survive the migration."""

import sqlite3
from pathlib import Path

import pytest

from app.cli.migrate_uuid.sqlite_to_sqlite import run_migration_sqlite_to_sqlite
from tests.cli.migrate_uuid._sources import build_sqlite_options
from tests.cli.migrate_uuid._sources import seed_source_db


def _sqlite_vec_available() -> bool:
    """Return True iff the sqlite-vec extension can be loaded."""
    try:
        import sqlite_vec
    except ImportError:
        return False
    conn = sqlite3.connect(':memory:')
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (AttributeError, sqlite3.OperationalError, sqlite3.NotSupportedError):
        return False
    finally:
        conn.close()


requires_sqlite_vec = pytest.mark.skipif(
    not _sqlite_vec_available(),
    reason='sqlite-vec extension not loadable on this platform',
)


@pytest.fixture
def source_with_embeddings(tmp_path: Path) -> Path:
    """Create a source DB that includes embedding metadata, chunks, and vec0 rows."""
    if not _sqlite_vec_available():
        pytest.skip('sqlite-vec required for this fixture')
    import sqlite_vec

    path = tmp_path / 'source-with-vec.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 'thread-x',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'embedding row one',
            'metadata': None,
            'created_at': '2025-05-01 09:00:00',
        },
        {
            'id': 2,
            'thread_id': 'thread-x',
            'source': 'agent',
            'content_type': 'text',
            'text_content': 'embedding row two',
            'metadata': None,
            'created_at': '2025-05-01 09:01:00',
        },
    ]
    seed_source_db(path, rows)

    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        conn.execute('CREATE VIRTUAL TABLE vec_context_embeddings USING vec0(embedding float[4])')
        conn.execute(
            'CREATE TABLE embedding_metadata ('
            'context_id INTEGER NOT NULL PRIMARY KEY, '
            'model_name TEXT NOT NULL, '
            'dimensions INTEGER NOT NULL, '
            'chunk_count INTEGER NOT NULL DEFAULT 1, '
            'created_at TEXT NOT NULL, '
            'updated_at TEXT NOT NULL)',
        )
        conn.execute(
            'CREATE TABLE embedding_chunks ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'context_id INTEGER NOT NULL, '
            'vec_rowid INTEGER NOT NULL, '
            'start_index INTEGER NOT NULL DEFAULT 0, '
            'end_index INTEGER NOT NULL DEFAULT 0, '
            'created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)',
        )
        emb_a = bytes([1, 2, 3, 4] * 4)  # 16 bytes = 4 float32 values
        emb_b = bytes([5, 6, 7, 8] * 4)
        conn.execute('INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (1, ?)', (emb_a,))
        conn.execute('INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (2, ?)', (emb_b,))
        conn.execute(
            'INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)',
            (1, 'model-a', 4, 1, '2025-05-01 09:00:00', '2025-05-01 09:00:00'),
        )
        conn.execute(
            'INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)',
            (2, 'model-a', 4, 1, '2025-05-01 09:01:00', '2025-05-01 09:01:00'),
        )
        conn.execute(
            'INSERT INTO embedding_chunks (id, context_id, vec_rowid, start_index, end_index) '
            'VALUES (?, ?, ?, ?, ?)',
            (1, 1, 1, 0, 17),
        )
        conn.execute(
            'INSERT INTO embedding_chunks (id, context_id, vec_rowid, start_index, end_index) '
            'VALUES (?, ?, ?, ?, ?)',
            (2, 2, 2, 0, 17),
        )
        conn.commit()
    finally:
        conn.close()
    return path


class TestEmbeddingPreservation:
    """Embeddings and chunk internals are preserved verbatim."""

    @requires_sqlite_vec
    def test_embeddings_copied_not_regenerated(
        self,
        source_with_embeddings: Path,
        new_target_db_path: Path,
    ) -> None:
        """vec_context_embeddings blob content matches between source and target."""
        import sqlite_vec

        options = build_sqlite_options(source_with_embeddings, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert not stats.errors, f'unexpected errors: {stats.errors}'

        src = sqlite3.connect(str(source_with_embeddings))
        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            for conn in (src, tgt):
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.row_factory = sqlite3.Row
            src_rows = list(src.execute(
                'SELECT rowid, embedding FROM vec_context_embeddings ORDER BY rowid',
            ))
            tgt_rows = list(tgt.execute(
                'SELECT rowid, embedding FROM vec_context_embeddings ORDER BY rowid',
            ))
            assert len(src_rows) == len(tgt_rows)
            for src_row, tgt_row in zip(src_rows, tgt_rows, strict=True):
                assert src_row['rowid'] == tgt_row['rowid']
                assert bytes(src_row['embedding']) == bytes(tgt_row['embedding'])
        finally:
            src.close()
            tgt.close()

    @requires_sqlite_vec
    def test_embedding_chunks_internals_preserved(
        self,
        source_with_embeddings: Path,
        new_target_db_path: Path,
    ) -> None:
        """``embedding_chunks.id`` and ``vec_rowid`` match between source and target."""
        options = build_sqlite_options(source_with_embeddings, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)

        src = sqlite3.connect(str(source_with_embeddings))
        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            src.row_factory = sqlite3.Row
            tgt.row_factory = sqlite3.Row
            src_rows = list(src.execute(
                'SELECT id, vec_rowid FROM embedding_chunks ORDER BY id',
            ))
            tgt_rows = list(tgt.execute(
                'SELECT id, vec_rowid FROM embedding_chunks ORDER BY id',
            ))
            assert len(src_rows) == len(tgt_rows)
            for src_row, tgt_row in zip(src_rows, tgt_rows, strict=True):
                assert src_row['id'] == tgt_row['id']
                assert src_row['vec_rowid'] == tgt_row['vec_rowid']
        finally:
            src.close()
            tgt.close()


class TestVec0Preservation:
    """The vec0 rowid identifier is preserved verbatim between source and target."""

    @requires_sqlite_vec
    def test_vec0_rowid_unchanged(
        self,
        source_with_embeddings: Path,
        new_target_db_path: Path,
    ) -> None:
        """Target vec_context_embeddings.rowid matches the source values."""
        import sqlite_vec

        options = build_sqlite_options(source_with_embeddings, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)
        src = sqlite3.connect(str(source_with_embeddings))
        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            for conn in (src, tgt):
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
            src_ids = [row[0] for row in src.execute('SELECT rowid FROM vec_context_embeddings')]
            tgt_ids = [row[0] for row in tgt.execute('SELECT rowid FROM vec_context_embeddings')]
            assert sorted(src_ids) == sorted(tgt_ids)
        finally:
            src.close()
            tgt.close()


@pytest.fixture
def source_pre_boundary_columns(tmp_path: Path) -> Path:
    """Create a source DB whose embedding_chunks table predates the start_index/end_index columns."""
    if not _sqlite_vec_available():
        pytest.skip('sqlite-vec required for this fixture')
    import sqlite_vec

    path = tmp_path / 'source-pre-boundary.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 'thread-x',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'legacy embedding row',
            'metadata': None,
            'created_at': '2025-05-01 09:00:00',
        },
    ]
    seed_source_db(path, rows)

    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        conn.execute('CREATE VIRTUAL TABLE vec_context_embeddings USING vec0(embedding float[4])')
        conn.execute(
            'CREATE TABLE embedding_metadata ('
            'context_id INTEGER NOT NULL PRIMARY KEY, '
            'model_name TEXT NOT NULL, '
            'dimensions INTEGER NOT NULL, '
            'chunk_count INTEGER NOT NULL DEFAULT 1, '
            'created_at TEXT NOT NULL, '
            'updated_at TEXT NOT NULL)',
        )
        # A chunk table without the boundary columns: NO start_index / end_index.
        conn.execute(
            'CREATE TABLE embedding_chunks ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'context_id INTEGER NOT NULL, '
            'vec_rowid INTEGER NOT NULL, '
            'created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)',
        )
        conn.execute('INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (1, ?)', (bytes([1, 2, 3, 4] * 4),))
        conn.execute(
            'INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)',
            (1, 'model-a', 4, 1, '2025-05-01 09:00:00', '2025-05-01 09:00:00'),
        )
        conn.execute(
            'INSERT INTO embedding_chunks (id, context_id, vec_rowid, created_at) VALUES (?, ?, ?, ?)',
            (1, 1, 1, '2025-05-01 09:00:00'),
        )
        conn.commit()
    finally:
        conn.close()
    return path


class TestPreBoundaryColumnSource:
    """A source embedding_chunks table lacking start_index/end_index is tolerated."""

    @requires_sqlite_vec
    def test_migration_succeeds_and_defaults_boundaries_to_zero(
        self,
        source_pre_boundary_columns: Path,
        new_target_db_path: Path,
    ) -> None:
        """The unguarded boundary-column SELECT would raise OperationalError; the guarded
        copier instead migrates the row and defaults start_index/end_index to 0 on a target
        that has the columns.
        """
        options = build_sqlite_options(source_pre_boundary_columns, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)

        assert not stats.errors, f'unexpected errors: {stats.errors}'
        assert stats.embedding_chunks_migrated == 1

        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            tgt.row_factory = sqlite3.Row
            row = tgt.execute(
                'SELECT id, vec_rowid, start_index, end_index FROM embedding_chunks',
            ).fetchone()
            assert row['id'] == 1
            assert row['vec_rowid'] == 1
            assert row['start_index'] == 0
            assert row['end_index'] == 0
        finally:
            tgt.close()


class TestFreeFormColumnsUntouched:
    """The rewrite reaches metadata only: numbers in free-form text columns are copied as written."""

    def test_text_content_not_rewritten(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """text_content containing integer-style mentions is copied verbatim."""
        options = build_sqlite_options(legacy_source_db, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated == 3

        src_conn = sqlite3.connect(str(legacy_source_db))
        tgt_conn = sqlite3.connect(str(new_target_db_path))
        try:
            src_rows = [row[0] for row in src_conn.execute(
                'SELECT text_content FROM context_entries ORDER BY created_at ASC, id ASC',
            )]
            tgt_rows = [row[0] for row in tgt_conn.execute(
                'SELECT text_content FROM context_entries ORDER BY created_at ASC',
            )]
        finally:
            src_conn.close()
            tgt_conn.close()
        assert tgt_rows == src_rows
        assert any('8944' in (text or '') for text in tgt_rows)

    def test_summary_not_rewritten(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """summary column is preserved byte-for-byte across migration."""
        options = build_sqlite_options(legacy_source_db, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)
        tgt_conn = sqlite3.connect(str(new_target_db_path))
        try:
            row = tgt_conn.execute(
                "SELECT summary FROM context_entries WHERE summary LIKE '%8944%'",
            ).fetchone()
        finally:
            tgt_conn.close()
        assert row is not None
        assert row[0] == 'A reply that mentions ID 8944 inline.'
