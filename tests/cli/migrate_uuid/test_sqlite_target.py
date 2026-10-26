"""Tests for the SQLite target initialization: the embedding dimension, the FTS rebuild and the base schema objects."""

import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.sqlite_to_sqlite import run_migration_sqlite_to_sqlite
from tests.cli.migrate_uuid._sources import HEX_32_RE
from tests.cli.migrate_uuid._sources import build_sqlite_options
from tests.cli.migrate_uuid._sources import seed_source_db


class TestInitializeTargetSqliteDimFallback:
    """initialize_target_sqlite uses get_settings().embedding.dim when embedding_dim is None.

    When the source ``embedding_metadata`` table exists but is empty,
    ``detect_source_embedding_dim`` returns ``None``. A hardcoded constant such as
    1024 would silently ignore the operator's ``EMBEDDING_DIM`` and provision a
    target vector column at the wrong width, so, like the PostgreSQL counterpart,
    the function keeps a detected non-None dim unchanged and otherwise resolves
    ``get_settings().embedding.dim``.
    """

    @staticmethod
    def _install_fakes(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[list[str], mock.MagicMock]:
        """Patch the file reader and vec-load so the semantic branch is entered without I/O.

        Returns:
            A tuple of (executed_sqls, conn) where executed_sqls collects every SQL
            string passed to executescript and conn is the mock connection.
        """
        import app.cli.migrate_uuid.sqlite_target as sqlite_target_mod

        executed_sqls: list[str] = []
        original_read = sqlite_target_mod.read_schema_file

        def _spy_read(filename: str) -> str:
            sql = original_read(filename)
            # Keep {EMBEDDING_DIM} as an inner sentinel so the real replacement logic
            # runs and we can inspect which numeric value was substituted.
            if filename == 'add_semantic_search_sqlite.sql':
                return sql.replace('{EMBEDDING_DIM}', '__DIM:{EMBEDDING_DIM}__')
            return sql

        monkeypatch.setattr(sqlite_target_mod, 'read_schema_file', _spy_read)
        monkeypatch.setattr(sqlite_target_mod, 'load_sqlite_vec_extension', lambda _conn: True)

        conn: mock.MagicMock = mock.MagicMock(spec=sqlite3.Connection)
        conn.executescript.side_effect = executed_sqls.append
        conn.execute.return_value = mock.MagicMock()
        return executed_sqls, conn

    def test_settings_fallback_used_when_embedding_dim_is_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """embedding_dim=None causes the settings EMBEDDING_DIM to be templated into the schema.

        When the source embedding_metadata table exists but is empty the detected dim is
        None.  The function must resolve get_settings().embedding.dim instead of falling
        back to the hardcoded constant 1024, so the target vector column width matches the
        operator's configured dimension.
        """
        from app.cli.migrate_uuid.sqlite_target import initialize_target_sqlite
        from app.settings import get_settings

        configured_dim = 2048
        monkeypatch.setenv('EMBEDDING_DIM', str(configured_dim))
        monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
        get_settings.cache_clear()
        try:
            executed_sqls, conn = self._install_fakes(monkeypatch)
            stats = MigrationStats()

            initialize_target_sqlite(
                conn,
                optional_tables={'embedding_metadata': True},
                embedding_dim=None,
                fts_tokenizer='porter unicode61',
                stats=stats,
            )
        finally:
            monkeypatch.delenv('EMBEDDING_DIM', raising=False)
            monkeypatch.delenv('ENABLE_EMBEDDING_GENERATION', raising=False)
            get_settings.cache_clear()

        sentinel = f'__DIM:{configured_dim}__'
        assert any(sentinel in sql for sql in executed_sqls), (
            f'Expected the semantic SQL to contain the configured dim {configured_dim} '
            f'(via settings fallback), but got: {executed_sqls}'
        )

    def test_detected_source_dim_used_unchanged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-None embedding_dim is passed through unchanged without consulting settings.

        When the source embedding_metadata table is non-empty and detect_source_embedding_dim
        returns a concrete dimension, initialize_target_sqlite must template THAT value into
        the vector column width -- not the settings EMBEDDING_DIM -- so the target column
        matches the actual source data.
        """
        from app.cli.migrate_uuid.sqlite_target import initialize_target_sqlite
        from app.settings import get_settings

        source_dim = 512
        settings_dim = 1024  # deliberately different to prove settings is not consulted
        monkeypatch.setenv('EMBEDDING_DIM', str(settings_dim))
        monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
        get_settings.cache_clear()
        try:
            executed_sqls, conn = self._install_fakes(monkeypatch)
            stats = MigrationStats()

            initialize_target_sqlite(
                conn,
                optional_tables={'embedding_metadata': True},
                embedding_dim=source_dim,
                fts_tokenizer='porter unicode61',
                stats=stats,
            )
        finally:
            monkeypatch.delenv('EMBEDDING_DIM', raising=False)
            monkeypatch.delenv('ENABLE_EMBEDDING_GENERATION', raising=False)
            get_settings.cache_clear()

        sentinel_source = f'__DIM:{source_dim}__'
        sentinel_settings = f'__DIM:{settings_dim}__'
        semantic_sqls = [s for s in executed_sqls if '__DIM:' in s]
        assert semantic_sqls, f'Semantic SQL was not intercepted; got: {executed_sqls}'
        assert any(sentinel_source in s for s in semantic_sqls), (
            f'Expected source dim {source_dim} to be templated; got: {semantic_sqls}'
        )
        assert not any(sentinel_settings in s for s in semantic_sqls), (
            f'Settings dim {settings_dim} must not be consulted when source dim is non-None; '
            f'got: {semantic_sqls}'
        )


@pytest.fixture
def source_with_fts(tmp_path: Path) -> Path:
    """Create a source DB that includes the FTS5 virtual table."""
    path = tmp_path / 'source-fts.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 't',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'the quick brown fox jumps over the lazy dog',
            'metadata': None,
            'created_at': '2025-08-10 10:00:00',
        },
        {
            'id': 2,
            'thread_id': 't',
            'source': 'agent',
            'content_type': 'text',
            'text_content': 'sphinx of black quartz judge my vow',
            'metadata': None,
            'created_at': '2025-08-10 10:05:00',
        },
    ]
    seed_source_db(path, rows)
    conn = sqlite3.connect(str(path))
    try:
        # Build an FTS5 table that mirrors the legacy external-content shape so
        # that detect_optional_tables returns True for context_entries_fts.
        conn.execute(
            "CREATE VIRTUAL TABLE context_entries_fts USING fts5("
            "text_content, content='context_entries', content_rowid='id', "
            "tokenize='porter unicode61')",
        )
        conn.execute("INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()
    return path


class TestFtsRebuild:
    """FTS5 index is rebuilt against the target's rowid_int surrogate."""

    def test_fts_rebuilt_after_migration_sqlite(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
    ) -> None:
        """An FTS5 MATCH against the target returns the migrated rows."""
        options = build_sqlite_options(source_with_fts, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.fts_rebuilt
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            conn.row_factory = sqlite3.Row
            rows = list(conn.execute(
                "SELECT rowid FROM context_entries_fts WHERE text_content MATCH 'fox'",
            ))
            assert len(rows) >= 1
        finally:
            conn.close()

    def test_fts_join_uses_rowid_int_surrogate(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
    ) -> None:
        """``context_entries_fts.rowid`` lines up with ``context_entries.rowid_int``."""
        options = build_sqlite_options(source_with_fts, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            conn.row_factory = sqlite3.Row
            rows = list(conn.execute(
                'SELECT ce.rowid_int AS rowid_int, ce.id AS public_id, fts.rowid AS fts_rowid '
                'FROM context_entries ce '
                'JOIN context_entries_fts fts ON fts.rowid = ce.rowid_int',
            ))
            assert len(rows) >= 1
            for row in rows:
                assert row['rowid_int'] == row['fts_rowid']
                assert HEX_32_RE.match(row['public_id'])
        finally:
            conn.close()

    @staticmethod
    def _read_target_fts_ddl(target_path: Path) -> str:
        """Return the stored CREATE statement for the target's FTS5 table."""
        conn = sqlite3.connect(str(target_path))
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'context_entries_fts'",
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0]
        return str(row[0])

    def test_fts_tokenizer_defaults_to_porter_for_english(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
    ) -> None:
        """With the default English language, the target FTS5 table uses the Porter stemmer."""
        options = build_sqlite_options(source_with_fts, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.fts_rebuilt
        ddl = self._read_target_fts_ddl(new_target_db_path)
        assert "tokenize='porter unicode61'" in ddl

    def test_fts_tokenizer_drops_porter_for_non_english_language(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-English FTS_LANGUAGE migrates to plain unicode61, not the English Porter stemmer."""
        from app.settings import get_settings

        monkeypatch.setenv('FTS_LANGUAGE', 'russian')
        get_settings.cache_clear()
        try:
            options = build_sqlite_options(source_with_fts, new_target_db_path)
            stats = run_migration_sqlite_to_sqlite(options)
            assert stats.fts_rebuilt
            ddl = self._read_target_fts_ddl(new_target_db_path)
            assert "tokenize='unicode61'" in ddl
            assert "tokenize='porter unicode61'" not in ddl
        finally:
            # Restore the cached singleton so the override does not leak into later tests.
            get_settings.cache_clear()


class TestBaseSchemaObjects:
    """The target holds every object the base schema declares before any server starts on it."""

    def test_the_target_holds_an_empty_generation_jobs_table_and_its_queue_index(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
    ) -> None:
        stats = run_migration_sqlite_to_sqlite(build_sqlite_options(source_with_fts, new_target_db_path))

        conn = sqlite3.connect(str(new_target_db_path))
        try:
            objects = conn.execute(
                "SELECT type, name FROM sqlite_master WHERE tbl_name = 'generation_jobs' "
                "AND name NOT LIKE 'sqlite_autoindex_%' ORDER BY type, name",
            ).fetchall()
            jobs = conn.execute('SELECT count(*) FROM generation_jobs').fetchone()[0]
        finally:
            conn.close()

        assert stats.rows_migrated == 2
        assert objects == [('index', 'idx_generation_jobs_queue'), ('table', 'generation_jobs')]
        assert jobs == 0
