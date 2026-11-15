"""Tests for the SQLite-to-SQLite migration runner: dry runs, schema detection and precision logging."""

import sqlite3
from pathlib import Path

import pytest

from app.cli.migrate import main as cli_main
from app.cli.migrate_uuid.sqlite_to_sqlite import run_migration_sqlite_to_sqlite
from tests.cli.migrate_uuid._sources import INTEGER_KEYED_SCHEMA_SQL
from tests.cli.migrate_uuid._sources import build_sqlite_options


class TestDryRunNoTargetWrites:
    """A dry run must not create or write the SQLite target on disk."""

    def test_dry_run_does_not_create_target_file(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """--dry-run leaves no file at a non-existent SQLite target path."""
        assert not new_target_db_path.exists()
        stats = run_migration_sqlite_to_sqlite(
            build_sqlite_options(legacy_source_db, new_target_db_path, dry_run=True),
        )
        assert not stats.errors
        assert not new_target_db_path.exists()


class TestDryRun:
    """`--dry-run` performs all logic without writing to the target."""

    def test_dry_run_no_writes_to_target(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """A dry run counts at least one row as migrated and creates no target file."""
        options = build_sqlite_options(legacy_source_db, new_target_db_path, dry_run=True)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated > 0
        assert not new_target_db_path.exists()

    def test_dry_run_returns_zero_exit_code(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """``main`` returns 0 in dry-run mode when no errors occur."""
        exit_code = cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
                '--dry-run',
            ],
        )
        assert exit_code == 0


class TestSchemaDetection:
    """Source schema detection guards against unnecessary work."""

    def test_already_migrated_source_warned_and_exit_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """A TEXT-keyed source DB produces a warning and exits 0 without writes."""
        source = tmp_path / 'already.db'
        conn = sqlite3.connect(str(source))
        try:
            conn.execute(
                'CREATE TABLE context_entries ('
                'rowid_int INTEGER PRIMARY KEY AUTOINCREMENT, '
                'id TEXT NOT NULL UNIQUE, '
                'thread_id TEXT NOT NULL, '
                'source TEXT NOT NULL, '
                'content_type TEXT NOT NULL, '
                'text_content TEXT, '
                'metadata JSON, '
                'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
            )
            conn.commit()
        finally:
            conn.close()
        target = tmp_path / 'target.db'
        options = build_sqlite_options(source, target)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated == 0
        assert any('nothing to migrate' in w for w in stats.warnings)
        assert not stats.errors

    def test_legacy_source_proceeds_normally(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """An integer-keyed source database is fully migrated."""
        options = build_sqlite_options(legacy_source_db, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated == 3
        assert not stats.errors

    def test_target_must_be_empty(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """If the target file contains rows, the CLI refuses with code 1."""
        # Pre-populate the target.
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            conn.executescript(INTEGER_KEYED_SCHEMA_SQL)
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, created_at, updated_at) '
                "VALUES (1, 't', 'user', 'text', 'existing', '2024-01-01', '2024-01-01')",
            )
            conn.commit()
        finally:
            conn.close()
        options = build_sqlite_options(legacy_source_db, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.errors
        assert any('already contains' in e for e in stats.errors)
        assert any('Recovery' in e for e in stats.errors)


class TestSecondPrecisionWarning:
    """The CLI logs an info line when source created_at has zero microsecond precision."""

    def test_second_precision_source_logs_info(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A second-precision source triggers the precision info log."""
        options = build_sqlite_options(legacy_source_db, new_target_db_path)
        with caplog.at_level('INFO', logger='app.cli.migrate_uuid.sqlite_to_sqlite'):
            run_migration_sqlite_to_sqlite(options)
        assert any('precision' in record.message for record in caplog.records)
