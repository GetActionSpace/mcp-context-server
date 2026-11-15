"""Tests for the migration entry point: its exit status, the report file and the stdout summary."""

import json
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate import main as cli_main
from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.id_paths import IdPath
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import LegacyIdCounts
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.run import print_summary
from app.cli.migrate_uuid.run import run_uuid_migration
from tests.cli._owner_connection import AnswersAsOwner


class TestReportOutput:
    """JSON report path and stdout summary behavior."""

    def test_report_written_to_path(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        tmp_path: Path,
    ) -> None:
        """``--report`` writes a JSON file matching MigrationStats.to_dict() shape."""
        report = tmp_path / 'report.json'
        exit_code = cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
                '--report',
                str(report),
            ],
        )
        assert exit_code == 0
        assert report.exists()
        loaded = json.loads(report.read_text(encoding='utf-8'))
        assert loaded['rows_migrated'] == 3
        assert loaded['id_paths'] == {DEFAULT_ID_PATH: {'rewritten': 3, 'not_migrated': 0, 'kept': 0}}
        assert loaded['legacy_id'] is None
        assert 'warnings' in loaded
        assert loaded['errors'] == []

    def test_summary_printed_to_stdout(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``--report``, the summary appears on stdout."""
        cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
            ],
        )
        captured = capsys.readouterr()
        assert 'Migration summary' in captured.out
        assert 'rows migrated' in captured.out

    def test_summary_prints_each_path_and_the_stamp(self, capsys: pytest.CaptureFixture[str]) -> None:
        """One line per path in the order of the run's ``id_paths``, then the stamp line, in singular or plural.

        ``links.*`` comes first although it sorts after the default path, so a summary that
        reordered the paths would fail.
        """
        stats = MigrationStats(
            id_paths={
                'links.*': IdPathCounts(rewritten=1, not_migrated=2, kept=3),
                DEFAULT_ID_PATH: IdPathCounts(rewritten=4, not_migrated=1, kept=1),
            },
            legacy_id=LegacyIdCounts(key='legacy_id', stamped=2, already_present=1, not_stamped=0),
        )

        print_summary(stats, 'sqlite:///source.db', 'sqlite:///target.db')

        lines = capsys.readouterr().out.splitlines()
        first = lines.index('  ids at links.*: 1 rewritten, 2 not ids of migrated entries, 3 other values kept')
        assert lines[first + 1] == (
            f'  ids at {DEFAULT_ID_PATH}: 4 rewritten, 1 not the id of a migrated entry, 1 other value kept'
        )
        assert lines[first + 2] == "  legacy id key 'legacy_id': 2 stamped, 1 already present, 0 not stamped"

    def test_summary_without_a_stamp_key_prints_no_stamp_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The stamp line appears only when ``--legacy-id-key`` is given."""
        stats = MigrationStats(id_paths={DEFAULT_ID_PATH: IdPathCounts()})

        print_summary(stats, 'sqlite:///source.db', 'sqlite:///target.db')

        output = capsys.readouterr().out
        assert f'  ids at {DEFAULT_ID_PATH}: 0 rewritten, 0 not ids of migrated entries, 0 other values kept' in output
        assert 'legacy id key' not in output


class _FakePostgreSQLConnection(AnswersAsOwner):
    """A PostgreSQL connection that accepts the session statements applied right after the dial."""

    async def execute(self, _sql: str, *_args: object) -> str:
        """Accept a session statement.

        Returns:
            The command tag.
        """
        return 'SET'

    async def close(self) -> None:
        """Close nothing."""


class TestPostgreSQLTargetHoldingRows:
    """A PostgreSQL target that already holds ``context_entries`` rows is refused, so a rerun needs an empty one."""

    @pytest.mark.parametrize('source_backend', ['sqlite', 'postgresql'])
    def test_refused_with_exit_status_1(
        self,
        source_backend: str,
        legacy_source_db: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Both runners that write to PostgreSQL record the refusal, store nothing and exit 1."""
        import asyncpg

        import app.cli.migrate_uuid.postgresql_to_postgresql as postgresql_runner
        import app.cli.migrate_uuid.sqlite_to_postgresql as sqlite_runner

        async def _connect(_dsn: str, **_kwargs: object) -> _FakePostgreSQLConnection:
            return _FakePostgreSQLConnection()

        async def _holds_rows(_conn: object, *, schema: str | None = None) -> bool:
            del schema
            return True

        initialize = mock.AsyncMock()
        monkeypatch.setattr(asyncpg, 'connect', _connect)
        for runner in (postgresql_runner, sqlite_runner):
            monkeypatch.setattr(runner, 'target_pg_has_data', _holds_rows)
            monkeypatch.setattr(runner, 'initialize_target_postgresql', initialize)
        source_url = (
            f'sqlite:///{legacy_source_db.as_posix()}' if source_backend == 'sqlite' else 'postgresql://u:p@localhost/src'
        )

        exit_code = run_uuid_migration(
            source_url,
            'postgresql://u:p@localhost/tgt',
            dry_run=False,
            report_path=None,
            id_paths=(IdPath.parse(DEFAULT_ID_PATH),),
            legacy_id_key=None,
        )

        assert exit_code == 1
        lines = capsys.readouterr().out.splitlines()
        assert '  rows migrated: 0' in lines
        assert any(
            line.startswith('    - target PostgreSQL database already contains context_entries rows. ') for line in lines
        )
        initialize.assert_not_awaited()
