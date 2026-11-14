"""Tests for the migration report file and the stdout summary."""

import json
from pathlib import Path

import pytest

from app.cli.migrate import main as cli_main
from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import LegacyIdCounts
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.run import print_summary


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
