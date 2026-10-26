"""Tests for the ingest client's run report: counts, the exit code, the summary and the JSON file."""

import json
from pathlib import Path
from typing import Literal

import pytest

from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import EXIT_TRANSIENT
from app.cli.ingest.report import DocumentOutcome
from app.cli.ingest.report import RunReport
from app.cli.ingest.report import render_summary
from app.cli.ingest.report import write_json

SERVER = 'http://localhost:8000/mcp'
THREAD_ID = 'kb:handbook'
CONTEXT_ID = '0190abcdef1234567890abcdef123456'
ZERO_COUNTS = {
    'created': 0,
    'updated': 0,
    'deleted': 0,
    'unchanged': 0,
    'kept': 0,
    'foreign': 0,
    'failed': 0,
    'withheld': 0,
    'generation_pending': 0,
}


def _report(*, server: str = SERVER, dry_run: bool = False) -> RunReport:
    """Return an empty report of a run against the handbook corpus."""
    return RunReport(server=server, thread_id=THREAD_ID, dry_run=dry_run)


class TestCounts:
    """Every outcome is counted under its own key, and the run-level counts are added directly."""

    def test_a_new_report_counts_nothing(self) -> None:
        """A report starts with every count at zero, the withheld deletes included, and exit code 0."""
        report = _report()

        assert report.counts == ZERO_COUNTS
        assert report.outcomes == []
        assert report.errors == []
        assert report.exit_code == EXIT_CONVERGED

    def test_each_document_outcome_is_counted(self) -> None:
        """Adding an outcome records it and counts it under its outcome."""
        report = _report()
        outcomes = [
            DocumentOutcome(source_id='a.md', outcome='created', context_id=CONTEXT_ID, error=None),
            DocumentOutcome(source_id='b.md', outcome='updated', context_id=CONTEXT_ID, error=None),
            DocumentOutcome(source_id='c.md', outcome='deleted', context_id=CONTEXT_ID, error=None),
            DocumentOutcome(source_id='d.md', outcome='unchanged', context_id=CONTEXT_ID, error=None),
            DocumentOutcome(source_id='e.md', outcome='unchanged', context_id=CONTEXT_ID, error=None),
        ]

        for outcome in outcomes:
            report.add(outcome)

        assert report.outcomes == outcomes
        assert report.counts == {**ZERO_COUNTS, 'created': 1, 'updated': 1, 'deleted': 1, 'unchanged': 2}
        assert report.exit_code == EXIT_CONVERGED

    def test_foreign_and_withheld_counts_accumulate(self) -> None:
        """Foreign members and withheld deletes are run-level counts added per plan slice."""
        report = _report()

        report.count('foreign', 2)
        report.count('foreign', 3)
        report.count('withheld', 4)

        assert report.counts == {**ZERO_COUNTS, 'foreign': 5, 'withheld': 4}
        assert report.outcomes == []

    def test_pending_generation_is_counted_without_changing_the_exit_code(self) -> None:
        """Written entries whose generation the server left pending are a run-level count, never a problem."""
        report = _report()
        report.add(DocumentOutcome(source_id='a.md', outcome='created', context_id=CONTEXT_ID, error=None))

        report.count('generation_pending', 1)
        report.count('generation_pending', 2)

        assert report.counts == {**ZERO_COUNTS, 'created': 1, 'generation_pending': 3}
        assert report.exit_code == EXIT_CONVERGED

    @pytest.mark.parametrize('outcome', ['failed', 'kept'])
    def test_a_failed_or_kept_document_makes_the_run_partial(self, outcome: Literal['failed', 'kept']) -> None:
        """A document the run could not synchronize, or could only keep, exits 1."""
        report = _report()

        report.add(DocumentOutcome(source_id='a.md', outcome=outcome, context_id=None, error='unreadable'))

        assert report.counts[outcome] == 1
        assert report.exit_code == EXIT_PARTIAL


class TestExitCode:
    """The exit code of a run is the most severe one any of its problems calls for."""

    @pytest.mark.parametrize(
        ('codes', 'expected'),
        [
            ([], EXIT_CONVERGED),
            ([EXIT_CONVERGED], EXIT_CONVERGED),
            ([EXIT_PARTIAL], EXIT_PARTIAL),
            ([EXIT_PARTIAL, EXIT_CONVERGED], EXIT_PARTIAL),
            ([EXIT_PARTIAL, EXIT_TRANSIENT], EXIT_TRANSIENT),
            ([EXIT_TRANSIENT, EXIT_PARTIAL], EXIT_TRANSIENT),
            ([EXIT_TRANSIENT, EXIT_CONFIG], EXIT_CONFIG),
            ([EXIT_CONFIG, EXIT_TRANSIENT], EXIT_CONFIG),
            ([EXIT_CONFIG, EXIT_PARTIAL], EXIT_CONFIG),
            ([EXIT_PARTIAL, EXIT_CONFIG, EXIT_TRANSIENT, EXIT_CONVERGED], EXIT_CONFIG),
        ],
    )
    def test_precedence_is_config_then_transient_then_partial(self, codes: list[int], expected: int) -> None:
        """2 outranks 69, which outranks 1, which outranks 0, in whatever order they arrive."""
        report = _report()

        for code in codes:
            report.escalate(code)

        assert report.exit_code == expected

    def test_an_unknown_exit_code_is_refused(self) -> None:
        """Only the four exit codes of the command can be recorded."""
        report = _report()

        with pytest.raises(ValueError, match='unknown exit code 3'):
            report.escalate(3)

    def test_a_run_level_error_is_recorded_with_its_exit_code(self) -> None:
        """A run-level error is kept in order and raises the exit code it calls for."""
        report = _report()

        report.fail('directory not listed: private: Permission denied', exit_code=EXIT_PARTIAL)
        report.fail('cannot reach http://localhost:8000/mcp: refused', exit_code=EXIT_TRANSIENT)

        assert report.errors == [
            'directory not listed: private: Permission denied',
            'cannot reach http://localhost:8000/mcp: refused',
        ]
        assert report.exit_code == EXIT_TRANSIENT


class TestSummary:
    """The human summary names the run, every count, each document problem and the exit code."""

    def test_summary_of_a_partial_run(self) -> None:
        """A run with a failed and a kept document lists both after the counts."""
        report = _report()
        report.add(DocumentOutcome(source_id='a.md', outcome='created', context_id=CONTEXT_ID, error=None))
        report.add(DocumentOutcome(source_id='b.md', outcome='failed', context_id=None, error='Context entry b not found'))
        report.add(DocumentOutcome(source_id='c.md', outcome='kept', context_id=None, error='the file cannot be read'))
        report.count('withheld', 2)

        assert render_summary(report) == (
            'Sync of kb:handbook with http://localhost:8000/mcp\n'
            'created 1, updated 0, deleted 0, unchanged 0, kept 1, foreign 0, failed 1, withheld 2, generation pending 0\n'
            'failed: b.md: Context entry b not found\n'
            'kept: c.md: the file cannot be read\n'
            'exit 1: completed with problems: some documents were not synchronized'
        )

    def test_summary_names_the_pending_generation(self) -> None:
        """The count of written entries whose generation is pending reads as words in the summary."""
        report = _report()
        report.add(DocumentOutcome(source_id='a.md', outcome='created', context_id=CONTEXT_ID, error=None))
        report.count('generation_pending', 1)

        assert render_summary(report).splitlines()[1:] == [
            'created 1, updated 0, deleted 0, unchanged 0, kept 0, foreign 0, failed 0, withheld 0, generation pending 1',
            'exit 0: completed',
        ]

    def test_summary_of_a_dry_run(self) -> None:
        """A dry run says it planned and wrote nothing, in its first line and in its exit line."""
        report = _report(dry_run=True)
        report.add(DocumentOutcome(source_id='a.md', outcome='created', context_id=None, error=None))

        assert render_summary(report) == (
            'Dry run of kb:handbook with http://localhost:8000/mcp: nothing was written\n'
            'created 1, updated 0, deleted 0, unchanged 0, kept 0, foreign 0, failed 0, withheld 0, generation pending 0\n'
            'exit 0: planned; nothing was written'
        )

    def test_a_dry_run_with_problems_keeps_the_partial_meaning(self) -> None:
        """Only a dry run that ends with exit 0 says it planned; a kept file still makes it partial."""
        report = _report(dry_run=True)
        report.add(DocumentOutcome(source_id='c.md', outcome='kept', context_id=None, error='the file cannot be read'))

        assert render_summary(report).splitlines()[-1] == (
            'exit 1: completed with problems: some documents were not synchronized'
        )

    @pytest.mark.parametrize(
        ('code', 'meaning'),
        [
            (EXIT_CONFIG, 'stopped on a configuration, authentication or plan error'),
            (EXIT_TRANSIENT, 'stopped on a transient failure; rerun later'),
        ],
    )
    @pytest.mark.parametrize('dry_run', [False, True])
    def test_summary_names_the_stopping_exit_codes(self, code: int, meaning: str, dry_run: bool) -> None:
        """The exit line explains a run that stopped, a dry run whose plan failed included."""
        report = _report(dry_run=dry_run)
        report.escalate(code)

        assert render_summary(report).splitlines()[-1] == f'exit {code}: {meaning}'


class TestJson:
    """The JSON report carries the run and its outcomes, and never a secret of the command line."""

    def test_report_file_holds_every_field(self, tmp_path: Path) -> None:
        """The file holds the server, thread, dry-run flag, exit code, counts, documents and errors."""
        report = _report()
        report.add(DocumentOutcome(source_id='a.md', outcome='created', context_id=CONTEXT_ID, error=None))
        report.add(DocumentOutcome(source_id='b.md', outcome='failed', context_id=None, error='not found'))
        report.count('foreign', 1)
        report.fail('directory not listed: private: Permission denied', exit_code=EXIT_PARTIAL)
        path = tmp_path / 'report.json'

        write_json(report, path)

        assert json.loads(path.read_text(encoding='utf-8')) == {
            'server': SERVER,
            'thread_id': THREAD_ID,
            'dry_run': False,
            'exit_code': EXIT_PARTIAL,
            'counts': {**ZERO_COUNTS, 'created': 1, 'failed': 1, 'foreign': 1},
            'documents': [
                {'source_id': 'a.md', 'outcome': 'created', 'context_id': CONTEXT_ID, 'error': None},
                {'source_id': 'b.md', 'outcome': 'failed', 'context_id': None, 'error': 'not found'},
            ],
            'errors': ['directory not listed: private: Permission denied'],
        }

    def test_non_ascii_source_ids_are_written_as_text(self, tmp_path: Path) -> None:
        """The file is UTF-8 with non-ASCII source ids written as characters, not escapes."""
        report = _report()
        report.add(DocumentOutcome(source_id='заметки/日本.md', outcome='created', context_id=CONTEXT_ID, error=None))
        path = tmp_path / 'report.json'

        write_json(report, path)

        assert 'заметки/日本.md' in path.read_text(encoding='utf-8')

    @pytest.mark.parametrize(
        ('server', 'shown'),
        [
            ('http://alice:s3cret-pass@localhost:8000/mcp', 'http://localhost:8000/mcp'),
            ('https://token-in-userinfo@kb.example.com/mcp', 'https://kb.example.com/mcp'),
            ('http://alice:s3cret-pass@[::1]:8001/mcp', 'http://[::1]:8001/mcp'),
            ('http://localhost:8000/mcp', 'http://localhost:8000/mcp'),
        ],
    )
    def test_userinfo_never_reaches_the_report(
        self, server: str, shown: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The report keeps the server URL without its userinfo, in memory, in the file and in the summary."""
        report = _report(server=server)
        path = tmp_path / 'report.json'

        write_json(report, path)
        print(render_summary(report))

        written = path.read_text(encoding='utf-8')
        printed = capsys.readouterr().out
        assert report.server == shown
        assert json.loads(written)['server'] == shown
        for secret in ('alice', 's3cret-pass', 'token-in-userinfo'):
            assert secret not in written
            assert secret not in printed
