"""Tests for the mcp-context-ingest command line: the flags, their validation and the exit codes main returns."""

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import get_default_environment

from app.cli.ingest import cli
from app.cli.ingest.apply import RunConfig
from app.cli.ingest.apply import run_sync
from app.cli.ingest.cli import build_parser
from app.cli.ingest.cli import main
from app.cli.ingest.cli import parse_config
from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import EXIT_TRANSIENT
from app.cli.ingest.report import RunReport
from app.models import MAX_THREAD_ID_LENGTH
from tests.cli.ingest._fake_server import FakeServer

SERVER = 'http://127.0.0.1:8765/mcp'
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _required(source_dir: Path, *extra: str) -> list[str]:
    """Return the required flags of a run over a directory, followed by extra arguments."""
    return ['--server', SERVER, '--corpus', 'handbook', '--source-dir', str(source_dir), *extra]


def _refused(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Parse arguments that must be refused and return the usage error argparse printed."""
    with pytest.raises(SystemExit) as raised:
        parse_config(argv)
    assert raised.value.code == EXIT_CONFIG
    return capsys.readouterr().err


def _fake_run(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make main run its sync against an in-memory server."""

    async def run(config: RunConfig) -> RunReport:
        return await run_sync(config, connect=server.connect)

    monkeypatch.setattr(cli, 'run_sync', run)


class TestDefaults:
    """Every flag left out takes its documented default."""

    def test_defaults(self, tmp_path: Path) -> None:
        """A run with only the required flags syncs Markdown files privately in batches of ten."""
        config = parse_config(_required(tmp_path))

        assert config == RunConfig(server=SERVER, corpus='handbook', source_dir=tmp_path)
        assert config.thread_id == 'kb:handbook'
        assert config.includes == ('**/*.md',)
        assert (config.excludes, config.tags, dict(config.metadata)) == ((), (), {})
        assert (config.visibility, config.defer_generation) == ('private', False)
        assert (config.token_file, config.token_env, config.report) == (None, None, None)
        assert (config.dry_run, config.allow_empty_corpus) == (False, False)
        assert (config.batch_size, config.max_request_bytes, config.call_timeout) == (10, 3_145_728, 3600.0)
        assert config.concurrency == 1

    def test_repeatable_flags_collect_in_order_and_an_include_replaces_the_default(self, tmp_path: Path) -> None:
        """--include, --exclude and --tag repeat; an explicit include drops the Markdown default."""
        config = parse_config(_required(
            tmp_path,
            '--include', 'docs/**', '--include', '*.txt', '--exclude', 'drafts/**', '--tag', 'guide', '--tag', 'ops',
            '--metadata-json', '{"team": "docs", "level": 2}', '--visibility', 'public', '--token-env', 'INGEST_TOKEN',
            '--dry-run', '--allow-empty-corpus', '--report', str(tmp_path / 'report.json'), '--batch-size', '100',
            '--max-request-bytes', '4194304', '--call-timeout', '90.5', '--concurrency', '4', '--defer-generation',
        ))

        assert config.includes == ('docs/**', '*.txt')
        assert config.excludes == ('drafts/**',)
        assert config.tags == ('guide', 'ops')
        assert dict(config.metadata) == {'team': 'docs', 'level': 2}
        assert (config.visibility, config.token_env, config.defer_generation) == ('public', 'INGEST_TOKEN', True)
        assert (config.dry_run, config.allow_empty_corpus) == (True, True)
        assert config.report == tmp_path / 'report.json'
        assert (config.batch_size, config.max_request_bytes, config.call_timeout) == (100, 4_194_304, 90.5)
        assert config.concurrency == 4


class TestValidation:
    """A flag value the command cannot use is a usage error, exit 2, before anything runs."""

    def test_token_file_and_token_env_are_mutually_exclusive(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Only one token source can be named."""
        error = _refused(_required(tmp_path, '--token-file', str(tmp_path / 't'), '--token-env', 'T'), capsys)

        assert 'not allowed with argument' in error

    def test_a_token_is_never_accepted_on_the_command_line(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """No flag carries a token value: the token flags name a file or a variable."""
        options = {option for action in build_parser()._actions for option in action.option_strings}

        assert '--token' not in options
        assert {'--token-file', '--token-env'} <= options
        assert '--token' in _refused(_required(tmp_path, '--token', 'secret-token'), capsys)

    @pytest.mark.parametrize('name', ['handbook', 'team.docs_v2-1', '0', 'a' * (MAX_THREAD_ID_LENGTH - 3)])
    def test_corpus_names_that_are_accepted(self, name: str, tmp_path: Path) -> None:
        """A lowercase name of letters, digits, dots, underscores and hyphens names the thread kb:<name>."""
        argv = ['--server', SERVER, '--corpus', name, '--source-dir', str(tmp_path)]

        assert parse_config(argv).thread_id == f'kb:{name}'

    @pytest.mark.parametrize('name', ['Handbook', '-docs', '.docs', 'my docs', 'kb:docs', 'a' * (MAX_THREAD_ID_LENGTH - 2)])
    def test_corpus_names_that_are_refused(self, name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Uppercase, a leading symbol, spaces, a colon and a thread id past the length bound are refused."""
        error = _refused(['--server', SERVER, '--corpus', name, '--source-dir', str(tmp_path)], capsys)

        assert '--corpus' in error

    @pytest.mark.parametrize(
        ('value', 'reason'),
        [
            ('not json', 'not valid JSON'),
            ('[1, 2]', 'a JSON object'),
            ('{"ingest": {}}', 'ingest'),
            ('{"score": NaN}', 'NaN'),
        ],
    )
    def test_metadata_json_must_be_an_object_without_the_reserved_key(
        self, value: str, reason: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--metadata-json takes a JSON object of plain values that does not set the ingest block."""
        error = _refused(_required(tmp_path, '--metadata-json', value), capsys)

        assert '--metadata-json' in error
        assert reason in error

    @pytest.mark.parametrize(
        ('flag', 'value'),
        [
            ('--batch-size', '0'),
            ('--batch-size', '101'),
            ('--batch-size', 'ten'),
            ('--max-request-bytes', '0'),
            ('--max-request-bytes', '4194305'),
            ('--call-timeout', '0'),
            ('--call-timeout', '-5'),
            ('--concurrency', '0'),
            ('--concurrency', '21'),
            ('--concurrency', 'two'),
            ('--visibility', 'shared'),
            ('--include', '[z-a].md'),
        ],
    )
    def test_out_of_range_values_are_refused(
        self, flag: str, value: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Batch sizes and concurrency stay within their bounds, the request budget within 4 MiB, timeouts positive."""
        error = _refused(_required(tmp_path, flag, value), capsys)

        assert flag in error

    def test_the_source_directory_must_exist(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A missing directory or a file is refused, so a mistyped path never reads as an empty corpus."""
        file = tmp_path / 'file.md'
        file.write_text('x', encoding='utf-8')

        assert '--source-dir' in _refused(_required(tmp_path / 'missing'), capsys)
        assert '--source-dir' in _refused(_required(file), capsys)

    @pytest.mark.parametrize('server', ['localhost:8000/mcp', 'ftp://alice:secret-pass@example.com/mcp', 'http:///mcp'])
    def test_the_server_must_be_an_http_url_and_its_userinfo_is_never_echoed(
        self, server: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--server takes an http or https URL with a host; a refused value is shown without userinfo."""
        error = _refused(['--server', server, '--corpus', 'handbook', '--source-dir', str(tmp_path)], capsys)

        assert '--server' in error
        assert 'secret-pass' not in error

    def test_help_names_every_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The help lists the flags of the command."""
        build_parser().print_help()
        printed = capsys.readouterr().out

        for flag in (
            '--server', '--corpus', '--source-dir', '--include', '--exclude', '--tag', '--metadata-json',
            '--visibility', '--token-file', '--token-env', '--dry-run', '--allow-empty-corpus', '--report',
            '--batch-size', '--max-request-bytes', '--call-timeout', '--concurrency', '--defer-generation',
        ):
            assert flag in printed


class TestMain:
    """main runs the sync, prints the summary and the errors, writes the report and returns the exit code."""

    def test_a_converged_run_exits_0_and_writes_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The summary goes to stdout and the JSON report to the named file."""
        corpus_root = tmp_path / 'corpus'
        corpus_root.mkdir()
        (corpus_root / 'a.md').write_text('alpha', encoding='utf-8')
        report_path = tmp_path / 'report.json'
        _fake_run(FakeServer(), monkeypatch)

        code = main(_required(corpus_root, '--report', str(report_path)))

        captured = capsys.readouterr()
        assert code == EXIT_CONVERGED
        assert 'Sync of kb:handbook with http://127.0.0.1:8765/mcp' in captured.out
        assert captured.out.rstrip().endswith('exit 0: completed')
        assert captured.err == ''
        written = json.loads(report_path.read_text(encoding='utf-8'))
        assert (written['exit_code'], written['counts']['created']) == (EXIT_CONVERGED, 1)

    def test_a_deferring_run_exits_0_and_counts_the_pending_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Entries left pending are counted in the summary and the report, and the run still converges."""
        corpus_root = tmp_path / 'corpus'
        corpus_root.mkdir()
        (corpus_root / 'a.md').write_text('alpha', encoding='utf-8')
        report_path = tmp_path / 'report.json'
        _fake_run(FakeServer(), monkeypatch)

        code = main(_required(corpus_root, '--defer-generation', '--report', str(report_path)))

        captured = capsys.readouterr()
        assert code == EXIT_CONVERGED
        assert 'withheld 0, generation pending 1' in captured.out
        assert captured.out.rstrip().endswith('exit 0: completed')
        written = json.loads(report_path.read_text(encoding='utf-8'))
        assert (written['exit_code'], written['counts']['generation_pending']) == (EXIT_CONVERGED, 1)

    def test_run_errors_go_to_stderr_and_set_the_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A partial run exits 1 and prints each run-level error on stderr."""
        (tmp_path / 'a.md').write_text('alpha', encoding='utf-8')
        server = FakeServer()
        server.failing_sources['a.md'] = 'embedding provider unavailable'
        _fake_run(server, monkeypatch)

        code = main(_required(tmp_path))

        captured = capsys.readouterr()
        assert code == EXIT_PARTIAL
        assert 'failed: a.md: embedding provider unavailable' in captured.out
        assert 'error: ' in captured.err
        assert 'did not converge' in captured.err

    def test_an_empty_source_directory_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Without --allow-empty-corpus a directory without documents ends the run before any contact."""
        code = main(_required(tmp_path))

        captured = capsys.readouterr()
        assert code == EXIT_CONFIG
        assert '--allow-empty-corpus' in captured.err
        assert captured.out.rstrip().endswith('exit 2: stopped on a configuration, authentication or plan error')

    def test_an_unwritable_report_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A --report path that cannot be written is a configuration error named on stderr."""
        corpus_root = tmp_path / 'corpus'
        corpus_root.mkdir()
        (corpus_root / 'a.md').write_text('alpha', encoding='utf-8')
        _fake_run(FakeServer(), monkeypatch)

        code = main(_required(corpus_root, '--report', str(tmp_path)))

        assert code == EXIT_CONFIG
        assert f'cannot write the report {tmp_path}' in capsys.readouterr().err

    def test_an_unreachable_server_exits_69(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A server nobody listens on is a transient failure."""
        (tmp_path / 'a.md').write_text('alpha', encoding='utf-8')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]

        code = main(['--server', f'http://127.0.0.1:{port}/mcp', '--corpus', 'handbook', '--source-dir', str(tmp_path)])

        assert code == EXIT_TRANSIENT
        assert f'cannot reach http://127.0.0.1:{port}/mcp' in capsys.readouterr().err

    def test_output_a_pipe_cannot_encode_never_ends_the_run(self, tmp_path: Path) -> None:
        """A source id the output encoding cannot represent is printed escaped instead of crashing the command."""
        (tmp_path / '日本.md').write_bytes(b'\xff not utf-8')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]

        completed = subprocess.run(
            [
                sys.executable, '-m', 'app.cli.ingest.cli',
                '--server', f'http://127.0.0.1:{port}/mcp', '--corpus', 'handbook', '--source-dir', str(tmp_path),
            ],
            env=get_default_environment(),
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
            check=False,
        )

        stderr = completed.stderr.decode('utf-8', errors='replace')
        assert completed.returncode == EXIT_TRANSIENT, stderr
        assert 'Traceback' not in stderr
        assert 'kept: ' in completed.stdout.decode('utf-8', errors='replace')
