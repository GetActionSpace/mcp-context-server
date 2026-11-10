"""Tests for the mcp-context-server-migrate dispatcher: argument parsing, configuration errors and the owner refusal."""

import importlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import asyncpg
import pytest

from app.cli._backend import MaintenanceConnectionError
from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main


class TestCliArgs:
    """Argparse plumbing for the migrate CLI."""

    def test_help_runs_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``--help`` exits with code 0 and prints usage."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(['--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert 'mcp-context-server-migrate' in captured.out
        assert '--source-url' in captured.out

    def test_missing_source_url_errors(self) -> None:
        """Calling without ``--source-url`` triggers a non-zero exit."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(['--target-url', 'sqlite:///dummy.db'])
        assert exc_info.value.code != 0

    def test_build_parser_has_required_flags(self) -> None:
        """The argparse parser declares all expected options."""
        parser = build_parser()
        actions = {action.dest for action in parser._actions}
        assert {'source_url', 'target_url', 'dry_run', 'report', 'reassign_owner'}.issubset(actions)

    @pytest.mark.parametrize('flag', ['--embed-missing', '--re-embed'])
    def test_the_embedding_modes_say_they_complete_the_generation_jobs_first(self, flag: str) -> None:
        """The help of both embedding modes says they complete the deferred writes' generation jobs first."""
        parser = build_parser()
        help_text = next(action.help for action in parser._actions if flag in action.option_strings)

        assert help_text is not None
        assert 'defer_generation' in help_text
        assert 'generation jobs' in help_text


class TestSettingsValidationExitCode:
    """A settings ValidationError exits EX_CONFIG (78) on every CLI path.

    The server's guarded import (app/server.py) classifies an import-time
    settings ValidationError as a permanent misconfiguration and exits 78,
    but mcp-context-server-migrate never imports app.server: its in-place
    flows reach get_settings() through the lazily imported backend modules.
    Without its own mapping, a value like RETRY_MAX_RETRIES=0 would surface
    there as a raw multi-frame pydantic traceback with the generic exit 1 --
    the failure mode the server guard eliminates, on a second first-class
    entry point.
    """

    def test_in_place_dispatch_maps_validation_error_to_exit_78(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--compress under an invalid env exits 78 with the pydantic detail."""
        from app.errors import ConfigurationError
        from app.settings import get_settings

        db = tmp_path / 'src.db'
        sqlite3.connect(str(db)).close()
        monkeypatch.setenv('RETRY_MAX_RETRIES', '0')
        get_settings.cache_clear()
        try:
            rc = cli_main(['--source-url', f'sqlite:///{db}', '--compress', '--dry-run'])
        finally:
            monkeypatch.delenv('RETRY_MAX_RETRIES', raising=False)
            get_settings.cache_clear()

        assert rc == ConfigurationError.EXIT_CODE
        err = capsys.readouterr().err
        assert 'Configuration invalid' in err
        assert 'RETRY_MAX_RETRIES' in err or 'retry_max_retries' in err


def test_build_parser_accepts_compress_flag() -> None:
    """``--compress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.compress is True
    assert args.decompress is False


def test_build_parser_accepts_decompress_flag() -> None:
    """``--decompress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--decompress',
    ])
    assert args.decompress is True
    assert args.compress is False


def test_build_parser_rejects_both_flags() -> None:
    """``--compress`` and ``--decompress`` are mutually exclusive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--decompress',
        ])


def test_build_parser_target_url_optional_when_compress() -> None:
    """``--target-url`` is optional when --compress is set."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.target_url is None


_OWNER_REFUSAL = (
    "This command must connect with the table owner's credentials: row-level security filters mcp_app on "
    "context_entries. Rerun with the owner's connection string (docs/migration-v2-to-v3.md)."
)
_PG_SOURCE = 'postgresql://mcp_app:app-secret@db.internal:5432/mcp_context'

# mode flags, the module and name of the async body the dispatcher reaches, and the compression switch the mode needs
_MODES = [
    pytest.param(
        ['--reassign-owner', 'local', 'alice'], 'app.cli.migrate_reassign_owner', '_reassign_async', None, id='reassign',
    ),
    pytest.param(['--compress'], 'app.cli.migrate_compression.compress', '_compress_async', 'true', id='compress'),
    pytest.param(['--decompress'], 'app.cli.migrate_compression.decompress', '_decompress_async', 'false', id='decompress'),
    pytest.param(['--re-embed'], 'app.cli.migrate_reembed', '_reembed_async', None, id='re-embed'),
    pytest.param(['--embed-missing'], 'app.cli.migrate_embeddings', '_embed_missing_async', None, id='embed-missing'),
    pytest.param(['--disable-rls'], 'app.cli.migrate_disable_rls', '_disable_rls_async', None, id='disable-rls'),
    pytest.param(
        ['--target-url', 'postgresql://mcp_owner:owner-secret@db.internal:5432/target'],
        'app.cli.migrate_uuid.run',
        'run_migration_postgresql',
        None,
        id='uuid-migration',
    ),
]


class TestOwnerRefusalExitCode:
    """Every PostgreSQL mode lets the owner refusal and a privilege error reach the dispatcher, which exits 1."""

    @pytest.fixture(autouse=True)
    def _fresh_settings(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        from app.settings import get_settings

        monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()

    @pytest.mark.parametrize(('flags', 'module', 'body', 'compression'), _MODES)
    @pytest.mark.parametrize(
        ('error', 'message'),
        [
            pytest.param(MaintenanceConnectionError(_OWNER_REFUSAL), _OWNER_REFUSAL, id='precondition'),
            pytest.param(
                asyncpg.exceptions.InsufficientPrivilegeError('permission denied for table context_entries'),
                "This command must connect with the table owner's credentials: the database refused a statement "
                "(permission denied for table context_entries). Rerun with the owner's connection string "
                '(docs/migration-v2-to-v3.md).',
                id='42501',
            ),
        ],
    )
    def test_the_refusal_exits_1_with_the_remediation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        flags: list[str],
        module: str,
        body: str,
        compression: str | None,
        error: Exception,
        message: str,
    ) -> None:
        from app.settings import get_settings

        if compression is not None:
            monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', compression)
            get_settings.cache_clear()

        async def _refuse(*_args: object, **_kwargs: object) -> int:
            raise error

        monkeypatch.setattr(importlib.import_module(module), body, _refuse)

        rc = cli_main(['--source-url', _PG_SOURCE, *flags])

        assert rc == 1
        assert capsys.readouterr().err.splitlines()[-1] == message
