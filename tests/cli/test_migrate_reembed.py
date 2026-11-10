"""Unit tests for the --re-embed CLI handler.

These tests cover the CLI surface: argparse flag handling, mutual
exclusivity, dispatcher logic, dry-run output shape, env-var validation,
the dimension-change refusal, and the completion of the outstanding
generation jobs before the re-embed, with the generation and the embedding
replaced by recorders. End-to-end coverage with a real SQLite
database lives in ``tests/integration/sqlite/test_migrate_reembed_e2e.py``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main
from app.settings import get_settings
from app.startup import get_summary_provider
from tests.cli._embedding_db import ENTRY_FAILED
from tests.cli._embedding_db import ENTRY_HELD
from tests.cli._embedding_db import ENTRY_PENDING
from tests.cli._embedding_db import ENTRY_WITHOUT_JOB
from tests.cli._embedding_db import DrainAndPass
from tests.cli._embedding_db import bootstrap_schema
from tests.cli._embedding_db import drop_jobs_table
from tests.cli._embedding_db import generation_env
from tests.cli._embedding_db import job_statuses
from tests.cli._embedding_db import provider_double
from tests.cli._embedding_db import seed_embedding_metadata
from tests.cli._embedding_db import seed_entry
from tests.cli._embedding_db import seed_jobs_database
from tests.cli._embedding_db import text_of


def test_build_parser_accepts_re_embed_flag() -> None:
    """``--re-embed`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args(['--source-url', 'sqlite:///fake.db', '--re-embed'])
    assert args.re_embed is True
    assert args.compress is False
    assert args.decompress is False


def test_re_embed_mutually_exclusive_with_compress() -> None:
    """``--re-embed --compress`` is rejected by argparse (mode_group)."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--source-url', 'sqlite:///fake.db', '--re-embed', '--compress'])


def test_re_embed_mutually_exclusive_with_decompress() -> None:
    """``--re-embed --decompress`` is rejected by argparse (mode_group)."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--source-url', 'sqlite:///fake.db', '--re-embed', '--decompress'])


def test_main_requires_generation_enabled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_reembed`` exits 1 when ENABLE_EMBEDDING_GENERATION=false."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'ENABLE_EMBEDDING_GENERATION=true' in err
    assert 'RE-EMBED ALL' in err  # warning printed before validation


def test_embedding_metadata_table_missing_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When ``embedding_metadata`` is absent, the CLI surfaces a clear error."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db, with_embedding_metadata=False)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'embedding_metadata table not present' in err


def test_dry_run_emits_plan_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` reports the entry count without invoking the provider."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    seed_entry(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A')
    seed_entry(db, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'doc B')

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed', '--dry-run'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Re-embedding 2 entries' in err
    assert '[DRY-RUN] Provider not called' in err


def test_dimension_change_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A configured EMBEDDING_DIM different from the stored dim is refused."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    seed_entry(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A')
    # Existing embedding recorded at a DIFFERENT dimension (512 != 1024).
    seed_embedding_metadata(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'old-model', 512)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'EMBEDDING_DIM=1024' in err
    assert '[512]' in err


def test_dispatch_re_embed_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--re-embed`` standalone dispatches directly to run_reembed."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_compression.compress.run_compress') as mock_compress,
        patch('app.cli.migrate_compression.decompress.run_decompress') as mock_decompress,
        patch('app.cli.migrate_reembed.run_reembed', return_value=0) as mock_reembed,
    ):
        rc = cli_main(['--source-url', 'sqlite:///fake.db', '--re-embed'])

    assert rc == 0
    mock_compress.assert_not_called()
    mock_decompress.assert_not_called()
    mock_reembed.assert_called_once()


def test_dispatch_re_embed_supersedes_embed_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--re-embed --embed-missing`` runs only re-embed (the superset)."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_reembed.run_reembed', return_value=0) as mock_reembed,
        patch('app.cli.migrate_embeddings.run_embed_missing') as mock_embed,
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--re-embed', '--embed-missing',
        ])

    assert rc == 0
    mock_reembed.assert_called_once()
    mock_embed.assert_not_called()


_EMBEDDING_FACTORY = 'app.cli._embedding_runtime.create_embedding_provider'
_SUMMARY_FACTORY = 'app.cli._summary_runtime.create_summary_provider'


def test_dry_run_reports_the_generation_jobs_and_leaves_their_entries_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` counts the pending and failed jobs, counts only entries without one, and changes nothing."""
    db = generation_env(monkeypatch, tmp_path)
    seed_jobs_database(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed', '--dry-run'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Re-embedding 1 entries without a generation job' in err
    assert 'Found 2 pending and 1 failed generation jobs' in err
    assert '[DRY-RUN] Provider not called' in err
    assert job_statuses(db) == {ENTRY_PENDING: 'pending', ENTRY_FAILED: 'failed', ENTRY_HELD: 'running'}


def test_a_database_without_the_job_table_re_embeds_every_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A database prepared before deferred generation has no jobs, so every entry counts."""
    db = generation_env(monkeypatch, tmp_path)
    bootstrap_schema(db)
    seed_entry(db, ENTRY_WITHOUT_JOB, 'doc a')
    seed_entry(db, ENTRY_PENDING, 'doc b')
    drop_jobs_table(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed', '--dry-run'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Re-embedding 2 entries' in err
    assert 'Found 0 pending and 0 failed generation jobs' in err


def test_the_drain_completes_pending_and_failed_jobs_before_the_re_embed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pending and failed jobs run first; the re-embed then covers every entry without a job, never the held one."""
    db = generation_env(monkeypatch, tmp_path)
    seed_jobs_database(db)
    recorder = DrainAndPass()
    recorder.install(monkeypatch)

    with patch(_EMBEDDING_FACTORY, return_value=provider_double()):
        rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 0
    assert recorder.texts('drain') == [text_of(ENTRY_PENDING), text_of(ENTRY_FAILED)]
    assert recorder.texts('pass') == [text_of(ENTRY_WITHOUT_JOB), text_of(ENTRY_PENDING), text_of(ENTRY_FAILED)]
    assert [stage for stage, _text in recorder.events] == ['drain', 'drain', 'pass', 'pass', 'pass']
    assert job_statuses(db) == {ENTRY_HELD: 'running'}
    err = capsys.readouterr().err
    assert 'Generation jobs: 2 completed' in err
    assert '1 pending and 0 failed generation jobs remain' in err


def test_an_unavailable_summary_provider_stops_before_the_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Jobs to drain with an unavailable summary provider exit 1 before any job or entry is touched."""
    db = generation_env(monkeypatch, tmp_path, summaries=True)
    seed_jobs_database(db)
    recorder = DrainAndPass()
    recorder.install(monkeypatch)
    embedding_provider = provider_double()

    with (
        patch(_EMBEDDING_FACTORY, return_value=embedding_provider),
        patch(_SUMMARY_FACTORY, return_value=provider_double(available=False)),
    ):
        rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 1
    assert "[ERROR] Summary provider 'fake' is not available" in capsys.readouterr().err
    assert recorder.events == []
    assert job_statuses(db) == {ENTRY_PENDING: 'pending', ENTRY_FAILED: 'failed', ENTRY_HELD: 'running'}
    embedding_provider.shutdown.assert_awaited_once()


def test_the_drain_runs_with_the_summary_provider_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With summary generation enabled the drain runs with the configured provider, which is shut down after."""
    db = generation_env(monkeypatch, tmp_path, summaries=True)
    seed_jobs_database(db)
    recorder = DrainAndPass()
    recorder.install(monkeypatch)
    summary_provider = provider_double()

    with (
        patch(_EMBEDDING_FACTORY, return_value=provider_double()),
        patch(_SUMMARY_FACTORY, return_value=summary_provider),
    ):
        rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 0
    assert recorder.summary_providers == [summary_provider, summary_provider]
    summary_provider.shutdown.assert_awaited_once()
    assert get_summary_provider() is None
