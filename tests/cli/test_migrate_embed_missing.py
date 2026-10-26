"""Unit tests for the embed-missing CLI handler.

These tests cover the CLI surface: argparse flag handling, dispatcher
logic, dry-run output shape, env-var validation, and the completion of
the outstanding generation jobs before the backfill, with the generation
and the embedding replaced by recorders. End-to-end coverage with real SQLite/PostgreSQL databases lives in
``tests/integration/sqlite/test_migrate_embed_missing_e2e.py``.
"""

import sqlite3
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


def test_build_parser_accepts_embed_missing_flag() -> None:
    """``--embed-missing`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--embed-missing',
    ])
    assert args.embed_missing is True
    assert args.compress is False
    assert args.decompress is False


def test_build_parser_embed_missing_compose_with_compress() -> None:
    """``--compress --embed-missing`` is NOT mutually exclusive."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress', '--embed-missing',
    ])
    assert args.compress is True
    assert args.embed_missing is True


def test_build_parser_embed_missing_with_decompress_parses() -> None:
    """``--decompress --embed-missing`` is accepted by argparse.

    The flag is OUTSIDE ``mode_group``, so argparse does not reject the
    combination. The dispatcher in ``main()`` honors ``--decompress``
    and skips ``--embed-missing`` per design. Document the parser-level
    acceptance here.
    """
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--decompress', '--embed-missing',
    ])
    assert args.decompress is True
    assert args.embed_missing is True


def test_main_requires_generation_enabled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_embed_missing`` exits 1 when ENABLE_EMBEDDING_GENERATION=false."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'ENABLE_EMBEDDING_GENERATION=true' in err
    assert 'EMBEDDING BACKFILL' in err  # warning printed before validation


def test_no_missing_entries_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the database has no context_entries rows, exits 0 with count=0."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Found 0 entries with missing embeddings' in err


def test_dry_run_emits_count_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` reports the missing count without invoking the provider."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)

    # Seed 2 context_entries rows without embedding_metadata counterparts.
    seed_entry(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A')
    seed_entry(db, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'doc B')

    rc = cli_main([
        '--source-url', f'sqlite:///{db}',
        '--embed-missing', '--dry-run',
    ])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Found 2 entries with missing embeddings' in err
    assert '[DRY-RUN] Provider not called' in err


def test_embedding_metadata_table_missing_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When ``embedding_metadata`` is absent, the CLI surfaces a clear error."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'

    bootstrap_schema(db, with_embedding_metadata=False)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'embedding_metadata table not present' in err


def test_dispatch_compress_then_embed_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--compress --embed-missing`` invokes both, in order."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    call_order: list[str] = []

    def _fake_compress(*_args: object, **_kwargs: object) -> int:
        call_order.append('compress')
        return 0

    def _fake_embed(*_args: object, **_kwargs: object) -> int:
        call_order.append('embed')
        return 0

    with (
        patch('app.cli.migrate_compression.compress.run_compress', side_effect=_fake_compress),
        patch('app.cli.migrate_embeddings.run_embed_missing', side_effect=_fake_embed),
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--embed-missing',
        ])

    assert rc == 0
    assert call_order == ['compress', 'embed']


def test_dispatch_embed_missing_skipped_when_compress_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --compress returns non-zero, --embed-missing is NOT attempted."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_compression.compress.run_compress', return_value=2) as mock_compress,
        patch('app.cli.migrate_embeddings.run_embed_missing') as mock_embed,
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--embed-missing',
        ])

    assert rc == 2
    mock_compress.assert_called_once()
    mock_embed.assert_not_called()


def test_dispatch_embed_missing_standalone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--embed-missing`` standalone dispatches directly to run_embed_missing."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_compression.compress.run_compress') as mock_compress,
        patch('app.cli.migrate_compression.decompress.run_decompress') as mock_decompress,
        patch('app.cli.migrate_embeddings.run_embed_missing', return_value=0) as mock_embed,
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--embed-missing',
        ])

    assert rc == 0
    mock_compress.assert_not_called()
    mock_decompress.assert_not_called()
    mock_embed.assert_called_once()


def test_embed_missing_refuses_model_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Existing embeddings from a different model block the backfill."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_MODEL', 'new-model')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    # One entry already embedded under a DIFFERENT model.
    seed_entry(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A')
    seed_embedding_metadata(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'old-model', 1024)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert "'old-model'" in err
    assert '--re-embed' in err  # points to the whole-corpus path


def test_embed_missing_refuses_dim_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Existing embeddings at a different dimension block the backfill."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_MODEL', 'same-model')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    # Existing embedding recorded at dimension 512 (config wants 1024).
    seed_entry(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A')
    seed_embedding_metadata(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'same-model', 512)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'EMBEDDING_DIM=1024' in err
    assert '[512]' in err


def _seed_compression_metadata(
    path: Path,
    dim: int,
    fingerprint: str | None = None,
) -> None:
    """Create + populate the singleton compression_metadata row at ``dim``."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
                variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
                seed INTEGER NOT NULL CHECK (seed >= 0),
                dim INTEGER NOT NULL CHECK (dim > 0),
                codebook_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 0, ?, ?)",
            (dim, fingerprint),
        )
        conn.commit()
    finally:
        conn.close()


def test_embed_missing_refuses_compression_dim_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A seed-locked compression dim different from EMBEDDING_DIM is refused.

    Regression test for the corruption path where the compressed database is
    sealed at one dimension (compression_metadata) but embedding_metadata is
    still empty: the per-row dimension scan finds nothing, so the guard MUST
    consult compression_metadata to block the mismatched backfill.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '0')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '512')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    # Database sealed at dim=1024 (no embeddings yet); config wants 512.
    _seed_compression_metadata(db, 1024)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'EMBEDDING_DIM=512' in err
    assert 'compression_metadata' in err


def test_embed_missing_refuses_compression_seed_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A COMPRESSION_SEED differing from the sealed seed is refused.

    Regression: the guard consulted only the sealed DIM, not the full sealed
    compression tuple. Backfilling with a different seed/bits/variant/provider
    encodes payloads against an incompatible codebook (seed is load-bearing for
    the TurboQuant rotation), so the guard MUST block it -- not just a dim change.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '7')  # sealed row has seed=0
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')  # matches the sealed dim
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    # Sealed at seed=0/bits=4/variant=ip/dim=1024; only the seed disagrees.
    _seed_compression_metadata(db, 1024)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'COMPRESSION_SEED=7' in err
    assert 'compression_metadata' in err


def test_embed_missing_refuses_codebook_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sealed codebook fingerprint this host cannot reproduce is refused.

    Regression: the guard compared only the sealed scalar tuple. On a host
    whose BLAS/LAPACK build materializes a DIFFERENT numpy.linalg.qr rotation
    for the same (dim, seed) -- the cross-host divergence the fingerprint
    machinery exists to catch -- every scalar matches, yet encoding would write
    payloads in the wrong rotation basis. The guard MUST compare the realized
    fingerprint like the server startup validator and the decompress guard do.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '0')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    # Every scalar matches the sealed row, but the recorded fingerprint is one
    # this host's rotation cannot reproduce (simulating the divergent-BLAS host).
    _seed_compression_metadata(db, 1024, fingerprint='0' * 64)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'fingerprint' in err
    assert 'rotation' in err


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

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing', '--dry-run'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Found 1 entries with missing embeddings and no generation job' in err
    assert 'Found 2 pending and 1 failed generation jobs' in err
    assert '[DRY-RUN] Provider not called' in err
    assert job_statuses(db) == {ENTRY_PENDING: 'pending', ENTRY_FAILED: 'failed', ENTRY_HELD: 'running'}


def test_a_database_without_the_job_table_backfills_every_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A database prepared before deferred generation has no jobs, so every entry lacking embeddings counts."""
    db = generation_env(monkeypatch, tmp_path)
    bootstrap_schema(db)
    seed_entry(db, ENTRY_WITHOUT_JOB, 'doc a')
    seed_entry(db, ENTRY_PENDING, 'doc b')
    drop_jobs_table(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing', '--dry-run'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Found 2 entries with missing embeddings' in err
    assert 'Found 0 pending and 0 failed generation jobs' in err


def test_the_drain_completes_pending_and_failed_jobs_before_the_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pending and failed jobs run first; the pass then covers every entry without a job, never the held one."""
    db = generation_env(monkeypatch, tmp_path)
    seed_jobs_database(db)
    recorder = DrainAndPass()
    recorder.install(monkeypatch)

    with patch(_EMBEDDING_FACTORY, return_value=provider_double()):
        rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 0
    assert recorder.texts('drain') == [text_of(ENTRY_PENDING), text_of(ENTRY_FAILED)]
    assert recorder.texts('pass') == [text_of(ENTRY_WITHOUT_JOB), text_of(ENTRY_PENDING), text_of(ENTRY_FAILED)]
    assert [stage for stage, _text in recorder.events] == ['drain', 'drain', 'pass', 'pass', 'pass']
    assert job_statuses(db) == {ENTRY_HELD: 'running'}
    err = capsys.readouterr().err
    assert (
        'Generation jobs: 2 completed, 0 left for a later attempt, 0 failed, 0 discarded '
        '(1 failed job returned to the queue first).'
    ) in err
    assert '1 pending and 0 failed generation jobs remain' in err


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
        rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 0
    assert recorder.summary_providers == [summary_provider, summary_provider]
    summary_provider.shutdown.assert_awaited_once()
    assert get_summary_provider() is None


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
        rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    assert "[ERROR] Summary provider 'fake' is not available" in capsys.readouterr().err
    assert recorder.events == []
    assert job_statuses(db) == {ENTRY_PENDING: 'pending', ENTRY_FAILED: 'failed', ENTRY_HELD: 'running'}
    embedding_provider.shutdown.assert_awaited_once()


def test_no_jobs_need_no_summary_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without jobs the backfill creates no summary provider, even with summary generation enabled."""
    db = generation_env(monkeypatch, tmp_path, summaries=True)
    bootstrap_schema(db)
    seed_entry(db, ENTRY_WITHOUT_JOB, 'doc a')
    recorder = DrainAndPass()
    recorder.install(monkeypatch)

    with (
        patch(_EMBEDDING_FACTORY, return_value=provider_double()),
        patch(_SUMMARY_FACTORY) as summary_factory,
    ):
        rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 0
    summary_factory.assert_not_called()
    assert recorder.events == [('pass', 'doc a')]
