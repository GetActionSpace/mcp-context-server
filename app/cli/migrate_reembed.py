"""Re-embed an existing corpus under the currently configured model.

Implements the ``--re-embed`` flag of ``mcp-context-server-migrate``. Unlike
``--embed-missing`` (which only fills entries that LACK an
``embedding_metadata`` row), ``--re-embed`` regenerates embeddings for EVERY
``context_entries`` row that has ``text_content``, deleting any existing
embeddings first. This is the one-command path for switching the embedding
MODEL on an existing database -- the operation the old manual "drop the
embedding tables, set the new model, regenerate" procedure described.

Scope -- model change at the SAME dimension:
    ``--re-embed`` re-embeds with the configured ``EMBEDDING_MODEL`` at the
    EXISTING vector dimension. It deliberately REFUSES a dimension change
    (a configured ``EMBEDDING_DIM`` different from the stored dimension)
    because a dimension change rewrites the vector-storage geometry: the
    fp32 ``vec_context_embeddings`` column width is fixed at creation, and
    under compression the dimension is part of the seed-locked
    ``compression_metadata`` codebook (immutable by design). A dimension
    change is the documented destructive rebuild (back up, recreate the
    database at the new dimension, re-store the data); see
    ``docs/migration-v2-to-v3.md``.

Generation jobs:
    Before the re-embed, the command completes the generation jobs that
    writes made with ``defer_generation`` left outstanding, failed jobs
    included (:mod:`app.cli._generation_drain`), with the configured
    embedding and summary providers. The re-embed then covers the entries
    that hold no job, the completed jobs' entries among them: an entry whose
    job a running server holds, or whose attempt failed, keeps its job and
    is left to it.

Layouts:
    Works for both fp32 and compressed databases. The delete and store paths
    branch on ``ENABLE_EMBEDDING_COMPRESSION`` internally
    (``EmbeddingRepository.delete_all_chunks`` /
    ``EmbeddingRepository.store_chunked``), and
    ``generate_compression_with_timeout`` populates the payload bytes when
    compression is enabled.

The CLI is single-backend: source is the database under operation;
``--target-url`` is ignored (mirrors ``--compress`` / ``--decompress`` /
``--embed-missing``).
"""

import asyncio
import logging
import sqlite3
import sys
import time
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.cli._backend import assert_maintenance_connection
from app.cli._backend import make_backend
from app.cli._backend import owner_refusal
from app.cli._backend import shutdown_backend
from app.cli._database_url import mask_credentials
from app.cli._embedding_introspect import dimension_conflict_error
from app.cli._embedding_introspect import distinct_embedding_models
from app.cli._embedding_introspect import embedding_metadata_table_exists
from app.cli._embedding_runtime import EmbeddingPipelineUnavailableError
from app.cli._embedding_runtime import initialize_cli_embedding_pipeline
from app.cli._embedding_runtime import shutdown_cli_embedding_pipeline
from app.cli._generation_drain import HOLDS_NO_JOB
from app.cli._generation_drain import JobCounts
from app.cli._generation_drain import complete_generation_jobs
from app.cli._generation_drain import count_generation_jobs
from app.cli._generation_drain import generation_jobs_table_exists
from app.cli._summary_runtime import SummaryProviderUnavailableError
from app.cli._summary_runtime import initialize_cli_summary_provider
from app.cli._summary_runtime import shutdown_cli_summary_provider
from app.embeddings.base import EmbeddingProvider
from app.settings import get_settings
from app.summary.base import SummaryProvider

logger = logging.getLogger(__name__)


def _print_warning(*, source_url: str, dry_run: bool) -> None:
    """Print the operator warning describing the cost surface."""
    lines = [
        '=' * 60,
        'RE-EMBED ALL -- this operation deletes every existing embedding',
        'and regenerates it with the configured EMBEDDING_MODEL, issuing',
        'live embedding-provider requests for every entry. Cost (API',
        'requests, latency) scales with the TOTAL number of entries.',
        'Outstanding generation jobs of deferred writes are completed',
        'first, which also calls the summary provider when summary',
        'generation is enabled.',
        'Back up the database before running without --dry-run.',
        '-' * 60,
        f'Source: {source_url}',
    ]
    if dry_run:
        lines.append('DRY-RUN: providers are not called; only the plan is reported')
    lines.append('=' * 60)
    print('\n'.join(lines), file=sys.stderr)


def run_reembed(source_url: str, *, dry_run: bool) -> int:
    """Public entry point for the ``--re-embed`` flag.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, report the plan only; no provider calls.

    Returns:
        Process exit code: 0 success or success-no-op, 1 invalid input or
        provider unavailable, 2 unrecoverable failure.
        A connection row-level security filters, or a privilege error, is not a
        failure of this mode: it propagates to the dispatcher, which prints the
        owner remediation and exits 1.
    """
    masked = mask_credentials(source_url)
    _print_warning(source_url=masked, dry_run=dry_run)
    settings = get_settings()
    if not settings.embedding.generation_enabled:
        print(
            '[ERROR] ENABLE_EMBEDDING_GENERATION=true must be set for '
            '--re-embed. The pipeline calls the embedding provider '
            'configured via EMBEDDING_PROVIDER/EMBEDDING_MODEL.',
            file=sys.stderr,
        )
        return 1

    try:
        return asyncio.run(_reembed_async(source_url, dry_run=dry_run))
    except Exception as exc:
        if owner_refusal(exc) is not None:
            raise
        logger.exception('re-embed failed: %s', exc)
        return 2


async def _reembed_async(source_url: str, *, dry_run: bool) -> int:
    """Async body: validate, complete the generation jobs, then delete + regenerate per entry atomically.

    Defers imports of ``app.tools._generation`` and ``app.repositories`` until
    runtime so the module can be imported by the CLI dispatcher without
    pulling in the embedding/compression dependency chain unconditionally.

    Args:
        source_url: URL passed to ``--source-url``.
        dry_run: When True, report the plan only; no provider calls.

    Returns:
        Process exit code: 0 on success or no-op, 1 on a structural or
        configuration error or when a provider the work needs is unavailable.
    """
    masked = mask_credentials(source_url)
    backend = make_backend(source_url)
    await backend.initialize()
    try:
        await assert_maintenance_connection(backend, get_settings().storage.postgresql_schema)

        from app.repositories import RepositoryContainer
        from app.tools._generation import generate_compression_with_timeout
        from app.tools._generation import generate_embeddings_with_timeout

        repos = RepositoryContainer(backend)

        if not await embedding_metadata_table_exists(backend):
            print(
                '[ERROR] embedding_metadata table not present in source. '
                'Initialize the database against the live server first '
                '(or run the semantic-search migration) so the embedding '
                'tables exist before re-embedding.',
                file=sys.stderr,
            )
            return 1

        settings = get_settings()
        configured_model = settings.embedding.model
        configured_dim = settings.embedding.dim

        # Refuse a dimension change: --re-embed swaps the MODEL at the
        # existing dimension. Changing the dimension rewrites the vector
        # storage geometry (and the seed-locked compressed codebook) and is
        # the documented destructive rebuild. Shared with --embed-missing so
        # the two guards cannot diverge.
        dim_error = await dimension_conflict_error(
            backend, configured_dim=configured_dim,
        )
        if dim_error is not None:
            print(f'[ERROR] {dim_error}', file=sys.stderr)
            return 1

        existing_models = await distinct_embedding_models(backend)
        # A database prepared before deferred generation has no job table and
        # therefore no jobs; every entry is then re-embedded.
        jobs_table = await generation_jobs_table_exists(backend)
        jobs = await count_generation_jobs(backend) if jobs_table else JobCounts(pending=0, failed=0)
        entries = await _find_all_entries(backend, skip_job_holders=jobs_table)

        print(
            f'Re-embedding {len(entries)} entries without a generation job, with model '
            f'{configured_model!r}, on {masked}.',
            file=sys.stderr,
        )
        print(f'Found {jobs.describe()}.', file=sys.stderr)
        changing_models = sorted(m for m in existing_models if m != configured_model)
        if changing_models:
            print(
                f'Existing embeddings used model(s) {changing_models}; '
                f'replacing all with {configured_model!r}.',
                file=sys.stderr,
            )
        if not entries and not jobs.total:
            return 0
        if dry_run:
            print(
                '[DRY-RUN] Provider not called; rerun without --dry-run '
                'to complete the generation jobs and re-embed.',
                file=sys.stderr,
            )
            return 0

        provider: EmbeddingProvider | None = None
        summary_provider: SummaryProvider | None = None
        try:
            try:
                provider = await initialize_cli_embedding_pipeline()
                if jobs.total:
                    summary_provider = await initialize_cli_summary_provider()
            except (EmbeddingPipelineUnavailableError, SummaryProviderUnavailableError) as exc:
                print(f'[ERROR] {exc}', file=sys.stderr)
                return 1

            if jobs.total:
                await complete_generation_jobs(repos, max_attempts=settings.generation_jobs.max_attempts)
                # The completed jobs' entries are re-embedded with the rest; the
                # entries whose job remains are left to it.
                entries = await _find_all_entries(backend, skip_job_holders=True)

            start = time.perf_counter()
            succeeded = 0
            for i, (entry_id, text) in enumerate(entries, start=1):
                chunk_embeddings = await generate_embeddings_with_timeout(text)
                if chunk_embeddings is None:
                    print(
                        f'[SKIP] entry {entry_id}: embedding provider returned '
                        f'no result',
                        file=sys.stderr,
                    )
                    continue

                # No-op when compression is disabled; populates payload bytes
                # when enabled. Mirrors store_context / update_context.
                chunk_embeddings = await generate_compression_with_timeout(chunk_embeddings)
                if chunk_embeddings is None:
                    continue

                # Delete-then-store inside ONE transaction so an entry is
                # never left without embeddings: delete_all_chunks removes the
                # old vectors + embedding_metadata, store_chunked writes the
                # new ones (and the new model_name).
                async with backend.begin_transaction(access_scope=MAINTENANCE) as txn:
                    await repos.embeddings.delete_all_chunks(entry_id, txn=txn, scope=MAINTENANCE)
                    await repos.embeddings.store_chunked(
                        context_id=entry_id,
                        chunk_embeddings=chunk_embeddings,
                        model=configured_model,
                        txn=txn,
                        upsert=False,
                        scope=MAINTENANCE,
                    )
                succeeded += 1

                if i % 50 == 0:
                    elapsed = time.perf_counter() - start
                    rate = i / max(elapsed, 1e-6)
                    eta = (len(entries) - i) / rate if rate > 0 else 0.0
                    print(
                        f'[PROGRESS] {i}/{len(entries)} '
                        f'({rate:.1f} entries/s, ETA {eta:.0f}s)',
                        file=sys.stderr,
                    )

            elapsed = time.perf_counter() - start
            print(
                f'Re-embed complete: {succeeded}/{len(entries)} entries '
                f'embedded with {configured_model!r} in {elapsed:.1f}s.',
                file=sys.stderr,
            )
            return 0
        finally:
            await shutdown_cli_summary_provider(summary_provider)
            await shutdown_cli_embedding_pipeline(provider)
    finally:
        await shutdown_backend(backend)


async def _find_all_entries(backend: StorageBackend, *, skip_job_holders: bool) -> list[tuple[str, str]]:
    """Find every context_entries row with non-NULL ``text_content``.

    ``ORDER BY ce.id`` produces a deterministic order for stable progress
    reporting and reproducible test runs.

    Args:
        backend: Storage backend to query.
        skip_job_holders: Leave out the entries that hold a generation job,
            whose job generates their embeddings; False only for a database
            without the ``generation_jobs`` table.

    Returns:
        List of ``(id, text_content)`` tuples for every row that has a
        non-NULL ``text_content``, regardless of whether the row currently
        has embeddings.
    """
    job_filter = f' AND {HOLDS_NO_JOB}' if skip_job_holders else ''
    sql = (
        'SELECT ce.id, ce.text_content FROM context_entries ce '
        f'WHERE ce.text_content IS NOT NULL{job_filter} ORDER BY ce.id ASC'
    )
    if backend.backend_type == 'sqlite':

        def _read_sqlite(conn: sqlite3.Connection) -> list[tuple[str, str]]:
            return [(str(r[0]), str(r[1])) for r in conn.execute(sql).fetchall()]

        return await backend.execute_read(_read_sqlite, access_scope=MAINTENANCE)

    async def _read_pg(conn: asyncpg.Connection) -> list[tuple[str, str]]:
        return [(str(r['id']), str(r['text_content'])) for r in await conn.fetch(sql)]

    return await backend.execute_read(cast(Any, _read_pg), access_scope=MAINTENANCE)


__all__ = ['run_reembed']
