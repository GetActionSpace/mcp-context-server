"""Backfill missing embeddings via the active embedding provider.

Implements the ``--embed-missing`` CLI flag of
``mcp-context-server-migrate``. Identifies context_entries rows that
lack an embedding_metadata row and runs the live embedding pipeline
against their ``text_content`` to fill the gap.

Generation jobs:
    Before the backfill, the command completes the generation jobs that
    writes made with ``defer_generation`` left outstanding, failed jobs
    included (:mod:`app.cli._generation_drain`), with the configured
    embedding and summary providers. The backfill then covers the entries
    that hold no job: an entry whose job a running server holds, or whose
    attempt failed, keeps its job and is left to it.

Modes:
    Standalone:
        Backfills against the existing storage layout. When
        ``ENABLE_EMBEDDING_COMPRESSION=true`` is set in the invocation
        env, the freshly generated embeddings land in
        ``vec_context_embeddings_compressed``; otherwise they land in
        the fp32 ``vec_context_embeddings`` table.

    Composed with ``--compress``:
        The dispatcher in :mod:`app.cli.migrate` runs ``--compress``
        first (fp32 -> compressed), then dispatches to this module to
        backfill any missing entries directly into the compressed
        layout.

The CLI is single-backend: source is the database under operation;
``--target-url`` is ignored (mirrors ``--compress`` / ``--decompress``).
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
        'EMBEDDING BACKFILL -- this operation issues live embedding-',
        'provider requests against text_content for every entry that',
        'lacks an embedding. Cost (API requests, latency) scales with',
        'the number of missing entries. Outstanding generation jobs of',
        'deferred writes are completed first, which also calls the',
        'summary provider when summary generation is enabled.',
        '-' * 60,
        f'Source: {source_url}',
    ]
    if dry_run:
        lines.append('DRY-RUN: providers are not called; only the counts are reported')
    lines.append('=' * 60)
    print('\n'.join(lines), file=sys.stderr)


def run_embed_missing(source_url: str, *, dry_run: bool) -> int:
    """Public entry point for the ``--embed-missing`` flag.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, only count and report; no provider calls.

    Returns:
        Process exit code: 0 success or success-no-op, 1 invalid input,
        2 unrecoverable failure.
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
            '--embed-missing. The pipeline calls the embedding provider '
            'configured via EMBEDDING_PROVIDER/EMBEDDING_MODEL.',
            file=sys.stderr,
        )
        return 1

    try:
        return asyncio.run(_embed_missing_async(source_url, dry_run=dry_run))
    except Exception as exc:
        if owner_refusal(exc) is not None:
            raise
        logger.exception('embed-missing failed: %s', exc)
        return 2


async def _embed_missing_async(source_url: str, *, dry_run: bool) -> int:
    """Async body: complete the generation jobs, then identify missing entries, generate, store atomically.

    Defers imports of ``app.tools._generation`` and ``app.repositories`` until
    runtime so the module can be imported by the CLI dispatcher without
    pulling in the embedding/compression dependency chain unconditionally.

    Args:
        source_url: URL passed to ``--source-url``.
        dry_run: When True, only count and report; no provider calls.

    Returns:
        Process exit code: 0 on success or no-op, 1 when the
        ``embedding_metadata`` table is absent from the source database, the
        existing embeddings disagree with the configuration, or a provider
        the work needs is unavailable.
    """
    masked = mask_credentials(source_url)
    backend = make_backend(source_url)
    await backend.initialize()
    try:
        await assert_maintenance_connection(backend, get_settings().storage.postgresql_schema)

        # Deferred imports: keep numpy / heavy provider machinery out of
        # the dispatcher's import graph.
        from app.repositories import RepositoryContainer
        from app.tools._generation import generate_compression_with_timeout
        from app.tools._generation import generate_embeddings_with_timeout

        repos = RepositoryContainer(backend)

        if not await embedding_metadata_table_exists(backend):
            # Schema predates the semantic-search migration; every row
            # would qualify as missing but the storage path would fail
            # because the target tables do not exist. Surface a clearer
            # error here first.
            print(
                '[ERROR] embedding_metadata table not present in source. '
                'Initialize the database against the live server first '
                '(or run the semantic-search migration) so the target '
                'tables exist.',
                file=sys.stderr,
            )
            return 1

        settings = get_settings()
        model = settings.embedding.model
        configured_dim = settings.embedding.dim

        # Pre-flight: refuse to backfill into a database whose EXISTING
        # embeddings use a different model or dimension. --embed-missing fills
        # only the entries that lack embeddings and leaves the rest untouched,
        # so a model/dim mismatch would mix incompatible embedding spaces in
        # one table and corrupt semantic search. Surfaced before generation so
        # no provider calls are wasted.
        conflict = await _embedding_consistency_error(
            backend, configured_model=model, configured_dim=configured_dim,
        )
        if conflict is not None:
            print(f'[ERROR] {conflict}', file=sys.stderr)
            return 1

        # A database prepared before deferred generation has no job table and
        # therefore no jobs; every entry lacking embeddings is then backfilled.
        jobs_table = await generation_jobs_table_exists(backend)
        jobs = await count_generation_jobs(backend) if jobs_table else JobCounts(pending=0, failed=0)
        missing = await _find_missing_entries(backend, skip_job_holders=jobs_table)
        print(
            f'Found {len(missing)} entries with missing embeddings and no generation job on {masked}.',
            file=sys.stderr,
        )
        print(f'Found {jobs.describe()}.', file=sys.stderr)
        if not missing and not jobs.total:
            return 0
        if dry_run:
            print(
                '[DRY-RUN] Provider not called; rerun without --dry-run '
                'to complete the generation jobs and backfill.',
                file=sys.stderr,
            )
            return 0

        # The CLI does not run the server lifespan, so the global embedding
        # provider + chunking service, and the summary provider the generation
        # jobs need, are unset. Initialize them exactly as the server does;
        # without this, get_embedding_provider() returns None and every entry
        # would be silently skipped.
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
                # The completed jobs' entries now hold their outputs; the entries
                # whose job remains are left to it.
                missing = await _find_missing_entries(backend, skip_job_holders=True)

            start = time.perf_counter()
            succeeded = 0
            for i, (entry_id, text) in enumerate(missing, start=1):
                chunk_embeddings = await generate_embeddings_with_timeout(text)
                if chunk_embeddings is None:
                    print(
                        f'[SKIP] entry {entry_id}: embedding provider returned '
                        f'no result',
                        file=sys.stderr,
                    )
                    continue

                # generate_compression_with_timeout is a no-op when
                # compression.enabled is False; populates the payload field
                # when True. Calling it unconditionally mirrors the live
                # server's store_context / update_context code path.
                chunk_embeddings = await generate_compression_with_timeout(chunk_embeddings)
                if chunk_embeddings is None:
                    continue

                async with backend.begin_transaction(access_scope=MAINTENANCE) as txn:
                    await repos.embeddings.store_chunked(
                        context_id=entry_id,
                        chunk_embeddings=chunk_embeddings,
                        model=model,
                        txn=txn,
                        upsert=False,
                        scope=MAINTENANCE,
                    )
                succeeded += 1

                if i % 50 == 0:
                    elapsed = time.perf_counter() - start
                    rate = i / max(elapsed, 1e-6)
                    eta = (len(missing) - i) / rate if rate > 0 else 0.0
                    print(
                        f'[PROGRESS] {i}/{len(missing)} '
                        f'({rate:.1f} entries/s, ETA {eta:.0f}s)',
                        file=sys.stderr,
                    )

            elapsed = time.perf_counter() - start
            print(
                f'Backfill complete: {succeeded}/{len(missing)} entries '
                f'embedded in {elapsed:.1f}s.',
                file=sys.stderr,
            )
            return 0
        finally:
            await shutdown_cli_summary_provider(summary_provider)
            await shutdown_cli_embedding_pipeline(provider)
    finally:
        await shutdown_backend(backend)


async def _embedding_consistency_error(
    backend: StorageBackend,
    *,
    configured_model: str,
    configured_dim: int,
) -> str | None:
    """Return an actionable message if existing embeddings disagree with config.

    ``--embed-missing`` fills entries that LACK embeddings and leaves existing
    embeddings untouched. If those existing embeddings were produced by a
    different model or dimension than the configured ``EMBEDDING_MODEL`` /
    ``EMBEDDING_DIM``, backfilling new entries would mix incompatible embedding
    spaces in one table and corrupt semantic search.

    Args:
        backend: Storage backend to query.
        configured_model: The process's configured ``EMBEDDING_MODEL``.
        configured_dim: The process's configured ``EMBEDDING_DIM``.

    Returns:
        An operator-actionable message, or ``None`` when there is no conflict
        (no existing embeddings -- the canonical post-cross-backend-migration
        case -- or they already match the configured model and dimension).
    """
    # Dimension guard, shared with --re-embed so the two cannot diverge. It
    # covers both the per-row embedding_metadata dimensions AND the seed-locked
    # compression_metadata dimension (the latter matters even when no
    # embeddings exist yet -- a freshly sealed compressed database).
    dim_error = await dimension_conflict_error(backend, configured_dim=configured_dim)
    if dim_error is not None:
        return dim_error

    models = await distinct_embedding_models(backend)
    conflicting_models = sorted(m for m in models if m != configured_model)
    if conflicting_models:
        return (
            f'existing embeddings were generated with model(s) '
            f'{conflicting_models} but EMBEDDING_MODEL={configured_model!r}. '
            f'Backfilling missing entries with a different model would mix '
            f'incompatible embedding spaces. To re-embed the whole corpus with '
            f'{configured_model!r}, run --re-embed. To backfill consistently, '
            f'set EMBEDDING_MODEL to the existing model.'
        )
    return None


async def _find_missing_entries(backend: StorageBackend, *, skip_job_holders: bool) -> list[tuple[str, str]]:
    """Find context_entries rows lacking an embedding_metadata row.

    ORDER BY ``ce.id`` produces a deterministic order for stable progress
    reporting and reproducible test runs.

    Args:
        backend: Storage backend to query.
        skip_job_holders: Leave out the entries that hold a generation job,
            whose job generates their embeddings; False only for a database
            without the ``generation_jobs`` table.

    Returns:
        List of ``(id, text_content)`` tuples for rows that have no
        corresponding ``embedding_metadata`` entry and a non-NULL
        ``text_content``.
    """
    job_filter = f' AND {HOLDS_NO_JOB}' if skip_job_holders else ''
    sql = (
        'SELECT ce.id, ce.text_content FROM context_entries ce '
        'LEFT JOIN embedding_metadata em ON em.context_id = ce.id '
        f'WHERE em.context_id IS NULL AND ce.text_content IS NOT NULL{job_filter} '
        'ORDER BY ce.id ASC'
    )
    if backend.backend_type == 'sqlite':

        def _read_sqlite(conn: sqlite3.Connection) -> list[tuple[str, str]]:
            return [(str(r[0]), str(r[1])) for r in conn.execute(sql).fetchall()]

        return await backend.execute_read(_read_sqlite, access_scope=MAINTENANCE)

    async def _read_pg(conn: asyncpg.Connection) -> list[tuple[str, str]]:
        return [(str(r['id']), str(r['text_content'])) for r in await conn.fetch(sql)]

    return await backend.execute_read(cast(Any, _read_pg), access_scope=MAINTENANCE)


__all__ = ['run_embed_missing']
