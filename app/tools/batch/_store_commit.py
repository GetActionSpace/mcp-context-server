"""The transaction pass of the store_context_batch tool.

An atomic batch stores every entry in one transaction, all or nothing; a non-atomic batch
gives each entry its own transaction and records a failure for that entry alone. Both retry
a lost connection, and both regenerate the outputs of an entry whose deduplication diverged
inside its transaction and run the transaction again.
"""

import asyncio
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.errors import format_exception_message
from app.tools._generation import node_layer_active
from app.tools._transactions import EmbeddingsReconcileRequiredError
from app.tools._transactions import StoreOutcome
from app.tools._transactions import execute_store_in_transaction
from app.tools._transactions import is_connection_error
from app.tools._transactions import transaction_heartbeat
from app.tools.batch._store_generation import StoreGeneration
from app.tools.batch._store_generation import regenerate_diverged_entries
from app.tools.batch._store_generation import regenerate_diverged_entry
from app.types import BulkStoreResultItemDict

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.embeddings import EmbeddingProvider
    from app.repositories import RepositoryContainer
    from app.summary import SummaryProvider

logger = logging.getLogger(__name__)


class StoreCommit(NamedTuple):
    """What the transaction pass of a batch store committed.

    Attributes:
        results: One result per entry the pass processed, stored or failed.
        embeddings_stored: Entries whose transaction wrote embeddings.
        deferred_committed: Whether a committed entry left a generation job.
    """

    results: list[BulkStoreResultItemDict]
    embeddings_stored: int
    deferred_committed: bool


async def _store_entry(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    ve_idx: int,
    entry: dict[str, Any],
    generation: StoreGeneration,
    *,
    scope: AccessScope,
    author_group_grants: frozenset[str],
    embedding_provider: 'EmbeddingProvider | None',
    embedding_model: str,
) -> StoreOutcome:
    """Store one entry with its generated outputs inside ``txn``.

    Args:
        repos: Repository container.
        txn: The open transaction.
        ve_idx: The entry's position among the validated entries.
        entry: The validated entry.
        generation: The batch's generated outputs.
        scope: The caller's scope.
        author_group_grants: The author groups a fresh entry grants read access to.
        embedding_provider: The active embedding provider, or None.
        embedding_model: The embedding model the stored embeddings record.

    Returns:
        What the transaction wrote for the entry.
    """
    return await execute_store_in_transaction(
        repos, txn,
        thread_id=entry['thread_id'],
        source=entry['source'],
        content_type=entry['content_type'],
        text_content=entry['text_content'],
        scope=scope,
        visibility=entry['visibility'],
        author_group_grants=author_group_grants,
        metadata_str=entry['metadata'],
        summary=generation.summaries.get(ve_idx),
        tags=entry.get('tags'),
        validated_images=entry.get('images', []),
        images_provided=bool(entry.get('images_provided')),
        deduplicate=entry['deduplicate'],
        defer_generation=ve_idx in generation.deferred,
        chunk_embeddings=generation.embeddings.get(ve_idx),
        embedding_model=embedding_model,
        embedding_generation_enabled=embedding_provider is not None,
        index_nodes=generation.index_nodes.get(ve_idx),
        nodes_pending=(
            node_layer_active()
            and generation.likely_duplicates.get(ve_idx) is not None
        ),
        summary_pending=ve_idx in generation.preserved_summaries,
    )


async def commit_stores_atomically(
    repos: 'RepositoryContainer',
    entries: list[tuple[int, dict[str, Any]]],
    generation: StoreGeneration,
    *,
    scope: AccessScope,
    author_group_grants: frozenset[str],
    embedding_provider: 'EmbeddingProvider | None',
    summary_provider: 'SummaryProvider | None',
    embedding_model: str,
) -> StoreCommit:
    """Store every entry in one transaction, all or nothing.

    A lost connection re-runs the transaction up to twice. A deduplication that diverged
    inside the transaction regenerates the skipped outputs outside it and re-runs it, at
    most once per entry.

    Args:
        repos: Repository container.
        entries: ``(position, entry)`` of each entry to store.
        generation: The batch's generated outputs.
        scope: The caller's scope.
        author_group_grants: The author groups a fresh entry grants read access to.
        embedding_provider: The active embedding provider, or None.
        summary_provider: The active summary provider, or None.
        embedding_model: The embedding model the stored embeddings record.

    Returns:
        The committed results, the number of entries that stored embeddings, and whether
        any entry left a generation job.

    Raises:
        ToolError: When the reconcile passes are exhausted.
    """
    backend = repos.context.backend
    max_retries = 2
    attempt = 0
    # A pre-check-skipped embedding can diverge once per distinct entry,
    # so bound reconciliation passes by the number of entries.
    reconcile_passes = 0
    max_reconcile_passes = len(entries)

    while True:
        try:
            results_attempt: list[BulkStoreResultItemDict] = []
            # Count stored embeddings per attempt; only report them once the
            # transaction commits, so connection retries and reconciliation
            # passes never double-count.
            stored_count_attempt = 0

            async with backend.begin_transaction(access_scope=scope) as txn:
                for idx, (ve_idx, entry) in enumerate(entries):
                    # Heartbeat between entries (skip first)
                    if idx > 0:
                        await transaction_heartbeat(txn)

                    outcome = await _store_entry(
                        repos, txn, ve_idx, entry, generation,
                        scope=scope,
                        author_group_grants=author_group_grants,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                    )

                    if outcome.embedding_stored:
                        stored_count_attempt += 1

                    results_attempt.append(BulkStoreResultItemDict(
                        index=entry['index'],
                        success=True,
                        context_id=outcome.context_id,
                        error=None,
                        generation_status=outcome.generation_status,
                    ))

                # COMMIT happens here - all or nothing

            return StoreCommit(results_attempt, stored_count_attempt, bool(generation.deferred))

        except EmbeddingsReconcileRequiredError as reconcile:
            # The read-only pre-check skipped generation for a likely duplicate,
            # but the transaction inserted a new entry. Regenerate OUTSIDE the
            # transaction (generation-first) for the diverging text -- and any
            # other entry sharing that exact text whose outputs were also
            # skipped -- then re-run the whole atomic transaction.
            if reconcile_passes >= max_reconcile_passes:
                raise ToolError(
                    'Failed to reconcile embeddings after deduplication '
                    'divergence (atomic batch store)',
                ) from None
            reconcile_passes += 1
            await regenerate_diverged_entries(
                generation, entries, reconcile.text_content,
                embedding_provider=embedding_provider,
                summary_provider=summary_provider,
            )
            continue

        except ToolError:
            raise  # Logical error -- do not retry
        except Exception as e:
            if is_connection_error(e) and attempt < max_retries:
                delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                attempt += 1
                logger.warning(
                    'Atomic batch store transaction failed, retrying in %.1fs '
                    '(attempt %d/%d): %s',
                    delay, attempt, max_retries, e,
                )
                await asyncio.sleep(delay)
                continue
            raise  # Non-connection error or max retries exceeded


async def commit_stores_individually(
    repos: 'RepositoryContainer',
    entries: list[tuple[int, dict[str, Any]]],
    generation: StoreGeneration,
    *,
    scope: AccessScope,
    author_group_grants: frozenset[str],
    embedding_provider: 'EmbeddingProvider | None',
    summary_provider: 'SummaryProvider | None',
    embedding_model: str,
) -> StoreCommit:
    """Store each entry in its own transaction, recording a failure for that entry alone.

    A lost connection re-runs the entry's transaction up to twice. A deduplication that
    diverged inside it regenerates the entry's skipped outputs outside it and re-runs it
    once. A failed entry's response-counter contributions are reversed.

    Args:
        repos: Repository container.
        entries: ``(position, entry)`` of each entry to store.
        generation: The batch's generated outputs.
        scope: The caller's scope.
        author_group_grants: The author groups a fresh entry grants read access to.
        embedding_provider: The active embedding provider, or None.
        summary_provider: The active summary provider, or None.
        embedding_model: The embedding model the stored embeddings record.

    Returns:
        One result per entry, the number of entries that stored embeddings, and whether a
        committed entry left a generation job.
    """
    backend = repos.context.backend
    results: list[BulkStoreResultItemDict] = []
    embeddings_stored = 0
    deferred_committed = False

    for ve_idx, entry in entries:
        original_idx = entry['index']
        max_retries = 2
        attempt = 0
        reconciled = False

        while True:
            try:
                async with backend.begin_transaction(access_scope=scope) as txn:
                    outcome = await _store_entry(
                        repos, txn, ve_idx, entry, generation,
                        scope=scope,
                        author_group_grants=author_group_grants,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                    )

                    # COMMIT happens here for this entry

                if outcome.embedding_stored:
                    embeddings_stored += 1
                if ve_idx in generation.deferred:
                    deferred_committed = True

                results.append(BulkStoreResultItemDict(
                    index=original_idx,
                    success=True,
                    context_id=outcome.context_id,
                    error=None,
                    generation_status=outcome.generation_status,
                ))
                break  # Success -- exit retry loop

            except EmbeddingsReconcileRequiredError as reconcile:
                # Pre-check skipped generation for a likely duplicate, but this
                # store inserted a new entry. Regenerate OUTSIDE the transaction
                # (generation-first) and retry this entry once.
                if reconciled:
                    logger.error(
                        'Failed to reconcile embeddings for entry at index %d '
                        'after deduplication divergence', original_idx,
                    )
                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=False,
                        context_id=None,
                        error='Failed to reconcile embeddings after deduplication divergence',
                        generation_status=None,
                    ))
                    generation.discard_counts(ve_idx)
                    break
                reconciled = True
                # The regeneration is abort-mandatory, so a failure fails ONLY this
                # entry: escaping the per-entry loop would abort the whole batch and
                # discard sibling results already collected, breaking the documented
                # per-entry isolation of atomic=false. (The atomic pass deliberately
                # lets it abort: all-or-nothing.)
                regeneration_error = await regenerate_diverged_entry(
                    generation, ve_idx, entry, reconcile.text_content,
                    embedding_provider=embedding_provider,
                    summary_provider=summary_provider,
                )
                if regeneration_error is not None:
                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=False,
                        context_id=None,
                        error=regeneration_error,
                        generation_status=None,
                    ))
                    break
                continue

            except ToolError as e:
                # Logical error -- do not retry, record as failure
                logger.error(f'Failed to store entry at index {original_idx}: {e}')
                results.append(BulkStoreResultItemDict(
                    index=original_idx,
                    success=False,
                    context_id=None,
                    error=format_exception_message(e),
                    generation_status=None,
                ))
                generation.discard_counts(ve_idx)
                break
            except Exception as e:
                if is_connection_error(e) and attempt < max_retries:
                    delay = 0.5 * (2 ** attempt)
                    attempt += 1
                    logger.warning(
                        'Non-atomic batch store entry %d failed with connection error, '
                        'retrying in %.1fs (attempt %d/%d): %s',
                        original_idx, delay, attempt, max_retries, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Non-connection error or max retries exceeded
                logger.error(f'Failed to store entry at index {original_idx}: {e}')
                results.append(BulkStoreResultItemDict(
                    index=original_idx,
                    success=False,
                    context_id=None,
                    error=format_exception_message(e),
                    generation_status=None,
                ))
                generation.discard_counts(ve_idx)
                break

    return StoreCommit(results, embeddings_stored, deferred_committed)
