"""The transaction pass of the update_context_batch tool.

Each update compare-and-sets against the version its entry had before generation. An atomic
batch applies every update in one transaction, all or nothing, and aborts on a concurrent
modification or a vanished entry; a non-atomic batch gives each update its own transaction,
re-reads the version after a conflict, and records a failure for that update alone. Both
retry a lost connection.
"""

import asyncio
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.errors import format_exception_message
from app.repositories.context_repository.records import VersionConflictError
from app.tools._transactions import EntryNotFoundError
from app.tools._transactions import is_connection_error
from app.tools._transactions import reread_entry_version
from app.tools._transactions import transaction_heartbeat
from app.tools._update_transaction import UpdateOutcome
from app.tools._update_transaction import execute_update_in_transaction
from app.tools.batch._update_generation import UpdateGeneration
from app.tools.batch.update_access import reraise_disambiguated_cas_conflict
from app.types import BulkUpdateResultItemDict

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer

logger = logging.getLogger(__name__)


class UpdateCommit(NamedTuple):
    """What the transaction pass of a batch update committed.

    Attributes:
        results: One result per update the pass processed, applied or failed.
        summaries_cleared: Committed updates that cleared the entry's summary.
        deferred_committed: Whether a committed update left a generation job.
    """

    results: list[BulkUpdateResultItemDict]
    summaries_cleared: int
    deferred_committed: bool


def _bumps_version(update: dict[str, Any]) -> bool:
    """Whether applying ``update`` bumps its entry's version.

    Mirrors the execute_update_in_transaction predicate: text, metadata, AND
    visibility all ride the compare-and-set write, which bumps the version by 1.

    Args:
        update: The validated update.

    Returns:
        True when the update writes text, full metadata or visibility.
    """
    return (
        update.get('text') is not None
        or update.get('metadata') is not None
        or update.get('visibility') is not None
    )


async def _update_entry(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    vu_idx: int,
    update: dict[str, Any],
    generation: UpdateGeneration,
    *,
    scope: AccessScope,
    embedding_model: str,
    expected_version: int | None,
) -> UpdateOutcome:
    """Apply one update with its generated outputs inside ``txn``.

    Args:
        repos: Repository container.
        txn: The open transaction.
        vu_idx: The update's position among the validated updates.
        update: The validated update.
        generation: The batch's generated outputs.
        scope: The caller's scope.
        embedding_model: The embedding model the stored embeddings record.
        expected_version: The version the entry's compare-and-set expects.

    Returns:
        What the transaction wrote for the update.
    """
    update_images = update.get('images')
    return await execute_update_in_transaction(
        repos, txn,
        context_id=update['context_id'],
        scope=scope,
        text=update.get('text'),
        metadata=update.get('metadata'),
        metadata_patch=update.get('metadata_patch'),
        summary=generation.summaries.get(vu_idx),
        clear_summary=vu_idx in generation.clear_summaries,
        visibility=update.get('visibility'),
        tags=update.get('tags'),
        images=update_images,
        validated_images=update_images or [],
        chunk_embeddings=generation.embeddings.get(vu_idx),
        embedding_model=embedding_model,
        index_nodes=generation.index_nodes.get(vu_idx),
        expected_version=expected_version,
        defer_generation=vu_idx in generation.deferred,
    )


async def commit_updates_atomically(
    repos: 'RepositoryContainer',
    updates: list[tuple[int, dict[str, Any]]],
    generation: UpdateGeneration,
    entry_versions: dict[str, int],
    *,
    scope: AccessScope,
    embedding_model: str,
) -> UpdateCommit:
    """Apply every update in one transaction, all or nothing.

    A lost connection re-runs the transaction up to twice, each run from the versions
    captured before generation.

    Args:
        repos: Repository container.
        updates: ``(position, update)`` of each update to apply.
        generation: The batch's generated outputs.
        entry_versions: The version of each updated entry captured before generation.
        scope: The caller's scope.
        embedding_model: The embedding model the stored embeddings record.

    Returns:
        The committed results, the number of updates that cleared a summary, and whether
        any update left a generation job.

    Raises:
        ToolError: When a concurrent writer modified an entry or an entry no longer exists.
    """
    backend = repos.context.backend
    max_retries = 2
    attempt = 0

    while True:
        try:
            results_attempt: list[BulkUpdateResultItemDict] = []
            cleared_attempt = 0
            # Running version per id within THIS attempt: a same-id update
            # later in the batch must present the version the earlier same-id
            # update bumped to (the CAS is against the in-transaction row).
            # Reset each attempt because a rolled-back attempt did not commit.
            live_versions = dict(entry_versions)

            async with backend.begin_transaction(access_scope=scope) as txn:
                for idx, (vu_idx, update) in enumerate(updates):
                    context_id = update['context_id']

                    # Heartbeat between entries (skip first)
                    if idx > 0:
                        await transaction_heartbeat(txn)

                    try:
                        outcome = await _update_entry(
                            repos, txn, vu_idx, update, generation,
                            scope=scope,
                            embedding_model=embedding_model,
                            expected_version=live_versions.get(context_id),
                        )
                    except VersionConflictError:
                        # A compare-and-set matching zero rows is ambiguous:
                        # the version changed under a concurrent writer
                        # (retryable, surfaced below as concurrent-
                        # modification advice) or the row was deleted or its
                        # write access withdrawn after its version was
                        # captured (permanent). The helper re-probes the row
                        # under the caller's write predicate on the
                        # transaction connection and re-raises the
                        # disambiguated exception, so such a row aborts with
                        # the standard not-found error every other update
                        # path emits instead of retry advice no retry can
                        # satisfy.
                        await reraise_disambiguated_cas_conflict(repos, txn, context_id, scope=scope)
                    if outcome.summary_cleared:
                        cleared_attempt += 1
                    if _bumps_version(update) and context_id in live_versions:
                        # update_context_entry bumped version by 1; a later
                        # same-id update in this batch must see the new value.
                        live_versions[context_id] += 1

                    results_attempt.append(BulkUpdateResultItemDict(
                        index=update['index'],
                        context_id=context_id,
                        success=True,
                        updated_fields=outcome.updated_fields,
                        error=None,
                        generation_status=outcome.generation_status,
                    ))

                # COMMIT happens here - all or nothing

            # The authoritative per-update cleared count of the committed attempt (a
            # rolled-back or retried attempt contributes nothing).
            return UpdateCommit(results_attempt, cleared_attempt, bool(generation.deferred))

        except VersionConflictError as e:
            # A concurrent writer modified one of these entries during
            # generation (the row still exists -- a row deleted mid-CAS is
            # disambiguated on the transaction connection and aborts as
            # not-found below). Atomic mode is all-or-nothing, so abort the
            # whole batch with a clear error; the client can retry the request.
            raise ToolError(
                f'Concurrent modification during atomic batch update: {e}. Retry the request.',
            ) from None
        except EntryNotFoundError as e:
            # One target entry no longer exists (deleted concurrently or a
            # stale id). Atomic mode is all-or-nothing, so abort the whole
            # batch with a clear error; EntryNotFoundError is a
            # ControlFlowError, so the failed write did not charge the
            # circuit breaker, and no retry can resurrect a deleted row.
            raise ToolError(f'{e}. No entries were updated (atomic batch).') from None
        except ToolError:
            raise  # Logical error -- do not retry
        except Exception as e:
            if is_connection_error(e) and attempt < max_retries:
                delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                attempt += 1
                logger.warning(
                    'Atomic batch update transaction failed, retrying in %.1fs '
                    '(attempt %d/%d): %s',
                    delay, attempt, max_retries, e,
                )
                await asyncio.sleep(delay)
                continue
            raise  # Non-connection error or max retries exceeded


async def commit_updates_individually(
    repos: 'RepositoryContainer',
    updates: list[tuple[int, dict[str, Any]]],
    generation: UpdateGeneration,
    entry_versions: dict[str, int],
    *,
    scope: AccessScope,
    embedding_model: str,
) -> UpdateCommit:
    """Apply each update in its own transaction, recording a failure for that update alone.

    A lost connection re-runs the update's transaction up to twice. A version conflict
    re-reads the entry's version and re-runs it, up to five times; an entry gone or no
    longer writable stops it. A failed update's response-counter contributions are
    reversed.

    Args:
        repos: Repository container.
        updates: ``(position, update)`` of each update to apply.
        generation: The batch's generated outputs.
        entry_versions: The version of each updated entry captured before generation.
        scope: The caller's scope.
        embedding_model: The embedding model the stored embeddings record.

    Returns:
        One result per update, the number of committed updates that cleared a summary,
        and whether a committed update left a generation job.
    """
    backend = repos.context.backend
    results: list[BulkUpdateResultItemDict] = []
    summaries_cleared = 0
    deferred_committed = False
    # Running version per id, persisting ACROSS updates (processed
    # sequentially): a same-id update later in the batch must see the
    # version the earlier same-id update committed.
    live_versions = dict(entry_versions)

    for vu_idx, update in updates:
        original_idx = update['index']
        context_id = update['context_id']
        max_retries = 2
        attempt = 0
        version_conflicts = 0
        max_version_conflicts = 5

        while True:
            try:
                async with backend.begin_transaction(access_scope=scope) as txn:
                    outcome = await _update_entry(
                        repos, txn, vu_idx, update, generation,
                        scope=scope,
                        embedding_model=embedding_model,
                        expected_version=live_versions.get(context_id),
                    )

                    # COMMIT happens here for this update

                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=True,
                    updated_fields=outcome.updated_fields,
                    error=None,
                    generation_status=outcome.generation_status,
                ))
                if outcome.summary_cleared:
                    summaries_cleared += 1
                if vu_idx in generation.deferred:
                    deferred_committed = True
                if _bumps_version(update) and context_id in live_versions:
                    live_versions[context_id] += 1
                break  # Success -- exit retry loop
            except VersionConflictError as e:
                # The row changed since we captured its version (a same-id
                # update earlier in this batch, a concurrent external writer,
                # or a commit whose ack was lost). Re-read the current version
                # and retry so this entry self-heals into a success instead of
                # a spurious failure (mirrors the single update_context path).
                if version_conflicts >= max_version_conflicts:
                    logger.warning('Version conflict updating entry at index %d: %s', original_idx, e)
                    results.append(BulkUpdateResultItemDict(
                        index=original_idx,
                        context_id=context_id,
                        success=False,
                        updated_fields=None,
                        error=format_exception_message(e),
                        generation_status=None,
                    ))
                    generation.discard_counts(vu_idx)
                    break
                version_conflicts += 1
                # Refresh the token before re-entering the write. A transient
                # fault during the refresh is retried inside the helper, on the
                # READ alone: re-running the write transaction with the token
                # whose compare-and-set just failed is doomed by construction
                # (version is monotonic) and would burn a conflict slot on what
                # was only a connection blip. Only an exhausted refresh records
                # a per-entry failure.
                try:
                    current = await reread_entry_version(repos, context_id, scope=scope)
                except Exception as reread_error:
                    logger.error(f'Failed to update entry at index {original_idx}: {reread_error}')
                    results.append(BulkUpdateResultItemDict(
                        index=original_idx,
                        context_id=context_id,
                        success=False,
                        updated_fields=None,
                        error=format_exception_message(reread_error),
                        generation_status=None,
                    ))
                    generation.discard_counts(vu_idx)
                    break
                if not current.exists or not current.can_write:
                    # Gone or no longer writable for the caller: terminal, no retry.
                    results.append(BulkUpdateResultItemDict(
                        index=original_idx,
                        context_id=context_id,
                        success=False,
                        updated_fields=None,
                        error=(
                            f'Context entry {context_id} not found' if not current.exists
                            else f'Not authorized to modify context entry {context_id}'
                        ),
                        generation_status=None,
                    ))
                    generation.discard_counts(vu_idx)
                    break
                assert current.version is not None  # exists=True guarantees a version
                live_versions[context_id] = current.version
                continue
            except EntryNotFoundError as e:
                # This entry no longer exists (deleted concurrently or a
                # stale id). Record a per-entry not-found failure and move
                # on; no retry can resurrect a deleted row, and
                # EntryNotFoundError is a ControlFlowError so the failed
                # write did not charge the circuit breaker.
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=format_exception_message(e),
                    generation_status=None,
                ))
                generation.discard_counts(vu_idx)
                break
            except ToolError as e:
                # Logical error -- do not retry, record as failure
                logger.error(f'Failed to update entry at index {original_idx}: {e}')
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=format_exception_message(e),
                    generation_status=None,
                ))
                generation.discard_counts(vu_idx)
                break
            except Exception as e:
                if is_connection_error(e) and attempt < max_retries:
                    delay = 0.5 * (2 ** attempt)
                    attempt += 1
                    logger.warning(
                        'Non-atomic batch update entry %d failed with connection error, '
                        'retrying in %.1fs (attempt %d/%d): %s',
                        original_idx, delay, attempt, max_retries, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Non-connection error or max retries exceeded
                logger.error(f'Failed to update entry at index {original_idx}: {e}')
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=format_exception_message(e),
                    generation_status=None,
                ))
                generation.discard_counts(vu_idx)
                break

    return UpdateCommit(results, summaries_cleared, deferred_committed)
