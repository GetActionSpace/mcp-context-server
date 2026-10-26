"""The update_context_batch tool: update many context entries in one call."""

import asyncio
import logging
import operator
from collections.abc import Awaitable
from typing import Annotated
from typing import Any
from typing import cast

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_effective_principal
from app.auth import resolve_write_scope
from app.auth.access import reject_read_only_principal
from app.errors import format_exception_message
from app.repositories.context_repository.records import VersionConflictError
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.startup import notify_generation_worker
from app.tools._generation import generate_compression_with_timeout
from app.tools._generation import generate_embeddings_with_timeout
from app.tools._generation import generate_index_nodes_with_timeout
from app.tools._generation import generate_summary_with_timeout
from app.tools._generation import generation_active
from app.tools._responses import build_batch_update_response_message
from app.tools._transactions import EntryNotFoundError
from app.tools._transactions import execute_update_in_transaction
from app.tools._transactions import is_connection_error
from app.tools._transactions import reread_entry_version
from app.tools._transactions import transaction_heartbeat
from app.tools.batch.entry_validation import validate_update_entry
from app.tools.batch.update_access import authorize_updates
from app.tools.batch.update_access import reraise_disambiguated_cas_conflict
from app.types import BulkUpdateResponseDict
from app.types import BulkUpdateResultItemDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def update_context_batch(
    updates: Annotated[
        list[dict[str, Any]],
        Field(
            description='List of update operations. Each must have context_id (str, accepts 32-char hex, '
            '36-char hyphenated UUID, or 8-31 char hex prefix). '
            'Optional: text (str), metadata (dict - full replace), '
            'metadata_patch (dict - RFC 7396 merge), tags (list[str]), images (list[dict]), '
            'visibility ("private" or "public"; owner-only, publishing as public '
            'may require a configured role), defer_generation (bool, default false; with text, '
            'true stores the new text at once and leaves its embeddings, summary and section '
            'summaries to a background job).',
            min_length=1,
            max_length=100,
        ),
    ],
    atomic: Annotated[
        bool,
        Field(
            description='If true, ALL updates succeed or NONE are applied. '
            'If false, partial success allowed.',
        ),
    ] = True,
) -> BulkUpdateResponseDict:
    """Update multiple context entries in a batch.

    Atomicity modes:
    - atomic=True (default): ALL updates succeed or NONE are applied (transaction rollback).
    - atomic=False: Each update processed independently with per-item error reporting.

    Update semantics per entry:
    - Each update is identified by context_id
    - Only provided fields are modified
    - Immutable fields (cannot be changed): id, thread_id, source, created_at, ownership
    - Auto-managed fields: content_type (recalculated based on images), updated_at
    - text, metadata, tags and images: the owner or a principal holding a write grant
    - visibility: owner-only; publishing as 'public' may require a configured role
    - An entry the caller may not read is reported as "Context entry {id} not found";
      one it may read but not modify as "Not authorized to modify context entry {id}"
      (atomic mode appends " at index {i}" and fails the whole batch)
    - Metadata options (MUTUALLY EXCLUSIVE per entry):
      - metadata: FULL REPLACEMENT of entire metadata object
      - metadata_patch: RFC 7396 JSON Merge Patch (new keys added, existing updated,
        null values DELETE keys; cannot store null values; arrays replaced entirely)
    - Tags and images use REPLACEMENT semantics (not merge)
    - A text change regenerates the entry's embeddings, summary and section summaries
      before the batch is applied, unless the update sets defer_generation=True, which
      stores the new text at once, removes the old text's outputs and leaves the new ones
      to a background job; an update without text generates nothing and leaves any job as
      it is
    - Each applied update's generation_status is 'ready' when the entry's generated outputs
      are complete, 'pending' while its generation job waits or runs, and 'failed' once the
      job exhausted its retries; a failed update reports null

    Size limits:
    - Maximum 100 entries per batch
    - Image limits per entry: 10MB each, 100MB total
    - Tag limits per entry: 100 tags, each at most 128 characters
    - Value under an indexed metadata field: at most 512 characters

    Returns:
        BulkUpdateResponseDict with success (bool), total (int), succeeded (int),
        failed (int), results (list of index, context_id, success, updated_fields, error,
        generation_status), message (str).

    Raises:
        ToolError: If validation fails, embedding generation fails (atomic), or batch operation fails.
    """
    try:
        # The caller's identity, resolved once before anything that can reveal whether
        # an entry exists. The read-only principal is refused before any entry is
        # validated or the repositories are reached; id resolution and the existence
        # probes below see only the entries the caller may read, and the publish gate
        # needs the caller's roles.
        principal = resolve_effective_principal()
        reject_read_only_principal(principal, tool='update_context_batch')
        repos = await ensure_repositories()
        scope = resolve_write_scope(principal)

        # === PHASE 1: Validate all updates before processing ===
        validated_updates: list[dict[str, Any]] = []
        validation_errors: list[tuple[int, str, str]] = []  # (index, context_id, error)

        for idx, update in enumerate(updates):
            validated_update, entry_context_id, entry_validation_error = await validate_update_entry(
                update, idx, repos.context, scope=scope,
            )
            if entry_validation_error is not None:
                validation_errors.append((idx, entry_context_id, entry_validation_error))
                continue
            assert validated_update is not None  # guaranteed by error is None
            validated_updates.append(validated_update)

        # In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at context_id {first_error[1]}: {first_error[2]}',
            )

        # Build results list including validation errors
        results: list[BulkUpdateResultItemDict] = []

        # Add validation errors to results
        for idx, context_id, error in validation_errors:
            results.append(BulkUpdateResultItemDict(
                index=idx,
                context_id=context_id,
                success=False,
                updated_fields=None,
                error=error,
                generation_status=None,
            ))

        if not validated_updates:
            # All updates failed validation
            return BulkUpdateResponseDict(
                success=False,
                total=len(updates),
                succeeded=0,
                failed=len(updates),
                results=results,
                message='All updates failed validation',
            )

        # === PHASE 2: Check all entries exist and authorize visibility changes
        # (fail fast in atomic mode) ===
        entry_sources, entry_versions, existence_errors = await authorize_updates(
            repos, validated_updates, principal, scope=scope, atomic=atomic,
        )

        # In non-atomic mode, add existence errors to results
        if not atomic:
            for original_idx, context_id, error in existence_errors:
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=error,
                    generation_status=None,
                ))

        # Filter out non-existent entries in non-atomic mode by ORIGINAL INDEX, not by
        # context_id: two updates may target the same context_id in one non-atomic batch,
        # and filtering by a context_id set would also drop the sibling that DID exist
        # (dropping it silently, with no result item for its index -- so len(results) <
        # total and a client mapping results by index cannot find it). Mirrors the
        # generation-error filter below and store_context_batch.
        if not atomic and existence_errors:
            missing_indices = {original_idx for original_idx, _, _ in existence_errors}
            validated_updates_filtered = [
                (vu_idx, u) for vu_idx, u in enumerate(validated_updates)
                if u['index'] not in missing_indices
            ]
        else:
            validated_updates_filtered = list(enumerate(validated_updates))

        if not validated_updates_filtered:
            # All entries failed (validation or existence)
            results.sort(key=operator.itemgetter('index'))
            return BulkUpdateResponseDict(
                success=False,
                total=len(updates),
                succeeded=0,
                failed=len(updates),
                results=results,
                message='All updates failed validation or entry not found',
            )

        # === PHASE 3: Generate Embeddings + Summaries per entry (Outside Transaction) ===
        # CRITICAL: Generation happens BEFORE any database modifications.
        # Each entry runs embedding+summary in parallel via asyncio.gather.
        # Entries are processed sequentially to limit provider load.
        # Only entries with text changes need generation.
        update_embeddings: dict[int, list[ChunkEmbedding] | None] = {}
        update_summaries: dict[int, str | None] = {}
        update_clear_summaries: set[int] = set()
        generation_errors: list[tuple[int, str, str]] = []  # (original_idx, context_id, error)

        embedding_provider = get_embedding_provider()
        summary_provider = get_summary_provider()
        min_content_length = settings.summary.min_content_length
        # The text updates whose generation is left to a job: those asking for it while a
        # generation provider is active. Without one there is nothing to defer, and such an
        # update is applied like any other.
        generation_can_defer = generation_active()
        deferred = {
            vu_idx for vu_idx, update in validated_updates_filtered
            if update['defer_generation'] and update.get('text') is not None and generation_can_defer
        }
        embeddings_generated_count = 0
        summaries_generated_count = 0
        summaries_cleared_count = 0

        for vu_idx, update in validated_updates_filtered:
            original_idx = update['index']
            context_id = update['context_id']
            text_content = update.get('text')

            # No text change -- skip generation entirely
            if text_content is None:
                update_embeddings[vu_idx] = None
                update_summaries[vu_idx] = None
                continue

            # A deferring text change generates nothing here; its transaction clears the
            # old text's summary, embeddings and node rows and leaves the new ones to a job.
            if vu_idx in deferred:
                update_embeddings[vu_idx] = None
                update_summaries[vu_idx] = None
                update_clear_summaries.add(vu_idx)
                continue

            tasks_to_run: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
            task_names: list[str] = []

            # Embedding task
            if embedding_provider is not None:
                tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                task_names.append('embedding')

            # Summary task (min_content_length pre-check OUTSIDE wrapper)
            if summary_provider is not None:
                if min_content_length > 0 and len(text_content) < min_content_length:
                    update_summaries[vu_idx] = None
                    update_clear_summaries.add(vu_idx)
                    logger.info(
                        'Skipping summary for context %s at index %d: '
                        'text length %d < min_content_length %d. Existing summary will be cleared.',
                        context_id, original_idx, len(text_content), min_content_length,
                    )
                else:
                    tasks_to_run.append(generate_summary_with_timeout(text_content, entry_sources[context_id]))
                    task_names.append('summary')
            else:
                # No summary provider at update time (summary generation disabled/absent).
                # The stored summary describes the REPLACED text, so CLEAR it instead of
                # leaving a stale summary -- mirroring the too-short branch above, the
                # single-update path (app/tools/context/update.py), and the stale-embedding / index_tree
                # node-row clears that already run on this same text-change path.
                update_summaries[vu_idx] = None
                update_clear_summaries.add(vu_idx)

            if not tasks_to_run:
                update_embeddings.setdefault(vu_idx, None)
                if vu_idx not in update_summaries:
                    update_summaries[vu_idx] = None
                continue

            gather_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            errors: list[tuple[str, BaseException]] = []
            for name, result in zip(task_names, gather_results, strict=True):
                if isinstance(result, BaseException):
                    errors.append((name, result))
                    logger.error(
                        'Generation failed for %s on context %s at index %d (retries exhausted): %s',
                        name, context_id, original_idx, result,
                    )
                elif name == 'embedding':
                    update_embeddings[vu_idx] = cast(list[ChunkEmbedding] | None, result)
                elif name == 'summary':
                    update_summaries[vu_idx] = cast(str | None, result)

            # Summary regeneration that ran but produced nothing (empty -> None)
            # on a text change must CLEAR the now-stale summary (it describes the
            # OLD text), mirroring the too-short branch above and the single-update
            # path. Only when the summary task itself did not error.
            summary_errored = any(name == 'summary' for name, _ in errors)
            if 'summary' in task_names and not summary_errored and update_summaries.get(vu_idx) is None:
                update_clear_summaries.add(vu_idx)

            if errors:
                error_details = '; '.join(
                    f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
                )
                if atomic:
                    raise ToolError(
                        f'Generation failed for context {context_id} at index {original_idx} '
                        f'after exhausting configured retries: {error_details}. No data was modified.',
                    )
                generation_errors.append((original_idx, context_id, f'Generation failed: {error_details}'))
                update_embeddings.setdefault(vu_idx, None)
                update_summaries.setdefault(vu_idx, None)
            else:
                update_embeddings.setdefault(vu_idx, None)
                if vu_idx not in update_summaries:
                    update_summaries[vu_idx] = None

                # Compress regenerated embeddings OUTSIDE the DB transaction
                # (generation-first invariant). No-op when compression is
                # disabled. Atomic mode aborts the batch; non-atomic marks
                # this entry failed and continues.
                try:
                    update_embeddings[vu_idx] = await generate_compression_with_timeout(
                        update_embeddings.get(vu_idx),
                    )
                except ToolError as compress_err:
                    if atomic:
                        raise ToolError(
                            f'Compression failed for context {context_id} at '
                            f'index {original_idx}: {compress_err}. '
                            f'No data was modified.',
                        ) from compress_err
                    generation_errors.append(
                        (original_idx, context_id,
                         f'Compression failed: {compress_err}'),
                    )
                    update_embeddings[vu_idx] = None
                    update_summaries[vu_idx] = None
                else:
                    # Count generated artifacts only AFTER compression succeeds, so a
                    # non-atomic compression failure (which discards this entry below)
                    # cannot inflate the generated counts in the response diagnostic.
                    if update_embeddings.get(vu_idx) is not None:
                        embeddings_generated_count += 1
                    if update_summaries.get(vu_idx) and isinstance(update_summaries[vu_idx], str):
                        summaries_generated_count += 1

        # In non-atomic mode, add generation errors to results and filter
        if not atomic and generation_errors:
            for original_idx, context_id, error in generation_errors:
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=error,
                    generation_status=None,
                ))
            # Filter by ORIGINAL INDEX, not by context_id: two updates may target
            # the same context_id in one non-atomic batch, and filtering by a
            # context_id set would also drop the sibling that did NOT fail
            # (silently losing a successful update). Mirrors store_context_batch.
            failed_indices = {original_idx for original_idx, _, _ in generation_errors}
            validated_updates_final = [
                (vu_idx, u) for vu_idx, u in validated_updates_filtered
                if u['index'] not in failed_indices
            ]
        else:
            validated_updates_final = validated_updates_filtered

        if not validated_updates_final:
            results.sort(key=operator.itemgetter('index'))
            return BulkUpdateResponseDict(
                success=False,
                total=len(updates),
                succeeded=0,
                failed=len(updates),
                results=results,
                message='All updates failed validation, entry not found, or generation',
            )

        # Rebuild index_tree per-node summaries for each update that changes text,
        # in a SEPARATE fenced never-raise pass (isolated from the abort-mandatory
        # generation above). Updates without a text change are absent from the map,
        # so .get() yields None and leaves the node table untouched.
        update_index_nodes: dict[int, list[IndexNodeRow] | None] = {}
        for vu_idx, update in validated_updates_final:
            new_text = update.get('text')
            if vu_idx in deferred:
                update_index_nodes[vu_idx] = []
            elif new_text is not None:
                rebuilt = await generate_index_nodes_with_timeout(new_text)
                # Text changed: a None result must CLEAR the rows describing the old text
                # ([]) because navigate_context attaches stored summaries by heading-slug
                # node_id, so a retained slug would otherwise receive its pre-edit summary.
                # The clear is UNCONDITIONAL (not gated on node_summaries_enabled) so a
                # disable/edit/re-enable cycle cannot resurface stale rows; it covers the
                # provider-removed case (None for lack of a provider) and total degradation,
                # mirroring the single-update path and the provider-independent
                # embedding/summary clears. replace_nodes_for_context pre-checks table
                # existence, so clearing when the table is absent is a safe no-op.
                update_index_nodes[vu_idx] = [] if rebuilt is None else rebuilt

        def discard_generation_counts(vu_idx: int) -> None:
            """Reverse a discarded update's response-counter contributions.

            A transaction-phase failure discards an update AFTER the
            generation phase bumped the response counters, so without
            compensation the response message would claim "embeddings
            regenerated" / "summaries regenerated" for updates that modified
            nothing. The decrement criteria mirror the bump criteria exactly:
            one embeddings bump iff the update holds compressed embeddings,
            one summary bump iff it holds a non-empty regenerated summary
            string.
            """
            nonlocal embeddings_generated_count, summaries_generated_count
            if update_embeddings.get(vu_idx) is not None:
                embeddings_generated_count -= 1
            summary_value = update_summaries.get(vu_idx)
            if isinstance(summary_value, str) and summary_value:
                summaries_generated_count -= 1

        # === PHASE 4: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend
        deferred_committed = False

        if atomic:
            # ATOMIC MODE: All updates in a single transaction with retry
            max_retries = 2

            for attempt in range(max_retries + 1):
                try:
                    results_attempt: list[BulkUpdateResultItemDict] = []
                    cleared_attempt = 0
                    # Running version per id within THIS attempt: a same-id update
                    # later in the batch must present the version the earlier same-id
                    # update bumped to (the CAS is against the in-transaction row).
                    # Reset each attempt because a rolled-back attempt did not commit.
                    live_versions = dict(entry_versions)

                    async with backend.begin_transaction(access_scope=scope) as txn:
                        for idx, (vu_idx, update) in enumerate(validated_updates_final):
                            original_idx = update['index']
                            context_id = update['context_id']

                            # Heartbeat between entries (skip first)
                            if idx > 0:
                                await transaction_heartbeat(txn)

                            update_images = update.get('images')
                            try:
                                outcome = (
                                    await execute_update_in_transaction(
                                        repos, txn,
                                        context_id=context_id,
                                        scope=scope,
                                        text=update.get('text'),
                                        metadata=update.get('metadata'),
                                        metadata_patch=update.get('metadata_patch'),
                                        summary=update_summaries.get(vu_idx),
                                        clear_summary=vu_idx in update_clear_summaries,
                                        visibility=update.get('visibility'),
                                        tags=update.get('tags'),
                                        images=update_images,
                                        validated_images=update_images or [],
                                        chunk_embeddings=update_embeddings.get(vu_idx),
                                        embedding_model=settings.embedding.model,
                                        index_nodes=update_index_nodes.get(vu_idx),
                                        expected_version=live_versions.get(context_id),
                                        defer_generation=vu_idx in deferred,
                                    )
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
                            bumps_version = (
                                update.get('text') is not None
                                or update.get('metadata') is not None
                                or update.get('visibility') is not None
                            )
                            if bumps_version and context_id in live_versions:
                                # update_context_entry bumped version by 1; a later
                                # same-id update in this batch must see the new value.
                                # Mirrors the execute_update_in_transaction predicate:
                                # text, metadata, AND visibility all ride the CAS write.
                                live_versions[context_id] += 1

                            results_attempt.append(BulkUpdateResultItemDict(
                                index=original_idx,
                                context_id=context_id,
                                success=True,
                                updated_fields=outcome.updated_fields,
                                error=None,
                                generation_status=outcome.generation_status,
                            ))

                        # COMMIT happens here - all or nothing

                    # Transaction committed successfully
                    results.extend(results_attempt)
                    # Accumulate the authoritative per-entry cleared count from this
                    # committed attempt (a rolled-back/retried attempt contributes nothing).
                    summaries_cleared_count += cleared_attempt
                    deferred_committed = bool(deferred)
                    break

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
                        logger.warning(
                            'Atomic batch update transaction failed, retrying in %.1fs '
                            '(attempt %d/%d): %s',
                            delay, attempt + 1, max_retries, e,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise  # Non-connection error or max retries exceeded
        else:
            # NON-ATOMIC MODE: Each update in its own transaction (with retry).
            # Running version per id, persisting ACROSS entries (processed
            # sequentially): a same-id update later in the batch must see the
            # version the earlier same-id update committed.
            live_versions = dict(entry_versions)
            for vu_idx, update in validated_updates_final:
                original_idx = update['index']
                context_id = update['context_id']
                max_retries = 2
                attempt = 0
                version_conflicts = 0
                max_version_conflicts = 5

                while True:
                    try:
                        async with backend.begin_transaction(access_scope=scope) as txn:
                            update_images = update.get('images')
                            outcome = (
                                await execute_update_in_transaction(
                                    repos, txn,
                                    context_id=context_id,
                                    scope=scope,
                                    text=update.get('text'),
                                    metadata=update.get('metadata'),
                                    metadata_patch=update.get('metadata_patch'),
                                    summary=update_summaries.get(vu_idx),
                                    clear_summary=vu_idx in update_clear_summaries,
                                    visibility=update.get('visibility'),
                                    tags=update.get('tags'),
                                    images=update_images,
                                    validated_images=update_images or [],
                                    chunk_embeddings=update_embeddings.get(vu_idx),
                                    embedding_model=settings.embedding.model,
                                    index_nodes=update_index_nodes.get(vu_idx),
                                    expected_version=live_versions.get(context_id),
                                    defer_generation=vu_idx in deferred,
                                )
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
                            summaries_cleared_count += 1
                        if vu_idx in deferred:
                            deferred_committed = True
                        bumps_version = (
                            update.get('text') is not None
                            or update.get('metadata') is not None
                            or update.get('visibility') is not None
                        )
                        if bumps_version and context_id in live_versions:
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
                            discard_generation_counts(vu_idx)
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
                            discard_generation_counts(vu_idx)
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
                            discard_generation_counts(vu_idx)
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
                        discard_generation_counts(vu_idx)
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
                        discard_generation_counts(vu_idx)
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
                        discard_generation_counts(vu_idx)
                        break

        # One wake-up covers every job the batch enqueued.
        if deferred_committed:
            notify_generation_worker()

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(updates) - succeeded

        logger.info(f'Batch update completed: {succeeded}/{len(updates)} succeeded')

        # Both modes now accumulate summaries_cleared_count inline from the authoritative
        # per-entry summary_cleared returned by execute_update_in_transaction (atomic in the
        # committed-attempt loop, non-atomic per entry). The earlier post-hoc cross-product
        # over results x validated_updates_final matched by context_id only and miscounted
        # when two updates targeted the same context_id.

        message = build_batch_update_response_message(
            succeeded=succeeded,
            total=len(updates),
            embeddings_generated_count=embeddings_generated_count,
            summaries_generated_count=summaries_generated_count,
            summaries_cleared_count=summaries_cleared_count,
            generation_pending_count=sum(1 for r in results if r['generation_status'] == 'pending'),
            generation_failed_count=sum(1 for r in results if r['generation_status'] == 'failed'),
        )

        return BulkUpdateResponseDict(
            success=failed == 0,
            total=len(updates),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=message,
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch update: {e}')
        raise ToolError(f'Batch update failed: {format_exception_message(e)}') from e
