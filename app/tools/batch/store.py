"""The store_context_batch tool: store many context entries in one call."""

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
from app.tools._generation import node_layer_active
from app.tools._responses import build_batch_store_response_message
from app.tools._transactions import EmbeddingsReconcileRequiredError
from app.tools._transactions import execute_store_in_transaction
from app.tools._transactions import is_connection_error
from app.tools._transactions import transaction_heartbeat
from app.tools.batch.entry_validation import validate_store_entry
from app.types import BulkStoreResponseDict
from app.types import BulkStoreResultItemDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def store_context_batch(
    entries: Annotated[
        list[dict[str, Any]],
        Field(
            description='List of context entries to store. Each entry must have: '
            'thread_id (str), source ("user" or "agent"), text (str). '
            'Optional: metadata (dict), tags (list[str]), images (list[dict]), '
            'visibility ("private" or "public"; omitted uses the server default, '
            'publishing as public may require a configured role), '
            'deduplicate (bool, default true; false always inserts a new entry owned by the caller), '
            'defer_generation (bool, default false; true stores the entry at once and leaves its '
            'embeddings, summary and section summaries to a background job).',
            min_length=1,
            max_length=100,
        ),
    ],
    atomic: Annotated[
        bool,
        Field(
            description='If true, ALL entries must succeed or NONE are stored (transaction rollback). '
            'If false, partial success is allowed with per-item error reporting.',
        ),
    ] = True,
) -> BulkStoreResponseDict:
    """Store multiple context entries in a batch.

    Batch processing is significantly faster than individual calls when storing many entries.
    Use for migrations, imports, or bulk operations.

    Atomicity modes:
    - atomic=True (default): ALL entries must succeed or NONE are stored (transaction rollback).
    - atomic=False: Each entry processed independently with per-item error reporting.

    Deduplication: the candidate is the latest entry of the same thread_id and source
    among the entries the caller may read; when the caller owns it and its text is
    identical, the existing entry is updated, otherwise a new entry owned by the caller
    is inserted. Deduplication is suppressed when opposite-source entries the caller
    may read exist after the candidate, preserving chronological ordering.
    Pre-check optimization skips embedding/summary generation for likely duplicates.
    An entry with deduplicate=False skips the candidate and always inserts a new entry
    owned by the caller; a later deduplicating entry of the same batch with identical
    text merges into that new entry, as it is then the latest one.

    Generation: an entry's embeddings, summary and section summaries are generated before
    it is stored, and a generation failure stores nothing of it. An entry with
    defer_generation=True skips the pre-check and generation: it is stored at once with a
    background job that generates its outputs, readable and full-text searchable
    immediately, and found by semantic search once the job has written its embeddings.
    Each stored item's generation_status is 'ready' when the entry's generated outputs are
    complete, 'pending' while its generation job waits or runs, and 'failed' once the job
    exhausted its retries; an item that was not stored reports null.

    Size limits:
    - Maximum 100 entries per batch
    - Image limits per entry: 10MB each, 100MB total
    - Tag limits per entry: 100 tags, each at most 128 characters
    - thread_id: at most 256 characters
    - Value under an indexed metadata field: at most 512 characters
    - Standard tag normalization (lowercase)

    Returns:
        BulkStoreResponseDict with success (bool), total (int), succeeded (int),
        failed (int), results (list of index, success, context_id, error,
        generation_status), message (str).

    Raises:
        ToolError: If validation fails, embedding generation fails (atomic), or batch operation fails.
    """
    try:
        # Resolve the effective principal ONCE for the whole batch (one request,
        # one caller identity) and refuse the read-only principal before any entry
        # is validated or the repositories are reached; then the scope every entry
        # deduplicates and stamps under, and the author-group grant policy. The
        # publish gate is enforced per entry on each entry's EFFECTIVE visibility.
        principal = resolve_effective_principal()
        reject_read_only_principal(principal, tool='store_context_batch')
        repos = await ensure_repositories()
        scope = resolve_write_scope(principal)
        author_group_grants: frozenset[str] = (
            principal.groups
            if settings.access_control.default_group_grants == 'author_groups'
            else frozenset()
        )

        # === PHASE 1: Validate all entries before processing ===
        validated_entries: list[dict[str, Any]] = []
        validation_errors: list[tuple[int, str]] = []

        for idx, entry in enumerate(entries):
            validated_entry, entry_validation_error = validate_store_entry(entry, idx, principal)
            if entry_validation_error is not None:
                validation_errors.append((idx, entry_validation_error))
                continue
            assert validated_entry is not None  # guaranteed by error is None
            validated_entries.append(validated_entry)

        # In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at index {first_error[0]}: {first_error[1]}',
            )

        # Build results list including validation errors
        results: list[BulkStoreResultItemDict] = []

        # Add validation errors to results
        for idx, error in validation_errors:
            results.append(BulkStoreResultItemDict(
                index=idx,
                success=False,
                context_id=None,
                error=error,
                generation_status=None,
            ))

        if not validated_entries:
            # All entries failed validation
            return BulkStoreResponseDict(
                success=False,
                total=len(entries),
                succeeded=0,
                failed=len(entries),
                results=results,
                message='All entries failed validation',
            )

        # === PHASE 2: Generate Embeddings + Summaries per entry (Outside Transaction) ===
        # CRITICAL: Generation happens BEFORE any database modifications.
        # Each entry runs embedding+summary in parallel via asyncio.gather.
        # Entries are processed sequentially to limit provider load.
        entry_embeddings: dict[int, list[ChunkEmbedding] | None] = {}
        entry_summaries: dict[int, str | None] = {}
        # Per-entry dedup pre-check decision, captured here so the index_tree node
        # pass below gates symmetrically with embeddings/summary.
        entry_likely_duplicate: dict[int, str | None] = {}
        generation_errors: list[tuple[int, str]] = []  # (original_idx, error_message)

        embedding_provider = get_embedding_provider()
        summary_provider = get_summary_provider()
        min_content_length = settings.summary.min_content_length
        # The entries whose generation is left to a job: those asking for it while a
        # generation provider is active. Without one there is nothing to defer, and such
        # an entry is stored like any other.
        generation_can_defer = generation_active()
        deferred = {
            ve_idx for ve_idx, entry in enumerate(validated_entries)
            if entry['defer_generation'] and generation_can_defer
        }
        embeddings_generated_count = 0
        embeddings_stored_count = 0
        summaries_generated_count = 0
        summaries_preserved_count = 0
        # ve_idx values whose summary was reused from a likely duplicate (pre-check),
        # so a non-atomic compression failure that discards the entry can decrement the
        # preserved count it provisionally bumped -- mirroring the generated counts,
        # which are now bumped only after compression succeeds.
        preserved_summary_indices: set[int] = set()

        for ve_idx, entry in enumerate(validated_entries):
            original_idx = entry['index']
            text_content = entry['text_content']
            if ve_idx in deferred:
                entry_embeddings[ve_idx] = None
                entry_summaries[ve_idx] = None
                entry_likely_duplicate[ve_idx] = None
                continue

            tasks_to_run: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
            task_names: list[str] = []

            # Performance optimization: pre-check for likely duplicates (read-only).
            # The candidate id and its stored summary come from ONE
            # statement-level snapshot (see DuplicateCandidate): a separate
            # later summary read could observe a row version a concurrent
            # update committed in between, pairing a reused summary with text
            # it does not describe. An entry that does not deduplicate always
            # inserts, so it skips the pre-check and every leg runs for it.
            likely_duplicate_id: str | None = None
            duplicate_summary: str | None = None
            if entry['deduplicate'] and (embedding_provider is not None or summary_provider is not None):
                duplicate_candidate = await repos.context.check_latest_is_duplicate(
                    thread_id=entry['thread_id'],
                    source=entry['source'],
                    text_content=text_content,
                    scope=scope,
                )
                if duplicate_candidate is not None:
                    likely_duplicate_id = duplicate_candidate.context_id
                    duplicate_summary = duplicate_candidate.summary
            entry_likely_duplicate[ve_idx] = likely_duplicate_id

            # Embedding task (with pre-check optimization)
            if embedding_provider is not None:
                if likely_duplicate_id is not None:
                    has_embeddings = await repos.embeddings.exists(likely_duplicate_id, scope=scope)
                    if has_embeddings:
                        logger.debug(
                            'Pre-check: skipping embedding generation for likely duplicate '
                            'of context %s at index %d',
                            likely_duplicate_id, original_idx,
                        )
                    else:
                        tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                        task_names.append('embedding')
                else:
                    tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                    task_names.append('embedding')

            # Summary task (with pre-check optimization)
            if summary_provider is not None:
                if min_content_length > 0 and len(text_content) < min_content_length:
                    logger.info(
                        'Skipping summary generation at index %d: '
                        'text length %d < min_content_length %d',
                        original_idx, len(text_content), min_content_length,
                    )
                    entry_summaries[ve_idx] = None
                elif likely_duplicate_id is not None:
                    if duplicate_summary is not None:
                        entry_summaries[ve_idx] = duplicate_summary
                        summaries_preserved_count += 1
                        preserved_summary_indices.add(ve_idx)
                        logger.debug(
                            'Pre-check: reusing existing summary for likely duplicate '
                            'of context %s at index %d',
                            likely_duplicate_id, original_idx,
                        )
                    else:
                        tasks_to_run.append(generate_summary_with_timeout(text_content, entry['source']))
                        task_names.append('summary')
                else:
                    tasks_to_run.append(generate_summary_with_timeout(text_content, entry['source']))
                    task_names.append('summary')

            if not tasks_to_run:
                entry_embeddings[ve_idx] = None
                if ve_idx not in entry_summaries:
                    entry_summaries[ve_idx] = None
                continue

            gather_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            errors: list[tuple[str, BaseException]] = []
            for name, result in zip(task_names, gather_results, strict=True):
                if isinstance(result, BaseException):
                    errors.append((name, result))
                    logger.error(
                        'Generation failed for %s at index %d (retries exhausted): %s',
                        name, original_idx, result,
                    )
                elif name == 'embedding':
                    entry_embeddings[ve_idx] = cast(list[ChunkEmbedding] | None, result)
                elif name == 'summary':
                    entry_summaries[ve_idx] = cast(str | None, result)

            if errors:
                error_details = '; '.join(
                    f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
                )
                if atomic:
                    raise ToolError(
                        f'Generation failed at index {original_idx} after exhausting '
                        f'configured retries: {error_details}. No data was saved.',
                    )
                generation_errors.append((original_idx, f'Generation failed: {error_details}'))
                entry_embeddings.setdefault(ve_idx, None)
                entry_summaries.setdefault(ve_idx, None)
                # This entry is discarded below; undo the preserved-summary count it
                # provisionally bumped in the pre-check, mirroring the
                # compression-failure compensation -- counts must reflect entries
                # surviving the generation phase.
                if ve_idx in preserved_summary_indices:
                    preserved_summary_indices.discard(ve_idx)
                    summaries_preserved_count -= 1
            else:
                entry_embeddings.setdefault(ve_idx, None)
                if ve_idx not in entry_summaries:
                    entry_summaries[ve_idx] = None

                # Compress embeddings OUTSIDE the DB transaction (mirrors
                # the generation-first invariant). No-op when
                # ENABLE_EMBEDDING_COMPRESSION=false. In atomic mode a
                # failure aborts the whole batch; in non-atomic mode this
                # entry is marked failed and processing continues.
                try:
                    entry_embeddings[ve_idx] = await generate_compression_with_timeout(
                        entry_embeddings.get(ve_idx),
                    )
                except ToolError as compress_err:
                    if atomic:
                        raise ToolError(
                            f'Compression failed at index {original_idx}: '
                            f'{compress_err}. No data was saved.',
                        ) from compress_err
                    generation_errors.append(
                        (original_idx, f'Compression failed: {compress_err}'),
                    )
                    entry_embeddings[ve_idx] = None
                    entry_summaries[ve_idx] = None
                    # This entry is discarded below; undo the preserved-summary count it
                    # provisionally bumped in the pre-check, mirroring the generated counts
                    # which are bumped only on compression success. Discard the index too
                    # so the set stays consistent with the count, like every sibling
                    # compensation site.
                    if ve_idx in preserved_summary_indices:
                        preserved_summary_indices.discard(ve_idx)
                        summaries_preserved_count -= 1
                else:
                    # Count generated artifacts only AFTER compression succeeds, so a
                    # non-atomic compression failure (which discards this entry below)
                    # cannot inflate the generated counts in the response diagnostic.
                    if entry_embeddings.get(ve_idx) is not None:
                        embeddings_generated_count += 1
                    # Only count a summary as GENERATED when a summary model call
                    # actually ran for this entry; a reused/preserved summary (no
                    # 'summary' task queued, accounted by summaries_preserved_count)
                    # must not be double-counted here, mirroring the single-store
                    # generated-vs-preserved split.
                    if (
                        'summary' in task_names
                        and entry_summaries.get(ve_idx)
                        and isinstance(entry_summaries[ve_idx], str)
                    ):
                        summaries_generated_count += 1

        # In non-atomic mode, add generation errors to results and filter
        if not atomic and generation_errors:
            for original_idx, error in generation_errors:
                results.append(BulkStoreResultItemDict(
                    index=original_idx,
                    success=False,
                    context_id=None,
                    error=error,
                    generation_status=None,
                ))
            failed_indices = {idx for idx, _ in generation_errors}
            validated_entries_filtered = [
                (ve_idx, e) for ve_idx, e in enumerate(validated_entries)
                if e['index'] not in failed_indices
            ]
        else:
            validated_entries_filtered = list(enumerate(validated_entries))

        if not validated_entries_filtered:
            results.sort(key=operator.itemgetter('index'))
            return BulkStoreResponseDict(
                success=False,
                total=len(entries),
                succeeded=0,
                failed=len(entries),
                results=results,
                message='All entries failed validation or generation',
            )

        # Build index_tree per-node summaries for each surviving entry in a
        # SEPARATE fenced never-raise pass, isolated from the abort-mandatory
        # embedding/summary/compression above: a node-summary failure never fails
        # an entry. None per entry when the feature is disabled OR the entry is a
        # likely deduplication retransmit (gated symmetrically with
        # embeddings/summary). Computed once and reused across reconcile/connection
        # retries below.
        entry_index_nodes: dict[int, list[IndexNodeRow] | None] = {}
        for ve_idx, entry in validated_entries_filtered:
            # Skip node regeneration for a likely deduplication retransmit (None
            # leaves stored rows untouched); reconcile-divergence INSERTs below
            # regenerate for the diverged text.
            entry_index_nodes[ve_idx] = (
                None
                if entry_likely_duplicate.get(ve_idx) is not None or ve_idx in deferred
                else await generate_index_nodes_with_timeout(entry['text_content'])
            )

        def discard_generation_counts(ve_idx: int) -> None:
            """Reverse a discarded entry's response-counter contributions.

            A transaction-phase failure discards an entry AFTER the generation
            phase bumped the response counters. Without compensation the
            response message computes not_stored = generated - stored and
            labels the whole gap "not stored - duplicates", so a FAILED
            never-stored entry would be reported as a dedup skip, and
            "summaries generated/preserved" would be claimed for entries that
            stored nothing. The decrement criteria mirror the bump criteria
            exactly: one embeddings bump iff the entry holds compressed
            embeddings, one preserved bump iff its index is still in the
            preserved set, else one generated-summary bump iff it holds a
            non-empty generated summary string.
            """
            nonlocal embeddings_generated_count, summaries_generated_count, summaries_preserved_count
            if entry_embeddings.get(ve_idx) is not None:
                embeddings_generated_count -= 1
            if ve_idx in preserved_summary_indices:
                preserved_summary_indices.discard(ve_idx)
                summaries_preserved_count -= 1
            else:
                summary_value = entry_summaries.get(ve_idx)
                if isinstance(summary_value, str) and summary_value:
                    summaries_generated_count -= 1

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend
        deferred_committed = False

        if atomic:
            # ATOMIC MODE: All entries in a single transaction with retry
            max_retries = 2
            attempt = 0
            # A pre-check-skipped embedding can diverge once per distinct entry,
            # so bound reconciliation passes by the number of entries.
            reconcile_passes = 0
            max_reconcile_passes = len(validated_entries_filtered)

            while True:
                try:
                    results_attempt: list[BulkStoreResultItemDict] = []
                    # Count stored embeddings per attempt; only fold into the
                    # outer total once the transaction commits, so connection
                    # retries and reconciliation passes never double-count.
                    stored_count_attempt = 0

                    async with backend.begin_transaction(access_scope=scope) as txn:
                        for idx, (ve_idx, entry) in enumerate(validated_entries_filtered):
                            original_idx = entry['index']

                            # Heartbeat between entries (skip first)
                            if idx > 0:
                                await transaction_heartbeat(txn)

                            outcome = (
                                await execute_store_in_transaction(
                                    repos, txn,
                                    thread_id=entry['thread_id'],
                                    source=entry['source'],
                                    content_type=entry['content_type'],
                                    text_content=entry['text_content'],
                                    scope=scope,
                                    visibility=entry['visibility'],
                                    author_group_grants=author_group_grants,
                                    metadata_str=entry['metadata'],
                                    summary=entry_summaries.get(ve_idx),
                                    tags=entry.get('tags'),
                                    validated_images=entry.get('images', []),
                                    images_provided=bool(entry.get('images_provided')),
                                    deduplicate=entry['deduplicate'],
                                    defer_generation=ve_idx in deferred,
                                    chunk_embeddings=entry_embeddings.get(ve_idx),
                                    embedding_model=settings.embedding.model,
                                    embedding_generation_enabled=embedding_provider is not None,
                                    index_nodes=entry_index_nodes.get(ve_idx),
                                    nodes_pending=(
                                        node_layer_active()
                                        and entry_likely_duplicate.get(ve_idx) is not None
                                    ),
                                    summary_pending=ve_idx in preserved_summary_indices,
                                )
                            )

                            if outcome.embedding_stored:
                                stored_count_attempt += 1

                            results_attempt.append(BulkStoreResultItemDict(
                                index=original_idx,
                                success=True,
                                context_id=outcome.context_id,
                                error=None,
                                generation_status=outcome.generation_status,
                            ))

                        # COMMIT happens here - all or nothing

                    # Transaction committed successfully
                    results.extend(results_attempt)
                    embeddings_stored_count += stored_count_attempt
                    deferred_committed = bool(deferred)
                    break

                except EmbeddingsReconcileRequiredError as reconcile:
                    # The read-only pre-check skipped embedding generation for a
                    # likely duplicate, but the transaction inserted a new entry.
                    # Regenerate OUTSIDE the transaction (generation-first) for the
                    # diverging text -- and any other filtered entry sharing that
                    # exact text whose embeddings were also skipped -- then re-run
                    # the whole atomic transaction.
                    if reconcile_passes >= max_reconcile_passes:
                        raise ToolError(
                            'Failed to reconcile embeddings after deduplication '
                            'divergence (atomic batch store)',
                        ) from None
                    reconcile_passes += 1
                    # Only re-run the embedding provider when at least one diverged
                    # entry actually lacks embeddings. A node-only reconcile (the
                    # nodes_pending gate fired while embeddings are present) must not
                    # re-invoke the provider: that wastes a call and a transient
                    # failure would spuriously abort the whole atomic batch.
                    needs_embeddings = embedding_provider is not None and any(
                        entry_embeddings.get(r_ve_idx) is None
                        and r_entry['text_content'] == reconcile.text_content
                        for r_ve_idx, r_entry in validated_entries_filtered
                    )
                    if needs_embeddings:
                        regenerated = await generate_compression_with_timeout(
                            await generate_embeddings_with_timeout(reconcile.text_content),
                        )
                        for r_ve_idx, r_entry in validated_entries_filtered:
                            if (entry_embeddings.get(r_ve_idx) is None
                                    and r_entry['text_content'] == reconcile.text_content):
                                entry_embeddings[r_ve_idx] = regenerated
                                # Count only embeddings actually produced. With
                                # generation disabled the no-op provider returns
                                # None, so a node-only reconcile must not inflate
                                # the generated count (parity with the single store).
                                if regenerated is not None:
                                    embeddings_generated_count += 1
                    # A REUSED summary was read from the since-diverged candidate
                    # and may describe different text; regenerate it for the
                    # diverged text so the divergence INSERT cannot persist a
                    # mismatched summary. Clearing the preserved marker also
                    # clears the summary_pending reconcile gate on retry. The
                    # diverged text is fixed and the prompt varies only by
                    # source, so ONE call per distinct source is broadcast to
                    # every matching entry -- mirroring the embedding block's
                    # compute-once pattern above instead of issuing redundant,
                    # sequential LLM round-trips for identical (text, source)
                    # pairs on this abort-sensitive path.
                    if summary_provider is not None:
                        regenerated_summaries: dict[str, str | None] = {}
                        for r_ve_idx, r_entry in validated_entries_filtered:
                            if (r_ve_idx in preserved_summary_indices
                                    and r_entry['text_content'] == reconcile.text_content):
                                r_source = r_entry['source']
                                if r_source not in regenerated_summaries:
                                    regenerated_summaries[r_source] = await generate_summary_with_timeout(
                                        reconcile.text_content, r_source,
                                    )
                                entry_summaries[r_ve_idx] = regenerated_summaries[r_source]
                                preserved_summary_indices.discard(r_ve_idx)
                                summaries_preserved_count -= 1
                                if entry_summaries[r_ve_idx] is not None:
                                    summaries_generated_count += 1
                    # Node summaries for the diverged text were gated off as a
                    # likely duplicate; this divergence INSERTs, so regenerate them.
                    # Coerce total degradation (None) to [] so the reconcile gate
                    # clears on retry: the node layer is never-raise and must not
                    # abort the store; a fresh INSERT has no stale node rows.
                    regenerated_nodes = await generate_index_nodes_with_timeout(reconcile.text_content) or []
                    for r_ve_idx, r_entry in validated_entries_filtered:
                        if (entry_index_nodes.get(r_ve_idx) is None
                                and r_entry['text_content'] == reconcile.text_content):
                            entry_index_nodes[r_ve_idx] = regenerated_nodes
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
        else:
            # NON-ATOMIC MODE: Each entry in its own transaction (with retry)
            for ve_idx, entry in validated_entries_filtered:
                original_idx = entry['index']
                max_retries = 2
                attempt = 0
                reconciled = False

                while True:
                    try:
                        async with backend.begin_transaction(access_scope=scope) as txn:
                            outcome = (
                                await execute_store_in_transaction(
                                    repos, txn,
                                    thread_id=entry['thread_id'],
                                    source=entry['source'],
                                    content_type=entry['content_type'],
                                    text_content=entry['text_content'],
                                    scope=scope,
                                    visibility=entry['visibility'],
                                    author_group_grants=author_group_grants,
                                    metadata_str=entry['metadata'],
                                    summary=entry_summaries.get(ve_idx),
                                    tags=entry.get('tags'),
                                    validated_images=entry.get('images', []),
                                    images_provided=bool(entry.get('images_provided')),
                                    deduplicate=entry['deduplicate'],
                                    defer_generation=ve_idx in deferred,
                                    chunk_embeddings=entry_embeddings.get(ve_idx),
                                    embedding_model=settings.embedding.model,
                                    embedding_generation_enabled=embedding_provider is not None,
                                    index_nodes=entry_index_nodes.get(ve_idx),
                                    nodes_pending=(
                                        node_layer_active()
                                        and entry_likely_duplicate.get(ve_idx) is not None
                                    ),
                                    summary_pending=ve_idx in preserved_summary_indices,
                                )
                            )

                            # COMMIT happens here for this entry

                        if outcome.embedding_stored:
                            embeddings_stored_count += 1
                        if ve_idx in deferred:
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
                        # Pre-check skipped embeddings for a likely duplicate, but
                        # this store inserted a new entry. Regenerate OUTSIDE the
                        # transaction (generation-first) and retry this entry once.
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
                            discard_generation_counts(ve_idx)
                            break
                        reconciled = True
                        # Only re-run the embedding provider when this entry's
                        # embeddings were actually skipped; a node-only reconcile
                        # must not re-invoke the provider (wasted call + a transient
                        # failure would spuriously fail an otherwise-good entry).
                        #
                        # This regeneration is abort-mandatory and raises ToolError on
                        # provider failure/timeout. In non-atomic mode that ToolError
                        # must fail ONLY this entry: without a local guard it would
                        # escape the per-entry loop to the function-level handler and
                        # abort the whole batch, discarding sibling results already
                        # collected -- breaking the documented per-entry isolation of
                        # atomic=false. Record a per-entry failure and stop this entry,
                        # mirroring the per-entry ToolError branch below. (The atomic
                        # branch deliberately lets it abort: all-or-nothing.)
                        try:
                            if embedding_provider is not None and entry_embeddings.get(ve_idx) is None:
                                entry_embeddings[ve_idx] = await generate_compression_with_timeout(
                                    await generate_embeddings_with_timeout(reconcile.text_content),
                                )
                                # Count only embeddings actually produced (the no-op
                                # provider returns None when generation is disabled).
                                if entry_embeddings[ve_idx] is not None:
                                    embeddings_generated_count += 1
                        except ToolError as e:
                            logger.error(
                                'Failed to regenerate embeddings for entry at index %d '
                                'after deduplication divergence: %s', original_idx, e,
                            )
                            results.append(BulkStoreResultItemDict(
                                index=original_idx,
                                success=False,
                                context_id=None,
                                error=format_exception_message(e),
                                generation_status=None,
                            ))
                            # This entry is discarded; undo every response-counter
                            # contribution it made -- counts must reflect entries
                            # surviving the generation phase.
                            discard_generation_counts(ve_idx)
                            break
                        # A REUSED summary was read from the since-diverged candidate
                        # and may describe different text; regenerate it for this
                        # entry's text. Same per-entry ToolError isolation as the
                        # embedding regeneration above. Clearing the preserved marker
                        # also clears the summary_pending reconcile gate on retry.
                        if summary_provider is not None and ve_idx in preserved_summary_indices:
                            try:
                                entry_summaries[ve_idx] = await generate_summary_with_timeout(
                                    reconcile.text_content, entry['source'],
                                )
                            except ToolError as e:
                                logger.error(
                                    'Failed to regenerate summary for entry at index %d '
                                    'after deduplication divergence: %s', original_idx, e,
                                )
                                results.append(BulkStoreResultItemDict(
                                    index=original_idx,
                                    success=False,
                                    context_id=None,
                                    error=format_exception_message(e),
                                    generation_status=None,
                                ))
                                # This entry is discarded; undo every response-counter
                                # contribution it made, including embeddings the first
                                # reconcile pass regenerated above (the success path
                                # below reverses only the preserved marker).
                                discard_generation_counts(ve_idx)
                                break
                            preserved_summary_indices.discard(ve_idx)
                            summaries_preserved_count -= 1
                            if entry_summaries[ve_idx] is not None:
                                summaries_generated_count += 1
                        # Node summaries were gated off as a likely duplicate; this
                        # entry actually INSERTs, so regenerate its index_tree nodes.
                        # Coerce total degradation (None) to [] so the reconcile gate
                        # clears on retry: the node layer is never-raise and must not
                        # abort this entry's store; a fresh INSERT has no stale rows.
                        if entry_index_nodes.get(ve_idx) is None:
                            entry_index_nodes[ve_idx] = await generate_index_nodes_with_timeout(
                                reconcile.text_content,
                            ) or []
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
                        discard_generation_counts(ve_idx)
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
                        discard_generation_counts(ve_idx)
                        break

        # One wake-up covers every job the batch enqueued.
        if deferred_committed:
            notify_generation_worker()

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(entries) - succeeded

        logger.info(f'Batch store completed: {succeeded}/{len(entries)} succeeded')

        message = build_batch_store_response_message(
            succeeded=succeeded,
            total=len(entries),
            embeddings_generated_count=embeddings_generated_count,
            embeddings_stored_count=embeddings_stored_count,
            summaries_generated_count=summaries_generated_count,
            summaries_preserved_count=summaries_preserved_count,
            generation_pending_count=sum(1 for r in results if r['generation_status'] == 'pending'),
            generation_failed_count=sum(1 for r in results if r['generation_status'] == 'failed'),
        )

        return BulkStoreResponseDict(
            success=failed == 0,
            total=len(entries),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=message,
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch store: {e}')
        raise ToolError(f'Batch store failed: {format_exception_message(e)}') from e
