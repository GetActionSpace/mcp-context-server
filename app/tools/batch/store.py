"""The store_context_batch tool: store many context entries in one call."""

import logging
import operator
from typing import Annotated
from typing import Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_effective_principal
from app.auth import resolve_write_scope
from app.auth.access import reject_read_only_principal
from app.errors import format_exception_message
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.startup import notify_generation_worker
from app.tools._responses import build_batch_store_response_message
from app.tools.batch._store_commit import commit_stores_atomically
from app.tools.batch._store_commit import commit_stores_individually
from app.tools.batch._store_generation import generate_store_index_nodes
from app.tools.batch._store_generation import generate_store_outputs
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
    is inserted. In such an update metadata, tags and images are replaced when the entry
    gives them and kept when it does not, and its visibility and owner never change.
    Deduplication is suppressed when opposite-source entries the caller
    may read exist after the candidate, preserving chronological ordering.
    Pre-check optimization skips embedding/summary generation for likely duplicates.
    An entry with deduplicate=False skips the candidate and always inserts a new entry
    owned by the caller; a later deduplicating entry of the same batch with identical
    text merges into that new entry, as it is then the latest one.

    Generation: an entry's embeddings, summary and section summaries are generated before
    it is stored; an embedding or summary failure stores nothing of it, while a section
    summary that fails is left out and the entry is stored without it. An entry with
    defer_generation=True skips the pre-check and generation: it is stored at once with a
    background job that generates its outputs, readable and full-text searchable
    immediately, and found by semantic search once the job has written its embeddings;
    one that deduplicates into an existing entry keeps a waiting generation job, restarts
    a failed one, and adds one only when an output is missing.
    Each stored item's generation_status is 'ready' when the entry's generated outputs are
    complete, 'pending' while its generation job waits or runs, and 'failed' once the job
    exhausted its retries; an item that was not stored reports null.

    Size limits:
    - Maximum 100 entries per batch
    - Image limits per entry: at most 10 images; by default 10MB each and 100MB in total
      (the operator may set other sizes)
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
        embedding_provider = get_embedding_provider()
        summary_provider = get_summary_provider()
        generation = await generate_store_outputs(
            repos, validated_entries,
            scope=scope,
            embedding_provider=embedding_provider,
            summary_provider=summary_provider,
            min_content_length=settings.summary.min_content_length,
            atomic=atomic,
        )

        # In non-atomic mode, add generation errors to results and filter
        if not atomic and generation.errors:
            for original_idx, error in generation.errors:
                results.append(BulkStoreResultItemDict(
                    index=original_idx,
                    success=False,
                    context_id=None,
                    error=error,
                    generation_status=None,
                ))
            failed_indices = {idx for idx, _ in generation.errors}
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

        # The node summaries of the surviving entries, computed once and reused across
        # the reconcile and connection retries of the transaction pass.
        await generate_store_index_nodes(generation, validated_entries_filtered)

        # === PHASE 3: Transactions for ALL Database Operations ===
        commit_entries = commit_stores_atomically if atomic else commit_stores_individually
        commit = await commit_entries(
            repos, validated_entries_filtered, generation,
            scope=scope,
            author_group_grants=author_group_grants,
            embedding_provider=embedding_provider,
            summary_provider=summary_provider,
            embedding_model=settings.embedding.model,
        )
        results.extend(commit.results)

        # One wake-up covers every job the batch enqueued.
        if commit.deferred_committed:
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
            embeddings_generated_count=generation.embeddings_generated,
            embeddings_stored_count=commit.embeddings_stored,
            summaries_generated_count=generation.summaries_generated,
            summaries_preserved_count=generation.summaries_preserved,
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
