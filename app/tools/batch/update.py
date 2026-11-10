"""The update_context_batch tool: update many context entries in one call."""

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
from app.tools._responses import build_batch_update_response_message
from app.tools.batch._update_commit import commit_updates_atomically
from app.tools.batch._update_commit import commit_updates_individually
from app.tools.batch._update_generation import generate_update_index_nodes
from app.tools.batch._update_generation import generate_update_outputs
from app.tools.batch.entry_validation import validate_update_entry
from app.tools.batch.update_access import authorize_updates
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
        # Only entries with text changes need generation.
        generation = await generate_update_outputs(
            validated_updates_filtered, entry_sources,
            embedding_provider=get_embedding_provider(),
            summary_provider=get_summary_provider(),
            min_content_length=settings.summary.min_content_length,
            atomic=atomic,
        )

        # In non-atomic mode, add generation errors to results and filter
        if not atomic and generation.errors:
            for original_idx, context_id, error in generation.errors:
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
            failed_indices = {original_idx for original_idx, _, _ in generation.errors}
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

        # The node summaries of the surviving text changes, rebuilt once before the
        # transaction pass.
        await generate_update_index_nodes(generation, validated_updates_final)

        # === PHASE 4: Transactions for ALL Database Operations ===
        commit_updates = commit_updates_atomically if atomic else commit_updates_individually
        commit = await commit_updates(
            repos, validated_updates_final, generation, entry_versions,
            scope=scope,
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
        failed = len(updates) - succeeded

        logger.info(f'Batch update completed: {succeeded}/{len(updates)} succeeded')

        # Both modes count cleared summaries from the summary_cleared each committed
        # transaction reports, so two updates of one context_id count separately.
        message = build_batch_update_response_message(
            succeeded=succeeded,
            total=len(updates),
            embeddings_generated_count=generation.embeddings_generated,
            summaries_generated_count=generation.summaries_generated,
            summaries_cleared_count=commit.summaries_cleared,
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
