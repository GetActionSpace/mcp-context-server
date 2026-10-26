"""The database writes of one update, run inside the update's open transaction.

Writes the text, metadata, summary and visibility, the metadata patch, the tags and the
images under the caller's access, keeps content_type and updated_at current, replaces or
removes the embeddings and node rows of a changed text, replaces the entry's generation
job, and reads the generation state the update reports. A write that finds the entry gone
or no longer writable ends the update as not found.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import asyncpg

from app.access_scope import AccessScope
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools._transactions import EntryNotFoundError
from app.tools._transactions import transaction_heartbeat
from app.tools._transactions import write_generation_outputs
from app.types import GenerationStatus

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer


@dataclass(frozen=True, slots=True)
class UpdateOutcome:
    """What the transaction of one update wrote.

    Attributes:
        updated_fields: The names of the fields the update changed.
        summary_cleared: Whether the update cleared the entry's summary.
        generation_status: The entry's generation state as the transaction leaves it.
    """

    updated_fields: list[str]
    summary_cleared: bool
    generation_status: GenerationStatus


async def execute_update_in_transaction(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    *,
    context_id: str,
    scope: AccessScope,
    text: str | None,
    metadata: dict[str, Any] | None,
    metadata_patch: dict[str, Any] | None,
    summary: str | None,
    clear_summary: bool,
    visibility: str | None = None,
    tags: list[str] | None,
    images: list[dict[str, str]] | None,
    validated_images: list[dict[str, str]],
    chunk_embeddings: list[ChunkEmbedding] | None,
    embedding_model: str,
    index_nodes: list[IndexNodeRow] | None = None,
    expected_version: int | None = None,
    defer_generation: bool,
) -> UpdateOutcome:
    """Execute all update operations within an existing transaction.

    Performs a complete update of a single context entry as the caller. Every
    statement on the entry row re-asserts the caller's access: owner-only when
    the visibility changes, owner or write grant otherwise, so access lost
    between the caller's pre-generation probe and this transaction ends the
    update as not found:
    1. Update text/metadata/summary/visibility via update_context_entry (CHECK success)
    2. Apply metadata_patch via patch_metadata (CHECK success)
    3. Replace tags if provided
    4. Replace images if provided (update content_type accordingly)
    5. Maintain the auto-managed fields: recompute content_type from actual image
       presence, and guarantee updated_at advanced for this update (a tags-only
       change writes no context_entries row of its own)
    6. Delete old + store new embeddings if text changed
    7. Replace the entry's generation job if text changed: removed when the caller
       generated the outputs, replaced by a new job when it defers generation
    8. Read the entry's generation state for the response

    Args:
        repos: Repository container.
        txn: Active transaction context.
        context_id: ID (32-char canonical hex) of entry to update.
        scope: The caller's scope, passed to every statement on the entry row.
        text: New text content or None.
        metadata: Full metadata replacement or None.
        metadata_patch: Metadata merge patch or None.
        summary: New summary or None.
        clear_summary: Whether to clear existing summary.
        visibility: New visibility value or None. The caller validates the
            value and checks ownership BEFORE the transaction; here it rides
            the update_context_entry write, which admits only an entry the
            scope owns and participates in the same compare-and-set as
            text/metadata.
        tags: New tags or None.
        images: Raw images parameter from caller (for None vs empty detection).
        validated_images: Validated image list (empty if images is None).
        chunk_embeddings: Regenerated embeddings or None.
        embedding_model: Model name for embedding storage.
        index_nodes: Replacement index_tree node rows; None leaves the stored
            rows untouched, an empty list clears them.
        expected_version: Optimistic-concurrency token captured before
            generation; None skips the compare-and-set.
        defer_generation: True when the caller generated nothing for the new text and
            leaves its outputs to the generation worker: the entry's job is replaced by
            a new one, claimable after every job enqueued before it. The caller then
            clears the summary and the node rows and passes no embeddings, so the stale
            outputs go with the old text. Ignored when text is None. The caller passes
            True only while a generation provider is active.

    Returns:
        The names of the fields the update changed, whether it cleared the summary, and
        the entry's generation state.

    Raises:
        EntryNotFoundError: If the target entry does not exist or the scope may no
            longer modify it -- update_context_entry or patch_metadata reports no
            matching row, the tags-only / images-only path finds the parent missing,
            or the entry's content type cannot be read. A ControlFlowError, so the
            failed write is not charged to the circuit breaker; the caller catches it
            outside the transaction and converts it to a not-found ToolError.
    """
    updated_fields: list[str] = []

    # ``updated_at`` is an auto-managed PUBLIC field: get_context_by_ids and every
    # search tool return it, and it is the only mutation timestamp the API exposes,
    # so clients key incremental sync and cache invalidation on it. It is stamped
    # ONLY by a write to context_entries itself (update_context_entry,
    # patch_metadata, update_content_type) -- a branch that touches just a child
    # table leaves it stale. Track whether such a write happened so the auto-managed
    # block below can stamp it exactly once for every update variant.
    entry_row_stamped = False

    # Update text content, metadata (full replacement), and/or visibility if provided
    if text is not None or metadata is not None or visibility is not None:
        metadata_str: str | None = None
        if metadata is not None:
            metadata_str = json.dumps(metadata, ensure_ascii=False)

        success, fields = await repos.context.update_context_entry(
            context_id=context_id,
            text_content=text,
            metadata=metadata_str,
            summary=summary,
            clear_summary=clear_summary,
            visibility=visibility,
            expected_version=expected_version,
            scope=scope,
            txn=txn,
        )

        if not success:
            # update_context_entry returns success=False only when no row the scope
            # may modify matched its id (a version mismatch is raised, not
            # returned), i.e. the entry was deleted concurrently, the caller lost
            # access to it, or the id is stale.
            raise EntryNotFoundError(context_id)

        updated_fields.extend(fields)
        entry_row_stamped = True

    # Apply metadata patch (partial update) if provided
    if metadata_patch is not None:
        success, fields = await repos.context.patch_metadata(
            context_id=context_id,
            patch=metadata_patch,
            scope=scope,
            txn=txn,
        )

        if not success:
            # patch_metadata returns success=False only when the row is gone or the
            # scope may no longer modify it.
            raise EntryNotFoundError(context_id)

        updated_fields.extend(fields)
        entry_row_stamped = True

    # A tags-only or images-only update issues no write that first confirms the
    # parent exists and admits the caller: the text/metadata and metadata_patch
    # branches each SELECT the row under the caller's access predicate (and raise
    # EntryNotFoundError above when it is gone), but neither ran here. Without that
    # guard, replacing tags/images against a deleted parent violates the child
    # foreign key -- a non-ControlFlowError that charges the circuit breaker --
    # or, with FK enforcement off, orphans the replacement rows, and the child
    # writes would reach an entry the caller may no longer modify. Confirm the
    # parent under the write predicate and raise the same breaker-exempt signal.
    if (
        text is None
        and metadata is None
        and metadata_patch is None
        and visibility is None
        and (tags is not None or images is not None)
        and not await repos.context.entry_exists(context_id, scope=scope, txn=txn)
    ):
        raise EntryNotFoundError(context_id)

    # Heartbeat between operation groups
    await transaction_heartbeat(txn)

    # Replace tags and/or images if provided. Both write child rows keyed on the
    # parent context_entries id. The entry_exists guard above (locking the parent
    # with FOR KEY SHARE on PostgreSQL) already confirms and holds the parent for
    # the tags-only / images-only path, so a concurrent delete cannot race these
    # writes. This catch is defense in depth for any residual foreign-key
    # violation, mapping it to the breaker-exempt EntryNotFoundError so a missing
    # parent surfaces as a clean not-found outcome rather than a raw asyncpg error
    # that charges the circuit breaker.
    try:
        # Replace tags if provided
        if tags is not None:
            await repos.tags.replace_tags_for_context(context_id, tags, txn=txn, scope=scope)
            updated_fields.append('tags')

        # Replace images if provided
        if images is not None:
            if len(images) == 0:
                await repos.images.replace_images_for_context(context_id, [], txn=txn, scope=scope)
                await repos.context.update_content_type(context_id, 'text', scope=scope, txn=txn)
                updated_fields.extend(['images', 'content_type'])
            else:
                await repos.images.replace_images_for_context(
                    context_id, validated_images, txn=txn, scope=scope,
                )
                await repos.context.update_content_type(
                    context_id, 'multimodal', scope=scope, txn=txn,
                )
                updated_fields.extend(['images', 'content_type'])
            # update_content_type writes context_entries, so it carried the stamp.
            entry_row_stamped = True
    except asyncpg.exceptions.ForeignKeyViolationError as exc:
        raise EntryNotFoundError(context_id) from exc

    # Enforce the two auto-managed fields centrally, for EVERY update variant that
    # changed something, instead of relying on whichever data write a branch happens
    # to issue:
    #   * content_type is recomputed from the entry's ACTUAL image rows (the explicit
    #     images branch above already wrote the matching value, so this only runs when
    #     the caller left images untouched);
    #   * updated_at is advanced whenever no branch has written context_entries yet --
    #     the tags-only variant writes only the child `tags` table, so without this it
    #     would report success while leaving the entry's public mutation timestamp at
    #     its previous value, and a client syncing or invalidating caches on
    #     updated_at would never observe the change.
    # The recomputed content_type is derived from the image rows inside this same
    # transaction, so it is correct by construction rather than a read-modify-write
    # of a field a concurrent writer could have moved. When it is already correct,
    # the timestamp is stamped EXPLICITLY (touch_updated_at) instead of rewriting an
    # unrelated column back to its own value just to carry the stamp along.
    if images is None and updated_fields:
        image_count = await repos.images.count_images_for_context(context_id, txn=txn, scope=scope)
        current_content_type = 'multimodal' if image_count > 0 else 'text'
        stored_content_type = await repos.context.get_content_type(context_id, scope=scope, txn=txn)
        if stored_content_type is None:
            # The row is gone or the scope may no longer modify it. Comparing None
            # with the recomputed type would read as a difference and issue a
            # content-type write, so the update ends as not found instead.
            raise EntryNotFoundError(context_id)
        if stored_content_type != current_content_type:
            await repos.context.update_content_type(
                context_id, current_content_type, scope=scope, txn=txn,
            )
            entry_row_stamped = True
            updated_fields.append('content_type')
        elif not entry_row_stamped:
            await repos.context.touch_updated_at(context_id, scope=scope, txn=txn)
            entry_row_stamped = True

    # Embeddings describe text_content, so a text change invalidates the stored vectors.
    if (
        chunk_embeddings is None
        and text is not None
        and await repos.embeddings.embedding_tables_exist(txn=txn, scope=scope)
    ):
        # Text changed but embeddings were NOT regenerated (no embedding provider at
        # update time -- generation disabled/absent). The stored chunks describe the
        # REPLACED text, so DELETE them rather than leave stale vectors that semantic
        # search would match against the old content. Guarded by embedding_tables_exist
        # so a database that never provisioned embeddings is a safe no-op. (Mirrors the
        # stale-summary clear on this same text-change path.)
        await transaction_heartbeat(txn)
        if await repos.embeddings.delete_all_chunks(context_id, txn=txn, scope=scope):
            updated_fields.append('embedding')

    # New embeddings replace the old chunks. Index_tree node summaries are replaced
    # atomically: None leaves the node table untouched (feature off, or text unchanged
    # so the caller did not recompute); an empty list clears stale rows when text
    # shrank below the summary thresholds.
    if await write_generation_outputs(
        repos, txn, context_id, chunk_embeddings, index_nodes, embedding_model=embedding_model, scope=scope,
    ):
        updated_fields.append('embedding')

    # A job belongs to the text it was enqueued for. New text either brought its outputs
    # with it, which retires the job, or leaves them to a new job at the end of the queue.
    if text is not None:
        await repos.generation_jobs.delete_for_entry(context_id, txn=txn)
        if defer_generation:
            await repos.generation_jobs.enqueue(context_id, txn=txn)

    return UpdateOutcome(
        updated_fields=updated_fields,
        summary_cleared=clear_summary,
        generation_status=await repos.generation_jobs.state_for(context_id, txn=txn),
    )
