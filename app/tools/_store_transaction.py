"""The database writes of one store, run inside the store's open transaction.

Stores the entry under the caller's scope, deduplicating into the caller's identical latest
entry or inserting a new one, then its author-group read grants, tags, images, embeddings
and node rows; leaves the generation job a store that defers generation needs; and reads
the generation state the store reports. A new entry whose generation legs the caller's
deduplication pre-check skipped or reused aborts the transaction for a regeneration.
"""

import logging
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.startup import get_embedding_provider
from app.tools._generation import summary_applies
from app.tools._transactions import EmbeddingsReconcileRequiredError
from app.tools._transactions import transaction_heartbeat
from app.types import GenerationStatus

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StoreOutcome:
    """What the transaction of one store wrote.

    Attributes:
        context_id: The id of the stored entry.
        was_updated: True when deduplication updated an existing entry, False when a new
            entry was inserted.
        embedding_stored: Whether embeddings were written.
        generation_status: The entry's generation state as the transaction leaves it.
    """

    context_id: str
    was_updated: bool
    embedding_stored: bool
    generation_status: GenerationStatus


async def execute_store_in_transaction(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    *,
    thread_id: str,
    source: str,
    content_type: str,
    text_content: str,
    scope: AccessScope,
    visibility: str,
    author_group_grants: 'Collection[str]' = (),
    metadata_str: str | None,
    summary: str | None,
    tags: list[str] | None,
    validated_images: list[dict[str, str]],
    images_provided: bool,
    deduplicate: bool,
    defer_generation: bool,
    chunk_embeddings: list[ChunkEmbedding] | None,
    embedding_model: str,
    embedding_generation_enabled: bool = False,
    index_nodes: list[IndexNodeRow] | None = None,
    nodes_pending: bool = False,
    summary_pending: bool = False,
) -> StoreOutcome:
    """Execute all store operations within an existing transaction.

    Performs the storage of a single context entry:
    1. Store the entry: deduplicating through store_with_deduplication when
       deduplicate is true, otherwise one fresh owner-stamped insert; either
       way under the caller's scope, stamping owner_id/visibility on a fresh INSERT
    2. Store author-group read grants on a fresh INSERT (when configured)
    3. Store/replace tags based on dedup outcome
    4. Store/replace images based on dedup outcome
    5. Store embeddings (skip if dedup + embeddings already exist) and node rows
    6. Leave the generation job the entry of a deferring store needs
    7. Read the entry's generation state for the response

    Args:
        repos: Repository container with context, tags, images, embeddings repos.
        txn: Active transaction context.
        thread_id: Thread identifier.
        source: 'user' or 'agent'.
        content_type: 'text' or 'multimodal'.
        text_content: The text content to store.
        scope: The caller's scope, built from the server-resolved effective
            principal (never caller-supplied at the tool boundary).
            Deduplication considers only entries it may read and merges only
            into an entry its principal owns; a fresh INSERT is owned by its
            principal, which is also recorded as the grantor of author-group
            grants.
        visibility: Validated visibility value stamped on a fresh INSERT. A
            deduplication UPDATE leaves the existing row's owner_id and
            visibility untouched.
        author_group_grants: Group ids that receive a read grant when this
            store INSERTs a new entry (the ACCESS_CONTROL_DEFAULT_GROUP_GRANTS
            author_groups policy, resolved by the caller; empty means none).
        metadata_str: JSON-serialized metadata or None.
        summary: Generated/preserved summary or None.
        tags: Tag list or None. None PRESERVES existing tags on a dedup UPDATE;
            a provided list (including []) REPLACES them, matching the
            documented replacement contract and update_context semantics.
        validated_images: Validated image list (may be empty).
        images_provided: Whether the CALLER passed an images value; True with
            an empty validated_images clears existing images on a dedup UPDATE
            instead of preserving them.
        deduplicate: True stores through store_with_deduplication, which
            updates the caller's identical latest entry of the thread and
            source instead of inserting a duplicate; False always inserts one
            new entry owned by the caller through insert_entry.
        defer_generation: True when the caller generated nothing and leaves the
            entry's outputs to the generation worker: a new entry receives a job;
            an entry a retransmit deduplicated into keeps a waiting or running job,
            has a failed job started over, and receives a job when it lacks
            embeddings while an embedding provider is active or lacks a summary
            its text gets. The caller passes True only while a generation provider
            is active. The reconcile triggers below never fire for such a store.
        chunk_embeddings: Generated embeddings or None.
        embedding_model: Model name for embedding storage.
        embedding_generation_enabled: True when an embedding provider is
            configured. When True and this store INSERTs a new entry
            (was_updated False) while chunk_embeddings is None -- which only
            happens when the caller's read-only pre-check skipped generation
            expecting a deduplication UPDATE -- the transaction is aborted via
            EmbeddingsReconcileRequiredError so the caller can regenerate
            embeddings outside the transaction and retry. Defaults to False so
            callers unaware of the pre-check optimization keep prior behavior.
        nodes_pending: True when the index_tree node layer is active and the
            caller's pre-check skipped node generation for a likely duplicate.
            When True and this store INSERTs a new entry (was_updated False)
            while index_nodes is None, the transaction aborts via
            EmbeddingsReconcileRequiredError so the caller regenerates node
            summaries outside the transaction and retries -- even when embedding
            generation is disabled. Defaults to False.
        summary_pending: True when the caller's pre-check REUSED the likely
            duplicate's stored summary instead of generating one. When True and
            this store INSERTs a new entry (was_updated False), the reused
            summary was read from a candidate that has since diverged and may
            describe different text, so the transaction aborts via
            EmbeddingsReconcileRequiredError for the caller to regenerate the
            summary outside the transaction and retry. Defaults to False.

    Returns:
        The stored entry's id, whether deduplication updated an existing entry,
        whether embeddings were written, and the entry's generation state.

    Raises:
        ToolError: If the store returns no context id.
        EmbeddingsReconcileRequiredError: If a store that does not defer generation
            inserted a new entry while the caller's pre-check had skipped embedding
            generation, or (when nodes_pending) node-summary generation; signals the
            caller to regenerate the skipped legs outside the transaction and retry.
    """
    if deduplicate:
        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            text_content=text_content,
            scope=scope,
            visibility=visibility,
            metadata=metadata_str,
            summary=summary,
            # Preserve the existing content_type on a dedup UPDATE only when no images
            # value was provided this call (images are preserved, not replaced).
            # Overwriting it then would flip a multimodal entry to 'text' while its
            # image rows remain, making them unretrievable. When images ARE provided
            # -- including an explicit empty list, which clears them below --
            # content_type is overwritten to match the request.
            preserve_content_type_on_dedup=not images_provided,
            txn=txn,
        )
    else:
        context_id = await repos.context.insert_entry(
            thread_id, source, content_type, text_content,
            scope=scope, visibility=visibility, metadata=metadata_str, summary=summary, txn=txn,
        )
        was_updated = False

    if not context_id:
        raise ToolError('Failed to store context')

    # Generation-first reconciliation: the caller's read-only pre-check skips
    # embedding AND node-summary generation (and may REUSE the candidate's
    # summary) when a likely duplicate already has them, expecting this store to
    # deduplicate into an UPDATE. If a concurrent same-thread write committed in
    # the meantime, store_with_deduplication can instead INSERT a brand-new
    # entry (was_updated False). Committing now would persist a row missing its
    # embeddings (when generation is enabled) or its per-node summaries (when
    # the node layer is active) -- or carrying a REUSED summary read from the
    # since-diverged candidate, which may describe DIFFERENT text (the summary
    # read happens after the hash check, so a commit between them poisons it).
    # Abort so the caller regenerates the skipped/reused legs OUTSIDE the
    # transaction and retries. The three reconcile triggers are decoupled so
    # each leg is repaired regardless of which others are active. A deferring
    # store generated nothing and skipped no leg: its new entry lacks every
    # output until the generation worker writes them.
    needs_embedding_reconcile = embedding_generation_enabled and chunk_embeddings is None
    needs_node_reconcile = nodes_pending and index_nodes is None
    needs_summary_reconcile = summary_pending
    if (
        not defer_generation
        and not was_updated
        and (needs_embedding_reconcile or needs_node_reconcile or needs_summary_reconcile)
    ):
        raise EmbeddingsReconcileRequiredError(text_content)

    # Heartbeat: keep connection alive between sequential operations
    await transaction_heartbeat(txn)

    # Author-group read grants land only with a fresh INSERT: a deduplication
    # UPDATE targets a row whose grants were stamped when it was inserted, and a
    # retransmit must not widen (or re-attribute) existing access.
    if not was_updated and author_group_grants:
        await repos.grants.store_group_read_grants(context_id, author_group_grants, scope=scope, txn=txn)

    # Store or replace tags depending on deduplication outcome. The documented
    # contract distinguishes PROVIDED from None: an explicitly provided empty
    # list REPLACES (clears) existing tags on a dedup UPDATE, exactly like
    # update_context's `if tags is not None` semantics; only None preserves.
    # On a fresh INSERT an empty list stores nothing, so the write is skipped.
    if tags is not None:
        if was_updated:
            await repos.tags.replace_tags_for_context(context_id, tags, txn=txn, scope=scope)
        elif tags:
            await repos.tags.store_tags(context_id, tags, txn=txn, scope=scope)

    # Store or replace images depending on deduplication outcome, with the same
    # provided-vs-None distinction. validated_images is always a list (the
    # validator normalizes None to []), so images_provided carries whether the
    # CALLER passed an images value; when it did, an empty list clears existing
    # images on a dedup UPDATE instead of preserving them.
    if images_provided:
        if was_updated:
            await repos.images.replace_images_for_context(
                context_id, validated_images, txn=txn, scope=scope,
            )
        elif validated_images:
            await repos.images.store_images(context_id, validated_images, txn=txn, scope=scope)

    # Store embeddings only if:
    # 1. New entry (not was_updated) - always store, OR
    # 2. Deduplicated entry (was_updated) but no embeddings exist yet
    # Skip if: Deduplicated entry AND embeddings already exist
    embedding_stored = False
    if chunk_embeddings is not None:
        # Heartbeat before potentially long embedding storage
        await transaction_heartbeat(txn)

        should_store = True
        if was_updated:
            embedding_exists = await repos.embeddings.exists(context_id, txn=txn, scope=scope)
            should_store = not embedding_exists
            if not should_store:
                logger.debug(
                    'Skipping embedding storage for deduplicated context %s '
                    '(embeddings already exist)',
                    context_id,
                )

        if should_store:
            await repos.embeddings.store_chunked(
                context_id=context_id,
                chunk_embeddings=chunk_embeddings,
                model=embedding_model,
                txn=txn,
                upsert=was_updated,
                scope=scope,
            )
            embedding_stored = True

    # Replace index_tree node summaries atomically. None means the per-node
    # summary feature is off, so the node table is left untouched. An empty list
    # clears stale rows, but only on a fresh INSERT: on a dedup UPDATE a
    # post-reconcile [] (coerced from total node-summary degradation) must NOT
    # wipe an existing entry's node rows, so an empty list is suppressed when
    # was_updated is True.
    if index_nodes is not None and (not was_updated or index_nodes):
        await repos.index_nodes.replace_nodes_for_context(context_id, index_nodes, txn=txn, scope=scope)

    if defer_generation:
        await _leave_deferred_store_job(
            repos, txn, context_id, inserted=not was_updated, text_content=text_content, scope=scope,
        )

    return StoreOutcome(
        context_id=context_id,
        was_updated=was_updated,
        embedding_stored=embedding_stored,
        generation_status=await repos.generation_jobs.state_for(context_id, txn=txn),
    )


async def _leave_deferred_store_job(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    context_id: str,
    *,
    inserted: bool,
    text_content: str,
    scope: AccessScope,
) -> None:
    """Leave the generation job the entry of a deferring store needs.

    A new entry receives a job. An entry a retransmit deduplicated into keeps a waiting
    or running job, has a failed job replaced so its attempts start over, and receives a
    job only when it lacks an output generation would produce.

    Args:
        repos: Repository container.
        txn: The store's transaction.
        context_id: The stored entry's id.
        inserted: Whether the store inserted the entry.
        text_content: The entry's text.
        scope: The caller's scope.
    """
    jobs = repos.generation_jobs
    if not inserted:
        state = await jobs.state_for(context_id, txn=txn)
        if state == 'pending':
            return
        if state == 'failed':
            await jobs.delete_for_entry(context_id, txn=txn)
        elif not await _outputs_missing(repos, txn, context_id, text_content=text_content, scope=scope):
            return
    await jobs.enqueue(context_id, txn=txn)


async def _outputs_missing(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    context_id: str,
    *,
    text_content: str,
    scope: AccessScope,
) -> bool:
    """Report whether an entry lacks an output generation would produce for it now.

    Args:
        repos: Repository container.
        txn: The open transaction.
        context_id: The entry's id.
        text_content: The entry's text.
        scope: The caller's scope.

    Returns:
        True when an embedding provider is active and the entry holds no embeddings, or
        its text gets a summary and the entry holds none.
    """
    if get_embedding_provider() is not None and not await repos.embeddings.exists(context_id, txn=txn, scope=scope):
        return True
    return summary_applies(text_content) and await repos.context.has_summary(context_id, scope=scope, txn=txn) is False
