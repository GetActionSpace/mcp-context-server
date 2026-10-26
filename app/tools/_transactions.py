"""Store and update transaction execution for the write tools.

Runs the database writes of one store or update inside an open transaction,
writes an entry's generated embeddings and node rows for the update path and the
generation worker alike, keeps that transaction alive with a heartbeat,
classifies connection faults, re-probes an entry for the compare-and-set retry,
and defines the control-flow exceptions these paths raise to their callers.
"""

import asyncio
import json
import logging
from collections.abc import Collection
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import asyncpg
from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.backends.base import BackendAccess
from app.backends.sqlite_backend.contention import is_sqlite_locked_error
from app.errors import ControlFlowError
from app.repositories.context_repository.records import EntryProbe
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer


logger = logging.getLogger(__name__)


class EmbeddingsReconcileRequiredError(ControlFlowError):
    """Internal control-flow signal raised inside ``execute_store_in_transaction``.

    The read-only deduplication pre-check (performed by the caller OUTSIDE the
    transaction) skips embedding generation when a likely duplicate already has
    embeddings, on the assumption that this store will deduplicate into an
    UPDATE. If a concurrent same-thread write commits in the window between the
    pre-check and the transaction, ``store_with_deduplication`` can instead
    INSERT a brand-new entry. Committing that entry would leave a row with no
    embeddings while embedding generation is enabled, silently violating the
    generation-first guarantee.

    Raising this exception rolls the open transaction back and instructs the
    caller to regenerate embeddings OUTSIDE the transaction and retry the store.
    It is deliberately NOT a ``ToolError`` (so the ``except ToolError`` fast-path
    does not swallow it) and NOT a connection error (so it is not treated as a
    transient retry). ``text_content`` lets the caller regenerate embeddings for
    the exact entry that diverged.
    """

    def __init__(self, text_content: str) -> None:
        super().__init__('Embedding reconciliation required after deduplication divergence')
        self.text_content = text_content


class EntryNotFoundError(ControlFlowError):
    """Internal control-flow signal: the target context entry does not exist.

    Raised INSIDE an update transaction when a write targets a row that is gone
    (deleted concurrently, or a stale/wrong id): update_context_entry or
    patch_metadata reports no such row, or the tags-only / images-only path finds
    the parent missing before replacing its children. It is deliberately a
    ``ControlFlowError`` -- NOT a ``ToolError`` (so the ``except ToolError``
    fast-path does not swallow it) and NOT a connection error -- so the failed
    write is a clean client-input outcome that is NOT charged to the circuit
    breaker (a missing row is not a backend fault). The update tools catch it
    OUTSIDE the transaction and convert it to a not-found ``ToolError``.
    """

    def __init__(self, context_id: str) -> None:
        super().__init__(f'Context entry with ID {context_id} not found')
        self.context_id = context_id


type RefusedAction = Literal['modify', 'delete', 'share', 'unshare', 'list_grants']

# The text each refusal reports, given the comma-separated ids it names.
_REFUSAL_MESSAGES: dict[RefusedAction, str] = {
    'modify': 'Not authorized to modify context entry with ID {ids}',
    'delete': 'Not authorized to delete context entries: {ids}',
    'share': 'Not authorized to share context entry with ID {ids}',
    'unshare': 'Not authorized to unshare context entry with ID {ids}',
    'list_grants': 'Not authorized to list the grants of context entry with ID {ids}',
}


class EntryNotAuthorizedError(ControlFlowError):
    """Internal control-flow signal: the caller may read the entries but not do what it asked with them.

    An entry the caller may not read is reported as not found instead, so this
    signal never reveals a hidden entry: the caller already knows these entries
    exist. Like ``EntryNotFoundError`` it is a ``ControlFlowError`` -- a refused
    request is a client-input outcome, never charged to the circuit breaker --
    and its message is the exact text the tools report:

    - ``modify``: ``Not authorized to modify context entry with ID {id}``
    - ``delete``: ``Not authorized to delete context entries: {ids}``, the ids
      comma-separated in the order the caller named them
    - ``share``: ``Not authorized to share context entry with ID {id}``
    - ``unshare``: ``Not authorized to unshare context entry with ID {id}``
    - ``list_grants``: ``Not authorized to list the grants of context entry with ID {id}``

    Attributes:
        context_ids: The refused entry ids, in the caller's order.
        action: What the caller was refused.
    """

    def __init__(self, context_ids: Sequence[str], *, action: RefusedAction) -> None:
        self.context_ids = tuple(context_ids)
        self.action = action
        super().__init__(_REFUSAL_MESSAGES[action].format(ids=', '.join(self.context_ids)))


# ---------------------------------------------------------------------------
# Transaction utilities
# ---------------------------------------------------------------------------


async def transaction_heartbeat(txn: object) -> None:
    """Send lightweight heartbeat to prevent network intermediary idle timeout.

    Executes SELECT 1 on the connection to generate wire-protocol traffic,
    preventing NAT/firewall/proxy from classifying the connection as idle
    and closing it during long-running transactions.

    This is a defense-in-depth measure complementing TCP keepalive:
    - TCP keepalive operates at kernel level (probes every ~15s)
    - Heartbeat operates at application level (between sequential DB operations)
    - Together they provide maximum protection against intermediary timeouts

    For SQLite connections this is a no-op since SQLite does not use network
    connections and is not subject to intermediary idle timeouts.

    Args:
        txn: Transaction context (TransactionContext) providing connection and backend_type.
             Accepts object type for compatibility across backends.
    """
    backend_type = getattr(txn, 'backend_type', None)
    if backend_type != 'postgresql':
        return
    conn = getattr(txn, 'connection', None)
    if conn is None:
        return
    pg_conn = cast(asyncpg.Connection, conn)
    await pg_conn.execute('SELECT 1')


def is_connection_error(exc: Exception) -> bool:
    """Check if an exception is a transient DB error that is safe to retry.

    The classifier covers four transient families, not only lost connections.
    A retry is safe because the failed attempt never committed: PostgreSQL aborts
    the transaction on families 2 and 3, SQLite refuses the write of family 4
    before it takes effect and the transaction rolls back, and family 1 aborts
    the transaction when the connection drops before COMMIT. ALL
    embedding/summary/compression generation has already completed OUTSIDE the
    transaction (generation-first invariant), so a retry re-runs only the
    rolled-back DB write and never regenerates or skips generation.

    The exception is a connection lost while COMMIT is in flight: the attempt's
    outcome is unknown, and the retry runs the write again. An update keyed by id
    or a delete by id still leaves no more than one attempt's effect, and a single
    deduplicating store merges into the row its committed attempt inserted, but a
    ``deduplicate=false`` store, or an atomic batch store of several entries, can
    insert a second copy of an entry; for a corpus entry, the next sync plan lists
    that copy as a duplicate delete.

    The four families:

    1. Connection-level failures (the connection was lost, not a logical/data
       error): asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError,
       ConnectionResetError, OSError -- EXCLUDING TimeoutError. TimeoutError is
       an OSError subclass on Python 3.12, but the pool-acquire TimeoutError that
       begin_transaction re-raises uncharged signals a SATURATED connection pool,
       not a lost connection: retrying it re-runs the full POSTGRESQL_POOL_TIMEOUT_S
       acquire wait each time, multiplying one saturation stall into several. So a
       saturated pool must fail fast at the tool layer after one bounded wait,
       matching execute_write's fail-fast handling of the identical signal.
    2. Statement / lock-wait timeouts: asyncpg.exceptions.QueryCanceledError
       (SQLSTATE 57014). PostgreSQL cancels the statement when it exceeds the
       connection's statement_timeout (set to ~0.9 * POSTGRESQL_COMMAND_TIMEOUT_S
       in app.backends.postgresql_backend.pool_callbacks.setup_pool_connection). Retrying with the SAME ceiling
       only helps a TRANSIENT lock-WAIT (the write was blocked behind a
       concurrent writer and the contention has since cleared); it does NOT
       help a write that is fundamentally slower than the ceiling -- for that
       case (notably fp32 mode, ENABLE_EMBEDDING_COMPRESSION=false, where each
       per-chunk INSERT performs in-transaction HNSW maintenance) the operator
       must also raise POSTGRESQL_COMMAND_TIMEOUT_S or keep compression ON. See
       docs/database-backends.md.
    3. Transaction-rollback failures: asyncpg.exceptions.TransactionRollbackError
       (SQLSTATE class 40 -- deadlock_detected 40P01, serialization_failure 40001,
       and siblings). PostgreSQL aborts one transaction to break a deadlock or a
       serialization cycle; by definition the loser is expected to retry, and the
       retry succeeds once the competing transaction has committed. Without this
       class a deadlock (e.g. two atomic update batches that lock the same rows in
       opposite order) is neither a ControlFlowError nor a connection error, so it
       would charge the circuit breaker instead of retrying -- turning a routine,
       self-clearing lock cycle into an outage.

    4. SQLite write contention: sqlite3.OperationalError in the SQLITE_BUSY /
       SQLITE_LOCKED family ('database is locked'), classified by the shared
       is_sqlite_locked_error predicate. The backend's write-queue path
       (execute_write) retries this family internally, but begin_transaction --
       the path every store/update transaction site uses -- bypasses the write
       queue and performs NO backend-level retry, so a cross-process lock
       collision (e.g. two MCP server processes sharing one SQLite database
       file) must be retried by these tool-layer loops, mirroring the
       PostgreSQL class-40 treatment above. The backend re-raises the family
       without charging the circuit breaker for the same reason.

    QueryCanceledError is PostgreSQL-only; the isinstance check is harmless on
    SQLite, which never raises it.

    Args:
        exc: The exception to classify

    Returns:
        True if the exception is a transient DB error safe for retry
    """
    # TimeoutError is an OSError subclass; exclude it so a saturated-pool acquire
    # timeout is NOT retried (see the family list above). asyncpg.InterfaceError,
    # ConnectionDoesNotExistError, and ConnectionResetError are not TimeoutError
    # subclasses, so they still match.
    if isinstance(exc, TimeoutError):
        return False
    return is_sqlite_locked_error(exc) or isinstance(exc, (
        asyncpg.InterfaceError,
        asyncpg.ConnectionDoesNotExistError,
        asyncpg.exceptions.QueryCanceledError,
        asyncpg.exceptions.TransactionRollbackError,
        ConnectionResetError,
        OSError,
    ))


async def reread_entry_version(
    repos: 'RepositoryContainer',
    context_id: str,
    *,
    scope: AccessScope,
    max_retries: int = 2,
) -> EntryProbe:
    """Re-read an entry's optimistic-concurrency version after a failed compare-and-set.

    The update write paths capture ``version`` before their (LLM-bound) generation
    pass and compare-and-set on commit. When a concurrent writer wins that race the
    repository raises ``VersionConflictError`` and the caller must refresh the token
    BEFORE re-entering the write: ``version`` is monotonic, so re-issuing the write
    with the token whose compare-and-set just failed matches zero rows by
    construction -- a guaranteed-doomed transaction that also consumes one of the
    caller's bounded conflict slots.

    The operation that actually needs retrying after a transient fault is therefore
    this READ, which is what this helper retries: a dropped connection or a
    self-clearing lock collision during the refresh is retried here, on the read
    alone, and the caller re-enters the write only once a fresh version is in hand.

    Args:
        repos: Repository container.
        context_id: ID (32-char canonical hex) of the entry whose version to refresh.
        scope: The caller's scope; the entry is re-read as the caller.
        max_retries: Maximum transient-fault retries (exponential backoff).

    Returns:
        The fresh probe of the entry as the caller: ``exists`` is False when the
        entry is gone or the caller may no longer read it, and ``can_write`` is
        False when the caller may still read it but no longer modify it; the
        caller re-enters the write with ``version`` only when both hold.
    """
    attempt = 0
    while True:
        try:
            probe = await repos.context.check_entry_exists(context_id, scope=scope)
        except Exception as exc:
            if is_connection_error(exc) and attempt < max_retries:
                delay = 0.5 * (2 ** attempt)
                attempt += 1
                logger.warning(
                    'Transient error re-reading the version of context %s; retrying in %.1fs '
                    '(attempt %d/%d): %s',
                    context_id, delay, attempt, max_retries, exc,
                )
                await asyncio.sleep(delay)
                continue
            raise
        return probe


# ---------------------------------------------------------------------------
# Transaction execution helpers for store and update operations
# ---------------------------------------------------------------------------


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
    chunk_embeddings: list[ChunkEmbedding] | None,
    embedding_model: str,
    embedding_generation_enabled: bool = False,
    index_nodes: list[IndexNodeRow] | None = None,
    nodes_pending: bool = False,
    summary_pending: bool = False,
) -> tuple[str, bool, bool]:
    """Execute all store operations within an existing transaction.

    Performs the storage of a single context entry:
    1. Store the entry: deduplicating through store_with_deduplication when
       deduplicate is true, otherwise one fresh owner-stamped insert; either
       way under the caller's scope, stamping owner_id/visibility on a fresh INSERT
    2. Store author-group read grants on a fresh INSERT (when configured)
    3. Store/replace tags based on dedup outcome
    4. Store/replace images based on dedup outcome
    5. Store embeddings (skip if dedup + embeddings already exist)
    6. Track embedding_stored flag for response message parity

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
        Tuple of (context_id, was_updated, embedding_stored):
        - context_id: ID of stored/updated entry
        - was_updated: True if deduplication updated existing entry
        - embedding_stored: True if embeddings were written to DB

    Raises:
        ToolError: If the store returns no context id.
        EmbeddingsReconcileRequiredError: If the store inserted a new entry while
            the caller's pre-check had skipped embedding generation, or (when
            nodes_pending) node-summary generation; signals the caller to
            regenerate the skipped legs outside the transaction and retry.
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
    # each leg is repaired regardless of which others are active.
    needs_embedding_reconcile = embedding_generation_enabled and chunk_embeddings is None
    needs_node_reconcile = nodes_pending and index_nodes is None
    needs_summary_reconcile = summary_pending
    if not was_updated and (needs_embedding_reconcile or needs_node_reconcile or needs_summary_reconcile):
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

    return context_id, was_updated, embedding_stored


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
) -> tuple[list[str], bool]:
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

    Returns:
        Tuple of (updated_fields, summary_cleared):
        - updated_fields: List of field names that were updated
        - summary_cleared: True if summary was cleared (for response message)

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

    return updated_fields, clear_summary


async def write_generation_outputs(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    context_id: str,
    chunk_embeddings: list[ChunkEmbedding] | None,
    index_nodes: list[IndexNodeRow] | None,
    *,
    embedding_model: str,
    scope: BackendAccess,
) -> bool:
    """Write an entry's generated embeddings and index-tree node rows on an open transaction.

    Generated embeddings replace every stored chunk of the entry; a node list, empty
    included, replaces its node rows, and None leaves them as they are. The update path
    writes its regenerated outputs through this function, and so does the generation
    worker, so both store the outputs of one text the same way.

    Args:
        repos: Repository container.
        txn: The transaction that writes the entry's outputs.
        context_id: The entry's id.
        chunk_embeddings: The generated embeddings, or None when none were generated.
        index_nodes: The generated node rows, or None to keep the stored ones.
        embedding_model: The model the embeddings are stored under.
        scope: The identity the writes run as.

    Returns:
        Whether embeddings were written.
    """
    embeddings_written = False
    if chunk_embeddings is not None:
        await transaction_heartbeat(txn)
        await repos.embeddings.delete_all_chunks(context_id, txn=txn, scope=scope)
        await repos.embeddings.store_chunked(
            context_id=context_id,
            chunk_embeddings=chunk_embeddings,
            model=embedding_model,
            txn=txn,
            scope=scope,
        )
        embeddings_written = True
    if index_nodes is not None:
        await repos.index_nodes.replace_nodes_for_context(context_id, index_nodes, txn=txn, scope=scope)
    return embeddings_written
