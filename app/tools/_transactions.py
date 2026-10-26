"""Shared pieces of the store and update transactions of the write tools.

Defines the control-flow exceptions the write paths raise to their callers, keeps a long
transaction alive with a heartbeat, classifies transient database faults, re-probes an
entry for the compare-and-set retry, and writes an entry's generated embeddings and node
rows for the update path and the generation worker alike.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import asyncpg

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
