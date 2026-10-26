"""Record types and the column list shared by the context repository and its callers.

Holds the optimistic-concurrency ``VersionConflictError``, the explicit
``CONTEXT_ENTRY_COLUMNS`` select list with its ``entry_columns`` renderer, the
``EntryProbe``, ``IdAccess`` and ``DuplicateCandidate`` result tuples, the
``GrepScanRow`` record of the grep pre-filter scan, the ``IngestRow`` record of
an ingest-managed entry, and the ``GenerationSource`` record of an entry whose
outputs are being generated.
"""

from dataclasses import dataclass
from typing import NamedTuple

from app.errors import ControlFlowError


class VersionConflictError(ControlFlowError):
    """Raised by ``update_context_entry`` when its optimistic-concurrency guard
    finds the row's ``version`` changed since the caller read it.

    A concurrent writer committed a newer update first, so applying this one
    would silently overwrite it (and leave index_tree node rows describing stale
    text). The caller re-reads the current version and retries the transaction so
    the update applies against the latest row instead of clobbering it. It is
    deliberately NOT a ``ToolError`` (so an ``except ToolError`` fast-path does
    not swallow it) and NOT a connection error (so it is not treated as a
    transient retry).
    """

    def __init__(self, context_id: str) -> None:
        super().__init__(f'Version conflict updating context {context_id}')
        self.context_id = context_id


# Explicit column list to avoid exposing internal database columns (e.g., text_search_vector)
# This constant is used in all SELECT queries that return context entries to ensure
# only the expected columns are returned, preventing internal PostgreSQL columns from
# leaking into API responses. ``visibility`` is reported on every entry read; the owner
# never is: a statement reports ownership as ``is_owner``, which needs the caller's bind
# and so is appended per statement.
CONTEXT_ENTRY_COLUMNS = (
    'id, thread_id, source, content_type, text_content, metadata, summary, created_at, updated_at, visibility'
)


def entry_columns(alias: str | None = None) -> str:
    """Render :data:`CONTEXT_ENTRY_COLUMNS` for a select list.

    Args:
        alias: The alias of the ``context_entries`` relation, or None for bare column names.

    Returns:
        The column list, with every column qualified by ``alias`` when one is given.
    """
    if alias is None:
        return CONTEXT_ENTRY_COLUMNS
    return ', '.join(f'{alias}.{column.strip()}' for column in CONTEXT_ENTRY_COLUMNS.split(','))


class EntryProbe(NamedTuple):
    """Existence probe result for one context entry (see ``check_entry_exists``).

    ``exists`` is True only for an entry the caller may read, so an entry the
    caller may not read probes exactly like a missing one. ``source``,
    ``version``, and ``owner_id`` are None and ``can_write`` is False when the
    entry does not exist for the caller. ``version`` is the optimistic-concurrency
    token the update paths capture BEFORE generation as their compare-and-set
    guard; ``owner_id`` backs the owner-only visibility-change authorization (it
    is immutable, so a pre-generation read of it cannot go stale); ``can_write``
    says whether the caller may modify the entry's content.
    """

    exists: bool
    source: str | None
    version: int | None
    owner_id: str | None
    can_write: bool


class IdAccess(NamedTuple):
    """What the caller may do with one readable entry (see ``probe_ids``).

    ``can_write`` says whether the caller may modify the entry's content;
    ``is_owner`` whether the caller owns it, the only role that may delete it or
    change its visibility.
    """

    can_write: bool
    is_owner: bool


class DuplicateCandidate(NamedTuple):
    """Statement-level snapshot of a likely-duplicate entry found by the pre-check.

    ``context_id`` and ``summary`` come from the SAME row read that matched the
    content hash, so the summary is guaranteed to describe the text the hash
    was computed from. Reading the summary in a separate later statement could
    observe a row version whose summary describes DIFFERENT text (a concurrent
    update landing between the two reads) -- and the dedup UPDATE's
    content-hash predicate cannot distinguish a revision-consistent row from a
    restored one, so the mismatched summary would persist via COALESCE.
    """

    context_id: str
    summary: str | None


@dataclass(frozen=True, slots=True)
class GrepScanRow:
    """One candidate entry of the grep pre-filter scan.

    Attributes:
        id: The entry id, as canonical 32-character lowercase hex.
        text_content: The entry's full text; an empty string when none is stored.
        visibility: The entry's stored visibility, 'private' or 'public'.
        is_owner: Whether the caller owns the entry.
    """

    id: str
    text_content: str
    visibility: str
    is_owner: bool


@dataclass(frozen=True, slots=True)
class IngestRow:
    """One ingest-managed entry as the ingest reads return it.

    Attributes:
        context_id: The entry id.
        source_id: The source id the entry's ``metadata.ingest`` block carries, as the
            statement selected it: the value the range bounds and the keyset compare.
        block: The parsed ``metadata.ingest`` block, unvalidated: its shape is checked
            by whoever plans or lists from it.
        content_hash: The stored hash of the entry's text; None for an entry stored
            before content hashes existed.
        is_owner: Whether the caller owns the entry.
    """

    context_id: str
    source_id: str
    block: object
    content_hash: str | None
    is_owner: bool


@dataclass(frozen=True, slots=True)
class GenerationSource:
    """What the generation of one entry reads: its text, and which outputs it already holds.

    Attributes:
        text_content: The entry's text; an empty string when none is stored.
        source: The entry's source, 'user' or 'agent'.
        content_hash: The stored hash of the text, which a completion is guarded on; None
            for an entry stored before content hashes existed.
        has_summary: Whether the entry stores a summary.
        has_embeddings: Whether the entry stores embeddings.
        node_count: The index-tree node rows the entry stores; 0 when node summaries were
            never provisioned.
    """

    text_content: str
    source: str
    content_hash: str | None
    has_summary: bool
    has_embeddings: bool
    node_count: int
