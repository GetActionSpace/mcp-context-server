"""Response pieces the tools share: the per-entry fields of an entry read and the success messages of writes.

``entry_access_fields`` shapes the ``visibility`` and ``is_owner`` every entry-carrying
read response reports; ``generation_statuses`` and ``attach_generation_status`` read the
``generation_status`` it reports beside them, once per response; the ``build_*_message``
functions write the success-message text of single-entry and batch store and update
responses, naming an entry's generation state while its outputs are pending or failed.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.access_scope import AccessScope
from app.types import EntryAccessFields
from app.types import EntryVisibility
from app.types import GenerationStatus

if TYPE_CHECKING:
    from app.repositories import RepositoryContainer


def entry_access_fields(row: Mapping[str, object]) -> EntryAccessFields:
    """Shape the access fields of one entry from a repository row.

    The repositories compute ``is_owner`` as an integer (1 or 0) in SQL; the response
    reports a bool, because clients validate it against the boolean output schema.

    Args:
        row: A row carrying the entry's ``visibility`` and the caller's ``is_owner``.

    Returns:
        The entry's ``visibility`` and whether the caller owns it.
    """
    return {'visibility': cast(EntryVisibility, str(row['visibility'])), 'is_owner': bool(row['is_owner'])}


async def generation_statuses(
    context_ids: Iterable[str], repos: 'RepositoryContainer', *, scope: AccessScope,
) -> dict[str, GenerationStatus]:
    """Read the generation state of the entries one response carries, in one read.

    Args:
        context_ids: Ids of the entries a scoped read returned for this response, in
            canonical form; an id given more than once is read once.
        repos: The repositories.
        scope: The caller's scope.

    Returns:
        The state of each id: 'ready' when the entry's generated outputs are complete,
        'pending' while its generation job waits or runs, 'failed' once the job spent
        every attempt.
    """
    return await repos.generation_jobs.states_for_ids(list(dict.fromkeys(context_ids)), scope=scope)


async def attach_generation_status(
    entries: Sequence[MutableMapping[str, Any]], repos: 'RepositoryContainer', *, scope: AccessScope,
) -> None:
    """Set ``generation_status`` on every entry of one response, from one read.

    Args:
        entries: The response's entries as a scoped read returned them, each carrying
            its ``id``; each gains its ``generation_status`` in place.
        repos: The repositories.
        scope: The caller's scope.
    """
    statuses = await generation_statuses((str(entry['id']) for entry in entries), repos, scope=scope)
    for entry in entries:
        entry['generation_status'] = statuses[str(entry['id'])]


def build_store_response_message(
    *,
    action: str,
    image_count: int,
    embedding_generated: bool,
    embedding_stored: bool,
    summary_generated: bool,
    summary_preserved: bool,
    generation_status: GenerationStatus,
) -> str:
    """Build a response message for a store operation.

    Constructs a human-readable message with parenthetical detail parts
    covering embedding status, summary status, generation state, and image count.

    Args:
        action: 'stored' or 'updated' (deduplication outcome)
        image_count: Number of validated images (0 suppresses image mention)
        embedding_generated: Whether embeddings were generated
        embedding_stored: Whether generated embeddings were stored to DB
        summary_generated: Whether a new summary was generated
        summary_preserved: Whether an existing summary was reused
        generation_status: The entry's generation state after the store; a pending or
            failed state is named, a ready one is not

    Returns:
        Formatted message string like 'Context stored (embedding generated, summary generated)'.
    """
    parts: list[str] = []

    if embedding_generated and not embedding_stored:
        parts.append('embedding generated but not stored - duplicate')
    elif embedding_stored:
        parts.append('embedding generated')

    if summary_generated:
        parts.append('summary generated')
    elif summary_preserved:
        parts.append('summary preserved')
    parts.extend(_generation_parts(generation_status))

    # Suppress "with 0 images" when no images
    base = f'Context {action} with {image_count} images' if image_count > 0 else f'Context {action}'

    # Single consolidated parenthetical
    return f'{base} ({", ".join(parts)})' if parts else base


def build_update_response_message(
    *,
    updated_fields_count: int,
    embedding_generated: bool,
    summary_generated: bool,
    summary_cleared: bool,
    generation_status: GenerationStatus,
) -> str:
    """Build a response message for an update operation.

    Args:
        updated_fields_count: Number of fields updated
        embedding_generated: Whether embeddings were regenerated
        summary_generated: Whether summary was regenerated
        summary_cleared: Whether existing summary was cleared
        generation_status: The entry's generation state after the update; a pending or
            failed state is named, a ready one is not

    Returns:
        Formatted message string.
    """
    parts: list[str] = []
    if embedding_generated:
        parts.append('embedding regenerated')
    if summary_generated:
        parts.append('summary regenerated')
    elif summary_cleared:
        parts.append('summary cleared')
    parts.extend(_generation_parts(generation_status))

    base = f'Successfully updated {updated_fields_count} field(s)'
    return f'{base} ({", ".join(parts)})' if parts else base


def build_batch_store_response_message(
    *,
    succeeded: int,
    total: int,
    embeddings_generated_count: int,
    embeddings_stored_count: int,
    summaries_generated_count: int,
    summaries_preserved_count: int,
    generation_pending_count: int,
    generation_failed_count: int,
) -> str:
    """Build a response message for a batch store operation.

    Args:
        succeeded: Number of successfully stored entries
        total: Total number of entries in the batch
        embeddings_generated_count: Number of entries with generated embeddings
        embeddings_stored_count: Number of entries where embeddings were stored
        summaries_generated_count: Number of entries with generated summaries
        summaries_preserved_count: Number of entries with preserved summaries
        generation_pending_count: Number of stored entries whose generation is pending
        generation_failed_count: Number of stored entries whose generation failed

    Returns:
        Formatted batch message string.
    """
    parts: list[str] = []
    if embeddings_generated_count > 0:
        not_stored = embeddings_generated_count - embeddings_stored_count
        if not_stored > 0:
            parts.append(f'embeddings generated ({not_stored} not stored - duplicates)')
        else:
            parts.append('embeddings generated')
    if summaries_generated_count > 0:
        parts.append('summaries generated')
    if summaries_preserved_count > 0:
        parts.append('summaries preserved')
    parts.extend(_generation_count_parts(generation_pending_count, generation_failed_count))
    base = f'Stored {succeeded}/{total} entries successfully'
    return f'{base} ({", ".join(parts)})' if parts else base


def build_batch_update_response_message(
    *,
    succeeded: int,
    total: int,
    embeddings_generated_count: int,
    summaries_generated_count: int,
    summaries_cleared_count: int,
    generation_pending_count: int,
    generation_failed_count: int,
) -> str:
    """Build a response message for a batch update operation.

    Args:
        succeeded: Number of successfully updated entries
        total: Total number of entries in the batch
        embeddings_generated_count: Number of entries with regenerated embeddings
        summaries_generated_count: Number of entries with regenerated summaries
        summaries_cleared_count: Number of entries with cleared summaries
        generation_pending_count: Number of updated entries whose generation is pending
        generation_failed_count: Number of updated entries whose generation failed

    Returns:
        Formatted batch message string.
    """
    parts: list[str] = []
    if embeddings_generated_count > 0:
        parts.append('embeddings regenerated')
    if summaries_generated_count > 0:
        parts.append('summaries regenerated')
    if summaries_cleared_count > 0:
        parts.append('summaries cleared')
    parts.extend(_generation_count_parts(generation_pending_count, generation_failed_count))
    base = f'Updated {succeeded}/{total} entries successfully'
    return f'{base} ({", ".join(parts)})' if parts else base


def _generation_parts(generation_status: GenerationStatus) -> list[str]:
    """Name an entry's generation state when its outputs are outstanding or failed."""
    return [] if generation_status == 'ready' else [f'generation {generation_status}']


def _generation_count_parts(pending: int, failed: int) -> list[str]:
    """Count the entries of a batch whose generation is pending and whose generation failed."""
    return [
        f'generation {state} for {count} {"entry" if count == 1 else "entries"}'
        for state, count in (('pending', pending), ('failed', failed))
        if count > 0
    ]
