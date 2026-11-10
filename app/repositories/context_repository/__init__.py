"""Context repository: every database operation on ``context_entries``.

``ContextRepository`` is assembled here from one mixin per group of operations:
browse search and grep scan, by-id reads and probes, the reads of ingest-managed
entries, the deduplicating store, updates, deletes, and the reads and writes that
generating an entry's outputs needs. Record types live in ``records`` and shared
pure helpers in ``helpers``.
"""

from app.repositories.context_repository.dedup import ContextDedupMixin
from app.repositories.context_repository.deletes import ContextDeleteMixin
from app.repositories.context_repository.generation import ContextGenerationMixin
from app.repositories.context_repository.ingest import ContextIngestMixin
from app.repositories.context_repository.reads import ContextReadMixin
from app.repositories.context_repository.search import ContextSearchMixin
from app.repositories.context_repository.updates import ContextUpdateMixin


class ContextRepository(
    ContextSearchMixin,
    ContextReadMixin,
    ContextIngestMixin,
    ContextDedupMixin,
    ContextUpdateMixin,
    ContextDeleteMixin,
    ContextGenerationMixin,
):
    """Repository for context entry operations.

    Handles storage, retrieval, search, and deletion of context entries
    with proper deduplication and transaction management.
    """
