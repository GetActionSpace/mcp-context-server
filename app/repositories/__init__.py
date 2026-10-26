"""
Repository pattern implementation for database operations.

This module provides clean separation of concerns by isolating all database operations
into focused repository classes
"""

# Type imports
from app.backends.base import StorageBackend
from app.repositories.context_repository import ContextRepository
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.fts_repository import FtsRepository
from app.repositories.generation_job_repository import GenerationJobRepository
from app.repositories.grant_repository import GrantRepository
from app.repositories.image_repository import ImageRepository
from app.repositories.index_node_repository import IndexNodeRepository
from app.repositories.statistics_repository import StatisticsRepository
from app.repositories.tag_repository import TagRepository


class RepositoryContainer:
    """Container for all repository instances providing dependency injection.

    This class manages repository instances and provides them to the server layer,
    ensuring proper separation of concerns and testability.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize repository container with storage backend.

        Args:
            backend: Storage backend for all repositories to use
        """
        self.context = ContextRepository(backend)
        self.tags = TagRepository(backend)
        self.images = ImageRepository(backend)
        self.statistics = StatisticsRepository(backend)
        self.embeddings = EmbeddingRepository(backend)
        self.fts = FtsRepository(backend)
        self.index_nodes = IndexNodeRepository(backend)
        self.grants = GrantRepository(backend)
        self.generation_jobs = GenerationJobRepository(backend)


__all__ = [
    'ContextRepository',
    'EmbeddingRepository',
    'FtsRepository',
    'GenerationJobRepository',
    'GrantRepository',
    'ImageRepository',
    'IndexNodeRepository',
    'StatisticsRepository',
    'TagRepository',
    'RepositoryContainer',
]
