"""Mock repositories, mock transactions and provider-state preservation shared by the batch tool tests."""

from collections.abc import Generator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import app.startup
from app.repositories.context_repository.records import EntryProbe
from tests.tools._generation_jobs import InMemoryGenerationJobs


def make_mock_txn() -> tuple[MagicMock, object]:
    """Create a mock transaction and an async context-manager factory that yields it."""
    mock_txn = MagicMock()
    mock_txn.connection = MagicMock()
    mock_txn.backend_type = 'sqlite'

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        _ = access_scope
        yield mock_txn

    return mock_txn, mock_begin_transaction


def create_mock_repositories() -> MagicMock:
    """Create mock repositories with transaction support for batch tool tests."""
    repos = MagicMock()

    mock_backend = MagicMock()

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        _ = access_scope
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        txn.connection = MagicMock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    repos.context = MagicMock()
    repos.context.backend = mock_backend
    repos.context.store_with_deduplication = AsyncMock(return_value=(100, False))
    repos.context.insert_entry = AsyncMock(return_value=100)
    repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)
    repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content', 'summary']))
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.update_content_type = AsyncMock(return_value=True)

    repos.tags = MagicMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    repos.images = MagicMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    repos.context.get_content_type = AsyncMock(return_value='text')

    repos.embeddings = MagicMock()
    repos.embeddings.exists = AsyncMock(return_value=False)
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.delete_all_chunks = AsyncMock()
    repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)

    repos.index_nodes = MagicMock()
    repos.index_nodes.replace_nodes_for_context = AsyncMock()
    repos.index_nodes.get_nodes_for_context = AsyncMock(return_value={})
    repos.index_nodes.count_all_nodes = AsyncMock(return_value=0)

    repos.generation_jobs = InMemoryGenerationJobs()

    return repos


@contextmanager
def preserved_providers() -> Generator[None, None, None]:
    """Restore the global summary and embedding providers on exit."""
    original_summary = app.startup.get_summary_provider()
    original_embedding = app.startup.get_embedding_provider()
    try:
        yield
    finally:
        app.startup.set_summary_provider(original_summary)
        app.startup.set_embedding_provider(original_embedding)
