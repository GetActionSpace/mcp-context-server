"""Fixtures shared by the update_context, get_context_by_ids and delete_context tool tests."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from app.repositories.context_repository.records import EntryProbe
from tests.tools._generation_jobs import NoGenerationJobs


@pytest.fixture
def mock_repositories():
    """Create mock repository container with all necessary repositories.

    Tools call ``backend.begin_transaction()`` and pass a ``txn`` argument to
    repository methods. Tests that assert on repository call arguments should
    use ``unittest.mock.ANY`` for the ``txn`` parameter.

    Returns:
        Mock: Repository container with mocked repositories.
    """
    from contextlib import asynccontextmanager

    repos = Mock()

    # Mock backend exposing begin_transaction() as an async context manager
    mock_backend = Mock()

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        _ = access_scope
        txn = Mock()
        txn.backend_type = 'sqlite'
        txn.connection = Mock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    # Mock context repository
    repos.context = Mock()
    repos.context.backend = mock_backend
    repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
    repos.context.entry_exists = AsyncMock(return_value=True)
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)
    repos.context.touch_updated_at = AsyncMock(return_value=True)
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))

    # Mock tags repository
    repos.tags = Mock()
    repos.tags.replace_tags_for_context = AsyncMock()

    # Mock images repository
    repos.images = Mock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    # Mock embeddings repository for generation-first transactional writes
    repos.embeddings = Mock()
    repos.embeddings.store_chunked = AsyncMock(return_value=None)
    repos.embeddings.delete_all_chunks = AsyncMock(return_value=None)
    repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)

    # Mock index_tree node-summary repository. With the per-node feature enabled (default),
    # a text-change update clears stale node rows via replace_nodes_for_context([]) -- gated
    # on node_summaries_enabled, NOT on a provider being present.
    repos.index_nodes = Mock()
    repos.index_nodes.replace_nodes_for_context = AsyncMock(return_value=None)
    repos.index_nodes.get_nodes_for_context = AsyncMock(return_value={})

    # Generation job repository: no entry holds a job, so every entry reads ready.
    repos.generation_jobs = NoGenerationJobs()

    return repos
