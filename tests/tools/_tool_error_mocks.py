"""Mock repository container and ensure_repositories patches shared by the tool error-handling tests."""

from collections.abc import Generator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from tests.tools._generation_jobs import InMemoryGenerationJobs


def build_mock_repos() -> MagicMock:
    """Create a mock repository container with transaction support."""
    repos = MagicMock()

    mock_backend = Mock()

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        _ = access_scope
        txn = Mock()
        txn.backend_type = 'sqlite'
        txn.connection = Mock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    repos.context = AsyncMock()
    repos.context.backend = mock_backend
    repos.tags = AsyncMock()
    repos.images = AsyncMock()
    repos.statistics = AsyncMock()
    repos.generation_jobs = InMemoryGenerationJobs()

    repos.embeddings = AsyncMock()
    repos.embeddings.store_chunked = AsyncMock(return_value=None)
    repos.embeddings.delete_all_chunks = AsyncMock(return_value=None)

    return repos


@contextmanager
def patch_tool_repositories(repos: MagicMock) -> Generator[None, None, None]:
    """Patch ensure_repositories in each tool module where it is imported to return repos."""
    with (
        patch('app.tools.context.store.ensure_repositories', return_value=repos),
        patch('app.tools.context.retrieve.ensure_repositories', return_value=repos),
        patch('app.tools.context.update.ensure_repositories', return_value=repos),
        patch('app.tools.context.delete.ensure_repositories', return_value=repos),
        patch('app.tools.search.browse.ensure_repositories', return_value=repos),
        patch('app.tools.search.semantic.ensure_repositories', return_value=repos),
        patch('app.tools.search.fts.ensure_repositories', return_value=repos),
        patch('app.tools.search.hybrid.ensure_repositories', return_value=repos),
        patch('app.tools.discovery.ensure_repositories', return_value=repos),
    ):
        yield
