"""Tests for summary provider initialization and shutdown in the server lifespan."""

from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.startup
from app.startup import set_backend
from app.startup import set_chunking_service
from app.startup import set_embedding_provider
from app.startup import set_repositories
from app.startup import set_reranking_provider
from app.startup import set_summary_provider
from tests.helpers import patch_database_setup_steps
from tests.helpers import patch_generation_worker_phase
from tests.helpers import preserve_summary_state


@pytest.fixture(autouse=True)
def reset_summary_state() -> Generator[None, None, None]:
    """Reset global summary state between tests."""
    with preserve_summary_state():
        yield


class TestSummaryLifespan:
    """Tests for summary provider initialization and shutdown in server lifespan."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_and_shuts_down_summary_provider(self) -> None:
        """Initialize the summary provider on startup and shut it down on exit."""
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        mock_summary_provider = MagicMock()
        mock_summary_provider.initialize = AsyncMock()
        mock_summary_provider.shutdown = AsyncMock()
        mock_summary_provider.is_available = AsyncMock(return_value=True)
        mock_summary_provider.provider_name = 'test-summary-provider'

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.semantic_search.mode = 'false'
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.summary.generation_enabled = True
        mock_settings.summary.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup.get_backend()
        original_repos = app.startup.get_repositories()
        original_embedding_provider = app.startup.get_embedding_provider()
        original_reranking_provider = app.startup.get_reranking_provider()
        original_chunking_service = app.startup.get_chunking_service()
        original_summary_provider = app.startup.get_summary_provider()

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch_database_setup_steps(),
                patch_generation_worker_phase(),
                patch('app.startup.tool_registration.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch(
                    'app.migrations.check_summary_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None}),
                ),
                patch('app.summary.create_summary_provider', return_value=mock_summary_provider),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

                async with lifespan(mock_mcp):
                    assert app.startup.get_summary_provider() is mock_summary_provider

                mock_summary_provider.initialize.assert_awaited_once()
                mock_summary_provider.shutdown.assert_awaited_once()
                assert app.startup.get_summary_provider() is None
        finally:
            set_backend(original_backend)
            set_repositories(original_repos)
            set_embedding_provider(original_embedding_provider)
            set_reranking_provider(original_reranking_provider)
            set_chunking_service(original_chunking_service)
            set_summary_provider(original_summary_provider)

    @pytest.mark.asyncio
    async def test_lifespan_summary_disabled(self) -> None:
        """Set summary provider to None when summary generation is disabled."""
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.semantic_search.mode = 'false'
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.summary.generation_enabled = False
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup.get_backend()
        original_repos = app.startup.get_repositories()
        original_embedding_provider = app.startup.get_embedding_provider()
        original_reranking_provider = app.startup.get_reranking_provider()
        original_chunking_service = app.startup.get_chunking_service()
        original_summary_provider = app.startup.get_summary_provider()

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch_database_setup_steps(),
                patch('app.startup.tool_registration.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

                async with lifespan(mock_mcp):
                    assert app.startup.get_summary_provider() is None

                assert app.startup.get_summary_provider() is None
        finally:
            set_backend(original_backend)
            set_repositories(original_repos)
            set_embedding_provider(original_embedding_provider)
            set_reranking_provider(original_reranking_provider)
            set_chunking_service(original_chunking_service)
            set_summary_provider(original_summary_provider)
