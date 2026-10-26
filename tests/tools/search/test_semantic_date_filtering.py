"""Tests for date range filtering in semantic_search_context.

Covers how start_date and end_date are validated and passed to the embedding repository search.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.repositories import RepositoryContainer
from tests.tools._generation_jobs import InMemoryGenerationJobs


@pytest.mark.usefixtures('mock_server_dependencies')
class TestSemanticSearchDateFiltering:
    """Test date filtering in semantic_search_context tool.

    Tests verify that date parameters are correctly validated and passed
    to the embedding repository for semantic search operations.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.mock_repos = MagicMock(spec=RepositoryContainer)
        self.mock_repos.context = AsyncMock()
        self.mock_repos.tags = AsyncMock()
        self.mock_repos.embeddings = AsyncMock()
        self.mock_repos.generation_jobs = InMemoryGenerationJobs()

    @pytest.mark.asyncio
    async def test_semantic_search_with_start_date(self) -> None:
        """Test semantic_search_context with start_date filter."""
        # Mock embedding service
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        # Mock search results (a tuple of rows and stats)
        self.mock_repos.embeddings.search = AsyncMock(return_value=([
            {
                'id': 1,
                'thread_id': 'test-thread',
                'source': 'user',
                'text_content': 'Test entry',
                'distance': 0.5,
                'visibility': 'private',
                'is_owner': 1,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).strftime('%Y-%m-%d')

        with (
            patch('app.tools.search.semantic.ensure_repositories', return_value=self.mock_repos),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
            patch('app.tools.search.ranking.settings', mock_settings),
        ):
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'
            mock_settings.search.truncation_length = 150

            # Import and get the actual function
            import app.tools
            semantic_search = app.tools.semantic_search_context

            result = await semantic_search(
                query='test query',
                start_date=today,
            limit=20,
            )

        # Verify search was called with start_date
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] == today
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_semantic_search_with_end_date(self) -> None:
        """Test semantic_search_context with end_date filter.

        Note: Date-only end_date is expanded to end-of-day (T23:59:59.999999).
        """
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).strftime('%Y-%m-%d')
        expected_end_date = f'{today}T23:59:59.999999'

        with (
            patch('app.tools.search.semantic.ensure_repositories', return_value=self.mock_repos),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
            patch('app.tools.search.ranking.settings', mock_settings),
        ):
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'

            import app.tools
            semantic_search = app.tools.semantic_search_context

            await semantic_search(
                query='test query',
                end_date=today,
            limit=20,
            )

        # Verify end_date was expanded to end-of-day
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_semantic_search_with_date_range(self) -> None:
        """Test semantic_search_context with both start_date and end_date."""
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        expected_end_date = f'{tomorrow}T23:59:59.999999'

        with (
            patch('app.tools.search.semantic.ensure_repositories', return_value=self.mock_repos),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
            patch('app.tools.search.ranking.settings', mock_settings),
        ):
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'

            import app.tools
            semantic_search = app.tools.semantic_search_context

            await semantic_search(
                query='test query',
                start_date=today,
                end_date=tomorrow,
            limit=20,
            )

        # Verify both dates were passed correctly
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] == today
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_semantic_search_invalid_date_format_raises_error(self) -> None:
        """Test semantic_search_context with invalid date format raises ToolError."""
        mock_embedding_provider = MagicMock()

        with (
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
        ):
            mock_settings.semantic_search.enabled = True

            import app.tools
            semantic_search = app.tools.semantic_search_context

            with pytest.raises(ToolError) as exc_info:
                await semantic_search(
                    query='test query',
                    start_date='invalid-date',
                limit=20,
                )
            assert 'Invalid start_date format' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_semantic_search_invalid_date_range_raises_error(self) -> None:
        """Test semantic_search_context with start_date > end_date raises ToolError."""
        mock_embedding_provider = MagicMock()

        with (
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
        ):
            mock_settings.semantic_search.enabled = True

            import app.tools
            semantic_search = app.tools.semantic_search_context

            with pytest.raises(ToolError) as exc_info:
                await semantic_search(
                    query='test query',
                    start_date='2025-12-01',
                    end_date='2025-11-01',
                limit=20,
                )
            assert 'Invalid date range' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_semantic_search_with_datetime_format(self) -> None:
        """Test semantic_search_context with full datetime format."""
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        start = '2025-11-29T10:00:00'
        end = '2025-11-29T18:00:00'

        with (
            patch('app.tools.search.semantic.ensure_repositories', return_value=self.mock_repos),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
            patch('app.tools.search.ranking.settings', mock_settings),
        ):
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'

            import app.tools
            semantic_search = app.tools.semantic_search_context

            await semantic_search(
                query='test query',
                start_date=start,
                end_date=end,
            limit=20,
            )

        # Verify datetime format is preserved (not expanded)
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] == start
        assert call_args[1]['end_date'] == end

    @pytest.mark.asyncio
    async def test_semantic_search_no_date_filter_passes_none(self) -> None:
        """Test semantic_search_context without date filter passes None to repository."""
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        with (
            patch('app.tools.search.semantic.ensure_repositories', return_value=self.mock_repos),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
            patch('app.tools.search.ranking.settings', mock_settings),
        ):
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'

            import app.tools
            semantic_search = app.tools.semantic_search_context

            await semantic_search(
                query='test query',
            limit=20,
            )

        # Verify None dates were passed
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] is None
        assert call_args[1]['end_date'] is None
