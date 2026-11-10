"""Tests for date range filtering in search_context.

Covers how start_date and end_date reach the repository, validation errors, and end-to-end filtering
against a real SQLite database.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories import RepositoryContainer
from tests.tools._generation_jobs import InMemoryGenerationJobs

# Get the actual async functions from app.tools
store_context = app.tools.store_context
search_context = app.tools.search_context


@pytest.mark.usefixtures('mock_server_dependencies')
class TestSearchContextDateFiltering:
    """Test date filtering in search_context tool."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.mock_repos = MagicMock(spec=RepositoryContainer)
        self.mock_repos.context = AsyncMock()
        self.mock_repos.tags = AsyncMock()
        self.mock_repos.images = AsyncMock()
        self.mock_repos.generation_jobs = InMemoryGenerationJobs()

    @pytest.mark.asyncio
    async def test_filter_by_start_date_future(self) -> None:
        """Test filtering with future start_date calls repository correctly."""
        # Mock search_contexts to return empty results
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        future_date = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')

        with patch('app.tools.search.browse.ensure_repositories', return_value=self.mock_repos):
            result = await search_context(
                thread_id='date-test-1',
                start_date=future_date,
            limit=50,
            )

        assert len(result['results']) == 0
        # Verify start_date was passed to repository
        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] == future_date

    @pytest.mark.asyncio
    async def test_filter_by_end_date_past(self) -> None:
        """Test filtering with past end_date calls repository correctly.

        Note: Date-only end_date is expanded to end-of-day (T23:59:59.999999) by validate_date_param().
        """
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        past_date = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        # Expected expanded value includes end-of-day time with microsecond precision
        expected_end_date = f'{past_date}T23:59:59.999999'

        with patch('app.tools.search.browse.ensure_repositories', return_value=self.mock_repos):
            result = await search_context(
                thread_id='date-test-2',
                end_date=past_date,
            limit=50,
            )

        assert len(result['results']) == 0
        call_args = self.mock_repos.context.search_contexts.call_args
        # Verify end_date was expanded to end-of-day
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self) -> None:
        """Test filtering with both start and end dates.

        Note: Date-only end_date is expanded to end-of-day (T23:59:59.999999) by validate_date_param().
        """
        mock_entry = {
            'id': 1,
            'thread_id': 'date-test-3',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'Test entry',
            'metadata': None,
            'created_at': '2025-11-29 10:00:00',
            'updated_at': '2025-11-29 10:00:00',
            'visibility': 'private',
            'is_owner': 1,
        }
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([mock_entry], {}))
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).date().strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).strftime('%Y-%m-%d')
        # Expected expanded end_date includes end-of-day time with microsecond precision
        expected_end_date = f'{tomorrow}T23:59:59.999999'

        with patch('app.tools.search.browse.ensure_repositories', return_value=self.mock_repos):
            result = await search_context(
                thread_id='date-test-3',
                start_date=today,
                end_date=tomorrow,
            limit=50,
            )

        assert len(result['results']) == 1
        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] == today
        # Verify end_date was expanded to end-of-day
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_invalid_date_format_raises_error(self) -> None:
        """Test invalid date format raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            await search_context(
                thread_id='any-thread',
                start_date='invalid-date',
            limit=50,
            )
        assert 'Invalid start_date format' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_date_range_raises_error(self) -> None:
        """Test start_date > end_date raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            await search_context(
                thread_id='any-thread',
                start_date='2025-12-01',
                end_date='2025-11-01',
            limit=50,
            )
        assert 'Invalid date range' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_filter_with_datetime_format(self) -> None:
        """Test filtering with full datetime format."""
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        now = datetime.now(UTC)
        start = (now - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')
        end = (now + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')

        with patch('app.tools.search.browse.ensure_repositories', return_value=self.mock_repos):
            await search_context(
                thread_id='date-test-6',
                start_date=start,
                end_date=end,
            limit=50,
            )

        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] == start
        assert call_args[1]['end_date'] == end

    @pytest.mark.asyncio
    async def test_no_date_filter_passes_none(self) -> None:
        """Test that no date filter passes None to repository."""
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        with patch('app.tools.search.browse.ensure_repositories', return_value=self.mock_repos):
            await search_context(
                thread_id='date-test-7',
            limit=50,
            )

        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] is None
        assert call_args[1]['end_date'] is None


@pytest.mark.usefixtures('mock_server_dependencies', 'initialized_server')
class TestSearchContextDateIntegration:
    """Integration tests for date filtering with actual database."""

    @pytest.mark.asyncio
    async def test_date_only_end_date_includes_entire_day(self) -> None:
        """Test that date-only end_date includes ALL entries on that day.

        When a user specifies end_date='2025-11-29', they expect to include ALL
        entries created on November 29th, not just entries before midnight, so a
        date-only end_date expands to <= 2025-11-29T23:59:59.999999 rather than
        <= 2025-11-29 00:00:00.

        This follows Elasticsearch precedent where missing time components are
        replaced with max values for 'lte' operations.
        """
        # Store an entry - it will be created at the current time (e.g., 18:56:45)
        await store_context(
            thread_id='end-date-end-of-day-test',
            source='user',
            text='End-of-day test entry',
        )

        # Use date-only end_date for TODAY
        # The entry is found because:
        #   end_date='2025-11-29' -> <= '2025-11-29T23:59:59' (end of day)
        #   Entry at 18:56:45 <= 23:59:59 -> INCLUDED
        # Without the expansion the bound is midnight and the entry is excluded.
        today = datetime.now(UTC).date().strftime('%Y-%m-%d')

        result = await search_context(
            thread_id='end-date-end-of-day-test',
            end_date=today,
        limit=50,
        )

        # Entry should be found because end_date is expanded to end-of-day
        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'End-of-day test entry'

    @pytest.mark.asyncio
    async def test_date_filter_with_real_database(self) -> None:
        """Test date filtering with real database operations."""
        # Store a test entry
        await store_context(
            thread_id='date-integration-1',
            source='user',
            text='Integration test entry',
        )

        # Get today and tomorrow dates
        today = datetime.now(UTC).date().strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).strftime('%Y-%m-%d')

        # Search with date range including today - should find the entry
        result = await search_context(
            thread_id='date-integration-1',
            start_date=today,
            end_date=tomorrow,
        limit=50,
        )

        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_future_start_date_returns_empty(self) -> None:
        """Test that future start_date returns empty results."""
        # Store a test entry
        await store_context(
            thread_id='date-integration-2',
            source='agent',
            text='Another test entry',
        )

        # Future date should return empty
        future_date = (datetime.now(UTC).date() + timedelta(days=10)).strftime('%Y-%m-%d')

        result = await search_context(
            thread_id='date-integration-2',
            start_date=future_date,
        limit=50,
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_past_end_date_returns_empty(self) -> None:
        """Test that past end_date returns empty results."""
        # Store a test entry
        await store_context(
            thread_id='date-integration-3',
            source='user',
            text='Yet another test entry',
        )

        # Past date should return empty
        past_date = (datetime.now(UTC).date() - timedelta(days=10)).strftime('%Y-%m-%d')

        result = await search_context(
            thread_id='date-integration-3',
            end_date=past_date,
        limit=50,
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_date_filter_combined_with_source(self) -> None:
        """Test date filtering combined with source filter."""
        # Store entries with different sources
        await store_context(
            thread_id='date-integration-4',
            source='user',
            text='User entry',
        )
        await store_context(
            thread_id='date-integration-4',
            source='agent',
            text='Agent entry',
        )

        today = datetime.now(UTC).date().strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).strftime('%Y-%m-%d')

        # Filter by source and date
        result = await search_context(
            thread_id='date-integration-4',
            source='user',
            start_date=today,
            end_date=tomorrow,
        limit=50,
        )

        assert len(result['results']) == 1
        assert result['results'][0]['source'] == 'user'
