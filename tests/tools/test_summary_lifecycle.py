"""Tests for summary generation across the tool lifecycle: the timeout-bounded summary helper in
``app.tools._generation``, summary generation alongside embeddings in ``store_context``, and summary
regeneration, preservation and truncated-text display through ``update_context``, deduplicating stores
and ``search_context``.
"""

import asyncio
from collections.abc import Generator
from contextlib import asynccontextmanager
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
import app.tools._generation as generation_module
from app.backends.base import MAINTENANCE
from app.repositories.context_repository.records import EntryProbe
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.startup import ensure_repositories
from tests.helpers import LOCAL_SCOPE
from tests.helpers import LOCAL_WRITE_SCOPE
from tests.helpers import preserve_summary_state
from tests.tools._generation_jobs import InMemoryGenerationJobs

store_context = app.tools.store_context
update_context = app.tools.update_context
search_context = app.tools.search_context


def _create_mock_repositories() -> MagicMock:
    """Create mock repositories with transaction support for tool tests."""
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
    repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)
    repos.context.store_with_deduplication = AsyncMock(return_value=(123, False))
    repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content', 'summary']))
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)

    repos.tags = MagicMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    repos.images = MagicMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

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


@pytest.fixture(autouse=True)
def reset_summary_state() -> Generator[None, None, None]:
    """Reset global summary state between tests."""
    with preserve_summary_state():
        yield


@pytest.mark.usefixtures('mock_server_dependencies')
class TestGenerateSummaryWithTimeout:
    """Tests for summary generation helper behavior."""

    @pytest.mark.asyncio
    async def test_returns_summary_from_provider(self) -> None:
        """Return the provider result when summary generation succeeds."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Generated summary')

        with (
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await generation_module.generate_summary_with_timeout('Long text', 'agent')

        assert result == 'Generated summary'
        mock_provider.summarize.assert_awaited_once_with('Long text', 'agent')

    @pytest.mark.asyncio
    async def test_timeout_raises_tool_error(self) -> None:
        """Raise ToolError when total timeout is exceeded."""

        async def slow_summary(_text: str, _source: str) -> str:
            await asyncio.sleep(0.2)
            return 'Too late'

        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(side_effect=slow_summary)

        with (
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=0.05),
            pytest.raises(ToolError, match='Summary generation exceeded total timeout'),
        ):
            await generation_module.generate_summary_with_timeout('Long text', 'agent')

    @pytest.mark.asyncio
    async def test_provider_none_skips_generation(self) -> None:
        """Return None when no summary provider is configured."""
        with (
            patch('app.tools._generation.get_summary_provider', return_value=None),
        ):
            result = await generation_module.generate_summary_with_timeout('Long text', 'agent')

        assert result is None


@pytest.mark.usefixtures('mock_server_dependencies')
class TestSummaryStoreWithMocks:
    """Tests for summary behavior in store_context with mocked repositories."""

    @pytest.mark.asyncio
    async def test_store_context_generates_summary_in_parallel_with_embeddings(self) -> None:
        """Run summary and embedding generation concurrently before the transaction."""
        repos = _create_mock_repositories()
        embedding_started = asyncio.Event()
        summary_started = asyncio.Event()

        async def fake_embedding(text: str) -> list[ChunkEmbedding]:
            embedding_started.set()
            await asyncio.wait_for(summary_started.wait(), timeout=0.2)
            return [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=len(text))]

        async def fake_summary(_text: str, _source: str) -> str:
            summary_started.set()
            await asyncio.wait_for(embedding_started.wait(), timeout=0.2)
            return 'Generated summary'

        with (
            patch('app.tools.context.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.context.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools._generation.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.store.get_summary_provider', return_value=MagicMock()),
            patch('app.tools._generation.get_summary_provider', return_value=MagicMock()),
            patch('app.tools._generation.generate_embeddings_with_timeout', side_effect=fake_embedding),
            patch('app.tools._generation.generate_summary_with_timeout', side_effect=fake_summary),
        ):
            long_text = 'x' * 500
            result = await store_context(
                thread_id='parallel-summary-thread',
                source='agent',
                text=long_text,
            )

        assert result['success'] is True
        assert 'embedding generated' in result['message']
        assert 'summary generated' in result['message']
        # No images provided -> content_type is preserved on a dedup UPDATE (so a multimodal
        # entry can't flip to 'text' while its image rows remain). See store_with_deduplication.
        repos.context.store_with_deduplication.assert_awaited_once_with(
            thread_id='parallel-summary-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            scope=LOCAL_WRITE_SCOPE,
            visibility='private',
            metadata=None,
            summary='Generated summary',
            preserve_content_type_on_dedup=True,
            txn=ANY,
        )
        repos.embeddings.store_chunked.assert_awaited_once()


@pytest.mark.usefixtures('initialized_server')
class TestSummaryIntegration:
    """Integration tests for summary behavior with the real SQLite repositories."""

    @pytest.mark.asyncio
    async def test_update_context_regenerates_summary_on_text_change(self) -> None:
        """Regenerate and store a new summary when text changes."""
        repos = await ensure_repositories()
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='update-summary-thread',
            source='agent',
            content_type='text',
            text_content='Original text',
            metadata=None,
            summary='Original summary',
        )

        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Updated summary')

        updated_text = 'x' * 500

        with (
            patch('app.tools.context.update.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context(
                context_id=context_id,
                text=updated_text,
            )

        rows = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert rows[0]['text_content'] == updated_text
        assert rows[0]['summary'] == 'Updated summary'
        assert '(summary regenerated)' in result['message']
        mock_provider.summarize.assert_awaited_once_with(updated_text, 'agent')

    @pytest.mark.asyncio
    async def test_update_context_preserves_summary_on_metadata_only_change(self) -> None:
        """Leave an existing summary unchanged when only metadata is updated."""
        repos = await ensure_repositories()
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='metadata-preserve-thread',
            source='agent',
            content_type='text',
            text_content='Text with summary',
            metadata='{"status": "old"}',
            summary='Existing summary',
        )

        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Should not be used')

        with (
            patch('app.tools.context.update.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
        ):
            result = await update_context(
                context_id=context_id,
                metadata={'status': 'new'},
            )

        rows = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert rows[0]['summary'] == 'Existing summary'
        assert '(summary regenerated)' not in result['message']
        mock_provider.summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_context_shows_truncated_text_and_summary(self) -> None:
        """Search results show truncated text_content alongside summary field."""
        repos = await ensure_repositories()
        long_text = 'A' * 400
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='search-summary-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            metadata=None,
            summary='Short summary for search results',
        )

        result = await search_context(thread_id='search-summary-thread', limit=10)

        entry = result['results'][0]
        # text_content is truncated original text, NOT summary
        assert entry['text_content'] != long_text
        assert len(entry['text_content']) <= 303  # 300 + '...'
        assert entry['is_text_content_truncated'] is True
        # summary is a separate field
        assert entry['summary'] == 'Short summary for search results'

    @pytest.mark.asyncio
    async def test_search_context_truncates_without_summary(self) -> None:
        """Truncate long text and show empty summary when no summary stored."""
        repos = await ensure_repositories()
        long_text = 'B' * 400
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='search-fallback-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            metadata=None,
            summary=None,
        )

        result = await search_context(thread_id='search-fallback-thread', limit=10)

        entry = result['results'][0]
        assert entry['text_content'] != long_text
        assert len(entry['text_content']) < len(long_text)
        assert entry['is_text_content_truncated'] is True
        assert entry['summary'] == ''

    @pytest.mark.asyncio
    async def test_dedup_preserves_existing_summary(self) -> None:
        """Reuse the existing summary for duplicate content instead of regenerating it."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(side_effect=['Original summary', 'Unexpected second summary'])

        with (
            patch('app.tools.context.store.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            dedup_text = 'x' * 500
            first_result = await store_context(
                thread_id='dedup-summary-thread',
                source='agent',
                text=dedup_text,
            )
            second_result = await store_context(
                thread_id='dedup-summary-thread',
                source='agent',
                text=dedup_text,
            )

        repos = await ensure_repositories()
        rows = await repos.context.get_by_ids([first_result['context_id']], scope=LOCAL_SCOPE)
        assert first_result['context_id'] == second_result['context_id']
        assert rows[0]['summary'] == 'Original summary'
        assert '(summary preserved)' in second_result['message']
        assert mock_provider.summarize.await_count == 1

    @pytest.mark.asyncio
    async def test_dedup_generates_summary_when_missing(self) -> None:
        """Generate a summary for a duplicate entry that lacks an existing summary."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Newly generated summary')

        dedup_text = 'y' * 500

        with (
            patch('app.tools.context.store.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            # First store creates the entry (summary generated)
            first_result = await store_context(
                thread_id='dedup-no-summary-thread',
                source='agent',
                text=dedup_text,
            )

        # Manually clear the summary from the database to simulate missing summary
        repos = await ensure_repositories()
        context_id = first_result['context_id']

        if repos.context.backend.backend_type == 'sqlite':
            import sqlite3

            def _clear_summary(conn: sqlite3.Connection) -> None:
                conn.execute(
                    'UPDATE context_entries SET summary = NULL WHERE id = ?',
                    (context_id,),
                )

            await repos.context.backend.execute_write(_clear_summary, access_scope=MAINTENANCE)

        # Now store duplicate - should detect duplicate but find no summary, so generate one
        mock_provider2 = MagicMock()
        mock_provider2.summarize = AsyncMock(return_value='Summary for missing case')

        with (
            patch('app.tools.context.store.get_summary_provider', return_value=mock_provider2),
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider2),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            second_result = await store_context(
                thread_id='dedup-no-summary-thread',
                source='agent',
                text=dedup_text,
            )

        assert second_result['context_id'] == context_id
        assert '(summary generated)' in second_result['message']
        mock_provider2.summarize.assert_awaited_once_with(dedup_text, 'agent')

        rows = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert rows[0]['summary'] == 'Summary for missing case'
