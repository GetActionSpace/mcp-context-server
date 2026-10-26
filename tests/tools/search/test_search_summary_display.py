"""Tests for summary display in semantic, FTS, and hybrid search results."""

from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.tools.search.fts import fts_search_context
from app.tools.search.hybrid import hybrid_search_context
from app.tools.search.semantic import semantic_search_context
from tests.helpers import preserve_summary_state
from tests.tools._generation_jobs import InMemoryGenerationJobs


@pytest.fixture(autouse=True)
def reset_summary_state() -> Generator[None, None, None]:
    """Reset global summary state between tests."""
    with preserve_summary_state():
        yield


class TestSummarySearchDisplay:
    """Tests for unified search display formatting in semantic, FTS, and hybrid search tools."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_server_dependencies')
    async def test_semantic_search_shows_truncated_text_and_summary(self) -> None:
        """Show truncated text_content and summary as separate fields in semantic search."""
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        mock_repos = MagicMock()
        mock_repos.generation_jobs = InMemoryGenerationJobs()
        mock_repos.embeddings.search = AsyncMock(return_value=([
            {
                'id': 1,
                'thread_id': 'sem-summary-thread',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Full text content that is very long',
                'metadata': None,
                'summary': 'Concise semantic summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'distance': 0.3,
                'visibility': 'private',
                'is_owner': 1,
                'matched_chunk_start': None,
                'matched_chunk_end': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.images.get_images_for_context = AsyncMock(return_value=[])

        with (
            patch('app.tools.search.semantic.ensure_repositories', new=AsyncMock(return_value=mock_repos)),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.semantic.settings') as mock_settings,
            patch('app.tools.search.ranking.settings', mock_settings),
        ):
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'
            mock_settings.search.truncation_length = 150

            result = await semantic_search_context(
                query='test query',
                limit=10,
            )

        assert len(result['results']) == 1
        entry = result['results'][0]
        # text_content is truncated original, not summary
        assert entry['text_content'] != 'Concise semantic summary'
        assert entry['is_text_content_truncated'] is False  # short enough, not truncated
        assert entry['summary'] == 'Concise semantic summary'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_server_dependencies')
    async def test_fts_search_shows_truncated_text_and_summary(self) -> None:
        """Show truncated text_content and summary as separate fields in FTS search."""
        mock_repos = MagicMock()
        mock_repos.generation_jobs = InMemoryGenerationJobs()
        mock_repos.fts.is_available = AsyncMock(return_value=True)
        mock_repos.fts.search = AsyncMock(return_value=([
            {
                'id': 2,
                'thread_id': 'fts-summary-thread',
                'source': 'user',
                'content_type': 'text',
                'text_content': 'Full text content for FTS indexing',
                'metadata': None,
                'summary': 'Concise FTS summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'score': 5.0,
                'visibility': 'private',
                'is_owner': 1,
                'highlighted': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.images.get_images_for_context = AsyncMock(return_value=[])

        mock_fts_status = MagicMock()
        mock_fts_status.in_progress = False

        with (
            patch('app.tools.search.fts.ensure_repositories', new=AsyncMock(return_value=mock_repos)),
            patch('app.tools.search.fts.settings') as mock_settings,
            patch('app.tools.search.legs.settings', mock_settings),
            patch('app.tools.search.ranking.settings', mock_settings),
            patch('app.tools.search.fts.get_fts_migration_status', return_value=mock_fts_status),
            patch('app.tools.search.legs.get_fts_migration_status', return_value=mock_fts_status),
        ):
            mock_settings.fts.enabled = True
            mock_settings.fts.language = 'english'
            mock_settings.reranking.enabled = False
            mock_settings.search.truncation_length = 150

            result = await fts_search_context(
                query='test query',
                limit=10,
            )

        assert len(result['results']) == 1
        entry = result['results'][0]
        # text_content is truncated original, not summary
        assert entry['text_content'] != 'Concise FTS summary'
        assert entry['is_text_content_truncated'] is False  # short enough, not truncated
        assert entry['summary'] == 'Concise FTS summary'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_server_dependencies')
    async def test_hybrid_search_shows_truncated_text_and_summary(self) -> None:
        """Show truncated text_content and summary as separate fields in hybrid search."""
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        mock_repos = MagicMock()
        mock_repos.generation_jobs = InMemoryGenerationJobs()
        mock_repos.fts.is_available = AsyncMock(return_value=True)
        # FTS returns results with summary
        mock_repos.fts.search = AsyncMock(return_value=([
            {
                'id': 3,
                'thread_id': 'hybrid-summary-thread',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Full text content for hybrid search',
                'metadata': None,
                'summary': 'Concise hybrid summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'score': 5.0,
                'visibility': 'private',
                'is_owner': 1,
                'highlighted': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        # Semantic returns same entry
        mock_repos.embeddings.search = AsyncMock(return_value=([
            {
                'id': 3,
                'thread_id': 'hybrid-summary-thread',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Full text content for hybrid search',
                'metadata': None,
                'summary': 'Concise hybrid summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'distance': 0.3,
                'visibility': 'private',
                'is_owner': 1,
                'matched_chunk_start': None,
                'matched_chunk_end': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.images.get_images_for_context = AsyncMock(return_value=[])

        mock_fts_status = MagicMock()
        mock_fts_status.in_progress = False

        with (
            patch('app.tools.search.hybrid.ensure_repositories', new=AsyncMock(return_value=mock_repos)),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.hybrid.settings') as mock_settings,
            patch('app.tools.search.legs.settings', mock_settings),
            patch('app.tools.search.ranking.settings', mock_settings),
            patch('app.tools.search.limits.settings', mock_settings),
            patch('app.tools.search.legs.get_fts_migration_status', return_value=mock_fts_status),
        ):
            mock_settings.hybrid_search.enabled = True
            mock_settings.hybrid_search.rrf_k = 60
            mock_settings.hybrid_search.rrf_overfetch = 2
            mock_settings.hybrid_search.fts_or_threshold = 4
            mock_settings.fts.enabled = True
            mock_settings.fts.language = 'english'
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'
            mock_settings.embedding.provider = 'ollama'
            mock_settings.reranking.enabled = False
            mock_settings.storage.backend_type = 'sqlite'
            mock_settings.search.truncation_length = 150

            result = await hybrid_search_context(
                query='test query',
                limit=10,
            )

        assert len(result['results']) == 1
        entry = result['results'][0]
        # text_content is truncated original, not summary
        assert entry['text_content'] != 'Concise hybrid summary'
        assert entry['is_text_content_truncated'] is False  # short enough, not truncated
        assert entry['summary'] == 'Concise hybrid summary'
