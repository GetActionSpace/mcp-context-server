"""
Tests for embedding concurrency controls and dynamic timeout computation.

Tests verify:
- compute_embedding_total_timeout() produces correct values for various settings
- Semaphore limits concurrent embedding requests, one slot per batch
- The total timeout applies per batch and raises ToolError naming the batch
- FTS/semantic error logging in hybrid search graceful degradation
"""

import asyncio
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.embeddings.batching import FixedBatchSizer
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.services.chunking_service import TextChunk

# --- Tests for compute_embedding_total_timeout() ---


@pytest.fixture
def mock_retry_settings():
    """Mock settings for compute_embedding_total_timeout tests."""
    with patch('app.embeddings.retry.get_settings') as mock:
        yield mock


def test_compute_timeout_default_settings(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with default settings (240s, 5 attempts, 1.0s delay)."""
    mock_retry_settings.return_value.embedding.timeout_s = 240.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 5
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 1.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # (5 * 240 + (min(1*1+1, 60) + min(1*2+1, 60) + min(1*4+1, 60) + min(1*8+1, 60))) * 1.1
    # = (1200 + (2 + 3 + 5 + 9)) * 1.1 = 1219 * 1.1 = 1340.9
    assert result == pytest.approx(1340.9, abs=0.1)


def test_compute_timeout_user_deployed_settings(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with user deployed settings (90s, 5 attempts, 1.0s delay)."""
    mock_retry_settings.return_value.embedding.timeout_s = 90.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 5
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 1.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # (5 * 90 + (2 + 3 + 5 + 9)) * 1.1 = 469 * 1.1 = 515.9
    assert result == pytest.approx(515.9, abs=0.1)


def test_compute_timeout_single_attempt(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with single attempt (no backoff intervals)."""
    mock_retry_settings.return_value.embedding.timeout_s = 240.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 1
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 1.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # (1 * 240 + 0) * 1.1 = 264.0 (no backoff with single attempt)
    assert result == pytest.approx(264.0, abs=0.1)


def test_compute_timeout_high_config(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with high configuration values."""
    mock_retry_settings.return_value.embedding.timeout_s = 120.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 5
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 2.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # attempt_time = 5 * 120 = 600
    # backoffs: min(2*1+2,60)=4, min(2*2+2,60)=6, min(2*4+2,60)=10, min(2*8+2,60)=18
    # total_backoff = 4 + 6 + 10 + 18 = 38
    # (600 + 38) * 1.1 = 701.8
    assert result == pytest.approx(701.8, abs=0.1)


def test_compute_timeout_backoff_cap(mock_retry_settings: MagicMock) -> None:
    """Test that backoff intervals are capped at 60 seconds."""
    mock_retry_settings.return_value.embedding.timeout_s = 10.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 10
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 10.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # attempt_time = 10 * 10 = 100
    # backoffs for i=0..8: min(10*(2^i)+10, 60)
    # i=0: min(20, 60)=20
    # i=1: min(30, 60)=30
    # i=2: min(50, 60)=50
    # i=3: min(90, 60)=60  (capped)
    # i=4..8: all capped at 60
    # total_backoff = 20+30+50+60*6 = 460
    # (100 + 460) * 1.1 = 616.0
    assert result == pytest.approx(616.0, abs=0.1)


# --- Tests for semaphore concurrency control ---


def _chunking_service(chunks_per_document: int) -> MagicMock:
    """An enabled chunking service splitting every document into a fixed number of chunks."""
    service = MagicMock()
    service.is_enabled = True
    service.chunk_size = 1500
    service.split_text = lambda text: [
        TextChunk(text=f'{text}-{i}', chunk_index=i, start_index=i, end_index=i + 1)
        for i in range(chunks_per_document)
    ]
    return service


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency() -> None:
    """Concurrent documents never have more provider requests in flight than EMBEDDING_MAX_CONCURRENT."""
    in_flight = 0
    peak = 0
    requests = 0

    class CountingProvider:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            nonlocal in_flight, peak, requests
            requests += 1
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return [[0.0] for _ in texts]

    max_concurrent_setting = 2

    with (
        patch('app.tools._generation.get_embedding_provider', return_value=CountingProvider()),
        patch('app.tools._generation.get_chunking_service', return_value=_chunking_service(6)),
        patch('app.tools._generation.compute_embedding_total_timeout', return_value=999.0),
        patch('app.tools._generation._batch_sizer', FixedBatchSizer(2)),
        patch('app.tools._generation.settings') as mock_settings,
    ):
        mock_settings.embedding.max_concurrent = max_concurrent_setting

        # Rebind the module-level semaphore against the patched setting so
        # the next acquisition uses the test's max_concurrent value.
        import app.tools._generation as generation_module

        original_semaphore = generation_module._embedding_semaphore
        generation_module._reset_embedding_semaphore()

        try:
            from app.tools._generation import generate_embeddings_with_timeout

            # Four documents of three batches each, embedded at once.
            await asyncio.gather(*(generate_embeddings_with_timeout(f'doc{n}') for n in range(4)))
        finally:
            generation_module._embedding_semaphore = original_semaphore

    assert requests == 12
    assert peak == max_concurrent_setting


@pytest.mark.asyncio
async def test_total_timeout_raises_tool_error() -> None:
    """A batch exceeding the total timeout fails the store and names the batch."""
    from fastmcp.exceptions import ToolError

    class SecondBatchHangsProvider:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if texts[0].endswith('-1'):
                await asyncio.sleep(10.0)
            return [[0.0] for _ in texts]

    mock_repos = MagicMock()
    mock_repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)
    provider = SecondBatchHangsProvider()

    with (
        patch('app.tools.context.store.get_embedding_provider', return_value=provider),
        patch('app.tools._generation.get_embedding_provider', return_value=provider),
        patch('app.tools._generation.get_chunking_service', return_value=_chunking_service(3)),
        patch('app.tools._generation.compute_embedding_total_timeout', return_value=0.05),
        patch('app.tools._generation._batch_sizer', FixedBatchSizer(1)),
        patch('app.tools._generation.settings') as mock_settings,
        patch('app.tools.context.store.ensure_repositories', new_callable=AsyncMock, return_value=mock_repos),
    ):
        mock_settings.embedding.max_concurrent = 3

        import app.tools._generation as generation_module

        original_semaphore = generation_module._embedding_semaphore
        generation_module._reset_embedding_semaphore()

        try:
            from app.tools.context.store import store_context

            # The first batch finishes well inside the timeout; the second one runs out of it.
            with pytest.raises(ToolError, match=r'total timeout \(0s\) on batch 2 of 3 \(chunk 2 of 3; 1 chunk embedded'):
                await store_context(
                    thread_id='test-thread',
                    source='agent',
                    text='Test text for embedding timeout',
                )
        finally:
            generation_module._embedding_semaphore = original_semaphore


@pytest.mark.asyncio
async def test_embedding_disabled_skips_semaphore() -> None:
    """Verify that when embedding provider is None, semaphore is not acquired."""
    mock_sem = MagicMock()
    mock_sem.__aenter__ = AsyncMock(return_value=mock_sem)
    mock_sem.__aexit__ = AsyncMock(return_value=False)

    with (
        patch('app.tools.context.store.get_embedding_provider', return_value=None),
        patch('app.tools._generation.get_embedding_provider', return_value=None),
        patch('app.tools._generation._embedding_semaphore', new=mock_sem),
        patch('app.tools.context.store.ensure_repositories', new_callable=AsyncMock) as mock_repos,
    ):
        mock_backend = MagicMock()
        mock_txn = MagicMock()
        mock_txn.__aenter__ = AsyncMock(return_value=mock_txn)
        mock_txn.__aexit__ = AsyncMock(return_value=False)
        mock_backend.begin_transaction.return_value = mock_txn
        mock_repos.return_value.context.backend = mock_backend
        mock_repos.return_value.context.store_with_deduplication = AsyncMock(return_value=(1, False))
        mock_repos.return_value.tags.store_tags = AsyncMock()
        mock_repos.return_value.images.store_images = AsyncMock()
        mock_repos.return_value.embeddings.store_chunked = AsyncMock()

        from app.tools.context.store import store_context

        result = await store_context(
            thread_id='test-thread',
            source='agent',
            text='Test text without embedding',
        )

        assert result['success'] is True
        # When the embedding provider is None, the semaphore must NOT be
        # acquired -- the generate_embeddings_with_timeout helper short-
        # circuits before entering the ``async with`` block.
        mock_sem.__aenter__.assert_not_called()


# --- Tests for hybrid search FTS/semantic error logging ---


@pytest.mark.asyncio
async def test_hybrid_search_logs_fts_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Verify warning logged when FTS fails but semantic succeeds."""
    with (
        patch('app.tools.search.hybrid.settings') as mock_settings,
        patch('app.tools.search.ranking.settings', mock_settings),
        patch('app.tools.search.hybrid.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
        patch('app.tools.search.hybrid.ensure_repositories', new_callable=AsyncMock) as mock_repos,
        patch('app.tools.search.hybrid.fts_search_raw', new_callable=AsyncMock) as mock_fts,
        patch('app.tools.search.hybrid.semantic_search_raw', new_callable=AsyncMock) as mock_semantic,
        patch('app.tools.search.hybrid.apply_reranking', new_callable=AsyncMock) as mock_rerank,
    ):
        mock_settings.hybrid_search.enabled = True
        mock_settings.hybrid_search.rrf_k = 60
        mock_settings.hybrid_search.rrf_overfetch = 2
        mock_settings.hybrid_search.fts_or_threshold = 4
        mock_settings.storage.backend_type = 'sqlite'
        mock_settings.fts.enabled = True
        mock_settings.semantic_search.enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.embedding.provider = 'ollama'
        mock_settings.search.truncation_length = 150

        # FTS fails, semantic succeeds
        mock_fts.side_effect = Exception('FTS index corrupted')
        mock_semantic.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text',
              'visibility': 'private', 'is_owner': 1}],
            None,
        )
        mock_rerank.return_value = [
            {'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text',
             'scores': {'rrf': 0.5}, 'visibility': 'private', 'is_owner': 1},
        ]
        mock_repos.return_value.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.return_value.images.get_images_for_context = AsyncMock(return_value=[])

        from app.tools.search.hybrid import hybrid_search_context

        with caplog.at_level(logging.WARNING, logger='app.tools.search'):
            await hybrid_search_context(query='test query')

        assert any('FTS sub-search failed' in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_hybrid_search_logs_semantic_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Verify warning logged when semantic fails but FTS succeeds."""
    with (
        patch('app.tools.search.hybrid.settings') as mock_settings,
        patch('app.tools.search.ranking.settings', mock_settings),
        patch('app.tools.search.hybrid.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
        patch('app.tools.search.hybrid.ensure_repositories', new_callable=AsyncMock) as mock_repos,
        patch('app.tools.search.hybrid.fts_search_raw', new_callable=AsyncMock) as mock_fts,
        patch('app.tools.search.hybrid.semantic_search_raw', new_callable=AsyncMock) as mock_semantic,
        patch('app.tools.search.hybrid.apply_reranking', new_callable=AsyncMock) as mock_rerank,
    ):
        mock_settings.hybrid_search.enabled = True
        mock_settings.hybrid_search.rrf_k = 60
        mock_settings.hybrid_search.rrf_overfetch = 2
        mock_settings.hybrid_search.fts_or_threshold = 4
        mock_settings.storage.backend_type = 'sqlite'
        mock_settings.fts.enabled = True
        mock_settings.semantic_search.enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.embedding.provider = 'ollama'
        mock_settings.search.truncation_length = 150

        # FTS succeeds, semantic fails
        mock_fts.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': '{"key": "val"}', 'source': 'agent',
              'content_type': 'text', 'visibility': 'private', 'is_owner': 1}],
            None,
        )
        mock_semantic.side_effect = Exception('Ollama unreachable')
        mock_rerank.return_value = [
            {'id': 1, 'text_content': 'test', 'metadata': {'key': 'val'}, 'source': 'agent', 'content_type': 'text',
             'scores': {'rrf': 0.5}, 'visibility': 'private', 'is_owner': 1},
        ]
        mock_repos.return_value.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.return_value.images.get_images_for_context = AsyncMock(return_value=[])

        from app.tools.search.hybrid import hybrid_search_context

        with caplog.at_level(logging.WARNING, logger='app.tools.search'):
            await hybrid_search_context(query='test query')

        assert any('semantic sub-search failed' in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_hybrid_search_no_warning_on_success(caplog: pytest.LogCaptureFixture) -> None:
    """Verify no warning logged when both sub-searches succeed."""
    with (
        patch('app.tools.search.hybrid.settings') as mock_settings,
        patch('app.tools.search.ranking.settings', mock_settings),
        patch('app.tools.search.hybrid.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
        patch('app.tools.search.hybrid.ensure_repositories', new_callable=AsyncMock) as mock_repos,
        patch('app.tools.search.hybrid.fts_search_raw', new_callable=AsyncMock) as mock_fts,
        patch('app.tools.search.hybrid.semantic_search_raw', new_callable=AsyncMock) as mock_semantic,
        patch('app.tools.search.hybrid.apply_reranking', new_callable=AsyncMock) as mock_rerank,
    ):
        mock_settings.hybrid_search.enabled = True
        mock_settings.hybrid_search.rrf_k = 60
        mock_settings.hybrid_search.rrf_overfetch = 2
        mock_settings.hybrid_search.fts_or_threshold = 4
        mock_settings.storage.backend_type = 'sqlite'
        mock_settings.fts.enabled = True
        mock_settings.semantic_search.enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.embedding.provider = 'ollama'
        mock_settings.search.truncation_length = 150

        mock_fts.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text',
              'visibility': 'private', 'is_owner': 1}],
            None,
        )
        mock_semantic.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text',
              'visibility': 'private', 'is_owner': 1}],
            None,
        )
        mock_rerank.return_value = [
            {'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text',
             'scores': {'rrf': 0.5}, 'visibility': 'private', 'is_owner': 1},
        ]
        mock_repos.return_value.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.return_value.images.get_images_for_context = AsyncMock(return_value=[])

        from app.tools.search.hybrid import hybrid_search_context

        with caplog.at_level(logging.WARNING, logger='app.tools.search'):
            await hybrid_search_context(query='test query')

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any('sub-search failed' in r.message for r in warning_records)


# --- Tests for EMBEDDING_MAX_CONCURRENT setting ---


def test_embedding_max_concurrent_setting_default() -> None:
    """Test that EMBEDDING_MAX_CONCURRENT has correct default value."""
    from app.settings.embedding import EmbeddingSettings

    settings = EmbeddingSettings()
    assert settings.max_concurrent == 3


def test_embedding_max_concurrent_setting_bounds() -> None:
    """Test that EMBEDDING_MAX_CONCURRENT enforces bounds."""
    from pydantic import ValidationError

    from app.settings.embedding import EmbeddingSettings

    with pytest.raises(ValidationError):
        EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 0})

    with pytest.raises(ValidationError):
        EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 21})

    # Valid boundary values
    settings_min = EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 1})
    assert settings_min.max_concurrent == 1

    settings_max = EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 20})
    assert settings_max.max_concurrent == 20


# --- Tests for generate_embeddings_with_timeout helper ---


@pytest.mark.asyncio
async def testgenerate_embeddings_with_timeout_returns_none_when_no_provider() -> None:
    """Verify helper returns None when embedding provider is not configured."""
    with (
        patch('app.tools._generation.get_embedding_provider', return_value=None),
    ):
        from app.tools._generation import generate_embeddings_with_timeout

        result = await generate_embeddings_with_timeout('test text')
        assert result is None


@pytest.mark.asyncio
async def testgenerate_embeddings_with_timeout_success() -> None:
    """Verify helper returns the single embedding when chunking is disabled."""
    provider = MagicMock()
    provider.embed_query = AsyncMock(return_value=[0.1, 0.2])

    with (
        patch('app.tools._generation.get_embedding_provider', return_value=provider),
        patch('app.tools._generation.get_chunking_service', return_value=None),
        patch('app.tools._generation.compute_embedding_total_timeout', return_value=999.0),
        patch('app.tools._generation.settings') as mock_settings,
    ):
        mock_settings.embedding.max_concurrent = 3

        import app.tools._generation as generation_module

        original_semaphore = generation_module._embedding_semaphore
        generation_module._reset_embedding_semaphore()

        try:
            from app.tools._generation import generate_embeddings_with_timeout

            result = await generate_embeddings_with_timeout('test text')
            assert result == [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=9)]
        finally:
            generation_module._embedding_semaphore = original_semaphore


@pytest.mark.asyncio
async def testgenerate_embeddings_with_timeout_raises_on_timeout() -> None:
    """Verify helper raises ToolError when the single embedding times out."""
    from fastmcp.exceptions import ToolError

    async def slow_embedding(_text: str) -> list[float]:
        await asyncio.sleep(10.0)
        return [0.0]

    provider = MagicMock()
    provider.embed_query = slow_embedding

    with (
        patch('app.tools._generation.get_embedding_provider', return_value=provider),
        patch('app.tools._generation.get_chunking_service', return_value=None),
        patch('app.tools._generation.compute_embedding_total_timeout', return_value=0.05),
        patch('app.tools._generation.settings') as mock_settings,
    ):
        mock_settings.embedding.max_concurrent = 3

        import app.tools._generation as generation_module

        original_semaphore = generation_module._embedding_semaphore
        generation_module._reset_embedding_semaphore()

        try:
            from app.tools._generation import generate_embeddings_with_timeout

            with pytest.raises(ToolError, match=r'^Embedding generation exceeded total timeout \(0s\)\. '):
                await generate_embeddings_with_timeout('test text')
        finally:
            generation_module._embedding_semaphore = original_semaphore
