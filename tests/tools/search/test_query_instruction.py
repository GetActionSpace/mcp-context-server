"""Tests for the EMBEDDING_QUERY_INSTRUCTION query-side prefix.

The instruction is prepended verbatim to the text embedded for search queries
(semantic_search_context and the semantic leg of hybrid_search_context) and is
never applied to document embeddings on the store/update path. Unset and empty
values leave the text handed to the embedding provider byte-identical to the
bare query. Downstream consumers of the query text (the response echo and the
hybrid FTS leg) always receive the bare query.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.settings import get_settings
from tests.tools._generation_jobs import InMemoryGenerationJobs

QUERY_INSTRUCTION = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Isolate the settings singleton from env vars set inside these tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _refresh_search_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuild the settings singleton and refresh the binding of the module that reads the query instruction."""
    import app.tools.search.legs as search_legs

    get_settings.cache_clear()
    monkeypatch.setattr(search_legs, 'settings', get_settings())


def _mock_embedding_provider(dim: int = 1024) -> MagicMock:
    """Create an embedding provider mock whose embed_query records its input text."""
    provider = MagicMock()
    provider.embed_query = AsyncMock(return_value=[0.1] * dim)
    return provider


def _mock_repos() -> MagicMock:
    """Create a repository container mock sufficient for empty search results."""
    repos = MagicMock()
    repos.embeddings.search = AsyncMock(return_value=([], {}))
    repos.tags.get_tags_for_context = AsyncMock(return_value=[])
    repos.images.get_images_for_context = AsyncMock(return_value=[])
    repos.generation_jobs = InMemoryGenerationJobs()
    return repos


@pytest.mark.asyncio
async def test_semantic_search_unset_instruction_sends_bare_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """With EMBEDDING_QUERY_INSTRUCTION unset the provider receives the bare query."""
    monkeypatch.delenv('EMBEDDING_QUERY_INSTRUCTION', raising=False)
    _refresh_search_settings(monkeypatch)
    provider = _mock_embedding_provider()

    with (
        patch('app.tools.search.semantic.get_embedding_provider', return_value=provider),
        patch('app.tools.search.semantic.get_reranking_provider', return_value=None),
        patch('app.tools.search.semantic.ensure_repositories', new_callable=AsyncMock, return_value=_mock_repos()),
    ):
        from app.tools.search.semantic import semantic_search_context

        result = await semantic_search_context(query='find the quarterly report')

    provider.embed_query.assert_awaited_once_with('find the quarterly report')
    assert result['query'] == 'find the quarterly report'


@pytest.mark.asyncio
async def test_semantic_search_empty_instruction_sends_bare_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty EMBEDDING_QUERY_INSTRUCTION behaves exactly like an unset one."""
    monkeypatch.setenv('EMBEDDING_QUERY_INSTRUCTION', '')
    _refresh_search_settings(monkeypatch)
    provider = _mock_embedding_provider()

    with (
        patch('app.tools.search.semantic.get_embedding_provider', return_value=provider),
        patch('app.tools.search.semantic.get_reranking_provider', return_value=None),
        patch('app.tools.search.semantic.ensure_repositories', new_callable=AsyncMock, return_value=_mock_repos()),
    ):
        from app.tools.search.semantic import semantic_search_context

        await semantic_search_context(query='find the quarterly report')

    provider.embed_query.assert_awaited_once_with('find the quarterly report')


@pytest.mark.asyncio
async def test_semantic_search_set_instruction_prefixes_embedded_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set instruction is prepended verbatim to the embedded text; the response echoes the bare query."""
    monkeypatch.setenv('EMBEDDING_QUERY_INSTRUCTION', QUERY_INSTRUCTION)
    _refresh_search_settings(monkeypatch)
    provider = _mock_embedding_provider()

    with (
        patch('app.tools.search.semantic.get_embedding_provider', return_value=provider),
        patch('app.tools.search.semantic.get_reranking_provider', return_value=None),
        patch('app.tools.search.semantic.ensure_repositories', new_callable=AsyncMock, return_value=_mock_repos()),
    ):
        from app.tools.search.semantic import semantic_search_context

        result = await semantic_search_context(query='find the quarterly report')

    provider.embed_query.assert_awaited_once_with(QUERY_INSTRUCTION + 'find the quarterly report')
    assert result['query'] == 'find the quarterly report'


@pytest.mark.asyncio
async def test_hybrid_semantic_leg_prefixes_embedded_text_and_fts_leg_stays_bare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hybrid semantic leg embeds the prefixed text while the FTS leg keeps the bare query."""
    monkeypatch.setenv('EMBEDDING_QUERY_INSTRUCTION', QUERY_INSTRUCTION)
    _refresh_search_settings(monkeypatch)
    provider = _mock_embedding_provider()

    with (
        patch('app.tools.search.hybrid.get_embedding_provider', return_value=provider),
        patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
        patch('app.tools.search.hybrid.ensure_repositories', new_callable=AsyncMock, return_value=_mock_repos()),
        patch('app.tools.search.hybrid.fts_search_raw', new_callable=AsyncMock, return_value=([], None)) as mock_fts,
    ):
        from app.tools.search.hybrid import hybrid_search_context

        result = await hybrid_search_context(query='alpha beta')

    provider.embed_query.assert_awaited_once_with(QUERY_INSTRUCTION + 'alpha beta')
    assert mock_fts.await_args is not None
    assert mock_fts.await_args.kwargs['query'] == 'alpha beta'
    assert result['query'] == 'alpha beta'


@pytest.mark.asyncio
async def test_store_path_document_embedding_never_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document embeddings on the store/update path receive the bare text even with the instruction set.

    With chunking disabled the store path embeds the whole document through the
    provider's embed_query method, so this test pins that the instruction never
    leaks into document embeddings through that shared provider method.
    """
    monkeypatch.setenv('EMBEDDING_QUERY_INSTRUCTION', QUERY_INSTRUCTION)
    import app.tools._generation as generation_module

    get_settings.cache_clear()
    monkeypatch.setattr(generation_module, 'settings', get_settings())
    provider = _mock_embedding_provider()

    with (
        patch('app.tools._generation.get_embedding_provider', return_value=provider),
        patch('app.tools._generation.get_chunking_service', return_value=None),
    ):
        from app.tools._generation import generate_embeddings_with_timeout

        chunk_embeddings = await generate_embeddings_with_timeout('stored document text')

    provider.embed_query.assert_awaited_once_with('stored document text')
    assert chunk_embeddings is not None
    assert len(chunk_embeddings) == 1
