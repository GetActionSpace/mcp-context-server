"""Tests for the OpenAI embedding provider in app.embeddings.providers.langchain_openai.

These tests use mocks to avoid requiring actual provider dependencies.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_with_key(self) -> MagicMock:
        """Create mock settings with API key."""
        mock = MagicMock()
        mock.embedding.model = 'text-embedding-3-small'
        mock.embedding.dim = 1536
        mock.embedding.openai_api_key = MagicMock()
        mock.embedding.openai_api_key.get_secret_value.return_value = 'test-api-key'
        mock.embedding.openai_api_base = None
        mock.embedding.openai_organization = None
        return mock

    @pytest.fixture
    def mock_settings_without_key(self) -> MagicMock:
        """Create mock settings without API key."""
        mock = MagicMock()
        mock.embedding.model = 'text-embedding-3-small'
        mock.embedding.dim = 1536
        mock.embedding.openai_api_key = None
        mock.embedding.openai_api_base = None
        mock.embedding.openai_organization = None
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_api_key(
        self,
        mock_settings_without_key: MagicMock,
    ) -> None:
        """Test initialization fails without API key."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_openai.get_settings',
                return_value=mock_settings_without_key,
            ),
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()

            with pytest.raises(ValueError, match='OPENAI_API_KEY is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_api_key(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful initialization with API key."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.OpenAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_openai.get_settings',
                return_value=mock_settings_with_key,
            ),
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'openai'

    def test_provider_name(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test provider_name property returns 'openai'."""
        with patch(
            'app.embeddings.providers.langchain_openai.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()

            assert provider.provider_name == 'openai'

    @pytest.mark.asyncio
    async def test_embed_query_passes_text_to_backend_unchanged(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """The provider hands its input text to the backend verbatim.

        EMBEDDING_QUERY_INSTRUCTION is applied at the search-tool call site,
        never inside the provider layer: cloud models such as
        text-embedding-3-small need no prefix, and the same embed_query method
        also embeds whole documents on the store path when chunking is disabled.
        """
        mock_settings_with_key.embedding.query_instruction = 'Instruct: retrieve passages\nQuery:'
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)

        with patch(
            'app.embeddings.providers.langchain_openai.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()
            provider._embeddings = mock_embeddings

            await provider.embed_query('bare query text')

            mock_embeddings.aembed_query.assert_awaited_once_with('bare query text')

    @pytest.mark.asyncio
    async def test_is_available_raises_configuration_error_on_4xx(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """is_available raises ConfigurationError on HTTP 4xx client errors."""
        from app.errors import ConfigurationError

        class FakeClientError(Exception):
            """Exception with status_code attribute simulating an HTTP client error."""

            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(
            side_effect=FakeClientError('invalid api key', status_code=401),
        )

        with patch(
            'app.embeddings.providers.langchain_openai.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ConfigurationError, match='client error'):
                await provider.is_available()

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_transient_error(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """is_available returns False for transient errors (no status_code)."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(
            side_effect=TimeoutError('Connection timed out'),
        )

        with patch(
            'app.embeddings.providers.langchain_openai.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_embed_documents_logs_the_request_at_debug(
        self,
        mock_settings_with_key: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One embed_documents call embeds one batch of a document, so its line is DEBUG.

        The batching loop logs each document's start and completion at INFO; an
        INFO line per request would repeat once per batch at the default level.
        """
        caplog.set_level(logging.DEBUG, logger='app.embeddings.providers.langchain_openai')
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])

        with patch(
            'app.embeddings.providers.langchain_openai.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()
            provider._embeddings = mock_embeddings

            await provider.embed_documents(['text1', 'text2'])

        records = [record for record in caplog.records if record.name == 'app.embeddings.providers.langchain_openai']
        assert [(record.levelno, record.getMessage()) for record in records] == [
            (logging.DEBUG, 'Generated 2 embeddings for 2 texts'),
        ]
