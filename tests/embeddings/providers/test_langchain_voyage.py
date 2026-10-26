"""Tests for the Voyage AI embedding provider in app.embeddings.providers.langchain_voyage.

These tests use mocks to avoid requiring actual provider dependencies.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestVoyageEmbeddingProvider:
    """Tests for VoyageEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_with_key(self) -> MagicMock:
        """Create mock settings with API key."""
        mock = MagicMock()
        mock.embedding.model = 'voyage-3'
        mock.embedding.dim = 1024
        mock.embedding.voyage_api_key = MagicMock()
        mock.embedding.voyage_api_key.get_secret_value.return_value = 'test-voyage-key'
        mock.embedding.voyage_truncation = False  # New default: disabled
        mock.embedding.voyage_batch_size = 7
        return mock

    @pytest.fixture
    def mock_settings_without_key(self) -> MagicMock:
        """Create mock settings without API key."""
        mock = MagicMock()
        mock.embedding.model = 'voyage-3'
        mock.embedding.dim = 1024
        mock.embedding.voyage_api_key = None
        mock.embedding.voyage_truncation = False  # New default: disabled
        mock.embedding.voyage_batch_size = 7
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_api_key(
        self,
        mock_settings_without_key: MagicMock,
    ) -> None:
        """Test initialization fails without API key."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_voyageai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_voyage.get_settings',
                return_value=mock_settings_without_key,
            ),
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()

            with pytest.raises(ValueError, match='VOYAGE_API_KEY is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_api_key(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful initialization with API key."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.VoyageAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_voyageai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_voyage.get_settings',
                return_value=mock_settings_with_key,
            ),
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'voyage'

    @pytest.mark.asyncio
    async def test_embed_query_dimension_validation(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test dimension validation in embed_query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 512)  # Wrong dimension

        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ValueError, match='Dimension mismatch'):
                await provider.embed_query('test')

    @pytest.mark.asyncio
    async def test_embed_query_success(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful embedding query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.embed_query('test text')

            assert len(result) == 1024
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_documents_success(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful batch embedding."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(
            return_value=[[0.1] * 1024, [0.2] * 1024],
        )

        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.embed_documents(['text1', 'text2'])

            assert len(result) == 2
            assert len(result[0]) == 1024
            assert len(result[1]) == 1024

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_initialized(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test is_available returns False when provider not initialized."""
        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            result = await provider.is_available()

            assert result is False

    def test_get_dimension_returns_configured_value(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test get_dimension returns the configured dimension."""
        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()

            assert provider.get_dimension() == 1024

    @pytest.mark.asyncio
    async def test_truncation_false_passed_by_default(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test truncation=False (new default) is passed to VoyageAIEmbeddings."""
        mock_settings_with_key.embedding.voyage_truncation = False

        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.VoyageAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_voyageai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_voyage.get_settings',
                return_value=mock_settings_with_key,
            ),
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            await provider.initialize()

            # Verify truncation was passed in kwargs
            call_kwargs = mock_langchain.VoyageAIEmbeddings.call_args[1]
            assert 'truncation' in call_kwargs
            assert call_kwargs['truncation'] is False

    @pytest.mark.asyncio
    async def test_truncation_true_passed_when_enabled(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test truncation=True is passed to VoyageAIEmbeddings when enabled."""
        mock_settings_with_key.embedding.voyage_truncation = True

        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.VoyageAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_voyageai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_voyage.get_settings',
                return_value=mock_settings_with_key,
            ),
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            await provider.initialize()

            # Verify truncation was passed in kwargs
            call_kwargs = mock_langchain.VoyageAIEmbeddings.call_args[1]
            assert 'truncation' in call_kwargs
            assert call_kwargs['truncation'] is True

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
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
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
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
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
        caplog.set_level(logging.DEBUG, logger='app.embeddings.providers.langchain_voyage')
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 1024, [0.2] * 1024])

        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            provider._embeddings = mock_embeddings

            await provider.embed_documents(['text1', 'text2'])

        records = [record for record in caplog.records if record.name == 'app.embeddings.providers.langchain_voyage']
        assert [(record.levelno, record.getMessage()) for record in records] == [
            (logging.DEBUG, 'Generated 2 embeddings for 2 texts'),
        ]
