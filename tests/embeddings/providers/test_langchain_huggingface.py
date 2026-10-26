"""Tests for the HuggingFace embedding provider in app.embeddings.providers.langchain_huggingface.

These tests use mocks to avoid requiring actual provider dependencies.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestHuggingFaceEmbeddingProvider:
    """Tests for HuggingFaceEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_with_token(self) -> MagicMock:
        """Create mock settings with API token."""
        mock = MagicMock()
        mock.embedding.model = 'sentence-transformers/all-MiniLM-L6-v2'
        mock.embedding.dim = 384
        mock.embedding.huggingface_api_key = MagicMock()
        mock.embedding.huggingface_api_key.get_secret_value.return_value = 'test-token'
        return mock

    @pytest.fixture
    def mock_settings_without_token(self) -> MagicMock:
        """Create mock settings without API token."""
        mock = MagicMock()
        mock.embedding.model = 'sentence-transformers/all-MiniLM-L6-v2'
        mock.embedding.dim = 384
        mock.embedding.huggingface_api_key = None
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_token(
        self,
        mock_settings_without_token: MagicMock,
    ) -> None:
        """Test initialization fails without API token."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_huggingface': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_huggingface.get_settings',
                return_value=mock_settings_without_token,
            ),
        ):
            from app.embeddings.providers.langchain_huggingface import HuggingFaceEmbeddingProvider

            provider = HuggingFaceEmbeddingProvider()

            with pytest.raises(ValueError, match='HUGGINGFACEHUB_API_TOKEN is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_token(
        self,
        mock_settings_with_token: MagicMock,
    ) -> None:
        """Test successful initialization with API token."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.HuggingFaceEndpointEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_huggingface': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_huggingface.get_settings',
                return_value=mock_settings_with_token,
            ),
        ):
            from app.embeddings.providers.langchain_huggingface import HuggingFaceEmbeddingProvider

            provider = HuggingFaceEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'huggingface'

    @pytest.mark.asyncio
    async def test_is_available_raises_configuration_error_on_4xx(
        self,
        mock_settings_with_token: MagicMock,
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
            side_effect=FakeClientError('invalid token', status_code=401),
        )

        with patch(
            'app.embeddings.providers.langchain_huggingface.get_settings',
            return_value=mock_settings_with_token,
        ):
            from app.embeddings.providers.langchain_huggingface import HuggingFaceEmbeddingProvider

            provider = HuggingFaceEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ConfigurationError, match='client error'):
                await provider.is_available()

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_transient_error(
        self,
        mock_settings_with_token: MagicMock,
    ) -> None:
        """is_available returns False for transient errors (no status_code)."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(
            side_effect=TimeoutError('Connection timed out'),
        )

        with patch(
            'app.embeddings.providers.langchain_huggingface.get_settings',
            return_value=mock_settings_with_token,
        ):
            from app.embeddings.providers.langchain_huggingface import HuggingFaceEmbeddingProvider

            provider = HuggingFaceEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_embed_documents_logs_the_request_at_debug(
        self,
        mock_settings_with_token: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One embed_documents call embeds one batch of a document, so its line is DEBUG.

        The batching loop logs each document's start and completion at INFO; an
        INFO line per request would repeat once per batch at the default level.
        """
        caplog.set_level(logging.DEBUG, logger='app.embeddings.providers.langchain_huggingface')
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])

        with patch(
            'app.embeddings.providers.langchain_huggingface.get_settings',
            return_value=mock_settings_with_token,
        ):
            from app.embeddings.providers.langchain_huggingface import HuggingFaceEmbeddingProvider

            provider = HuggingFaceEmbeddingProvider()
            provider._embeddings = mock_embeddings

            await provider.embed_documents(['text1', 'text2'])

        records = [record for record in caplog.records if record.name == 'app.embeddings.providers.langchain_huggingface']
        assert [(record.levelno, record.getMessage()) for record in records] == [
            (logging.DEBUG, 'Generated 2 embeddings for 2 texts'),
        ]
