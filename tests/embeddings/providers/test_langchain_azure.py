"""Tests for the Azure OpenAI embedding provider in app.embeddings.providers.langchain_azure.

These tests use mocks to avoid requiring actual provider dependencies.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestAzureEmbeddingProvider:
    """Tests for AzureEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_incomplete(self) -> MagicMock:
        """Create mock settings with missing Azure settings."""
        mock = MagicMock()
        mock.embedding.dim = 1536
        mock.embedding.azure_openai_api_key = None
        mock.embedding.azure_openai_endpoint = None
        mock.embedding.azure_openai_deployment_name = None
        mock.embedding.azure_openai_api_version = '2024-02-01'
        return mock

    @pytest.fixture
    def mock_settings_complete(self) -> MagicMock:
        """Create mock settings with all Azure settings."""
        mock = MagicMock()
        mock.embedding.dim = 1536
        mock.embedding.azure_openai_api_key = MagicMock()
        mock.embedding.azure_openai_api_key.get_secret_value.return_value = 'test-api-key'
        mock.embedding.azure_openai_endpoint = 'https://test.openai.azure.com'
        mock.embedding.azure_openai_deployment_name = 'test-deployment'
        mock.embedding.azure_openai_api_version = '2024-02-01'
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_api_key(
        self,
        mock_settings_incomplete: MagicMock,
    ) -> None:
        """Test initialization fails without API key."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_azure.get_settings',
                return_value=mock_settings_incomplete,
            ),
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()

            with pytest.raises(ValueError, match='AZURE_OPENAI_API_KEY is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_raises_without_endpoint(
        self,
        mock_settings_incomplete: MagicMock,
    ) -> None:
        """Test initialization fails without endpoint."""
        mock_settings_incomplete.embedding.azure_openai_api_key = MagicMock()
        mock_settings_incomplete.embedding.azure_openai_api_key.get_secret_value.return_value = 'test-key'
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_azure.get_settings',
                return_value=mock_settings_incomplete,
            ),
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()

            with pytest.raises(ValueError, match='AZURE_OPENAI_ENDPOINT is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_complete_settings(
        self,
        mock_settings_complete: MagicMock,
    ) -> None:
        """Test successful initialization with complete settings."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.AzureOpenAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_azure.get_settings',
                return_value=mock_settings_complete,
            ),
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'azure'

    @pytest.mark.asyncio
    async def test_is_available_raises_configuration_error_on_4xx(
        self,
        mock_settings_complete: MagicMock,
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
            side_effect=FakeClientError('deployment not found', status_code=404),
        )

        with patch(
            'app.embeddings.providers.langchain_azure.get_settings',
            return_value=mock_settings_complete,
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ConfigurationError, match='client error'):
                await provider.is_available()

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_transient_error(
        self,
        mock_settings_complete: MagicMock,
    ) -> None:
        """is_available returns False for transient errors (no status_code)."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(
            side_effect=TimeoutError('Connection timed out'),
        )

        with patch(
            'app.embeddings.providers.langchain_azure.get_settings',
            return_value=mock_settings_complete,
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_embed_documents_logs_the_request_at_debug(
        self,
        mock_settings_complete: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One embed_documents call embeds one batch of a document, so its line is DEBUG.

        The batching loop logs each document's start and completion at INFO; an
        INFO line per request would repeat once per batch at the default level.
        """
        caplog.set_level(logging.DEBUG, logger='app.embeddings.providers.langchain_azure')
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])

        with patch(
            'app.embeddings.providers.langchain_azure.get_settings',
            return_value=mock_settings_complete,
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()
            provider._embeddings = mock_embeddings

            await provider.embed_documents(['text1', 'text2'])

        records = [record for record in caplog.records if record.name == 'app.embeddings.providers.langchain_azure']
        assert [(record.levelno, record.getMessage()) for record in records] == [
            (logging.DEBUG, 'Generated 2 embeddings for 2 texts'),
        ]
