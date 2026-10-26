"""Tests for the Ollama embedding provider in app.embeddings.providers.langchain_ollama.

These tests use mocks to avoid requiring actual provider dependencies.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestOllamaEmbeddingProvider:
    """Tests for OllamaEmbeddingProvider."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for Ollama provider."""
        mock = MagicMock()
        mock.embedding.model = 'test-model'
        mock.ollama.host = 'http://localhost:11434'
        mock.embedding.dim = 768
        mock.embedding.ollama_truncate = False
        mock.embedding.ollama_num_ctx = 4096
        return mock

    @pytest.fixture
    def mock_ollama_embeddings(self) -> MagicMock:
        """Create mock OllamaEmbeddings class."""
        mock = MagicMock()
        mock.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock.aembed_documents = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        return mock

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        mock_settings: MagicMock,
        mock_ollama_embeddings: MagicMock,
    ) -> None:
        """Test successful initialization."""
        mock_langchain = MagicMock()
        mock_langchain.OllamaEmbeddings = MagicMock(return_value=mock_ollama_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_ollama': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_ollama.get_settings',
                return_value=mock_settings,
            ),
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'ollama'

    @pytest.mark.asyncio
    async def test_embed_query_success(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test successful embedding query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.embed_query('test text')

            assert len(result) == 768
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_query_passes_text_to_backend_unchanged(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """The provider hands its input text to the backend verbatim.

        EMBEDDING_QUERY_INSTRUCTION is applied at the search-tool call site,
        never inside the provider layer: the same embed_query method also
        embeds whole documents on the store path when chunking is disabled,
        so a provider-level prefix would leak into document embeddings.
        """
        mock_settings.embedding.query_instruction = 'Instruct: retrieve passages\nQuery:'
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            await provider.embed_query('bare query text')

            mock_embeddings.aembed_query.assert_awaited_once_with('bare query text')

    @pytest.mark.asyncio
    async def test_embed_query_dimension_validation(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test dimension validation in embed_query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 512)  # Wrong dimension

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ValueError, match='Dimension mismatch'):
                await provider.embed_query('test')

    @pytest.mark.asyncio
    async def test_embed_query_not_initialized_raises_error(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test embed_query raises error when not initialized."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()

            with pytest.raises(RuntimeError, match='Provider not initialized'):
                await provider.embed_query('test')

    @pytest.mark.asyncio
    async def test_embed_documents_dimension_validation(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test dimension validation in embed_documents."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(
            return_value=[[0.1] * 768, [0.2] * 512],  # Second has wrong dimension
        )

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ValueError, match='Embedding 1 dimension mismatch'):
                await provider.embed_documents(['text1', 'text2'])

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_initialized(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test is_available returns False when provider not initialized."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            result = await provider.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_working(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test is_available returns True when provider works."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test is_available returns False when API fails."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(side_effect=Exception('Connection failed'))

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_raises_configuration_error_on_4xx(
        self,
        mock_settings: MagicMock,
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
            side_effect=FakeClientError('invalid model', status_code=400),
        )

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ConfigurationError, match='client error'):
                await provider.is_available()

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_transient_error(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """is_available returns False for transient errors (no status_code)."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(
            side_effect=TimeoutError('Connection timed out'),
        )

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()
            assert result is False

    def test_get_dimension_returns_configured_value(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test get_dimension returns the configured dimension."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()

            assert provider.get_dimension() == 768

    @pytest.mark.asyncio
    async def test_shutdown_clears_embeddings(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test shutdown properly clears the embeddings instance."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = MagicMock()  # Simulate initialized state

            await provider.shutdown()

            assert provider._embeddings is None

    def test_convert_to_python_floats(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test numpy type conversion utility."""
        # Create mock numpy-like objects
        class MockNumpyFloat:
            def __init__(self, val: float) -> None:
                self._val = val

            def item(self) -> float:
                return self._val

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()

            # Test with mix of numpy-like and Python floats
            input_data = [MockNumpyFloat(1.5), 2.5, MockNumpyFloat(3.5)]
            result = provider._convert_to_python_floats(input_data)

            assert result == [1.5, 2.5, 3.5]
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_truncate_not_passed_to_embeddings(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test that truncate is NOT passed to OllamaEmbeddings (library doesn't support it)."""
        mock_settings.embedding.ollama_truncate = False

        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.OllamaEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_ollama': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_ollama.get_settings',
                return_value=mock_settings,
            ),
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            await provider.initialize()

            # Verify truncate was NOT passed (langchain-ollama doesn't support it)
            call_kwargs = mock_langchain.OllamaEmbeddings.call_args[1]
            assert 'truncate' not in call_kwargs
            assert call_kwargs['model'] == mock_settings.embedding.model
            assert call_kwargs['base_url'] == mock_settings.ollama.host

    @pytest.mark.asyncio
    async def test_text_length_validation_when_truncate_disabled(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test that text length is validated when EMBEDDING_OLLAMA_TRUNCATE=false."""
        mock_settings.embedding.ollama_truncate = False
        mock_settings.embedding.ollama_num_ctx = 1000  # Small context for testing
        mock_settings.embedding.model = 'qwen3-embedding:0.6b'

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            # Long text that exceeds estimated context (~1666 tokens, exceeds 1000)
            long_text = 'a' * 5000

            with pytest.raises(ValueError, match='may exceed context window'):
                await provider.embed_query(long_text)

    @pytest.mark.asyncio
    async def test_no_validation_when_truncate_enabled(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test that text length is NOT validated when EMBEDDING_OLLAMA_TRUNCATE=true."""
        mock_settings.embedding.ollama_truncate = True
        mock_settings.embedding.ollama_num_ctx = 1000
        mock_settings.embedding.model = 'qwen3-embedding:0.6b'

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            # Long text - should NOT raise error when truncate=True
            long_text = 'a' * 5000
            result = await provider.embed_query(long_text)
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_documents_validation_when_truncate_disabled(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test that embed_documents validates all texts when EMBEDDING_OLLAMA_TRUNCATE=false."""
        mock_settings.embedding.ollama_truncate = False
        mock_settings.embedding.ollama_num_ctx = 1000
        mock_settings.embedding.model = 'qwen3-embedding:0.6b'

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 768])

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            # Second text exceeds context
            texts = ['short text', 'a' * 5000]

            with pytest.raises(ValueError, match='Text 1 validation failed'):
                await provider.embed_documents(texts)

    @pytest.mark.asyncio
    async def test_embed_documents_logs_the_request_at_debug(
        self,
        mock_settings: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One embed_documents call embeds one batch of a document, so its line is DEBUG.

        The batching loop logs each document's start and completion at INFO; an
        INFO line per request would repeat once per batch at the default level.
        """
        caplog.set_level(logging.DEBUG, logger='app.embeddings.providers.langchain_ollama')
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            await provider.embed_documents(['text1', 'text2'])

        records = [record for record in caplog.records if record.name == 'app.embeddings.providers.langchain_ollama']
        assert [(record.levelno, record.getMessage()) for record in records] == [
            (logging.DEBUG, 'Generated 2 embeddings for 2 texts'),
        ]
