"""
OpenAI embedding provider using LangChain integration.

This provider uses langchain-openai package for OpenAI's embedding API.
"""


import logging
from typing import Any

from app.embeddings.retry import with_retry_and_timeout
from app.embeddings.tracing import traced_embedding
from app.errors import ConfigurationError
from app.errors import is_client_error
from app.settings import get_settings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider using LangChain integration.

    Implements the EmbeddingProvider protocol for OpenAI models.
    Uses LangChain's OpenAIEmbeddings for native async support.

    Environment Variables:
        EMBEDDING_PROVIDER: Must be 'openai'
        OPENAI_API_KEY: OpenAI API key (required)
        EMBEDDING_MODEL: Model name (default: text-embedding-3-small)
        EMBEDDING_DIM: Vector dimensions (default: 1536)
        OPENAI_API_BASE: Custom base URL for OpenAI-compatible APIs (optional)
        OPENAI_ORGANIZATION: Organization ID (optional)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.embedding.model
        self._dimension = settings.embedding.dim
        self._api_key = settings.embedding.openai_api_key
        self._api_base = settings.embedding.openai_api_base
        self._organization = settings.embedding.openai_organization
        self._embeddings: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain OpenAIEmbeddings client.

        Raises:
            ImportError: If langchain-openai is not installed
            ValueError: If API key is not configured
        """
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                'langchain-openai package required',
            ) from e

        if self._api_key is None:
            raise ValueError(
                'OPENAI_API_KEY is required for OpenAI embedding provider. '
                'Set the environment variable or use a different provider.',
            )

        # Build kwargs for OpenAIEmbeddings
        # Disable internal retry (max_retries=0) - universal wrapper handles retries
        # Disable internal timeout (timeout=None) - universal wrapper handles timeout
        kwargs: dict[str, Any] = {
            'model': self._model,
            'api_key': self._api_key.get_secret_value(),
            'max_retries': 0,
            'timeout': None,
        }

        if self._api_base:
            kwargs['base_url'] = self._api_base

        if self._organization:
            kwargs['organization'] = self._organization

        self._embeddings = OpenAIEmbeddings(**kwargs)
        logger.info(f'Initialized OpenAI embedding provider: {self._model}')

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._embeddings = None
        logger.info('OpenAI embedding provider shut down')

    @traced_embedding
    async def embed_query(self, text: str) -> list[float]:
        """Generate single embedding using async method.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If provider not initialized
            ValueError: If embedding dimension mismatch
        """
        if self._embeddings is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        async def _embed() -> list[Any]:
            result: list[Any] = await self._embeddings.aembed_query(text)
            return result

        embedding = await with_retry_and_timeout(_embed, f'{self.provider_name}_embed_query')
        embedding = self._convert_to_python_floats(embedding)

        # Key operational event: shows embedding generation worked
        logger.info(f'Generated query embedding: text_len={len(text)}, dim={len(embedding)}')

        # Validate dimension
        if len(embedding) != self._dimension:
            raise ValueError(
                f'Dimension mismatch: expected {self._dimension}, '
                f'got {len(embedding)}. Check EMBEDDING_DIM setting.',
            )

        return embedding

    @traced_embedding
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings using async method.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If provider not initialized
            ValueError: If any embedding dimension mismatch
        """
        if self._embeddings is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        async def _embed() -> list[list[Any]]:
            result: list[list[Any]] = await self._embeddings.aembed_documents(texts)
            return result

        embeddings = await with_retry_and_timeout(_embed, f'{self.provider_name}_embed_documents')

        # One provider request; the batching loop logs the document's outcome at INFO.
        logger.debug(f'Generated {len(embeddings)} embeddings for {len(texts)} texts')

        result: list[list[float]] = []
        for i, emb in enumerate(embeddings):
            emb = self._convert_to_python_floats(emb)
            if len(emb) != self._dimension:
                raise ValueError(
                    f'Embedding {i} dimension mismatch: '
                    f'expected {self._dimension}, got {len(emb)}',
                )
            result.append(emb)

        return result

    async def is_available(self) -> bool:
        """Check if OpenAI API is available.

        Returns:
            True if provider is ready to generate embeddings

        Raises:
            ConfigurationError: If the API returns a client error (4xx) indicating
                a permanent configuration problem (e.g., invalid OPENAI_API_KEY or
                EMBEDDING_MODEL)
        """
        if self._embeddings is None:
            return False

        try:
            await self._embeddings.aembed_query('test')
        except Exception as e:
            if is_client_error(e):
                raise ConfigurationError(
                    f'OpenAI API returned a client error during availability check: {e}. '
                    'This indicates a permanent configuration problem (e.g., invalid '
                    'OPENAI_API_KEY or EMBEDDING_MODEL). '
                    'Fix: Check the error message above and correct the configuration.',
                ) from e
            logger.warning(f'OpenAI embedding not available: {e}')
            return False
        else:
            return True

    def get_dimension(self) -> int:
        """Return configured embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'openai'

    @staticmethod
    def _convert_to_python_floats(embedding: list[Any]) -> list[float]:
        """Convert numpy types to Python float if needed."""
        return [
            x.item() if hasattr(x, 'item') else float(x)
            for x in embedding
        ]
