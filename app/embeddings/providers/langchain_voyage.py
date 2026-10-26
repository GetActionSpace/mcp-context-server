"""
Voyage AI embedding provider using LangChain integration.

This provider uses langchain-voyageai package for Voyage AI's embedding API.
"""


import logging
from typing import Any

from app.embeddings.retry import with_retry_and_timeout
from app.embeddings.tracing import traced_embedding
from app.errors import ConfigurationError
from app.errors import is_client_error
from app.settings import get_settings

logger = logging.getLogger(__name__)


class VoyageEmbeddingProvider:
    """Voyage AI embedding provider using LangChain integration.

    Implements the EmbeddingProvider protocol for Voyage AI models.
    Uses LangChain's VoyageAIEmbeddings for native async support.

    Environment Variables:
        EMBEDDING_PROVIDER: Must be 'voyage'
        VOYAGE_API_KEY: Voyage AI API key (required)
        EMBEDDING_MODEL: Model name (default: voyage-3)
        EMBEDDING_DIM: Vector dimensions (default: 1024)
        VOYAGE_TRUNCATION: Control truncation behavior (default: false = error on exceed)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.embedding.model
        self._dimension = settings.embedding.dim
        self._api_key = settings.embedding.voyage_api_key
        self._truncation = settings.embedding.voyage_truncation
        self._batch_size = settings.embedding.voyage_batch_size
        self._embeddings: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain VoyageAIEmbeddings client.

        Raises:
            ImportError: If langchain-voyageai is not installed
            ValueError: If API key is not configured
        """
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except ImportError as e:
            raise ImportError(
                'langchain-voyageai package required',
            ) from e

        if self._api_key is None:
            raise ValueError(
                'VOYAGE_API_KEY is required for Voyage AI embedding provider. '
                'Set the environment variable or use a different provider.',
            )

        # Build kwargs for VoyageAIEmbeddings
        # Note: VoyageAI underlying client has max_retries=0 by default
        # Universal wrapper handles all retry logic
        # Note: Using kwargs pattern because pyright type stubs don't recognize voyage_api_key
        kwargs: dict[str, Any] = {
            'model': self._model,
            'voyage_api_key': self._api_key.get_secret_value(),
            'batch_size': self._batch_size,
            'truncation': self._truncation,
        }
        self._embeddings = VoyageAIEmbeddings(**kwargs)
        truncation_mode = 'disabled (errors on exceed)' if not self._truncation else 'enabled (silent truncation)'
        logger.info(f'Initialized Voyage AI embedding provider: {self._model}, truncation={truncation_mode}')

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._embeddings = None
        logger.info('Voyage AI embedding provider shut down')

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
        """Check if Voyage AI API is available.

        Returns:
            True if provider is ready to generate embeddings

        Raises:
            ConfigurationError: If the API returns a client error (4xx) indicating
                a permanent configuration problem (e.g., invalid VOYAGE_API_KEY or
                EMBEDDING_MODEL)
        """
        if self._embeddings is None:
            return False

        try:
            await self._embeddings.aembed_query('test')
        except Exception as e:
            if is_client_error(e):
                raise ConfigurationError(
                    f'Voyage AI API returned a client error during availability check: {e}. '
                    'This indicates a permanent configuration problem (e.g., invalid '
                    'VOYAGE_API_KEY or EMBEDDING_MODEL). '
                    'Fix: Check the error message above and correct the configuration.',
                ) from e
            logger.warning(f'Voyage AI embedding not available: {e}')
            return False
        else:
            return True

    def get_dimension(self) -> int:
        """Return configured embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'voyage'

    @staticmethod
    def _convert_to_python_floats(embedding: list[Any]) -> list[float]:
        """Convert numpy types to Python float if needed."""
        return [
            x.item() if hasattr(x, 'item') else float(x)
            for x in embedding
        ]
