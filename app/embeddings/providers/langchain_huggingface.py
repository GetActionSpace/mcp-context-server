"""
HuggingFace embedding provider using LangChain integration.

This provider uses langchain-huggingface package for HuggingFace Hub models.
"""


import logging
from typing import Any

from app.embeddings.retry import with_retry_and_timeout
from app.embeddings.tracing import traced_embedding
from app.errors import ConfigurationError
from app.errors import is_client_error
from app.settings import get_settings

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider:
    """HuggingFace embedding provider using LangChain integration.

    Implements the EmbeddingProvider protocol for HuggingFace models.
    Uses LangChain's HuggingFaceEndpointEmbeddings for API-based embeddings.

    Environment Variables:
        EMBEDDING_PROVIDER: Must be 'huggingface'
        HUGGINGFACEHUB_API_TOKEN: HuggingFace Hub API token (required)
        EMBEDDING_MODEL: Model name (default: sentence-transformers/all-MiniLM-L6-v2)
        EMBEDDING_DIM: Vector dimensions (default: 384)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.embedding.model
        self._dimension = settings.embedding.dim
        self._api_token = settings.embedding.huggingface_api_key
        self._embeddings: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain HuggingFaceEndpointEmbeddings client.

        Raises:
            ImportError: If langchain-huggingface is not installed
            ValueError: If API token is not configured
        """
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
        except ImportError as e:
            raise ImportError(
                'langchain-huggingface package required',
            ) from e

        if self._api_token is None:
            raise ValueError(
                'HUGGINGFACEHUB_API_TOKEN is required for HuggingFace embedding provider. '
                'Set the environment variable or use a different provider.',
            )

        # HuggingFaceEndpointEmbeddings has no built-in retry
        # Universal wrapper handles all retry logic
        self._embeddings = HuggingFaceEndpointEmbeddings(
            model=self._model,
            huggingfacehub_api_token=self._api_token.get_secret_value(),
        )
        logger.info(f'Initialized HuggingFace embedding provider: {self._model}')

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._embeddings = None
        logger.info('HuggingFace embedding provider shut down')

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

        # Convert numpy types to Python float if needed
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

        # Convert numpy types and validate dimensions
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
        """Check if HuggingFace API is available.

        Returns:
            True if provider is ready to generate embeddings

        Raises:
            ConfigurationError: If the API returns a client error (4xx) indicating
                a permanent configuration problem (e.g., invalid HUGGINGFACEHUB_API_TOKEN
                or EMBEDDING_MODEL)
        """
        if self._embeddings is None:
            return False

        try:
            await self._embeddings.aembed_query('test')
        except Exception as e:
            if is_client_error(e):
                raise ConfigurationError(
                    f'HuggingFace API returned a client error during availability check: {e}. '
                    'This indicates a permanent configuration problem (e.g., invalid '
                    'HUGGINGFACEHUB_API_TOKEN or EMBEDDING_MODEL). '
                    'Fix: Check the error message above and correct the configuration.',
                ) from e
            logger.warning(f'HuggingFace embedding not available: {e}')
            return False
        else:
            return True

    def get_dimension(self) -> int:
        """Return configured embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'huggingface'

    @staticmethod
    def _convert_to_python_floats(embedding: list[Any]) -> list[float]:
        """Convert numpy types to Python float if needed."""
        return [
            x.item() if hasattr(x, 'item') else float(x)
            for x in embedding
        ]
