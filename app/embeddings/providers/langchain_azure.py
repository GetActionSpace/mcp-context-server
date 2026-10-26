"""
Azure OpenAI embedding provider using LangChain integration.

This provider uses langchain-openai package for Azure OpenAI Service.
"""


import logging
from typing import Any

from app.embeddings.retry import with_retry_and_timeout
from app.embeddings.tracing import traced_embedding
from app.errors import ConfigurationError
from app.errors import is_client_error
from app.settings import get_settings

logger = logging.getLogger(__name__)


class AzureEmbeddingProvider:
    """Azure OpenAI embedding provider using LangChain integration.

    Implements the EmbeddingProvider protocol for Azure OpenAI models.
    Uses LangChain's AzureOpenAIEmbeddings for native async support.

    Environment Variables:
        EMBEDDING_PROVIDER: Must be 'azure'
        AZURE_OPENAI_API_KEY: Azure OpenAI API key (required)
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL (required)
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: Deployment name (required)
        AZURE_OPENAI_API_VERSION: API version (default: 2024-02-01)
        EMBEDDING_DIM: Vector dimensions (default: 1536)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._dimension = settings.embedding.dim
        self._api_key = settings.embedding.azure_openai_api_key
        self._endpoint = settings.embedding.azure_openai_endpoint
        self._deployment = settings.embedding.azure_openai_deployment_name
        self._api_version = settings.embedding.azure_openai_api_version
        self._embeddings: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain AzureOpenAIEmbeddings client.

        Raises:
            ImportError: If langchain-openai is not installed
            ValueError: If required Azure settings are not configured
        """
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                'langchain-openai package required',
            ) from e

        # Validate required settings
        if self._api_key is None:
            raise ValueError('AZURE_OPENAI_API_KEY is required for Azure embedding provider.')

        if self._endpoint is None:
            raise ValueError('AZURE_OPENAI_ENDPOINT is required for Azure embedding provider.')

        if self._deployment is None:
            raise ValueError('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is required for Azure embedding provider.')

        # Build kwargs dict for AzureOpenAIEmbeddings
        # Disable internal retry (max_retries=0) - universal wrapper handles retries
        # Disable internal timeout (timeout=None) - universal wrapper handles timeout
        kwargs: dict[str, Any] = {
            'openai_api_key': self._api_key,
            'azure_endpoint': self._endpoint,
            'deployment': self._deployment,
            'openai_api_version': self._api_version,
            'max_retries': 0,
            'timeout': None,
        }
        self._embeddings = AzureOpenAIEmbeddings(**kwargs)
        logger.info(f'Initialized Azure OpenAI embedding provider: {self._deployment}')

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._embeddings = None
        logger.info('Azure OpenAI embedding provider shut down')

    @traced_embedding
    async def embed_query(self, text: str) -> list[float]:
        """Generate single embedding using async method."""
        if self._embeddings is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        async def _embed() -> list[Any]:
            result: list[Any] = await self._embeddings.aembed_query(text)
            return result

        embedding = await with_retry_and_timeout(_embed, f'{self.provider_name}_embed_query')
        embedding = self._convert_to_python_floats(embedding)

        # Key operational event: shows embedding generation worked
        logger.info(f'Generated query embedding: text_len={len(text)}, dim={len(embedding)}')

        if len(embedding) != self._dimension:
            raise ValueError(
                f'Dimension mismatch: expected {self._dimension}, '
                f'got {len(embedding)}. Check EMBEDDING_DIM setting.',
            )

        return embedding

    @traced_embedding
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings using async method."""
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
        """Check if Azure OpenAI API is available.

        Returns:
            True if provider is ready to generate embeddings

        Raises:
            ConfigurationError: If the API returns a client error (4xx) indicating
                a permanent configuration problem (e.g., invalid AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_ENDPOINT, or AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)
        """
        if self._embeddings is None:
            return False

        try:
            await self._embeddings.aembed_query('test')
        except Exception as e:
            if is_client_error(e):
                raise ConfigurationError(
                    f'Azure OpenAI API returned a client error during availability check: {e}. '
                    'This indicates a permanent configuration problem (e.g., invalid '
                    'AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, or '
                    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME). '
                    'Fix: Check the error message above and correct the configuration.',
                ) from e
            logger.warning(f'Azure OpenAI embedding not available: {e}')
            return False
        else:
            return True

    def get_dimension(self) -> int:
        """Return configured embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'azure'

    @staticmethod
    def _convert_to_python_floats(embedding: list[Any]) -> list[float]:
        """Convert numpy types to Python float if needed."""
        return [
            x.item() if hasattr(x, 'item') else float(x)
            for x in embedding
        ]
