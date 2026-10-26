"""
Ollama embedding provider using LangChain integration.

This provider uses langchain-ollama package for native async support
and consistent interface with other LangChain providers.
"""


import logging
from typing import Any

from app.embeddings.retry import with_retry_and_timeout
from app.embeddings.tracing import traced_embedding
from app.errors import ConfigurationError
from app.errors import is_client_error
from app.settings import get_settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    """Ollama embedding provider using LangChain integration.

    Implements the EmbeddingProvider protocol for Ollama models.
    Uses LangChain's OllamaEmbeddings for native async support.

    Environment Variables:
        EMBEDDING_PROVIDER: Must be 'ollama' (default)
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
        EMBEDDING_MODEL: Model name (default: qwen3-embedding:0.6b)
        EMBEDDING_DIM: Vector dimensions (default: 1024)
        EMBEDDING_OLLAMA_NUM_CTX: Context length in tokens (default: 4096)
        EMBEDDING_OLLAMA_TRUNCATE: Control truncation behavior (default: false = error on exceed)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.embedding.model
        self._base_url = settings.ollama.host
        self._dimension = settings.embedding.dim
        self._truncate = settings.embedding.ollama_truncate
        self._num_ctx = settings.embedding.ollama_num_ctx
        self._embeddings: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain OllamaEmbeddings client.

        Raises:
            ImportError: If langchain-ollama is not installed
        """
        # Import httpx here to avoid unused import warnings
        # when Ollama provider is not used
        import httpx

        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as e:
            raise ImportError(
                'langchain-ollama package required',
            ) from e

        settings = get_settings()

        # Configure httpx limits for connection pooling
        # This prevents per-request connection creation overhead
        httpx_limits = httpx.Limits(
            max_connections=20,  # Total connections allowed
            max_keepalive_connections=10,  # Connections kept alive for reuse
        )

        # Configure timeouts aligned with embedding settings
        httpx_timeout = httpx.Timeout(
            connect=10.0,  # Connection establishment timeout
            read=settings.embedding.timeout_s,  # Read timeout (matches EMBEDDING_TIMEOUT_S)
            write=30.0,  # Write timeout
            pool=5.0,  # Pool acquisition timeout
        )

        # OllamaEmbeddings has no built-in retry
        # Universal wrapper handles all retry logic
        # Note: truncate parameter not supported by langchain-ollama library.
        # Truncation control is handled via pre-validation in _validate_text_length()
        self._embeddings = OllamaEmbeddings(
            model=self._model,
            base_url=self._base_url,
            # httpx client configuration for connection reuse
            client_kwargs={
                'limits': httpx_limits,
                'timeout': httpx_timeout,
            },
        )
        logger.info(
            f'Initialized Ollama embedding provider: {self._model} at {self._base_url}, '
            f'num_ctx={self._num_ctx}, httpx_limits=(max={httpx_limits.max_connections}, '
            f'keepalive={httpx_limits.max_keepalive_connections})',
        )
        if not self._truncate:
            logger.info(
                'EMBEDDING_OLLAMA_TRUNCATE=false. Text length validation enabled. '
                'Texts exceeding context limit will raise error before embedding.',
            )

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._embeddings = None
        logger.info('Ollama embedding provider shut down')

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

        # Pre-validate text length when truncation disabled
        if not self._truncate:
            self._validate_text_length(text)

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

        # Pre-validate all texts when truncation disabled
        if not self._truncate:
            for i, text in enumerate(texts):
                try:
                    self._validate_text_length(text)
                except ValueError as e:
                    raise ValueError(f'Text {i} validation failed: {e}') from e

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
        """Check if Ollama model is available.

        Returns:
            True if provider is ready to generate embeddings

        Raises:
            ConfigurationError: If the API returns a client error (4xx) indicating
                a permanent configuration problem (e.g., invalid EMBEDDING_MODEL)
        """
        if self._embeddings is None:
            return False

        try:
            # Quick test embedding
            await self._embeddings.aembed_query('test')
        except Exception as e:
            if is_client_error(e):
                raise ConfigurationError(
                    f'Ollama API returned a client error during availability check: {e}. '
                    'This indicates a permanent configuration problem (e.g., invalid '
                    'EMBEDDING_MODEL). '
                    'Fix: Check the error message above and correct the configuration.',
                ) from e
            logger.warning(f'Ollama embedding not available: {e}')
            return False
        else:
            return True

    def get_dimension(self) -> int:
        """Return configured embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'ollama'

    def _validate_text_length(self, text: str) -> None:
        """Validate text length against estimated context window.

        When EMBEDDING_OLLAMA_TRUNCATE=false, this provides fail-fast behavior by checking
        if text is likely to exceed the context window BEFORE calling the embedding API.

        Args:
            text: Text to validate

        Raises:
            ValueError: If text likely exceeds context window and truncation disabled
        """
        # Import here to avoid circular imports
        from app.embeddings.context_limits import get_model_spec

        # Get model-specific max_tokens if known, else use EMBEDDING_OLLAMA_NUM_CTX setting
        spec = get_model_spec(self._model)
        if spec:
            max_tokens = min(spec.max_tokens, self._num_ctx)
            source = f'model spec ({spec.max_tokens}) capped by EMBEDDING_OLLAMA_NUM_CTX ({self._num_ctx})'
        else:
            max_tokens = self._num_ctx
            source = f'EMBEDDING_OLLAMA_NUM_CTX ({self._num_ctx})'

        # Heuristic: 1 token ~ 3-4 characters for English
        # Use conservative estimate (3 chars/token) to avoid false negatives
        estimated_tokens = len(text) / 3

        if estimated_tokens > max_tokens:
            raise ValueError(
                f'Text length ({len(text)} chars, ~{int(estimated_tokens)} estimated tokens) '
                f'may exceed context window ({max_tokens} tokens from {source}) for model {self._model}. '
                f'Options: 1) Enable chunking (ENABLE_CHUNKING=true, default), '
                f'2) Increase EMBEDDING_OLLAMA_NUM_CTX, 3) Set EMBEDDING_OLLAMA_TRUNCATE=true to allow silent truncation.',
            )

    @staticmethod
    def _convert_to_python_floats(embedding: list[Any]) -> list[float]:
        """Convert numpy.float32 or similar to Python float.

        asyncpg with pgvector requires Python float, not numpy.float32.

        Args:
            embedding: Embedding vector potentially containing numpy types

        Returns:
            Embedding vector with Python float values
        """
        return [
            x.item() if hasattr(x, 'item') else float(x)
            for x in embedding
        ]
