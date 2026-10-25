"""Embedding generation, text chunking, and embedding compression settings."""

import logging
import os
from typing import Annotated
from typing import Literal
from typing import Self

from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator

from app.settings.base import CommonSettings

logger = logging.getLogger(__name__)


class EmbeddingSettings(CommonSettings):
    """Embedding provider settings following LangChain conventions.

    All environment variable names follow LangChain documentation conventions
    for maximum compatibility and user familiarity.
    """

    # Embedding generation toggle
    # CRITICAL: generation_enabled default=True is INTENTIONAL and MUST NOT be changed.
    #
    # Rationale:
    # 1. Embeddings are fundamental infrastructure - users should explicitly opt OUT, not opt IN
    # 2. Fail-fast semantics prevent silent embedding gaps in stored content
    # 3. Users who don't want embeddings MUST explicitly set ENABLE_EMBEDDING_GENERATION=false
    # 4. This ensures no surprises - if embeddings are missing, user explicitly disabled them
    # 5. ENABLE_SEMANTIC_SEARCH=true requires embeddings; if ENABLE_EMBEDDING_GENERATION=false
    #    and ENABLE_SEMANTIC_SEARCH=true, semantic_search_context tool will NOT be registered
    #
    # DO NOT change this default without understanding the full architectural implications.
    # This default=True is part of the breaking change in v1.0.0.
    generation_enabled: bool = Field(
        default=True,
        alias='ENABLE_EMBEDDING_GENERATION',
        description='Enable embedding generation for stored context entries. '
                    'If true and dependencies are not met, server will NOT start. '
                    'Set to false to disable embeddings entirely.',
    )

    # Provider selection
    provider: Literal['ollama', 'openai', 'azure', 'huggingface', 'voyage'] = Field(
        default='ollama',
        alias='EMBEDDING_PROVIDER',
        description='Embedding provider: ollama (default), openai, azure, huggingface, voyage',
    )

    # Common settings
    model: str = Field(
        default='qwen3-embedding:0.6b',
        alias='EMBEDDING_MODEL',
        description='Embedding model name',
    )
    dim: int = Field(
        default=1024,
        alias='EMBEDDING_DIM',
        gt=0,
        le=4096,
        description='Embedding vector dimensions',
    )
    query_instruction: str | None = Field(
        default=None,
        alias='EMBEDDING_QUERY_INSTRUCTION',
        description='Instruction prefix prepended verbatim to the text embedded for search queries '
                    '(semantic_search_context and the semantic leg of hybrid_search_context). '
                    'Document embeddings on the store/update path always stay bare. '
                    'Unset or empty string leaves query text unchanged. '
                    'Instruct-aware models such as the default qwen3-embedding:0.6b prescribe an instructed '
                    "query side ('Instruct: {task}\\nQuery:{query}'); cloud models such as OpenAI "
                    'text-embedding-3-small need no prefix.',
    )

    # Timeout and retry settings
    timeout_s: float = Field(
        default=240.0,
        alias='EMBEDDING_TIMEOUT_S',
        gt=0,
        le=3600,
        description='Timeout in seconds for one attempt of one embedding request: one batch of '
                    'chunks on the store/update path, one query on search',
    )
    retry_max_attempts: int = Field(
        default=5,
        alias='EMBEDDING_RETRY_MAX_ATTEMPTS',
        ge=1,
        le=10,
        description='Maximum number of retry attempts for embedding generation',
    )
    retry_base_delay_s: float = Field(
        default=1.0,
        alias='EMBEDDING_RETRY_BASE_DELAY_S',
        gt=0,
        le=30,
        description='Base delay in seconds between retry attempts (with exponential backoff)',
    )

    # Concurrency control
    max_concurrent: int = Field(
        default=3,
        alias='EMBEDDING_MAX_CONCURRENT',
        ge=1,
        le=20,
        description='Maximum concurrent embedding requests on the store/update path. '
                    'A chunked document holds one slot per batch and releases it between batches. '
                    'Limits parallel Ollama/provider requests to prevent overload. '
                    'Default 3 balances throughput vs resource contention.',
    )

    # Document batching: a chunked document is embedded in batches, one provider
    # request per batch, so the timeout and retries apply per batch.
    batch_size: Literal['auto'] | Annotated[int, Field(ge=1, le=512)] = Field(
        default='auto',
        alias='EMBEDDING_BATCH_SIZE',
        description="Chunks per embedding request when a document is embedded. 'auto' (default) "
                    'sizes each batch to take about EMBEDDING_BATCH_TARGET_S from the measured '
                    'seconds per chunk, between 1 and 64 chunks; an integer from 1 to 512 fixes the size.',
    )
    batch_target_s: float = Field(
        default=5.0,
        alias='EMBEDDING_BATCH_TARGET_S',
        gt=0,
        le=3600,
        description="Wall time in seconds one embedding batch should take when EMBEDDING_BATCH_SIZE is 'auto'",
    )

    # Ollama-specific context and truncation settings
    ollama_num_ctx: int = Field(
        default=4096,
        alias='EMBEDDING_OLLAMA_NUM_CTX',
        ge=512,
        le=2097152,
        description='Ollama embedding context length in tokens. Default 4096. '
                    'Must match or exceed model capabilities.',
    )
    ollama_truncate: bool = Field(
        default=False,
        alias='EMBEDDING_OLLAMA_TRUNCATE',
        description='Control text truncation when exceeding embedding context length. '
                    'False (default): Returns error on exceeded context. '
                    'True: Silently truncates input (may degrade embedding quality).',
    )

    # OpenAI-specific (matches LangChain docs: OPENAI_API_KEY)
    openai_api_key: SecretStr | None = Field(
        default=None,
        alias='OPENAI_API_KEY',
        description='OpenAI API key',
    )
    openai_api_base: str | None = Field(
        default=None,
        alias='OPENAI_API_BASE',
        description='Custom base URL for OpenAI-compatible APIs',
    )
    openai_organization: str | None = Field(
        default=None,
        alias='OPENAI_ORGANIZATION',
        description='OpenAI organization ID',
    )

    # Azure OpenAI-specific (matches LangChain docs)
    azure_openai_api_key: SecretStr | None = Field(
        default=None,
        alias='AZURE_OPENAI_API_KEY',
        description='Azure OpenAI API key',
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        alias='AZURE_OPENAI_ENDPOINT',
        description='Azure OpenAI endpoint URL',
    )
    azure_openai_api_version: str = Field(
        default='2024-02-01',
        alias='AZURE_OPENAI_API_VERSION',
        description='Azure OpenAI API version',
    )
    azure_openai_deployment_name: str | None = Field(
        default=None,
        alias='AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME',
        description='Azure OpenAI embedding deployment name',
    )

    # HuggingFace-specific (matches LangChain docs)
    huggingface_api_key: SecretStr | None = Field(
        default=None,
        alias='HUGGINGFACEHUB_API_TOKEN',
        description='HuggingFace Hub API token',
    )

    # Voyage AI-specific (matches LangChain docs: VOYAGE_API_KEY)
    voyage_api_key: SecretStr | None = Field(
        default=None,
        alias='VOYAGE_API_KEY',
        description='Voyage AI API key',
    )
    voyage_truncation: bool = Field(
        default=False,
        alias='VOYAGE_TRUNCATION',
        description='Control text truncation when exceeding context length. '
                    'False (default): Returns error on exceeded context. '
                    'True: Silently truncates input (may degrade embedding quality).',
    )
    voyage_batch_size: int = Field(
        default=7,
        alias='VOYAGE_BATCH_SIZE',
        ge=1,
        le=128,
        description='Number of texts per API call (default: 7)',
    )

    @field_validator('batch_size', mode='before')
    @classmethod
    def _coerce_batch_size(cls, value: object) -> object:
        """Turn a digit string into an int and any casing of 'auto' into 'auto'.

        Returns:
            The int or 'auto' for those inputs, otherwise the value unchanged for
            field validation to reject.
        """
        if isinstance(value, str):
            text = value.strip()
            if text.lower() == 'auto':
                return 'auto'
            if text.isdigit():
                return int(text)
        return value

    @field_validator('dim')
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """Warn when the dimension is not a multiple of 64 (the Field's ``le=4096`` already enforces the ceiling)."""
        if v % 64 != 0:
            logger.warning(
                f'EMBEDDING_DIM={v} is not a multiple of 64. '
                f'Most embedding models use dimensions divisible by 64.',
            )
        return v


class ChunkingSettings(CommonSettings):
    """Text chunking settings for semantic search.

    Controls how long documents are split into smaller chunks for embedding.
    Chunking improves semantic search quality for documents longer than ~500 tokens.
    """

    enabled: bool = Field(
        default=True,
        alias='ENABLE_CHUNKING',
        description='Enable text chunking for embedding generation',
    )
    size: int = Field(
        default=1500,
        alias='CHUNK_SIZE',
        ge=100,
        le=10000,
        description='Target chunk size in characters (default: 1500)',
    )
    overlap: int = Field(
        default=150,
        alias='CHUNK_OVERLAP',
        ge=0,
        le=500,
        description='Overlap between chunks in characters (default: 150)',
    )
    aggregation: Literal['max'] = Field(
        default='max',
        alias='CHUNK_AGGREGATION',
        description='How to aggregate chunk scores (currently only max is supported; '
                    'avg and sum will be added in future releases)',
    )

    @model_validator(mode='after')
    def validate_overlap_less_than_size(self) -> Self:
        """Ensure overlap is strictly less than chunk size."""
        if self.overlap >= self.size:
            raise ValueError(
                f'CHUNK_OVERLAP ({self.overlap}) must be less than CHUNK_SIZE ({self.size})',
            )
        return self


class CompressionSettings(CommonSettings):
    """Embedding compression settings.

    Carries configuration for the TurboQuant embedding compression
    subsystem. The runtime storage and search paths consult these
    settings when ``ENABLE_EMBEDDING_COMPRESSION`` is true; a startup
    validator enforces the seed-locked invariant against the
    ``compression_metadata`` table.
    """

    enabled: bool = Field(
        default=True,
        alias='ENABLE_EMBEDDING_COMPRESSION',
        description='Enable TurboQuant embedding compression at storage time. '
                    'Default true. fp32 embeddings are replaced with bit-packed '
                    'compressed payloads (~8x storage reduction at the default '
                    'bits=4). Set ENABLE_EMBEDDING_COMPRESSION=false to disable '
                    'compression and keep fp32 storage.',
    )

    provider: Literal['turboquant'] = Field(
        default='turboquant',
        alias='COMPRESSION_PROVIDER',
        description='Compression provider. v3.0.0 supports only turboquant.',
    )

    bits: int = Field(
        default=4,
        ge=2,
        le=4,
        alias='COMPRESSION_BITS',
        description='Bits per coordinate. 2 = 16x compression, 3 = ~11x, 4 = 8x. '
                    'Default 4 = ~8x with high recall. The lower bound of 2 is '
                    "required by variant='ip' (the inner-product variant reserves "
                    'one bit for the QJL sign).',
    )

    variant: Literal['mse', 'ip'] = Field(
        default='ip',
        alias='COMPRESSION_VARIANT',
        description="'ip' (default): Algorithm 2 with QJL, unbiased inner-product "
                    "estimator. 'mse': Algorithm 1, L2-optimal reconstruction.",
    )

    seed: int = Field(
        default=0,
        ge=0,
        le=4294967295,
        alias='COMPRESSION_SEED',
        description='Rotation matrix seed. Load-bearing invariant: rotations are '
                    'deterministic given the seed; changing the seed AFTER any '
                    'compressed data has been stored will corrupt all decode/search '
                    'operations. Default 0. Pick any stable integer in the range '
                    '[0, 4294967295] (the seed is packed into the compressed payload '
                    'as an unsigned 32-bit field) and keep it constant for the '
                    'lifetime of the database; the value is persisted in '
                    'compression_metadata at first startup and validated on each '
                    'subsequent start (exit 78 on mismatch).',
    )

    max_concurrent: int = Field(
        default_factory=lambda: min(os.cpu_count() or 4, 4),
        ge=1,
        le=32,
        alias='COMPRESSION_MAX_CONCURRENT',
        description='Max concurrent compression encode workers dispatched to '
                    'threads. The CPU-bound codec section is serialized by a '
                    'process-wide BLAS-limits lock, so this bounds worker '
                    'fan-out (thread count and memory), not CPU parallelism. '
                    'Separate from I/O-bound EMBEDDING_MAX_CONCURRENT and '
                    'SUMMARY_MAX_CONCURRENT. Default min(cpu_count, 4).',
    )
