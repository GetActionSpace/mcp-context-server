"""Application settings: the ``AppSettings`` composition root and the cached ``get_settings()`` accessor.

Each configuration domain lives in its own submodule (``base``, ``server``, ``auth``, ``storage``, ``providers``,
``embedding``, ``summary``, ``generation``, ``search``, ``navigation``, ``sync``). ``AppSettings`` composes one
instance of each domain class and runs the validators that span several domains. Only ``AppSettings`` and
``get_settings`` are imported from this package; import a domain class from its defining submodule.
"""

import logging
from functools import lru_cache
from typing import Self

from pydantic import Field
from pydantic import model_validator

from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.pgvector_limits import exceeds_pgvector_index_dim_limit
from app.settings.auth import AccessControlSettings
from app.settings.auth import AuthSettings
from app.settings.base import CommonSettings
from app.settings.embedding import ChunkingSettings
from app.settings.embedding import CompressionSettings
from app.settings.embedding import EmbeddingSettings
from app.settings.generation import GenerationJobSettings
from app.settings.navigation import ContextNavigationSettings
from app.settings.navigation import ContextRangeSettings
from app.settings.navigation import GrepContextSettings
from app.settings.providers import LangSmithSettings
from app.settings.providers import OllamaSettings
from app.settings.search import FtsPassageSettings
from app.settings.search import FtsSettings
from app.settings.search import HybridSearchSettings
from app.settings.search import RerankingSettings
from app.settings.search import RetrievalSettings
from app.settings.search import SearchSettings
from app.settings.search import SemanticSearchSettings
from app.settings.server import InstructionsSettings
from app.settings.server import LoggingSettings
from app.settings.server import ToolManagementSettings
from app.settings.server import TransportSettings
from app.settings.storage import StorageSettings
from app.settings.summary import IndexTreeNodeSummarySettings
from app.settings.summary import SummarySettings
from app.settings.sync import ContextSyncSettings

logger = logging.getLogger(__name__)


class AppSettings(CommonSettings):
    """Root settings object: one instance of every domain settings class plus the validators that span domains."""

    # Core settings
    logging: LoggingSettings = Field(default_factory=lambda: LoggingSettings())
    tools: ToolManagementSettings = Field(default_factory=lambda: ToolManagementSettings())
    storage: StorageSettings = Field(default_factory=lambda: StorageSettings())

    # Search-related settings
    search: SearchSettings = Field(default_factory=lambda: SearchSettings())
    semantic_search: SemanticSearchSettings = Field(default_factory=lambda: SemanticSearchSettings())
    fts: FtsSettings = Field(default_factory=lambda: FtsSettings())
    hybrid_search: HybridSearchSettings = Field(default_factory=lambda: HybridSearchSettings())
    fts_passage: FtsPassageSettings = Field(default_factory=lambda: FtsPassageSettings())
    retrieval: RetrievalSettings = Field(default_factory=lambda: RetrievalSettings())
    grep_context: GrepContextSettings = Field(default_factory=lambda: GrepContextSettings())
    context_range: ContextRangeSettings = Field(default_factory=lambda: ContextRangeSettings())
    context_sync: ContextSyncSettings = Field(default_factory=lambda: ContextSyncSettings())
    context_navigation: ContextNavigationSettings = Field(default_factory=lambda: ContextNavigationSettings())
    index_tree: IndexTreeNodeSummarySettings = Field(default_factory=lambda: IndexTreeNodeSummarySettings())

    # Embedding and processing settings
    embedding: EmbeddingSettings = Field(default_factory=lambda: EmbeddingSettings())
    summary: SummarySettings = Field(default_factory=lambda: SummarySettings())
    generation_jobs: GenerationJobSettings = Field(default_factory=lambda: GenerationJobSettings())
    chunking: ChunkingSettings = Field(default_factory=lambda: ChunkingSettings())
    reranking: RerankingSettings = Field(default_factory=lambda: RerankingSettings())

    # Embedding compression settings
    compression: CompressionSettings = Field(default_factory=lambda: CompressionSettings())

    # Shared Ollama settings
    ollama: OllamaSettings = Field(default_factory=lambda: OllamaSettings())

    # Infrastructure settings
    transport: TransportSettings = Field(default_factory=lambda: TransportSettings())
    auth: AuthSettings = Field(default_factory=lambda: AuthSettings())
    access_control: AccessControlSettings = Field(default_factory=lambda: AccessControlSettings())
    instructions: InstructionsSettings = Field(default_factory=lambda: InstructionsSettings())
    langsmith: LangSmithSettings = Field(default_factory=lambda: LangSmithSettings())

    @model_validator(mode='after')
    def validate_chunk_size_vs_context_limit(self) -> Self:
        """Validate CHUNK_SIZE against model context window from context_limits.py.

        This is a UNIVERSAL validator that works for ALL embedding providers.

        When ENABLE_CHUNKING=true:
            - Validates CHUNK_SIZE against model's max_tokens
            - Warns if chunk size may exceed context window

        When ENABLE_CHUNKING=false:
            - Warns about potential issues with large documents
            - Document sizes are unknown at startup, so only general warning is possible

        Returns:
            Self: The validated settings instance.
        """
        # Import here to avoid circular imports at module load time
        try:
            from app.embeddings.context_limits import get_model_spec
            from app.embeddings.context_limits import get_provider_default_context
        except ImportError:
            # context_limits module not available - skip validation
            return self

        # Get model specification from context_limits.py
        model_spec = get_model_spec(self.embedding.model)

        # Determine max_tokens for the model
        # truncation_behavior can be 'error', 'silent', 'configurable', or None (unknown)
        truncation_behavior: str | None
        if model_spec:
            max_tokens = model_spec.max_tokens
            truncation_behavior = model_spec.truncation_behavior
            source = f'model spec for {model_spec.model}'
        else:
            # Unknown model - use provider default
            max_tokens = get_provider_default_context(self.embedding.provider)
            truncation_behavior = None  # Unknown behavior
            source = f'provider default for {self.embedding.provider}'
            logger.warning(
                f'Model "{self.embedding.model}" not found in context_limits.py. '
                f'Using provider default context limit ({max_tokens} tokens). '
                f'Consider adding model spec to app/embeddings/context_limits.py for accurate validation.',
            )

        if not self.chunking.enabled:
            # ENABLE_CHUNKING=false - warn about potential issues
            if truncation_behavior == 'silent':
                logger.warning(
                    f'ENABLE_CHUNKING=false with provider "{self.embedding.provider}". '
                    f'Model "{self.embedding.model}" ALWAYS silently truncates (cannot be disabled). '
                    f'Documents exceeding {max_tokens} tokens ({source}) will be truncated without warning.',
                )
            elif truncation_behavior == 'configurable':
                # Determine current truncation setting for this provider
                truncation_enabled = self._get_truncation_setting_for_provider()
                if truncation_enabled:
                    logger.warning(
                        f'ENABLE_CHUNKING=false with truncation enabled. '
                        f'Large documents will be silently truncated to {max_tokens} tokens ({source}). '
                        f'Consider enabling chunking for better embedding quality.',
                    )
                else:
                    logger.warning(
                        f'ENABLE_CHUNKING=false with truncation disabled. '
                        f'Documents exceeding {max_tokens} tokens ({source}) will cause embedding errors. '
                        f'Consider enabling chunking to handle large documents.',
                    )
            elif truncation_behavior == 'error':
                logger.warning(
                    f'ENABLE_CHUNKING=false with provider "{self.embedding.provider}". '
                    f'Model "{self.embedding.model}" returns error on context exceed (no truncation). '
                    f'Documents exceeding {max_tokens} tokens ({source}) will fail embedding. '
                    f'Consider enabling chunking to handle large documents.',
                )
            else:
                # Unknown truncation behavior
                logger.warning(
                    f'ENABLE_CHUNKING=false. Document sizes unknown at startup. '
                    f'Documents exceeding {max_tokens} tokens ({source}) may cause issues. '
                    f'Consider enabling chunking for better reliability.',
                )
            return self

        # ENABLE_CHUNKING=true - validate CHUNK_SIZE against max_tokens
        # Heuristic: 1 token ~ 3-4 characters for English
        chunk_tokens_estimate = self.chunking.size / 3

        if chunk_tokens_estimate > max_tokens:
            # Determine consequence based on truncation behavior
            if truncation_behavior == 'silent':
                consequence = 'will be silently truncated (quality degradation)'
            elif truncation_behavior == 'configurable':
                truncation_enabled = self._get_truncation_setting_for_provider()
                consequence = 'will be silently truncated' if truncation_enabled else 'will cause embedding errors'
            elif truncation_behavior == 'error':
                consequence = 'will cause embedding errors'
            else:
                consequence = 'may cause issues'

            logger.warning(
                f'CHUNK_SIZE ({self.chunking.size} chars, '
                f'~{int(chunk_tokens_estimate)} tokens estimate) exceeds '
                f'model context limit ({max_tokens} tokens from {source}). '
                f'Chunks {consequence}. '
                f'Recommendation: Reduce CHUNK_SIZE to ~{int(max_tokens * 3 * 0.8)} chars '
                f'(80% of context window).',
            )

        return self

    @model_validator(mode='after')
    def validate_fts_passage_vs_reranking(self) -> Self:
        """Validate FTS passage settings against cross-encoder token limits.

        When reranking is enabled, validates that FTS passage extraction settings
        are configured appropriately for the cross-encoder's max_length limit.

        Uses configurable chars_per_token ratio for token estimation, allowing
        users to tune based on their content type (English prose ~4.5, code ~3.5).

        Returns:
            Self: The validated settings instance.
        """
        # Skip validation if reranking is disabled
        if not self.reranking.enabled:
            return self

        # Calculate estimated passage size for a single FTS match with context windows
        boundary_expansion = 400  # max_search * 2 from expand_to_boundary
        single_match_estimate = self.fts_passage.rerank_window_size * 2 + boundary_expansion

        # Estimate token usage
        estimated_tokens = single_match_estimate / self.reranking.chars_per_token

        if estimated_tokens > self.reranking.max_length:
            optimal_window = int(
                (self.reranking.max_length * self.reranking.chars_per_token - boundary_expansion) / 2,
            )
            logger.warning(
                f'Single FTS match may produce ~{int(estimated_tokens)} tokens '
                f'(using {self.reranking.chars_per_token} chars/token), exceeding RERANKING_MAX_LENGTH '
                f'({self.reranking.max_length} tokens). Cross-encoder will truncate. '
                f'Recommendations: '
                f'1. Reduce FTS_RERANK_WINDOW_SIZE to ~{optimal_window} chars, OR '
                f'2. Increase RERANKING_CHARS_PER_TOKEN if your content has longer words',
            )

        return self

    @model_validator(mode='after')
    def validate_pgvector_dimension_limit(self) -> Self:
        """Reject an fp32 PostgreSQL configuration whose embedding dimension exceeds pgvector's index cap.

        pgvector caps HNSW (and IVFFlat) index dimensionality at
        ``PGVECTOR_INDEX_DIM_LIMIT`` (2000) for the ``vector`` type -- the cap is
        shared via ``app.pgvector_limits`` with the migration CLIs, which
        pre-flight the same limit before rebuilding the fp32 layout. On
        PostgreSQL with embedding generation enabled and compression OFF, the
        fp32 write path provisions ``vec_context_embeddings`` as ``vector(dim)``
        and builds ``idx_vec_context_embeddings_hnsw`` over it, so an
        EMBEDDING_DIM above the cap makes the semantic-search migration crash at
        CREATE INDEX time -- a boot failure for a setting that is already invalid
        the moment it is read. Enabling compression removes the constraint
        (compressed payloads are stored as BYTEA with no pgvector index), and
        SQLite's sqlite-vec has no equivalent per-dimension index cap, so the guard
        is scoped to the PostgreSQL fp32 path. Rejecting it here turns the deferred
        migration crash into a clean configuration error the supervisor will not
        restart-loop on.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If the PostgreSQL fp32 path is configured with an embedding
                dimension above the pgvector index limit.
        """
        if (
            self.storage.backend_type == 'postgresql'
            and self.embedding.generation_enabled
            and not self.compression.enabled
            and exceeds_pgvector_index_dim_limit(self.embedding.dim)
        ):
            raise ValueError(
                f'EMBEDDING_DIM ({self.embedding.dim}) exceeds the pgvector index limit of '
                f'{PGVECTOR_INDEX_DIM_LIMIT} dimensions for fp32 vectors on PostgreSQL. '
                f'The semantic-search migration would fail building the HNSW index on '
                f'vec_context_embeddings. Either reduce EMBEDDING_DIM to '
                f'{PGVECTOR_INDEX_DIM_LIMIT} or below, or set ENABLE_EMBEDDING_COMPRESSION=true '
                f'(compressed payloads are stored as BYTEA with no pgvector dimension cap).',
            )
        return self

    @model_validator(mode='after')
    def validate_postgresql_rls_configuration(self) -> Self:
        """Reject a row-level security configuration that cannot enforce anything.

        Row-level security is a PostgreSQL feature, and it filters only a role that
        does not own the tables: the migration role must own them and requests must
        connect as a different role. This check sees configured names only, so it
        compares the two users only in the component form; with
        POSTGRESQL_CONNECTION_STRING the request user lives inside the secret, and
        startup compares the roles both connections actually resolve to, which also
        covers pooler-suffixed names and forced logins.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If row-level security is enabled with the SQLite backend,
                without a migration user, or with a migration user equal to the
                configured request user.
        """
        if not self.access_control.postgresql_rls_enabled:
            return self
        if self.storage.backend_type != 'postgresql':
            raise ValueError('ENABLE_POSTGRESQL_RLS requires STORAGE_BACKEND=postgresql')
        migration_user = self.storage.postgresql_migration_user
        if not migration_user:
            raise ValueError(
                'Row-level security needs a separate migration role that owns the tables: '
                'set POSTGRESQL_MIGRATION_USER and POSTGRESQL_MIGRATION_PASSWORD',
            )
        # An empty connection string selects the component form, as in the backend's connection builder.
        if not self.storage.postgresql_connection_string and migration_user == self.storage.postgresql_user:
            # The role grammar excludes the double quote, so wrapping the name yields its quoted identifier.
            policy_role = f'"{self.access_control.postgresql_rls_policy_role}"'
            raise ValueError(
                f"ENABLE_POSTGRESQL_RLS=true, but POSTGRESQL_MIGRATION_USER and POSTGRESQL_USER name the same "
                f"PostgreSQL role '{migration_user}'. The table owner bypasses row-level security, so nothing "
                f'would be enforced. Create a separate login role for requests (CREATE ROLE {policy_role} NOLOGIN; '
                f"CREATE ROLE <login> LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS; GRANT {policy_role} TO <login>;) "
                f"and set POSTGRESQL_USER to it (docs/database-backends.md, 'Row-level security (PostgreSQL)').",
            )
        return self

    def _get_truncation_setting_for_provider(self) -> bool:
        """Get current truncation setting for the configured provider.

        Returns:
            bool: True if truncation is enabled, False otherwise
        """
        # Provider is Literal['ollama', 'openai', 'azure', 'huggingface', 'voyage']
        # All cases are exhaustively covered
        match self.embedding.provider:
            case 'ollama':
                return self.embedding.ollama_truncate
            case 'voyage':
                return self.embedding.voyage_truncation
            case 'openai' | 'azure':
                return False  # OpenAI/Azure always error on exceed
            case 'huggingface':
                return True  # HuggingFace always silently truncates


@lru_cache
def get_settings() -> AppSettings:
    """Return the process-wide ``AppSettings``, built on the first call and cached for the process lifetime."""
    return AppSettings()
