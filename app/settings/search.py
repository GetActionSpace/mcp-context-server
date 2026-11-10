"""Search settings: search behavior, the semantic, FTS and hybrid tool toggles, FTS passages, reranking, and retrieval."""

from typing import Literal

from pydantic import Field
from pydantic import field_validator

from app.settings.base import CommonSettings
from app.settings.base import FeatureToggleSettings

# The values FTS_LANGUAGE accepts: PostgreSQL's built-in text search configurations
# (SELECT cfgname FROM pg_ts_config).
FTS_LANGUAGES = frozenset({
    'simple', 'arabic', 'armenian', 'basque', 'catalan', 'danish', 'dutch',
    'english', 'finnish', 'french', 'german', 'greek', 'hindi', 'hungarian',
    'indonesian', 'irish', 'italian', 'lithuanian', 'nepali', 'norwegian',
    'portuguese', 'romanian', 'russian', 'serbian', 'spanish', 'swedish',
    'tamil', 'turkish', 'yiddish',
})


class RerankingSettings(CommonSettings):
    """Cross-encoder reranking settings.

    Reranking improves search precision by using a cross-encoder model
    to re-score and reorder initial search results.
    """

    enabled: bool = Field(
        default=True,
        alias='ENABLE_RERANKING',
        description='Enable cross-encoder reranking of search results',
    )
    provider: str = Field(
        default='flashrank',
        alias='RERANKING_PROVIDER',
        description='Reranking provider (default: flashrank)',
    )
    model: str = Field(
        default='ms-marco-MiniLM-L-12-v2',
        alias='RERANKING_MODEL',
        description='Reranking model name (default: ms-marco-MiniLM-L-12-v2, 34MB)',
    )
    max_length: int = Field(
        default=512,
        alias='RERANKING_MAX_LENGTH',
        ge=128,
        le=2048,
        description='Maximum input length for reranking (default: 512 tokens)',
    )
    cache_dir: str | None = Field(
        default=None,
        alias='RERANKING_CACHE_DIR',
        description='Directory for caching reranking models (default: system cache)',
    )
    chars_per_token: float = Field(
        default=4.0,
        alias='RERANKING_CHARS_PER_TOKEN',
        ge=2.0,
        le=8.0,
        description='Estimated characters per token for passage size validation. '
                    'Default 4.0 for English. Use 3.0-3.5 for multilingual/code.',
    )
    intra_op_threads: int = Field(
        default=0,
        alias='RERANKING_INTRA_OP_THREADS',
        ge=0,
        description='ONNX Runtime intra-operation parallelism threads for reranking. '
                    '0 = auto-detect (uses all available cores). '
                    'In containerized environments with CPU limits, set to match the CPU quota '
                    'to prevent thread explosion (e.g., 2 for a 2-core container).',
    )
    cpu_mem_arena: bool = Field(
        default=False,
        alias='RERANKING_CPU_MEM_ARENA',
        description='Enable ONNX Runtime CPU memory arena for reranking. '
                    'When False (default), prevents permanent retention of intermediate '
                    'tensor buffers, reducing RAM usage in containerized deployments. '
                    'Set True to enable arena for slightly faster inference at the cost '
                    'of higher memory consumption.',
    )
    batch_size: int = Field(
        default=32,
        alias='RERANKING_BATCH_SIZE',
        gt=0,
        description='Maximum number of passages per ONNX Runtime inference batch during reranking. '
                    'Prevents OOM from unbounded batch sizes with large result sets. '
                    'Typical workloads (20-40 passages) fit in a single batch at default value.',
    )


class FtsPassageSettings(CommonSettings):
    """FTS passage extraction settings for reranking.

    Controls how text passages are extracted from FTS results with highlighted
    matches for use in cross-encoder reranking. These settings affect the quality
    and size of passages sent to the reranker.
    """

    rerank_window_size: int = Field(
        default=750,
        alias='FTS_RERANK_WINDOW_SIZE',
        ge=100,
        le=2000,
        description='Characters of context around each FTS match for reranking passage extraction (default: 750)',
    )

    rerank_gap_merge: int = Field(
        default=100,
        alias='FTS_RERANK_GAP_MERGE',
        ge=0,
        le=500,
        description='Merge FTS match regions within this character distance (default: 100)',
    )


class SemanticSearchSettings(FeatureToggleSettings):
    """Semantic search feature configuration.

    Controls whether the semantic_search_context tool is registered. Requires an
    embedding provider, which is initialized whenever ENABLE_EMBEDDING_GENERATION
    resolves on.
    """

    mode: Literal['auto', 'true', 'false'] = Field(
        default='auto',
        alias='ENABLE_SEMANTIC_SEARCH',
        description='Semantic search tool registration: auto and true both register the '
                    'tool when an embedding provider is available (a warning is logged for '
                    'true when none is, since semantic search has no backend without a '
                    'provider), false (force off).',
    )


class FtsSettings(FeatureToggleSettings):
    """Full-text search feature configuration.

    Controls FTS tool registration and language/tokenizer settings. Full-text
    search uses built-in database capabilities and needs no extra dependencies,
    so 'auto' registers it by default.
    """

    mode: Literal['auto', 'true', 'false'] = Field(
        default='auto',
        alias='ENABLE_FTS',
        description='Full-text search tool registration: auto (register; uses '
                    'built-in database FTS, no extra dependencies), true (force '
                    'on), false (force off).',
    )

    language: str = Field(
        default='english',
        alias='FTS_LANGUAGE',
        description='Language for FTS stemming (e.g., english, german, french)',
    )

    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate FTS language is a known PostgreSQL text search configuration.

        PostgreSQL FTS requires a valid text search configuration. Invalid values
        cause runtime failures when applying migrations or executing queries.
        This validator fails fast at startup to prevent runtime errors.

        Returns:
            str: The validated language name normalized to lowercase.

        Raises:
            ValueError: If the language is not a valid PostgreSQL text search configuration.
        """
        v_lower = v.lower()
        if v_lower not in FTS_LANGUAGES:
            raise ValueError(
                f"FTS_LANGUAGE='{v}' is not a valid PostgreSQL text search configuration. "
                f'Valid options: {", ".join(sorted(FTS_LANGUAGES))}',
            )
        return v_lower


class HybridSearchSettings(FeatureToggleSettings):
    """Hybrid search configuration using Reciprocal Rank Fusion (RRF).

    Combines FTS and semantic search results for improved relevance.
    """

    mode: Literal['auto', 'true', 'false'] = Field(
        default='auto',
        alias='ENABLE_HYBRID_SEARCH',
        description='Hybrid search tool registration: auto and true both register the '
                    'tool when at least one of full-text or semantic search is available '
                    '(a warning is logged when neither is, since hybrid has no underlying '
                    'mode to fuse), false (force off).',
    )

    rrf_k: int = Field(
        default=60,
        alias='HYBRID_RRF_K',
        ge=1,
        le=1000,
        description='RRF smoothing constant for hybrid search (default 60)',
    )

    rrf_overfetch: int = Field(
        default=2,
        alias='HYBRID_RRF_OVERFETCH',
        ge=1,
        le=10,
        description='Multiplier for over-fetching results before RRF fusion (default: 2x)',
    )

    fts_or_threshold: int = Field(
        default=4,
        alias='HYBRID_FTS_OR_THRESHOLD',
        ge=2,
        le=20,
        description='Minimum number of significant query terms to switch FTS from AND to OR logic (default: 4)',
    )


class SearchSettings(CommonSettings):
    """General search behavior configuration.

    Settings that apply across all search types (FTS, semantic, hybrid).
    """

    truncation_length: int = Field(
        default=300,
        ge=50,
        le=1000,
        alias='SEARCH_TRUNCATION_LENGTH',
        description='Maximum character length for truncated text_content in search results (default: 300)',
    )


class RetrievalSettings(CommonSettings):
    """Retrieval-tool response-shape configuration.

    Settings that govern the response shape of by-ID retrieval tools
    (currently `get_context_by_ids`). Distinct from SearchSettings,
    which governs search tools' truncation and ranking behavior.

    This class is the home for future per-tool response-shape toggles
    targeting retrieval-by-ID tools.
    """

    include_summary: bool = Field(
        default=False,
        alias='GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY',
        description='Whether get_context_by_ids includes the summary field in each '
                    'returned entry. Default false: the tool already returns the full '
                    'text_content, so the AI-generated summary is redundant and inflates '
                    'token usage. Set true to include the summary anyway. Does not affect '
                    'search tools, which always return summary because they truncate '
                    'text_content.',
    )
