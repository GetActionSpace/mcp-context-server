"""The response types of the discovery tools ``list_threads`` and ``get_statistics``."""

from typing import NotRequired
from typing import TypedDict


class ThreadInfoDict(TypedDict):
    """Type definition for individual thread info."""

    thread_id: str
    entry_count: int
    source_types: int
    multimodal_count: int
    first_entry: str
    last_entry: str
    last_id: str


class ThreadListDict(TypedDict):
    """Type definition for thread list response."""

    threads: list[ThreadInfoDict]
    total_threads: int


# Statistics TypedDicts (for get_statistics tool response)


class ConnectionMetricsDict(TypedDict, total=False):
    """Type definition for backend connection metrics.

    This TypedDict is what FastMCP derives the advertised ``connection_metrics``
    portion of the ``get_statistics`` outputSchema from, so every key either
    backend can emit MUST be declared here -- a key that is missing is dropped by
    any client reading the tool through its declared schema, which is exactly how
    operator-critical health data (circuit-breaker state, failure counters, last
    error) becomes invisible while the database is unhealthy.

    Fields are the UNION of what both backends return, all optional because the
    shape is backend-dependent:

    - Emitted by BOTH backends: ``backend_type``, ``pool_size``,
      ``total_queries``, ``failed_queries``, ``circuit_state``,
      ``consecutive_failures``, ``last_error``, ``last_error_time``.
      ``pool_size`` is the asyncpg pool's current size on PostgreSQL and the
      reader-pool bound on SQLite (which serializes writes onto one writer).
    - SQLite only: ``total_connections``, ``active_connections``,
      ``failed_connections``, ``write_queue_size``.
    - PostgreSQL only: ``pool_idle``, ``pool_min_size``, ``pool_max_size``, and
      -- once the corresponding detection has run -- ``pgpool_detected``,
      ``pgpool_version``, ``session_mode_pooler_detected``.
    """

    backend_type: str
    pool_size: int
    total_queries: int
    failed_queries: int
    circuit_state: str
    consecutive_failures: int
    last_error: str | None
    last_error_time: float | None
    # SQLite-only
    total_connections: int
    active_connections: int
    failed_connections: int
    write_queue_size: int
    # PostgreSQL-only
    pool_idle: int
    pool_min_size: int
    pool_max_size: int
    pgpool_detected: bool
    pgpool_version: str | None
    session_mode_pooler_detected: bool


class MostActiveThreadDict(TypedDict):
    """One row of ``most_active_threads`` in the stats response."""

    thread_id: str
    count: int


class TopTagDict(TypedDict):
    """One row of ``top_tags`` in the stats response."""

    tag: str
    count: int


class SemanticSearchStatsDict(TypedDict, total=False):
    """Type definition for the ``semantic_search`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.

    The field name ``embedding_count`` refers to the number of stored
    embedding rows on disk (one row per text chunk, since the storage layer
    persists one embedding vector per chunk). The text chunks themselves
    are NOT persisted; chunking is a transient pre-processing step purely
    for embedding quality.
    """

    enabled: bool
    available: bool
    backend: str
    model: str
    dimensions: int
    context_count: int  # COUNT(*) from embedding_metadata (entry count)
    embedding_count: int  # SUM(chunk_count) from embedding_metadata (stored embedding rows)
    average_chunks_per_entry: float
    coverage_percentage: float
    message: str  # Present when available is False


class FtsStatsDict(TypedDict, total=False):
    """Type definition for the ``fts`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.
    """

    enabled: bool
    available: bool
    language: str
    backend: str
    engine: str
    indexed_entries: int
    coverage_percentage: float
    message: str  # Present when available is False


class ChunkingStatsDict(TypedDict):
    """Type definition for the ``chunking`` sub-block in get_statistics.

    All fields ALWAYS present (no conditional shape).
    """

    enabled: bool
    available: bool
    chunk_size: int
    chunk_overlap: int
    aggregation: str


class RerankingStatsDict(TypedDict, total=False):
    """Type definition for the ``reranking`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.
    """

    enabled: bool
    available: bool
    provider: str
    model: str
    message: str  # Present when available is False


class SummaryStatsDict(TypedDict, total=False):
    """Type definition for the ``summary`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.
    """

    enabled: bool
    available: bool
    provider: str
    model: str
    summary_count: int
    coverage_percentage: float
    min_content_length: int
    message: str  # Present when available is False


class CompressionStatsDict(TypedDict, total=False):
    """Type definition for the ``compression`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason. Values
    originate from the singleton ``compression_metadata`` row (DB-truth)
    except ``max_concurrent`` which comes from runtime settings.
    """

    enabled: bool
    available: bool
    provider: str
    bits: int
    variant: str
    seed: int
    dim: int
    max_concurrent: int
    message: str  # Present when available is False


class IndexTreeStatsDict(TypedDict, total=False):
    """Type definition for the ``index_tree`` sub-block in get_statistics.

    ``enabled`` reflects ENABLE_INDEX_TREE_NODE_SUMMARIES. ``node_count`` is the
    total stored per-node summaries when the feature is enabled. It is 0 whenever
    the feature is off -- reported without querying, mirroring the compression
    block's enabled-gated pattern, so any residual rows from a previously-enabled
    run are intentionally NOT counted -- and 0 when the table is absent.
    """

    enabled: bool
    node_count: int


class GenerationStatsDict(TypedDict):
    """Type definition for the ``generation`` block in get_statistics.

    Counts the caller's readable entries whose generated outputs are outstanding:
    ``pending`` while their generation job waits or runs, ``failed`` once it spent
    every attempt. An entry whose outputs are complete is in neither count.
    """

    pending: int
    failed: int


class StatisticsResponseDict(TypedDict):
    """Type definition for the get_statistics tool response.

    Top-level fields and the ``generation`` block come from
    ``StatisticsRepository.get_database_statistics``; sub-blocks
    (``semantic_search``, ``fts``, ``chunking``, ``reranking``,
    ``summary``, ``compression``, ``index_tree``) are added by
    ``app.tools.discovery.get_statistics`` and are always present.

    ``database_size_mb`` reflects the whole database on PostgreSQL
    (``pg_database_size``) and the on-disk database file on SQLite. It is
    omitted only for in-memory or missing-file SQLite databases.

    ``embeddings_size_mb`` reflects the active vector payload table and is not
    byte-comparable across backends (on-disk relation size including indexes on
    PostgreSQL; raw payload bytes or an fp32 estimate on SQLite). When the
    SQLite figure is the fp32 estimate, ``embeddings_size_estimated`` is True.
    """

    total_entries: int
    by_source: dict[str, int]
    by_content_type: dict[str, int]
    total_images: int
    unique_tags: int
    total_threads: int
    avg_entries_per_thread: float
    most_active_threads: list[MostActiveThreadDict]
    top_tags: list[TopTagDict]
    backend: str
    generation: GenerationStatsDict
    database_size_mb: NotRequired[float]  # Whole DB on PostgreSQL; on-disk file on SQLite
    # Embedding vector payload size (not byte-comparable across backends)
    embeddings_size_mb: NotRequired[float]
    embeddings_size_estimated: NotRequired[bool]  # True only for the SQLite fp32 estimate
    connection_metrics: ConnectionMetricsDict
    semantic_search: SemanticSearchStatsDict
    fts: FtsStatsDict
    chunking: ChunkingStatsDict
    reranking: RerankingStatsDict
    summary: SummaryStatsDict
    compression: CompressionStatsDict
    # Always emitted by get_statistics (both toggle branches set it), so required.
    index_tree: IndexTreeStatsDict
