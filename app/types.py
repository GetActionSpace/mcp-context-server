"""Type definitions for the MCP context server.

This module provides type definitions to replace explicit Any usage
and ensure strict type safety throughout the codebase. The response
types of the discovery tools ``list_threads`` and ``get_statistics``
live in ``app/discovery_types.py``.
"""

from typing import Literal
from typing import NotRequired
from typing import Required
from typing import TypedDict

# JSON value types - recursive union for JSON-like data structures
type JsonValue = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']

# Metadata value types - simpler non-recursive type for metadata fields
type MetadataValue = str | int | float | bool | None

# Metadata dictionary type for use in models - supports nested JSON structures
type MetadataDict = dict[str, JsonValue]

# The stored visibility of an entry, as every entry read reports it.
type EntryVisibility = Literal['private', 'public']

# The state of an entry's generated outputs: 'ready' when the entry holds no generation job,
# 'pending' while its job waits or runs, 'failed' when its job spent every attempt.
type GenerationStatus = Literal['ready', 'pending', 'failed']


class EntryAccessFields(TypedDict):
    """The access fields every entry-carrying read response reports for one entry.

    ``visibility`` is the entry's stored visibility. ``is_owner`` says whether the
    caller owns the entry, the only role that may change its visibility or delete
    it; a principal reading through a grant or a ``public`` visibility sees false.
    The owner itself is never reported.
    """

    visibility: EntryVisibility
    is_owner: bool


# API Response TypedDicts for proper return type annotations
class ImageDict(TypedDict):
    """Type definition for image data in API responses.

    ``mime_type`` is always present; ``data`` is present only when images are fetched
    with ``include_data=True``; ``metadata`` is present only when the image carries
    metadata.

    Per-image ``metadata`` crosses the tool boundary as a JSON-ENCODED STRING (the
    write tools type ``images`` as ``list[dict[str, str]]``), so the read path
    round-trips that string back verbatim and the canonical returned shape is a
    ``str``. It is declared as an open ``JsonValue`` rather than ``str`` because
    this TypedDict is what FastMCP derives the strict ``get_context_by_ids``
    outputSchema from, and rows stored before the write-path shape check existed
    can hold any JSON value; a narrower declaration would make those entries
    permanently unreadable through the tool instead of merely unusual.
    """

    mime_type: str
    data: NotRequired[str]
    metadata: NotRequired[JsonValue]


class ContextEntryDict(TypedDict, total=False):
    """Type definition for context entry responses.

    Uses total=False to handle optional fields properly; ``visibility`` and
    ``is_owner`` (see :class:`EntryAccessFields`) and ``generation_status`` are
    required on every entry.
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str | None
    metadata: MetadataDict | None
    summary: str  # Conditional presence handled by TypedDict(total=False); absence == "feature disabled"
    created_at: str
    updated_at: str
    tags: list[str]
    images: list[ImageDict] | None
    is_text_content_truncated: bool | None
    visibility: Required[EntryVisibility]
    is_owner: Required[bool]
    generation_status: Required[GenerationStatus]


class ClampedLimitDict(TypedDict):
    """Type definition for clamped limit hint in search responses."""

    requested: int
    applied: int


class RankDepthLimitDict(TypedDict):
    """Type definition for the ranked-depth hint in ranked search responses.

    The ranked search tools (semantic, FTS, hybrid) decide their final order after
    the database returns rows, so they serve every page from ONE ordering of fixed
    depth (``rank_depth``); that is what keeps successive pages from repeating or
    skipping rows. A request whose ``offset + limit`` reaches past that depth is
    served short -- empty when the offset alone is past it -- and carries this hint,
    so a client can tell a window outside the ranked ordering apart from an
    exhausted result set.
    """

    requested_offset: int
    requested_limit: int
    rank_depth: int


class SearchContextResponseDict(TypedDict, total=False):
    """Type definition for search_context response.

    Uses total=False to handle optional fields (stats, error, validation_errors).
    """

    results: list[ContextEntryDict]
    count: int
    stats: dict[str, object] | None  # Only present when explain_query=True
    error: str | None
    validation_errors: list[str] | None
    clamped_limit: ClampedLimitDict | None


class GrepFileDict(TypedDict):
    """A grep result row in ``files_with_matches`` mode (locator only), with the entry's access fields and generation state."""

    context_id: str
    match_count: int
    visibility: EntryVisibility
    is_owner: bool
    generation_status: GenerationStatus


class GrepContentMatchDict(TypedDict):
    """A grep result row in ``content`` mode (one per match), with the access fields and generation state of the matched entry.

    Offsets are Unicode code-point indices into the entry's ``text_content`` and
    compose directly with ``read_context_range`` (start_char/end_char).
    """

    context_id: str
    line_number: int
    line: str
    match_start: int
    match_end: int
    before: list[str]
    after: list[str]
    visibility: EntryVisibility
    is_owner: bool
    generation_status: GenerationStatus


class GrepCountDict(TypedDict):
    """A grep result row in ``count`` mode (per-entry match tally), with the entry's access fields and generation state."""

    context_id: str
    count: int
    visibility: EntryVisibility
    is_owner: bool
    generation_status: GenerationStatus


class GrepContextResultDict(TypedDict, total=False):
    """Type definition for the grep_context response.

    ``results`` holds one of the three row shapes depending on ``mode``; ``error``
    and ``validation_errors`` appear only when a client-supplied filter is invalid
    (a metadata filter fails validation or the tags filter exceeds its member cap).
    ``timed_out_context_ids`` appears only when one or more entries were skipped
    because their regex match exceeded the per-entry timeout (the result is then
    incomplete for those entries).
    """

    mode: str
    total_matches: int
    truncated: bool
    results: list[GrepFileDict] | list[GrepContentMatchDict] | list[GrepCountDict]
    error: str | None
    validation_errors: list[str] | None
    timed_out_context_ids: list[str]


class ReadContextRangeDict(TypedDict):
    """Type definition for the read_context_range response.

    Echoes the RESOLVED span actually returned (after clamping out-of-range
    offsets to ``[0, len]``) so a caller always knows exactly what it received.
    Offsets are Unicode code-point indices. ``visibility`` and ``is_owner`` are the
    entry's access fields and ``generation_status`` its generation state.
    """

    context_id: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    text: str
    visibility: EntryVisibility
    is_owner: bool
    generation_status: GenerationStatus


class OutlineNodeDict(TypedDict, total=False):
    """A node in the navigate_context outline tree.

    ``char_start``/``char_end`` are Unicode code-point offsets feeding directly
    into read_context_range. ``summary`` mirrors the entry summary on the root
    (by reference) and carries an optional per-node LLM summary on descendants.
    """

    node_id: str
    level: int
    ordinal: int
    title: str
    char_start: int
    char_end: int
    summary: str | None
    children: list['OutlineNodeDict']


class NavigateContextResultDict(TypedDict, total=False):
    """Type definition for the navigate_context response.

    ``visibility`` and ``is_owner`` are the entry's access fields and
    ``generation_status`` its generation state, reported at the top level only,
    never on an outline node.
    """

    context_id: str
    total_chars: int
    node_count: int
    root: OutlineNodeDict
    visibility: Required[EntryVisibility]
    is_owner: Required[bool]
    generation_status: Required[GenerationStatus]


class StoreContextSuccessDict(TypedDict):
    """Type definition for successful store context response."""

    success: bool
    context_id: str
    thread_id: str
    message: str


class UpdateContextSuccessDict(TypedDict):
    """Type definition for successful update context response."""

    success: bool
    context_id: str
    updated_fields: list[str]
    message: str


# Sharing TypedDicts


class ShareContextResultDict(TypedDict):
    """Response of share_context: the grant it names and whether this call created it."""

    success: bool
    context_id: str
    principal_type: Literal['user', 'group']
    principal_id: str
    permission: Literal['read', 'write']
    created: bool
    message: str


class UnshareContextResultDict(TypedDict):
    """Response of unshare_context: the grant it names, whether this call removed it, and what the grantee keeps.

    ``remaining_permissions`` lists the permissions the same principal type and id
    still hold on the entry; access through a group the grantee belongs to is not
    counted.
    """

    success: bool
    context_id: str
    principal_type: Literal['user', 'group']
    principal_id: str
    permission: Literal['read', 'write']
    removed: bool
    remaining_permissions: list[Literal['read', 'write']]
    message: str


class ContextGrantDict(TypedDict):
    """One grant on an entry, as list_context_grants reports it to the entry's owner."""

    principal_type: Literal['user', 'group']
    principal_id: str
    permission: Literal['read', 'write']
    granted_by: str
    created_at: str


class ListContextGrantsResultDict(TypedDict):
    """Response of list_context_grants: every grant of the entry in principal type, id and permission order."""

    context_id: str
    grants: list[ContextGrantDict]
    count: int


# Bulk operation TypedDicts


class BulkStoreResultItemDict(TypedDict):
    """Type definition for a single result in bulk store response."""

    index: int
    success: bool
    context_id: str | None
    error: str | None


class BulkStoreResponseDict(TypedDict):
    """Type definition for bulk store response."""

    success: bool
    total: int
    succeeded: int
    failed: int
    results: list[BulkStoreResultItemDict]
    message: str


class BulkUpdateResultItemDict(TypedDict):
    """Type definition for a single result in bulk update response."""

    index: int
    context_id: str
    success: bool
    updated_fields: list[str] | None
    error: str | None


class BulkUpdateResponseDict(TypedDict):
    """Type definition for bulk update response."""

    success: bool
    total: int
    succeeded: int
    failed: int
    results: list[BulkUpdateResultItemDict]
    message: str


class BulkDeleteResponseDict(TypedDict):
    """Type definition for bulk delete response."""

    success: bool
    deleted_count: int
    criteria_used: list[str]
    message: str


# FTS (Full-Text Search) TypedDicts


class ScoresDict(TypedDict, total=False):
    """Unified scores breakdown for all search tools.

    All search tools return this structure with applicable fields populated:
    - FTS search: fts_score, fts_rank, rerank_score
    - Semantic search: semantic_distance, semantic_rank, rerank_score
    - Hybrid search: All fields

    Score Polarity Reference:
    - fts_score: HIGHER = better match (BM25/ts_rank relevance)
    - fts_rank: LOWER = better (1 = best)
    - semantic_distance: LOWER = better; L2 for fp32/mse, negated inner product for the ip compression variant
    - semantic_rank: LOWER = better (1 = best)
    - rrf: HIGHER = better (combined RRF score)
    - rerank_score: HIGHER = better (cross-encoder relevance, 0.0-1.0)
    """

    # FTS scores
    fts_score: float | None  # BM25/ts_rank relevance (HIGHER = better)
    fts_rank: int | None  # Rank in FTS results (1-based, LOWER = better)

    # Semantic scores
    semantic_distance: float | None  # Lower = better; L2 for fp32/mse, negated inner product for ip variant
    semantic_rank: int | None  # Rank in semantic results (1-based, LOWER = better)

    # RRF score (hybrid only)
    rrf: float | None  # Combined RRF score (HIGHER = better)

    # Rerank score (all tools when reranking enabled)
    rerank_score: float | None  # Cross-encoder relevance (HIGHER = better, 0.0-1.0)


class FtsSearchResultDict(TypedDict, total=False):
    """Type definition for FTS search result entry.

    The `scores` object contains:
    - fts_score: BM25/ts_rank relevance score (HIGHER = better)
    - fts_rank: Always null for standalone FTS (no ranking)
    - rerank_score: Present when reranking is enabled (HIGHER = better)

    ``visibility`` and ``is_owner`` (see :class:`EntryAccessFields`) and ``generation_status`` are
    required on every result.
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str
    summary: str
    is_text_content_truncated: bool
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    scores: ScoresDict
    highlighted: str | None
    images: list[ImageDict] | None
    visibility: Required[EntryVisibility]
    is_owner: Required[bool]
    generation_status: Required[GenerationStatus]


class FtsSearchResponseDict(TypedDict, total=False):
    """Type definition for fts_search_context response.

    Uses total=False to handle optional fields (stats, error, validation_errors).
    """

    query: str
    mode: str
    results: list[FtsSearchResultDict]
    count: int
    language: str
    stats: dict[str, object] | None  # Only present when explain_query=True
    error: str | None
    validation_errors: list[str] | None
    clamped_limit: ClampedLimitDict | None
    # Only present when the requested page reaches past the fixed ranked depth
    rank_depth_limit: RankDepthLimitDict | None


class SemanticSearchResultDict(TypedDict, total=False):
    """Type definition for semantic search result entry.

    The `scores` object contains:
    - semantic_distance: LOWER = more similar; L2 Euclidean for fp32/mse, negated inner product (~ -1..0) for the ip variant
    - semantic_rank: Always null for standalone semantic (no ranking)
    - rerank_score: Present when reranking is enabled (HIGHER = better)

    ``visibility`` and ``is_owner`` (see :class:`EntryAccessFields`) and ``generation_status`` are
    required on every result.
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str
    summary: str
    is_text_content_truncated: bool
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    scores: ScoresDict
    images: list[ImageDict] | None
    visibility: Required[EntryVisibility]
    is_owner: Required[bool]
    generation_status: Required[GenerationStatus]


class SemanticSearchResponseDict(TypedDict, total=False):
    """Type definition for semantic_search_context response.

    Uses total=False to handle optional fields (stats, error, validation_errors).
    """

    query: str
    results: list[SemanticSearchResultDict]
    count: int
    model: str
    stats: dict[str, object] | None  # Only present when explain_query=True
    error: str | None
    validation_errors: list[str] | None
    clamped_limit: ClampedLimitDict | None
    # Only present when the requested page reaches past the fixed ranked depth
    rank_depth_limit: RankDepthLimitDict | None


class FtsMigrationInProgressDict(TypedDict):
    """Type definition for FTS migration in progress response.

    Returned by fts_search_context when the FTS index is being rebuilt
    due to a language/tokenizer change. Provides informative feedback
    to clients with estimated completion time.
    """

    migration_in_progress: bool
    message: str
    started_at: str  # ISO 8601 timestamp
    estimated_remaining_seconds: int
    old_language: str
    new_language: str
    suggestion: str


# Hybrid Search TypedDicts


class HybridScoresDict(TypedDict, total=False):
    """Type definition for hybrid search scores breakdown.

    Contains scores from individual search methods and the combined RRF score.
    The `rerank_score` is present when reranking is enabled.
    """

    rrf: float  # Combined RRF score (HIGHER = better)
    fts_rank: int | None  # Rank in FTS results (1-based), None if not in FTS results
    semantic_rank: int | None  # Rank in semantic results (1-based), None if not in semantic results
    fts_score: float | None  # Original FTS score (BM25/ts_rank)
    semantic_distance: float | None  # Lower = better; L2 for fp32/mse, negated inner product for ip variant
    rerank_score: float | None  # Cross-encoder reranking score (HIGHER = better, 0.0-1.0)


class HybridFusedResultDict(TypedDict, total=False):
    """A hybrid search result as reciprocal rank fusion builds it from the two legs.

    Combines fields from both FTS and semantic search results with
    hybrid-specific scoring information; ``visibility`` and ``is_owner``
    (see :class:`EntryAccessFields`) are required on every result. The search
    display format turns it into a :class:`HybridSearchResultDict`.
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str
    summary: str
    is_text_content_truncated: bool
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    scores: HybridScoresDict  # Hybrid scoring breakdown
    rerank_text: str | None  # Internal: chunk text for reranking (removed before API response)
    images: list[ImageDict] | None
    visibility: Required[EntryVisibility]
    is_owner: Required[bool]


class HybridSearchResultDict(HybridFusedResultDict, total=False):
    """Type definition for hybrid search result entry.

    A fused result (see :class:`HybridFusedResultDict`) with its entry's
    ``generation_status``, which is required on every result beside
    ``visibility`` and ``is_owner``.
    """

    generation_status: Required[GenerationStatus]


# Hybrid Search Stats TypedDicts (for explain_query parameter)
# NOTE: Stats types defined before HybridSearchResponseDict to avoid forward reference


class HybridFtsStatsDict(TypedDict, total=False):
    """Type definition for FTS statistics in hybrid search.

    Contains timing and filter information from the FTS portion
    of hybrid search. This is also the shape of the ``stats`` payload the
    standalone ``fts_search_context`` tool returns, including on its
    validation-error path, so ``backend`` (the active storage backend type)
    is always present alongside the timing and filter counters.
    """

    execution_time_ms: float
    filters_applied: int
    rows_returned: int
    query_plan: str | None
    backend: str


class HybridSemanticStatsDict(TypedDict, total=False):
    """Type definition for semantic search statistics in hybrid search.

    Contains timing and filter information from the semantic portion
    of hybrid search. This is also the shape of the ``stats`` payload the
    standalone ``semantic_search_context`` tool returns, including on its
    validation-error path, so ``backend`` (the active storage backend type)
    is always present alongside the timing and filter counters.
    ``embedding_generation_ms`` is the measured wall-clock duration (rounded,
    milliseconds) of the query ``embed_query`` call, injected by
    ``semantic_search_raw`` (app/tools/search/legs.py) so both the standalone tool and the semantic leg
    of hybrid search surface it (zeroed on the validation-error path).
    """

    execution_time_ms: float
    embedding_generation_ms: float
    filters_applied: int
    rows_returned: int
    backend: str
    query_plan: str | None


class HybridFusionStatsDict(TypedDict):
    """Type definition for RRF fusion statistics in hybrid search.

    Contains overlap and distribution information about how results
    were combined from FTS and semantic search.
    """

    rrf_k: int
    total_unique_documents: int
    documents_in_both: int
    documents_fts_only: int
    documents_semantic_only: int


class HybridSearchStatsDict(TypedDict, total=False):
    """Type definition for complete hybrid search statistics.

    Aggregates stats from FTS, semantic search, and fusion operations.
    Only present in response when explain_query=True.
    """

    execution_time_ms: float  # Total hybrid search time
    fts_stats: HybridFtsStatsDict | None
    semantic_stats: HybridSemanticStatsDict | None
    fusion_stats: HybridFusionStatsDict
    adaptive_fts_mode: str  # 'match' or 'boolean' - indicates FTS mode used


class HybridSearchResponseDict(TypedDict, total=False):
    """Type definition for hybrid_search_context response.

    Uses total=False to handle optional stats field.
    """

    query: str
    results: list[HybridSearchResultDict]
    count: int
    fusion_method: str  # 'rrf'
    search_modes_used: list[str]  # Actual modes executed, e.g., ['fts', 'semantic']
    fts_count: int  # Number of results from FTS
    semantic_count: int  # Number of results from semantic search
    stats: HybridSearchStatsDict | None
    clamped_limit: ClampedLimitDict | None
    # Only present when the requested page reaches past the fixed ranked depth
    rank_depth_limit: RankDepthLimitDict | None
    warnings: list[str] | None  # Degradation warnings when sub-searches fail (e.g., FTS or semantic)
    error: str | None  # Set when all available modes failed on filter validation
    # Per-filter details captured from a failing sub-search. Present both when all
    # available modes failed (alongside ``error``) and when the response is only
    # partially degraded (one mode succeeded while the other failed validation), so a
    # client can correct an invalid sub-query even when results were still returned.
    validation_errors: list[str] | None
