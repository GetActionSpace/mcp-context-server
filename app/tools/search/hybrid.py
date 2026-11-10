"""The hybrid_search_context tool: FTS and semantic search fused with RRF and reranked once.

Also holds the helpers only hybrid search uses: the per-leg overfetch clamp, the zeroed
fusion stats of a request that ran no leg, and the adaptive AND/OR FTS query preparation.
"""

import asyncio
import json
import logging
import re
from collections.abc import Coroutine
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.repositories.fts_repository.query import sanitize_sqlite_fts_terms
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_reranking_provider
from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range
from app.tools._validation import reject_unstorable_input
from app.tools.search.legs import fts_search_raw
from app.tools.search.legs import semantic_search_raw
from app.tools.search.limits import MAX_FILTER_TAGS
from app.tools.search.limits import MAX_FTS_QUERY_LENGTH
from app.tools.search.limits import MAX_METADATA_FILTERS
from app.tools.search.limits import MAX_METADATA_KEYS
from app.tools.search.limits import MAX_SEARCH_LIMIT
from app.tools.search.limits import MAX_SEARCH_OFFSET
from app.tools.search.limits import RANKED_SEARCH_DEPTH
from app.tools.search.limits import empty_page_beyond_rank_depth
from app.tools.search.limits import filter_caps_error
from app.tools.search.limits import rank_depth_hint
from app.tools.search.limits import structural_filter_errors
from app.tools.search.ranking import apply_reranking
from app.tools.search.ranking import apply_search_display_format

logger = logging.getLogger(__name__)
settings = get_settings()


# Hard ceiling for a computed candidate window before it reaches a LIMIT/OFFSET bind.
# The largest legitimate window is RANKED_SEARCH_DEPTH rows times the biggest overfetch
# factor the module applies (HYBRID_RRF_OVERFETCH, max 10) for the hybrid legs, so this
# ceiling keeps an order of magnitude of headroom above it. It never clamps a valid
# request, because the window does not grow with the requested page; it is defense in
# depth so no factor change can grow the bind out of range.
MAX_OVERFETCH_ROWS = RANKED_SEARCH_DEPTH * 200


def _clamp_overfetch(rows: int) -> int:
    """Clamp a computed candidate window to the safe ceiling before it is bound.

    Args:
        rows: The computed candidate window (the ranked depth times the overfetch
            factors).

    Returns:
        The window bounded to MAX_OVERFETCH_ROWS so the row count reaching a
        LIMIT/OFFSET bind stays independent of the individual overfetch multipliers.
    """
    return min(rows, MAX_OVERFETCH_ROWS)


def _zeroed_fusion_stats(rrf_k: int) -> dict[str, Any]:
    """Build the zeroed fusion_stats dict for a hybrid validation-error response.

    Carries the same keys as the success path's fusion_stats (including the
    resolved ``rrf_k``) with all document counters at zero, so a client sees the
    same stats keys whether the hybrid search executed or failed validation.

    Args:
        rrf_k: The resolved RRF smoothing constant for this request.

    Returns:
        The zeroed fusion_stats dict.
    """
    return {
        'rrf_k': rrf_k,
        'total_unique_documents': 0,
        'documents_in_both': 0,
        'documents_fts_only': 0,
        'documents_semantic_only': 0,
    }


def _prepare_hybrid_fts_query(
    query: str,
    or_threshold: int,
    backend_type: str,
    language: str = 'english',
) -> tuple[str, Literal['match', 'boolean']]:
    """Prepare FTS query with adaptive AND/OR logic for hybrid search.

    Short queries (below threshold) use 'match' mode (AND logic).
    Long queries (at or above threshold) use 'boolean' mode with
    OR keywords inserted between terms, enabling partial-match recall.

    Quoted phrases (e.g., "error handling") are preserved as single
    tokens to maintain phrase semantics in boolean mode.

    Args:
        query: Original search query.
        or_threshold: Minimum significant terms to switch to OR mode.
        backend_type: Storage backend type ('sqlite' or 'postgresql').
        language: Configured FTS_LANGUAGE; governs whether and/or/not operator barewords are
            dropped as stopwords on the SQLite OR-join path (to mirror PostgreSQL for that
            language, so the two backends do not diverge for a non-English deployment).

    Returns:
        Tuple of (transformed_query, fts_mode) where fts_mode is
        either 'match' or 'boolean'.
    """
    tokens = re.findall(r'"[^"]*"|\S+', query.strip())
    significant = [t for t in tokens if len(t.strip('"')) > 1]
    use_or = len(significant) >= or_threshold

    if backend_type == 'sqlite':
        # Short ('match') queries return the RAW query so the single canonical 'match'
        # transform in transform_query_sqlite sanitizes them EXACTLY ONCE -- identically to
        # standalone fts_search_context. Long ('OR') queries are sanitized HERE because
        # boolean mode is passed through transform_query_sqlite unchanged; a bare FTS5
        # operator/special char left on that path raises 'fts5: syntax error' (the recurring
        # crash), so the surviving significant terms are OR-joined for partial-match recall.
        #
        # The match case must NOT pre-sanitize: transform_query_sqlite('match') re-runs
        # sanitize_sqlite_fts_terms over whatever it is handed, so pre-sanitizing here would
        # send an ALREADY-QUOTED term list through a second wrapping pass. Returning the raw
        # query keeps exactly one transform between the user's text and MATCH, which is what
        # guarantees hybrid recall is identical to standalone fts_search_context for the same
        # input rather than merely similar.
        if not use_or:
            return query, 'match'
        terms = sanitize_sqlite_fts_terms(significant, language)
        if not terms:
            # Every term was an operator/stopword bareword: return the '' match-nothing
            # sentinel (re-transformed by _search_sqlite to an empty result), in parity with
            # PostgreSQL's empty tsquery. A literal-phrase fallback ('"and"') would instead
            # MATCH every document containing the dropped word on SQLite (FTS5 keeps stopwords
            # as tokens) while PostgreSQL returns zero -- a cross-backend divergence.
            return '', 'match'
        return ' OR '.join(terms), 'boolean'  # FTS5 OR is uppercase (case-sensitive)

    # PostgreSQL: plainto_tsquery / websearch_to_tsquery sanitize operators and stopwords
    # server-side, so a raw query never raises -- only the OR-mode join needs building. Hyphens
    # are replaced to prevent websearch_to_tsquery NOT interpretation; quoted phrases preserved.
    if not use_or:
        return query, 'match'
    sanitized: list[str] = []
    for token in significant:
        # >= 2 chars so a lone '"' is never treated as a balanced phrase (consistent with the
        # shared SQLite sanitizer; the `significant` filter already excludes it here too).
        if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
            sanitized.append(token)
            continue
        # SPLIT a bare token on an embedded double quote, exactly as
        # sanitize_sqlite_fts_terms does, instead of emitting it raw. The fragments are
        # joined into the server-synthesized ' or ' string below, so an unbalanced quote
        # passed through verbatim would open a websearch_to_tsquery phrase that swallows
        # every following OR term into an adjacency phrase -- collapsing hybrid recall to
        # near zero for ordinary input such as don"t or a pasted KeyError: "foo".
        for fragment in token.replace('-', ' ').split('"'):
            clean = fragment.strip()
            if not clean:
                continue
            if ' ' in clean:
                # A hyphenated fragment becomes a QUOTED PHRASE so its parts keep
                # ordered-adjacency semantics (websearch_to_tsquery parses "a b"
                # as a <-> b), matching the SQLite branch where
                # sanitize_sqlite_fts_terms wraps the same fragment as the FTS5
                # phrase literal "a b". Left unquoted, websearch_to_tsquery ANDs
                # the parts (unordered), so the two backends would return
                # different recall for the identical hybrid query. The token
                # itself contains no whitespace (re.findall '\\S+'), so a space
                # here can only come from the hyphen replacement.
                sanitized.append(f'"{clean}"')
                continue
            sanitized.append(clean)
    if not sanitized:
        return query, 'match'
    return ' or '.join(sanitized), 'boolean'  # websearch_to_tsquery 'or' is lowercase


async def hybrid_search_context(
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_FTS_QUERY_LENGTH,
            description=f'Natural language search query (1-{MAX_FTS_QUERY_LENGTH} characters)',
        ),
    ],
    limit: Annotated[int, Field(ge=1, description='Maximum results to return (1-100, default: 5)')] = 5,
    offset: Annotated[int, Field(ge=0, le=MAX_SEARCH_OFFSET, description='Pagination offset (default: 0)')] = 0,
    fusion_method: Annotated[
        Literal['rrf'],
        Field(description="Fusion algorithm: 'rrf' (Reciprocal Rank Fusion, default)"),
    ] = 'rrf',
    rrf_k: Annotated[
        int | None,
        Field(
            ge=1,
            le=1000,
            description='RRF smoothing constant (default from HYBRID_RRF_K env var, typically 60). '
            'Higher values give more weight to lower-ranked documents.',
        ),
    ] = None,
    thread_id: Annotated[str | None, Field(min_length=1, description='Optional filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Optional filter by source type')] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None, Field(description='Filter by content type (text or multimodal)'),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(max_length=MAX_FILTER_TAGS, description=f'Filter by any of these tags (OR logic; at most {MAX_FILTER_TAGS})'),
    ] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(
            max_length=MAX_METADATA_KEYS,
            description=f'Simple metadata filters (key=value equality; at most {MAX_METADATA_KEYS} keys)',
        ),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            max_length=MAX_METADATA_FILTERS,
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            f'starts_with, ends_with, is_null, is_not_null, array_contains. At most {MAX_METADATA_FILTERS} filters',
        ),
    ] = None,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
) -> dict[str, Any]:
    """Hybrid search combining FTS and semantic search with Reciprocal Rank Fusion (RRF).

    Executes both full-text search and semantic search in parallel, then fuses results.
    Documents appearing in both result sets score higher.

    Graceful degradation:
    - If only FTS is available, returns FTS results only
    - If only semantic search is available, returns semantic results only
    - If neither is available, raises ToolError

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags; at most 100 tags per request)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching (at most 100 keys per request)
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.);
      at most 100 filters per request, in/not_in value lists accept at most 100 members

    The `scores` object contains:
    - rrf: Combined fusion score (HIGHER = better)
    - fts_rank: Rank in full-text results (LOWER = better, 1 = best)
    - semantic_rank: Rank in semantic results (LOWER = better, 1 = best)
    - fts_score: BM25/ts_rank relevance (HIGHER = better match). On SQLite the BM25
      score draws on statistics of the whole FTS5 index, entries the caller cannot
      read included, so the scores of readable entries and their relative order can
      shift as other principals' entries change, while the set of matching entries
      and the number of results per page depend only on what the caller can read.
      PostgreSQL's ts_rank_cd scores each entry on its own.
    - semantic_distance: LOWER = more similar (L2 for fp32/mse storage, negated inner product for the ip compression variant)
    - rerank_score: Cross-encoder relevance (HIGHER = better), present when reranking enabled

    When explain_query=True, the `stats` field contains:
    - execution_time_ms: Total hybrid search time
    - fts_stats: {execution_time_ms, filters_applied, rows_returned, backend, query_plan} or None
    - semantic_stats: {execution_time_ms, embedding_generation_ms, filters_applied, rows_returned, backend, query_plan} or None
    - fusion_stats: {rrf_k, total_unique_documents, documents_in_both, documents_fts_only, documents_semantic_only}
    - adaptive_fts_mode: 'match' or 'boolean' (the FTS mode the adaptive AND/OR switch selected)

    Pagination:
    - A query has ONE ranking, at most 100 rows deep; limit and offset select a window
      inside it, so paging never repeats a row on two pages nor skips one entirely.
    - A window reaching past that depth is served short (empty when the offset itself is
      past it) and the response carries rank_depth_limit.

    Returns:
        Dict with query (str), results (list with id, thread_id, source,
        text_content (truncated), summary, is_text_content_truncated,
        metadata, scores, tags, visibility, is_owner (whether the caller owns
        the entry), generation_status ('ready' once the entry's embeddings and
        summaries are generated, 'pending' while their generation waits or runs,
        'failed' when it gave up after its retries; an entry without embeddings
        yet is found by the full-text leg only)), count (int), fusion_method (str),
        search_modes_used (list), fts_count (int), semantic_count (int), stats (only
        when explain_query=True), and rank_depth_limit (only when the requested
        window reaches past the ranked depth).

    Raises:
        ToolError: If hybrid search is not available or all search modes fail.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Reject an embedded NUL or unpaired UTF-16 surrogate in thread_id/tags before
    # they reach the PostgreSQL bind, where asyncpg would raise a non-ControlFlowError
    # that charges the circuit breaker (SQLite binds them silently -- a divergence).
    reject_unstorable_input(thread_id=thread_id, tags=tags)

    # Resolve rrf_k before any early return: the validation-error stats blocks
    # below carry the resolved constant in their zeroed fusion_stats.
    effective_rrf_k = rrf_k if rrf_k is not None else settings.hybrid_search.rrf_k

    # Boundary cap re-check behind the wire-schema max_length: an oversized tags
    # list, metadata dict, or metadata_filters list is a structured validation
    # error before any sub-search runs, mirroring the all-modes-failed response
    # shape below. The stats block is attached under explain_query so a validation
    # rejection and a successful search expose the same keys (uniform with the
    # standalone fts/semantic siblings); no sub-search ran, so the sub-stats are
    # None and the fusion counters zero.
    caps_error = filter_caps_error(tags, metadata, metadata_filters)
    if caps_error is not None:
        caps_response: dict[str, Any] = {
            'query': query,
            'results': [],
            'count': 0,
            'fusion_method': fusion_method,
            'search_modes_used': [],
            'fts_count': 0,
            'semantic_count': 0,
            'error': caps_error,
            'validation_errors': [caps_error],
        }
        if explain_query:
            caps_response['stats'] = {
                'execution_time_ms': 0.0,
                'fts_stats': None,
                'semantic_stats': None,
                'fusion_stats': _zeroed_fusion_stats(effective_rrf_k),
                'adaptive_fts_mode': 'match',
            }
        return caps_response

    # Check if hybrid search is enabled
    if not settings.hybrid_search.enabled:
        raise ToolError(
            'Hybrid search is not available. '
            'Set ENABLE_HYBRID_SEARCH to auto (default) or true to enable this feature. '
            'Also ensure ENABLE_FTS and/or ENABLE_SEMANTIC_SEARCH are not force-disabled.',
        )

    # Determine available search modes
    fts_available = settings.fts.enabled
    embedding_provider = get_embedding_provider()
    semantic_available = settings.semantic_search.enabled and embedding_provider is not None

    # Collect all available modes
    available_modes: list[str] = []
    if fts_available:
        available_modes.append('fts')
    if semantic_available:
        available_modes.append('semantic')

    if not available_modes:
        unavailable_reasons: list[str] = []
        if not fts_available:
            unavailable_reasons.append('FTS requires ENABLE_FTS not force-disabled (auto/true)')
        if not semantic_available:
            unavailable_reasons.append(
                f'Semantic search requires ENABLE_SEMANTIC_SEARCH not force-disabled (auto/true) and '
                f'{settings.embedding.provider} provider properly configured',
            )
        raise ToolError(
            f'No search modes available. '
            f'Issues: {"; ".join(unavailable_reasons)}',
        )

    try:
        # Clamp limit to prevent excessive memory use (Postel's Law for LLM clients)
        original_limit = limit
        if limit > MAX_SEARCH_LIMIT:
            limit = MAX_SEARCH_LIMIT
            logger.warning(
                'hybrid_search_context: requested limit=%d exceeds maximum %d, clamped to %d',
                original_limit, MAX_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
            )

        import time as time_module

        total_start_time = time_module.time()

        # Import fusion module
        from app.fusion import count_unique_results
        from app.fusion import reciprocal_rank_fusion

        # Get repositories for tag/image enrichment
        repos = await ensure_repositories()

        # Per-leg candidate depth: the fixed ranked depth times the RRF overfetch
        # factor, so each leg reaches far enough for fusion to find the documents the
        # two modes share. It is page-independent (see RANKED_SEARCH_DEPTH) because
        # RRF is itself a reordering stage even with the cross-encoder absent -- a
        # document present in BOTH legs outscores a higher-placed document present in
        # only one -- so legs sized from the requested page fused into a different
        # ordering for every page.
        reranking_provider = get_reranking_provider()
        over_fetch_limit = _clamp_overfetch(RANKED_SEARCH_DEPTH * settings.hybrid_search.rrf_overfetch)

        # Determine if we need highlights for internal reranking
        need_highlight_for_rerank = reranking_provider is not None and settings.reranking.enabled

        # Execute searches in parallel using raw functions (Layer 1 - no reranking)
        fts_results: list[dict[str, Any]] = []
        semantic_results: list[dict[str, Any]] = []
        fts_error: str | None = None
        semantic_error: str | None = None
        # Structured validation details captured from FtsValidationError /
        # MetadataFilterValidationError, so the all-modes-failed response can
        # carry the same validation_errors key the sibling search tools return
        # instead of flattening them into an opaque string.
        fts_validation_errors: list[str] | None = None
        semantic_validation_errors: list[str] | None = None

        # Stats collection for explain_query
        fts_stats: dict[str, Any] | None = None
        semantic_stats: dict[str, Any] | None = None

        from app.repositories.embedding_repository.records import MetadataFilterValidationError
        from app.repositories.fts_repository.faults import FtsValidationError

        # Determine adaptive FTS mode for hybrid search
        adaptive_query, adaptive_mode = _prepare_hybrid_fts_query(
            query=query,
            or_threshold=settings.hybrid_search.fts_or_threshold,
            backend_type=settings.storage.backend_type,
            language=settings.fts.language,
        )

        # A page starting at or past the ranked depth is deterministically empty, so it is
        # answered from the client's arguments alone rather than after both legs run --
        # which on this tool means an embedding round trip, two searches, RRF fusion and a
        # cross-encoder pass whose every row the slice would then discard. An unbindable
        # query or a structurally invalid filter still takes precedence: the request falls
        # through to the normal path, whose per-leg degradation reports it. No leg executed,
        # so search_modes_used is empty and the counters are zero; rank_depth_limit says why.
        from app.repositories.fts_repository.query import fts_query_validation_errors

        if (
            offset >= RANKED_SEARCH_DEPTH
            and fts_query_validation_errors(query) is None
            and structural_filter_errors(tags, metadata, metadata_filters) is None
        ):
            depth_stats: dict[str, Any] | None = None
            if explain_query:
                depth_stats = {
                    'execution_time_ms': round((time_module.time() - total_start_time) * 1000, 2),
                    'fts_stats': None,
                    'semantic_stats': None,
                    'fusion_stats': _zeroed_fusion_stats(effective_rrf_k),
                    'adaptive_fts_mode': adaptive_mode,
                }
            return empty_page_beyond_rank_depth(
                {
                    'query': query,
                    'fusion_method': fusion_method,
                    'search_modes_used': [],
                    'fts_count': 0,
                    'semantic_count': 0,
                },
                offset=offset,
                limit=limit,
                original_limit=original_limit,
                stats=depth_stats,
            )

        # Both legs run as the caller through one scope object, so the fusion, the
        # reranking, the counts and the explain stats all derive from rows it may read.
        scope = resolve_access_scope()

        async def run_fts_search() -> None:
            nonlocal fts_results, fts_error, fts_stats, fts_validation_errors
            try:
                results, stats = await fts_search_raw(
                    query=adaptive_query,
                    limit=over_fetch_limit,
                    mode=adaptive_mode,
                    offset=0,
                    thread_id=thread_id,
                    source=source,
                    content_type=content_type,
                    tags=tags,
                    start_date=start_date,
                    end_date=end_date,
                    metadata=metadata,
                    metadata_filters=metadata_filters,
                    highlight=False,  # Client doesn't want highlighted field in hybrid
                    internal_highlight_for_rerank=need_highlight_for_rerank,
                    explain_query=explain_query,
                    repos=repos,
                    scope=scope,
                )
                fts_results = results
                if explain_query:
                    fts_stats = stats
            except FtsValidationError as e:
                fts_error = e.message
                fts_validation_errors = e.validation_errors
            except ToolError as e:
                fts_error = format_exception_message(e)
            except Exception as e:
                fts_error = format_exception_message(e)

        async def run_semantic_search() -> None:
            nonlocal semantic_results, semantic_error, semantic_stats, semantic_validation_errors
            try:
                results, stats = await semantic_search_raw(
                    query=query,
                    limit=over_fetch_limit,
                    offset=0,
                    thread_id=thread_id,
                    source=source,
                    content_type=content_type,
                    tags=tags,
                    start_date=start_date,
                    end_date=end_date,
                    metadata=metadata,
                    metadata_filters=metadata_filters,
                    extract_rerank_text=need_highlight_for_rerank,
                    explain_query=explain_query,
                    repos=repos,
                    embedding_provider=embedding_provider,
                    scope=scope,
                )
                semantic_results = results
                if explain_query:
                    semantic_stats = stats
            except MetadataFilterValidationError as e:
                semantic_error = e.message
                semantic_validation_errors = e.validation_errors
            except ToolError as e:
                semantic_error = format_exception_message(e)
            except Exception as e:
                semantic_error = format_exception_message(e)

        # Run searches in parallel
        tasks: list[Coroutine[Any, Any, None]] = []
        if 'fts' in available_modes:
            tasks.append(run_fts_search())
        if 'semantic' in available_modes:
            tasks.append(run_semantic_search())

        await asyncio.gather(*tasks)

        # Log individual sub-search failures for observability.
        # When only one sub-search fails, the other's results are used (graceful degradation),
        # but the failure should not be silently swallowed.
        if fts_error and not semantic_error:
            logger.warning(
                'Hybrid search: FTS sub-search failed (semantic succeeded): %s',
                fts_error,
            )
        if semantic_error and not fts_error:
            logger.warning(
                'Hybrid search: semantic sub-search failed (FTS succeeded): %s',
                semantic_error,
            )

        # Build warnings list for client visibility when sub-searches degrade.
        # Include the specific sub-search failure so a permanently-invalid
        # sub-query (e.g. bad metadata_filters reaching only one mode) is
        # correctable from the response instead of hidden behind a generic notice.
        search_warnings: list[str] = []
        if fts_error:
            search_warnings.append(
                f'FTS sub-search failed; results may be based on semantic search only. FTS: {fts_error}',
            )
        if semantic_error:
            search_warnings.append(
                f'Semantic sub-search failed; results may be based on FTS only. Semantic: {semantic_error}',
            )

        # Deduplicate validation details captured from either sub-search once;
        # both the all-modes-failed response below and the partial-degradation
        # response surface them under the same validation_errors key.
        combined_raw: list[str] = [
            *(fts_validation_errors or []),
            *(semantic_validation_errors or []),
        ]
        combined_validation_errors = list(dict.fromkeys(combined_raw))

        # Determine which available modes executed successfully (not errored).
        modes_used: list[str] = []
        if 'fts' in available_modes and not fts_error:
            modes_used.append('fts')
        if 'semantic' in available_modes and not semantic_error:
            modes_used.append('semantic')

        # If NO available mode succeeded, surface the error instead of returning
        # an empty success. Requiring both modes to have failed would miss a
        # single-mode deployment whose only available mode errored (e.g. an
        # invalid metadata_filters), masking the failure as zero results.
        if not modes_used:
            details = '. '.join(
                part for part in (
                    f'FTS: {fts_error}' if fts_error else '',
                    f'Semantic: {semantic_error}' if semantic_error else '',
                ) if part
            )
            # When a sub-search failed on FILTER VALIDATION, return the same
            # structured error response the sibling search tools produce (error
            # + validation_errors, count 0) instead of an opaque raised
            # ToolError: the caller needs the per-filter details to correct the
            # query rather than retry a permanently-invalid request. Both modes
            # validate the SAME filters and build identical messages, so the
            # order-preserving dedup above keeps each defect listed once.
            # The error response carries the same always-present response-shape keys
            # the success path, the docstring, and HybridSearchResponseDict declare
            # (fusion_method, fts_count, semantic_count), so a client reading them
            # never hits a KeyError on the error branch. Under explain_query it also
            # carries the same stats keys the success path builds: the real elapsed
            # time, whatever sub-search stats were captured before the failure (None
            # when a mode failed validation), a zeroed fusion_stats with the resolved
            # rrf_k, and the adaptive FTS mode the request selected.
            if combined_validation_errors:
                failed_response: dict[str, Any] = {
                    'query': query,
                    'results': [],
                    'count': 0,
                    'fusion_method': fusion_method,
                    'search_modes_used': [],
                    'fts_count': 0,
                    'semantic_count': 0,
                    'error': f'All available search modes failed. {details}',
                    'validation_errors': combined_validation_errors,
                }
                if explain_query:
                    failed_response['stats'] = {
                        'execution_time_ms': round((time_module.time() - total_start_time) * 1000, 2),
                        'fts_stats': fts_stats,
                        'semantic_stats': semantic_stats,
                        'fusion_stats': _zeroed_fusion_stats(effective_rrf_k),
                        'adaptive_fts_mode': adaptive_mode,
                    }
                return failed_response
            raise ToolError(f'All available search modes failed. {details}')

        # Parse FTS metadata (returned as JSON strings from DB)
        for result in fts_results:
            metadata_raw = result.get('metadata')
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):
                try:
                    result['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    result['metadata'] = None

        # Fuse results using RRF (no reranking yet - Layer 1 results only) down to the
        # fixed ranked depth, so the fused ordering every page is cut from is the same.
        fused_results = reciprocal_rank_fusion(
            fts_results=fts_results,
            semantic_results=semantic_results,
            k=effective_rrf_k,
            limit=RANKED_SEARCH_DEPTH,
        )

        # Apply reranking (Layer 3 - single reranking after fusion)
        fused_results_any: list[dict[str, Any]] = cast(list[dict[str, Any]], fused_results)
        reranked_results = await apply_reranking(
            query=query,
            results=fused_results_any,
            limit=RANKED_SEARCH_DEPTH,
            reranking_provider=reranking_provider,
        )

        # Cut the requested page out of the single ranked ordering
        final_results = reranked_results[offset:][:limit]

        # Clean up internal fields from final results
        for result in final_results:
            result.pop('rerank_text', None)  # Internal field, not for client

        # Enrich results with tags and optionally images
        for result in final_results:
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(str(context_id), scope=scope)
                result['tags'] = tags_result
                # Fetch images if requested and applicable
                if include_images and result.get('content_type') == 'multimodal':
                    images_result = await repos.images.get_images_for_context(str(context_id), include_data=True, scope=scope)
                    result['images'] = images_result
            else:
                result['tags'] = []

        # Apply unified search display formatting, the results' generation states included
        await apply_search_display_format(final_results, repos, scope=scope)

        logger.info(
            f'Hybrid search found {len(final_results)} results for query: "{query[:50]}..." '
            f'(fts={len(fts_results)}, semantic={len(semantic_results)}, '
            f'available={available_modes}, executed={modes_used})',
        )

        # Build response
        response: dict[str, Any] = {
            'query': query,
            'results': final_results,
            'count': len(final_results),
            'fusion_method': fusion_method,
            'search_modes_used': modes_used,
            'fts_count': len(fts_results),
            'semantic_count': len(semantic_results),
        }

        # Add stats if explain_query is enabled
        if explain_query:
            # Calculate fusion stats
            fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)
            fusion_stats: dict[str, Any] = {
                'rrf_k': effective_rrf_k,
                'total_unique_documents': fts_only + semantic_only + overlap,
                'documents_in_both': overlap,
                'documents_fts_only': fts_only,
                'documents_semantic_only': semantic_only,
            }

            # Calculate total execution time
            total_execution_time_ms = (time_module.time() - total_start_time) * 1000

            response['stats'] = {
                'execution_time_ms': round(total_execution_time_ms, 2),
                'fts_stats': fts_stats,
                'semantic_stats': semantic_stats,
                'fusion_stats': fusion_stats,
                'adaptive_fts_mode': adaptive_mode,
            }

        if original_limit != limit:
            response['clamped_limit'] = {
                'requested': original_limit,
                'applied': limit,
            }
        depth_hint = rank_depth_hint(offset, limit)
        if depth_hint is not None:
            response['rank_depth_limit'] = depth_hint
        if search_warnings:
            response['warnings'] = search_warnings
        # A partially-degraded response carries the captured per-filter details
        # under the same validation_errors key the all-modes-failed branch uses,
        # so a client can fix an invalid sub-query even when the other mode
        # produced results.
        if combined_validation_errors:
            response['validation_errors'] = combined_validation_errors
        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in hybrid search: {e}')
        raise ToolError(f'Hybrid search failed: {format_exception_message(e)}') from e
