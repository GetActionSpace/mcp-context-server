"""The semantic_search_context tool: vector similarity search with optional reranking."""

import logging
from typing import Annotated
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_reranking_provider
from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range
from app.tools._validation import reject_unstorable_input
from app.tools.search.legs import semantic_search_raw
from app.tools.search.limits import MAX_FILTER_TAGS
from app.tools.search.limits import MAX_METADATA_FILTERS
from app.tools.search.limits import MAX_METADATA_KEYS
from app.tools.search.limits import MAX_SEARCH_LIMIT
from app.tools.search.limits import MAX_SEARCH_OFFSET
from app.tools.search.limits import RANKED_SEARCH_DEPTH
from app.tools.search.limits import empty_page_beyond_rank_depth
from app.tools.search.limits import empty_stats_for_unexecuted_query
from app.tools.search.limits import filter_caps_error
from app.tools.search.limits import rank_depth_hint
from app.tools.search.limits import structural_filter_errors
from app.tools.search.ranking import apply_reranking
from app.tools.search.ranking import apply_search_display_format

logger = logging.getLogger(__name__)
settings = get_settings()


async def semantic_search_context(
    query: Annotated[str, Field(min_length=1, description='Natural language search query')],
    limit: Annotated[int, Field(ge=1, description='Maximum results to return (1-100, default: 5)')] = 5,
    offset: Annotated[int, Field(ge=0, le=MAX_SEARCH_OFFSET, description='Pagination offset (default: 0)')] = 0,
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
    """Find semantically similar context with optional metadata filtering.

    Finds entries with similar MEANING even without matching keywords.
    Use for: finding related concepts, similar discussions, thematic grouping.

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags; at most 100 tags per request)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching (at most 100 keys per request)
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.);
      at most 100 filters per request, in/not_in value lists accept at most 100 members

    The `scores` object contains:
    - semantic_distance: LOWER = more similar. The metric depends on embedding storage:
      Euclidean L2 (>= 0) for uncompressed/mse storage, or a negated inner product
      (~ -1..0 for normalized embeddings, where more negative = more similar) when the
      default ip compression variant is active. Compare values within one result set
      rather than against fixed thresholds, since the range differs by storage variant.
    - semantic_rank: Always null for standalone semantic search
    - rerank_score: Cross-encoder relevance (HIGHER = better), present when reranking enabled

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
        yet is never a result)), count (int), model (str), stats (only when
        explain_query=True), and rank_depth_limit (only when the requested window
        reaches past the ranked depth).

    Raises:
        ToolError: If semantic search is not available or search operation fails.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Reject an embedded NUL or unpaired UTF-16 surrogate in thread_id/tags before
    # they reach the PostgreSQL bind, where asyncpg would raise a non-ControlFlowError
    # that charges the circuit breaker (SQLite binds them silently -- a divergence).
    reject_unstorable_input(thread_id=thread_id, tags=tags)

    # Boundary cap re-check behind the wire-schema max_length: an oversized tags
    # list, metadata dict, or metadata_filters list is a structured validation
    # error before any repository work runs.
    caps_error = filter_caps_error(tags, metadata, metadata_filters)
    if caps_error is not None:
        caps_error_response: dict[str, Any] = {
            'query': query,
            'results': [],
            'count': 0,
            'model': settings.embedding.model,
            'error': caps_error,
            'validation_errors': [caps_error],
        }
        if explain_query:
            caps_error_response['stats'] = empty_stats_for_unexecuted_query(include_embedding_ms=True)
        return caps_error_response

    try:
        # Clamp limit to prevent excessive memory use (Postel's Law for LLM clients)
        original_limit = limit
        if limit > MAX_SEARCH_LIMIT:
            limit = MAX_SEARCH_LIMIT
            logger.warning(
                'semantic_search_context: requested limit=%d exceeds maximum %d, clamped to %d',
                original_limit, MAX_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
            )

        # The candidate depth is fixed and page-independent (see
        # RANKED_SEARCH_DEPTH), so every page is cut from the SAME ordering. It also
        # over-fetches for the reranker on its own: the cross-encoder scores the full
        # ranked window however few rows the caller asked for.
        reranking_provider = get_reranking_provider()
        need_reranking = reranking_provider is not None and settings.reranking.enabled
        embedding_provider = get_embedding_provider()

        # Import exception here to avoid circular imports at module level
        from app.repositories.embedding_repository.records import MetadataFilterValidationError

        # A page starting at or past the ranked depth is deterministically empty, so it is
        # answered from the client's arguments alone rather than after an embedding round
        # trip, a vector search and a cross-encoder pass whose every row the slice below
        # would then discard. A structurally invalid filter still takes precedence: the
        # request falls through to the normal path, which rejects it (before the embedding
        # call) with the per-filter detail the client needs.
        if offset >= RANKED_SEARCH_DEPTH and structural_filter_errors(tags, metadata, metadata_filters) is None:
            return empty_page_beyond_rank_depth(
                {'query': query, 'model': settings.embedding.model},
                offset=offset,
                limit=limit,
                original_limit=original_limit,
                stats=empty_stats_for_unexecuted_query(include_embedding_ms=True) if explain_query else None,
            )

        repos = await ensure_repositories()
        # The search runs as the caller: entries it may not read never take a rank
        # position, a page slot or a count.
        scope = resolve_access_scope()

        try:
            # Call raw search (Layer 1) for the full ranked depth
            # Extract rerank_text (matched chunk) when reranking is enabled
            search_results, search_stats = await semantic_search_raw(
                query=query,
                limit=RANKED_SEARCH_DEPTH,
                offset=0,  # The page is cut from the ranked window below
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata=metadata,
                metadata_filters=metadata_filters,
                extract_rerank_text=need_reranking,
                explain_query=explain_query,
                repos=repos,
                embedding_provider=embedding_provider,
                scope=scope,
            )
        except MetadataFilterValidationError as e:
            # Return error response (unified with search_context behavior)
            error_response: dict[str, Any] = {
                'query': query,
                'results': [],
                'count': 0,
                'model': settings.embedding.model,
                'error': e.message,
                'validation_errors': e.validation_errors,
            }
            if explain_query:
                # Uniform validation-error stats shape, mirroring the FTS error
                # path: zeroed counters plus the always-present backend key.
                error_response['stats'] = empty_stats_for_unexecuted_query(include_embedding_ms=True)
            return error_response

        # Transform results to use scores object (before reranking)
        for result in search_results:
            # Move distance into scores object
            distance_value = result.pop('distance', None)
            result['scores'] = {
                'semantic_distance': distance_value,
                'semantic_rank': None,  # Standalone semantic has no ranking
            }

        # Apply reranking (Layer 2) if available, over the whole fixed-depth window
        reranked_results = await apply_reranking(
            query=query,
            results=search_results,
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

        logger.info(f'Semantic search found {len(final_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'results': final_results,
            'count': len(final_results),
            'model': settings.embedding.model,
        }
        if explain_query:
            response['stats'] = search_stats
        if original_limit != limit:
            response['clamped_limit'] = {
                'requested': original_limit,
                'applied': limit,
            }
        depth_hint = rank_depth_hint(offset, limit)
        if depth_hint is not None:
            response['rank_depth_limit'] = depth_hint
        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in semantic search: {e}')
        raise ToolError(f'Semantic search failed: {format_exception_message(e)}') from e
