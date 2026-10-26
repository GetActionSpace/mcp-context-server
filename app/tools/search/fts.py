"""The fts_search_context tool: full-text search with linguistic analysis and optional reranking."""

import json
import logging
from datetime import UTC
from datetime import datetime
from typing import Annotated
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.migrations import get_fts_migration_status
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_reranking_provider
from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range
from app.tools._validation import reject_unstorable_input
from app.tools.search.legs import fts_search_raw
from app.tools.search.limits import MAX_FILTER_TAGS
from app.tools.search.limits import MAX_FTS_QUERY_LENGTH
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


async def fts_search_context(
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_FTS_QUERY_LENGTH,
            description=f'Full-text search query (1-{MAX_FTS_QUERY_LENGTH} characters)',
        ),
    ],
    limit: Annotated[int, Field(ge=1, description='Maximum results to return (1-100, default: 5)')] = 5,
    mode: Annotated[
        Literal['match', 'prefix', 'phrase', 'boolean'],
        Field(
            description="Search mode: 'match' (default, natural language), "
            "'prefix' (wildcard with *), 'phrase' (exact phrase), "
            "'boolean' (AND/OR/NOT operators)",
        ),
    ] = 'match',
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
    offset: Annotated[int, Field(ge=0, le=MAX_SEARCH_OFFSET, description='Pagination offset (default: 0)')] = 0,
    highlight: Annotated[bool, Field(description='Include highlighted snippets in results')] = False,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
) -> dict[str, Any]:
    """Full-text search with linguistic analysis (stemming, ranking, boolean queries).

    Search modes:
    - match: Natural language query (default) - words are stemmed and matched
    - prefix: Wildcard search - "search*" matches "searching", "searched"
    - phrase: Exact phrase matching - "exact phrase" must appear as-is
    - boolean: Boolean operators - "python AND (async OR await) NOT blocking"

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags; at most 100 tags per request)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching (at most 100 keys per request)
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.);
      at most 100 filters per request, in/not_in value lists accept at most 100 members

    Only entries the caller may read are matched, returned and counted.

    The `scores` object contains:
    - fts_score: BM25/ts_rank relevance (HIGHER = better match). On SQLite the BM25
      score draws on statistics of the whole FTS5 index, entries the caller cannot
      read included, so the scores of readable entries and their relative order can
      shift as other principals' entries change, while the set of matching entries
      and the number of results per page depend only on what the caller can read.
      PostgreSQL's ts_rank_cd scores each entry on its own.
    - fts_rank: Always null for standalone FTS search
    - rerank_score: Cross-encoder relevance (HIGHER = better), present when reranking enabled

    Pagination:
    - A query has ONE ranking, at most 100 rows deep; limit and offset select a window
      inside it, so paging never repeats a row on two pages nor skips one entirely.
    - A window reaching past that depth is served short (empty when the offset itself is
      past it) and the response carries rank_depth_limit.

    Returns:
        Dict with query (str), mode (str), results (list with id, thread_id, source,
        text_content (truncated), summary, is_text_content_truncated,
        metadata, scores, highlighted, tags, visibility, is_owner (whether the
        caller owns the entry)), count (int), language (str),
        stats (only when explain_query=True), and rank_depth_limit (only when the
        requested window reaches past the ranked depth).

    Raises:
        ToolError: If FTS is not available or search operation fails.
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
            'mode': mode,
            'results': [],
            'count': 0,
            'language': settings.fts.language,
            'error': caps_error,
            'validation_errors': [caps_error],
        }
        if explain_query:
            caps_error_response['stats'] = empty_stats_for_unexecuted_query()
        return caps_error_response

    # Check if migration is in progress - return informative response for graceful degradation
    fts_status = get_fts_migration_status()
    if fts_status.in_progress:
        if fts_status.started_at is not None and fts_status.estimated_seconds is not None:
            elapsed = (datetime.now(tz=UTC) - fts_status.started_at).total_seconds()
            remaining = max(0, fts_status.estimated_seconds - int(elapsed))
        else:
            remaining = 60  # Default estimate if no timing info available

        old_lang = fts_status.old_language or 'unknown'
        new_lang = fts_status.new_language or settings.fts.language

        return {
            'migration_in_progress': True,
            'message': f'FTS index is being rebuilt with language/tokenizer "{new_lang}". '
            'Search functionality will be available shortly.',
            'started_at': fts_status.started_at.isoformat() if fts_status.started_at else '',
            'estimated_remaining_seconds': remaining,
            'old_language': old_lang,
            'new_language': new_lang,
            'suggestion': f'Please retry in {remaining + 5} seconds.',
        }

    try:
        # Clamp limit to prevent excessive memory use (Postel's Law for LLM clients)
        original_limit = limit
        if limit > MAX_SEARCH_LIMIT:
            limit = MAX_SEARCH_LIMIT
            logger.warning(
                'fts_search_context: requested limit=%d exceeds maximum %d, clamped to %d',
                original_limit, MAX_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
            )

        # The candidate depth is fixed and page-independent (see
        # RANKED_SEARCH_DEPTH), so every page is cut from the SAME ordering.
        reranking_provider = get_reranking_provider()

        # Determine if we need highlights for internal reranking
        need_highlight_for_rerank = reranking_provider is not None and settings.reranking.enabled

        # Import exception here to avoid circular imports
        from app.repositories.fts_repository.faults import FtsValidationError
        from app.repositories.fts_repository.query import fts_query_validation_errors

        # A page starting at or past the ranked depth is deterministically empty, so it is
        # answered from the client's arguments alone rather than after a full-depth search
        # and a cross-encoder pass over up to RANKED_SEARCH_DEPTH documents whose every row
        # the slice below would then discard. An unbindable query or a structurally invalid
        # filter still takes precedence: the request falls through to the normal path,
        # which rejects it with the detail the client needs to correct it.
        if (
            offset >= RANKED_SEARCH_DEPTH
            and fts_query_validation_errors(query) is None
            and structural_filter_errors(tags, metadata, metadata_filters) is None
        ):
            return empty_page_beyond_rank_depth(
                {'query': query, 'mode': mode, 'language': settings.fts.language},
                offset=offset,
                limit=limit,
                original_limit=original_limit,
                stats=empty_stats_for_unexecuted_query() if explain_query else None,
            )

        repos = await ensure_repositories()
        # The search runs as the caller: entries it may not read never take a rank
        # position, a page slot or a count.
        scope = resolve_access_scope()

        try:
            # Call raw search (Layer 1) for the full ranked depth
            search_results, stats = await fts_search_raw(
                query=query,
                limit=RANKED_SEARCH_DEPTH,
                mode=mode,
                offset=0,  # The page is cut from the ranked window below
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata=metadata,
                metadata_filters=metadata_filters,
                highlight=highlight,
                internal_highlight_for_rerank=need_highlight_for_rerank,
                explain_query=explain_query,
                repos=repos,
                scope=scope,
            )
        except FtsValidationError as e:
            # Return error response (unified with search_context behavior)
            error_response: dict[str, Any] = {
                'query': query,
                'mode': mode,
                'results': [],
                'count': 0,
                'language': settings.fts.language,
                'error': e.message,
                'validation_errors': e.validation_errors,
            }
            if explain_query:
                error_response['stats'] = empty_stats_for_unexecuted_query()
            return error_response

        # Transform results to use scores object (before reranking)
        for result in search_results:
            # Move score into scores object
            fts_score_value = result.pop('score', None)
            result['scores'] = {
                'fts_score': fts_score_value,
                'fts_rank': None,  # Standalone FTS has no ranking
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

        # Process results: parse metadata and enrich with tags
        for result in final_results:
            # Parse JSON metadata - database stores as JSON string
            metadata_raw = result.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    result['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    result['metadata'] = None

            # Get normalized tags
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

        logger.info(f'FTS search found {len(final_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'mode': mode,
            'results': final_results,
            'count': len(final_results),
            'language': settings.fts.language,
        }
        if explain_query:
            response['stats'] = stats
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
        logger.error(f'Error in FTS search: {e}')
        raise ToolError(f'FTS search failed: {format_exception_message(e)}') from e
