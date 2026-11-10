"""The search_context tool: filter and browse context entries without a free-text query."""

import json
import logging
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.startup import ensure_repositories
from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range
from app.tools._validation import reject_unstorable_input
from app.tools.search.limits import MAX_FILTER_TAGS
from app.tools.search.limits import MAX_METADATA_FILTERS
from app.tools.search.limits import MAX_METADATA_KEYS
from app.tools.search.limits import MAX_SEARCH_LIMIT
from app.tools.search.limits import MAX_SEARCH_OFFSET
from app.tools.search.limits import empty_stats_for_unexecuted_query
from app.tools.search.limits import filter_caps_error
from app.tools.search.ranking import apply_search_display_format
from app.types import ContextEntryDict

logger = logging.getLogger(__name__)


async def search_context(
    limit: Annotated[int, Field(ge=1, description='Maximum results to return (1-100, default: 30)')] = 30,
    thread_id: Annotated[str | None, Field(min_length=1, description='Filter by thread (indexed)')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Filter by source type (indexed)')] = None,
    tags: Annotated[
        list[str] | None,
        Field(max_length=MAX_FILTER_TAGS, description=f'Filter by any of these tags (OR logic; at most {MAX_FILTER_TAGS})'),
    ] = None,
    content_type: Annotated[Literal['text', 'multimodal'] | None, Field(description='Filter by content type')] = None,
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
    offset: Annotated[int, Field(ge=0, le=MAX_SEARCH_OFFSET, description='Pagination offset (default: 0)')] = 0,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
) -> dict[str, Any]:
    """Search context entries with filtering. Returns TRUNCATED text_content.

    Filtering options:
    - thread_id, source: Indexed for fast filtering (always prefer specifying thread_id)
    - tags: OR logic (matches ANY of provided tags; at most 100 tags per request)
    - metadata: Simple key=value equality matching (at most 100 keys per request)
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.);
      at most 100 filters per request, in/not_in value lists accept at most 100 members
    - start_date/end_date: Filter by creation timestamp (ISO 8601)

    Returns:
        Dict with results (list of ContextEntryDict with truncated text_content,
        summary always present as string, is_text_content_truncated flag, visibility,
        is_owner telling whether the caller owns the entry, and generation_status:
        'ready' once the entry's embeddings and summaries are generated, 'pending'
        while their generation waits or runs, 'failed' when it gave up after its
        retries), count (int), and stats (dict, only when explain_query=True).

    Raises:
        ToolError: If search operation fails.
    """
    try:
        # Clamp limit to prevent excessive memory use (Postel's Law for LLM clients)
        original_limit = limit
        if limit > MAX_SEARCH_LIMIT:
            limit = MAX_SEARCH_LIMIT
            logger.warning(
                'search_context: requested limit=%d exceeds maximum %d, clamped to %d',
                original_limit, MAX_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
            )

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
        # error before any repository work runs. The stats block is attached under
        # explain_query so a validation rejection and a successful search expose the
        # same keys (uniform with the fts/semantic siblings).
        caps_error = filter_caps_error(tags, metadata, metadata_filters)
        if caps_error is not None:
            caps_response: dict[str, Any] = {
                'results': [],
                'count': 0,
                'error': caps_error,
                'validation_errors': [caps_error],
            }
            if explain_query:
                caps_response['stats'] = empty_stats_for_unexecuted_query()
            return caps_response

        repos = await ensure_repositories()
        # The search runs as the caller: entries it may not read never match, never
        # take a page slot and never count.
        scope = resolve_access_scope()

        result = await repos.context.search_contexts(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            metadata=metadata,
            metadata_filters=metadata_filters,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
            explain_query=explain_query,
            scope=scope,
        )

        # Always expect tuple from repository
        rows, stats = result

        # Check for validation errors in stats
        if 'error' in stats:
            # Return the error response with validation details. The stats block is
            # attached under explain_query so a validation rejection and a successful
            # search expose the same keys (uniform with the fts/semantic siblings).
            error_response: dict[str, Any] = {
                'results': [],
                'count': 0,
                'error': stats.get('error', 'Unknown error'),
            }
            if 'validation_errors' in stats:
                error_response['validation_errors'] = stats['validation_errors']
            if explain_query:
                error_response['stats'] = empty_stats_for_unexecuted_query()
            return error_response

        entries: list[ContextEntryDict] = []

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))

            # Parse JSON metadata - database stores as JSON string
            metadata_raw = entry.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    entry['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    entry['metadata'] = None

            # Get normalized tags
            entry_id_raw = entry.get('id')
            if entry_id_raw is not None:
                entry_id = str(entry_id_raw)
                tags_result = await repos.tags.get_tags_for_context(entry_id, scope=scope)
                entry['tags'] = tags_result
            else:
                entry['tags'] = []

            # Fetch images if requested and applicable
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id = str(entry.get('id', ''))
                images_result = await repos.images.get_images_for_context(entry_id, include_data=True, scope=scope)
                entry['images'] = images_result

            entries.append(entry)

        # Apply unified search display formatting, the entries' access fields and generation states included
        await apply_search_display_format(cast(list[dict[str, Any]], entries), repos, scope=scope)

        # Return dict with results, count, and optional stats
        response: dict[str, Any] = {'results': entries, 'count': len(entries)}
        if explain_query:
            response['stats'] = stats
        if original_limit != limit:
            response['clamped_limit'] = {
                'requested': original_limit,
                'applied': limit,
            }
        return response
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error searching context: {e}')
        raise ToolError(f'Failed to search context: {format_exception_message(e)}') from e
