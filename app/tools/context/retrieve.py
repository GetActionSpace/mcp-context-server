"""The get_context_by_ids tool: fetch context entries by ID with their full text."""

import json
import logging
from typing import Annotated
from typing import cast

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.ids import resolve_or_normalize_ids
from app.repositories.base import canonical_timestamp
from app.settings import get_settings
from app.startup import ensure_repositories
from app.tools._responses import entry_access_fields
from app.tools._responses import generation_statuses
from app.types import ContextEntryDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_context_by_ids(
    context_ids: Annotated[
        list[str],
        Field(min_length=1, max_length=100, description='List of context entry IDs to retrieve (max 100 per call)'),
    ],
    include_images: Annotated[bool, Field(description='Whether to include image data')] = True,
) -> list[ContextEntryDict]:
    """Fetch specific context entries by their IDs with FULL (non-truncated) text content.

    Use this when you have specific context IDs from previous operations
    and need the complete, untruncated content.

    Full IDs of entries that do not exist or that the caller may not read are
    silently skipped alike; only the readable entries found are returned. An ID
    may also be an 8-31 character hex prefix of a readable entry; a malformed ID,
    or a prefix matching no readable entry or several, fails the whole call.
    Accepts at most 100 IDs per call (the same cap as the batch tools); an
    oversized list is rejected at the tool boundary as a validation error
    before any database work. Fetch larger sets in successive calls.

    Returns:
        List of ContextEntryDict with id, thread_id, source, content_type, text_content,
        metadata, tags, created_at, updated_at, visibility, is_owner and
        generation_status fields, plus images (each with mime_type, base64 data and any
        image metadata) on an entry whose content_type is 'multimodal' when include_images
        is true; a text entry, and every entry fetched with include_images false, has no
        images key. is_owner says whether the caller owns the entry,
        and generation_status is 'ready' once the entry's embeddings and summaries
        are generated, 'pending' while their generation waits or runs, and 'failed'
        when it gave up after its retries. The summary field follows a
        tri-state contract controlled by GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY:

        - When disabled (the default), the summary key is omitted entirely; consumers
          reading entry.get('summary') will receive None, which is the conventional
          Python signal for "feature disabled, no value to surface".
        - When enabled and the stored summary is a non-empty string, the value is
          returned verbatim.
        - When enabled but the stored summary is NULL or empty (e.g., generation was
          skipped because text was shorter than SUMMARY_MIN_CONTENT_LENGTH, or no
          provider is configured), the value is normalized to an empty string ''.
          This mirrors the search-tool contract (search tools always emit summary as
          a string, never None) and provides an explicit "feature on, no data yet"
          signal distinct from the "feature disabled" None.

    Raises:
        ToolError: If an ID is malformed, if a prefix matches no readable entry
            or several, or if fetching context entries fails.
    """
    try:
        # Get repositories first; prefix resolution below needs the context repo.
        repos = await ensure_repositories()
        # Every read below runs as the caller, so an entry it may not read is
        # skipped exactly like a missing one.
        scope = resolve_access_scope()

        # Resolve incoming IDs at the boundary: accept full 32/36-char IDs or
        # 8-31 char hex prefixes (uniform with update_context/delete_context).
        try:
            context_ids = await resolve_or_normalize_ids(context_ids, repos.context, scope=scope)
        except ValueError as e:
            raise ToolError(f'Invalid context ID: {e}') from e

        # Fetch the readable entries; tags, images and generation states below are read only for these ids.
        rows = await repos.context.get_by_ids(context_ids, scope=scope)
        statuses = await generation_statuses((str(row['id']) for row in rows), repos, scope=scope)
        entries: list[ContextEntryDict] = []
        include_summary = settings.retrieval.include_summary

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))
            access = entry_access_fields(row)
            entry['visibility'] = access['visibility']
            entry['is_owner'] = access['is_owner']
            entry['generation_status'] = statuses[str(row['id'])]

            # Canonical timestamp wire format: identical across SQLite and PostgreSQL
            # and to the search tools (text_content stays FULL here -- get_context_by_ids
            # is never truncated). See app.repositories.base.canonical_timestamp.
            created_at_val = entry.get('created_at')
            if created_at_val is not None:
                entry['created_at'] = cast(str, canonical_timestamp(created_at_val))
            updated_at_val = entry.get('updated_at')
            if updated_at_val is not None:
                entry['updated_at'] = cast(str, canonical_timestamp(updated_at_val))

            if include_summary:
                # Mirror search-tool normalization (apply_search_display_format in app/tools/search/ranking.py):
                # surface an empty string for the "feature ON but no data yet" state.
                # Tri-state contract:
                #   include_summary=False (default)              -> key omitted (consumers see entry.get('summary') == None)
                #   include_summary=True  + stored non-empty str -> verbatim stored string
                #   include_summary=True  + DB NULL/empty        -> '' (explicit signal "feature on, no data yet")
                summary = entry.get('summary')
                if isinstance(summary, str) and summary.strip():
                    entry['summary'] = summary
                else:
                    entry['summary'] = ''
            else:
                entry.pop('summary', None)

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

            # Fetch images
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id_img = entry.get('id')
                if entry_id_img is not None:
                    images_result = await repos.images.get_images_for_context(
                        str(entry_id_img), include_data=True, scope=scope,
                    )
                    entry['images'] = images_result
                else:
                    entry['images'] = []

            entries.append(entry)

        return entries
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error fetching context by IDs: {e}')
        raise ToolError(f'Failed to fetch context entries: {format_exception_message(e)}') from e
