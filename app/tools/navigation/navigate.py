"""The ``navigate_context`` tool: an on-demand Markdown outline (index_tree) of one context entry.

The outline lets an agent orient in a long record before reading it. Every node
carries ``char_start``/``char_end`` in the Unicode code-point offset contract that
``read_context_range`` reads, and a node's ``node_id`` addresses the same span
there -- navigate, then extract.

The response reports the entry's ``visibility``, whether the caller owns it
(``is_owner``) and the state of its generated outputs (``generation_status``).
"""

import asyncio
import logging
from typing import Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.ids import resolve_or_normalize_id
from app.repositories.index_node_repository import StoredNodeSummaries
from app.services.outline_service import ROOT_NODE_ID
from app.services.outline_service import OutlineNode
from app.services.outline_service import count_nodes
from app.services.outline_service import parse_outline
from app.services.text_lines import should_offload_line_scan
from app.settings import get_settings
from app.startup import ensure_repositories
from app.tools._responses import entry_access_fields
from app.tools._responses import generation_statuses
from app.types import EntryAccessFields
from app.types import NavigateContextResultDict
from app.types import OutlineNodeDict

logger = logging.getLogger(__name__)
settings = get_settings()

# The outline parse over unbounded stored entry text is LINE-oriented, so it gates on
# should_offload_line_scan (app.services.text_lines) rather than on a size-only
# threshold, like the line split of read_context_range. The grep matcher keeps its own
# size-based gate (grep_service._OFFLOAD_MIN_CHARS) because its literal path is a single
# C-level whole-text scan and its regex path offloads unconditionally, so there size
# really is the cost driver.


def _outline_to_dict(node: OutlineNode, stored_nodes: StoredNodeSummaries) -> OutlineNodeDict:
    """Convert an OutlineNode (and its children) to a response dict.

    Each node's ``summary`` is looked up in the stored per-node LLM summaries
    (empty when that layer is off): primarily by ``node_id``, then by exact
    ``(char_start, char_end)`` span. A ``node_id`` hit is trusted ONLY when
    the stored row's span equals the computed node's span: node ids are
    algorithm-versioned, so a row written by an OLDER slug algorithm can
    collide with a DIFFERENT section's current id (a pre-cap sibling whose
    short title equals the capped prefix of a longer sibling's slug) and
    would otherwise attach that sibling's summary here. The span fallback
    then re-attaches rows whose node_id the current parse no longer computes
    for the SAME section -- spans derive from the text rather than the slug,
    and node rows are replaced on every text change, so an exact-span match
    against the same revision is deterministic. The caller overrides the
    root node's summary with the entry summary afterward.

    Returns:
        The node as an OutlineNodeDict, children converted recursively.
    """
    stored = stored_nodes.by_node_id.get(node.node_id)
    summary = (
        stored.summary
        if stored is not None and (stored.char_start, stored.char_end) == (node.char_start, node.char_end)
        else None
    )
    if summary is None and node.node_id != ROOT_NODE_ID:
        # The synthetic root is excluded from the span fallback: its summary
        # mirrors the entry summary by reference (assigned by the caller), and
        # a document-spanning first heading shares the root's span, so the
        # fallback would leak that section's summary onto the root.
        summary = stored_nodes.by_span.get((node.char_start, node.char_end))
    return {
        'node_id': node.node_id,
        'level': node.level,
        'ordinal': node.ordinal,
        'title': node.title,
        'char_start': node.char_start,
        'char_end': node.char_end,
        'summary': summary,
        'children': [_outline_to_dict(child, stored_nodes) for child in node.children],
    }


async def navigate_context(
    context_id: Annotated[
        str,
        Field(min_length=8, description='Context entry id (32/36-char UUID or 8-31 char hex prefix)'),
    ],
    max_depth: Annotated[
        int,
        Field(ge=1, le=6, description='Deepest Markdown heading level to include (deeper headings fold into their section)'),
    ] = 6,
    include_node_summaries: Annotated[
        bool,
        Field(description='Attach stored per-node LLM summaries to descendant nodes (when that layer is enabled)'),
    ] = False,
) -> NavigateContextResultDict:
    """Build a navigable Markdown outline (index_tree) for one context entry.

    Parses the entry's headings on demand into a hierarchical table of contents
    so an agent can orient in a long record before reading. Each node carries
    ``char_start``/``char_end`` (Unicode code-point offsets) that feed straight
    into ``read_context_range`` (by offset or by ``node_id``). The synthetic root
    spans the whole document and its ``summary`` mirrors the entry's stored
    summary by reference; an entry with no headings yields a childless root.

    The tree itself is code-derived (no LLM), computed fresh each call, so it is
    never stale and works for every entry. When ``include_node_summaries`` is True
    and the per-node summary layer is enabled, descendant nodes are enriched with
    their stored LLM section abstracts.

    Returns:
        NavigateContextResultDict with ``context_id``, ``total_chars``,
        ``node_count`` (headings, excluding the root), ``root`` -- a recursive
        node carrying ``node_id``, ``level``, ``ordinal``, ``title``, ``char_start``,
        ``char_end``, ``summary`` and ``children`` -- and the entry's ``visibility``,
        ``is_owner`` (whether the caller owns it) and ``generation_status``
        ('ready' once the entry's embeddings and summaries are generated,
        'pending' while their generation waits or runs, 'failed' when it gave up
        after its retries).

    Raises:
        ToolError: If the id is invalid or the entry is not found.
    """
    try:
        repos = await ensure_repositories()
        # Every read below runs as the caller: an entry it may not read is not found,
        # and its node summaries are never read.
        scope = resolve_access_scope()
        try:
            resolved_id = await resolve_or_normalize_id(context_id, repos.context, scope=scope)
        except ValueError as exc:
            raise ToolError(f'Invalid context ID: {exc}') from exc

        # The entry row and the stored node summaries live in different tables and
        # are read on separate connections (no shared snapshot). A text-change
        # update commits new text AND replacement node rows atomically, so a commit
        # landing between the two reads would pair the OLD text's outline with the
        # NEW node summaries -- mis-attaching a post-edit section summary to a
        # pre-edit span via a retained heading slug. Bracket both reads with probes
        # of the monotonic version token (bumped by exactly the writes that can
        # replace node rows) and re-take the whole snapshot (bounded) when the
        # token moved inside the window; the race is milliseconds-wide, so one
        # retry virtually always converges. updated_at is NOT a usable probe:
        # SQLite CURRENT_TIMESTAMP has second granularity, so two commits inside
        # the same wall-clock second compare equal and a torn pairing would be
        # accepted. The probes also catch a revocation: when an unshare removes the
        # caller's read access inside the window, the closing probe finds no readable
        # entry (no version), the snapshot is retaken, and the retaken read reports
        # the entry as not found, so node summaries read inside the window are never
        # returned.
        text = ''
        root_summary: str | None = None
        access: EntryAccessFields = {'visibility': 'private', 'is_owner': False}
        stored_nodes = StoredNodeSummaries({}, {})
        want_node_summaries = include_node_summaries and settings.index_tree.node_summaries_enabled
        for _snapshot_attempt in range(3):
            version_before: int | None = None
            if want_node_summaries:
                version_before = (await repos.context.check_entry_exists(resolved_id, scope=scope)).version
            rows = await repos.context.get_by_ids([resolved_id], scope=scope)
            if not rows:
                raise ToolError(f'Context entry not found: {context_id}')
            row = rows[0]
            text_value = row['text_content']
            text = text_value if text_value is not None else ''
            access = entry_access_fields(row)

            # Root summary mirrors the live summary column by reference; an empty or
            # whitespace-only stored summary normalizes to None per the optional-field
            # convention (OutlineNodeDict.summary is str | None), which is distinct from
            # the search tools' '' (empty string) for an absent summary. The flat
            # summary column is never overloaded.
            summary_value = row['summary']
            root_summary = summary_value if isinstance(summary_value, str) and summary_value.strip() else None

            if not want_node_summaries:
                break

            # Stored per-node summaries are fetched by id and do NOT depend on the
            # parse, so they are loaded before the CPU-bound parse + serialize block.
            stored_nodes = await repos.index_nodes.get_nodes_for_context(resolved_id, scope=scope)
            version_after = (await repos.context.check_entry_exists(resolved_id, scope=scope)).version
            if version_after == version_before:
                break
            logger.debug(
                'navigate_context %s: entry changed between text and node-summary '
                'reads; retaking snapshot', resolved_id,
            )
        else:
            logger.warning(
                'navigate_context %s: snapshot retries exhausted under sustained '
                'concurrent updates; the returned outline and node summaries may '
                'pair text and section abstracts from different revisions',
                resolved_id,
            )

        def _build_outline() -> tuple[OutlineNodeDict, int]:
            built = parse_outline(text, max_depth=max_depth)
            return _outline_to_dict(built, stored_nodes), count_nodes(built)

        # parse_outline + serialization are CPU work whose cost tracks LINE count
        # over unbounded entry text; a large OR line-dense entry is offloaded to a
        # worker thread so it cannot pin the event loop (see should_offload_line_scan).
        if should_offload_line_scan(text):
            root_dict, node_count = await asyncio.to_thread(_build_outline)
        else:
            root_dict, node_count = _build_outline()

        # The root mirrors the entry summary by reference; the stored node
        # summaries never contain the synthetic root id, and the span fallback
        # excludes the root explicitly.
        root_dict['summary'] = root_summary

        statuses = await generation_statuses([resolved_id], repos, scope=scope)

        return {
            'context_id': resolved_id,
            'total_chars': len(text),
            'node_count': node_count,
            'root': root_dict,
            'visibility': access['visibility'],
            'is_owner': access['is_owner'],
            'generation_status': statuses[resolved_id],
        }
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error in navigate_context: {exc}')
        raise ToolError(f'Failed to navigate context: {format_exception_message(exc)}') from exc
