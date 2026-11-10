"""The ``read_context_range`` tool: partial read of one context entry by character range, line range, or outline node.

It slices the full stored ``text_content`` directly, so it works for every entry,
and addresses it in the one Unicode code-point offset contract that ``grep_context``
match offsets and ``navigate_context`` node offsets share -- locate, then extract.

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
from app.services.outline_service import resolve_node_span
from app.services.text_lines import line_index_for_offset
from app.services.text_lines import should_offload_line_scan
from app.services.text_lines import split_lines_with_offsets
from app.startup import ensure_repositories
from app.tools._responses import entry_access_fields
from app.tools._responses import generation_statuses
from app.types import ReadContextRangeDict

logger = logging.getLogger(__name__)

# The line split and the node-span parse over unbounded stored entry text are
# LINE-oriented, so both gate on should_offload_line_scan (app.services.text_lines)
# rather than on a size-only threshold, like the outline parse of navigate_context. The
# grep matcher keeps its own size-based gate (grep_service._OFFLOAD_MIN_CHARS) because
# its literal path is a single C-level whole-text scan and its regex path offloads
# unconditionally, so there size really is the cost driver.


async def read_context_range(
    context_id: Annotated[
        str,
        Field(min_length=8, description='Context entry id (32/36-char UUID or 8-31 char hex prefix)'),
    ],
    start_char: Annotated[
        int | None,
        Field(ge=0, description='Start of a character range (Unicode code-point offset, inclusive)'),
    ] = None,
    end_char: Annotated[
        int | None,
        Field(ge=0, description='End of a character range (code-point offset, exclusive)'),
    ] = None,
    start_line: Annotated[
        int | None,
        Field(ge=1, description='Start of a line range (1-based, inclusive)'),
    ] = None,
    end_line: Annotated[
        int | None,
        Field(ge=1, description='End of a line range (1-based, inclusive)'),
    ] = None,
    node_id: Annotated[
        str | None,
        Field(description="Outline node id from navigate_context (e.g. 'setup/install' or 'root')"),
    ] = None,
) -> ReadContextRangeDict:
    """Read part of one context entry by character range, line range, or outline node.

    Slices the full stored ``text_content`` directly, so it works for EVERY
    entry regardless of embeddings. Provide exactly ONE addressing mode: a
    character range (``start_char``/``end_char``), a line range
    (``start_line``/``end_line``), or a ``node_id`` from ``navigate_context``.
    Pair it with ``grep_context`` (content mode) by feeding a match's
    ``match_start``/``match_end`` into ``start_char``/``end_char``, or with
    ``navigate_context`` by passing a section's ``node_id``.

    Out-of-range offsets are CLAMPED to ``[0, len(text)]`` and the response
    ECHOES the resolved span, so an offset or node_id captured in a prior turn
    degrades gracefully if the entry was edited since.

    Returns:
        ReadContextRangeDict with ``context_id``, the resolved
        ``start_char``/``end_char`` and ``start_line``/``end_line`` (1-based), the
        ``text`` of the slice, and the entry's ``visibility``, ``is_owner``
        (whether the caller owns it) and ``generation_status`` ('ready' once the
        entry's embeddings and summaries are generated, 'pending' while their
        generation waits or runs, 'failed' when it gave up after its retries).
        Offsets are Unicode code-point indices.

    Raises:
        ToolError: If the id is invalid, not exactly one addressing mode is
            given, the entry is not found, or the node_id does not resolve.
    """
    try:
        repos = await ensure_repositories()
        # The read below runs as the caller, so an entry it may not read is not found.
        scope = resolve_access_scope()
        try:
            resolved_id = await resolve_or_normalize_id(context_id, repos.context, scope=scope)
        except ValueError as exc:
            raise ToolError(f'Invalid context ID: {exc}') from exc

        char_mode = start_char is not None or end_char is not None
        line_mode = start_line is not None or end_line is not None
        node_mode = node_id is not None
        if sum((char_mode, line_mode, node_mode)) != 1:
            raise ToolError(
                'Provide exactly one addressing mode: a character range (start_char/end_char), '
                'a line range (start_line/end_line), or a node_id.',
            )

        rows = await repos.context.get_by_ids([resolved_id], scope=scope)
        if not rows:
            raise ToolError(f'Context entry not found: {context_id}')
        text_value = rows[0]['text_content']
        text = text_value if text_value is not None else ''
        access = entry_access_fields(rows[0])
        statuses = await generation_statuses([resolved_id], repos, scope=scope)
        length = len(text)
        # Line splitting costs one slice and one list append PER LINE, so like the
        # outline parse its cost tracks LINE count rather than text size: a
        # line-dense entry well under the size threshold still blocks the loop for
        # a tenth of a second on EVERY read_context_range call, in every addressing
        # mode. Gate it on the same line-aware predicate (see
        # should_offload_line_scan).
        if should_offload_line_scan(text):
            lines, line_starts = await asyncio.to_thread(split_lines_with_offsets, text)
        else:
            lines, line_starts = split_lines_with_offsets(text)

        # In line mode the clamped line numbers ARE the resolved span; echo them
        # verbatim. Recomputing from the exclusive end offset would under-report by
        # one when the requested range ends on the trailing empty line (its content
        # start equals the document length, whose preceding code point belongs to
        # the prior line).
        resolved_lines: tuple[int, int] | None = None
        if char_mode:
            start = start_char if start_char is not None else 0
            end = end_char if end_char is not None else length
        elif line_mode:
            line_count = len(lines)
            first_line = start_line if start_line is not None else 1
            last_line = end_line if end_line is not None else line_count
            first_line = max(1, min(first_line, line_count))
            last_line = max(1, min(last_line, line_count))
            if last_line < first_line:
                last_line = first_line
            start = line_starts[first_line - 1]
            end = line_starts[last_line - 1] + len(lines[last_line - 1])
            resolved_lines = (first_line, last_line)
        else:
            # node_id mode: resolve against the on-demand outline (node_id is not
            # None here because exactly one mode is active). resolve_node_span
            # re-parses the outline, whose cost tracks LINE count; offload a large
            # OR line-dense entry so the parse cannot pin the event loop -- even a
            # request returning a five-character slice pays the full parse (see
            # should_offload_line_scan).
            if should_offload_line_scan(text):
                span = await asyncio.to_thread(resolve_node_span, text, node_id or '')
            else:
                span = resolve_node_span(text, node_id or '')
            if span is None:
                raise ToolError(
                    f'node_id {node_id!r} did not resolve to a section in this entry '
                    f'(it may have changed since navigate_context was called).',
                )
            start, end = span

        # Mandatory clamp so stale grep/prior-turn offsets degrade gracefully.
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end < start:
            end = start

        snippet = text[start:end]
        if resolved_lines is not None:
            start_line_out, end_line_out = resolved_lines
        else:
            start_line_out = line_index_for_offset(line_starts, start) + 1
            end_anchor = end - 1 if end > start else start
            end_line_out = line_index_for_offset(line_starts, end_anchor) + 1

        return {
            'context_id': resolved_id,
            'start_char': start,
            'end_char': end,
            'start_line': start_line_out,
            'end_line': end_line_out,
            'text': snippet,
            'visibility': access['visibility'],
            'is_owner': access['is_owner'],
            'generation_status': statuses[resolved_id],
        }
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error in read_context_range: {exc}')
        raise ToolError(f'Failed to read context range: {format_exception_message(exc)}') from exc
