"""The ``grep_context`` tool: server-side grep over stored context.

Grep lets an agent work with stored context the way it works with a codebase:
locate exact text, then read only the relevant span. It is literal/regex,
line-oriented, UNRANKED pattern matching over ``text_content``, the complement of
the full-text and semantic search tools, not a replacement: FTS matches
stemmed/ranked lexemes and returns whole entries, while grep matches raw
characters/regex and returns match LOCATIONS (line number + code-point offsets).
All matching runs in Python (``re`` for literals, the ``regex`` engine for user
regex) for byte-identical SQLite/PostgreSQL results.

``content`` rows carry ``match_start``/``match_end`` in the one Unicode code-point
offset contract the navigation tools share: they feed straight into
``read_context_range`` ``start_char``/``end_char`` -- locate, then extract.

Every grep row reports its entry's ``visibility``, whether the caller owns it
(``is_owner``) and the state of its generated outputs (``generation_status``).
"""

import asyncio
import logging
import re
import time
from collections.abc import Mapping
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

import regex
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.services.grep_service import GrepEntryResult
from app.services.grep_service import compile_pattern
from app.services.grep_service import extract_ascii_literal
from app.services.grep_service import match_entry
from app.settings import get_settings
from app.startup import ensure_repositories
from app.tools._responses import entry_access_fields
from app.tools._responses import generation_statuses
from app.tools._validation import reject_unstorable_input
from app.tools.search.limits import MAX_FILTER_TAGS
from app.tools.search.limits import MAX_METADATA_FILTERS
from app.tools.search.limits import filter_caps_error
from app.types import EntryAccessFields
from app.types import GenerationStatus
from app.types import GrepContentMatchDict
from app.types import GrepContextResultDict
from app.types import GrepCountDict
from app.types import GrepFileDict

logger = logging.getLogger(__name__)
settings = get_settings()


def _shape_grep_output(
    output_mode: str,
    entry_results: list[GrepEntryResult],
    access: Mapping[str, EntryAccessFields],
    statuses: Mapping[str, GenerationStatus],
    total_matches: int,
    truncated: bool,
    timed_out_ids: list[str] | None = None,
) -> GrepContextResultDict:
    """Build the grep response dict for the requested output mode.

    Returns:
        The response, every row of it carrying the access fields of its entry, taken
        from ``access`` by entry id, and its generation state, taken from ``statuses``.
    """
    if output_mode == 'content':
        content_rows: list[GrepContentMatchDict] = [
            {
                'context_id': entry.context_id,
                'line_number': match.line_number,
                'line': match.line,
                'match_start': match.char_start,
                'match_end': match.char_end,
                'before': list(match.before),
                'after': list(match.after),
                'visibility': access[entry.context_id]['visibility'],
                'is_owner': access[entry.context_id]['is_owner'],
                'generation_status': statuses[entry.context_id],
            }
            for entry in entry_results
            for match in entry.matches
        ]
        results: list[Any] = content_rows
    elif output_mode == 'count':
        count_rows: list[GrepCountDict] = [
            {
                'context_id': entry.context_id,
                'count': entry.match_count,
                'visibility': access[entry.context_id]['visibility'],
                'is_owner': access[entry.context_id]['is_owner'],
                'generation_status': statuses[entry.context_id],
            }
            for entry in entry_results
        ]
        results = count_rows
    else:  # files_with_matches (default)
        file_rows: list[GrepFileDict] = [
            {
                'context_id': entry.context_id,
                'match_count': entry.match_count,
                'visibility': access[entry.context_id]['visibility'],
                'is_owner': access[entry.context_id]['is_owner'],
                'generation_status': statuses[entry.context_id],
            }
            for entry in entry_results
        ]
        results = file_rows

    response = cast(
        GrepContextResultDict,
        {
            'mode': output_mode,
            'total_matches': total_matches,
            'truncated': truncated,
            'results': results,
        },
    )
    # Surface entries skipped because their regex match timed out, so the caller
    # knows the result is incomplete for them (present only when some entry did).
    if timed_out_ids:
        response['timed_out_context_ids'] = timed_out_ids
    return response


def _trim_entry_results(
    entry_results: list[GrepEntryResult],
    limit: int,
) -> tuple[list[GrepEntryResult], int]:
    """Trim collected matches to exactly ``limit`` total, rebuilding the boundary entry.

    The grep loop collects one match past ``max_matches`` to detect genuine
    truncation truthfully; this drops the surplus and marks the entry straddling
    the cap as ``capped`` so the response stays internally consistent.

    Returns:
        The trimmed entry list and the new total match count (== ``limit``).
    """
    trimmed: list[GrepEntryResult] = []
    total = 0
    for entry in entry_results:
        if total >= limit:
            break
        room = limit - total
        if entry.match_count <= room:
            trimmed.append(entry)
            total += entry.match_count
            continue
        kept = entry.matches[:room]
        trimmed.append(
            GrepEntryResult(
                context_id=entry.context_id,
                matches=kept,
                match_count=len(kept),
                capped=True,
                timed_out=entry.timed_out,
            ),
        )
        total += len(kept)
        break
    return trimmed, total


async def grep_context(
    pattern: Annotated[
        str,
        Field(
            min_length=1,
            max_length=settings.grep_context.max_pattern_chars,
            description='Literal substring (default) or regular expression to match, raw and unstemmed '
                        f'(at most {settings.grep_context.max_pattern_chars} characters)',
        ),
    ],
    thread_id: Annotated[
        str | None,
        Field(description='Scope to one thread (recommended: bounds the scan to the indexed thread)'),
    ] = None,
    source: Annotated[
        Literal['user', 'agent'] | None,
        Field(description="Filter by entry source ('user' or 'agent')"),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(max_length=MAX_FILTER_TAGS, description=f'Filter by tags (OR logic; at most {MAX_FILTER_TAGS})'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            max_length=MAX_METADATA_FILTERS,
            description='Advanced metadata filters with operators (gt, lt, contains, exists, etc.); '
                        f'at most {MAX_METADATA_FILTERS} filters',
        ),
    ] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None,
        Field(description='Filter by content type'),
    ] = None,
    is_regex: Annotated[
        bool,
        Field(description='Treat pattern as a Python regular expression. Default False: literal substring (auto-escaped)'),
    ] = False,
    case_sensitive: Annotated[
        bool,
        Field(description='Match case-sensitively. Default False (Unicode-aware case-insensitive).'),
    ] = False,
    output_mode: Annotated[
        Literal['files_with_matches', 'content', 'count'],
        Field(
            description="'files_with_matches' (default: context_ids + match_count), "
                        "'content' (each match with line + offsets + context), 'count' (per-entry tally)",
        ),
    ] = 'files_with_matches',
    context_lines: Annotated[
        int,
        Field(ge=0, le=100, description='Surrounding lines before/after each match in content mode (like grep -C)'),
    ] = 0,
    max_matches: Annotated[
        int,
        Field(ge=1, le=10000, description='Maximum total matches to return (bounded output; clamped to the server cap)'),
    ] = 100,
    max_entries_scanned: Annotated[
        int,
        Field(ge=1, le=1000000, description='Maximum entries the scan visits (clamped to the server cap)'),
    ] = 1000,
) -> GrepContextResultDict:
    """Server-side grep over stored context: literal/regex, line-oriented, unranked.

    Use this to find EXACT text -- identifiers, IDs, dates, code, error strings,
    punctuation -- that linguistic search cannot pin down. Unlike
    ``fts_search_context`` (stemmed, ranked, whole-entry) and
    ``semantic_search_context`` (meaning-based), grep matches raw characters and
    returns precise LOCATIONS so you can then read just that span with
    ``read_context_range``.

    Matching runs in Python (identical on SQLite and PostgreSQL): literals use the
    stdlib ``re`` engine; ``is_regex=True`` uses the ``regex`` engine with ``^``/``$``
    anchored per line (line-oriented: matching runs per logical line split on LF/CRLF,
    so a pattern never spans a line break and ``^``/``$`` anchor at CRLF as well as LF
    boundaries, like ripgrep with ``--crlf``), bounded by both a real per-entry
    wall-clock timeout AND an aggregate scan budget: a catastrophic pattern is
    preempted and that entry is skipped, and once the aggregate budget is exhausted
    the scan stops with ``truncated`` True -- rather than freezing the server. Output
    is always bounded by ``max_matches``; ``truncated`` is True only when matches or
    the scan were genuinely capped. Scope with ``thread_id`` whenever possible -- an
    unscoped pattern scans entries sequentially. The ``pattern`` accepts at most
    GREP_MAX_PATTERN_CHARS characters (default 32768), the ``tags`` filter at most
    100 tags, and ``metadata_filters`` at most 100 filters per request; an oversized
    input is rejected at the tool boundary as a validation error before any scan,
    while an invalid metadata filter returns a structured validation-error response
    with ``error`` and ``validation_errors``.

    Returns:
        GrepContextResultDict with ``mode``, ``total_matches``, ``truncated`` and
        ``results``. In ``content`` mode each result carries ``context_id``,
        ``line_number``, ``line``, ``match_start``/``match_end`` (Unicode
        code-point offsets into text_content), and ``before``/``after`` context
        lines. In ``files_with_matches`` mode each result is ``context_id`` +
        ``match_count``; in ``count`` mode ``context_id`` + ``count``. Every result also
        carries its entry's ``visibility``, ``is_owner`` (whether the caller owns
        the entry) and ``generation_status`` ('ready' once the entry's embeddings
        and summaries are generated, 'pending' while their generation waits or
        runs, 'failed' when it gave up after its retries).
        ``timed_out_context_ids`` lists entries skipped because their regex match
        exceeded the timeout (present only when some entry timed out -- the result
        is then incomplete for those entries).

    Raises:
        ToolError: If the regular expression is invalid or the scan fails.
    """
    try:
        repos = await ensure_repositories()
        # The scan runs as the caller: entries it may not read are never scanned and
        # never count toward the scan cap or the truncated flag.
        scope = resolve_access_scope()

        # Reject an embedded NUL or unpaired UTF-16 surrogate in thread_id or a tag before
        # it reaches the PostgreSQL bind, where asyncpg would raise a non-ControlFlowError
        # that charges the circuit breaker (SQLite binds it silently -- a divergence). Both
        # feed grep_scan_text_contents below (thread_id and tags both parameterize the scan
        # query), so both must be screened, matching the sibling search tools.
        # A NUL in the grep pattern is handled separately by extract_ascii_literal, which
        # returns None for it so the pattern never reaches the LIKE/ILIKE pre-filter bind.
        reject_unstorable_input(thread_id=thread_id, tags=tags)

        grep_settings = settings.grep_context

        # Boundary cap re-checks behind the wire-schema max_length constraints,
        # returned as structured validation errors before any SQL executes or any
        # pattern compiles (never charging the circuit breaker), matching the
        # sibling search tools: each tag and each metadata filter binds SQL
        # placeholders in the scan query, and pattern COMPILATION cost grows with
        # pattern size (it runs before any matching timeout applies, so an
        # unbounded pattern could stall the whole request just to compile).
        caps_error = filter_caps_error(tags, metadata_filters=metadata_filters)
        if caps_error is None and len(pattern) > grep_settings.max_pattern_chars:
            caps_error = (
                f'Pattern too long: {len(pattern)} characters exceeds the maximum of '
                f'{grep_settings.max_pattern_chars}. Shorten the pattern.'
            )
        if caps_error is not None:
            return {
                'mode': output_mode,
                'total_matches': 0,
                'truncated': False,
                'results': [],
                'error': caps_error,
                'validation_errors': [caps_error],
            }

        max_matches = min(max_matches, grep_settings.max_matches_cap)
        context_lines = min(context_lines, grep_settings.max_context_lines)
        max_entries_scanned = min(max_entries_scanned, grep_settings.max_entries_scanned)

        try:
            if is_regex:
                # User-regex compilation is pure-Python parsing whose cost grows
                # with pattern size and structure; run it in a worker thread so
                # even a slow compile of a legal-size pattern cannot pin the
                # event loop. Literal compilation (re.escape + re.compile) is
                # linear and sub-millisecond at the enforced size cap, so it
                # stays inline to avoid a per-call thread hop.
                compiled = await asyncio.to_thread(
                    compile_pattern, pattern, is_regex=True, case_sensitive=case_sensitive,
                )
            else:
                compiled = compile_pattern(pattern, is_regex=False, case_sensitive=case_sensitive)
        except (re.error, regex.error) as exc:
            raise ToolError(f'Invalid regular expression: {exc}') from exc

        ascii_literal = extract_ascii_literal(pattern, is_regex=is_regex, case_sensitive=case_sensitive)

        rows, scan_stats = await repos.context.grep_scan_text_contents(
            ascii_literal=ascii_literal,
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            metadata_filters=metadata_filters,
            max_entries_scanned=max_entries_scanned,
            aggregate_bytes_budget=grep_settings.aggregate_bytes_budget,
            scope=scope,
        )

        scan_validation_errors = scan_stats.get('validation_errors')
        if scan_validation_errors:
            return {
                'mode': output_mode,
                'total_matches': 0,
                'truncated': False,
                'results': [],
                'error': 'Metadata filter validation failed',
                'validation_errors': cast(list[str], scan_validation_errors),
            }

        entry_results: list[GrepEntryResult] = []
        collected = 0
        match_truncated = False
        timed_out_ids: list[str] = []
        # Collect up to max_matches + 1 so a genuine extra match is detectable
        # (truthful ``truncated``): a trailing candidate that yields no match must
        # NOT flip the flag. The +1 overflow is trimmed back before shaping.
        overflow_cap = max_matches + 1
        # Aggregate wall-clock budget for a regex scan: the per-entry
        # regex_timeout_s bounds ONE entry, but an unscoped regex can visit up to
        # max_entries_scanned rows, so without a cumulative cap a catastrophic
        # pattern could hold one request for max_entries_scanned * regex_timeout_s.
        # The deadline applies only to is_regex (literals are linear-time); when it
        # fires the scan stops and returns the matches collected so far, flagged
        # truncated -- never aborting the read-only tool.
        regex_deadline = (
            time.monotonic() + grep_settings.regex_total_timeout_s if is_regex else None
        )
        scan_deadline_exceeded = False
        access = {
            row.id: entry_access_fields({'visibility': row.visibility, 'is_owner': row.is_owner}) for row in rows
        }
        for row in rows:
            context_id, text = row.id, row.text_content
            if collected >= overflow_cap:
                break
            if regex_deadline is not None and time.monotonic() >= regex_deadline:
                scan_deadline_exceeded = True
                break
            result = await match_entry(
                context_id,
                text,
                compiled,
                context_lines=context_lines,
                max_matches=overflow_cap - collected,
                is_regex=is_regex,
                regex_timeout_s=grep_settings.regex_timeout_s,
            )
            if result.timed_out:
                timed_out_ids.append(context_id)
            if result.capped:
                match_truncated = True
            if result.match_count == 0:
                continue
            entry_results.append(result)
            collected += result.match_count

        if collected > max_matches:
            # A real match beyond the cap exists -> genuine truncation; trim the
            # collected matches back to exactly max_matches before shaping.
            match_truncated = True
            entry_results, collected = _trim_entry_results(entry_results, max_matches)

        truncated = bool(scan_stats.get('truncated')) or match_truncated or scan_deadline_exceeded
        # One generation state read for the entries the response carries, never for the scanned rows without a match.
        statuses = await generation_statuses((entry.context_id for entry in entry_results), repos, scope=scope)
        return _shape_grep_output(output_mode, entry_results, access, statuses, collected, truncated, timed_out_ids)
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error in grep_context: {exc}')
        raise ToolError(f'Failed to grep context: {format_exception_message(exc)}') from exc
