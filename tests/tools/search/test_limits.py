"""Tests for the search argument bounds in ``app.tools.search.limits``: the offset ceiling and the tags and
metadata filter caps on every search tool and ``grep_context``.
"""

from collections.abc import Callable
from typing import Any
from typing import cast
from typing import get_args
from typing import get_type_hints
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from annotated_types import Ge
from annotated_types import Le
from annotated_types import MaxLen
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from pydantic.fields import FieldInfo

import app.tools
from tests.helpers import argument_errors
from tests.tools._generation_jobs import NoGenerationJobs

_SEARCH_TOOLS: list[Callable[..., object]] = [
    app.tools.search_context,
    app.tools.semantic_search_context,
    app.tools.fts_search_context,
    app.tools.hybrid_search_context,
]


class TestSearchOffsetUpperBound:
    """The offset parameter carries an explicit upper bound on every search tool.

    Bounding offset at the tool boundary keeps a deep-pagination request a clean
    client-facing validation error instead of letting the reranking/RRF overfetch
    multipliers grow a LIMIT/OFFSET bind past the int64 column width inside
    connection scope (which would charge the circuit breaker for invalid input).
    """

    @staticmethod
    def _offset_field_info(fn: Callable[..., object]) -> FieldInfo:
        """Extract the pydantic FieldInfo attached to a tool's ``offset`` parameter."""
        hints = get_type_hints(fn, include_extras=True)
        annotation = hints['offset']
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, FieldInfo):
                return meta
        msg = f'offset annotation for {getattr(fn, "__name__", fn)!r} carries no FieldInfo'
        raise AssertionError(msg)

    @pytest.mark.parametrize('tool', _SEARCH_TOOLS)
    def test_offset_declares_le_max_search_offset(self, tool: Callable[..., object]) -> None:
        """Every search tool's offset Field declares le=MAX_SEARCH_OFFSET."""
        from app.tools.search.limits import MAX_SEARCH_OFFSET

        tool_name = getattr(tool, '__name__', tool)
        field_info = self._offset_field_info(tool)
        le_values = [constraint.le for constraint in field_info.metadata if isinstance(constraint, Le)]
        assert le_values == [MAX_SEARCH_OFFSET], (
            f'{tool_name!r} offset must declare le=MAX_SEARCH_OFFSET, got {le_values}'
        )

    @pytest.mark.parametrize('tool', _SEARCH_TOOLS)
    def test_offset_declares_ge_zero(self, tool: Callable[..., object]) -> None:
        """The offset lower bound (ge=0) is declared alongside the upper bound."""
        tool_name = getattr(tool, '__name__', tool)
        field_info = self._offset_field_info(tool)
        ge_values = [constraint.ge for constraint in field_info.metadata if isinstance(constraint, Ge)]
        assert ge_values == [0], f'{tool_name!r} offset must declare ge=0, got {ge_values}'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_offset_above_bound_rejected(self) -> None:
        """An offset above MAX_SEARCH_OFFSET is rejected at the boundary.

        Calling the tool through the pydantic-validated wrapper rejects the
        out-of-range offset before any repository work runs; the le constraint
        surfaces as a FastMCP ValidationError, chained from the pydantic error,
        naming the offset field and its ceiling.
        """
        from fastmcp.tools import Tool

        from app.tools.search.limits import MAX_SEARCH_OFFSET

        validated = Tool.from_function(app.tools.search_context)
        with pytest.raises(FastMCPValidationError) as exc_info:
            await validated.run({'offset': MAX_SEARCH_OFFSET + 1})
        errors = argument_errors(exc_info)
        assert any(
            err['type'] == 'less_than_equal' and err['loc'] == ('offset',)
            for err in errors
        ), errors

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_offset_at_bound_accepted(self) -> None:
        """An offset exactly at MAX_SEARCH_OFFSET passes boundary validation.

        The ceiling is inclusive (le), so the maximum legitimate value is
        accepted and reaches the tool body (returning the normal response shape).
        """
        from fastmcp.tools import Tool

        from app.tools.search.limits import MAX_SEARCH_OFFSET

        validated = Tool.from_function(app.tools.search_context)
        result = await validated.run({'thread_id': 'offset_bound_thread', 'offset': MAX_SEARCH_OFFSET})
        payload = result.structured_content
        assert payload is not None
        assert payload['results'] == []
        assert payload['count'] == 0


_TAGS_FILTER_TOOLS: list[Callable[..., object]] = [
    *_SEARCH_TOOLS,
    app.tools.grep_context,
]


class TestTagsFilterUpperBound:
    """The tags filter carries an explicit member cap on every search and grep tool.

    Each tag expands into one SQL bind placeholder in a single statement on both
    backends, so an unbounded schema-legal tags array could overflow the backend's
    bind limit inside connection scope and charge the circuit breaker for purely
    invalid client input. The cap is advertised in the wire schema
    (Field(max_length=MAX_FILTER_TAGS)) and re-checked in the tool body as a
    structured validation error before any SQL executes.
    """

    @staticmethod
    def _tags_field_info(fn: Callable[..., object]) -> FieldInfo:
        """Extract the pydantic FieldInfo attached to a tool's ``tags`` parameter."""
        hints = get_type_hints(fn, include_extras=True)
        annotation = hints['tags']
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, FieldInfo):
                return meta
        msg = f'tags annotation for {getattr(fn, "__name__", fn)!r} carries no FieldInfo'
        raise AssertionError(msg)

    @pytest.mark.parametrize('tool', _TAGS_FILTER_TOOLS)
    def test_tags_declares_max_length_cap(self, tool: Callable[..., object]) -> None:
        """Every tags-accepting tool's Field declares max_length=MAX_FILTER_TAGS."""
        from app.tools.search.limits import MAX_FILTER_TAGS

        tool_name = getattr(tool, '__name__', tool)
        field_info = self._tags_field_info(tool)
        max_values = [constraint.max_length for constraint in field_info.metadata if isinstance(constraint, MaxLen)]
        assert max_values == [MAX_FILTER_TAGS], (
            f'{tool_name!r} tags must declare max_length=MAX_FILTER_TAGS, got {max_values}'
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_tags_over_cap_rejected_at_boundary(self) -> None:
        """A tags list above the cap is rejected by the wire-schema validation with a FastMCP ValidationError."""
        from fastmcp.tools import Tool

        from app.tools.search.limits import MAX_FILTER_TAGS

        validated = Tool.from_function(app.tools.search_context)
        oversized = [f'tag-{i}' for i in range(MAX_FILTER_TAGS + 1)]
        with pytest.raises(FastMCPValidationError) as exc_info:
            await validated.run({'tags': oversized})
        errors = argument_errors(exc_info)
        assert any(err['type'] == 'too_long' for err in errors), errors

    @pytest.mark.asyncio
    async def test_search_context_oversized_tags_structured_error_before_sql(self) -> None:
        """A direct call with oversized tags returns a structured validation error
        without touching the repository layer (nothing reaches SQL)."""
        from app.tools.search.limits import MAX_FILTER_TAGS

        oversized = [f'tag-{i}' for i in range(MAX_FILTER_TAGS + 1)]
        ensure_repos = AsyncMock()
        with patch('app.tools.search.browse.ensure_repositories', ensure_repos):
            result = await app.tools.search_context(tags=oversized)

        assert result['results'] == []
        assert result['count'] == 0
        assert 'exceeds the maximum' in result['error']
        assert result['validation_errors'] == [result['error']]
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_context_tags_at_cap_accepted(self) -> None:
        """A tags list of exactly MAX_FILTER_TAGS members passes the cap (inclusive)."""
        from app.tools.search.limits import MAX_FILTER_TAGS

        at_cap = [f'tag-{i}' for i in range(MAX_FILTER_TAGS)]
        repos = MagicMock()
        repos.context.search_contexts = AsyncMock(return_value=([], {}))
        repos.generation_jobs = NoGenerationJobs()
        with patch('app.tools.search.browse.ensure_repositories', AsyncMock(return_value=repos)):
            result = await app.tools.search_context(tags=at_cap)

        assert result['results'] == []
        assert result['count'] == 0
        assert 'error' not in result

    @pytest.mark.asyncio
    async def test_grep_context_oversized_tags_structured_error_before_sql(self) -> None:
        """grep_context rejects oversized tags as a structured validation error
        before the scan query runs (the repository scan is never invoked)."""
        from app.tools.search.limits import MAX_FILTER_TAGS

        oversized = [f'tag-{i}' for i in range(MAX_FILTER_TAGS + 1)]
        repos = MagicMock()
        repos.context.grep_scan_text_contents = AsyncMock()
        with patch('app.tools.navigation.grep.ensure_repositories', AsyncMock(return_value=repos)):
            result = await app.tools.grep_context(pattern='needle', tags=oversized)

        # GrepContextResultDict is total=False; widen for direct key assertions.
        payload = cast('dict[str, Any]', result)
        assert payload['mode'] == 'files_with_matches'
        assert payload['results'] == []
        assert payload['total_matches'] == 0
        assert payload['truncated'] is False
        assert 'exceeds the maximum' in payload['error']
        assert payload['validation_errors'] == [payload['error']]
        repos.context.grep_scan_text_contents.assert_not_awaited()


class TestMetadataFilterCapsUpperBound:
    """metadata_filters and the simple metadata dict carry explicit caps on every filter tool.

    The sibling caps bound the per-item dimensions (tags list members, IN/NOT_IN
    value-list members), but capped dimensions multiply: without a cap on the
    filter LIST length (and the simple metadata dict key count) a few hundred
    schema-legal IN filters expand past the backend's per-statement bind limit
    inside connection scope and charge the circuit breaker. The caps are
    advertised in the wire schema (max_length on the Field) and re-checked in
    each tool body as a structured validation error before any SQL executes.
    """

    @staticmethod
    def _field_info(fn: Callable[..., object], param: str) -> FieldInfo:
        """Extract the pydantic FieldInfo attached to a tool's named parameter."""
        hints = get_type_hints(fn, include_extras=True)
        annotation = hints[param]
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, FieldInfo):
                return meta
        msg = f'{param} annotation for {getattr(fn, "__name__", fn)!r} carries no FieldInfo'
        raise AssertionError(msg)

    @pytest.mark.parametrize('tool', _TAGS_FILTER_TOOLS)
    def test_metadata_filters_declares_max_length_cap(self, tool: Callable[..., object]) -> None:
        """Every filter tool's metadata_filters Field declares max_length=MAX_METADATA_FILTERS."""
        from app.tools.search.limits import MAX_METADATA_FILTERS

        tool_name = getattr(tool, '__name__', tool)
        field_info = self._field_info(tool, 'metadata_filters')
        max_values = [constraint.max_length for constraint in field_info.metadata if isinstance(constraint, MaxLen)]
        assert max_values == [MAX_METADATA_FILTERS], (
            f'{tool_name!r} metadata_filters must declare max_length=MAX_METADATA_FILTERS, got {max_values}'
        )

    @pytest.mark.parametrize('tool', _SEARCH_TOOLS)
    def test_metadata_dict_declares_max_length_cap(self, tool: Callable[..., object]) -> None:
        """Every search tool's simple metadata Field declares max_length=MAX_METADATA_KEYS."""
        from app.tools.search.limits import MAX_METADATA_KEYS

        tool_name = getattr(tool, '__name__', tool)
        field_info = self._field_info(tool, 'metadata')
        max_values = [constraint.max_length for constraint in field_info.metadata if isinstance(constraint, MaxLen)]
        assert max_values == [MAX_METADATA_KEYS], (
            f'{tool_name!r} metadata must declare max_length=MAX_METADATA_KEYS, got {max_values}'
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_metadata_filters_over_cap_rejected_at_boundary(self) -> None:
        """A metadata_filters list above the cap is rejected by the wire-schema validation with a FastMCP ValidationError."""
        from fastmcp.tools import Tool

        from app.tools.search.limits import MAX_METADATA_FILTERS

        validated = Tool.from_function(app.tools.search_context)
        oversized = [{'key': 'status', 'operator': 'eq', 'value': 'x'}] * (MAX_METADATA_FILTERS + 1)
        with pytest.raises(FastMCPValidationError) as exc_info:
            await validated.run({'metadata_filters': oversized})
        errors = argument_errors(exc_info)
        assert any(err['type'] == 'too_long' for err in errors), errors

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_metadata_dict_over_cap_rejected_at_boundary(self) -> None:
        """A metadata dict above the key cap is rejected by the wire-schema validation with a FastMCP ValidationError."""
        from fastmcp.tools import Tool

        from app.tools.search.limits import MAX_METADATA_KEYS

        validated = Tool.from_function(app.tools.search_context)
        oversized = {f'key{i}': i for i in range(MAX_METADATA_KEYS + 1)}
        with pytest.raises(FastMCPValidationError) as exc_info:
            await validated.run({'metadata': oversized})
        errors = argument_errors(exc_info)
        assert any(err['type'] == 'too_long' for err in errors), errors

    @pytest.mark.asyncio
    async def test_search_context_oversized_metadata_filters_structured_error_before_sql(self) -> None:
        """A direct call with an oversized metadata_filters list returns a structured
        validation error without touching the repository layer (nothing reaches SQL)."""
        from app.tools.search.limits import MAX_METADATA_FILTERS

        oversized = [{'key': 'status', 'operator': 'eq', 'value': 'x'}] * (MAX_METADATA_FILTERS + 1)
        ensure_repos = AsyncMock()
        with patch('app.tools.search.browse.ensure_repositories', ensure_repos):
            result = await app.tools.search_context(metadata_filters=oversized)

        assert result['results'] == []
        assert result['count'] == 0
        assert 'exceeds the maximum' in result['error']
        assert result['validation_errors'] == [result['error']]
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_context_oversized_metadata_dict_structured_error_before_sql(self) -> None:
        """A direct call with an oversized simple metadata dict returns a structured
        validation error without touching the repository layer."""
        from app.tools.search.limits import MAX_METADATA_KEYS

        oversized: dict[str, str | int | float | bool] = {f'key{i}': i for i in range(MAX_METADATA_KEYS + 1)}
        ensure_repos = AsyncMock()
        with patch('app.tools.search.browse.ensure_repositories', ensure_repos):
            result = await app.tools.search_context(metadata=oversized)

        assert result['results'] == []
        assert result['count'] == 0
        assert 'exceeds the maximum' in result['error']
        assert result['validation_errors'] == [result['error']]
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_context_metadata_filters_at_cap_accepted(self) -> None:
        """A metadata_filters list of exactly MAX_METADATA_FILTERS members passes the cap."""
        from app.tools.search.limits import MAX_METADATA_FILTERS

        at_cap = [{'key': 'status', 'operator': 'eq', 'value': 'x'}] * MAX_METADATA_FILTERS
        repos = MagicMock()
        repos.context.search_contexts = AsyncMock(return_value=([], {}))
        repos.generation_jobs = NoGenerationJobs()
        with patch('app.tools.search.browse.ensure_repositories', AsyncMock(return_value=repos)):
            result = await app.tools.search_context(metadata_filters=at_cap)

        assert result['results'] == []
        assert result['count'] == 0
        assert 'error' not in result

    @pytest.mark.asyncio
    async def test_grep_context_oversized_metadata_filters_structured_error_before_sql(self) -> None:
        """grep_context rejects an oversized metadata_filters list as a structured
        validation error before the scan query runs."""
        from app.tools.search.limits import MAX_METADATA_FILTERS

        oversized = [{'key': 'status', 'operator': 'eq', 'value': 'x'}] * (MAX_METADATA_FILTERS + 1)
        repos = MagicMock()
        repos.context.grep_scan_text_contents = AsyncMock()
        with patch('app.tools.navigation.grep.ensure_repositories', AsyncMock(return_value=repos)):
            result = await app.tools.grep_context(pattern='needle', metadata_filters=oversized)

        # GrepContextResultDict is total=False; widen for direct key assertions.
        payload = cast('dict[str, Any]', result)
        assert payload['results'] == []
        assert payload['total_matches'] == 0
        assert 'exceeds the maximum' in payload['error']
        assert payload['validation_errors'] == [payload['error']]
        repos.context.grep_scan_text_contents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_context_caps_error_attaches_stats_under_explain_query(self) -> None:
        """The caps validation-error response carries the zeroed stats dict when
        explain_query=True, matching the fts/semantic siblings' error-path shape."""
        import app.tools.search.limits as search_limits
        from app.tools.search.limits import MAX_FILTER_TAGS

        oversized = [f'tag-{i}' for i in range(MAX_FILTER_TAGS + 1)]
        with patch('app.tools.search.browse.ensure_repositories', AsyncMock()):
            result = await app.tools.search_context(tags=oversized, explain_query=True)

        assert 'exceeds the maximum' in result['error']
        # query_plan is carried as an explicit null rather than omitted: the documented
        # contract lets a client read stats['query_plan'] unconditionally under
        # explain_query, so dropping the key here would make that access raise KeyError.
        assert result['stats'] == {
            'execution_time_ms': 0.0,
            'filters_applied': 0,
            'rows_returned': 0,
            'backend': search_limits.settings.storage.backend_type,
            'query_plan': None,
        }

    @pytest.mark.asyncio
    async def test_search_context_caps_error_omits_stats_without_explain_query(self) -> None:
        """Without explain_query the caps validation-error response carries no stats."""
        from app.tools.search.limits import MAX_FILTER_TAGS

        oversized = [f'tag-{i}' for i in range(MAX_FILTER_TAGS + 1)]
        with patch('app.tools.search.browse.ensure_repositories', AsyncMock()):
            result = await app.tools.search_context(tags=oversized, explain_query=False)

        assert 'exceeds the maximum' in result['error']
        assert 'stats' not in result
