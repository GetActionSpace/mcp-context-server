"""Tests for the ``grep_context`` pattern-size cap and off-loop regex compilation in ``app.tools.navigation.grep``."""

from collections.abc import Callable
from typing import Any
from typing import cast
from typing import get_args
from typing import get_type_hints
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from annotated_types import MaxLen
from pydantic.fields import FieldInfo

import app.tools
from tests.tools._generation_jobs import InMemoryGenerationJobs


class TestGrepPatternCap:
    """grep_context bounds the pattern size and compiles user regex off the event loop.

    Pattern compilation is pure-Python parsing that runs BEFORE any matching
    timeout applies and whose cost grows with pattern size, so an unbounded
    multi-megabyte pattern could pin the whole event loop for seconds just to
    compile. The cap is advertised in the wire schema (max_length on the pattern
    Field) and re-checked in the tool body; a legal-size user regex is compiled
    under asyncio.to_thread so even a slow compile cannot stall other requests.
    """

    def test_pattern_declares_max_length_cap(self) -> None:
        """The pattern Field declares max_length=GREP_MAX_PATTERN_CHARS."""
        import app.tools.navigation.grep as grep_mod

        hints = get_type_hints(app.tools.grep_context, include_extras=True)
        field_info = next(
            meta for meta in get_args(hints['pattern'])[1:] if isinstance(meta, FieldInfo)
        )
        max_values = [constraint.max_length for constraint in field_info.metadata if isinstance(constraint, MaxLen)]
        assert max_values == [grep_mod.settings.grep_context.max_pattern_chars]

    @pytest.mark.asyncio
    async def test_oversized_pattern_structured_error_before_compile(self) -> None:
        """A direct call with an oversized pattern returns a structured validation
        error before the pattern is compiled and before the scan query runs."""
        import app.tools.navigation.grep as grep_mod

        cap = grep_mod.settings.grep_context.max_pattern_chars
        repos = MagicMock()
        repos.context.grep_scan_text_contents = AsyncMock()
        compile_spy = MagicMock()
        with (
            patch('app.tools.navigation.grep.ensure_repositories', AsyncMock(return_value=repos)),
            patch('app.tools.navigation.grep.compile_pattern', compile_spy),
        ):
            result = await app.tools.grep_context(pattern='x' * (cap + 1))

        payload = cast('dict[str, Any]', result)
        assert payload['results'] == []
        assert payload['total_matches'] == 0
        assert 'Pattern too long' in payload['error']
        assert payload['validation_errors'] == [payload['error']]
        compile_spy.assert_not_called()
        repos.context.grep_scan_text_contents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pattern_at_cap_accepted(self) -> None:
        """A pattern of exactly the cap length passes the boundary check (inclusive)."""
        import app.tools.navigation.grep as grep_mod

        cap = grep_mod.settings.grep_context.max_pattern_chars
        repos = MagicMock()
        repos.context.grep_scan_text_contents = AsyncMock(return_value=([], {}))
        repos.generation_jobs = InMemoryGenerationJobs()
        with patch('app.tools.navigation.grep.ensure_repositories', AsyncMock(return_value=repos)):
            result = await app.tools.grep_context(pattern='x' * cap)

        payload = cast('dict[str, Any]', result)
        assert 'error' not in payload
        assert payload['total_matches'] == 0

    @pytest.mark.asyncio
    async def test_regex_compile_runs_in_worker_thread(self) -> None:
        """is_regex compilation is offloaded via asyncio.to_thread (never inline on the loop)."""
        import asyncio as asyncio_module

        from app.services.grep_service import compile_pattern as real_compile

        real_to_thread = asyncio_module.to_thread
        offloaded: list[Callable[..., object]] = []

        async def spy_to_thread(fn: Callable[..., object], /, *args: object, **kwargs: object) -> object:
            offloaded.append(fn)
            return await real_to_thread(fn, *args, **kwargs)

        repos = MagicMock()
        repos.context.grep_scan_text_contents = AsyncMock(return_value=([], {}))
        repos.generation_jobs = InMemoryGenerationJobs()
        with (
            patch('app.tools.navigation.grep.ensure_repositories', AsyncMock(return_value=repos)),
            patch('app.tools.navigation.grep.asyncio.to_thread', side_effect=spy_to_thread),
        ):
            result = await app.tools.grep_context(pattern='nee.dle', is_regex=True)

        payload = cast('dict[str, Any]', result)
        assert 'error' not in payload
        assert real_compile in offloaded
