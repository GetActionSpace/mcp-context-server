"""Tests for the summary generation error contract and the generation availability checks in app.tools._generation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.summary.retry import SummaryRetryExhaustedError
from app.summary.retry import SummaryTimeoutError
from app.tools._generation import generate_summary_with_timeout
from app.tools._generation import generation_active
from app.tools._generation import summary_applies


class TestGenerateSummaryErrorContract:
    """generate_summary_with_timeout normalizes EVERY failure to ToolError.

    Its sibling abort-mandatory leg (generate_embeddings_with_timeout) already
    converts any provider failure into a ToolError, and call sites that isolate a
    per-entry failure rely on that shared contract with ``except ToolError``. The
    summary helper used to convert only its own outer timeout, so the retry layer's
    SummaryTimeoutError / SummaryRetryExhaustedError and the providers' bare
    RuntimeError / ValueError escaped raw and slipped past those guards.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'failure',
        [
            SummaryRetryExhaustedError('retries exhausted'),
            SummaryTimeoutError('provider timed out'),
            RuntimeError('ollama unreachable'),
            ValueError('text too long for the model context'),
        ],
        ids=['retries-exhausted', 'provider-timeout', 'runtime-error', 'value-error'],
    )
    async def test_provider_failures_become_tool_error(self, failure: Exception) -> None:
        """Every provider failure class arrives as a ToolError naming the leg."""
        provider = MagicMock()
        provider.summarize = AsyncMock(side_effect=failure)
        with (
            patch('app.tools._generation.get_summary_provider', return_value=provider),
            pytest.raises(ToolError, match='Summary generation failed'),
        ):
            await generate_summary_with_timeout('body text', 'agent')

    @pytest.mark.asyncio
    async def test_outer_timeout_keeps_its_dedicated_message(self) -> None:
        """The total-timeout message stays distinct from the generic failure message."""

        async def _never_returns(text: str, source: str) -> str:
            _ = (text, source)
            await asyncio.sleep(10)
            return 'unreachable'

        provider = MagicMock()
        provider.summarize = _never_returns
        with (
            patch('app.tools._generation.get_summary_provider', return_value=provider),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=0.01),
            pytest.raises(ToolError, match='exceeded total timeout'),
        ):
            await generate_summary_with_timeout('body text', 'agent')

    @pytest.mark.asyncio
    async def test_cancellation_still_propagates(self) -> None:
        """Cancellation is NOT swallowed: run_generation cancels this leg on abort."""
        started = asyncio.Event()

        async def _block(text: str, source: str) -> str:
            _ = (text, source)
            started.set()
            await asyncio.sleep(10)
            return 'unreachable'

        provider = MagicMock()
        provider.summarize = _block
        with patch('app.tools._generation.get_summary_provider', return_value=provider):
            task = asyncio.create_task(generate_summary_with_timeout('body text', 'agent'))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


class TestGenerationActive:
    """generation_active reports whether an active provider can generate an entry's outputs."""

    @pytest.mark.parametrize(
        ('embedding', 'summary', 'expected'),
        [(None, None, False), (MagicMock(), None, True), (None, MagicMock(), True), (MagicMock(), MagicMock(), True)],
    )
    def test_any_active_provider_makes_generation_active(
        self, embedding: MagicMock | None, summary: MagicMock | None, *, expected: bool,
    ) -> None:
        with (
            patch('app.tools._generation.get_embedding_provider', return_value=embedding),
            patch('app.tools._generation.get_summary_provider', return_value=summary),
        ):
            assert generation_active() is expected


def _min_length_settings(min_content_length: int) -> SimpleNamespace:
    """Settings carrying only the summary minimum length summary_applies reads."""
    return SimpleNamespace(summary=SimpleNamespace(min_content_length=min_content_length))


class TestSummaryApplies:
    """summary_applies reports whether a summary is generated for a text."""

    def test_no_summary_without_a_provider(self) -> None:
        with patch('app.tools._generation.get_summary_provider', return_value=None):
            assert summary_applies('x' * 10_000) is False

    def test_text_shorter_than_the_minimum_gets_no_summary(self) -> None:
        with (
            patch('app.tools._generation.get_summary_provider', return_value=MagicMock()),
            patch('app.tools._generation.settings', _min_length_settings(100)),
        ):
            assert summary_applies('x' * 99) is False
            assert summary_applies('x' * 100) is True

    def test_a_zero_minimum_summarizes_every_text(self) -> None:
        with (
            patch('app.tools._generation.get_summary_provider', return_value=MagicMock()),
            patch('app.tools._generation.settings', _min_length_settings(0)),
        ):
            assert summary_applies('x') is True
