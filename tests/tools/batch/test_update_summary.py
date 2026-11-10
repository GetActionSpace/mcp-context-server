"""Tests for summary and per-node summary handling in update_context_batch, in atomic and non-atomic modes."""

from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from tests.tools.batch._mocks import create_mock_repositories
from tests.tools.batch._mocks import preserved_providers

if TYPE_CHECKING:
    from app.settings import AppSettings

update_context_batch = app.tools.update_context_batch


def _settings_with_node_summaries(enabled: bool) -> 'AppSettings':
    """Return an AppSettings copy with ``index_tree.node_summaries_enabled`` overridden.

    CommonSettings is frozen, so the per-node toggle is flipped via ``model_copy`` and the
    whole module-level ``settings`` binding is patched (the nested attribute cannot be set).

    Returns:
        An AppSettings copy identical to the cached settings except the per-node toggle.
    """
    from app.settings import get_settings

    base = get_settings()
    return base.model_copy(
        update={'index_tree': base.index_tree.model_copy(update={'node_summaries_enabled': enabled})},
    )


@pytest.fixture(autouse=True)
def reset_providers() -> Generator[None, None, None]:
    """Reset global provider state between tests."""
    with preserved_providers():
        yield


@pytest.mark.usefixtures('mock_server_dependencies')
class TestUpdateContextBatchWithSummary:
    """Tests for summary generation in update_context_batch."""

    @pytest.mark.asyncio
    async def test_update_batch_text_change_regenerates_summary(self) -> None:
        """Generate new summary when text is changed in batch update."""
        repos = create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Updated batch summary')

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500},
            {'context_id': '0190abcdef1234567890abcd00000002', 'text': 'y' * 500},
        ]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 2
        assert '(summaries regenerated)' in result['message']
        assert mock_summary.summarize.await_count == 2

        # Verify summary passed to update_context_entry
        for call in repos.context.update_context_entry.call_args_list:
            assert call.kwargs.get('summary') == 'Updated batch summary'

    @pytest.mark.asyncio
    async def test_update_batch_no_text_change_skips_summary(self) -> None:
        """Skip summary generation when only metadata is updated."""
        repos = create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not appear')

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000001', 'metadata': {'key': 'value'}},
        ]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        mock_summary.summarize.assert_not_awaited()

        # Verify summary=None passed when no text change
        call_kwargs = repos.context.update_context_entry.call_args.kwargs
        assert call_kwargs.get('summary') is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize('atomic', [True, False])
    async def test_update_batch_text_change_no_provider_clears_stale_summary(self, atomic: bool) -> None:
        """A text-change batch update with NO summary provider clears the now-stale summary.

        Mirrors the single-update contract (test_update_text_content_only): the stored summary
        describes the REPLACED text, so update_context_batch must pass clear_summary=True /
        summary=None to update_context_entry instead of leaving a stale summary that no longer
        matches the entry's text. Parametrized over atomic to cover both the atomic and
        non-atomic execute_update_in_transaction call sites, which both read the
        update_clear_summaries set.
        """
        repos = create_mock_repositories()

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
        ):
            result = await update_context_batch(updates=updates, atomic=atomic)

        assert result['success'] is True
        # The stale summary is cleared (not preserved, not regenerated).
        call_kwargs = repos.context.update_context_entry.call_args.kwargs
        assert call_kwargs.get('clear_summary') is True
        assert call_kwargs.get('summary') is None

    @pytest.mark.asyncio
    async def test_update_batch_atomic_summary_failure(self) -> None:
        """Fail entire atomic batch when summary generation fails."""
        repos = create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=RuntimeError('Provider down'))

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            pytest.raises(ToolError, match='Generation failed'),
        ):
            await update_context_batch(updates=updates, atomic=True)

        # No data should have been modified
        repos.context.update_context_entry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_batch_non_atomic_partial_summary_failure(self) -> None:
        """Report per-entry errors in non-atomic mode when summary fails for some."""
        repos = create_mock_repositories()

        call_count = 0

        async def selective_summary(_text: str, _source: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError('LLM overloaded')
            return 'Generated summary'

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=selective_summary)

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500},
            {'context_id': '0190abcdef1234567890abcd00000002', 'text': 'y' * 500},
        ]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=False)

        assert result['succeeded'] == 1
        assert result['failed'] == 1

        failed_results = [r for r in result['results'] if not r['success']]
        assert len(failed_results) == 1
        assert failed_results[0]['error'] is not None
        assert 'Generation failed' in failed_results[0]['error']

    @pytest.mark.asyncio
    async def test_update_batch_text_change_total_node_degradation_clears_stale_nodes(self) -> None:
        """A text-change batch update whose per-node summaries return None while the per-node
        layer is ENABLED clears the stale rows ([] -> replace), rather than preserving
        summaries describing the OLD text. The clear is gated on
        settings.index_tree.node_summaries_enabled (the gate navigate_context reads), NOT on
        a provider being present.
        """
        repos = create_mock_repositories()
        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='flat ok')

        updates = [{'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500}]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch(
                'app.tools.batch._update_generation.generate_index_nodes_with_timeout',
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch('app.tools._generation.settings', _settings_with_node_summaries(True)),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        # index_nodes passed positionally as (context_id, index_nodes, txn=...).
        repos.index_nodes.replace_nodes_for_context.assert_awaited_once()
        assert repos.index_nodes.replace_nodes_for_context.call_args.args[1] == []

    @pytest.mark.asyncio
    async def test_update_batch_text_change_feature_off_clears_stale_nodes(self) -> None:
        """A text-change batch update CLEARS stale node rows even when the per-node layer is
        DISABLED. The clear is UNCONDITIONAL (not gated on node_summaries_enabled), so a
        disable/edit/re-enable cycle cannot resurface pre-edit rows that navigate_context would
        mis-attach to a reused heading slug once the feature is turned back on.
        replace_nodes_for_context pre-checks table existence, so clearing while the table is
        absent is a safe no-op.
        """
        repos = create_mock_repositories()
        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='flat ok')

        updates = [{'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500}]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch(
                'app.tools.batch._update_generation.generate_index_nodes_with_timeout',
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch('app.tools._generation.settings', _settings_with_node_summaries(False)),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        # feature off + text change -> stale rows still cleared ([] -> replace).
        repos.index_nodes.replace_nodes_for_context.assert_awaited_once()
        assert repos.index_nodes.replace_nodes_for_context.call_args.args[1] == []
