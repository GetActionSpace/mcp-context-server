"""Tests for update_context generation gating: embedding and summary providers, and clearing stale index_tree
node rows on a text change."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

import app.tools
from app.tools._transactions import UpdateOutcome

if TYPE_CHECKING:
    from app.settings import AppSettings

# Tools are plain async functions registered at server startup, so tests call them directly.
update_context = app.tools.update_context


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


class TestUpdateContext:
    """Test suite for update_context tool."""

    @pytest.mark.asyncio
    async def test_update_context_no_embedding_task_when_provider_none(self, mock_repositories):
        """No embedding generation when embedding provider is None."""
        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
            patch('app.tools._generation.generate_embeddings_with_timeout') as mock_embed,
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='Updated text',
                metadata=None,
                tags=None,
                images=None,
            )

        assert result['success'] is True
        mock_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_context_embedding_task_when_provider_exists(self, mock_repositories):
        """Embedding task IS appended when embedding provider exists."""
        mock_provider = AsyncMock()
        mock_embeddings_result = [{'chunk_index': 0, 'text': 'Updated text', 'embedding': [0.1] * 1024}]

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
            patch(
                'app.tools._generation.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=mock_embeddings_result,
            ),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='Updated text',
                metadata=None,
                tags=None,
                images=None,
            )

        assert result['success'] is True
        assert 'embedding' in result['updated_fields']

    @pytest.mark.asyncio
    async def test_update_context_text_change_no_provider_skips_generation(self, mock_repositories):
        """Text change with no providers skips both embedding and summary generation."""
        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
            patch('app.tools._generation.generate_embeddings_with_timeout') as mock_embed,
            patch('app.tools._generation.generate_summary_with_timeout') as mock_summary,
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='New text without providers',
                metadata=None,
                tags=None,
                images=None,
            )

        assert result['success'] is True
        mock_embed.assert_not_called()
        mock_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_change_total_node_degradation_clears_stale_nodes(self, mock_repositories):
        """A text-change update whose per-node summaries return None while the per-node layer
        is ENABLED remaps index_nodes to [] so the transaction CLEARS the stale rows (which
        describe the OLD text) rather than preserving them.

        The clear is gated on settings.index_tree.node_summaries_enabled -- the SAME gate
        navigate_context reads -- NOT on a provider being present. A regression dropping the
        clear remap would pass None through and leave the stale rows.
        """
        captured: dict[str, object] = {}

        async def capture_update(_repos, _txn, **kwargs):
            captured['index_nodes'] = kwargs.get('index_nodes')
            return UpdateOutcome(['text_content'], True, 'ready')

        with (
            patch('app.tools.context.update.settings', _settings_with_node_summaries(True)),
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools.context.update.run_generation', new_callable=AsyncMock, return_value=(None, None, None)),
            patch('app.tools.context.update.execute_update_in_transaction', side_effect=capture_update),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='brand new body',
            )

        assert result['success'] is True
        assert captured['index_nodes'] == []  # node layer enabled + None on text change -> cleared

    @pytest.mark.asyncio
    async def test_text_change_no_provider_feature_on_clears_stale_nodes(self, mock_repositories):
        """Provider-removed-but-feature-on is a DISTINCT clear case.

        With ENABLE_INDEX_TREE_NODE_SUMMARIES on but NO summary provider, node generation
        returns None for lack of a provider (not total degradation). navigate_context still
        surfaces stored node rows (it gates only on node_summaries_enabled, no provider), so a
        text-change update MUST clear the stale rows. A prior version gated the clear on
        node_layer_active() (which additionally required a provider) and wrongly PRESERVED
        them, letting navigate_context mis-attach an old summary to a new same-slug section.
        """
        captured: dict[str, object] = {}

        async def capture_update(_repos, _txn, **kwargs):
            captured['index_nodes'] = kwargs.get('index_nodes')
            return UpdateOutcome(['text_content'], True, 'ready')

        with (
            patch('app.tools.context.update.settings', _settings_with_node_summaries(True)),
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools.context.update.run_generation', new_callable=AsyncMock, return_value=(None, None, None)),
            patch('app.tools.context.update.execute_update_in_transaction', side_effect=capture_update),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='brand new body',
            )

        assert result['success'] is True
        assert captured['index_nodes'] == []  # feature on, provider gone -> stale rows cleared

    @pytest.mark.asyncio
    async def test_text_change_feature_off_clears_stale_nodes(self, mock_repositories):
        """A text-change update CLEARS stale node rows even when the per-node layer is DISABLED.

        The clear is UNCONDITIONAL (not gated on node_summaries_enabled), so a
        disable/edit/re-enable cycle cannot resurface pre-edit rows: without this, an admin
        could turn the feature off (a documented reversible cost lever), edit the entry's text
        while off (leaving the stale rows in place), then re-enable -- and navigate_context
        would mis-attach the pre-edit summary onto the changed section by its reused heading
        slug. replace_nodes_for_context pre-checks table existence, so clearing while the table
        is absent is a safe no-op.
        """
        captured: dict[str, object] = {}

        async def capture_update(_repos, _txn, **kwargs):
            captured['index_nodes'] = kwargs.get('index_nodes')
            return UpdateOutcome(['text_content'], True, 'ready')

        with (
            patch('app.tools.context.update.settings', _settings_with_node_summaries(False)),
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools.context.update.run_generation', new_callable=AsyncMock, return_value=(None, None, None)),
            patch('app.tools.context.update.execute_update_in_transaction', side_effect=capture_update),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='brand new body',
            )

        assert result['success'] is True
        assert captured['index_nodes'] == []  # feature off + text change -> stale rows still cleared
