"""Generation-first transactional integrity for store_context_batch and update_context_batch.

An atomic batch saves nothing when generation fails for any entry; a non-atomic batch records the
failing entry and saves the others. An update without a text change skips generation, and text
below the summary length threshold clears the stored summary.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories.context_repository.records import EntryProbe
from tests.tools._generation_jobs import InMemoryGenerationJobs

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_mock_repositories() -> MagicMock:
    """Create mock repositories with transaction support for batch tool tests."""
    repos = MagicMock()

    mock_backend = MagicMock()

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        _ = access_scope
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        txn.connection = MagicMock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    repos.context = MagicMock()
    repos.context.backend = mock_backend
    repos.context.store_with_deduplication = AsyncMock(return_value=(100, False))
    repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)
    repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
    repos.context.update_context_entry = AsyncMock(
        return_value=(True, ['text_content', 'summary']),
    )
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.update_content_type = AsyncMock(return_value=True)

    repos.tags = MagicMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    repos.images = MagicMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    repos.context.get_content_type = AsyncMock(return_value='text')

    repos.embeddings = MagicMock()
    repos.embeddings.exists = AsyncMock(return_value=False)
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.delete_all_chunks = AsyncMock()
    repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)

    # Mock index_tree node-summary repository (text-change updates clear stale node rows).
    repos.index_nodes = MagicMock()
    repos.index_nodes.replace_nodes_for_context = AsyncMock(return_value=None)
    repos.index_nodes.get_nodes_for_context = AsyncMock(return_value={})

    repos.generation_jobs = InMemoryGenerationJobs()

    return repos


# ---------------------------------------------------------------------------
# store_context_batch tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures('mock_server_dependencies')
class TestStoreContextBatchGenerationFirst:
    """Tests for store_context_batch generation-first pattern."""

    @pytest.mark.asyncio
    async def test_atomic_embedding_fails_no_data_saved(self) -> None:
        """Atomic batch: embedding fails on one entry -- entire batch fails."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(101, False))

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch(
                'app.tools.batch.store.get_embedding_provider',
                return_value=MagicMock(),  # non-None so embedding task is added
            ),
            patch('app.tools.batch.store.get_summary_provider', return_value=None),
            patch(
                'app.tools.batch._store_generation.generate_embeddings_with_timeout',
                new=AsyncMock(side_effect=ToolError('Embedding generation timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed at index 0'),
        ):
            await app.tools.store_context_batch(
                entries=[
                    {'thread_id': 'bf-1', 'source': 'agent', 'text': 'Entry 1'},
                ],
                atomic=True,
            )

        repos.context.store_with_deduplication.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_atomic_summary_fails_no_data_saved(self) -> None:
        """Atomic batch: summary fails -- entire batch fails, no data saved."""
        repos = _create_mock_repositories()

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.batch._store_generation.generate_summary_with_timeout',
                new=AsyncMock(side_effect=ToolError('Summary generation timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed at index 0'),
        ):
            await app.tools.store_context_batch(
                entries=[
                    {'thread_id': 'bf-2', 'source': 'agent', 'text': 'x' * 500},
                ],
                atomic=True,
            )

        repos.context.store_with_deduplication.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_atomic_partial_generation_failure(self) -> None:
        """Non-atomic batch: one entry fails generation, others succeed."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(200, False))

        call_count = 0

        async def selective_summary(_text: str, _source: str) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError('Provider overloaded')
            return 'Summary ok'

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.batch._store_generation.generate_summary_with_timeout',
                new=AsyncMock(side_effect=selective_summary),
            ),
        ):
            result = await app.tools.store_context_batch(
                entries=[
                    {'thread_id': 'bf-3', 'source': 'agent', 'text': 'x' * 500},
                    {'thread_id': 'bf-3', 'source': 'agent', 'text': 'y' * 500},
                ],
                atomic=False,
            )

        assert result['succeeded'] == 1
        assert result['failed'] == 1
        failed = [r for r in result['results'] if not r['success']]
        assert len(failed) == 1
        assert 'Generation failed' in (failed[0].get('error') or '')

    @pytest.mark.asyncio
    async def test_atomic_both_succeed_data_saved(self) -> None:
        """Atomic batch: both embedding+summary succeed -- data saved."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(
            side_effect=[(101, False), (102, False)],
        )

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.batch._store_generation.generate_summary_with_timeout',
                new=AsyncMock(return_value='Batch summary'),
            ),
        ):
            result = await app.tools.store_context_batch(
                entries=[
                    {'thread_id': 'bf-4', 'source': 'agent', 'text': 'x' * 500},
                    {'thread_id': 'bf-4', 'source': 'agent', 'text': 'y' * 500},
                ],
                atomic=True,
            )

        assert result['success'] is True
        assert result['succeeded'] == 2


# ---------------------------------------------------------------------------
# update_context_batch tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures('mock_server_dependencies')
class TestUpdateContextBatchGenerationFirst:
    """Tests for update_context_batch generation-first pattern."""

    @pytest.mark.asyncio
    async def test_atomic_embedding_fails_no_data_modified(self) -> None:
        """Atomic update batch: embedding fails -- no data modified."""
        repos = _create_mock_repositories()

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch(
                'app.tools.batch.update.get_embedding_provider',
                return_value=MagicMock(),
            ),
            patch('app.tools.batch.update.get_summary_provider', return_value=None),
            patch(
                'app.tools.batch.update.generate_embeddings_with_timeout',
                new=AsyncMock(side_effect=ToolError('Embedding generation timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed for context 0190abcdef1234567890abcd00000001'),
        ):
            await app.tools.update_context_batch(
                updates=[{'context_id': '0190abcdef1234567890abcd00000001', 'text': 'Updated text'}],
                atomic=True,
            )

        repos.context.update_context_entry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_atomic_summary_fails_no_data_modified(self) -> None:
        """Atomic update batch: summary fails -- no data modified."""
        repos = _create_mock_repositories()

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.batch.update.generate_summary_with_timeout',
                new=AsyncMock(side_effect=ToolError('Summary timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed for context 0190abcdef1234567890abcd00000001'),
        ):
            await app.tools.update_context_batch(
                updates=[{'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500}],
                atomic=True,
            )

        repos.context.update_context_entry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_atomic_partial_generation_failure(self) -> None:
        """Non-atomic update batch: one entry fails, others succeed."""
        repos = _create_mock_repositories()

        call_count = 0

        async def selective_summary(_text: str, _source: str) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError('Provider overloaded')
            return 'Summary ok'

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.batch.update.generate_summary_with_timeout',
                new=AsyncMock(side_effect=selective_summary),
            ),
        ):
            result = await app.tools.update_context_batch(
                updates=[
                    {'context_id': '0190abcdef1234567890abcd00000001', 'text': 'x' * 500},
                    {'context_id': '0190abcdef1234567890abcd00000002', 'text': 'y' * 500},
                ],
                atomic=False,
            )

        assert result['succeeded'] == 1
        assert result['failed'] == 1

    @pytest.mark.asyncio
    async def test_no_text_change_skips_generation(self) -> None:
        """Update batch: metadata-only update skips generation entirely."""
        repos = _create_mock_repositories()

        mock_gen_emb = AsyncMock(return_value=None)
        mock_gen_sum = AsyncMock(return_value=None)

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools._generation.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.batch.update.get_summary_provider', return_value=MagicMock()),
            patch('app.tools.batch.update.generate_embeddings_with_timeout', new=mock_gen_emb),
            patch('app.tools.batch.update.generate_summary_with_timeout', new=mock_gen_sum),
        ):
            result = await app.tools.update_context_batch(
                updates=[{'context_id': '0190abcdef1234567890abcd00000001', 'metadata': {'key': 'value'}}],
                atomic=True,
            )

        assert result['success'] is True
        # Generation should NOT have been called for metadata-only update
        mock_gen_emb.assert_not_awaited()
        mock_gen_sum.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_summary_semantics_preserved(self) -> None:
        """Update batch: text shorter than min_content_length clears existing summary."""
        repos = _create_mock_repositories()

        mock_gen_sum = AsyncMock(return_value='Should not be called')

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=MagicMock()),
            patch('app.tools.batch.update.generate_summary_with_timeout', new=mock_gen_sum),
        ):
            # Text is short (< default min_content_length of 500)
            result = await app.tools.update_context_batch(
                updates=[{'context_id': '0190abcdef1234567890abcd00000001', 'text': 'Short text'}],
                atomic=True,
            )

        assert result['success'] is True
        # Summary wrapper should NOT be called (text too short)
        mock_gen_sum.assert_not_awaited()
        # Verify update was called with summary=None and clear_summary flag
        repos.context.update_context_entry.assert_awaited_once()
        call_kwargs = repos.context.update_context_entry.call_args.kwargs
        assert call_kwargs.get('summary') is None
