"""Tests for the generated, preserved and regenerated counts in batch store and update response messages."""

from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories.context_repository.records import DuplicateCandidate
from app.repositories.context_repository.records import EntryProbe
from tests.tools.batch._mocks import create_mock_repositories
from tests.tools.batch._mocks import preserved_providers

store_context_batch = app.tools.store_context_batch
update_context_batch = app.tools.update_context_batch


@pytest.fixture(autouse=True)
def reset_providers() -> Generator[None, None, None]:
    """Reset global provider state between tests."""
    with preserved_providers():
        yield


@pytest.mark.usefixtures('mock_server_dependencies')
class TestBatchMessageAccuracy:
    """Tests that batch message reflects actual generation, not provider availability."""

    @pytest.mark.asyncio
    async def test_store_batch_short_text_no_summary_message(self) -> None:
        """Message omits 'summaries generated' when all entries skip summary due to min_content_length."""
        repos = create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(
            side_effect=[(101, False), (102, False)],
        )

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not be called')

        entries = [
            {'thread_id': 'batch-short', 'source': 'user', 'text': 'Short text'},
            {'thread_id': 'batch-short', 'source': 'agent', 'text': 'Also short'},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert 'summaries generated' not in result['message']
        mock_summary.summarize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_batch_preserved_summary_not_counted_as_generated(self) -> None:
        """A reused (preserved) summary is not also counted as 'generated'.

        When a likely-duplicate entry reuses its existing summary AND has absent
        embeddings, an embedding task is queued so the gather runs; the success
        block must not count the preserved summary as generated, or the message
        claims BOTH 'summaries generated' and 'summaries preserved' for the same
        entry. The count is gated on a summary task having actually run.
        """
        repos = create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(500, True))
        # Likely-duplicate with an existing summary to REUSE, but absent embeddings
        # (so an embedding task IS queued and the gather runs).
        repos.context.check_latest_is_duplicate = AsyncMock(
            return_value=DuplicateCandidate(
                context_id='0190abcdef1234567890abcd00000009',
                summary='Existing preserved summary',
            ),
        )
        repos.embeddings.exists = AsyncMock(return_value=False)

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should NOT be generated')

        entries = [
            {'thread_id': 'preserve-sum', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools._generation.get_chunking_service', return_value=None),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        # The reused summary is reported as preserved, NEVER as generated.
        assert 'summaries preserved' in result['message']
        assert 'summaries generated' not in result['message']
        # No summary model call ran for the reused summary.
        mock_summary.summarize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nonatomic_discarded_entry_not_counted_as_preserved(self) -> None:
        """A preserved-summary entry discarded by a generation failure is not counted.

        Counts must reflect entries surviving the generation phase: the
        pre-check provisionally bumps the preserved count when a likely
        duplicate's stored summary is reused, and the compression-failure
        branch already compensates on discard. The sibling embedding-failure
        discard branch must compensate the same way, or the response claims
        'summaries preserved' for an entry that was never stored.
        """
        repos = create_mock_repositories()
        # Likely-duplicate with a reusable summary but absent embeddings, so
        # an embedding task IS queued -- and then fails.
        repos.context.check_latest_is_duplicate = AsyncMock(
            return_value=DuplicateCandidate(
                context_id='0190abcdef1234567890abcd00000010',
                summary='Existing preserved summary',
            ),
        )
        repos.embeddings.exists = AsyncMock(return_value=False)

        mock_embedding = MagicMock()
        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should NOT be generated')

        entries = [
            {'thread_id': 'preserve-discard', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch(
                'app.tools.batch._store_generation.generate_embeddings_with_timeout',
                new=AsyncMock(side_effect=ToolError('embedding: provider unavailable')),
            ),
        ):
            result = await store_context_batch(entries=entries, atomic=False)

        # The lone entry was discarded during generation, so no summary
        # survived to be preserved and the message must not claim one.
        assert result['results'][0]['success'] is False
        assert 'summaries preserved' not in result['message']

    @pytest.mark.asyncio
    async def test_nonatomic_transaction_failure_not_labeled_duplicate(self) -> None:
        """A transaction-phase failure is not reported as a dedup skip.

        The response message computes not_stored = generated - stored and
        labels the whole gap 'not stored - duplicates', so an entry that
        survived generation but failed at commit time (a logical ToolError in
        its own transaction) would otherwise be reported as a duplicate skip with
        its generated summary still claimed. The failure branches reverse the
        entry's generated-counter contributions, so a batch with zero stored
        entries claims nothing.
        """
        repos = create_mock_repositories()

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='A generated summary')

        entries = [
            {'thread_id': 'txn-fail', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools._generation.get_chunking_service', return_value=None),
            patch(
                'app.tools.batch._store_commit.execute_store_in_transaction',
                new=AsyncMock(side_effect=ToolError('Failed to store context')),
            ),
        ):
            result = await store_context_batch(entries=entries, atomic=False)

        assert result['results'][0]['success'] is False
        assert 'duplicates' not in result['message']
        assert 'embeddings generated' not in result['message']
        assert 'summaries generated' not in result['message']

    @pytest.mark.asyncio
    async def test_nonatomic_transaction_failure_not_counted_as_preserved(self) -> None:
        """A preserved-summary entry failing at commit time is not counted.

        The pre-check provisionally bumps the preserved count when a likely
        duplicate's stored summary is reused; the generation-phase discard
        branches already compensate. The transaction-phase failure branches
        must compensate the same way, or the response claims 'summaries
        preserved' for an entry that was never stored.
        """
        repos = create_mock_repositories()
        repos.context.check_latest_is_duplicate = AsyncMock(
            return_value=DuplicateCandidate(
                context_id='0190abcdef1234567890abcd00000011',
                summary='Existing preserved summary',
            ),
        )
        repos.embeddings.exists = AsyncMock(return_value=False)

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should NOT be generated')

        entries = [
            {'thread_id': 'preserve-txn-fail', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools._generation.get_chunking_service', return_value=None),
            patch(
                'app.tools.batch._store_commit.execute_store_in_transaction',
                new=AsyncMock(side_effect=ToolError('Failed to store context')),
            ),
        ):
            result = await store_context_batch(entries=entries, atomic=False)

        assert result['results'][0]['success'] is False
        assert 'summaries preserved' not in result['message']
        assert 'duplicates' not in result['message']

    @pytest.mark.asyncio
    async def test_nonatomic_update_transaction_failure_not_counted_as_regenerated(self) -> None:
        """An update failing at commit time is not reported as regenerated.

        The update response message claims 'embeddings regenerated' /
        'summaries regenerated' whenever the generation-phase counters are
        positive, so an update that regenerated both but failed in its own
        transaction would otherwise produce 'Updated 0/1 ... (embeddings
        regenerated, summaries regenerated)'. The failure branches reverse the
        update's generated-counter contributions.
        """
        repos = create_mock_repositories()

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='A regenerated summary')

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000012', 'text': 'y' * 600},
        ]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools._generation.get_chunking_service', return_value=None),
            patch(
                'app.tools.batch.update.execute_update_in_transaction',
                new=AsyncMock(side_effect=ToolError('Failed to update context')),
            ),
        ):
            result = await update_context_batch(updates=updates, atomic=False)

        assert result['results'][0]['success'] is False
        assert 'regenerated' not in result['message']

    @pytest.mark.asyncio
    async def test_update_batch_short_text_no_summary_message(self) -> None:
        """Message omits 'summaries regenerated' when all entries skip summary due to min_content_length."""
        repos = create_mock_repositories()
        repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
        repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not be called')

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000001', 'text': 'Short text'},
            {'context_id': '0190abcdef1234567890abcd00000002', 'text': 'Also short'},
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
        assert 'summaries regenerated' not in result['message']
        assert 'summaries generated' not in result['message']
        mock_summary.summarize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_batch_no_text_change_no_regeneration_message(self) -> None:
        """Message omits generation info when only metadata is updated (no text changes)."""
        repos = create_mock_repositories()
        repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
        repos.context.update_context_entry = AsyncMock(return_value=(True, ['metadata']))

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not be called')

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2])

        updates = [
            {'context_id': '0190abcdef1234567890abcd00000001', 'metadata': {'key': 'val1'}},
            {'context_id': '0190abcdef1234567890abcd00000002', 'metadata': {'key': 'val2'}},
        ]

        with (
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.update.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert 'embeddings regenerated' not in result['message']
        assert 'summaries regenerated' not in result['message']
        mock_summary.summarize.assert_not_awaited()
        mock_embedding.embed_query.assert_not_awaited()
