"""Tests for update_context_batch: failed entry writes in atomic and non-atomic modes, the 'summaries cleared'
response, and non-atomic updates sharing a context_id."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.access_scope import Scope
from app.repositories.context_repository.records import EntryProbe
from app.tools.context.retrieve import get_context_by_ids
from tests.helpers import LOCAL_WRITE_SCOPE
from tests.tools._read_only import read_only_caller
from tests.tools.batch._mocks import make_mock_txn


@pytest.mark.usefixtures('initialized_server')
class TestUpdateBatchFailureHandling:
    """update_context_batch surfaces a failed entry write (update or metadata patch) instead of swallowing it."""

    @pytest.mark.asyncio
    async def test_update_batch_atomic_raises_on_update_entry_failure(self):
        """Atomic mode aborts the whole batch when the target entry no longer exists."""
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-atomic-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = make_mock_txn()

        with patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            with pytest.raises(ToolError, match=r'No entries were updated \(atomic batch\)'):
                await update_context_batch(
                    updates=[{
                        'context_id': context_id,
                        'text': 'Updated text',
                    }],
                    atomic=True,
                )

    @pytest.mark.asyncio
    async def test_update_batch_atomic_raises_on_patch_metadata_failure(self):
        """Atomic mode aborts the whole batch when the target entry no longer exists on the patch path."""
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-atomic-patch-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = make_mock_txn()

        with patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            with pytest.raises(ToolError, match=r'No entries were updated \(atomic batch\)'):
                await update_context_batch(
                    updates=[{
                        'context_id': context_id,
                        'metadata_patch': {'key': 'value'},
                    }],
                    atomic=True,
                )

    @pytest.mark.asyncio
    async def test_update_batch_nonatomic_reports_update_entry_failure(self):
        """Non-atomic mode records a per-entry not-found failure when the target entry no longer exists."""
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-nonatomic-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = make_mock_txn()

        with patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'text': 'Updated text',
                }],
                atomic=False,
            )
            assert result['failed'] == 1
            failed_entry = result['results'][0]
            assert failed_entry['success'] is False
            assert failed_entry['error'] is not None

    @pytest.mark.asyncio
    async def test_update_batch_nonatomic_reports_patch_metadata_failure(self):
        """Non-atomic mode records a per-entry not-found failure when the target entry no longer exists on the patch path."""
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-nonatomic-patch-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = make_mock_txn()

        with patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'metadata_patch': {'key': 'value'},
                }],
                atomic=False,
            )
            assert result['failed'] == 1
            failed_entry = result['results'][0]
            assert failed_entry['success'] is False
            assert failed_entry['error'] is not None
            assert 'not found' in str(failed_entry['error'])


@pytest.mark.usefixtures('initialized_server')
class TestBatchUpdateResponseParity:
    """Batch update response includes 'summaries cleared'."""

    @pytest.mark.asyncio
    async def test_update_batch_reports_summaries_cleared(self):
        """Batch update reports 'summaries cleared' when summaries are removed."""
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-summary-cleared',
                'source': 'user',
                'text': 'A' * 1000,
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = make_mock_txn()

        mock_settings = MagicMock()
        mock_settings.embedding.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.summary.enabled = True
        mock_settings.summary.min_content_length = 500

        with (
            patch('app.tools.batch.update.settings', mock_settings),
            patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=MagicMock()),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(True, ['text']))
            mock_repos.images.count_images_for_context = AsyncMock(return_value=0)

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Update with short text (below min_content_length) triggers summary clearing
            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'text': 'Short',
                }],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert 'summaries cleared' in result['message']


@pytest.mark.usefixtures('initialized_server')
class TestUpdateBatchSiblingNotDropped:
    """A sibling update sharing a context_id with a failed one must NOT be dropped."""

    @pytest.mark.asyncio
    async def test_nonatomic_failed_generation_keeps_sibling_same_context_id(self):
        """Two non-atomic updates target the same context_id; only the failing one is dropped.

        index 0 (text update) fails embedding generation; index 1 (metadata-only,
        no generation) shares the same context_id. The non-atomic filter must drop
        only index 0 by ORIGINAL INDEX, leaving index 1 to be applied. Filtering by a
        context_id set would drop both and silently lose index 1.
        """
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'sibling-drop-thread',
                'source': 'user',
                'text': 'A' * 600,
            }],
        )
        cid = store_result['results'][0]['context_id']
        assert cid is not None

        _, mock_begin_transaction = make_mock_txn()

        async def emb_side_effect(text):
            if 'FAILING' in text:
                raise ToolError('boom: embedding generation failed')
            return [('chunk-0', [0.1, 0.2, 0.3])]

        with (
            patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.update.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.batch.update.get_summary_provider', return_value=None),
            patch(
                'app.tools.batch.update.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                side_effect=emb_side_effect,
            ),
            patch(
                'app.tools.batch.update.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            # index 1 is a metadata-only update -> patch_metadata applies it.
            mock_repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
            mock_repos.images.count_images_for_context = AsyncMock(return_value=0)

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[
                    {'context_id': cid, 'text': 'FAILING update text content'},
                    {'context_id': cid, 'metadata_patch': {'reviewed': True}},
                ],
                atomic=False,
            )

        by_index = {r['index']: r for r in result['results']}
        # Failing text update is reported as failed...
        assert by_index[0]['success'] is False
        assert by_index[0]['error'] is not None
        # ...but its same-context_id sibling is NOT dropped -- it is applied.
        assert by_index[1]['success'] is True
        assert result['total'] == 2
        assert result['succeeded'] == 1
        assert result['failed'] == 1

    @pytest.mark.asyncio
    async def test_nonatomic_missing_entry_keeps_existing_sibling_same_context_id(self):
        """The EXISTENCE filter drops only the missing index, not its same-context_id sibling.

        Two non-atomic updates target the same context_id; a concurrent delete makes the
        SECOND existence check fail while the FIRST passed. The filter must drop only the
        missing entry by ORIGINAL INDEX. Filtering by a context_id set would drop BOTH,
        leaving the existing sibling with no result item (len(results) < total) and never
        applied. The existence filter drops by index exactly like the generation-error filter.
        """
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'existence-divergence-thread',
                'source': 'user',
                'text': 'B' * 600,
            }],
        )
        cid = store_result['results'][0]['context_id']
        assert cid is not None

        _, mock_begin_transaction = make_mock_txn()

        # index 0's existence check passes; index 1's fails, as if a concurrent delete of
        # the shared context_id committed between the two sequential checks.
        exists_results = [EntryProbe(True, 'user', 0, 'local', True), EntryProbe(False, None, None, None, False)]

        async def exists_side_effect(_context_id: str, *, scope: Scope) -> EntryProbe:
            assert scope == LOCAL_WRITE_SCOPE
            return exists_results.pop(0)

        with (
            patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools.batch.update.get_summary_provider', return_value=None),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(side_effect=exists_side_effect)
            mock_repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
            mock_repos.images.count_images_for_context = AsyncMock(return_value=0)

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[
                    {'context_id': cid, 'metadata_patch': {'first': True}},
                    {'context_id': cid, 'metadata_patch': {'second': True}},
                ],
                atomic=False,
            )

        by_index = {r['index']: r for r in result['results']}
        # Every input index gets exactly one result item -- the sibling is not silently lost.
        assert set(by_index) == {0, 1}
        assert len(result['results']) == 2
        assert by_index[0]['success'] is True   # existence check passed -> applied
        assert by_index[1]['success'] is False  # concurrent delete -> not found
        assert result['total'] == 2
        assert result['succeeded'] == 1
        assert result['failed'] == 1


_INGEST_BLOCK: dict[str, object] = {
    'scope': 'kb:handbook',
    'source_id': 'guides/setup.md',
    'meta_digest': 'ab' * 32,
    'synced_at': '2026-10-18T12:00:00Z',
}
_PATCH_SHAPE_ERROR = (
    'metadata_patch.ingest must be null or an object with exactly the keys meta_digest, scope, source_id, synced_at'
)


@pytest.mark.usefixtures('initialized_server')
class TestUpdateBatchIngestBlock:
    """update_context_batch refuses a malformed ingest block per entry, before any generation."""

    @staticmethod
    async def _store_three() -> list[str]:
        from app.tools.batch.store import store_context_batch

        stored = await store_context_batch(
            entries=[
                {'thread_id': 'kb:handbook', 'source': 'user', 'text': text, 'metadata': {'ingest': _INGEST_BLOCK}}
                for text in ('Setup guide', 'FAQ', 'Release notes')
            ],
        )
        ids = [item['context_id'] for item in stored['results']]
        assert all(context_id is not None for context_id in ids)
        return [context_id for context_id in ids if context_id is not None]

    @pytest.mark.asyncio
    async def test_nonatomic_fails_only_the_malformed_update(self) -> None:
        """A partial block in a patch fails that update; a null patch and a complete block apply."""
        from app.tools.batch.update import update_context_batch

        first, second, third = await self._store_three()
        replacement = {**_INGEST_BLOCK, 'meta_digest': 'cd' * 32}

        result = await update_context_batch(
            updates=[
                {'context_id': first, 'metadata_patch': {'ingest': None}},
                {'context_id': second, 'metadata_patch': {'ingest': {'meta_digest': 'cd' * 32}}},
                {'context_id': third, 'metadata': {'ingest': replacement}},
            ],
            atomic=False,
        )

        by_index = {item['index']: item for item in result['results']}
        assert by_index[1] == {
            'index': 1, 'context_id': second, 'success': False, 'updated_fields': None, 'error': _PATCH_SHAPE_ERROR,
            'generation_status': None,
        }
        assert by_index[0]['success'] is True
        assert by_index[2]['success'] is True
        fetched = {entry.get('id'): entry for entry in await get_context_by_ids(context_ids=[first, second, third])}
        assert fetched[first].get('metadata') == {}
        assert fetched[second].get('metadata') == {'ingest': _INGEST_BLOCK}
        assert fetched[third].get('metadata') == {'ingest': replacement}

    @pytest.mark.asyncio
    async def test_atomic_aborts_naming_the_malformed_entry_before_generation(self) -> None:
        """The atomic batch aborts naming the entry; no provider is reached and nothing changes."""
        from app.tools.batch.update import update_context_batch

        first, second, _ = await self._store_three()
        embedding_provider = AsyncMock()
        summary_provider = AsyncMock()
        with (
            patch('app.tools.batch.update.get_embedding_provider', return_value=embedding_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=embedding_provider),
            patch('app.tools.batch.update.get_summary_provider', return_value=summary_provider),
            pytest.raises(ToolError) as exc_info,
        ):
            await update_context_batch(
                updates=[
                    {'context_id': first, 'text': 'Setup guide, revised'},
                    {'context_id': second, 'metadata': {'ingest': {**_INGEST_BLOCK, 'scope': ''}}},
                ],
                atomic=True,
            )

        assert str(exc_info.value) == (
            f'Validation failed for 1 entries. First error at context_id {second}: metadata.ingest.scope cannot be empty'
        )
        assert embedding_provider.mock_calls == []
        assert summary_provider.mock_calls == []
        fetched = {entry.get('id'): entry for entry in await get_context_by_ids(context_ids=[first, second])}
        assert fetched[first].get('text_content') == 'Setup guide'
        assert fetched[second].get('metadata') == {'ingest': _INGEST_BLOCK}


class TestBatchUpdateReadOnlyPrincipal:
    """The read-only principal is refused before any entry validation, id resolution or generation."""

    @pytest.mark.asyncio
    async def test_read_only_principal_is_refused_first(self) -> None:
        """A batch naming no valid id and no field still gets the read-only refusal, and nothing runs."""
        from app.tools.batch.update import update_context_batch

        with (
            read_only_caller(),
            patch('app.tools.batch.update.ensure_repositories', new=AsyncMock()) as ensure,
            patch('app.tools.batch.update.validate_update_entry', new=AsyncMock()) as validate,
            patch('app.tools.batch.update.get_embedding_provider') as embedding_provider,
            patch('app.tools.batch.update.get_summary_provider') as summary_provider,
            pytest.raises(ToolError) as exc_info,
        ):
            await update_context_batch(updates=[{'context_id': 'not-a-context-id'}])

        assert str(exc_info.value) == 'The anonymous principal is read-only: update_context_batch is not permitted'
        ensure.assert_not_awaited()
        validate.assert_not_awaited()
        embedding_provider.assert_not_called()
        summary_provider.assert_not_called()
