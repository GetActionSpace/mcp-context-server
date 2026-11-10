"""Tests for execute_update_in_transaction in app.tools._update_transaction."""

from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.access_scope import AccessScope
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools._transactions import EntryNotFoundError
from app.tools._update_transaction import execute_update_in_transaction
from tests.helpers import LOCAL_SCOPE
from tests.tools._generation_jobs import InMemoryGenerationJobs

CONTEXT_ID = '0190abcdef1234567890abcd00000001'


class TestExecuteUpdateInTransaction:
    """Test execute_update_in_transaction shared function."""

    @pytest.fixture
    def mock_repos(self) -> MagicMock:
        """Create a mock RepositoryContainer with all required sub-repositories."""
        repos = MagicMock()
        repos.context.update_context_entry = AsyncMock(return_value=(True, ['text']))
        repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
        repos.context.update_content_type = AsyncMock()
        repos.context.touch_updated_at = AsyncMock(return_value=True)
        repos.context.get_content_type = AsyncMock(return_value='text')
        repos.context.entry_exists = AsyncMock(return_value=True)
        repos.tags.replace_tags_for_context = AsyncMock()
        repos.images.replace_images_for_context = AsyncMock()
        repos.images.count_images_for_context = AsyncMock(return_value=0)
        repos.embeddings.delete_all_chunks = AsyncMock()
        repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)
        repos.embeddings.store_chunked = AsyncMock()
        repos.generation_jobs = InMemoryGenerationJobs()
        return repos

    @pytest.fixture
    def mock_txn(self) -> MagicMock:
        """Create a mock transaction context."""
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        return txn

    @pytest.mark.asyncio
    async def test_basic_text_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Update text field returns updated_fields with text."""
        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text='New text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
            defer_generation=False,
        )
        assert 'text' in outcome.updated_fields
        assert outcome.summary_cleared is False
        mock_repos.context.update_context_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_patch_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Apply metadata_patch calls patch_metadata."""
        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text=None,
            metadata=None,
            metadata_patch={'key': 'value'},
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
            defer_generation=False,
        )
        assert 'metadata' in outcome.updated_fields
        mock_repos.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_tags_replacement(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Tags provided triggers replace_tags_for_context."""
        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text=None,
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=['new-tag'],
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
            defer_generation=False,
        )
        assert 'tags' in outcome.updated_fields
        mock_repos.tags.replace_tags_for_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_images_removal(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Empty images list removes all images and sets content_type to text."""
        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text=None,
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=[],
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
            defer_generation=False,
        )
        assert 'images' in outcome.updated_fields
        assert 'content_type' in outcome.updated_fields
        mock_repos.images.replace_images_for_context.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', [], txn=mock_txn, scope=LOCAL_SCOPE,
        )
        mock_repos.context.update_content_type.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', 'text', scope=LOCAL_SCOPE, txn=mock_txn,
        )

    @pytest.mark.asyncio
    async def test_embeddings_regeneration(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Chunk embeddings provided triggers delete+store cycle."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text='New text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=chunk_embeddings,
            embedding_model='m',
            defer_generation=False,
        )
        assert 'embedding' in outcome.updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', txn=mock_txn, scope=LOCAL_SCOPE,
        )
        mock_repos.embeddings.store_chunked.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_change_without_provider_clears_stale_embeddings(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Text changed, no new embeddings, tables exist -> stale chunks deleted.

        When an update changes text but no embedding provider regenerates vectors,
        the stored chunks describe the REPLACED text and must be DELETEd so semantic
        search cannot match the old content. Guarded by embedding_tables_exist.
        """
        mock_repos.images.count_images_for_context = AsyncMock(return_value=0)
        mock_repos.context.get_content_type = AsyncMock(return_value='text')
        mock_repos.context.update_content_type = AsyncMock()
        mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=True)
        mock_repos.embeddings.delete_all_chunks = AsyncMock(return_value=True)

        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text='Replaced text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
            defer_generation=False,
        )

        assert 'embedding' in outcome.updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', txn=mock_txn, scope=LOCAL_SCOPE,
        )
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_change_without_provider_tables_absent_is_noop(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Same text-change-without-provider path but embeddings were never
        provisioned -> safe no-op (no delete, 'embedding' not in updated_fields)."""
        mock_repos.images.count_images_for_context = AsyncMock(return_value=0)
        mock_repos.context.get_content_type = AsyncMock(return_value='text')
        mock_repos.context.update_content_type = AsyncMock()
        mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)
        mock_repos.embeddings.delete_all_chunks = AsyncMock(return_value=True)

        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text='Replaced text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
            defer_generation=False,
        )

        assert 'embedding' not in outcome.updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_raises_on_failed_entry_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises EntryNotFoundError when update_context_entry reports no such row."""
        mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                scope=LOCAL_SCOPE,
                text='New text',
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=None,
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
                defer_generation=False,
            )

    @pytest.mark.asyncio
    async def test_update_raises_on_failed_metadata_patch(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises EntryNotFoundError when patch_metadata reports no such row."""
        mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                scope=LOCAL_SCOPE,
                text=None,
                metadata=None,
                metadata_patch={'key': 'value'},
                summary=None,
                clear_summary=False,
                tags=None,
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
                defer_generation=False,
            )

    @pytest.mark.asyncio
    async def test_tags_only_update_missing_parent_raises_not_found(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A tags-only update against a vanished parent row raises EntryNotFoundError.

        The parent can disappear between the pre-generation existence check and this
        transaction (concurrent delete). Without the guard the tags write would fire a
        foreign-key insert against a missing parent, charging the circuit breaker.
        """
        mock_repos.context.entry_exists = AsyncMock(return_value=False)
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                scope=LOCAL_SCOPE,
                text=None,
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=['new-tag'],
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
                defer_generation=False,
            )
        mock_repos.tags.replace_tags_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_images_only_update_missing_parent_raises_not_found(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """An images-only update against a vanished parent row raises EntryNotFoundError.

        Same concurrent-delete race as the tags-only path; the guard stops the image
        write from touching a missing parent and charging the circuit breaker.
        """
        mock_repos.context.entry_exists = AsyncMock(return_value=False)
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                scope=LOCAL_SCOPE,
                text=None,
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=None,
                images=[],
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
                defer_generation=False,
            )
        mock_repos.images.replace_images_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_cleared_flag(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """clear_summary=True is returned as summary_cleared."""
        outcome = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            scope=LOCAL_SCOPE,
            text='Short',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=True,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
            defer_generation=False,
        )
        assert outcome.summary_cleared is True

    @pytest.mark.asyncio
    async def test_every_gate_receives_the_scope(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Each statement on the entry row runs as the caller, so each re-asserts its access."""
        bob = AccessScope('bob', frozenset({'team-x'}))
        context_id = '0190abcdef1234567890abcd00000001'
        common: dict[str, Any] = {
            'context_id': context_id, 'scope': bob, 'summary': None, 'clear_summary': False,
            'chunk_embeddings': None, 'embedding_model': 'm',
        }

        await execute_update_in_transaction(
            mock_repos, mock_txn, text='New text', metadata=None, metadata_patch={'k': 'v'}, visibility='public',
            tags=None, images=None, validated_images=[], **common,
            defer_generation=False,
        )
        await execute_update_in_transaction(
            mock_repos, mock_txn, text=None, metadata=None, metadata_patch=None,
            tags=['tag'], images=None, validated_images=[], **common,
            defer_generation=False,
        )
        await execute_update_in_transaction(
            mock_repos, mock_txn, text=None, metadata=None, metadata_patch=None,
            tags=None, images=[], validated_images=[], **common,
            defer_generation=False,
        )

        assert mock_repos.context.update_context_entry.await_args.kwargs['scope'] is bob
        mock_repos.context.patch_metadata.assert_awaited_once_with(
            context_id=context_id, patch={'k': 'v'}, scope=bob, txn=mock_txn,
        )
        mock_repos.context.entry_exists.assert_awaited_with(context_id, scope=bob, txn=mock_txn)
        mock_repos.context.get_content_type.assert_awaited_with(context_id, scope=bob, txn=mock_txn)
        mock_repos.context.touch_updated_at.assert_awaited_once_with(context_id, scope=bob, txn=mock_txn)
        mock_repos.context.update_content_type.assert_awaited_once_with(context_id, 'text', scope=bob, txn=mock_txn)

    @pytest.mark.asyncio
    async def test_unreadable_content_type_raises_not_found_without_writing(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """No content type for the caller means the row is gone or no longer writable: nothing is written.

        Comparing None with the recomputed type would report a difference and issue a
        content-type write, so the missing type ends the update as not found instead.
        """
        mock_repos.context.get_content_type = AsyncMock(return_value=None)
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                scope=LOCAL_SCOPE,
                text=None,
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=['new-tag'],
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
                defer_generation=False,
            )
        mock_repos.context.update_content_type.assert_not_called()
        mock_repos.context.touch_updated_at.assert_not_called()

    @pytest.mark.asyncio
    async def test_generated_outputs_are_written_through_the_shared_helper(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """The regenerated embeddings and node rows reach the store through write_generation_outputs."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        nodes = cast(list[IndexNodeRow], [MagicMock()])
        helper = AsyncMock(return_value=True)

        with patch('app.tools._update_transaction.write_generation_outputs', helper):
            outcome = await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id=CONTEXT_ID,
                scope=LOCAL_SCOPE,
                text='New text',
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=None,
                images=None,
                validated_images=[],
                chunk_embeddings=chunk_embeddings,
                embedding_model='m',
                index_nodes=nodes,
                defer_generation=False,
            )

        helper.assert_awaited_once_with(
            mock_repos, mock_txn, CONTEXT_ID, chunk_embeddings, nodes, embedding_model='m', scope=LOCAL_SCOPE,
        )
        assert 'embedding' in outcome.updated_fields
