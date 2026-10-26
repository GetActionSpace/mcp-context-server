"""Tests for execute_update_in_transaction and write_generation_outputs in app.tools._transactions."""

from typing import Any
from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.backends.base import MAINTENANCE
from app.errors import ControlFlowError
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools._transactions import EntryNotAuthorizedError
from app.tools._transactions import EntryNotFoundError
from app.tools._transactions import execute_update_in_transaction
from app.tools._transactions import write_generation_outputs
from tests.helpers import LOCAL_SCOPE


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
        updated_fields, summary_cleared = await execute_update_in_transaction(
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
        )
        assert 'text' in updated_fields
        assert summary_cleared is False
        mock_repos.context.update_context_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_patch_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Apply metadata_patch calls patch_metadata."""
        updated_fields, _ = await execute_update_in_transaction(
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
        )
        assert 'metadata' in updated_fields
        mock_repos.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_tags_replacement(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Tags provided triggers replace_tags_for_context."""
        updated_fields, _ = await execute_update_in_transaction(
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
        )
        assert 'tags' in updated_fields
        mock_repos.tags.replace_tags_for_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_images_removal(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Empty images list removes all images and sets content_type to text."""
        updated_fields, _ = await execute_update_in_transaction(
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
        )
        assert 'images' in updated_fields
        assert 'content_type' in updated_fields
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
        updated_fields, _ = await execute_update_in_transaction(
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
        )
        assert 'embedding' in updated_fields
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

        updated_fields, _ = await execute_update_in_transaction(
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
        )

        assert 'embedding' in updated_fields
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

        updated_fields, _ = await execute_update_in_transaction(
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
        )

        assert 'embedding' not in updated_fields
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
            )
        mock_repos.images.replace_images_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_cleared_flag(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """clear_summary=True is returned as summary_cleared."""
        _, summary_cleared = await execute_update_in_transaction(
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
        )
        assert summary_cleared is True

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
        )
        await execute_update_in_transaction(
            mock_repos, mock_txn, text=None, metadata=None, metadata_patch=None,
            tags=['tag'], images=None, validated_images=[], **common,
        )
        await execute_update_in_transaction(
            mock_repos, mock_txn, text=None, metadata=None, metadata_patch=None,
            tags=None, images=[], validated_images=[], **common,
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

        with patch('app.tools._transactions.write_generation_outputs', helper):
            updated_fields, _ = await execute_update_in_transaction(
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
            )

        helper.assert_awaited_once_with(
            mock_repos, mock_txn, CONTEXT_ID, chunk_embeddings, nodes, embedding_model='m', scope=LOCAL_SCOPE,
        )
        assert 'embedding' in updated_fields


CONTEXT_ID = '0190abcdef1234567890abcd00000001'


class TestWriteGenerationOutputs:
    """write_generation_outputs writes an entry's generated embeddings and node rows on the given transaction."""

    @pytest.fixture
    def mock_repos(self) -> MagicMock:
        """A repository container whose embedding and node writers record their calls."""
        repos = MagicMock()
        repos.embeddings.delete_all_chunks = AsyncMock(return_value=1)
        repos.embeddings.store_chunked = AsyncMock()
        repos.index_nodes.replace_nodes_for_context = AsyncMock()
        return repos

    @pytest.fixture
    def mock_txn(self) -> MagicMock:
        """A SQLite transaction context."""
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        return txn

    @pytest.mark.asyncio
    async def test_embeddings_replace_the_stored_chunks(self, mock_repos: MagicMock, mock_txn: MagicMock) -> None:
        """Generated embeddings delete the stored chunks first, then store the new ones under the model."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock(), MagicMock()])
        calls = MagicMock()
        calls.attach_mock(mock_repos.embeddings.delete_all_chunks, 'delete_all_chunks')
        calls.attach_mock(mock_repos.embeddings.store_chunked, 'store_chunked')

        written = await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, chunk_embeddings, None, embedding_model='model-a', scope=MAINTENANCE,
        )

        assert written is True
        assert calls.mock_calls == [
            call.delete_all_chunks(CONTEXT_ID, txn=mock_txn, scope=MAINTENANCE),
            call.store_chunked(
                context_id=CONTEXT_ID, chunk_embeddings=chunk_embeddings, model='model-a', txn=mock_txn, scope=MAINTENANCE,
            ),
        ]

    @pytest.mark.asyncio
    async def test_no_embeddings_leaves_the_stored_chunks(self, mock_repos: MagicMock, mock_txn: MagicMock) -> None:
        """Without generated embeddings no chunk is deleted or stored."""
        written = await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, None, None, embedding_model='model-a', scope=LOCAL_SCOPE,
        )

        assert written is False
        mock_repos.embeddings.delete_all_chunks.assert_not_called()
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('nodes', [[], [MagicMock()]], ids=['empty', 'one'])
    async def test_a_node_list_replaces_the_stored_nodes(
        self, mock_repos: MagicMock, mock_txn: MagicMock, nodes: list[IndexNodeRow],
    ) -> None:
        """A node list, empty included, replaces the entry's node rows."""
        await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, None, nodes, embedding_model='model-a', scope=LOCAL_SCOPE,
        )

        mock_repos.index_nodes.replace_nodes_for_context.assert_awaited_once_with(
            CONTEXT_ID, nodes, txn=mock_txn, scope=LOCAL_SCOPE,
        )

    @pytest.mark.asyncio
    async def test_no_node_list_keeps_the_stored_nodes(self, mock_repos: MagicMock, mock_txn: MagicMock) -> None:
        """None for the nodes leaves the entry's node rows as they are."""
        await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, cast(list[ChunkEmbedding], [MagicMock()]), None,
            embedding_model='model-a', scope=LOCAL_SCOPE,
        )

        mock_repos.index_nodes.replace_nodes_for_context.assert_not_called()


class TestEntryNotAuthorizedError:
    """The denial for a readable entry the caller may not modify or delete."""

    def test_modify_message_names_the_entry(self) -> None:
        """A modify denial reads exactly like the update tool reports it."""
        error = EntryNotAuthorizedError(['0190abcdef1234567890abcd00000001'], action='modify')
        assert str(error) == 'Not authorized to modify context entry with ID 0190abcdef1234567890abcd00000001'
        assert error.context_ids == ('0190abcdef1234567890abcd00000001',)
        assert error.action == 'modify'

    def test_delete_message_lists_the_entries_in_input_order(self) -> None:
        """A delete denial lists every refused entry in the order the caller named them."""
        error = EntryNotAuthorizedError(['b' * 32, 'a' * 32], action='delete')
        assert str(error) == f'Not authorized to delete context entries: {"b" * 32}, {"a" * 32}'
        assert error.context_ids == ('b' * 32, 'a' * 32)

    @pytest.mark.parametrize(
        ('action', 'message'),
        [
            ('share', 'Not authorized to share context entry with ID 0190abcdef1234567890abcd00000001'),
            ('unshare', 'Not authorized to unshare context entry with ID 0190abcdef1234567890abcd00000001'),
            ('list_grants', 'Not authorized to list the grants of context entry with ID 0190abcdef1234567890abcd00000001'),
        ],
    )
    def test_grant_messages_name_the_entry(
        self, action: Literal['share', 'unshare', 'list_grants'], message: str,
    ) -> None:
        """A refused grant change or listing reads exactly like the sharing tools report it."""
        error = EntryNotAuthorizedError(['0190abcdef1234567890abcd00000001'], action=action)
        assert str(error) == message
        assert error.action == action

    def test_is_breaker_exempt_control_flow(self) -> None:
        """A denial is client-input control flow, never a backend fault or a ToolError."""
        error = EntryNotAuthorizedError(['0190abcdef1234567890abcd00000001'], action='modify')
        assert isinstance(error, ControlFlowError)
        assert not isinstance(error, ToolError)
