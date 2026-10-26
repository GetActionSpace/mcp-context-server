"""Tests for the ``store_context`` tool: text and multimodal stores, input validation, database errors, and
unusual input (unicode, large metadata, SQL metacharacters).
"""

import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.access_scope import AccessScope
from app.backends.base import BackendAccess
from app.repositories.context_repository.records import DuplicateCandidate
from app.tools._transactions import StoreOutcome
from tests.helpers import as_principal
from tests.tools._read_only import read_only_caller

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
search_context = app.tools.search_context
get_context_by_ids = app.tools.get_context_by_ids
get_statistics = app.tools.get_statistics


@pytest.mark.usefixtures('initialized_server')
class TestStoreContext:
    """Test the store_context MCP tool."""

    @pytest.mark.asyncio
    async def test_store_text_context(
        self,
        sample_context_data: dict[str, Any],
    ) -> None:
        """Test storing a simple text context entry."""
        result = await store_context(
            thread_id=sample_context_data['thread_id'],
            source=sample_context_data['source'],
            text=sample_context_data['text'],
            metadata=sample_context_data['metadata'],
            tags=sample_context_data['tags'],
        )

        assert result['success'] is True
        assert 'context_id' in result
        assert result['thread_id'] == sample_context_data['thread_id']
        assert 'Context stored' in result['message']

    @pytest.mark.asyncio
    async def test_store_multimodal_context(
        self,
        sample_multimodal_data: dict[str, Any],
    ) -> None:
        """Test storing context with images."""
        result = await store_context(
            thread_id=sample_multimodal_data['thread_id'],
            source=sample_multimodal_data['source'],
            text=sample_multimodal_data['text'],
            images=sample_multimodal_data['images'],
            metadata=sample_multimodal_data['metadata'],
            tags=sample_multimodal_data['tags'],
        )

        assert result['success'] is True
        assert 'context_id' in result
        assert 'Context stored with 1 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_no_content(self) -> None:
        """Test that empty text is properly validated in the function body.

        We test validation in the function body, not Pydantic.
        """
        with pytest.raises(ToolError, match='text cannot be empty or whitespace'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='',  # Empty text should fail in function validation
            )

    @pytest.mark.asyncio
    async def test_store_context_invalid_source(self) -> None:
        """Test that invalid source bypasses Pydantic and hits database CHECK constraint.

        Note: Pydantic Literal['user', 'agent'] handles validation at FastMCP level.
        This test uses cast() to bypass Pydantic and verify database constraint works.
        """
        with pytest.raises(ToolError, match='CHECK constraint failed|source'):
            await store_context(
                thread_id='test_thread',
                source=cast(Literal['user', 'agent'], 'invalid_source'),
                text='Some text',
            )

    @pytest.mark.asyncio
    async def test_store_context_oversized_image(
        self,
        large_image_data: dict[str, str],
    ) -> None:
        """Test error when image exceeds size limit."""
        with pytest.raises(ToolError, match='exceeds.*MB limit'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='Text with large image',
                images=[large_image_data],
            )

    @pytest.mark.asyncio
    async def test_store_context_invalid_base64(self) -> None:
        """Test error with invalid base64 image data."""
        with pytest.raises(ToolError, match='Image 0 has invalid base64 encoding'):
            await store_context(
                thread_id='test_thread',
                source='agent',
                text='Text with bad image',
                images=[{'data': 'not-valid-base64!', 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    async def test_store_multiple_images(
        self,
        sample_image_data: dict[str, str],
    ) -> None:
        """Test storing multiple images."""
        import json as _json

        images = [
            sample_image_data,
            {**sample_image_data, 'metadata': _json.dumps({'position': 1})},
            {**sample_image_data, 'metadata': _json.dumps({'position': 2})},
        ]

        result = await store_context(
            thread_id='test_multi_image',
            source='agent',
            text='Multiple images attached',
            images=images,
        )

        assert result['success'] is True
        assert 'Context stored with 3 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_database_error(
        self,
        temp_db_path: Path,
    ) -> None:
        """Test handling of database errors."""
        # Mock the repository method to raise an error
        _ = temp_db_path  # Acknowledge unused parameter
        with patch('app.repositories.context_repository.ContextRepository.store_with_deduplication') as mock_store:
            mock_store.side_effect = sqlite3.OperationalError('Database error')
            with pytest.raises(ToolError, match='Failed to store context'):
                await store_context(
                    thread_id='test',
                    source='user',
                    text='This should fail',
                )


@pytest.mark.usefixtures('initialized_server')
class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_unicode_content(self) -> None:
        """Test handling of Unicode content."""
        unicode_text = 'Hello 世界 🌍 مرحبا мир'
        result = await store_context(
            thread_id='unicode_test',
            source='user',
            text=unicode_text,
            tags=['unicode', '中文', 'عربي'],
        )

        assert result['success'] is True

        # Verify retrieval
        search_result = await search_context(limit=50, thread_id='unicode_test')
        assert isinstance(search_result, dict)
        assert search_result['results'][0]['text_content'] == unicode_text

    @pytest.mark.asyncio
    async def test_large_metadata(self) -> None:
        """Test handling of large metadata objects."""
        from app.types import JsonValue

        large_metadata = cast(
            'dict[str, JsonValue]',
            {
                'nested': {
                    'level': {
                        'data': ['item'] * 100,
                        'numbers': list(range(1000)),
                    },
                },
                'description': 'x' * 10000,
            },
        )

        result = await store_context(
            thread_id='metadata_test',
            source='agent',
            text='Large metadata',
            metadata=large_metadata,
        )

        assert result['success'] is True

        # Verify retrieval
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(fetched[0])
        assert entry['metadata'] == large_metadata

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self) -> None:
        """Test that SQL injection attempts are prevented."""
        malicious_thread = "'; DROP TABLE context_entries; --"
        malicious_tag = "'; DELETE FROM tags; --"

        # Should handle malicious input safely
        result = await store_context(
            thread_id=malicious_thread,
            source='user',
            text='Test SQL injection',
            tags=[malicious_tag],
        )

        assert result['success'] is True

        # Verify data integrity
        search_result = await search_context(limit=50, thread_id=malicious_thread)
        assert isinstance(search_result, dict)
        assert len(search_result['results']) == 1
        # Tag should be normalized to lowercase
        normalized_tag = malicious_tag.strip().lower()
        assert normalized_tag in search_result['results'][0]['tags']

        # Tables should still exist
        stats = await get_statistics()
        assert stats['total_entries'] > 0


@pytest.mark.usefixtures('initialized_server')
class TestStoreScope:
    """store_context runs its pre-check and its store under the caller's scope."""

    @pytest.mark.asyncio
    async def test_scope_reaches_the_precheck_and_the_store(self) -> None:
        """The scope of the request principal, groups included, reaches both repository calls."""
        mock_repos = AsyncMock()
        mock_backend = MagicMock()
        mock_backend.begin_transaction = _mock_begin_transaction
        mock_repos.context.backend = mock_backend
        precheck = AsyncMock(return_value=DuplicateCandidate(context_id='dup-id', summary='Stored summary'))
        mock_repos.context.check_latest_is_duplicate = precheck

        with (
            as_principal('bob', groups=['team-x']),
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.store.execute_store_in_transaction',
                new_callable=AsyncMock,
                return_value=StoreOutcome('dup-id', True, False, 'ready'),
            ) as mock_store,
        ):
            result = await store_context(thread_id='scope-thread', source='user', text='Re-sent text')

        expected = AccessScope('bob', frozenset({'team-x'}), may_publish=True, may_write=True)
        precheck_call = precheck.await_args
        store_call = mock_store.await_args
        assert result['context_id'] == 'dup-id'
        assert precheck_call is not None
        assert precheck_call.kwargs['scope'] == expected
        assert store_call is not None
        assert store_call.kwargs['scope'] == expected
        assert store_call.kwargs['deduplicate'] is True


class TestStoreWithoutDeduplication:
    """store_context with deduplicate=False inserts a fresh entry and skips every deduplication read."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_skips_the_precheck_and_runs_every_generation_leg(self) -> None:
        """The pre-check and the embedding-existence probe are never awaited; every leg runs; False is forwarded."""
        mock_repos = AsyncMock()
        mock_backend = MagicMock()
        mock_backend.begin_transaction = _mock_begin_transaction
        mock_repos.context.backend = mock_backend
        precheck = AsyncMock(return_value=DuplicateCandidate(context_id='dup-id', summary='Stored summary'))
        mock_repos.context.check_latest_is_duplicate = precheck
        mock_repos.embeddings.exists = AsyncMock(return_value=True)
        text = 'Corpus document body. ' * 30

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.store.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.store.run_generation',
                new_callable=AsyncMock,
                return_value=([('chunk-0', [0.1, 0.2])], 'Fresh summary', []),
            ) as mock_generation,
            patch(
                'app.tools.context.store.execute_store_in_transaction',
                new_callable=AsyncMock,
                return_value=StoreOutcome('new-id', False, True, 'ready'),
            ) as mock_store,
        ):
            result = await store_context(thread_id='kb:handbook', source='user', text=text, deduplicate=False)

        assert result['context_id'] == 'new-id'
        precheck.assert_not_awaited()
        mock_repos.embeddings.exists.assert_not_awaited()
        generation_call = mock_generation.await_args
        assert generation_call is not None
        assert generation_call.kwargs == {'run_embedding': True, 'run_summary': True, 'run_nodes': True}
        store_call = mock_store.await_args
        assert store_call is not None
        assert store_call.kwargs['deduplicate'] is False
        assert store_call.kwargs['summary'] == 'Fresh summary'
        assert store_call.kwargs['nodes_pending'] is False
        assert store_call.kwargs['summary_pending'] is False

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_identical_text_twice_gives_two_entries(self) -> None:
        """Two identical stores without deduplication insert two entries; a default store then merges into the latest."""
        first = await store_context(thread_id='kb:twins', source='user', text='Shared body', deduplicate=False)
        second = await store_context(thread_id='kb:twins', source='user', text='Shared body', deduplicate=False)
        merged = await store_context(thread_id='kb:twins', source='user', text='Shared body')

        assert first['context_id'] != second['context_id']
        assert merged['context_id'] == second['context_id']
        listed = await search_context(limit=50, thread_id='kb:twins')
        assert isinstance(listed, dict)
        assert sorted(row['id'] for row in listed['results']) == sorted([first['context_id'], second['context_id']])


_INGEST_BLOCK: dict[str, Any] = {
    'scope': 'kb:handbook',
    'source_id': 'guides/setup.md',
    'meta_digest': 'ab' * 32,
    'synced_at': '2026-10-18T12:00:00Z',
}


class TestStoreIngestBlock:
    """store_context validates the reserved ingest block before any generation or database work."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('block', 'message'),
        [
            (
                {'scope': 'kb:handbook'},
                'metadata.ingest must be an object with exactly the keys meta_digest, scope, source_id, synced_at',
            ),
            ({**_INGEST_BLOCK, 'source_id': ''}, 'metadata.ingest.source_id cannot be empty'),
            (
                {**_INGEST_BLOCK, 'meta_digest': 'AB' * 32},
                'metadata.ingest.meta_digest must be 64 lowercase hexadecimal characters',
            ),
            (
                {**_INGEST_BLOCK, 'synced_at': '2026-10-18'},
                'metadata.ingest.synced_at must be a UTC timestamp such as 2026-10-18T12:00:00Z',
            ),
        ],
        ids=['partial', 'empty-source-id', 'uppercase-digest', 'date-only'],
    )
    async def test_malformed_block_refused_before_generation(self, block: dict[str, Any], message: str) -> None:
        """A malformed block raises its fixed message; no provider and no repository is reached."""
        embedding_provider = AsyncMock()
        summary_provider = AsyncMock()
        with (
            patch('app.tools.context.store.ensure_repositories', new_callable=AsyncMock) as ensure,
            patch('app.tools.context.store.get_embedding_provider', return_value=embedding_provider),
            patch('app.tools.context.store.get_summary_provider', return_value=summary_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=embedding_provider),
            pytest.raises(ToolError) as exc_info,
        ):
            await store_context(
                thread_id='kb:handbook', source='user', text='Setup guide', metadata={'title': 'Setup', 'ingest': block},
            )

        assert str(exc_info.value) == message
        ensure.assert_not_awaited()
        assert embedding_provider.mock_calls == []
        assert summary_provider.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_valid_block_is_stored_as_sent(self) -> None:
        """A complete valid block passes the boundary and is stored unchanged."""
        metadata: dict[str, Any] = {'title': 'Setup', 'ingest': _INGEST_BLOCK}
        result = await store_context(thread_id='kb:handbook', source='user', text='Setup guide', metadata=metadata)

        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        assert dict(fetched[0])['metadata'] == metadata


@asynccontextmanager
async def _mock_begin_transaction(*, access_scope: BackendAccess) -> AsyncGenerator[MagicMock, None]:
    """Yield a mock transaction for a mocked backend."""
    _ = access_scope
    yield MagicMock()


class TestStoreReadOnlyPrincipal:
    """The read-only principal is refused before any argument check, repository access or generation."""

    @pytest.mark.asyncio
    async def test_read_only_principal_is_refused_first(self) -> None:
        """Blank arguments that would fail validation still get the read-only refusal, and nothing runs."""
        with (
            read_only_caller(),
            patch('app.tools.context.store.ensure_repositories', new=AsyncMock()) as ensure,
            patch('app.tools.context.store.get_embedding_provider') as embedding_provider,
            patch('app.tools.context.store.get_summary_provider') as summary_provider,
            pytest.raises(ToolError) as exc_info,
        ):
            await store_context(thread_id='   ', source='agent', text='   ', visibility='public')

        assert str(exc_info.value) == 'The anonymous principal is read-only: store_context is not permitted'
        ensure.assert_not_awaited()
        embedding_provider.assert_not_called()
        summary_provider.assert_not_called()
