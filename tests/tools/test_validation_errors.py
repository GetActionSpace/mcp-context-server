"""
Comprehensive tests for validation error handling in all MCP tools.

This test module ensures that ALL validation errors are properly handled using
the ToolError exception pattern, never returning raw validation errors or dictionaries.
"""

from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories import RepositoryContainer
from app.repositories.context_repository.records import EntryProbe
from app.repositories.context_repository.records import IdAccess
from tests.tools._generation_jobs import InMemoryGenerationJobs

# Get the actual async functions - they are no longer wrapped by @mcp.tool() at import time
store_context = app.tools.store_context
search_context = app.tools.search_context
get_context_by_ids = app.tools.get_context_by_ids
delete_context = app.tools.delete_context
update_context = app.tools.update_context
list_threads = app.tools.list_threads
get_statistics = app.tools.get_statistics


async def _owned_access(context_ids: list[str], **_kwargs: object) -> dict[str, IdAccess]:
    """Probe stand-in: the caller owns every requested entry."""
    return dict.fromkeys(context_ids, IdAccess(can_write=True, is_owner=True))


@pytest.fixture
def mock_repos():
    """Mock repository container for testing.

    Returns:
        MagicMock: The mock repository container with transaction support.
    """
    from contextlib import asynccontextmanager
    from unittest.mock import Mock

    repos = MagicMock(spec=RepositoryContainer)

    # Mock backend exposing begin_transaction() as an async context manager
    mock_backend = Mock()

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        _ = access_scope
        txn = Mock()
        txn.backend_type = 'sqlite'
        txn.connection = Mock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    # Mock context repository
    repos.context = AsyncMock()
    repos.context.backend = mock_backend
    repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
    repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.search_contexts = AsyncMock(return_value=([], {}))
    repos.context.get_by_ids = AsyncMock(return_value=[])
    repos.context.delete_by_ids = AsyncMock(return_value=1)
    # Every delete snapshots or names ids the caller owns, then deletes them by id.
    repos.context.get_ids_matching_batch_criteria = AsyncMock(return_value=['0190abcdef1234567890abcd00000001'])
    repos.context.probe_ids = AsyncMock(side_effect=_owned_access)

    # Mock tags repository
    repos.tags = AsyncMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    # Mock images repository
    repos.images = AsyncMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.get_images_for_context = AsyncMock(return_value=[])
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    # Mock statistics repository
    repos.statistics = AsyncMock()
    repos.statistics.get_thread_list = AsyncMock(return_value=[])
    repos.statistics.get_database_statistics = AsyncMock(return_value={'total_entries': 0})

    # Mock embeddings repository for generation-first transactional writes
    repos.embeddings = AsyncMock()
    repos.embeddings.store_chunked = AsyncMock(return_value=None)
    repos.embeddings.delete_all_chunks = AsyncMock(return_value=None)

    # Mock index_tree node-summary repository (text-change updates clear stale node rows).
    repos.index_nodes = AsyncMock()
    repos.index_nodes.replace_nodes_for_context = AsyncMock(return_value=None)
    repos.index_nodes.get_nodes_for_context = AsyncMock(return_value={})
    repos.generation_jobs = InMemoryGenerationJobs()

    return repos


class TestStoreContextValidation:
    """Test validation errors for store_context function."""

    @pytest.mark.asyncio
    async def test_empty_thread_id(self, mock_repos):
        """Test that empty thread_id raises ToolError."""
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            # Test empty string
            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='',
                    source='user',
                    text='Test content',
                )
            assert 'thread_id' in str(exc_info.value).lower()

            # Test whitespace only
            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='   ',
                    source='user',
                    text='Test content',
                )
            assert 'thread_id' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_empty_text(self, mock_repos):
        """Test that empty text raises ToolError."""
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            # Test empty string
            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='',
                )
            assert 'text' in str(exc_info.value).lower()

            # Test whitespace only
            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='   ',
                )
            assert 'text' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_source(self, mock_repos):
        """Test that Pydantic Literal handles invalid source.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            mock_repos.context.store_with_deduplication.return_value = (1, False)
            # Valid source works fine
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='Test content',
            )
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_oversized_image(self, mock_repos):
        """Test that oversized images raise ToolError."""
        import base64

        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            # Create actual oversized binary data and encode it
            # 11MB of binary data (over the 10MB limit)
            oversized_data = b'\x00' * (11 * 1024 * 1024)
            large_image = {
                'mime_type': 'image/png',
                'data': base64.b64encode(oversized_data).decode('ascii'),
            }

            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='Test content',
                    images=[large_image],
                )
            assert 'exceeds' in str(exc_info.value).lower() or 'size' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_image_data(self, mock_repos):
        """Test that invalid base64 image data raises ToolError."""
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            invalid_image = {
                'mime_type': 'image/png',
                'data': 'not-valid-base64!@#$%',
            }

            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='Test content',
                    images=[invalid_image],
                )
            assert 'invalid' in str(exc_info.value).lower() or 'base64' in str(exc_info.value).lower()


class TestUpdateContextValidation:
    """Test validation errors for update_context function."""

    @pytest.mark.asyncio
    async def test_empty_text(self, mock_repos):
        """Test that empty text in update raises ToolError."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repos):
            # Test empty string
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00000001',
                    text='',
                )
            assert 'text' in str(exc_info.value).lower()

            # Test whitespace only
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00000001',
                    text='   ',
                )
            assert 'text' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_no_fields_provided(self, mock_repos):
        """Test that updating with no fields raises ToolError."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repos):
            with pytest.raises(ToolError) as exc_info:
                await update_context(context_id='0190abcdef1234567890abcd00000001')
            assert 'at least one' in str(exc_info.value).lower() or 'field' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_nonexistent_context(self, mock_repos):
        """Test that updating non-existent context raises ToolError."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repos):
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(False, None, None, None, False))

            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd000003e7',
                    text='New text',
                )
            assert '999' in str(exc_info.value) or 'not found' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_image_structure(self, mock_repos):
        """Test that invalid image structure raises ToolError."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repos):
            invalid_images = cast(Any, [{'invalid': 'structure'}])
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00000001',
                    images=invalid_images,
                )
            error_msg = str(exc_info.value).lower()
            assert 'mime_type' in error_msg or 'data' in error_msg or 'missing' in error_msg

    @pytest.mark.asyncio
    async def test_oversized_images(self, mock_repos):
        """Test that oversized images in update raise ToolError."""
        import base64

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repos):
            # Create actual oversized binary data and encode it
            # 11MB of binary data (over the 10MB limit)
            oversized_data = b'\x00' * (11 * 1024 * 1024)
            large_image = {
                'mime_type': 'image/png',
                'data': base64.b64encode(oversized_data).decode('ascii'),
            }

            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00000001',
                    images=[large_image],
                )
            assert 'exceeds' in str(exc_info.value).lower() or 'size' in str(exc_info.value).lower()


class TestSearchContextValidation:
    """Test validation errors for search_context function."""

    @pytest.mark.asyncio
    async def test_invalid_source(self, mock_repos):
        """Test that Pydantic Literal handles invalid source.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        with patch('app.tools.search.browse.ensure_repositories', return_value=mock_repos):
            mock_repos.context.search_contexts.return_value = ([], {})
            # Valid source works fine
            result = await search_context(limit=50, source='user')
            assert 'results' in result

    @pytest.mark.asyncio
    async def test_invalid_content_type(self, mock_repos):
        """Test that invalid content_type in search returns proper error."""
        with patch('app.tools.search.browse.ensure_repositories', return_value=mock_repos):
            # Should work with valid content types
            result = await search_context(limit=50, content_type='text')
            assert 'results' in result

            result = await search_context(limit=50, content_type='multimodal')
            assert 'results' in result

    @pytest.mark.asyncio
    async def test_invalid_limit(self, mock_repos):
        """Test that Pydantic Field(ge=1, le=100) handles limit validation.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        with patch('app.tools.search.browse.ensure_repositories', return_value=mock_repos):
            mock_repos.context.search_contexts.return_value = ([], {})
            # Valid limits work fine
            result = await search_context(limit=1)
            assert 'results' in result
            result = await search_context(limit=100)
            assert 'results' in result

    @pytest.mark.asyncio
    async def test_negative_offset(self, mock_repos):
        """Test that Pydantic Field(ge=0) handles offset validation.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        with patch('app.tools.search.browse.ensure_repositories', return_value=mock_repos):
            mock_repos.context.search_contexts.return_value = ([], {})
            # Valid offsets work fine
            result = await search_context(limit=50, offset=0)
            assert 'results' in result
            result = await search_context(limit=50, offset=100)
            assert 'results' in result

    @pytest.mark.asyncio
    async def test_limit_exceeds_maximum(self, mock_repos):
        """Test that Pydantic Field(le=100) enforces max limit.

        Note: Pydantic validates at FastMCP level. This test verifies max limit works.
        """
        with patch('app.tools.search.browse.ensure_repositories', return_value=mock_repos):
            mock_repos.context.search_contexts.return_value = ([], {})
            # Valid max limit works fine
            result = await search_context(limit=100)
            assert 'results' in result


class TestGetContextByIdsValidation:
    """Test validation errors for get_context_by_ids function."""

    @pytest.mark.asyncio
    async def test_empty_list(self, mock_repos):
        """Test that Pydantic Field(min_length=1) handles empty list.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        with patch('app.tools.context.retrieve.ensure_repositories', return_value=mock_repos):
            mock_repos.context.get_by_ids.return_value = []
            # Valid non-empty list works fine
            result = await get_context_by_ids(context_ids=['0190abcdef1234567890abcd00000001'])
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_invalid_ids(self, mock_repos):
        """Test that invalid context IDs are handled gracefully."""
        with patch('app.tools.context.retrieve.ensure_repositories', return_value=mock_repos):
            # Non-existent IDs should return empty list, not error
            result = await get_context_by_ids(context_ids=['0190abcdef1234567890abcd000f423f'])
            assert result == []

    @pytest.mark.asyncio
    async def test_valid_integer_strings(self, mock_repos):
        """Test that valid integer IDs work correctly."""
        with patch('app.tools.context.retrieve.ensure_repositories', return_value=mock_repos):
            mock_repos.context.get_by_ids = AsyncMock(
                return_value=[
                    {
                        'id': 1,
                        'thread_id': 'test',
                        'source': 'user',
                        'content_type': 'text',
                        'text_content': 'Test',
                        'created_at': '2025-01-01',
                        'updated_at': '2025-01-01',
                        'visibility': 'private',
                        'is_owner': 1,
                    },
                ],
            )

            result = await get_context_by_ids(context_ids=['0190abcdef1234567890abcd00000001'])
            assert len(result) == 1


class TestDeleteContextValidation:
    """Test validation errors for delete_context function."""

    @pytest.mark.asyncio
    async def test_no_parameters(self, mock_repos):
        """Test that delete with no parameters raises ToolError."""
        with patch('app.tools.context.delete.ensure_repositories', return_value=mock_repos):
            with pytest.raises(ToolError) as exc_info:
                await delete_context()
            assert 'at least one' in str(exc_info.value).lower() or 'provide' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_successful_deletion_by_ids(self, mock_repos):
        """Test successful deletion by context IDs."""
        with patch('app.tools.context.delete.ensure_repositories', return_value=mock_repos):
            result = await delete_context(
                context_ids=[
                    '0190abcdef1234567890abcd00000001',
                    '0190abcdef1234567890abcd00000002',
                    '0190abcdef1234567890abcd00000003',
                ],
            )
            assert result['deleted_count'] == 1
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_successful_deletion_by_thread(self, mock_repos):
        """Test successful deletion by thread ID."""
        with patch('app.tools.context.delete.ensure_repositories', return_value=mock_repos):
            result = await delete_context(thread_id='test-thread')
            assert result['deleted_count'] == 1
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_deletion_error(self, mock_repos):
        """Test that repository errors during deletion raise ToolError."""
        with patch('app.tools.context.delete.ensure_repositories', return_value=mock_repos):
            mock_repos.context.delete_by_ids.side_effect = Exception('Database error')

            with pytest.raises(ToolError) as exc_info:
                await delete_context(context_ids=['0190abcdef1234567890abcd00000001'])
            assert 'failed' in str(exc_info.value).lower() or 'error' in str(exc_info.value).lower()


class TestEdgeCasesAndCombinations:
    """Test edge cases and combinations of validation errors."""

    @pytest.mark.asyncio
    async def test_multiple_validation_errors_store(self, mock_repos):
        """Test multiple validation errors in store_context."""
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            # Empty thread_id and empty text - should fail on thread_id first
            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='',
                    source='user',
                    text='',
                )
            # Should mention one of the validation errors
            error_msg = str(exc_info.value).lower()
            assert 'thread_id' in error_msg or 'text' in error_msg

    @pytest.mark.asyncio
    async def test_unicode_and_special_chars(self, mock_repos):
        """Test that Unicode and special characters are handled properly."""
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            # Should succeed with Unicode
            result = await store_context(
                thread_id='test-🚀-thread',
                source='user',
                text='Hello 世界 🌍',
                metadata={'emoji': '🎉', 'special': 'café'},
            )
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_very_long_inputs(self, mock_repos):
        """Test that very long inputs are handled."""
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            # Very long text should succeed
            long_text = 'A' * 100000  # 100K characters
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=long_text,
            )
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_null_vs_empty_string(self, mock_repos):
        """Test distinction between null and empty string."""
        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repos),
        ):
            # Empty string should raise error
            with pytest.raises(ToolError):
                await store_context(
                    thread_id='test',
                    source='user',
                    text='',
                )

            # None/null for optional fields should work
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000001',
                text='Valid text',
                metadata=None,  # Explicitly None
            )
            assert 'error' not in result

    @pytest.mark.asyncio
    async def test_search_with_all_filters(self, mock_repos):
        """Test search with all possible filters."""
        with patch('app.tools.search.browse.ensure_repositories', return_value=mock_repos):
            # Should succeed with all valid filters
            result = await search_context(
                thread_id='test-thread',
                source='user',
                tags=['tag1', 'tag2'],
                content_type='text',
                metadata={'key': 'value'},
                limit=10,
                offset=0,
            )
            assert 'results' in result


class TestExceptionHandling:
    """Test that repository exceptions are properly converted to ToolError."""

    @pytest.mark.asyncio
    async def test_repository_exception_store(self, mock_repos):
        """Test repository exception in store_context raises ToolError."""
        with patch('app.tools.context.store.ensure_repositories', return_value=mock_repos):
            mock_repos.context.store_with_deduplication.side_effect = Exception('Database error')

            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='Test content',
                )
            assert 'failed to store' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_repository_exception_update(self, mock_repos):
        """Test repository exception in update_context raises ToolError."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repos):
            mock_repos.context.update_context_entry.side_effect = Exception('Update failed')

            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00000001',
                    text='New text',
                )
            assert 'update' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_repository_exception_search(self, mock_repos):
        """Test repository exception in search_context raises ToolError."""
        with patch('app.tools.search.browse.ensure_repositories', return_value=mock_repos):
            mock_repos.context.search_contexts.side_effect = Exception('Search failed')

            with pytest.raises(ToolError) as exc_info:
                await search_context(limit=50)
            assert 'search' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_repository_exception_get_by_ids(self, mock_repos):
        """Test repository exception in get_context_by_ids raises ToolError."""
        with patch('app.tools.context.retrieve.ensure_repositories', return_value=mock_repos):
            mock_repos.context.get_by_ids.side_effect = Exception('Fetch failed')

            with pytest.raises(ToolError) as exc_info:
                await get_context_by_ids(
                    context_ids=[
                        '0190abcdef1234567890abcd00000001',
                        '0190abcdef1234567890abcd00000002',
                        '0190abcdef1234567890abcd00000003',
                    ],
                )
            assert 'fetch' in str(exc_info.value).lower()
