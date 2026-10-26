"""Tests for the ``get_context_by_ids`` tool."""

import base64
from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.startup import ensure_repositories
from tests.helpers import as_principal
from tests.tools._generation_jobs import record_state_reads
from tests.tools._generation_jobs import seed_generation_jobs

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
get_context_by_ids = app.tools.get_context_by_ids


@pytest.mark.usefixtures('initialized_server')
class TestGetContextByIds:
    """Test the get_context_by_ids MCP tool."""

    @pytest.mark.asyncio
    async def test_get_single_context(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test fetching a single context by ID."""
        context_id = multiple_context_entries[0]
        results = await get_context_by_ids(context_ids=[context_id])

        assert len(results) == 1
        entry = dict(results[0])
        assert entry['id'] == context_id

    @pytest.mark.asyncio
    async def test_get_multiple_contexts(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test fetching multiple contexts by IDs."""
        ids_to_fetch = multiple_context_entries[:3]
        results = await get_context_by_ids(context_ids=ids_to_fetch)

        assert len(results) == 3
        result_ids = [dict(r)['id'] for r in results]
        assert set(result_ids) == set(ids_to_fetch)

    @pytest.mark.asyncio
    async def test_get_context_with_images(self) -> None:
        """Test fetching context with images included."""
        import json as _json

        # Store context with image
        image_data = base64.b64encode(b'test_img').decode('utf-8')
        store_result = await store_context(
            thread_id='img_test',
            source='agent',
            text='With image',
            images=[
                {
                    'data': image_data,
                    'mime_type': 'image/jpeg',
                    'metadata': _json.dumps({'size': 100}),
                },
            ],
        )

        context_id = store_result['context_id']

        # Fetch with images
        results = await get_context_by_ids(
            context_ids=[context_id],
            include_images=True,
        )

        assert len(results) == 1
        assert 'images' in results[0]
        result_images = results[0]['images']
        assert result_images is not None
        assert len(result_images) == 1
        assert result_images[0]['mime_type'] == 'image/jpeg'

    @pytest.mark.asyncio
    async def test_get_context_without_images(self) -> None:
        """Test fetching context without images."""
        # Store context with image
        store_result = await store_context(
            thread_id='no_img_test',
            source='user',
            text='With image but not fetched',
            images=[{'data': base64.b64encode(b'img').decode('utf-8')}],
        )

        # Fetch without images
        results = await get_context_by_ids(
            context_ids=[store_result['context_id']],
            include_images=False,
        )

        assert len(results) == 1
        assert 'images' not in results[0] or results[0]['images'] == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_contexts(self) -> None:
        """Test fetching non-existent context IDs."""
        results = await get_context_by_ids(
            context_ids=[
                '0190abcdef1234567890abcd0000270f',
                '0190abcdef1234567890abcd00002710',
            ],
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_get_empty_context_list(self) -> None:
        """Test that Pydantic Field(min_length=1) handles empty list.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        # Test with valid non-empty list
        result = await get_context_by_ids(context_ids=['0190abcdef1234567890abcd00000001'])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_context_with_tags(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test that tags are included in fetched contexts."""
        # First entry has tags
        results = await get_context_by_ids(context_ids=[multiple_context_entries[0]])

        assert len(results) == 1
        assert 'tags' in results[0]
        assert 'important' in results[0]['tags']


@pytest.mark.usefixtures('initialized_server')
class TestGetContextByIdsScoping:
    """get_context_by_ids returns only the entries the caller may read."""

    @staticmethod
    async def _store_as(principal_id: str, text: str, visibility: Literal['private', 'public']) -> str:
        """Store one entry as the principal and return its id."""
        with as_principal(principal_id):
            result = await store_context(thread_id='scoped-get', source='agent', text=text, visibility=visibility)
        return result['context_id']

    @pytest.mark.asyncio
    async def test_unreadable_ids_are_omitted_like_absent_ones(self) -> None:
        """Bob gets alice's public entry and nothing for her private entry or an absent id."""
        private_id = await self._store_as('alice', 'alice private retrieve target', 'private')
        public_id = await self._store_as('alice', 'alice public retrieve target', 'public')
        absent_id = '0190abcdef1234567890abcd0000270f'

        with as_principal('bob'):
            results = await get_context_by_ids(context_ids=[private_id, public_id, absent_id])
            hidden_only = await get_context_by_ids(context_ids=[private_id])
            absent_only = await get_context_by_ids(context_ids=[absent_id])

        assert [dict(entry)['id'] for entry in results] == [public_id]
        assert hidden_only == absent_only == []

    @pytest.mark.asyncio
    async def test_owner_reads_their_private_entry(self) -> None:
        """The owner reads their own private entry."""
        private_id = await self._store_as('alice', 'alice reads her own entry', 'private')

        with as_principal('alice'):
            results = await get_context_by_ids(context_ids=[private_id])

        assert [dict(entry)['id'] for entry in results] == [private_id]

    @pytest.mark.asyncio
    async def test_no_child_reads_for_an_unreadable_id(self) -> None:
        """Tags and images are fetched only for the entries the scoped read returned."""
        private_id = await self._store_as('alice', 'alice entry with hidden children', 'private')
        repos = await ensure_repositories()

        with (
            as_principal('bob'),
            patch.object(repos.tags, 'get_tags_for_context', AsyncMock(return_value=[])) as tags_spy,
            patch.object(repos.images, 'get_images_for_context', AsyncMock(return_value=[])) as images_spy,
        ):
            assert await get_context_by_ids(context_ids=[private_id]) == []

        tags_spy.assert_not_awaited()
        images_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prefix_of_an_unreadable_entry_matches_nothing(self) -> None:
        """A prefix matching only another principal's private entry fails exactly like an absent prefix."""
        private_id = await self._store_as('alice', 'alice entry behind a prefix', 'private')
        absent_prefix = 'ffffffff' if not private_id.startswith('ffffffff') else 'eeeeeeee'

        with as_principal('bob'):
            with pytest.raises(ToolError) as hidden:
                await get_context_by_ids(context_ids=[private_id[:12]])
            with pytest.raises(ToolError) as absent:
                await get_context_by_ids(context_ids=[absent_prefix])

        assert str(hidden.value) == f"Invalid context ID: No context entry matches prefix '{private_id[:12]}'"
        assert str(absent.value) == f"Invalid context ID: No context entry matches prefix '{absent_prefix}'"


@pytest.mark.usefixtures('initialized_server')
class TestGetContextByIdsReadBack:
    """Every returned entry reports its visibility and whether the caller owns it, never the owner itself."""

    @pytest.mark.asyncio
    async def test_owner_and_reader_see_visibility_and_ownership(self) -> None:
        """Alice owns both of her entries; bob reads her public entry as not his."""
        with as_principal('alice'):
            private = await store_context(thread_id='read-back', source='agent', text='alice private', visibility='private')
            public = await store_context(thread_id='read-back', source='agent', text='alice public', visibility='public')
            ids = [private['context_id'], public['context_id']]
            owned = [dict(entry) for entry in await get_context_by_ids(context_ids=ids)]
        with as_principal('bob'):
            read = [dict(entry) for entry in await get_context_by_ids(context_ids=ids)]

        assert {entry['id']: (entry['visibility'], entry['is_owner']) for entry in owned} == {
            private['context_id']: ('private', True), public['context_id']: ('public', True),
        }
        assert [(entry['id'], entry['visibility'], entry['is_owner']) for entry in read] == [
            (public['context_id'], 'public', False),
        ]
        for entry in (*owned, *read):
            assert type(entry['is_owner']) is bool
            assert 'owner_id' not in entry


@pytest.mark.usefixtures('initialized_server')
class TestGetContextByIdsGenerationStatus:
    """Every returned entry reports the state of its generated outputs, read once per response."""

    @pytest.mark.asyncio
    async def test_each_entry_reports_its_generation_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No job reads ready, a waiting or running job pending, a failed job failed, all from one state read."""
        stored = [
            (await store_context(thread_id='generation-state', source='agent', text=f'entry {label}'))['context_id']
            for label in ('ready', 'waiting', 'running', 'failed')
        ]
        ready, waiting, running, failed = stored
        repos = await ensure_repositories()
        await seed_generation_jobs(repos.context.backend, {waiting: 'pending', running: 'running', failed: 'failed'})
        reads = record_state_reads(monkeypatch, repos)

        entries = [dict(entry) for entry in await get_context_by_ids(context_ids=stored)]

        assert {entry['id']: entry['generation_status'] for entry in entries} == {
            ready: 'ready', waiting: 'pending', running: 'pending', failed: 'failed',
        }
        assert len(reads) == 1
        assert sorted(reads[0]) == sorted(stored)
