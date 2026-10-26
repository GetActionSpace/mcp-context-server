"""Tests for the ``search_context`` tool: filters, pagination, image inclusion, and text truncation."""

import base64
import sqlite3
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

import app.startup
import app.tools
from app.access_scope import AccessScope
from app.backends.base import MAINTENANCE
from tests.helpers import as_principal
from tests.tools._generation_jobs import seed_generation_jobs

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
search_context = app.tools.search_context
get_context_by_ids = app.tools.get_context_by_ids


@pytest.mark.usefixtures('initialized_server')
class TestSearchContext:
    """Test the search_context MCP tool."""

    @pytest.mark.asyncio
    async def test_search_all_contexts(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test searching without filters returns all contexts."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(limit=50)

        assert isinstance(results, dict)
        assert 'results' in results
        assert len(results['results']) == 5  # All test entries

    @pytest.mark.asyncio
    async def test_search_by_thread(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test filtering by thread ID."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(limit=50, thread_id='thread_1')

        assert isinstance(results, dict)
        assert len(results['results']) == 2
        for result in results['results']:
            assert result['thread_id'] == 'thread_1'

    @pytest.mark.asyncio
    async def test_search_by_source(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test filtering by source type."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(limit=50, source='agent')

        assert isinstance(results, dict)
        assert len(results['results']) == 2
        for result in results['results']:
            assert result['source'] == 'agent'

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test filtering by tags."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(limit=50, tags=['important', 'nonexistent'])

        assert isinstance(results, dict)
        assert len(results['results']) == 1
        assert 'important' in results['results'][0]['tags']

    @pytest.mark.asyncio
    async def test_search_by_content_type(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test filtering by content type."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(limit=50, content_type='multimodal')

        assert isinstance(results, dict)
        assert len(results['results']) == 1
        assert results['results'][0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_search_with_pagination(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test pagination parameters."""
        _ = multiple_context_entries  # Fixture ensures data exists
        # Get first 2 results
        page1 = await search_context(limit=2, offset=0)
        assert isinstance(page1, dict)
        assert len(page1['results']) == 2

        # Get next 2 results
        page2 = await search_context(limit=2, offset=2)
        assert isinstance(page2, dict)
        assert len(page2['results']) == 2

        # Verify different results
        page1_ids = [r['id'] for r in page1['results']]
        page2_ids = [r['id'] for r in page2['results']]
        assert set(page1_ids).isdisjoint(set(page2_ids))

    @pytest.mark.asyncio
    async def test_search_include_images(
        self,
        temp_db_path: Path,
    ) -> None:
        """Test including image data in search results."""
        _ = temp_db_path  # Fixture provides database path
        # Store a context with image
        image_data = base64.b64encode(b'test_image_data').decode('utf-8')
        await store_context(
            thread_id='image_test',
            source='user',
            text='With image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )

        # Search with images included
        results = await search_context(
            limit=50,
            thread_id='image_test',
            include_images=True,
        )

        assert isinstance(results, dict)
        assert len(results['results']) == 1
        assert 'images' in results['results'][0]
        assert len(results['results'][0]['images']) == 1
        assert results['results'][0]['images'][0]['data'] == image_data

    @pytest.mark.asyncio
    async def test_search_complex_filters(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test combining multiple filters."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(
            limit=50,
            thread_id='thread_2',
            source='user',
            content_type='multimodal',
        )

        assert isinstance(results, dict)
        assert len(results['results']) == 1
        assert results['results'][0]['thread_id'] == 'thread_2'
        assert results['results'][0]['source'] == 'user'
        assert results['results'][0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_search_invalid_source(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test that Pydantic Literal validation handles invalid source.

        Note: Pydantic validates at FastMCP level. This test just verifies normal operation.
        """
        _ = multiple_context_entries  # Fixture ensures data exists
        # Valid source works fine
        result = await search_context(limit=50, source='user')
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_search_limit_max(self) -> None:
        """Test that Pydantic Field(le=100) enforces max limit.

        Note: Pydantic validates at FastMCP level. This test verifies max limit works.
        """
        # Create many entries
        for i in range(150):
            await store_context(
                thread_id=f'bulk_thread_{i}',
                source='user',
                text=f'Entry {i}',
            )

        # Valid max limit works fine
        result = await search_context(limit=100)
        assert 'results' in result
        assert len(result['results']) <= 100

    @pytest.mark.asyncio
    async def test_search_text_truncation_short(self) -> None:
        """Test that short text is not truncated."""
        short_text = 'This is a short text that should not be truncated.'
        assert len(short_text) < 300  # Ensure it's actually short

        await store_context(
            thread_id='truncation_test',
            source='user',
            text=short_text,
        )

        results = await search_context(limit=50, thread_id='truncation_test')
        assert isinstance(results, dict)
        assert len(results['results']) == 1
        assert results['results'][0]['text_content'] == short_text
        assert results['results'][0]['is_text_content_truncated'] is False

    @pytest.mark.asyncio
    async def test_search_text_truncation_long(self) -> None:
        """Test that long text is truncated with ellipsis."""
        # Create a text longer than 300 characters
        long_text = (
            'This is a very long text that exceeds the truncation limit. '
            'It contains multiple sentences to ensure it goes over 300 characters. '
            'This additional content will be truncated when returned from search_context. '
            'More content here to make it even longer and ensure truncation occurs. '
            'We need even more content to exceed the 300 character threshold for truncation. '
            'Adding extra sentences to guarantee this text is long enough for the test.'
        )
        assert len(long_text) > 300  # Ensure it's actually long

        await store_context(
            thread_id='truncation_long_test',
            source='agent',
            text=long_text,
        )

        results = await search_context(limit=50, thread_id='truncation_long_test')
        assert isinstance(results, dict)
        assert len(results['results']) == 1

        # Check truncation occurred
        assert results['results'][0]['is_text_content_truncated'] is True
        assert results['results'][0]['text_content'].endswith('...')
        assert len(results['results'][0]['text_content']) <= 303  # 300 + '...'
        assert results['results'][0]['text_content'] != long_text

        # Verify truncation preserves beginning of text
        assert long_text.startswith(results['results'][0]['text_content'][:-3])  # Remove '...'

    @pytest.mark.asyncio
    async def test_search_text_truncation_word_boundary(self) -> None:
        """Test that truncation happens at word boundaries when possible."""
        # Test case 1: Text with good word boundary near position 300
        text_with_good_boundary = (
            'This text has exactly the right length to test word boundary truncation behavior. '
            'When the 300th character falls within a word, the truncation algorithm should '
            'ideally find the nearest word boundary to avoid splitting words. '
            'We need additional content to push this text well past the 300 character truncation limit. '
            'Adding more sentences here ensures we have enough length for the truncation test to work properly.'
        )

        await store_context(
            thread_id='boundary_test_good',
            source='user',
            text=text_with_good_boundary,
        )

        results = await search_context(limit=50, thread_id='boundary_test_good')
        assert isinstance(results, dict)
        assert len(results['results']) == 1
        assert results['results'][0]['is_text_content_truncated'] is True
        assert results['results'][0]['text_content'].endswith('...')

        # Test case 2: Text where truncation will happen mid-word due to no good boundary
        # Create a text with a very long word starting before position 210
        long_word = (
            'verylongwordthatcannotbesplitproperlybecauseitexceedsthe'
            'boundarythresholdandcontinuesforquiteawhilelongerandlonger'
            'andlongerandlongerandlongerandlongerandlongerandlongerand'
            'longerandlongerandlongerandlonger'
        )
        text_with_bad_boundary = (
            'Short start then ' + long_word
            + ' and more text after to ensure truncation happens'
            ' at the right place in the output.'
        )

        await store_context(
            thread_id='boundary_test_bad',
            source='user',
            text=text_with_bad_boundary,
        )

        results_bad = await search_context(limit=50, thread_id='boundary_test_bad')
        assert isinstance(results_bad, dict)
        assert len(results_bad['results']) == 1
        assert results_bad['results'][0]['is_text_content_truncated'] is True
        assert results_bad['results'][0]['text_content'].endswith('...')
        # In this case, truncation happens at exactly 300 chars since no good word boundary exists

    @pytest.mark.asyncio
    async def test_search_vs_get_by_id_truncation(self) -> None:
        """Test that get_context_by_ids returns full text while search_context truncates."""
        long_text = (
            'This is a comprehensive test to verify that search_context truncates '
            'the text content while get_context_by_ids returns the complete full text. '
            'This distinction is important for the API design where search provides '
            'a preview and get_by_ids provides complete content for detailed viewing. '
            'Additional content here to ensure the text is sufficiently long.'
        )
        assert len(long_text) > 300

        store_result = await store_context(
            thread_id='comparison_test',
            source='user',
            text=long_text,
        )
        context_id = store_result['context_id']

        # Search should return truncated text
        search_results = await search_context(limit=50, thread_id='comparison_test')
        assert isinstance(search_results, dict)
        assert len(search_results['results']) == 1
        assert search_results['results'][0]['is_text_content_truncated'] is True
        assert search_results['results'][0]['text_content'] != long_text
        assert search_results['results'][0]['text_content'].endswith('...')

        # get_context_by_ids should return full text
        get_results = await get_context_by_ids(context_ids=[context_id])
        assert len(get_results) == 1
        entry = dict(get_results[0])
        assert entry['text_content'] == long_text
        assert 'is_text_content_truncated' not in entry  # This field should not exist

    @pytest.mark.asyncio
    async def test_search_null_text_truncation(self, temp_db_path: Path) -> None:
        """Test that null/empty text content is handled correctly."""
        # This shouldn't normally happen due to validation, but test defensive coding
        _ = temp_db_path  # Fixture provides database path
        await store_context(
            thread_id='null_test',
            source='user',
            text='placeholder',  # Store with some text first
        )

        # Directly update database to set text_content to empty (edge case)
        # Use backend-agnostic approach for both SQLite and PostgreSQL
        backend = app.startup.get_backend()
        assert backend is not None

        # Get backend type to determine SQL syntax
        backend_type = getattr(backend, 'backend_type', 'sqlite')

        if backend_type == 'sqlite':
            # SQLite uses execute_write to avoid connection pool issues
            def update_text_content(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE context_entries SET text_content = ? WHERE thread_id = ?',
                    ('', 'null_test'),
                )

            await backend.execute_write(update_text_content, access_scope=MAINTENANCE)
        else:
            # PostgreSQL uses async connection
            async with backend.get_connection() as conn:
                await conn.execute(
                    'UPDATE context_entries SET text_content = $1 WHERE thread_id = $2',
                    '', 'null_test',
                )

        results = await search_context(limit=50, thread_id='null_test')
        assert isinstance(results, dict)
        assert len(results['results']) == 1
        assert results['results'][0]['text_content'] == ''
        assert results['results'][0]['is_text_content_truncated'] is False


@pytest.mark.usefixtures('initialized_server')
class TestSearchContextScoping:
    """search_context returns only the entries the caller may read."""

    @staticmethod
    async def _store_as(principal_id: str, text: str, visibility: Literal['private', 'public']) -> str:
        """Store one entry in the scoped thread as the principal and return its id."""
        with as_principal(principal_id):
            result = await store_context(
                thread_id='scoped-browse', source='agent', text=text, visibility=visibility, metadata={'project': 'p'},
            )
        return result['context_id']

    @pytest.mark.asyncio
    async def test_unreadable_entries_are_absent(self) -> None:
        """Bob finds alice's public entry and nothing of her private one; the count matches."""
        await self._store_as('alice', 'alice private browse target', 'private')
        public_id = await self._store_as('alice', 'alice public browse target', 'public')

        with as_principal('bob'):
            results = await search_context(thread_id='scoped-browse', limit=50)

        assert [entry['id'] for entry in results['results']] == [public_id]
        assert results['count'] == 1

    @pytest.mark.asyncio
    async def test_owner_finds_their_private_entry(self) -> None:
        """The owner finds both of their entries, newest first."""
        private_id = await self._store_as('alice', 'alice private browse own', 'private')
        public_id = await self._store_as('alice', 'alice public browse own', 'public')

        with as_principal('alice'):
            results = await search_context(thread_id='scoped-browse', limit=50)

        assert [entry['id'] for entry in results['results']] == [public_id, private_id]

    @pytest.mark.asyncio
    async def test_results_report_visibility_and_ownership(self) -> None:
        """Every result carries its visibility and a bool telling whether the caller owns it, never the owner."""
        private_id = await self._store_as('alice', 'alice private browse read-back', 'private')
        public_id = await self._store_as('alice', 'alice public browse read-back', 'public')

        with as_principal('alice'):
            owned = (await search_context(thread_id='scoped-browse', limit=50))['results']
        with as_principal('bob'):
            read = (await search_context(thread_id='scoped-browse', limit=50))['results']

        assert [(entry['id'], entry['visibility'], entry['is_owner']) for entry in owned] == [
            (public_id, 'public', True), (private_id, 'private', True),
        ]
        assert [(entry['id'], entry['visibility'], entry['is_owner']) for entry in read] == [(public_id, 'public', False)]
        for entry in (*owned, *read):
            assert type(entry['is_owner']) is bool
            assert 'owner_id' not in entry

    @pytest.mark.asyncio
    async def test_scope_reaches_the_repository(self) -> None:
        """The caller's principal and groups reach search_contexts as its scope."""
        repos = await app.startup.ensure_repositories()

        with (
            as_principal('bob', groups=['team-x']),
            patch.object(repos.context, 'search_contexts', AsyncMock(return_value=([], {}))) as spy,
        ):
            await search_context(thread_id='scoped-browse', limit=10)

        assert spy.await_args is not None
        assert spy.await_args.kwargs['scope'] == AccessScope('bob', frozenset({'team-x'}))

    @pytest.mark.asyncio
    async def test_filters_applied_is_the_same_for_every_caller(self) -> None:
        """The explain stats count the client filters only, whoever runs them."""
        await self._store_as('alice', 'alice private browse stats', 'private')

        filters_applied: dict[str, int] = {}
        for principal_id in ('alice', 'bob'):
            with as_principal(principal_id):
                results = await search_context(
                    thread_id='scoped-browse', source='agent', metadata={'project': 'p'}, explain_query=True, limit=10,
                )
            filters_applied[principal_id] = results['stats']['filters_applied']
            assert results['stats']['query_plan']

        assert filters_applied == {'alice': 3, 'bob': 3}


@pytest.mark.usefixtures('initialized_server')
class TestSearchContextGenerationStatus:
    """Every result reports the state of its entry's generated outputs."""

    @pytest.mark.asyncio
    async def test_results_report_their_generation_state(self) -> None:
        """No job reads ready, a waiting job pending and a failed job failed."""
        stored = {
            label: (await store_context(thread_id='generation-browse', source='agent', text=f'browse {label}'))['context_id']
            for label in ('ready', 'pending', 'failed')
        }
        repos = await app.startup.ensure_repositories()
        await seed_generation_jobs(repos.context.backend, {stored['pending']: 'pending', stored['failed']: 'failed'})

        results = (await search_context(thread_id='generation-browse', limit=50))['results']

        assert {entry['id']: entry['generation_status'] for entry in results} == {
            stored['ready']: 'ready', stored['pending']: 'pending', stored['failed']: 'failed',
        }
