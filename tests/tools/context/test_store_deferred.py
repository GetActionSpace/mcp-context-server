"""Tests for store_context with defer_generation: the entry lands at once and a generation job completes it later.

A deferred store calls no provider and runs no generation: its transaction holds the entry,
its tags, images and author-group grants, and the job; the response reports the state the
transaction leaves the entry in, and the worker is woken after the commit. A deferred
retransmit keeps a waiting job, starts a failed one over, and enqueues one only when an
output is missing. Without an active provider a deferred store is a plain store.
"""

import sqlite3
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.startup
from app.backends.base import MAINTENANCE
from app.startup import ensure_repositories
from app.tools import store_context
from app.tools.context import store as store_module
from tests.helpers import as_principal
from tests.tools._generation_jobs import stored_jobs

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('initialized_server')]

LONG_TEXT = 'A deferred entry long enough to get a summary. ' * 20
PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='


@pytest.fixture
def providers() -> Iterator[dict[str, Any]]:
    """Activate an embedding and a summary provider; a test may set either to None through the returned dict."""
    original = (app.startup.get_embedding_provider(), app.startup.get_summary_provider())
    active: dict[str, Any] = {'embedding': MagicMock(name='embedding'), 'summary': MagicMock(name='summary')}
    app.startup.set_embedding_provider(active['embedding'])
    app.startup.set_summary_provider(active['summary'])
    try:
        yield active
    finally:
        app.startup.set_embedding_provider(original[0])
        app.startup.set_summary_provider(original[1])


def _deactivate(providers: dict[str, Any], *names: str) -> None:
    """Deactivate the named providers for the rest of the test."""
    for name in names:
        providers[name] = None
    app.startup.set_embedding_provider(providers['embedding'])
    app.startup.set_summary_provider(providers['summary'])


@pytest.fixture
def generation() -> Iterator[AsyncMock]:
    """Replace the store's generation so a test sees whether it ran; it returns a summary and nothing else."""
    with patch('app.tools.context.store.run_generation', AsyncMock(return_value=(None, 'A summary.', None))) as run:
        yield run


@pytest.fixture
def notify() -> Iterator[MagicMock]:
    """Record every wake-up of the generation worker."""
    with patch('app.tools.context.store.notify_generation_worker') as wake:
        yield wake


async def _row(context_id: str) -> sqlite3.Row:
    """Read the entry's summary and its child row counts as the table owner."""
    repos = await ensure_repositories()

    def _read(conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute(
            'SELECT summary, '
            '(SELECT COUNT(*) FROM tags WHERE context_entry_id = ce.id) AS tag_count, '
            '(SELECT COUNT(*) FROM image_attachments WHERE context_entry_id = ce.id) AS image_count, '
            '(SELECT COUNT(*) FROM context_entry_grants WHERE context_entry_id = ce.id) AS grant_count '
            'FROM context_entries ce WHERE id = ?',
            (context_id,),
        ).fetchone()
        assert row is not None
        return row

    return await repos.context.backend.execute_read(_read, access_scope=MAINTENANCE)


@pytest.mark.usefixtures('providers')
class TestDeferredStore:
    """A deferred store of a new entry."""

    async def test_the_entry_lands_with_a_job_and_no_generation(
        self, generation: AsyncMock, notify: MagicMock,
    ) -> None:
        repos = await ensure_repositories()
        with patch.object(repos.context, 'check_latest_is_duplicate', AsyncMock()) as precheck:
            result = await store_context(
                thread_id='deferred', source='agent', text=LONG_TEXT, tags=['alpha', 'beta'],
                images=[{'data': PNG, 'mime_type': 'image/png'}], defer_generation=True,
            )

        assert result['generation_status'] == 'pending'
        assert 'generation pending' in result['message']
        assert await stored_jobs(repos.context.backend) == {result['context_id']: 'pending'}
        row = await _row(result['context_id'])
        assert (row['summary'], row['tag_count'], row['image_count']) == (None, 2, 1)
        generation.assert_not_awaited()
        precheck.assert_not_awaited()
        notify.assert_called_once_with()

    async def test_author_group_grants_land_with_the_entry(
        self, monkeypatch: pytest.MonkeyPatch, generation: AsyncMock,
    ) -> None:
        del generation
        settings = store_module.settings
        access = settings.access_control.model_copy(update={'default_group_grants': 'author_groups'})
        monkeypatch.setattr(store_module, 'settings', settings.model_copy(update={'access_control': access}))

        with as_principal('alice', groups=['team-x']):
            result = await store_context(thread_id='deferred', source='agent', text=LONG_TEXT, defer_generation=True)

        assert (await _row(result['context_id']))['grant_count'] == 1
        assert result['generation_status'] == 'pending'

    async def test_a_store_without_deduplication_lands_with_a_job(self, generation: AsyncMock) -> None:
        del generation
        first = await store_context(thread_id='deferred', source='agent', text=LONG_TEXT, defer_generation=True)
        second = await store_context(
            thread_id='deferred', source='agent', text=LONG_TEXT, deduplicate=False, defer_generation=True,
        )

        repos = await ensure_repositories()
        assert first['context_id'] != second['context_id']
        assert await stored_jobs(repos.context.backend) == {first['context_id']: 'pending', second['context_id']: 'pending'}

    async def test_without_an_active_provider_the_store_is_a_plain_store(
        self, providers: dict[str, Any], generation: AsyncMock, notify: MagicMock,
    ) -> None:
        _deactivate(providers, 'embedding', 'summary')

        result = await store_context(thread_id='deferred', source='agent', text=LONG_TEXT, defer_generation=True)

        repos = await ensure_repositories()
        assert result['generation_status'] == 'ready'
        assert 'generation pending' not in result['message']
        assert await stored_jobs(repos.context.backend) == {}
        generation.assert_awaited_once()
        notify.assert_not_called()


@pytest.mark.usefixtures('providers')
class TestSynchronousStore:
    """The default store keeps generating before its transaction."""

    async def test_the_default_store_generates_and_leaves_no_job(
        self, providers: dict[str, Any], generation: AsyncMock, notify: MagicMock,
    ) -> None:
        _deactivate(providers, 'embedding')

        result = await store_context(thread_id='synchronous', source='agent', text=LONG_TEXT)

        repos = await ensure_repositories()
        assert result['generation_status'] == 'ready'
        assert (await _row(result['context_id']))['summary'] == 'A summary.'
        assert await stored_jobs(repos.context.backend) == {}
        generation.assert_awaited_once()
        notify.assert_not_called()

    async def test_a_retransmit_into_a_pending_entry_leaves_its_job(
        self, providers: dict[str, Any], generation: AsyncMock,
    ) -> None:
        _deactivate(providers, 'embedding')
        first = await store_context(thread_id='synchronous', source='agent', text=LONG_TEXT, defer_generation=True)

        second = await store_context(thread_id='synchronous', source='agent', text=LONG_TEXT)

        repos = await ensure_repositories()
        assert second['context_id'] == first['context_id']
        assert second['generation_status'] == 'pending'
        assert await stored_jobs(repos.context.backend) == {first['context_id']: 'pending'}
        generation.assert_awaited_once()


@pytest.mark.usefixtures('providers')
class TestDeferredRetransmit:
    """A deferred store whose text the caller's latest entry already holds."""

    async def test_a_waiting_job_is_kept(self, generation: AsyncMock, notify: MagicMock) -> None:
        del generation
        first = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT, defer_generation=True)
        repos = await ensure_repositories()

        second = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT, defer_generation=True)

        assert second['context_id'] == first['context_id']
        assert second['generation_status'] == 'pending'
        assert await stored_jobs(repos.context.backend) == {first['context_id']: 'pending'}
        assert notify.call_count == 2

    async def test_a_failed_job_starts_over(self, generation: AsyncMock) -> None:
        del generation
        first = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT, defer_generation=True)
        repos = await ensure_repositories()
        await _fail_job(first['context_id'])

        second = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT, defer_generation=True)

        assert second['generation_status'] == 'pending'
        assert await stored_jobs(repos.context.backend) == {first['context_id']: 'pending'}

    async def test_a_ready_entry_with_its_outputs_receives_no_job(
        self, providers: dict[str, Any], generation: AsyncMock,
    ) -> None:
        del generation
        _deactivate(providers, 'embedding')
        first = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT)

        second = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT, defer_generation=True)

        repos = await ensure_repositories()
        assert second['context_id'] == first['context_id']
        assert second['generation_status'] == 'ready'
        assert await stored_jobs(repos.context.backend) == {}

    async def test_a_missing_output_brings_a_job(self, providers: dict[str, Any], generation: AsyncMock) -> None:
        """An entry stored while no summary provider was active lacks a summary the text now gets."""
        _deactivate(providers, 'embedding', 'summary')
        generation.return_value = (None, None, None)
        first = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT)
        generation.assert_awaited_once()
        providers['summary'] = MagicMock(name='summary')
        app.startup.set_summary_provider(providers['summary'])

        second = await store_context(thread_id='retransmit', source='agent', text=LONG_TEXT, defer_generation=True)

        repos = await ensure_repositories()
        assert second['context_id'] == first['context_id']
        assert second['generation_status'] == 'pending'
        assert await stored_jobs(repos.context.backend) == {first['context_id']: 'pending'}


async def _fail_job(context_id: str) -> None:
    """Mark the entry's job failed, as a worker does after its last attempt."""
    repos = await ensure_repositories()

    def _fail(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE generation_jobs SET status = 'failed' WHERE context_entry_id = ?", (context_id,))

    await repos.context.backend.execute_write(_fail, access_scope=MAINTENANCE)
