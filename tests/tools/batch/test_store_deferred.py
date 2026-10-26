"""Tests for the per-entry defer_generation key of store_context_batch.

A deferring entry skips the pre-check and the per-entry generation pass and is stored
with a generation job; an entry without the key generates as before. Every stored item
reports the generation state its transaction left the entry in, a failed item reports
None, the message counts the pending entries, and the worker is woken once per batch.
"""

import sqlite3
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.startup
from app.backends.base import MAINTENANCE
from app.startup import ensure_repositories
from app.tools.batch.store import store_context_batch
from tests.tools._generation_jobs import stored_jobs

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('initialized_server', 'summary_provider')]

TEXT = 'A batch entry long enough to get a summary. ' * 20


@pytest.fixture
def summary_provider() -> Iterator[None]:
    """Activate a summary provider and no embedding provider for the test."""
    original = (app.startup.get_embedding_provider(), app.startup.get_summary_provider())
    app.startup.set_embedding_provider(None)
    app.startup.set_summary_provider(MagicMock(name='summary'))
    try:
        yield
    finally:
        app.startup.set_embedding_provider(original[0])
        app.startup.set_summary_provider(original[1])


@pytest.fixture
def summaries() -> Iterator[AsyncMock]:
    """Replace the batch's summary generation; every generated summary reads 'A summary.'."""
    with patch('app.tools.batch.store.generate_summary_with_timeout', AsyncMock(return_value='A summary.')) as generate:
        yield generate


@pytest.fixture
def nodes() -> Iterator[AsyncMock]:
    """Replace the batch's node-summary pass; it produces no node rows."""
    with patch('app.tools.batch.store.generate_index_nodes_with_timeout', AsyncMock(return_value=[])) as generate:
        yield generate


@pytest.fixture
def notify() -> Iterator[MagicMock]:
    """Record every wake-up of the generation worker."""
    with patch('app.tools.batch.store.notify_generation_worker') as wake:
        yield wake


async def _summaries() -> dict[str, object]:
    """Read every entry's summary as the table owner."""
    repos = await ensure_repositories()

    def _read(conn: sqlite3.Connection) -> dict[str, object]:
        return {str(row[0]): row[1] for row in conn.execute('SELECT id, summary FROM context_entries').fetchall()}

    return await repos.context.backend.execute_read(_read, access_scope=MAINTENANCE)


def _entry(thread: str, *, defer: object = None) -> dict[str, object]:
    """A batch entry of TEXT in ``thread``, carrying the defer_generation key unless ``defer`` is None."""
    entry: dict[str, object] = {'thread_id': thread, 'source': 'agent', 'text': TEXT}
    if defer is not None:
        entry['defer_generation'] = defer
    return entry


@pytest.mark.parametrize('atomic', [True, False])
async def test_mixed_flags_defer_only_the_flagged_entries(
    summaries: AsyncMock, nodes: AsyncMock, notify: MagicMock, *, atomic: bool,
) -> None:
    result = await store_context_batch(
        entries=[_entry('first', defer=True), _entry('second'), _entry('third', defer='true')], atomic=atomic,
    )

    repos = await ensure_repositories()
    ids = [str(item['context_id']) for item in result['results']]
    assert result['succeeded'] == 3
    assert [item['generation_status'] for item in result['results']] == ['pending', 'ready', 'pending']
    assert await stored_jobs(repos.context.backend) == {ids[0]: 'pending', ids[2]: 'pending'}
    stored = await _summaries()
    assert [stored[context_id] for context_id in ids] == [None, 'A summary.', None]
    summaries.assert_awaited_once()
    nodes.assert_awaited_once()
    assert 'generation pending for 2 entries' in result['message']
    notify.assert_called_once_with()


async def test_a_batch_without_deferring_entries_wakes_no_worker(
    summaries: AsyncMock, nodes: AsyncMock, notify: MagicMock,
) -> None:
    del summaries, nodes
    result = await store_context_batch(entries=[_entry('only')])

    assert [item['generation_status'] for item in result['results']] == ['ready']
    notify.assert_not_called()


async def test_a_deferring_entry_skips_the_pre_check(summaries: AsyncMock, nodes: AsyncMock) -> None:
    del summaries, nodes
    repos = await ensure_repositories()

    with patch.object(repos.context, 'check_latest_is_duplicate', AsyncMock(return_value=None)) as precheck:
        await store_context_batch(entries=[_entry('deferred', defer=True), _entry('synchronous')])

    assert precheck.await_count == 1
    assert precheck.await_args is not None
    assert precheck.await_args.kwargs['thread_id'] == 'synchronous'


async def test_a_non_boolean_key_fails_its_entry(summaries: AsyncMock, nodes: AsyncMock) -> None:
    del summaries, nodes
    result = await store_context_batch(entries=[_entry('bad', defer='maybe'), _entry('good', defer=True)], atomic=False)

    failed, stored = result['results']
    assert (failed['success'], failed['error'], failed['generation_status']) == (
        False, 'defer_generation must be a boolean', None,
    )
    assert (stored['success'], stored['generation_status']) == (True, 'pending')


async def test_without_an_active_provider_deferring_entries_are_plain_entries(
    summaries: AsyncMock, nodes: AsyncMock, notify: MagicMock,
) -> None:
    del summaries, nodes
    app.startup.set_summary_provider(None)

    result = await store_context_batch(entries=[_entry('plain', defer=True)])

    repos = await ensure_repositories()
    assert [item['generation_status'] for item in result['results']] == ['ready']
    assert await stored_jobs(repos.context.backend) == {}
    notify.assert_not_called()
