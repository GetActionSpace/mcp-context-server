"""Tests for the per-entry defer_generation key of update_context_batch.

A deferring text update skips the per-entry generation pass: its transaction stores the
new text, clears the old summary and replaces the entry's job. A synchronous text update
removes the job, an update without text leaves it, and the key changes nothing for an
update without text. Every item reports the generation state its transaction left the
entry in, a failed item reports None, and the worker is woken once per batch.
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
from app.tools._transactions import execute_update_in_transaction
from app.tools.batch.update import update_context_batch
from app.tools.context.store import store_context
from tests.tools._generation_jobs import seed_generation_jobs
from tests.tools._generation_jobs import stored_jobs

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('initialized_server')]

NEW_TEXT = 'Batch-updated text long enough to get a summary. ' * 20


@pytest.fixture
def summaries() -> Iterator[AsyncMock]:
    """Replace the batch's summary generation; every generated summary reads 'A summary.'."""
    with patch('app.tools.batch.update.generate_summary_with_timeout', AsyncMock(return_value='A summary.')) as generate:
        yield generate


@pytest.fixture
def nodes() -> Iterator[AsyncMock]:
    """Replace the batch's node-summary pass; it produces no node rows."""
    with patch('app.tools.batch.update.generate_index_nodes_with_timeout', AsyncMock(return_value=[])) as generate:
        yield generate


@pytest.fixture
def notify() -> Iterator[MagicMock]:
    """Record every wake-up of the generation worker."""
    with patch('app.tools.batch.update.notify_generation_worker') as wake:
        yield wake


async def _entries(jobs: list[str | None]) -> list[str]:
    """Store one summarized entry per item of ``jobs``, each with a job in the named stored status.

    The entries are stored before a provider is active, so storing them generates nothing; a
    summary provider is active once they are stored.

    Returns:
        The ids of the entries, in the order of ``jobs``.
    """
    ids = [
        (await store_context(thread_id='batch-update-deferred', source='agent', text=f'Original entry {index}.'))['context_id']
        for index in range(len(jobs))
    ]
    repos = await ensure_repositories()

    def _summarize(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE context_entries SET summary = 'The original summary.'")

    await repos.context.backend.execute_write(_summarize, access_scope=MAINTENANCE)
    await seed_generation_jobs(
        repos.context.backend, {context_id: job for context_id, job in zip(ids, jobs, strict=True) if job is not None},
    )
    app.startup.set_summary_provider(MagicMock(name='summary'))
    return ids


async def _summaries() -> dict[str, object]:
    """Read every entry's summary as the table owner."""
    repos = await ensure_repositories()

    def _read(conn: sqlite3.Connection) -> dict[str, object]:
        return {str(row[0]): row[1] for row in conn.execute('SELECT id, summary FROM context_entries').fetchall()}

    return await repos.context.backend.execute_read(_read, access_scope=MAINTENANCE)


@pytest.fixture(autouse=True)
def restore_providers() -> Iterator[None]:
    """Run every test without an embedding provider and restore both providers afterwards."""
    original = (app.startup.get_embedding_provider(), app.startup.get_summary_provider())
    app.startup.set_embedding_provider(None)
    app.startup.set_summary_provider(None)
    try:
        yield
    finally:
        app.startup.set_embedding_provider(original[0])
        app.startup.set_summary_provider(original[1])


@pytest.mark.parametrize('atomic', [True, False])
async def test_each_update_leaves_the_job_its_kind_calls_for(
    summaries: AsyncMock, nodes: AsyncMock, notify: MagicMock, *, atomic: bool,
) -> None:
    deferred, synchronous, metadata_only = await _entries(['failed', 'pending', 'failed'])

    result = await update_context_batch(
        updates=[
            {'context_id': deferred, 'text': NEW_TEXT, 'defer_generation': True},
            {'context_id': synchronous, 'text': NEW_TEXT},
            {'context_id': metadata_only, 'metadata': {'k': 'v'}, 'defer_generation': 'yes'},
        ],
        atomic=atomic,
    )

    repos = await ensure_repositories()
    assert result['succeeded'] == 3
    assert [item['generation_status'] for item in result['results']] == ['pending', 'ready', 'failed']
    assert await stored_jobs(repos.context.backend) == {deferred: 'pending', metadata_only: 'failed'}
    stored = await _summaries()
    assert [stored[context_id] for context_id in (deferred, synchronous, metadata_only)] == [
        None, 'A summary.', 'The original summary.',
    ]
    summaries.assert_awaited_once()
    nodes.assert_awaited_once()
    assert 'generation pending for 1 entry' in result['message']
    assert 'generation failed for 1 entry' in result['message']
    notify.assert_called_once_with()


async def test_a_deferring_update_removes_the_old_outputs(summaries: AsyncMock, nodes: AsyncMock) -> None:
    """Its transaction receives no embeddings and no summary, clears the summary and clears the node rows."""
    del summaries
    (context_id,) = await _entries([None])

    with patch(
        'app.tools.batch.update.execute_update_in_transaction', side_effect=execute_update_in_transaction,
    ) as transaction:
        await update_context_batch(updates=[{'context_id': context_id, 'text': NEW_TEXT, 'defer_generation': True}])

    assert transaction.await_args is not None
    arguments = transaction.await_args.kwargs
    assert arguments['defer_generation'] is True
    assert (arguments['chunk_embeddings'], arguments['summary'], arguments['clear_summary']) == (None, None, True)
    assert arguments['index_nodes'] == []
    nodes.assert_not_awaited()


async def test_a_batch_without_deferring_updates_wakes_no_worker(
    summaries: AsyncMock, nodes: AsyncMock, notify: MagicMock,
) -> None:
    del summaries, nodes
    (context_id,) = await _entries([None])

    result = await update_context_batch(updates=[{'context_id': context_id, 'text': NEW_TEXT}])

    assert [item['generation_status'] for item in result['results']] == ['ready']
    notify.assert_not_called()


async def test_a_non_boolean_key_fails_its_update(summaries: AsyncMock, nodes: AsyncMock) -> None:
    del summaries, nodes
    bad, good = await _entries([None, None])

    result = await update_context_batch(
        updates=[
            {'context_id': bad, 'text': NEW_TEXT, 'defer_generation': 'maybe'},
            {'context_id': good, 'text': NEW_TEXT, 'defer_generation': True},
        ],
        atomic=False,
    )

    failed, updated = result['results']
    assert (failed['success'], failed['error'], failed['generation_status']) == (
        False, 'defer_generation must be a boolean', None,
    )
    assert (updated['success'], updated['generation_status']) == (True, 'pending')


async def test_without_an_active_provider_a_deferring_update_is_a_plain_update(
    summaries: AsyncMock, nodes: AsyncMock, notify: MagicMock,
) -> None:
    del summaries, nodes
    (context_id,) = await _entries(['failed'])
    app.startup.set_summary_provider(None)

    result = await update_context_batch(updates=[{'context_id': context_id, 'text': NEW_TEXT, 'defer_generation': True}])

    repos = await ensure_repositories()
    assert [item['generation_status'] for item in result['results']] == ['ready']
    assert await stored_jobs(repos.context.backend) == {}
    notify.assert_not_called()
