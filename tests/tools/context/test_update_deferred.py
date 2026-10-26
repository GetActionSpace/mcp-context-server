"""Tests for update_context with defer_generation: new text lands at once and a generation job completes it later.

A deferred text update runs no generation: its compare-and-set transaction writes the
text, clears the summary the old text had, and replaces the entry's job with a new one,
and the worker is woken after the commit. A synchronous text update removes the job its
new outputs supersede. An update without text never touches the job, and the flag changes
nothing for it.
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
from app.backends.base import TransactionContext
from app.repositories import RepositoryContainer
from app.repositories.context_repository.records import VersionConflictError
from app.startup import ensure_repositories
from app.tools import store_context
from app.tools import update_context
from app.tools._transactions import UpdateOutcome
from app.tools._transactions import execute_update_in_transaction
from tests.tools._generation_jobs import seed_generation_jobs
from tests.tools._generation_jobs import stored_jobs

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('initialized_server')]

NEW_TEXT = 'The rewritten entry, long enough to get a summary. ' * 20


@pytest.fixture
def providers() -> Iterator[None]:
    """Activate an embedding and a summary provider for the test."""
    original = (app.startup.get_embedding_provider(), app.startup.get_summary_provider())
    app.startup.set_embedding_provider(MagicMock(name='embedding'))
    app.startup.set_summary_provider(MagicMock(name='summary'))
    try:
        yield
    finally:
        app.startup.set_embedding_provider(original[0])
        app.startup.set_summary_provider(original[1])


@pytest.fixture
def generation() -> Iterator[AsyncMock]:
    """Replace the update's generation so a test sees whether it ran; it returns a summary and nothing else."""
    with patch('app.tools.context.update.run_generation', AsyncMock(return_value=(None, 'A summary.', []))) as run:
        yield run


@pytest.fixture
def notify() -> Iterator[MagicMock]:
    """Record every wake-up of the generation worker."""
    with patch('app.tools.context.update.notify_generation_worker') as wake:
        yield wake


async def _summarized_entry(job: str | None = None) -> str:
    """Store an entry holding a summary, and give it a job in the stored status ``job`` when one is named.

    The entry is stored while no provider is active, so storing it generates nothing.

    Returns:
        The entry's id.
    """
    active = (app.startup.get_embedding_provider(), app.startup.get_summary_provider())
    app.startup.set_embedding_provider(None)
    app.startup.set_summary_provider(None)
    try:
        result = await store_context(thread_id='update-deferred', source='agent', text='The original entry.')
    finally:
        app.startup.set_embedding_provider(active[0])
        app.startup.set_summary_provider(active[1])
    context_id = result['context_id']
    repos = await ensure_repositories()

    def _summarize(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE context_entries SET summary = 'The original summary.' WHERE id = ?", (context_id,))

    await repos.context.backend.execute_write(_summarize, access_scope=MAINTENANCE)
    if job is not None:
        await seed_generation_jobs(repos.context.backend, {context_id: job})
    return context_id


async def _entry(context_id: str) -> sqlite3.Row:
    """Read the entry's text and summary as the table owner."""
    repos = await ensure_repositories()

    def _read(conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute('SELECT text_content, summary FROM context_entries WHERE id = ?', (context_id,)).fetchone()
        assert row is not None
        return row

    return await repos.context.backend.execute_read(_read, access_scope=MAINTENANCE)


@pytest.mark.usefixtures('providers')
class TestDeferredTextUpdate:
    """A text update that defers generation."""

    @pytest.mark.parametrize('job', [None, 'pending', 'running', 'failed'])
    async def test_the_text_lands_with_a_new_job_and_no_generation(
        self, generation: AsyncMock, notify: MagicMock, job: str | None,
    ) -> None:
        context_id = await _summarized_entry(job)

        result = await update_context(context_id=context_id, text=NEW_TEXT, defer_generation=True)

        repos = await ensure_repositories()
        assert result['generation_status'] == 'pending'
        assert 'generation pending' in result['message']
        assert await stored_jobs(repos.context.backend) == {context_id: 'pending'}
        row = await _entry(context_id)
        assert (row['text_content'], row['summary']) == (NEW_TEXT.strip(), None)
        generation.assert_not_awaited()
        notify.assert_called_once_with()

    async def test_the_old_outputs_go_with_the_old_text(self, generation: AsyncMock) -> None:
        """The transaction receives no embeddings, clears the summary and clears the node rows."""
        del generation
        context_id = await _summarized_entry()

        with patch(
            'app.tools.context.update.execute_update_in_transaction', side_effect=execute_update_in_transaction,
        ) as transaction:
            await update_context(context_id=context_id, text=NEW_TEXT, defer_generation=True)

        assert transaction.await_args is not None
        arguments = transaction.await_args.kwargs
        assert arguments['defer_generation'] is True
        assert (arguments['chunk_embeddings'], arguments['summary'], arguments['clear_summary']) == (None, None, True)
        assert arguments['index_nodes'] == []

    async def test_a_version_conflict_retries_the_deferred_update(self, generation: AsyncMock) -> None:
        del generation
        context_id = await _summarized_entry()
        calls: list[dict[str, Any]] = []

        async def _conflict_once(repos: RepositoryContainer, txn: TransactionContext, **kwargs: Any) -> UpdateOutcome:
            calls.append(kwargs)
            if len(calls) == 1:
                raise VersionConflictError(context_id)
            return await execute_update_in_transaction(repos, txn, **kwargs)

        with patch('app.tools.context.update.execute_update_in_transaction', side_effect=_conflict_once):
            result = await update_context(context_id=context_id, text=NEW_TEXT, defer_generation=True)

        repos = await ensure_repositories()
        assert [call['defer_generation'] for call in calls] == [True, True]
        assert result['generation_status'] == 'pending'
        assert await stored_jobs(repos.context.backend) == {context_id: 'pending'}

    async def test_without_an_active_provider_the_update_is_a_plain_update(
        self, generation: AsyncMock, notify: MagicMock,
    ) -> None:
        context_id = await _summarized_entry('failed')
        app.startup.set_embedding_provider(None)
        app.startup.set_summary_provider(None)

        result = await update_context(context_id=context_id, text=NEW_TEXT, defer_generation=True)

        repos = await ensure_repositories()
        assert result['generation_status'] == 'ready'
        assert await stored_jobs(repos.context.backend) == {}
        generation.assert_awaited_once()
        notify.assert_not_called()


@pytest.mark.usefixtures('providers')
class TestSynchronousTextUpdate:
    """The default text update keeps generating before its transaction."""

    @pytest.mark.parametrize('job', ['pending', 'failed'])
    async def test_new_outputs_remove_the_job(self, generation: AsyncMock, notify: MagicMock, job: str) -> None:
        context_id = await _summarized_entry(job)

        result = await update_context(context_id=context_id, text=NEW_TEXT)

        repos = await ensure_repositories()
        assert result['generation_status'] == 'ready'
        assert await stored_jobs(repos.context.backend) == {}
        assert (await _entry(context_id))['summary'] == 'A summary.'
        generation.assert_awaited_once()
        notify.assert_not_called()


@pytest.mark.usefixtures('providers')
class TestUpdateWithoutText:
    """An update that changes no text leaves the job and reports its state."""

    @pytest.mark.parametrize('defer_generation', [False, True])
    @pytest.mark.parametrize(
        'change',
        [{'metadata': {'k': 'v'}}, {'metadata_patch': {'k': 'v'}}, {'tags': ['tag']}, {'visibility': 'public'}],
        ids=['metadata', 'metadata_patch', 'tags', 'visibility'],
    )
    async def test_the_job_is_left_as_it_is(
        self, generation: AsyncMock, notify: MagicMock, change: dict[str, Any], *, defer_generation: bool,
    ) -> None:
        context_id = await _summarized_entry('failed')

        result = await update_context(context_id=context_id, defer_generation=defer_generation, **change)

        repos = await ensure_repositories()
        assert result['generation_status'] == 'failed'
        assert 'generation failed' in result['message']
        assert await stored_jobs(repos.context.backend) == {context_id: 'failed'}
        assert (await _entry(context_id))['summary'] == 'The original summary.'
        generation.assert_not_awaited()
        notify.assert_not_called()
