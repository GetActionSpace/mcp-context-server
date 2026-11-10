"""Tests for the generation jobs the update transaction of app.tools._update_transaction leaves behind.

A text update replaces the entry's job when it defers generation and removes it when it
generated the outputs itself; an update that leaves the text alone leaves the job. The
transaction reports the generation state it leaves the entry in.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.tools._update_transaction import UpdateOutcome
from app.tools._update_transaction import execute_update_in_transaction
from tests.helpers import LOCAL_SCOPE
from tests.tools._generation_jobs import InMemoryGenerationJobs
from tests.tools._transaction_doubles import record_calls
from tests.tools._transaction_doubles import sqlite_txn
from tests.tools._transaction_doubles import transaction_repos

ENTRY_ID = 'a' * 32
EMBEDDINGS = [ChunkEmbedding(embedding=[0.1], start_index=0, end_index=4)]

pytestmark = pytest.mark.asyncio


@pytest.fixture
def jobs() -> InMemoryGenerationJobs:
    """The job repository double, starting without a job."""
    return InMemoryGenerationJobs()


@pytest.fixture
def repos(jobs: InMemoryGenerationJobs) -> MagicMock:
    """Repositories whose writes succeed on the entry, which holds its embeddings and its summary."""
    return transaction_repos(jobs, ENTRY_ID)


@pytest.fixture
def txn() -> MagicMock:
    """A transaction double on SQLite."""
    return sqlite_txn()


async def _update(repos: MagicMock, txn: MagicMock, *, defer_generation: bool, **overrides: Any) -> UpdateOutcome:
    """Run one update of the entry, a deferred-shaped text change unless a test says otherwise."""
    arguments: dict[str, Any] = {
        'context_id': ENTRY_ID, 'scope': LOCAL_SCOPE, 'text': 'New text.', 'metadata': None, 'metadata_patch': None,
        'summary': None, 'clear_summary': True, 'tags': None, 'images': None, 'validated_images': [],
        'chunk_embeddings': None, 'embedding_model': 'model', 'index_nodes': [], 'expected_version': 0,
        'defer_generation': defer_generation,
    }
    arguments.update(overrides)
    return await execute_update_in_transaction(repos, txn, **arguments)


class TestUpdateJobs:
    """The job an update leaves its entry, and the state it reports."""

    @pytest.mark.parametrize('status', [None, 'pending', 'running', 'failed'])
    async def test_a_deferred_text_update_replaces_the_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs, status: str | None,
    ) -> None:
        if status is not None:
            jobs.jobs[ENTRY_ID] = status

        outcome = await _update(repos, txn, defer_generation=True)

        assert jobs.jobs == {ENTRY_ID: 'pending'}
        assert outcome.generation_status == 'pending'

    async def test_a_deferred_text_update_removes_the_old_outputs(self, repos: MagicMock, txn: MagicMock) -> None:
        await _update(repos, txn, defer_generation=True)

        repos.embeddings.delete_all_chunks.assert_awaited_once_with(ENTRY_ID, txn=txn, scope=LOCAL_SCOPE)
        repos.index_nodes.replace_nodes_for_context.assert_awaited_once_with(ENTRY_ID, [], txn=txn, scope=LOCAL_SCOPE)
        assert repos.context.update_context_entry.await_args.kwargs['clear_summary'] is True

    async def test_a_synchronous_text_update_removes_the_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        jobs.jobs[ENTRY_ID] = 'running'

        outcome = await _update(
            repos, txn, defer_generation=False, chunk_embeddings=EMBEDDINGS, summary='Fresh.', clear_summary=False,
        )

        assert jobs.jobs == {}
        assert outcome.generation_status == 'ready'

    @pytest.mark.parametrize('defer_generation', [False, True])
    @pytest.mark.parametrize(
        'change',
        [
            {'metadata': {'k': 'v'}}, {'metadata_patch': {'k': 'v'}}, {'tags': ['tag']}, {'images': []},
            {'visibility': 'public'},
        ],
        ids=['metadata', 'metadata_patch', 'tags', 'images', 'visibility'],
    )
    async def test_an_update_without_text_leaves_the_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
        change: dict[str, Any], *, defer_generation: bool,
    ) -> None:
        jobs.jobs[ENTRY_ID] = 'failed'

        outcome = await _update(
            repos, txn, defer_generation=defer_generation, text=None, clear_summary=False, index_nodes=None, **change,
        )

        assert jobs.jobs == {ENTRY_ID: 'failed'}
        assert outcome.generation_status == 'failed'

    async def test_the_job_follows_the_tags_and_images(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        order: list[str] = []
        record_calls(monkeypatch, repos.tags, 'replace_tags_for_context', order, 'tags')
        record_calls(monkeypatch, repos.images, 'replace_images_for_context', order, 'images')
        record_calls(monkeypatch, jobs, 'delete_for_entry', order, 'job')

        await _update(repos, txn, defer_generation=True, tags=['tag'], images=[])

        assert order == ['tags', 'images', 'job']

    async def test_the_outcome_carries_the_updated_fields_and_the_summary_clear(
        self, repos: MagicMock, txn: MagicMock,
    ) -> None:
        outcome = await _update(repos, txn, defer_generation=True)

        assert outcome == UpdateOutcome(
            updated_fields=['text_content'], summary_cleared=True, generation_status='pending',
        )
