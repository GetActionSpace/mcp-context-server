"""Tests for the generation jobs the store transaction of app.tools._store_transaction leaves behind.

A store that defers generation leaves its new entry a job; a deferred store that a
retransmit deduplicates into an existing entry keeps a waiting job, starts a failed one
over, and adds one only when an output generation would produce is missing. The
transaction reports the generation state it leaves the entry in.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.tools._store_transaction import StoreOutcome
from app.tools._store_transaction import execute_store_in_transaction
from tests.helpers import LOCAL_SCOPE
from tests.tools._generation_jobs import InMemoryGenerationJobs
from tests.tools._transaction_doubles import record_calls
from tests.tools._transaction_doubles import sqlite_txn
from tests.tools._transaction_doubles import transaction_repos

ENTRY_ID = 'a' * 32
TEXT = 'A retransmitted text long enough to summarize.'
EMBEDDINGS = [ChunkEmbedding(embedding=[0.1], start_index=0, end_index=4)]

pytestmark = pytest.mark.asyncio


@pytest.fixture
def jobs() -> InMemoryGenerationJobs:
    """The job repository double, starting without a job."""
    return InMemoryGenerationJobs()


@pytest.fixture
def repos(jobs: InMemoryGenerationJobs) -> MagicMock:
    """Repositories whose writes succeed; a store inserts a new entry unless a test says otherwise."""
    return transaction_repos(jobs, ENTRY_ID)


@pytest.fixture
def txn() -> MagicMock:
    """A transaction double on SQLite."""
    return sqlite_txn()


@pytest.fixture
def providers() -> Iterator[dict[str, Any]]:
    """Both providers active and every text summarized; a test switches either off through the returned dict."""
    active: dict[str, Any] = {'embedding': MagicMock(), 'summary': True}
    with (
        patch('app.tools._store_transaction.get_embedding_provider', side_effect=lambda: active['embedding']),
        patch('app.tools._store_transaction.summary_applies', side_effect=lambda _text: active['summary']),
    ):
        yield active


async def _store(repos: MagicMock, txn: MagicMock, *, defer_generation: bool, **overrides: Any) -> StoreOutcome:
    """Run one store of TEXT with no tags, images or outputs unless a test says otherwise."""
    arguments: dict[str, Any] = {
        'thread_id': 'thread', 'source': 'agent', 'content_type': 'text', 'text_content': TEXT,
        'scope': LOCAL_SCOPE, 'visibility': 'private', 'metadata_str': None, 'summary': None, 'tags': None,
        'validated_images': [], 'images_provided': False, 'deduplicate': True,
        'defer_generation': defer_generation, 'chunk_embeddings': None, 'embedding_model': 'model',
    }
    arguments.update(overrides)
    return await execute_store_in_transaction(repos, txn, **arguments)


@pytest.mark.usefixtures('providers')
class TestStoreJobs:
    """The job a store leaves its entry, and the state it reports."""

    async def test_a_deferred_new_entry_receives_a_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {ENTRY_ID: 'pending'}
        assert outcome == StoreOutcome(
            context_id=ENTRY_ID, was_updated=False, embedding_stored=False, generation_status='pending',
        )

    async def test_a_deferred_store_without_deduplication_receives_a_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        outcome = await _store(repos, txn, defer_generation=True, deduplicate=False)

        repos.context.insert_entry.assert_awaited_once()
        assert jobs.jobs == {ENTRY_ID: 'pending'}
        assert outcome.generation_status == 'pending'

    async def test_a_synchronous_new_entry_receives_no_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        outcome = await _store(repos, txn, defer_generation=False, chunk_embeddings=EMBEDDINGS)

        assert jobs.jobs == {}
        assert outcome.embedding_stored is True
        assert outcome.generation_status == 'ready'

    async def test_a_deferred_store_never_asks_for_reconciliation(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        """A deferred store generates nothing before its transaction, so a new entry without outputs is expected."""
        outcome = await _store(
            repos, txn, defer_generation=True,
            embedding_generation_enabled=True, nodes_pending=True, summary_pending=True,
        )

        assert outcome.generation_status == 'pending'
        assert jobs.jobs == {ENTRY_ID: 'pending'}

    async def test_the_job_follows_the_grants_tags_and_images(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        order: list[str] = []
        record_calls(monkeypatch, repos.grants, 'store_group_read_grants', order, 'grants')
        record_calls(monkeypatch, repos.tags, 'store_tags', order, 'tags')
        record_calls(monkeypatch, repos.images, 'store_images', order, 'images')
        record_calls(monkeypatch, jobs, 'enqueue', order, 'job')

        await _store(
            repos, txn, defer_generation=True, author_group_grants=frozenset({'team'}), tags=['tag'],
            validated_images=[{'data': 'x', 'mime_type': 'image/png'}], images_provided=True,
        )

        assert order == ['grants', 'tags', 'images', 'job']


@pytest.mark.usefixtures('providers')
class TestDeferredDeduplicationJobs:
    """A deferred retransmit that deduplicates into an existing entry."""

    @pytest.fixture(autouse=True)
    def deduplicates(self, repos: MagicMock) -> None:
        """Every store deduplicates into the existing entry."""
        repos.context.store_with_deduplication = AsyncMock(return_value=(ENTRY_ID, True))

    async def test_a_waiting_job_is_kept(self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs) -> None:
        """The entry still lacks the outputs its job will write; the job covers them, so no second job is added."""
        jobs.jobs[ENTRY_ID] = 'running'
        repos.embeddings.exists = AsyncMock(return_value=False)
        repos.context.has_summary = AsyncMock(return_value=False)

        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {ENTRY_ID: 'running'}
        assert outcome.was_updated is True
        assert outcome.generation_status == 'pending'

    async def test_a_failed_job_starts_over(self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs) -> None:
        jobs.jobs[ENTRY_ID] = 'failed'

        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {ENTRY_ID: 'pending'}
        assert outcome.generation_status == 'pending'

    async def test_a_ready_entry_with_every_output_receives_no_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {}
        assert outcome.generation_status == 'ready'
        repos.embeddings.exists.assert_awaited_once_with(ENTRY_ID, txn=txn, scope=LOCAL_SCOPE)
        repos.context.has_summary.assert_awaited_once_with(ENTRY_ID, scope=LOCAL_SCOPE, txn=txn)

    async def test_missing_embeddings_bring_a_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        repos.embeddings.exists = AsyncMock(return_value=False)

        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {ENTRY_ID: 'pending'}
        assert outcome.generation_status == 'pending'

    async def test_missing_embeddings_without_an_embedding_provider_bring_no_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs, providers: dict[str, Any],
    ) -> None:
        providers['embedding'] = None
        repos.embeddings.exists = AsyncMock(return_value=False)

        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {}
        assert outcome.generation_status == 'ready'
        repos.embeddings.exists.assert_not_awaited()

    async def test_a_missing_summary_brings_a_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs,
    ) -> None:
        repos.context.has_summary = AsyncMock(return_value=False)

        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {ENTRY_ID: 'pending'}
        assert outcome.generation_status == 'pending'

    async def test_a_missing_summary_the_text_would_not_get_brings_no_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs, providers: dict[str, Any],
    ) -> None:
        providers['summary'] = False
        repos.context.has_summary = AsyncMock(return_value=False)

        outcome = await _store(repos, txn, defer_generation=True)

        assert jobs.jobs == {}
        assert outcome.generation_status == 'ready'
        repos.context.has_summary.assert_not_awaited()

    @pytest.mark.parametrize('status', ['pending', 'failed'])
    async def test_a_synchronous_retransmit_leaves_the_job(
        self, repos: MagicMock, txn: MagicMock, jobs: InMemoryGenerationJobs, status: str,
    ) -> None:
        jobs.jobs[ENTRY_ID] = status

        outcome = await _store(repos, txn, defer_generation=False)

        assert jobs.jobs == {ENTRY_ID: status}
        assert outcome.generation_status == status
