"""Tests for the generation jobs the store and update transactions of app.tools._transactions leave behind.

A store that defers generation leaves its new entry a job; a deferred store that a
retransmit deduplicates into an existing entry keeps a waiting job, starts a failed one
over, and adds one only when an output generation would produce is missing. A text update
replaces the entry's job when it defers generation and removes it when it generated the
outputs itself. Each transaction reports the generation state it leaves the entry in.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.tools._transactions import StoreOutcome
from app.tools._transactions import UpdateOutcome
from app.tools._transactions import execute_store_in_transaction
from app.tools._transactions import execute_update_in_transaction
from tests.helpers import LOCAL_SCOPE
from tests.tools._generation_jobs import InMemoryGenerationJobs

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
    repos = MagicMock()
    repos.context.store_with_deduplication = AsyncMock(return_value=(ENTRY_ID, False))
    repos.context.insert_entry = AsyncMock(return_value=ENTRY_ID)
    repos.context.has_summary = AsyncMock(return_value=True)
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)
    repos.context.touch_updated_at = AsyncMock(return_value=True)
    repos.context.entry_exists = AsyncMock(return_value=True)
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)
    repos.grants.store_group_read_grants = AsyncMock()
    repos.embeddings.exists = AsyncMock(return_value=True)
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.delete_all_chunks = AsyncMock(return_value=False)
    repos.embeddings.embedding_tables_exist = AsyncMock(return_value=True)
    repos.index_nodes.replace_nodes_for_context = AsyncMock()
    repos.generation_jobs = jobs
    return repos


@pytest.fixture
def txn() -> MagicMock:
    """A transaction double on SQLite."""
    txn = MagicMock()
    txn.backend_type = 'sqlite'
    return txn


@pytest.fixture
def providers() -> Iterator[dict[str, Any]]:
    """Both providers active and every text summarized; a test switches either off through the returned dict."""
    active: dict[str, Any] = {'embedding': MagicMock(), 'summary': True}
    with (
        patch('app.tools._transactions.get_embedding_provider', side_effect=lambda: active['embedding']),
        patch('app.tools._transactions.summary_applies', side_effect=lambda _text: active['summary']),
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


def _record_calls(
    monkeypatch: pytest.MonkeyPatch, target: object, name: str, order: list[str], label: str,
) -> None:
    """Record ``label`` in ``order`` whenever ``target.name`` is awaited, letting the call through."""
    method = getattr(target, name)

    async def _recording(*args: object, **kwargs: object) -> object:
        order.append(label)
        return await method(*args, **kwargs)

    monkeypatch.setattr(target, name, _recording)


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
        _record_calls(monkeypatch, repos.grants, 'store_group_read_grants', order, 'grants')
        _record_calls(monkeypatch, repos.tags, 'store_tags', order, 'tags')
        _record_calls(monkeypatch, repos.images, 'store_images', order, 'images')
        _record_calls(monkeypatch, jobs, 'enqueue', order, 'job')

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
        _record_calls(monkeypatch, repos.tags, 'replace_tags_for_context', order, 'tags')
        _record_calls(monkeypatch, repos.images, 'replace_images_for_context', order, 'images')
        _record_calls(monkeypatch, jobs, 'delete_for_entry', order, 'job')

        await _update(repos, txn, defer_generation=True, tags=['tag'], images=[])

        assert order == ['tags', 'images', 'job']

    async def test_the_outcome_carries_the_updated_fields_and_the_summary_clear(
        self, repos: MagicMock, txn: MagicMock,
    ) -> None:
        outcome = await _update(repos, txn, defer_generation=True)

        assert outcome == UpdateOutcome(
            updated_fields=['text_content'], summary_cleared=True, generation_status='pending',
        )
