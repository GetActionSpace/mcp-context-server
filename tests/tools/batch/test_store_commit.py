"""Tests for the transaction pass of store_context_batch.

An atomic batch stores every entry in one transaction, regenerating and re-running it when a
deduplication diverges and retrying a lost connection; a non-atomic batch gives each entry its
own transaction and records a failure for that entry alone.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.tools._transactions import EmbeddingsReconcileRequiredError
from app.tools._transactions import StoreOutcome
from app.tools.batch._store_commit import StoreCommit
from app.tools.batch._store_commit import commit_stores_atomically
from app.tools.batch._store_commit import commit_stores_individually
from app.tools.batch._store_generation import StoreGeneration
from app.types import GenerationStatus
from tests.helpers import LOCAL_WRITE_SCOPE
from tests.tools.batch._mocks import create_mock_repositories

MODULE = 'app.tools.batch._store_commit'
TEXT = 'An entry of the batch.'
COMPRESSED = [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=10, payload=b'packed')]


def _entries(count: int) -> list[tuple[int, dict[str, Any]]]:
    """``count`` validated entries at their positions, each at the original index ten times its position."""
    return [
        (ve_idx, {
            'index': ve_idx * 10,
            'thread_id': 'thread',
            'source': 'agent',
            'content_type': 'text',
            'text_content': f'{TEXT} {ve_idx}',
            'metadata': None,
            'tags': None,
            'images': [],
            'images_provided': False,
            'visibility': 'private',
            'deduplicate': True,
            'defer_generation': False,
        })
        for ve_idx in range(count)
    ]


def _outcome(context_id: str, *, embedding_stored: bool = True, status: GenerationStatus = 'ready') -> StoreOutcome:
    """The outcome of one stored entry."""
    return StoreOutcome(context_id, was_updated=False, embedding_stored=embedding_stored, generation_status=status)


@pytest.fixture
def execute() -> Iterator[AsyncMock]:
    """Replace the per-entry store transaction; every entry stores under the id of its call order."""
    calls = AsyncMock()
    calls.side_effect = lambda *_args, **kwargs: _outcome(f'id-{kwargs["text_content"][-1]}')
    with patch(f'{MODULE}.execute_store_in_transaction', calls):
        yield calls


async def _atomically(generation: StoreGeneration, entries: list[tuple[int, dict[str, Any]]]) -> StoreCommit:
    """Commit ``entries`` in one transaction with an embedding and a summary provider."""
    return await commit_stores_atomically(
        create_mock_repositories(), entries, generation,
        scope=LOCAL_WRITE_SCOPE,
        author_group_grants=frozenset(),
        embedding_provider=MagicMock(name='embedding'),
        summary_provider=MagicMock(name='summary'),
        embedding_model='model',
    )


async def _individually(generation: StoreGeneration, entries: list[tuple[int, dict[str, Any]]]) -> StoreCommit:
    """Commit each of ``entries`` in its own transaction with an embedding and a summary provider."""
    return await commit_stores_individually(
        create_mock_repositories(), entries, generation,
        scope=LOCAL_WRITE_SCOPE,
        author_group_grants=frozenset(),
        embedding_provider=MagicMock(name='embedding'),
        summary_provider=MagicMock(name='summary'),
        embedding_model='model',
    )


@pytest.mark.asyncio
async def test_an_atomic_commit_stores_every_entry_with_its_outputs(execute: AsyncMock) -> None:
    generation = StoreGeneration(
        deferred={1}, embeddings={0: COMPRESSED}, summaries={0: 'A summary.'}, likely_duplicates={1: 'abc'},
        preserved_summaries={1}, index_nodes={0: []},
    )

    commit = await _atomically(generation, _entries(2))

    assert [(item['index'], item['context_id'], item['success']) for item in commit.results] == [
        (0, 'id-0', True), (10, 'id-1', True),
    ]
    assert (commit.embeddings_stored, commit.deferred_committed) == (2, True)
    first, second = (call.kwargs for call in execute.await_args_list)
    assert (first['chunk_embeddings'], first['summary'], first['index_nodes'], first['defer_generation']) == (
        COMPRESSED, 'A summary.', [], False,
    )
    assert (second['defer_generation'], second['summary_pending'], second['embedding_model']) == (True, True, 'model')


@pytest.mark.asyncio
async def test_an_atomic_commit_regenerates_and_reruns_a_diverged_deduplication(execute: AsyncMock) -> None:
    outcomes: list[object] = [EmbeddingsReconcileRequiredError('diverged'), _outcome('id-0'), _outcome('id-1')]
    execute.side_effect = outcomes
    entries = _entries(2)
    generation = StoreGeneration(deferred=set())

    with patch(f'{MODULE}.regenerate_diverged_entries', AsyncMock()) as regenerate:
        commit = await _atomically(generation, entries)

    regenerate.assert_awaited_once()
    assert regenerate.await_args is not None
    assert regenerate.await_args.args == (generation, entries, 'diverged')
    assert [item['context_id'] for item in commit.results] == ['id-0', 'id-1']


@pytest.mark.asyncio
async def test_an_atomic_commit_bounds_its_reconcile_passes_by_the_entry_count(execute: AsyncMock) -> None:
    execute.side_effect = EmbeddingsReconcileRequiredError('diverged')

    with (
        patch(f'{MODULE}.regenerate_diverged_entries', AsyncMock()) as regenerate,
        pytest.raises(ToolError, match=r'after deduplication divergence \(atomic batch store\)'),
    ):
        await _atomically(StoreGeneration(deferred=set()), _entries(2))

    assert regenerate.await_count == 2


@pytest.mark.asyncio
async def test_an_atomic_commit_retries_a_lost_connection_twice(execute: AsyncMock) -> None:
    execute.side_effect = ConnectionResetError('reset')

    with patch(f'{MODULE}.asyncio.sleep', AsyncMock()) as sleep, pytest.raises(ConnectionResetError):
        await _atomically(StoreGeneration(deferred=set()), _entries(1))

    assert [call.args for call in sleep.await_args_list] == [(0.5,), (1.0,)]
    assert execute.await_count == 3


@pytest.mark.asyncio
async def test_an_individual_commit_fails_only_the_entry_that_failed(execute: AsyncMock) -> None:
    outcomes: list[object] = [ToolError('refused'), _outcome('id-1', status='pending')]
    execute.side_effect = outcomes
    generation = StoreGeneration(
        deferred={1}, embeddings={0: COMPRESSED}, summaries={0: 'A summary.'}, embeddings_generated=1, summaries_generated=1,
    )

    commit = await _individually(generation, _entries(2))

    assert [(item['index'], item['success'], item['error']) for item in commit.results] == [
        (0, False, 'refused'), (10, True, None),
    ]
    assert (commit.embeddings_stored, commit.deferred_committed) == (1, True)
    assert (generation.embeddings_generated, generation.summaries_generated) == (0, 0)


@pytest.mark.asyncio
async def test_an_individual_commit_wakes_no_worker_when_no_deferring_entry_committed(execute: AsyncMock) -> None:
    execute.side_effect = ToolError('refused')

    commit = await _individually(StoreGeneration(deferred={0}), _entries(1))

    assert commit.deferred_committed is False


@pytest.mark.asyncio
async def test_an_individual_commit_reconciles_a_diverged_entry_once(execute: AsyncMock) -> None:
    execute.side_effect = EmbeddingsReconcileRequiredError('diverged')

    with patch(f'{MODULE}.regenerate_diverged_entry', AsyncMock(return_value=None)) as regenerate:
        commit = await _individually(StoreGeneration(deferred=set()), _entries(1))

    regenerate.assert_awaited_once()
    assert [item['error'] for item in commit.results] == ['Failed to reconcile embeddings after deduplication divergence']


@pytest.mark.asyncio
async def test_an_individual_commit_reports_a_failed_regeneration(execute: AsyncMock) -> None:
    execute.side_effect = EmbeddingsReconcileRequiredError('diverged')

    with patch(f'{MODULE}.regenerate_diverged_entry', AsyncMock(return_value='provider down')):
        commit = await _individually(StoreGeneration(deferred=set()), _entries(1))

    assert [(item['success'], item['error']) for item in commit.results] == [(False, 'provider down')]
    assert execute.await_count == 1


@pytest.mark.asyncio
async def test_an_individual_commit_records_a_connection_failure_after_its_retries(execute: AsyncMock) -> None:
    execute.side_effect = ConnectionResetError('reset')

    with patch(f'{MODULE}.asyncio.sleep', AsyncMock()):
        commit = await _individually(StoreGeneration(deferred=set()), _entries(1))

    assert [(item['success'], item['error']) for item in commit.results] == [(False, 'reset')]
    assert execute.await_count == 3
