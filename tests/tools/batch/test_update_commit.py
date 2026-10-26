"""Tests for the transaction pass of update_context_batch.

An atomic batch applies every update in one transaction against the versions captured before
generation, aborting on a concurrent modification or a vanished entry and retrying a lost
connection; a non-atomic batch gives each update its own transaction, re-reads the version
after a conflict, and records a failure for that update alone.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.repositories.context_repository.records import EntryProbe
from app.repositories.context_repository.records import VersionConflictError
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.tools._transactions import EntryNotFoundError
from app.tools._update_transaction import UpdateOutcome
from app.tools.batch._update_commit import UpdateCommit
from app.tools.batch._update_commit import commit_updates_atomically
from app.tools.batch._update_commit import commit_updates_individually
from app.tools.batch._update_generation import UpdateGeneration
from app.types import GenerationStatus
from tests.helpers import LOCAL_WRITE_SCOPE
from tests.tools.batch._mocks import create_mock_repositories

MODULE = 'app.tools.batch._update_commit'
COMPRESSED = [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=10, payload=b'packed')]


def _update(position: int, context_id: str, **fields: Any) -> tuple[int, dict[str, Any]]:
    """A validated update of ``context_id`` at its position, original index ten times the position."""
    return position, {'index': position * 10, 'context_id': context_id, **fields}


def _outcome(*, cleared: bool = False, status: GenerationStatus = 'ready') -> UpdateOutcome:
    """The outcome of one applied update."""
    return UpdateOutcome(['text_content'], summary_cleared=cleared, generation_status=status)


def _probe(*, exists: bool = True, can_write: bool = True, version: int | None = 7) -> EntryProbe:
    """The re-read of an entry after a failed compare-and-set."""
    return EntryProbe(exists, 'agent' if exists else None, version if exists else None, 'local', can_write)


@pytest.fixture
def execute() -> Iterator[AsyncMock]:
    """Replace the per-update transaction; every update applies and clears no summary."""
    calls = AsyncMock(return_value=_outcome())
    with patch(f'{MODULE}.execute_update_in_transaction', calls):
        yield calls


async def _atomically(
    generation: UpdateGeneration, updates: list[tuple[int, dict[str, Any]]], versions: dict[str, int],
) -> UpdateCommit:
    """Apply ``updates`` in one transaction."""
    return await commit_updates_atomically(
        create_mock_repositories(), updates, generation, versions, scope=LOCAL_WRITE_SCOPE, embedding_model='model',
    )


async def _individually(
    generation: UpdateGeneration, updates: list[tuple[int, dict[str, Any]]], versions: dict[str, int],
) -> UpdateCommit:
    """Apply each of ``updates`` in its own transaction."""
    return await commit_updates_individually(
        create_mock_repositories(), updates, generation, versions, scope=LOCAL_WRITE_SCOPE, embedding_model='model',
    )


@pytest.mark.asyncio
async def test_an_atomic_commit_applies_every_update_with_its_outputs(execute: AsyncMock) -> None:
    execute.side_effect = [_outcome(cleared=True), _outcome(status='pending')]
    generation = UpdateGeneration(
        deferred={1}, embeddings={0: COMPRESSED}, summaries={0: 'A summary.'}, clear_summaries={1}, index_nodes={0: []},
    )
    updates = [_update(0, 'ctx-a', text='New text.', tags=['x']), _update(1, 'ctx-b', text='Deferred text.')]

    commit = await _atomically(generation, updates, {'ctx-a': 3, 'ctx-b': 5})

    assert [(item['index'], item['context_id'], item['generation_status']) for item in commit.results] == [
        (0, 'ctx-a', 'ready'), (10, 'ctx-b', 'pending'),
    ]
    assert (commit.summaries_cleared, commit.deferred_committed) == (1, True)
    first, second = (call.kwargs for call in execute.await_args_list)
    assert (first['chunk_embeddings'], first['summary'], first['index_nodes'], first['tags']) == (
        COMPRESSED, 'A summary.', [], ['x'],
    )
    assert (first['expected_version'], first['clear_summary'], first['defer_generation']) == (3, False, False)
    assert (second['expected_version'], second['clear_summary'], second['defer_generation']) == (5, True, True)
    assert second['embedding_model'] == 'model'


@pytest.mark.asyncio
async def test_an_atomic_commit_presents_the_bumped_version_to_a_later_update_of_the_same_entry(
    execute: AsyncMock,
) -> None:
    updates = [
        _update(0, 'ctx-a', text='First.'),
        _update(1, 'ctx-a', tags=['only-tags']),
        _update(2, 'ctx-a', visibility='public'),
        _update(3, 'ctx-a', metadata={'k': 'v'}),
    ]

    await _atomically(UpdateGeneration(deferred=set()), updates, {'ctx-a': 3})

    assert [call.kwargs['expected_version'] for call in execute.await_args_list] == [3, 4, 4, 5]


@pytest.mark.asyncio
async def test_an_atomic_commit_aborts_on_a_concurrent_modification(execute: AsyncMock) -> None:
    execute.side_effect = VersionConflictError('ctx-a')

    with (
        patch(f'{MODULE}.reraise_disambiguated_cas_conflict', AsyncMock(side_effect=VersionConflictError('ctx-a'))) as
        disambiguate,
        pytest.raises(ToolError, match=r'Concurrent modification during atomic batch update: .*Retry the request\.'),
    ):
        await _atomically(UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.')], {'ctx-a': 3})

    disambiguate.assert_awaited_once()
    assert disambiguate.await_args is not None
    assert disambiguate.await_args.args[2] == 'ctx-a'


@pytest.mark.asyncio
async def test_an_atomic_commit_aborts_when_an_entry_vanished(execute: AsyncMock) -> None:
    execute.side_effect = EntryNotFoundError('ctx-a')

    with pytest.raises(ToolError, match=r'No entries were updated \(atomic batch\)\.'):
        await _atomically(UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.')], {'ctx-a': 3})


@pytest.mark.asyncio
async def test_an_atomic_commit_retries_a_lost_connection_twice_from_the_captured_versions(execute: AsyncMock) -> None:
    execute.side_effect = [_outcome(), ConnectionResetError('reset'), _outcome(), _outcome()]
    updates = [_update(0, 'ctx-a', text='First.'), _update(1, 'ctx-a', text='Second.')]

    with patch(f'{MODULE}.asyncio.sleep', AsyncMock()) as sleep:
        commit = await _atomically(UpdateGeneration(deferred=set()), updates, {'ctx-a': 3})

    assert [call.args for call in sleep.await_args_list] == [(0.5,)]
    assert [call.kwargs['expected_version'] for call in execute.await_args_list] == [3, 4, 3, 4]
    assert len(commit.results) == 2


@pytest.mark.asyncio
async def test_an_atomic_commit_gives_up_after_two_connection_retries(execute: AsyncMock) -> None:
    execute.side_effect = ConnectionResetError('reset')

    with patch(f'{MODULE}.asyncio.sleep', AsyncMock()) as sleep, pytest.raises(ConnectionResetError):
        await _atomically(UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.')], {'ctx-a': 3})

    assert [call.args for call in sleep.await_args_list] == [(0.5,), (1.0,)]
    assert execute.await_count == 3


@pytest.mark.asyncio
async def test_an_individual_commit_fails_only_the_update_that_failed(execute: AsyncMock) -> None:
    execute.side_effect = [ToolError('refused'), _outcome(cleared=True, status='pending')]
    generation = UpdateGeneration(
        deferred={1}, embeddings={0: COMPRESSED}, summaries={0: 'A summary.'}, embeddings_generated=1, summaries_generated=1,
    )

    commit = await _individually(generation, [_update(0, 'ctx-a', text='New.'), _update(1, 'ctx-b', text='Later.')], {})

    assert [(item['index'], item['success'], item['error']) for item in commit.results] == [
        (0, False, 'refused'), (10, True, None),
    ]
    assert (commit.summaries_cleared, commit.deferred_committed) == (1, True)
    assert (generation.embeddings_generated, generation.summaries_generated) == (0, 0)


@pytest.mark.asyncio
async def test_an_individual_commit_wakes_no_worker_when_no_deferring_update_committed(execute: AsyncMock) -> None:
    execute.side_effect = ToolError('refused')

    commit = await _individually(UpdateGeneration(deferred={0}), [_update(0, 'ctx-a', text='New.')], {})

    assert commit.deferred_committed is False


@pytest.mark.asyncio
async def test_an_individual_commit_carries_the_bumped_version_across_updates(execute: AsyncMock) -> None:
    updates = [_update(0, 'ctx-a', text='First.'), _update(1, 'ctx-a', images=[]), _update(2, 'ctx-a', text='Third.')]

    await _individually(UpdateGeneration(deferred=set()), updates, {'ctx-a': 3})

    assert [call.kwargs['expected_version'] for call in execute.await_args_list] == [3, 4, 4]


@pytest.mark.asyncio
async def test_an_individual_commit_retries_a_conflict_with_the_reread_version(execute: AsyncMock) -> None:
    execute.side_effect = [VersionConflictError('ctx-a'), _outcome()]

    with patch(f'{MODULE}.reread_entry_version', AsyncMock(return_value=_probe(version=9))):
        commit = await _individually(UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.')], {'ctx-a': 3})

    assert [call.kwargs['expected_version'] for call in execute.await_args_list] == [3, 9]
    assert [item['success'] for item in commit.results] == [True]


@pytest.mark.asyncio
async def test_an_individual_commit_bounds_its_conflict_retries(execute: AsyncMock) -> None:
    execute.side_effect = VersionConflictError('ctx-a')
    generation = UpdateGeneration(deferred=set(), embeddings={0: COMPRESSED}, embeddings_generated=1)

    with patch(f'{MODULE}.reread_entry_version', AsyncMock(return_value=_probe())) as reread:
        commit = await _individually(generation, [_update(0, 'ctx-a', text='New.')], {'ctx-a': 3})

    assert reread.await_count == 5
    assert execute.await_count == 6
    assert [(item['success'], item['error']) for item in commit.results] == [
        (False, 'Version conflict updating context ctx-a'),
    ]
    assert generation.embeddings_generated == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('probe', 'error'),
    [
        (_probe(exists=False, can_write=False), 'Context entry ctx-a not found'),
        (_probe(can_write=False), 'Not authorized to modify context entry ctx-a'),
    ],
)
async def test_an_individual_commit_stops_on_an_entry_gone_or_no_longer_writable(
    execute: AsyncMock, probe: EntryProbe, error: str,
) -> None:
    execute.side_effect = VersionConflictError('ctx-a')

    with patch(f'{MODULE}.reread_entry_version', AsyncMock(return_value=probe)):
        commit = await _individually(UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.')], {'ctx-a': 3})

    assert [(item['success'], item['error']) for item in commit.results] == [(False, error)]
    assert execute.await_count == 1


@pytest.mark.asyncio
async def test_an_individual_commit_records_a_failed_reread(execute: AsyncMock) -> None:
    execute.side_effect = VersionConflictError('ctx-a')

    with patch(f'{MODULE}.reread_entry_version', AsyncMock(side_effect=RuntimeError('database gone'))):
        commit = await _individually(UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.')], {'ctx-a': 3})

    assert [(item['success'], item['error']) for item in commit.results] == [(False, 'database gone')]


@pytest.mark.asyncio
async def test_an_individual_commit_records_a_vanished_entry(execute: AsyncMock) -> None:
    execute.side_effect = [EntryNotFoundError('ctx-a'), _outcome()]

    commit = await _individually(
        UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.'), _update(1, 'ctx-b', text='Other.')], {},
    )

    assert [(item['context_id'], item['success'], item['error']) for item in commit.results] == [
        ('ctx-a', False, 'Context entry with ID ctx-a not found'), ('ctx-b', True, None),
    ]


@pytest.mark.asyncio
async def test_an_individual_commit_records_a_connection_failure_after_its_retries(execute: AsyncMock) -> None:
    execute.side_effect = ConnectionResetError('reset')

    with patch(f'{MODULE}.asyncio.sleep', AsyncMock()) as sleep:
        commit = await _individually(UpdateGeneration(deferred=set()), [_update(0, 'ctx-a', text='New.')], {})

    assert [(item['success'], item['error']) for item in commit.results] == [(False, 'reset')]
    assert [call.args for call in sleep.await_args_list] == [(0.5,), (1.0,)]
    assert execute.await_count == 3
