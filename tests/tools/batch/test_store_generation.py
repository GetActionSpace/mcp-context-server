"""Tests for the generation pass of store_context_batch.

The pass generates each validated entry's embeddings and summary before any transaction,
reuses a likely duplicate's outputs, records or raises a failure by the batch's mode, builds
the node summaries of the surviving entries, and regenerates the outputs of an entry whose
deduplication diverged.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.repositories.context_repository.records import DuplicateCandidate
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools.batch._store_generation import StoreGeneration
from app.tools.batch._store_generation import generate_store_index_nodes
from app.tools.batch._store_generation import generate_store_outputs
from app.tools.batch._store_generation import regenerate_diverged_entries
from app.tools.batch._store_generation import regenerate_diverged_entry
from tests.helpers import LOCAL_WRITE_SCOPE

MODULE = 'app.tools.batch._store_generation'
TEXT = 'An entry long enough to summarize. ' * 4
EMBEDDINGS = [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=10)]
COMPRESSED = [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=10, payload=b'packed')]
NODES = [IndexNodeRow('a', 1, 0, 'A', 'About A.', 0, 10)]


def _entry(index: int, *, text: str = TEXT, source: str = 'agent', **flags: Any) -> dict[str, Any]:
    """A validated entry at ``index`` that deduplicates and generates unless ``flags`` say otherwise."""
    return {
        'index': index,
        'thread_id': 'thread',
        'source': source,
        'text_content': text,
        'deduplicate': flags.get('deduplicate', True),
        'defer_generation': flags.get('defer_generation', False),
    }


def _repos(candidate: DuplicateCandidate | None = None, *, has_embeddings: bool = False) -> MagicMock:
    """Repositories whose pre-check finds ``candidate`` and whose embedding probe answers ``has_embeddings``."""
    repos = MagicMock()
    repos.context.check_latest_is_duplicate = AsyncMock(return_value=candidate)
    repos.embeddings.exists = AsyncMock(return_value=has_embeddings)
    return repos


class Legs:
    """The generation calls of the pass, replaced by recording mocks."""

    def __init__(self) -> None:
        self.embeddings = AsyncMock(return_value=EMBEDDINGS)
        self.summary = AsyncMock(return_value='A summary.')
        self.compression = AsyncMock(return_value=COMPRESSED)
        self.nodes = AsyncMock(return_value=NODES)
        self.active = MagicMock(return_value=True)


@pytest.fixture
def legs() -> Iterator[Legs]:
    """Replace every generation call the pass makes."""
    replaced = Legs()
    with (
        patch(f'{MODULE}.generate_embeddings_with_timeout', replaced.embeddings),
        patch(f'{MODULE}.generate_summary_with_timeout', replaced.summary),
        patch(f'{MODULE}.generate_compression_with_timeout', replaced.compression),
        patch(f'{MODULE}.generate_index_nodes_with_timeout', replaced.nodes),
        patch(f'{MODULE}.generation_active', replaced.active),
    ):
        yield replaced


async def _generate(repos: MagicMock, entries: list[dict[str, Any]], *, atomic: bool = False) -> StoreGeneration:
    """Run the pass with an embedding and a summary provider and no minimum summary length."""
    return await generate_store_outputs(
        repos, entries,
        scope=LOCAL_WRITE_SCOPE,
        embedding_provider=MagicMock(name='embedding'),
        summary_provider=MagicMock(name='summary'),
        min_content_length=0,
        atomic=atomic,
    )


@pytest.mark.asyncio
async def test_generated_outputs_are_compressed_and_counted(legs: Legs) -> None:
    generation = await _generate(_repos(), [_entry(0)])

    assert generation.embeddings == {0: COMPRESSED}
    assert generation.summaries == {0: 'A summary.'}
    assert (generation.embeddings_generated, generation.summaries_generated, generation.summaries_preserved) == (1, 1, 0)
    legs.compression.assert_awaited_once_with(EMBEDDINGS)


@pytest.mark.asyncio
async def test_a_deferring_entry_generates_nothing(legs: Legs) -> None:
    repos = _repos()

    generation = await _generate(repos, [_entry(0, defer_generation=True)])

    assert generation.deferred == {0}
    assert (generation.embeddings, generation.summaries, generation.likely_duplicates) == ({0: None}, {0: None}, {0: None})
    repos.context.check_latest_is_duplicate.assert_not_awaited()
    legs.embeddings.assert_not_awaited()
    legs.summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_entry_asking_to_defer_without_an_active_provider_is_not_deferred(legs: Legs) -> None:
    legs.active.return_value = False

    generation = await _generate(_repos(), [_entry(0, defer_generation=True)])

    assert generation.deferred == set()


@pytest.mark.asyncio
async def test_a_likely_duplicate_reuses_its_summary_and_its_embeddings(legs: Legs) -> None:
    repos = _repos(DuplicateCandidate(context_id='abc', summary='Stored summary.'), has_embeddings=True)

    generation = await _generate(repos, [_entry(0)])

    assert generation.likely_duplicates == {0: 'abc'}
    assert generation.summaries == {0: 'Stored summary.'}
    assert generation.preserved_summaries == {0}
    assert (generation.embeddings_generated, generation.summaries_generated, generation.summaries_preserved) == (0, 0, 1)
    legs.embeddings.assert_not_awaited()
    legs.summary.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.usefixtures('legs')
async def test_an_entry_that_does_not_deduplicate_skips_the_pre_check() -> None:
    repos = _repos(DuplicateCandidate(context_id='abc', summary='Stored summary.'))

    generation = await _generate(repos, [_entry(0, deduplicate=False)])

    repos.context.check_latest_is_duplicate.assert_not_awaited()
    assert generation.likely_duplicates == {0: None}
    assert generation.summaries == {0: 'A summary.'}


@pytest.mark.asyncio
async def test_a_failed_leg_is_recorded_per_entry_and_its_preserved_summary_uncounted(legs: Legs) -> None:
    legs.embeddings.side_effect = RuntimeError('provider down')
    repos = _repos(DuplicateCandidate(context_id='abc', summary='Stored summary.'))

    generation = await _generate(repos, [_entry(4)])

    assert generation.errors == [(4, 'Generation failed: embedding: RuntimeError: provider down')]
    assert generation.summaries_preserved == 0
    assert generation.preserved_summaries == set()


@pytest.mark.asyncio
async def test_a_failed_leg_aborts_an_atomic_batch(legs: Legs) -> None:
    legs.summary.side_effect = RuntimeError('model gone')

    with pytest.raises(ToolError, match=r'Generation failed at index 2 after exhausting configured retries: '
                                        r'summary: RuntimeError: model gone\. No data was saved\.'):
        await _generate(_repos(), [_entry(2)], atomic=True)


@pytest.mark.asyncio
async def test_a_compression_failure_discards_the_entry_outputs(legs: Legs) -> None:
    legs.compression.side_effect = ToolError('codebook mismatch')

    generation = await _generate(_repos(), [_entry(1)])

    assert generation.errors == [(1, 'Compression failed: codebook mismatch')]
    assert (generation.embeddings, generation.summaries) == ({0: None}, {0: None})
    assert (generation.embeddings_generated, generation.summaries_generated) == (0, 0)


@pytest.mark.asyncio
async def test_a_compression_failure_aborts_an_atomic_batch(legs: Legs) -> None:
    legs.compression.side_effect = ToolError('codebook mismatch')

    with pytest.raises(ToolError, match=r'Compression failed at index 1: codebook mismatch\. No data was saved\.'):
        await _generate(_repos(), [_entry(1)], atomic=True)


@pytest.mark.asyncio
async def test_a_short_text_gets_no_summary(legs: Legs) -> None:
    generation = await generate_store_outputs(
        _repos(), [_entry(0, text='short')],
        scope=LOCAL_WRITE_SCOPE,
        embedding_provider=None,
        summary_provider=MagicMock(name='summary'),
        min_content_length=100,
        atomic=False,
    )

    assert generation.summaries == {0: None}
    legs.summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_node_summaries_skip_deferring_entries_and_likely_duplicates(legs: Legs) -> None:
    generation = StoreGeneration(deferred={1}, likely_duplicates={0: None, 1: None, 2: 'abc'})

    await generate_store_index_nodes(generation, [(0, _entry(0)), (1, _entry(1)), (2, _entry(2))])

    assert generation.index_nodes == {0: NODES, 1: None, 2: None}
    legs.nodes.assert_awaited_once_with(TEXT)


def test_discarding_an_entry_reverses_each_count_it_added() -> None:
    generation = StoreGeneration(
        deferred=set(),
        embeddings={0: COMPRESSED, 1: None, 2: None},
        summaries={0: 'Generated.', 1: 'Reused.', 2: ''},
        preserved_summaries={1},
        embeddings_generated=1,
        summaries_generated=1,
        summaries_preserved=1,
    )

    for ve_idx in (0, 1, 2):
        generation.discard_counts(ve_idx)

    assert (generation.embeddings_generated, generation.summaries_generated, generation.summaries_preserved) == (0, 0, 0)
    assert generation.preserved_summaries == set()


@pytest.mark.asyncio
async def test_a_diverged_batch_regenerates_once_per_text_and_source(legs: Legs) -> None:
    other = 'Different text.'
    entries = [(0, _entry(0)), (1, _entry(1, source='user')), (2, _entry(2)), (3, _entry(3, text=other))]
    generation = StoreGeneration(
        deferred=set(),
        embeddings={0: None, 1: None, 2: COMPRESSED, 3: None},
        summaries={0: 'Reused.', 1: 'Reused.', 2: 'Reused.', 3: None},
        preserved_summaries={0, 1, 2},
        index_nodes={0: None, 1: None, 2: None, 3: None},
        summaries_preserved=3,
    )
    legs.nodes.return_value = None

    await regenerate_diverged_entries(
        generation, entries, TEXT, embedding_provider=MagicMock(), summary_provider=MagicMock(),
    )

    legs.embeddings.assert_awaited_once_with(TEXT)
    assert generation.embeddings == {0: COMPRESSED, 1: COMPRESSED, 2: COMPRESSED, 3: None}
    assert generation.embeddings_generated == 2
    assert [call.args for call in legs.summary.await_args_list] == [(TEXT, 'agent'), (TEXT, 'user')]
    assert (generation.preserved_summaries, generation.summaries_preserved, generation.summaries_generated) == (set(), 0, 3)
    assert generation.index_nodes == {0: [], 1: [], 2: [], 3: None}


@pytest.mark.asyncio
async def test_a_diverged_batch_with_embeddings_present_calls_no_embedding_provider(legs: Legs) -> None:
    generation = StoreGeneration(deferred=set(), embeddings={0: COMPRESSED}, index_nodes={0: None})

    await regenerate_diverged_entries(
        generation, [(0, _entry(0))], TEXT, embedding_provider=MagicMock(), summary_provider=None,
    )

    legs.embeddings.assert_not_awaited()
    assert generation.index_nodes == {0: NODES}


@pytest.mark.asyncio
@pytest.mark.usefixtures('legs')
async def test_a_diverged_entry_regenerates_its_outputs() -> None:
    generation = StoreGeneration(
        deferred=set(), embeddings={0: None}, summaries={0: 'Reused.'}, preserved_summaries={0},
        index_nodes={0: None}, summaries_preserved=1,
    )

    error = await regenerate_diverged_entry(
        generation, 0, _entry(0), TEXT, embedding_provider=MagicMock(), summary_provider=MagicMock(),
    )

    assert error is None
    assert (generation.embeddings, generation.summaries, generation.index_nodes) == (
        {0: COMPRESSED}, {0: 'A summary.'}, {0: NODES},
    )
    assert (generation.embeddings_generated, generation.summaries_generated, generation.summaries_preserved) == (1, 1, 0)


@pytest.mark.asyncio
async def test_a_diverged_entry_whose_embeddings_fail_reports_and_uncounts(legs: Legs) -> None:
    legs.embeddings.side_effect = ToolError('provider down')
    generation = StoreGeneration(
        deferred=set(), embeddings={0: None}, summaries={0: 'Reused.'}, preserved_summaries={0}, summaries_preserved=1,
    )

    error = await regenerate_diverged_entry(
        generation, 0, _entry(0), TEXT, embedding_provider=MagicMock(), summary_provider=MagicMock(),
    )

    assert error == 'provider down'
    assert (generation.summaries_preserved, generation.preserved_summaries) == (0, set())
    legs.summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_diverged_entry_whose_summary_fails_reports_and_uncounts(legs: Legs) -> None:
    legs.summary.side_effect = ToolError('model gone')
    generation = StoreGeneration(
        deferred=set(), embeddings={0: None}, summaries={0: 'Reused.'}, preserved_summaries={0}, summaries_preserved=1,
    )

    error = await regenerate_diverged_entry(
        generation, 0, _entry(0), TEXT, embedding_provider=MagicMock(), summary_provider=MagicMock(),
    )

    assert error == 'model gone'
    assert (generation.embeddings_generated, generation.summaries_preserved) == (0, 0)
    legs.nodes.assert_not_awaited()
