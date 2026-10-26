"""Tests for the generation pass of update_context_batch.

The pass generates the embeddings and summary of each text change before any transaction,
marks the summaries the new text leaves stale for clearing, records or raises a failure by the
batch's mode, and rebuilds the node summaries of the surviving text changes.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools.batch._update_generation import UpdateGeneration
from app.tools.batch._update_generation import generate_update_index_nodes
from app.tools.batch._update_generation import generate_update_outputs

MODULE = 'app.tools.batch._update_generation'
TEXT = 'A replacement text long enough to summarize. ' * 4
EMBEDDINGS = [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=10)]
COMPRESSED = [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=10, payload=b'packed')]
NODES = [IndexNodeRow('a', 1, 0, 'A', 'About A.', 0, 10)]
SOURCES = {'ctx-0': 'agent', 'ctx-1': 'user', 'ctx-2': 'agent'}


def _update(position: int, *, text: str | None = TEXT, defer_generation: bool = False) -> tuple[int, dict[str, Any]]:
    """A validated update of ``ctx-<position>`` at its position, original index ten times the position."""
    return position, {
        'index': position * 10,
        'context_id': f'ctx-{position}',
        'text': text,
        'defer_generation': defer_generation,
    }


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


async def _generate(
    updates: list[tuple[int, dict[str, Any]]],
    *,
    atomic: bool = False,
    min_content_length: int = 0,
) -> UpdateGeneration:
    """Run the pass with an embedding and a summary provider."""
    return await generate_update_outputs(
        updates, SOURCES,
        embedding_provider=MagicMock(name='embedding'),
        summary_provider=MagicMock(name='summary'),
        min_content_length=min_content_length,
        atomic=atomic,
    )


@pytest.mark.asyncio
async def test_a_text_change_gets_compressed_embeddings_and_a_summary_in_its_source(legs: Legs) -> None:
    generation = await _generate([_update(1)])

    assert generation.embeddings == {1: COMPRESSED}
    assert generation.summaries == {1: 'A summary.'}
    assert generation.clear_summaries == set()
    assert (generation.embeddings_generated, generation.summaries_generated) == (1, 1)
    legs.summary.assert_awaited_once_with(TEXT, 'user')
    legs.compression.assert_awaited_once_with(EMBEDDINGS)


@pytest.mark.asyncio
async def test_an_update_without_text_generates_nothing(legs: Legs) -> None:
    generation = await _generate([_update(0, text=None)])

    assert (generation.embeddings, generation.summaries) == ({0: None}, {0: None})
    assert generation.deferred == set()
    legs.embeddings.assert_not_awaited()
    legs.summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_deferring_text_change_generates_nothing_and_clears_its_summary(legs: Legs) -> None:
    generation = await _generate([_update(0, defer_generation=True)])

    assert generation.deferred == {0}
    assert (generation.embeddings, generation.summaries, generation.clear_summaries) == ({0: None}, {0: None}, {0})
    legs.embeddings.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.usefixtures('legs')
async def test_a_deferring_update_without_text_is_not_deferred() -> None:
    generation = await _generate([_update(0, text=None, defer_generation=True)])

    assert generation.deferred == set()


@pytest.mark.asyncio
async def test_a_text_change_asking_to_defer_without_an_active_provider_is_not_deferred(legs: Legs) -> None:
    legs.active.return_value = False

    generation = await _generate([_update(0, defer_generation=True)])

    assert generation.deferred == set()
    legs.embeddings.assert_awaited_once_with(TEXT)


@pytest.mark.asyncio
async def test_a_short_text_clears_the_stored_summary(legs: Legs) -> None:
    generation = await _generate([_update(0, text='short')], min_content_length=100)

    assert generation.summaries == {0: None}
    assert generation.clear_summaries == {0}
    legs.summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_a_summary_provider_the_stored_summary_is_cleared(legs: Legs) -> None:
    generation = await generate_update_outputs(
        [_update(0)], SOURCES, embedding_provider=None, summary_provider=None, min_content_length=0, atomic=False,
    )

    assert generation.clear_summaries == {0}
    assert generation.embeddings == {0: None}
    legs.compression.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_empty_generated_summary_clears_the_stored_summary(legs: Legs) -> None:
    legs.summary.return_value = None

    generation = await _generate([_update(0)])

    assert generation.clear_summaries == {0}
    assert generation.summaries_generated == 0


@pytest.mark.asyncio
async def test_a_failed_leg_is_recorded_per_update(legs: Legs) -> None:
    legs.embeddings.side_effect = RuntimeError('provider down')

    generation = await _generate([_update(2)])

    assert generation.errors == [(20, 'ctx-2', 'Generation failed: embedding: RuntimeError: provider down')]
    assert (generation.embeddings_generated, generation.summaries_generated) == (0, 0)
    legs.compression.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_summary_leaves_the_stored_summary_unmarked(legs: Legs) -> None:
    legs.summary.side_effect = RuntimeError('model gone')

    generation = await _generate([_update(0)])

    assert generation.clear_summaries == set()
    assert generation.errors == [(0, 'ctx-0', 'Generation failed: summary: RuntimeError: model gone')]


@pytest.mark.asyncio
async def test_a_failed_leg_aborts_an_atomic_batch(legs: Legs) -> None:
    legs.summary.side_effect = RuntimeError('model gone')

    with pytest.raises(ToolError, match=r'Generation failed for context ctx-1 at index 10 after exhausting configured '
                                        r'retries: summary: RuntimeError: model gone\. No data was modified\.'):
        await _generate([_update(1)], atomic=True)


@pytest.mark.asyncio
async def test_a_compression_failure_discards_the_update_outputs(legs: Legs) -> None:
    legs.compression.side_effect = ToolError('codebook mismatch')

    generation = await _generate([_update(1)])

    assert generation.errors == [(10, 'ctx-1', 'Compression failed: codebook mismatch')]
    assert (generation.embeddings, generation.summaries) == ({1: None}, {1: None})
    assert (generation.embeddings_generated, generation.summaries_generated) == (0, 0)


@pytest.mark.asyncio
async def test_a_compression_failure_aborts_an_atomic_batch(legs: Legs) -> None:
    legs.compression.side_effect = ToolError('codebook mismatch')

    with pytest.raises(ToolError, match=r'Compression failed for context ctx-1 at index 10: codebook mismatch\. '
                                        r'No data was modified\.'):
        await _generate([_update(1)], atomic=True)


@pytest.mark.asyncio
async def test_node_summaries_are_rebuilt_for_text_changes_only(legs: Legs) -> None:
    generation = UpdateGeneration(deferred={1})

    await generate_update_index_nodes(generation, [_update(0), _update(1), _update(2, text=None)])

    assert generation.index_nodes == {0: NODES, 1: []}
    legs.nodes.assert_awaited_once_with(TEXT)


@pytest.mark.asyncio
async def test_node_summaries_that_could_not_be_built_clear_the_stored_rows(legs: Legs) -> None:
    legs.nodes.return_value = None
    generation = UpdateGeneration(deferred=set())

    await generate_update_index_nodes(generation, [_update(0)])

    assert generation.index_nodes == {0: []}


def test_discarding_an_update_reverses_each_count_it_added() -> None:
    generation = UpdateGeneration(
        deferred=set(),
        embeddings={0: COMPRESSED, 1: None, 2: None},
        summaries={0: 'Generated.', 1: '', 2: None},
        embeddings_generated=1,
        summaries_generated=1,
    )

    for vu_idx in (0, 1, 2):
        generation.discard_counts(vu_idx)

    assert (generation.embeddings_generated, generation.summaries_generated) == (0, 0)
