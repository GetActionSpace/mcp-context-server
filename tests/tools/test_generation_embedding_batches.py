"""Batched document embedding through ``generate_embeddings_with_timeout``.

A chunked document is embedded one batch per provider request, each batch under
its own slot of the embedding semaphore and its own total timeout; the error
messages name the batch and the progress made.
"""

import asyncio
import contextlib
from collections.abc import Generator
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools._generation as generation_module
from app.embeddings.batching import AdaptiveBatchSizer
from app.embeddings.batching import BatchSizer
from app.embeddings.batching import FixedBatchSizer
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.services.chunking_service import TextChunk
from app.tools._generation import generate_embeddings_with_timeout


class _PipeChunkingService:
    """Enabled chunking service that splits on ``|`` and keeps exact boundaries."""

    is_enabled = True
    chunk_size = 1500

    def split_text(self, text: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        start = 0
        for index, part in enumerate(text.split('|')):
            chunks.append(TextChunk(text=part, chunk_index=index, start_index=start, end_index=start + len(part)))
            start += len(part) + 1
        return chunks


class _Provider:
    """Records every request; can fail or hang on the batch holding a given chunk."""

    def __init__(self, *, fail_on: str | None = None, hang_on: str | None = None) -> None:
        self.batches: list[list[str]] = []
        self.queries: list[str] = []
        self._fail_on = fail_on
        self._hang_on = hang_on

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self._fail_on in texts:
            raise RuntimeError('model not loaded')
        if self._hang_on in texts:
            await asyncio.Event().wait()
        return [[float(len(text))] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.5]


@contextlib.contextmanager
def _wired(
    provider: object,
    *,
    sizer: BatchSizer | None = None,
    concurrency: int = 3,
    chunking: object | None = None,
) -> Generator[asyncio.Semaphore]:
    semaphore = asyncio.Semaphore(concurrency)
    chunking_service = chunking if chunking is not None else _PipeChunkingService()
    with (
        patch.object(generation_module, 'get_embedding_provider', return_value=provider),
        patch.object(generation_module, 'get_chunking_service', return_value=chunking_service),
        patch.object(generation_module, '_batch_sizer', sizer if sizer is not None else FixedBatchSizer(2)),
        patch.object(generation_module, '_embedding_semaphore', semaphore),
    ):
        yield semaphore


@pytest.mark.asyncio
async def test_document_chunks_are_embedded_batch_by_batch() -> None:
    provider = _Provider()
    with _wired(provider):
        result = await generate_embeddings_with_timeout('a|bb|ccc|dddd|eeeee')

    assert provider.batches == [['a', 'bb'], ['ccc', 'dddd'], ['eeeee']]
    assert result == [
        ChunkEmbedding(embedding=[1.0], start_index=0, end_index=1),
        ChunkEmbedding(embedding=[2.0], start_index=2, end_index=4),
        ChunkEmbedding(embedding=[3.0], start_index=5, end_index=8),
        ChunkEmbedding(embedding=[4.0], start_index=9, end_index=13),
        ChunkEmbedding(embedding=[5.0], start_index=14, end_index=19),
    ]


@pytest.mark.asyncio
async def test_a_failed_batch_is_named_in_the_error() -> None:
    with _wired(_Provider(fail_on='ccc')), pytest.raises(ToolError) as excinfo:
        await generate_embeddings_with_timeout('a|bb|ccc|dddd|eeeee')

    assert excinfo.match(
        r'^Embedding generation failed: model not loaded '
        r'\(batch 2 of 3, chunks 3-4 of 5; 2 chunks embedded in \d+\.\d s\)$',
    )


@pytest.mark.asyncio
async def test_the_total_timeout_applies_per_batch_and_names_the_batch() -> None:
    with (
        _wired(_Provider(hang_on='ccc')),
        patch.object(generation_module, 'compute_embedding_total_timeout', return_value=0.2),
        pytest.raises(ToolError) as excinfo,
    ):
        await generate_embeddings_with_timeout('a|bb|ccc|dddd|eeeee')

    assert excinfo.match(
        r'^Embedding generation exceeded total timeout \(0s\) on batch 2 of 3 '
        r'\(chunks 3-4 of 5; 2 chunks embedded in \d+\.\d s\)\. '
        r'This may indicate the embedding provider is overloaded or unreachable\.$',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(('sizer', 'mode'), [(FixedBatchSizer(2), '2'), (AdaptiveBatchSizer(target_s=5.0), 'auto')])
async def test_document_start_and_completion_are_logged(
    sizer: BatchSizer, mode: str, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level('INFO')
    with _wired(_Provider(), sizer=sizer):
        await generate_embeddings_with_timeout('a|bb|ccc|dddd|eeeee')

    messages = [record.getMessage() for record in caplog.records]
    assert f'Generating embeddings: text_len=19, chunks=5, batch_size={mode}' in messages
    assert any(m.startswith('Embeddings generated: chunks=5, batches=') for m in messages)


@pytest.mark.asyncio
async def test_a_waiting_document_is_served_after_the_current_batch() -> None:
    gate = asyncio.Event()
    first_call_started = asyncio.Event()
    order: list[str] = []

    class GatedProvider:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            order.append(texts[0])
            first_call_started.set()
            await gate.wait()
            return [[0.0] for _ in texts]

    with _wired(GatedProvider(), concurrency=1):
        long_document = asyncio.create_task(generate_embeddings_with_timeout('a0|a1|a2|a3|a4|a5'))
        await first_call_started.wait()
        short_document = asyncio.create_task(generate_embeddings_with_timeout('b0'))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(long_document, short_document)

    assert order == ['a0', 'b0', 'a2', 'a4']


@pytest.mark.asyncio
async def test_single_embedding_path_holds_one_semaphore_slot() -> None:
    provider = _Provider()
    chunking = MagicMock()
    chunking.is_enabled = False
    with _wired(provider, concurrency=1, chunking=chunking) as semaphore:
        await semaphore.acquire()
        task = asyncio.create_task(generate_embeddings_with_timeout('whole text'))
        await asyncio.sleep(0.01)
        assert provider.queries == []
        semaphore.release()
        result = await task

    assert provider.queries == ['whole text']
    assert provider.batches == []
    assert result == [ChunkEmbedding(embedding=[0.5], start_index=0, end_index=10)]


def test_reset_batch_sizer_rebuilds_from_settings() -> None:
    mock_settings = MagicMock()
    try:
        with patch.object(generation_module, 'settings', mock_settings):
            mock_settings.embedding.batch_size = 7
            generation_module._reset_batch_sizer()
            assert generation_module._get_batch_sizer().mode == '7'

            mock_settings.embedding.batch_size = 'auto'
            mock_settings.embedding.batch_target_s = 5.0
            generation_module._reset_batch_sizer()
            assert isinstance(generation_module._get_batch_sizer(), AdaptiveBatchSizer)
    finally:
        generation_module._reset_batch_sizer()
