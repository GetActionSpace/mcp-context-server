"""The store/update embedding leg offloads a splitting entry's chunking off the loop.

``generate_embeddings_with_timeout`` calls ``ChunkingService.split_text`` ->
``RecursiveCharacterTextSplitter.create_documents``, which is pure CPU over
UNBOUNDED stored entry text and runs on the embedding leg of ``run_generation``
(the store/update/batch write path). The recursive split costs roughly a
microsecond per character, so a plain one-megabyte document takes the better part
of a second -- far too long to run inline on the single event loop, where it would
starve every concurrent MCP request.

The cost switch is the splitter's own fast path: at or below ``chunk_size``
``split_text`` returns one chunk immediately, and above it the recursive split runs
in full. The offload gate therefore keys on ``chunk_size``, not on a large fixed
character threshold, which would leave exactly the expensive plain-text case inline.
Mirrors the read-path (navigation/test_offload.py::TestLargeEntryOffloadNonBlocking),
the index_tree node leg (test_index_node_generation.py::TestLargeEntryWritePathOffloadNonBlocking),
and the grep matcher (test_grep_matcher.py).
"""

import threading
from unittest.mock import patch

import pytest

import app.tools._generation as generation_module
from app.services.chunking_service import TextChunk
from app.tools._generation import generate_embeddings_with_timeout


class _FakeEmbeddingProvider:
    """Minimal embedding provider returning a fixed vector per text."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        _ = text
        return [0.1, 0.2, 0.3]


class _SpyChunkingService:
    """Enabled chunking service whose split_text records its executing thread."""

    is_enabled = True
    chunk_size = 1500

    def __init__(self) -> None:
        self.on_main: bool | None = None

    def split_text(self, text: str) -> list[TextChunk]:
        self.on_main = threading.current_thread() is threading.main_thread()
        return [TextChunk(text=text, chunk_index=0, start_index=0, end_index=len(text))]


@pytest.mark.asyncio
async def test_splitting_entry_chunking_offloaded() -> None:
    spy = _SpyChunkingService()
    big = 'a' * (spy.chunk_size + 10)  # long enough that the recursive split runs
    with (
        patch.object(generation_module, 'get_embedding_provider', lambda: _FakeEmbeddingProvider()),
        patch.object(generation_module, 'get_chunking_service', lambda: spy),
    ):
        result = await generate_embeddings_with_timeout(big)
    assert spy.on_main is False  # chunking ran on a worker thread, not the event loop
    assert result is not None
    assert len(result) == 1


@pytest.mark.asyncio
async def test_plain_text_below_the_old_size_threshold_is_still_offloaded() -> None:
    """A plain document under a megabyte splits for ~0.8s inline, so it must offload.

    The previous gate gave this exact shape a pass: no line density to trip a
    line-count signal, and a size comfortably below a one-megabyte threshold, while
    the recursive split still ran for hundreds of milliseconds on the event loop.
    """
    spy = _SpyChunkingService()
    text = 'a' * 999_000
    assert len(text) < 1_000_000
    with (
        patch.object(generation_module, 'get_embedding_provider', lambda: _FakeEmbeddingProvider()),
        patch.object(generation_module, 'get_chunking_service', lambda: spy),
    ):
        await generate_embeddings_with_timeout(text)
    assert spy.on_main is False


@pytest.mark.asyncio
async def test_small_entry_chunking_inline() -> None:
    spy = _SpyChunkingService()
    with (
        patch.object(generation_module, 'get_embedding_provider', lambda: _FakeEmbeddingProvider()),
        patch.object(generation_module, 'get_chunking_service', lambda: spy),
    ):
        await generate_embeddings_with_timeout('short text body')
    assert spy.on_main is True  # split_text short-circuits below chunk_size; no thread hop
