"""Embed a document's chunks in batches, one provider request per batch.

A chunked document goes to the embedding provider in batches rather than in one
request, so the provider's retry and per-attempt timeout (``EMBEDDING_TIMEOUT_S``)
cover one batch, a retry re-sends only the batch that failed, and the caller's
timeout bounds each batch instead of the whole document.

The caller's semaphore is acquired for each batch and released between batches on
purpose: ``asyncio.Semaphore`` wakes its waiters in FIFO order, so a caller that
arrives while another document is embedding gets a slot after that document's
current batch instead of after its whole run.

Batch sizes come from a ``BatchSizer``. ``AdaptiveBatchSizer`` (``EMBEDDING_BATCH_SIZE``
``auto``) sizes each batch to take about ``EMBEDDING_BATCH_TARGET_S`` from the seconds
per chunk measured so far, which needs no knowledge of the hardware or service
behind the provider; ``FixedBatchSizer`` uses the configured integer size.
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.settings.embedding import EmbeddingSettings

logger = logging.getLogger(__name__)

# Chunks in an auto-sized batch before any batch has been measured.
AUTO_FIRST_BATCH = 4
# Largest auto-sized batch.
AUTO_MAX_BATCH = 64
# Weight of the newest measurement in the seconds-per-chunk estimate.
AUTO_SMOOTHING = 0.5


class DocumentEmbedder(Protocol):
    """The part of an embedding provider the batched loop uses."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in one provider request.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text.
        """
        ...


class BatchSizer(Protocol):
    """Decides how many chunks the next embedding batch carries."""

    @property
    def mode(self) -> str:
        """The sizing mode for logs: ``'auto'`` or the fixed size."""
        ...

    def next_size(self, remaining: int) -> int:
        """Return the size of the next batch.

        Args:
            remaining: Chunks of the document not yet embedded (at least 1).

        Returns:
            A size from 1 to ``remaining``.
        """
        ...

    def observe(self, chunks: int, elapsed_s: float) -> None:
        """Record how long a batch took.

        Args:
            chunks: Chunks in the batch.
            elapsed_s: Wall time of the provider request in seconds.
        """
        ...


class AdaptiveBatchSizer:
    """Sizes each batch to take about a target wall time.

    Keeps an exponentially weighted estimate of the seconds one chunk takes and
    sizes the next batch as ``round(target_s / seconds_per_chunk)``, clamped to
    ``1..AUTO_MAX_BATCH``. Before the first measurement a batch is
    ``AUTO_FIRST_BATCH`` chunks. The estimate is shared by every document embedded
    through the same sizer, so it follows the provider's current speed, queueing
    at the provider included.
    """

    def __init__(self, target_s: float) -> None:
        """Create the sizer.

        Args:
            target_s: Wall time one batch should take, in seconds.
        """
        self._target_s = target_s
        self._seconds_per_chunk: float | None = None

    @property
    def mode(self) -> str:
        """Return ``'auto'``."""
        return 'auto'

    @property
    def seconds_per_chunk(self) -> float | None:
        """The current estimate, or None before the first measurement."""
        return self._seconds_per_chunk

    def next_size(self, remaining: int) -> int:
        """Return the size of the next batch.

        Args:
            remaining: Chunks of the document not yet embedded (at least 1).

        Returns:
            The target-sized batch, clamped to ``1..AUTO_MAX_BATCH`` and ``remaining``.
        """
        if self._seconds_per_chunk is None:
            size = AUTO_FIRST_BATCH
        else:
            size = round(min(self._target_s / self._seconds_per_chunk, AUTO_MAX_BATCH))
        return max(1, min(size, remaining))

    def observe(self, chunks: int, elapsed_s: float) -> None:
        """Fold a batch's seconds per chunk into the estimate.

        A measurement that is not a positive, finite time carries no rate and is
        ignored.

        Args:
            chunks: Chunks in the batch.
            elapsed_s: Wall time of the provider request in seconds.
        """
        if chunks < 1 or not math.isfinite(elapsed_s) or elapsed_s <= 0:
            return
        sample = elapsed_s / chunks
        if self._seconds_per_chunk is None:
            self._seconds_per_chunk = sample
        else:
            self._seconds_per_chunk += AUTO_SMOOTHING * (sample - self._seconds_per_chunk)


class FixedBatchSizer:
    """Uses one configured batch size."""

    def __init__(self, size: int) -> None:
        """Create the sizer.

        Args:
            size: Chunks per batch (at least 1).
        """
        self._size = size

    @property
    def mode(self) -> str:
        """Return the size as text."""
        return str(self._size)

    def next_size(self, remaining: int) -> int:
        """Return the configured size, or ``remaining`` when fewer chunks are left.

        Args:
            remaining: Chunks of the document not yet embedded (at least 1).

        Returns:
            The batch size.
        """
        return min(self._size, remaining)

    def observe(self, chunks: int, elapsed_s: float) -> None:
        """Ignore the measurement; the size is fixed.

        Args:
            chunks: Chunks in the batch.
            elapsed_s: Wall time of the provider request in seconds.
        """
        _ = (chunks, elapsed_s)


def create_batch_sizer(settings: EmbeddingSettings) -> BatchSizer:
    """Build the sizer ``EMBEDDING_BATCH_SIZE`` selects.

    Args:
        settings: The embedding settings.

    Returns:
        An ``AdaptiveBatchSizer`` for ``'auto'``, otherwise a ``FixedBatchSizer``.
    """
    if settings.batch_size == 'auto':
        return AdaptiveBatchSizer(target_s=settings.batch_target_s)
    return FixedBatchSizer(settings.batch_size)


@dataclass(frozen=True, slots=True)
class BatchPosition:
    """Where a document's embedding stands at a batch.

    Chunk positions are 1-based. ``batch_count`` is projected from the batch's
    size, so it is exact for a fixed size and an estimate in auto mode.

    Attributes:
        batch_number: The batch, counted from 1.
        batch_count: Batches the document takes.
        first_chunk: Position of the batch's first chunk.
        last_chunk: Position of the batch's last chunk.
        total_chunks: Chunks in the document.
        embedded_chunks: Chunks embedded before this batch.
        elapsed_s: Seconds since the document's embedding started.
    """

    batch_number: int
    batch_count: int
    first_chunk: int
    last_chunk: int
    total_chunks: int
    embedded_chunks: int
    elapsed_s: float

    @property
    def batch_label(self) -> str:
        """For example ``'batch 3 of 41'``."""
        return f'batch {self.batch_number} of {self.batch_count}'

    @property
    def chunk_label(self) -> str:
        """For example ``'chunks 9-15 of 280'``."""
        if self.first_chunk == self.last_chunk:
            return f'chunk {self.first_chunk} of {self.total_chunks}'
        return f'chunks {self.first_chunk}-{self.last_chunk} of {self.total_chunks}'

    @property
    def progress_label(self) -> str:
        """For example ``'8 chunks embedded in 12.3 s'``."""
        noun = 'chunk' if self.embedded_chunks == 1 else 'chunks'
        return f'{self.embedded_chunks} {noun} embedded in {self.elapsed_s:.1f} s'


class EmbeddingBatchError(Exception):
    """A batch of a document's chunks failed to embed."""

    def __init__(self, position: BatchPosition, cause: Exception) -> None:
        """Record where the document's embedding stopped and why.

        Args:
            position: The failing batch's position.
            cause: The provider's exception, or the TimeoutError of the batch timeout.
        """
        self.position = position
        self.cause = cause
        super().__init__(f'{position.batch_label}, {position.chunk_label}; {position.progress_label}: {cause!r}')


class EmbeddingBatchTimeoutError(EmbeddingBatchError):
    """A batch did not finish within the per-batch timeout."""


def _position(*, batch_number: int, embedded: int, size: int, total: int, elapsed_s: float) -> BatchPosition:
    return BatchPosition(
        batch_number=batch_number,
        batch_count=batch_number + math.ceil((total - embedded - size) / size),
        first_chunk=embedded + 1,
        last_chunk=embedded + size,
        total_chunks=total,
        embedded_chunks=embedded,
        elapsed_s=elapsed_s,
    )


async def embed_in_batches(
    provider: DocumentEmbedder,
    texts: Sequence[str],
    *,
    sizer: BatchSizer,
    semaphore: asyncio.Semaphore,
    batch_timeout_s: float,
    clock: Callable[[], float] = time.perf_counter,
) -> list[list[float]]:
    """Embed ``texts`` in batches, one ``embed_documents`` request per batch.

    For each batch the loop acquires ``semaphore``, awaits the provider under
    ``batch_timeout_s``, releases the semaphore, and feeds the request's wall time
    to ``sizer``.

    Args:
        provider: The embedding provider.
        texts: The chunk texts, in document order.
        sizer: Decides each batch's size and learns from its timing.
        semaphore: Bounds concurrent provider requests; held for one batch at a time.
        batch_timeout_s: Timeout for one batch's provider request, retries included.
        clock: Monotonic clock in seconds.

    Returns:
        One vector per text, in the order of ``texts``.

    Raises:
        EmbeddingBatchTimeoutError: If a batch exceeds ``batch_timeout_s``.
        EmbeddingBatchError: If the provider fails a batch.
    """
    total = len(texts)
    vectors: list[list[float]] = []
    started = clock()
    embedded = 0
    batch_number = 0
    while embedded < total:
        size = sizer.next_size(total - embedded)
        batch_number += 1
        batch = list(texts[embedded:embedded + size])
        async with semaphore:
            batch_started = clock()
            try:
                async with asyncio.timeout(batch_timeout_s):
                    batch_vectors = await provider.embed_documents(batch)
            except Exception as exc:
                position = _position(
                    batch_number=batch_number, embedded=embedded, size=size, total=total, elapsed_s=clock() - started,
                )
                if isinstance(exc, TimeoutError):
                    raise EmbeddingBatchTimeoutError(position, exc) from exc
                raise EmbeddingBatchError(position, exc) from exc
            elapsed_s = clock() - batch_started
        sizer.observe(size, elapsed_s)
        vectors.extend(batch_vectors)
        logger.debug(
            'Embedded batch %d: chunks %d-%d of %d in %.2f s',
            batch_number, embedded + 1, embedded + size, total, elapsed_s,
        )
        embedded += size
    logger.info(
        'Embeddings generated: chunks=%d, batches=%d, elapsed=%.1fs',
        total, batch_number, clock() - started,
    )
    return vectors
