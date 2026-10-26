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
per character of chunk text measured so far and the lengths of the chunks the batch
would carry, which needs no knowledge of the hardware or service behind the
provider; ``FixedBatchSizer`` uses the configured integer size.
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
# How fast older measurements fade: each measured batch scales the running sums of
# seconds and characters by ``1 - AUTO_SMOOTHING`` before adding its own.
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

    def next_size(self, remaining_lengths: Sequence[int]) -> int:
        """Return the size of the next batch.

        Args:
            remaining_lengths: Character counts of the document's chunks not yet
                embedded, in document order (at least one).

        Returns:
            A size from 1 to ``len(remaining_lengths)``.
        """
        ...

    def observe(self, chars: int, elapsed_s: float) -> None:
        """Record how long a batch took.

        Args:
            chars: Characters in the batch's chunks.
            elapsed_s: Wall time of the provider request in seconds.
        """
        ...


class AdaptiveBatchSizer:
    """Sizes each batch to take about a target wall time.

    Learns the seconds one character of chunk text takes: every measured batch adds
    its seconds and its characters to two running sums, after scaling both by
    ``1 - AUTO_SMOOTHING``, and the rate is their ratio. The next batch takes the
    document's next chunks in order while their predicted time (the rate times
    their characters) stays within ``target_s``: at least one chunk and at most
    ``AUTO_MAX_BATCH``. Before the first measurement a batch is ``AUTO_FIRST_BATCH``
    chunks. Counting characters rather than chunks keeps a short text, such as a
    one-line message embedded as one chunk, from making full-size chunks look
    cheap. The rate is shared by every document embedded through the same sizer, so
    it follows the provider's current speed, queueing at the provider included.
    """

    def __init__(self, target_s: float) -> None:
        """Create the sizer.

        Args:
            target_s: Wall time one batch should take, in seconds.
        """
        self._target_s = target_s
        self._seconds = 0.0
        self._chars = 0.0

    @property
    def mode(self) -> str:
        """Return ``'auto'``."""
        return 'auto'

    @property
    def seconds_per_char(self) -> float | None:
        """The current rate, or None before the first measurement."""
        if self._chars == 0:
            return None
        return self._seconds / self._chars

    def next_size(self, remaining_lengths: Sequence[int]) -> int:
        """Return the size of the next batch.

        Args:
            remaining_lengths: Character counts of the document's chunks not yet
                embedded, in document order (at least one).

        Returns:
            As many of the next chunks as fit the target at the current rate, from
            1 to ``AUTO_MAX_BATCH``, or ``AUTO_FIRST_BATCH`` before the first
            measurement; never more than ``len(remaining_lengths)``.
        """
        rate = self.seconds_per_char
        if rate is None:
            return min(AUTO_FIRST_BATCH, len(remaining_lengths))
        limit = min(AUTO_MAX_BATCH, len(remaining_lengths))
        size = 1
        chars = remaining_lengths[0]
        while size < limit and rate * (chars + remaining_lengths[size]) <= self._target_s:
            chars += remaining_lengths[size]
            size += 1
        return size

    def observe(self, chars: int, elapsed_s: float) -> None:
        """Fold a batch's seconds and characters into the rate.

        A measurement that is not a positive, finite time carries no rate and is
        ignored.

        Args:
            chars: Characters in the batch's chunks.
            elapsed_s: Wall time of the provider request in seconds.
        """
        if not math.isfinite(elapsed_s) or elapsed_s <= 0:
            return
        keep = 1 - AUTO_SMOOTHING
        self._seconds = self._seconds * keep + elapsed_s
        self._chars = self._chars * keep + chars


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

    def next_size(self, remaining_lengths: Sequence[int]) -> int:
        """Return the configured size, or the remaining chunk count when fewer are left.

        Args:
            remaining_lengths: Character counts of the document's chunks not yet
                embedded, in document order (at least one).

        Returns:
            The batch size.
        """
        return min(self._size, len(remaining_lengths))

    def observe(self, chars: int, elapsed_s: float) -> None:
        """Ignore the measurement; the size is fixed.

        Args:
            chars: Characters in the batch's chunks.
            elapsed_s: Wall time of the provider request in seconds.
        """
        _ = (chars, elapsed_s)


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
        elapsed_s: Seconds from the document's start to its last embedded chunk.
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

    For each batch the loop asks ``sizer`` for a size from the lengths of the chunks
    not yet embedded, acquires ``semaphore``, awaits the provider under
    ``batch_timeout_s``, releases the semaphore, and feeds the batch's characters and
    the request's wall time to ``sizer``.

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
    lengths = [len(text) for text in texts]
    vectors: list[list[float]] = []
    started = clock()
    # The time the latest batch's vectors arrived. A failure reports its progress up
    # to here, leaving out the failing batch's own attempts, backoff and timeout.
    progress_at = started
    embedded = 0
    batch_number = 0
    while embedded < total:
        size = sizer.next_size(lengths[embedded:])
        batch_number += 1
        batch = list(texts[embedded:embedded + size])
        async with semaphore:
            batch_started = clock()
            try:
                async with asyncio.timeout(batch_timeout_s):
                    batch_vectors = await provider.embed_documents(batch)
            except Exception as exc:
                position = _position(
                    batch_number=batch_number, embedded=embedded, size=size, total=total, elapsed_s=progress_at - started,
                )
                if isinstance(exc, TimeoutError):
                    raise EmbeddingBatchTimeoutError(position, exc) from exc
                raise EmbeddingBatchError(position, exc) from exc
            progress_at = clock()
            elapsed_s = progress_at - batch_started
        sizer.observe(sum(lengths[embedded:embedded + size]), elapsed_s)
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
