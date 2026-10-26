"""Tests for app.embeddings.batching: batch sizers and the batched embed loop."""

import asyncio
from typing import override
from unittest.mock import patch

import pytest

from app.embeddings.batching import AdaptiveBatchSizer
from app.embeddings.batching import BatchPosition
from app.embeddings.batching import EmbeddingBatchError
from app.embeddings.batching import EmbeddingBatchTimeoutError
from app.embeddings.batching import FixedBatchSizer
from app.embeddings.batching import _position
from app.embeddings.batching import create_batch_sizer
from app.embeddings.batching import embed_in_batches
from app.embeddings.retry import with_retry_and_timeout
from app.settings.embedding import EmbeddingSettings


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingProvider:
    """Embedding provider that records every batch it receives.

    Each text ``t<i>`` embeds to ``[float(i)]``, so the result order is checkable.
    ``seconds_per_chunk`` advances ``clock`` per call to model provider speed.
    """

    def __init__(self, clock: FakeClock | None = None, seconds_per_chunk: float = 0.0) -> None:
        self.batches: list[list[str]] = []
        self._clock = clock
        self._seconds_per_chunk = seconds_per_chunk

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self._clock is not None:
            self._clock.now += self._seconds_per_chunk * len(texts)
        return [[float(text[1:])] for text in texts]


def _texts(count: int) -> list[str]:
    return [f't{i}' for i in range(count)]


class TestAdaptiveBatchSizer:
    """The auto sizer aims each batch at the target wall time."""

    def test_first_batch_is_four_chunks(self) -> None:
        assert AdaptiveBatchSizer(target_s=5.0).next_size(100) == 4

    def test_first_batch_never_exceeds_remaining(self) -> None:
        assert AdaptiveBatchSizer(target_s=5.0).next_size(3) == 3

    def test_sizes_from_observed_seconds_per_chunk(self) -> None:
        sizer = AdaptiveBatchSizer(target_s=5.0)
        sizer.observe(4, 2.0)
        assert sizer.next_size(100) == 10

    def test_estimate_is_exponentially_weighted(self) -> None:
        sizer = AdaptiveBatchSizer(target_s=5.0)
        sizer.observe(4, 2.0)
        sizer.observe(10, 10.0)
        # 0.5 s/chunk, then a 1.0 s/chunk batch: the estimate moves halfway, to 0.75.
        assert sizer.seconds_per_chunk == pytest.approx(0.75)
        assert sizer.next_size(100) == 7

    def test_converges_on_a_steady_provider(self) -> None:
        sizer = AdaptiveBatchSizer(target_s=5.0)
        sizer.observe(4, 0.4)
        for _ in range(12):
            size = sizer.next_size(1000)
            sizer.observe(size, size * 1.0)
        assert sizer.next_size(1000) == 5

    def test_ceiling_is_sixty_four(self) -> None:
        sizer = AdaptiveBatchSizer(target_s=5.0)
        sizer.observe(4, 0.004)
        assert sizer.next_size(1000) == 64

    def test_floor_is_one(self) -> None:
        sizer = AdaptiveBatchSizer(target_s=5.0)
        sizer.observe(1, 60.0)
        assert sizer.next_size(1000) == 1

    def test_never_exceeds_remaining_after_observation(self) -> None:
        sizer = AdaptiveBatchSizer(target_s=5.0)
        sizer.observe(4, 2.0)
        assert sizer.next_size(3) == 3

    @pytest.mark.parametrize('elapsed_s', [0.0, -1.0])
    def test_non_positive_elapsed_time_is_ignored(self, elapsed_s: float) -> None:
        sizer = AdaptiveBatchSizer(target_s=5.0)
        sizer.observe(4, elapsed_s)
        assert sizer.seconds_per_chunk is None
        assert sizer.next_size(100) == 4
        sizer.observe(4, 2.0)
        sizer.observe(4, elapsed_s)
        assert sizer.seconds_per_chunk == pytest.approx(0.5)

    def test_mode_is_auto(self) -> None:
        assert AdaptiveBatchSizer(target_s=5.0).mode == 'auto'


class TestFixedBatchSizer:
    """The integer setting fixes the batch size."""

    def test_returns_the_configured_size(self) -> None:
        assert FixedBatchSizer(8).next_size(100) == 8

    def test_never_exceeds_remaining(self) -> None:
        assert FixedBatchSizer(8).next_size(5) == 5

    def test_observations_do_not_change_the_size(self) -> None:
        sizer = FixedBatchSizer(8)
        sizer.observe(8, 100.0)
        assert sizer.next_size(100) == 8

    def test_mode_is_the_size(self) -> None:
        assert FixedBatchSizer(8).mode == '8'


class TestCreateBatchSizer:
    """The factory builds the sizer the settings select."""

    def test_auto_builds_an_adaptive_sizer_with_the_target(self) -> None:
        sizer = create_batch_sizer(
            EmbeddingSettings.model_validate({'EMBEDDING_BATCH_SIZE': 'auto', 'EMBEDDING_BATCH_TARGET_S': 2.0}),
        )
        assert isinstance(sizer, AdaptiveBatchSizer)
        sizer.observe(4, 2.0)
        assert sizer.next_size(100) == 4

    def test_integer_builds_a_fixed_sizer(self) -> None:
        sizer = create_batch_sizer(EmbeddingSettings.model_validate({'EMBEDDING_BATCH_SIZE': '16'}))
        assert isinstance(sizer, FixedBatchSizer)
        assert sizer.next_size(100) == 16


class TestEmbedInBatches:
    """The loop sends one provider request per batch and keeps the chunk order."""

    @pytest.mark.asyncio
    async def test_batches_follow_the_sizer_in_order(self) -> None:
        provider = RecordingProvider()
        vectors = await embed_in_batches(
            provider, _texts(10), sizer=FixedBatchSizer(4), semaphore=asyncio.Semaphore(1), batch_timeout_s=10.0,
        )
        assert provider.batches == [['t0', 't1', 't2', 't3'], ['t4', 't5', 't6', 't7'], ['t8', 't9']]
        assert vectors == [[float(i)] for i in range(10)]

    @pytest.mark.asyncio
    async def test_adaptive_sizes_follow_the_measured_speed(self) -> None:
        clock = FakeClock()
        provider = RecordingProvider(clock, seconds_per_chunk=0.5)
        await embed_in_batches(
            provider,
            _texts(30),
            sizer=AdaptiveBatchSizer(target_s=5.0),
            semaphore=asyncio.Semaphore(1),
            batch_timeout_s=10.0,
            clock=clock,
        )
        assert [len(batch) for batch in provider.batches] == [4, 10, 10, 6]

    @pytest.mark.asyncio
    async def test_a_retried_batch_resends_only_itself(self) -> None:
        sent: list[list[str]] = []
        failed_once: set[str] = set()

        class RetryingProvider:
            async def embed_documents(self, texts: list[str]) -> list[list[float]]:
                async def _call() -> list[list[float]]:
                    sent.append(list(texts))
                    if texts[0] == 't2' and 't2' not in failed_once:
                        failed_once.add('t2')
                        raise ConnectionError('connection reset')
                    return [[float(text[1:])] for text in texts]

                return await with_retry_and_timeout(_call, 'test_embed_documents')

        with patch('app.embeddings.retry.get_settings') as mock_settings:
            mock_settings.return_value.embedding.timeout_s = 5.0
            mock_settings.return_value.embedding.retry_max_attempts = 3
            mock_settings.return_value.embedding.retry_base_delay_s = 0.01
            vectors = await embed_in_batches(
                RetryingProvider(),
                _texts(6),
                sizer=FixedBatchSizer(2),
                semaphore=asyncio.Semaphore(1),
                batch_timeout_s=10.0,
            )

        assert sent == [['t0', 't1'], ['t2', 't3'], ['t2', 't3'], ['t4', 't5']]
        assert vectors == [[float(i)] for i in range(6)]

    @pytest.mark.asyncio
    async def test_failure_reports_the_batch_position(self) -> None:
        clock = FakeClock()
        cause = RuntimeError('model not loaded')

        class FailingProvider(RecordingProvider):
            @override
            async def embed_documents(self, texts: list[str]) -> list[list[float]]:
                if texts[0] == 't3':
                    clock.now += 7.0  # the failing request's own attempts and backoff
                    raise cause
                return await super().embed_documents(texts)

        with pytest.raises(EmbeddingBatchError) as excinfo:
            await embed_in_batches(
                FailingProvider(clock, seconds_per_chunk=0.5),
                _texts(8),
                sizer=FixedBatchSizer(3),
                semaphore=asyncio.Semaphore(1),
                batch_timeout_s=10.0,
                clock=clock,
            )

        error = excinfo.value
        assert not isinstance(error, EmbeddingBatchTimeoutError)
        assert error.cause is cause
        assert error.__cause__ is cause
        position = error.position
        assert (position.batch_number, position.batch_count) == (2, 3)
        assert (position.first_chunk, position.last_chunk, position.total_chunks) == (4, 6, 8)
        assert position.embedded_chunks == 3
        assert position.elapsed_s == pytest.approx(1.5)
        assert position.batch_label == 'batch 2 of 3'
        assert position.chunk_label == 'chunks 4-6 of 8'
        assert position.progress_label == '3 chunks embedded in 1.5 s'

    @pytest.mark.asyncio
    async def test_failure_progress_leaves_out_the_wait_for_the_failing_batch(self) -> None:
        clock = FakeClock()
        semaphore = asyncio.Semaphore(1)

        async def other_caller() -> None:
            async with semaphore:
                clock.now += 100.0  # another document's batch holds the slot

        class Provider:
            def __init__(self) -> None:
                self.other: asyncio.Task[None] | None = None

            async def embed_documents(self, texts: list[str]) -> list[list[float]]:
                if self.other is not None:
                    raise RuntimeError('model not loaded')
                self.other = asyncio.create_task(other_caller())
                await asyncio.sleep(0)  # the other caller queues for the slot
                clock.now += 1.0
                return [[0.0] for _ in texts]

        provider = Provider()
        with pytest.raises(EmbeddingBatchError) as excinfo:
            await embed_in_batches(
                provider, _texts(4), sizer=FixedBatchSizer(2), semaphore=semaphore, batch_timeout_s=10.0, clock=clock,
            )
        assert provider.other is not None
        await provider.other

        assert clock.now == pytest.approx(101.0)
        assert excinfo.value.position.progress_label == '2 chunks embedded in 1.0 s'

    def test_labels_read_naturally_for_one_chunk(self) -> None:
        position = BatchPosition(
            batch_number=2,
            batch_count=2,
            first_chunk=2,
            last_chunk=2,
            total_chunks=2,
            embedded_chunks=1,
            elapsed_s=0.25,
        )
        assert position.chunk_label == 'chunk 2 of 2'
        assert position.progress_label == '1 chunk embedded in 0.2 s'

    def test_auto_mode_projects_the_batch_count_from_the_current_size(self) -> None:
        assert _position(batch_number=2, embedded=4, size=10, total=95, elapsed_s=1.0).batch_count == 11

    @pytest.mark.asyncio
    async def test_the_timeout_bounds_each_batch_not_the_document(self) -> None:
        clock = FakeClock()
        never = asyncio.Event()

        class SlowProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def embed_documents(self, texts: list[str]) -> list[list[float]]:
                self.calls += 1
                if self.calls == 5:
                    try:
                        await never.wait()
                    except asyncio.CancelledError:
                        clock.now += 0.25  # the hang lasts until the batch timeout cancels it
                        raise
                await asyncio.sleep(0.1)
                clock.now += 0.1
                return [[0.0] for _ in texts]

        # Four 0.1 s batches together take longer than the 0.25 s timeout, so only a
        # per-batch timeout lets them finish; the fifth batch never answers.
        with pytest.raises(EmbeddingBatchTimeoutError) as excinfo:
            await embed_in_batches(
                SlowProvider(),
                _texts(5),
                sizer=FixedBatchSizer(1),
                semaphore=asyncio.Semaphore(1),
                batch_timeout_s=0.25,
                clock=clock,
            )

        error = excinfo.value
        assert (error.position.batch_number, error.position.batch_count) == (5, 5)
        assert error.position.embedded_chunks == 4
        assert isinstance(error.cause, TimeoutError)
        # The progress covers the four embedded chunks and leaves out the timed-out batch.
        assert error.position.elapsed_s == pytest.approx(0.4)
        assert error.position.progress_label == '4 chunks embedded in 0.4 s'

    @pytest.mark.asyncio
    async def test_a_waiting_caller_is_served_after_the_current_batch(self) -> None:
        gate = asyncio.Event()
        first_call_started = asyncio.Event()
        order: list[str] = []

        class GatedProvider:
            async def embed_documents(self, texts: list[str]) -> list[list[float]]:
                order.append(texts[0])
                first_call_started.set()
                await gate.wait()
                return [[0.0] for _ in texts]

        semaphore = asyncio.Semaphore(1)
        provider = GatedProvider()
        long_document = asyncio.create_task(
            embed_in_batches(
                provider,
                ['a0', 'a1', 'a2', 'a3', 'a4', 'a5'],
                sizer=FixedBatchSizer(2),
                semaphore=semaphore,
                batch_timeout_s=10.0,
            ),
        )
        await first_call_started.wait()
        short_document = asyncio.create_task(
            embed_in_batches(provider, ['b0'], sizer=FixedBatchSizer(2), semaphore=semaphore, batch_timeout_s=10.0),
        )
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(long_document, short_document)

        assert order == ['a0', 'b0', 'a2', 'a4']

    @pytest.mark.asyncio
    async def test_the_sizer_learns_request_time_not_queue_time(self) -> None:
        clock = FakeClock()
        observed: list[tuple[int, float]] = []

        class RecordingSizer(FixedBatchSizer):
            @override
            def observe(self, chunks: int, elapsed_s: float) -> None:
                observed.append((chunks, elapsed_s))

        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        task = asyncio.create_task(
            embed_in_batches(
                RecordingProvider(clock, seconds_per_chunk=1.0),
                _texts(2),
                sizer=RecordingSizer(2),
                semaphore=semaphore,
                batch_timeout_s=10.0,
                clock=clock,
            ),
        )
        await asyncio.sleep(0)
        clock.now += 100.0  # time spent waiting for the slot
        semaphore.release()
        await task

        assert observed == [(2, 2.0)]

    @pytest.mark.asyncio
    async def test_batches_and_completion_are_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level('DEBUG', logger='app.embeddings.batching')
        await embed_in_batches(
            RecordingProvider(), _texts(5), sizer=FixedBatchSizer(2), semaphore=asyncio.Semaphore(1), batch_timeout_s=10.0,
        )
        messages = [record.getMessage() for record in caplog.records]
        assert any(m.startswith('Embedded batch 1: chunks 1-2 of 5 in ') for m in messages)
        assert any(m.startswith('Embedded batch 3: chunks 5-5 of 5 in ') for m in messages)
        assert any(m.startswith('Embeddings generated: chunks=5, batches=3, elapsed=') for m in messages)
