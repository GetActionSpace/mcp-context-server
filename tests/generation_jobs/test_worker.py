"""Tests for the generation worker in app/generation_jobs/worker.py.

The worker resets the failed jobs when it starts, claims jobs oldest first while a slot is
free, claims again as soon as a job finishes, a notification arrives or the poll interval
passes, and on stop cancels its running jobs and gives their claims back within a bounded
time. Most cases run on a fake job queue with the executor replaced; the last one runs the
real executor on a SQLite database and stops the worker in the middle of a generation.
"""

import asyncio
import logging
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

import app.generation_jobs.worker as worker_module
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.generation_jobs.executor import JobOutcome
from app.generation_jobs.worker import GenerationWorker
from app.repositories.generation_job_repository import ClaimedJob
from tests.generation_jobs.conftest import InstallGeneration
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._generation_job_cases import enqueue
from tests.repositories._generation_job_cases import require_job

MAX_ATTEMPTS = 3

# The longest a test waits for the worker to reach a state.
WAIT_S = 5.0


def _job(position: int) -> ClaimedJob:
    """Return the claimed job of the entry at ``position`` in the queue."""
    return ClaimedJob(f'0190abcdef1234567890abcd{position:08d}', 1, f'{position:032x}')


class Changes:
    """Lets a test wait until the fakes reach a state: every change of a fake is noted here."""

    def __init__(self) -> None:
        self._noted = asyncio.Event()

    def note(self) -> None:
        """Record that a fake changed."""
        self._noted.set()

    async def until(self, condition: Callable[[], bool]) -> None:
        """Wait until ``condition`` holds, failing after :data:`WAIT_S` seconds."""
        async with asyncio.timeout(WAIT_S):
            while True:
                self._noted.clear()
                if condition():
                    return
                await self._noted.wait()


class FakeJobs:
    """A job queue that hands out its jobs in order and records what the worker asked of it."""

    def __init__(self, jobs: list[ClaimedJob], changes: Changes, *, failed: int = 0, claim_errors: int = 0) -> None:
        self.changes = changes
        self.queue = list(jobs)
        self.failed = failed
        self.claim_errors = claim_errors
        self.events: list[str] = []
        self.released: list[str] = []

    async def reset_failed(self, *, scope: BackendAccess) -> int:
        """Record the reset and report the failed jobs it returned."""
        assert scope is MAINTENANCE
        self.events.append('reset')
        self.changes.note()
        return self.failed

    async def claim_next(self, *, max_attempts: int, lease_s: int, scope: BackendAccess) -> ClaimedJob | None:
        """Hand out the next job, after failing as often as configured."""
        assert (max_attempts, scope) == (MAX_ATTEMPTS, MAINTENANCE)
        assert lease_s > 0
        self.events.append('claim')
        self.changes.note()
        if self.claim_errors:
            self.claim_errors -= 1
            raise RuntimeError('database unavailable')
        return self.queue.pop(0) if self.queue else None

    async def release_claims(self, claim_tokens: list[str], *, scope: BackendAccess) -> int:
        """Record the released tokens."""
        assert scope is MAINTENANCE
        self.released.extend(claim_tokens)
        return len(claim_tokens)


class FakeRuns:
    """Stands in for the executor: records each job it runs and holds it until the gate opens."""

    def __init__(self) -> None:
        self.changes = Changes()
        self.gate = asyncio.Event()
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.running = 0
        self.peak = 0
        self.failing: set[str] = set()

    async def __call__(self, _repos: object, job: ClaimedJob, *, access: BackendAccess, max_attempts: int) -> JobOutcome:
        """Run one job until the gate opens."""
        assert (access, max_attempts) == (MAINTENANCE, MAX_ATTEMPTS)
        self.started.append(job.context_id)
        self.running += 1
        self.peak = max(self.peak, self.running)
        self.changes.note()
        try:
            if job.context_id in self.failing:
                raise RuntimeError('unexpected executor fault')
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled.append(job.context_id)
            raise
        finally:
            self.running -= 1
            self.changes.note()
        return JobOutcome.COMPLETED


def _worker(jobs: FakeJobs, *, concurrency: int = 1) -> GenerationWorker:
    """Return a worker over a repository container whose job queue is ``jobs``."""
    repos = MagicMock()
    repos.generation_jobs = jobs
    return GenerationWorker(repos, access=MAINTENANCE, concurrency=concurrency, max_attempts=MAX_ATTEMPTS)


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> FakeRuns:
    """Replace the executor the worker runs and make the poll interval long, so only events wake the worker."""
    fake = FakeRuns()
    monkeypatch.setattr(worker_module, 'run_job', fake)
    monkeypatch.setattr(worker_module, 'POLL_INTERVAL_S', 60.0)
    return fake


class TestStart:
    """Starting the worker returns the failed jobs to the queue before the first claim."""

    @pytest.mark.asyncio
    async def test_the_failed_jobs_are_reset_before_the_first_claim(
        self, runs: FakeRuns, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The reset runs first, and its count is logged."""
        jobs = FakeJobs([], runs.changes, failed=2)
        worker = _worker(jobs)

        with caplog.at_level(logging.INFO, logger=worker_module.__name__):
            await worker.start()
            await runs.changes.until(lambda: 'claim' in jobs.events)
            await worker.stop()

        assert jobs.events[:2] == ['reset', 'claim']
        assert any('2 failed generation job(s)' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_stopping_a_worker_that_never_started_does_nothing(self, runs: FakeRuns) -> None:
        """Stop before start touches no job."""
        jobs = FakeJobs([_job(1)], runs.changes)

        await _worker(jobs).stop()

        assert (jobs.events, jobs.released) == ([], [])


class TestClaiming:
    """Jobs run oldest first, never more at once than the concurrency, and a free slot is filled at once."""

    @pytest.mark.asyncio
    async def test_jobs_start_oldest_first_and_a_finished_job_frees_its_slot(self, runs: FakeRuns) -> None:
        """With one slot each job starts once the one before it finished, without waiting for a poll."""
        runs.gate.set()
        worker = _worker(FakeJobs([_job(1), _job(2), _job(3)], runs.changes))

        await worker.start()
        await runs.changes.until(lambda: len(runs.started) == 3)
        await worker.stop()

        assert runs.started == [_job(1).context_id, _job(2).context_id, _job(3).context_id]

    @pytest.mark.asyncio
    async def test_no_more_jobs_run_at_once_than_the_concurrency(self, runs: FakeRuns) -> None:
        """Two slots hold two jobs; the rest wait until a slot frees."""
        worker = _worker(FakeJobs([_job(position) for position in range(1, 6)], runs.changes), concurrency=2)

        await worker.start()
        await runs.changes.until(lambda: len(runs.started) == 2)
        await asyncio.sleep(0.1)
        assert len(runs.started) == 2

        runs.gate.set()
        await runs.changes.until(lambda: len(runs.started) == 5)
        await worker.stop()

        assert runs.peak == 2

    @pytest.mark.asyncio
    async def test_a_notification_wakes_the_worker(self, runs: FakeRuns) -> None:
        """A job enqueued after the worker found the queue empty runs as soon as the worker is notified."""
        jobs = FakeJobs([], runs.changes)
        worker = _worker(jobs)
        await worker.start()
        await runs.changes.until(lambda: 'claim' in jobs.events)

        jobs.queue.append(_job(1))
        worker.notify()
        await runs.changes.until(lambda: runs.started == [_job(1).context_id])
        await worker.stop()

    @pytest.mark.asyncio
    async def test_the_worker_polls_without_a_notification(self, runs: FakeRuns, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job enqueued by another process runs at the next poll."""
        monkeypatch.setattr(worker_module, 'POLL_INTERVAL_S', 0.05)
        jobs = FakeJobs([], runs.changes)
        worker = _worker(jobs)
        await worker.start()
        await runs.changes.until(lambda: 'claim' in jobs.events)

        jobs.queue.append(_job(1))
        await runs.changes.until(lambda: runs.started == [_job(1).context_id])
        await worker.stop()

    @pytest.mark.asyncio
    async def test_a_failed_claim_is_retried_at_the_next_poll(
        self, runs: FakeRuns, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A claim that fails is logged and tried again; the worker keeps running."""
        monkeypatch.setattr(worker_module, 'POLL_INTERVAL_S', 0.05)
        worker = _worker(FakeJobs([_job(1)], runs.changes, claim_errors=1))

        with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
            await worker.start()
            await runs.changes.until(lambda: runs.started == [_job(1).context_id])
            await worker.stop()

        assert any('database unavailable' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_job_that_ends_unexpectedly_is_logged_and_frees_its_slot(
        self, runs: FakeRuns, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An executor fault is logged at ERROR and the next job still runs."""
        runs.failing.add(_job(1).context_id)
        runs.gate.set()
        worker = _worker(FakeJobs([_job(1), _job(2)], runs.changes))

        with caplog.at_level(logging.ERROR, logger=worker_module.__name__):
            await worker.start()
            await runs.changes.until(lambda: len(runs.started) == 2)
            await worker.stop()

        errors = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert len(errors) == 1
        assert _job(1).context_id in errors[0].getMessage()


class TestStop:
    """Stopping cancels the running jobs and gives their claims back, within a bounded time."""

    @pytest.mark.asyncio
    async def test_stop_cancels_the_running_jobs_and_releases_their_claims(self, runs: FakeRuns) -> None:
        """Every running job is cancelled and every claim is released; no further job is claimed."""
        jobs = FakeJobs([_job(1), _job(2), _job(3)], runs.changes)
        worker = _worker(jobs, concurrency=2)
        await worker.start()
        await runs.changes.until(lambda: len(runs.started) == 2)

        await worker.stop()

        assert sorted(runs.cancelled) == [_job(1).context_id, _job(2).context_id]
        assert sorted(jobs.released) == [_job(1).claim_token, _job(2).claim_token]
        assert jobs.queue == [_job(3)]

    @pytest.mark.asyncio
    async def test_stop_returns_within_its_bound_when_a_job_does_not_end(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A job that ignores its cancellation cannot hold the stop past the bound; a warning says so."""
        monkeypatch.setattr(worker_module, 'STOP_TIMEOUT_S', 0.2)
        finish = asyncio.Event()
        started = asyncio.Event()

        async def stubborn(_repos: object, _job: ClaimedJob, *, access: BackendAccess, max_attempts: int) -> JobOutcome:
            assert (access, max_attempts) == (MAINTENANCE, MAX_ATTEMPTS)
            started.set()
            while not finish.is_set():
                try:
                    await finish.wait()
                except asyncio.CancelledError:
                    continue
            return JobOutcome.COMPLETED

        monkeypatch.setattr(worker_module, 'run_job', stubborn)
        worker = _worker(FakeJobs([_job(1)], Changes()))
        await worker.start()
        await asyncio.wait_for(started.wait(), timeout=5)

        with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
            await asyncio.wait_for(worker.stop(), timeout=2)

        assert any(record.levelno == logging.WARNING for record in caplog.records)
        finish.set()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('active_legs')
    async def test_stop_returns_a_running_job_to_the_queue_with_its_attempt(
        self, db: ScopedDb, generation: InstallGeneration, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On a real database the generation is cancelled and the job is pending again with no attempt spent."""
        monkeypatch.setattr(worker_module, 'POLL_INTERVAL_S', 60.0)
        fake = generation(block=True)
        context_id = await db.add_entry('doc', owner='alice', visibility='private')
        await enqueue(db, context_id)
        worker = GenerationWorker(db.repos, access=MAINTENANCE, concurrency=1, max_attempts=MAX_ATTEMPTS)
        await worker.start()
        await asyncio.wait_for(fake.started.wait(), timeout=5)

        await worker.stop()

        assert fake.cancelled is True
        stored = await require_job(db, context_id)
        assert (stored.status, stored.attempts, stored.claim_token) == ('pending', 0, None)
        assert stored.available_in_s <= 0
