"""The generation worker: drains the generation job queue in the background of the server.

The worker claims the oldest claimable job while it runs fewer jobs than its concurrency,
and runs each through the executor. It claims again as soon as a job finishes, when a
write that enqueued a job notifies it, and every :data:`POLL_INTERVAL_S` seconds, which
picks up jobs other processes enqueued and jobs whose backoff or lease ended.

Starting the worker returns every failed job to the queue, so each server start retries
them. Stopping it cancels the running jobs and gives their claims back without spending an
attempt, all within :data:`STOP_TIMEOUT_S` seconds; a claim it could not give back is
claimed again once its lease ends.
"""

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from app.backends.base import BackendAccess
from app.errors import format_exception_message
from app.generation_jobs.executor import LEASE_S
from app.generation_jobs.executor import JobOutcome
from app.generation_jobs.executor import run_job
from app.repositories.generation_job_repository import ClaimedJob

if TYPE_CHECKING:
    from app.repositories import RepositoryContainer

logger = logging.getLogger(__name__)

# The seconds between two looks at the queue when nothing wakes the worker.
POLL_INTERVAL_S = 5.0

# The longest a stop waits for the running jobs to end and their claims to be given back.
STOP_TIMEOUT_S = 10.0

_STOP_TIMED_OUT = 'Generation worker did not stop within %.0f s; its running jobs are claimed again once their leases end'


class GenerationWorker:
    """Runs the claimable generation jobs, at most ``concurrency`` at a time, oldest first."""

    def __init__(
        self, repos: 'RepositoryContainer', *, access: BackendAccess, concurrency: int, max_attempts: int,
    ) -> None:
        """Configure the worker; nothing runs until :meth:`start`.

        Args:
            repos: Repository container over the backend the jobs run on.
            access: The identity every statement of the worker runs as.
            concurrency: The most jobs that run at once.
            max_attempts: The attempts a job may spend.
        """
        self._repos = repos
        self._access = access
        self._concurrency = concurrency
        self._max_attempts = max_attempts
        self._wake = asyncio.Event()
        self._running: dict[asyncio.Task[JobOutcome], ClaimedJob] = {}
        self._loop: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Return the failed jobs to the queue, then start claiming in the background."""
        reset = await self._repos.generation_jobs.reset_failed(scope=self._access)
        if reset:
            logger.info('%d failed generation job(s) returned to the queue', reset)
        self._loop = asyncio.create_task(self._claim_loop(), name='generation-worker')
        logger.info(
            'Generation worker started: up to %d job(s) at a time, %d attempt(s) per job',
            self._concurrency, self._max_attempts,
        )

    def notify(self) -> None:
        """Wake the worker to claim at once, after a write enqueued a job."""
        self._wake.set()

    async def stop(self) -> None:
        """Stop claiming, cancel the running jobs and give their claims back, within :data:`STOP_TIMEOUT_S`."""
        claim_loop, self._loop = self._loop, None
        if claim_loop is None:
            return
        claim_loop.cancel()
        running = dict(self._running)
        for task in running:
            task.cancel()
        deadline = asyncio.get_running_loop().time() + STOP_TIMEOUT_S
        # asyncio.wait returns at the deadline even when a task ignores its cancellation,
        # which awaiting the tasks themselves would not.
        _ended, unended = await asyncio.wait([claim_loop, *running], timeout=STOP_TIMEOUT_S)
        if unended:
            logger.warning(_STOP_TIMED_OUT, STOP_TIMEOUT_S)
            return
        try:
            async with asyncio.timeout_at(deadline):
                released = await self._repos.generation_jobs.release_claims(
                    [job.claim_token for job in running.values()], scope=self._access,
                )
        except TimeoutError:
            logger.warning(_STOP_TIMED_OUT, STOP_TIMEOUT_S)
            return
        except Exception as error:
            logger.warning(
                'Generation worker could not give its claims back (%s); the jobs are claimed again once their leases end',
                format_exception_message(error),
            )
            return
        logger.info('Generation worker stopped; %d running job(s) returned to the queue', released)

    async def _claim_loop(self) -> None:
        """Claim whenever a slot is free and something may have become claimable."""
        while True:
            self._wake.clear()
            await self._claim_free_slots()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(POLL_INTERVAL_S):
                    await self._wake.wait()

    async def _claim_free_slots(self) -> None:
        """Claim and start jobs until every slot is taken or no job is claimable."""
        while len(self._running) < self._concurrency:
            try:
                job = await self._repos.generation_jobs.claim_next(
                    max_attempts=self._max_attempts, lease_s=LEASE_S, scope=self._access,
                )
            except Exception as error:
                logger.warning(
                    'Generation worker could not claim a job and tries again in %.0f s: %s',
                    POLL_INTERVAL_S, format_exception_message(error),
                )
                return
            if job is None:
                return
            task = asyncio.create_task(
                run_job(self._repos, job, access=self._access, max_attempts=self._max_attempts),
                name=f'generation-job-{job.context_id}',
            )
            self._running[task] = job
            task.add_done_callback(self._job_ended)

    def _job_ended(self, task: 'asyncio.Task[JobOutcome]') -> None:
        """Free the job's slot, log a fault the executor did not handle, and wake the worker to claim again."""
        job = self._running.pop(task, None)
        if not task.cancelled() and (error := task.exception()) is not None and job is not None:
            logger.error(
                'Generation job for context entry %s ended unexpectedly; it is claimed again once its lease ends',
                job.context_id, exc_info=error,
            )
        self._wake.set()
