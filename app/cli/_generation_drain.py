"""Complete outstanding generation jobs from the migration CLI.

A write made with ``defer_generation`` stores its entry together with a generation job, and a
server's generation worker generates the entry's embeddings, summary and index-tree node
summaries later. ``--embed-missing`` and ``--re-embed`` complete the outstanding jobs in their
own process before their own pass, so a database whose server is stopped still receives every
deferred output: the failed jobs return to the queue, then the oldest claimable job is claimed
under a lease, as a worker claims it, and run once through the executor the worker runs, until
no job is claimable. A job a running server holds stays with that server, a job whose attempt
fails waits out its backoff or stays failed after its last attempt, and the entry of every job
that remains is left out of the CLI's own pass.

Every statement runs as the maintenance identity on the CLI's table-owner connection. A
database prepared by a release without deferred generation has no ``generation_jobs`` table
and therefore no jobs; :func:`generation_jobs_table_exists` tells the CLI which case it has.
"""

import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE

if TYPE_CHECKING:
    from app.repositories import RepositoryContainer

# The condition a context entry aliased ``ce`` meets while it holds no generation job.
HOLDS_NO_JOB = 'NOT EXISTS (SELECT 1 FROM generation_jobs gj WHERE gj.context_entry_id = ce.id)'

_COUNT_SQL = 'SELECT status, COUNT(*) FROM generation_jobs GROUP BY status'


@dataclass(frozen=True, slots=True)
class JobCounts:
    """The outstanding generation jobs of a database.

    Attributes:
        pending: Jobs waiting for a claim or running under one.
        failed: Jobs whose last attempt failed.
    """

    pending: int
    failed: int

    @property
    def total(self) -> int:
        """Every outstanding job."""
        return self.pending + self.failed

    def describe(self) -> str:
        """Name the counts, as in ``2 pending and 1 failed generation jobs``."""
        return f'{self.pending} pending and {self.failed} failed generation jobs'


@dataclass(frozen=True, slots=True)
class DrainReport:
    """What a drain did.

    Attributes:
        reset: Failed jobs returned to the queue before the drain claimed any.
        completed: Jobs whose outputs were stored.
        later: Jobs whose attempt failed with attempts left, or whose failure could not be
            recorded; each is claimed again once its backoff or lease ends.
        failed: Jobs whose last attempt failed.
        discarded: Jobs whose entry, text or claim changed while they ran; nothing was stored.
    """

    reset: int
    completed: int
    later: int
    failed: int
    discarded: int

    def describe(self) -> str:
        """Name every count in one sentence."""
        returned = f'{self.reset} failed job' if self.reset == 1 else f'{self.reset} failed jobs'
        return (
            f'Generation jobs: {self.completed} completed, {self.later} left for a later attempt, '
            f'{self.failed} failed, {self.discarded} discarded ({returned} returned to the queue first).'
        )


async def generation_jobs_table_exists(backend: StorageBackend) -> bool:
    """Return whether the database holds the ``generation_jobs`` table.

    Args:
        backend: The CLI's backend.

    Returns:
        True when the table is reachable by its unqualified name: ``sqlite_master`` on
        SQLite, ``to_regclass`` through the connection's search_path on PostgreSQL.
    """
    if backend.backend_type == 'sqlite':

        def _probe_sqlite(conn: sqlite3.Connection) -> bool:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'generation_jobs'").fetchone()
            return row is not None

        return await backend.execute_read(_probe_sqlite, access_scope=MAINTENANCE)

    async def _probe_postgresql(conn: asyncpg.Connection) -> bool:
        return bool(await conn.fetchval("SELECT to_regclass('generation_jobs') IS NOT NULL"))

    return await backend.execute_read(cast(Any, _probe_postgresql), access_scope=MAINTENANCE)


async def count_generation_jobs(backend: StorageBackend) -> JobCounts:
    """Count the outstanding generation jobs, a running job as pending.

    Args:
        backend: The CLI's backend; the database must hold the ``generation_jobs`` table.

    Returns:
        The pending and failed jobs.
    """
    if backend.backend_type == 'sqlite':

        def _count_sqlite(conn: sqlite3.Connection) -> list[tuple[str, int]]:
            return [(str(row[0]), int(row[1])) for row in conn.execute(_COUNT_SQL).fetchall()]

        by_status = await backend.execute_read(_count_sqlite, access_scope=MAINTENANCE)
    else:

        async def _count_postgresql(conn: asyncpg.Connection) -> list[tuple[str, int]]:
            return [(str(row[0]), int(row[1])) for row in await conn.fetch(_COUNT_SQL)]

        by_status = await backend.execute_read(cast(Any, _count_postgresql), access_scope=MAINTENANCE)
    failed = sum(count for status, count in by_status if status == 'failed')
    return JobCounts(pending=sum(count for _status, count in by_status) - failed, failed=failed)


async def drain_generation_jobs(repos: 'RepositoryContainer', *, max_attempts: int) -> DrainReport:
    """Return the failed jobs to the queue, then run every claimable job once, oldest first.

    Each job is claimed under a lease, so a server running at the same time never runs the
    same job, and a job the server holds is never claimed here. The drain ends once no job is
    claimable: a job whose attempt failed waits out its backoff, so no job runs twice.

    Args:
        repos: Repository container over the CLI's backend.
        max_attempts: The attempts a job may spend.

    Returns:
        What the drain did.
    """
    # The executor brings in the whole generation pipeline; importing it here keeps that
    # chain out of the import of the CLI modules, which defer their generation imports too.
    from app.generation_jobs.executor import LEASE_S
    from app.generation_jobs.executor import JobOutcome
    from app.generation_jobs.executor import run_job

    reset = await repos.generation_jobs.reset_failed(scope=MAINTENANCE)
    outcomes: Counter[JobOutcome] = Counter()
    while True:
        job = await repos.generation_jobs.claim_next(max_attempts=max_attempts, lease_s=LEASE_S, scope=MAINTENANCE)
        if job is None:
            break
        outcomes[await run_job(repos, job, access=MAINTENANCE, max_attempts=max_attempts)] += 1
    return DrainReport(
        reset=reset,
        completed=outcomes[JobOutcome.COMPLETED],
        later=outcomes[JobOutcome.RETRYING] + outcomes[JobOutcome.UNRECORDED],
        failed=outcomes[JobOutcome.FAILED],
        discarded=outcomes[JobOutcome.DISCARDED],
    )


async def complete_generation_jobs(repos: 'RepositoryContainer', *, max_attempts: int) -> None:
    """Drain the job queue and print what the drain did and which jobs remain.

    Args:
        repos: Repository container over the CLI's backend.
        max_attempts: The attempts a job may spend.
    """
    report = await drain_generation_jobs(repos, max_attempts=max_attempts)
    print(report.describe(), file=sys.stderr)
    remaining = await count_generation_jobs(repos.generation_jobs.backend)
    if remaining.total:
        print(
            f'{remaining.describe()} remain (held by a running server, waiting for a retry, or failed); '
            'their entries are left out of this pass.',
            file=sys.stderr,
        )


__all__ = [
    'HOLDS_NO_JOB',
    'DrainReport',
    'JobCounts',
    'complete_generation_jobs',
    'count_generation_jobs',
    'drain_generation_jobs',
    'generation_jobs_table_exists',
]
