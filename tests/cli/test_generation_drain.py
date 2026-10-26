"""Tests for the migration CLI's generation job drain in app/cli/_generation_drain.py.

Every case of ``tests/cli/_generation_drain_cases.py`` runs here against a seeded SQLite
database the server's own preparation built, with the executor the worker runs and the
generation replaced by a recorder: the drain returns the failed jobs to the queue, runs every
claimable job once, oldest first, leaves a job another process holds to it, and ends once no
job is claimable. ``tests/integration/postgresql/test_generation_drain_postgresql.py`` runs
the same cases on PostgreSQL. The tests below them cover what does not depend on the
backend: how each outcome is reported and what a completion prints.
"""

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable

import pytest
import pytest_asyncio

import app.generation_jobs.executor as executor
from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from app.cli._generation_drain import DrainReport
from app.cli._generation_drain import complete_generation_jobs
from app.cli._generation_drain import drain_generation_jobs
from app.generation_jobs.executor import JobOutcome
from app.repositories import RepositoryContainer
from app.repositories.generation_job_repository import ClaimedJob
from tests.cli._generation_drain_cases import DRAIN_CASES
from tests.cli._generation_drain_cases import DrainGeneration
from tests.cli._generation_drain_cases import drain_report
from tests.cli._generation_drain_cases import entry_with_job
from tests.cli._generation_drain_cases import fail_job
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import sqlite_scoped_db
from tests.repositories._generation_job_cases import MAX_ATTEMPTS
from tests.repositories._generation_job_cases import claim


@pytest_asyncio.fixture
async def db(async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """A seeded SQLite database prepared as the server prepares one, with fp32 embeddings."""
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as scoped:
        yield scoped


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch) -> DrainGeneration:
    """Activate the embedding and summary legs and replace the executor's generation with a recorder."""
    recorder = DrainGeneration()
    recorder.install(monkeypatch)
    return recorder


@pytest.mark.asyncio
@pytest.mark.parametrize('case', DRAIN_CASES, ids=lambda case: case.__name__.removeprefix('case_'))
async def test_drain_case(
    db: ScopedDb, generation: DrainGeneration, case: Callable[[ScopedDb, DrainGeneration], Awaitable[None]],
) -> None:
    """Each shared case holds on SQLite."""
    await case(db, generation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('outcome', 'expected'),
    [
        pytest.param(JobOutcome.COMPLETED, drain_report(completed=1), id='completed'),
        pytest.param(JobOutcome.RETRYING, drain_report(later=1), id='retrying'),
        pytest.param(JobOutcome.UNRECORDED, drain_report(later=1), id='unrecorded'),
        pytest.param(JobOutcome.FAILED, drain_report(failed=1), id='failed'),
        pytest.param(JobOutcome.DISCARDED, drain_report(discarded=1), id='discarded'),
    ],
)
async def test_each_outcome_is_reported(
    db: ScopedDb, monkeypatch: pytest.MonkeyPatch, outcome: JobOutcome, expected: DrainReport,
) -> None:
    """Every outcome the executor reports is counted under its heading."""

    async def run_job(repos: RepositoryContainer, job: ClaimedJob, *, access: BackendAccess, max_attempts: int) -> JobOutcome:
        del max_attempts
        await repos.generation_jobs.delete_job(job.context_id, job.claim_token, scope=access)
        return outcome

    monkeypatch.setattr(executor, 'run_job', run_job)
    await entry_with_job(db, 'only')

    assert await drain_generation_jobs(db.repos, max_attempts=MAX_ATTEMPTS) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures('generation')
async def test_the_completion_reports_each_outcome_and_the_remaining_jobs(
    db: ScopedDb, capsys: pytest.CaptureFixture[str],
) -> None:
    """A drain that completes two jobs and leaves a held one says so, and says the held entry is skipped."""
    await entry_with_job(db, 'held', created_in_s=-10)
    assert await claim(db) is not None
    await entry_with_job(db, 'done')
    await fail_job(db, await entry_with_job(db, 'reset'))

    await complete_generation_jobs(db.repos, max_attempts=MAX_ATTEMPTS)

    err = capsys.readouterr().err
    assert (
        'Generation jobs: 2 completed, 0 left for a later attempt, 0 failed, 0 discarded '
        '(1 failed job returned to the queue first).'
    ) in err
    assert '1 pending and 0 failed generation jobs remain' in err
    assert 'their entries are left out of this pass' in err


@pytest.mark.asyncio
@pytest.mark.usefixtures('generation')
async def test_a_complete_drain_reports_no_remaining_jobs(db: ScopedDb, capsys: pytest.CaptureFixture[str]) -> None:
    """A drain that leaves no job prints no remaining-jobs line."""
    await entry_with_job(db, 'done')

    await complete_generation_jobs(db.repos, max_attempts=MAX_ATTEMPTS)

    err = capsys.readouterr().err
    assert 'Generation jobs: 1 completed' in err
    assert '(0 failed jobs returned to the queue first)' in err
    assert 'remain' not in err
