"""The generation job drain cases both backends run.

Each case takes a prepared, seeded database (:class:`~tests.repositories._access_scope_cases.ScopedDb`)
and the :class:`DrainGeneration` recorder that stands in for the generation the executor runs,
and drives :mod:`app.cli._generation_drain` and the entry reads of the two embedding modes
against it. ``tests/cli/test_generation_drain.py`` runs them on SQLite and
``tests/integration/postgresql/test_generation_drain_postgresql.py`` on PostgreSQL.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from unittest.mock import MagicMock

import pytest
from fastmcp.exceptions import ToolError

import app.generation_jobs.executor as executor
import app.tools._generation as generation_module
from app.cli._generation_drain import DrainReport
from app.cli._generation_drain import JobCounts
from app.cli._generation_drain import count_generation_jobs
from app.cli._generation_drain import drain_generation_jobs
from app.cli._generation_drain import generation_jobs_table_exists
from app.cli.migrate_embeddings import _find_missing_entries
from app.cli.migrate_reembed import _find_all_entries
from app.ids import normalize_id
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.generation_job_repository import retry_delay_s
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import vector_at
from tests.repositories._generation_job_cases import CLOCK_SLACK_S
from tests.repositories._generation_job_cases import MAX_ATTEMPTS
from tests.repositories._generation_job_cases import claim
from tests.repositories._generation_job_cases import enqueue
from tests.repositories._generation_job_cases import read_job
from tests.repositories._generation_job_cases import require_job
from tests.repositories._generation_job_cases import shift_job

GENERATED_SUMMARY = 'Generated summary.'

# Long enough to reach SUMMARY_MIN_CONTENT_LENGTH, so the summary leg runs.
_BODY = ' A sentence long enough to be summarized.' * 20


@dataclass
class DrainGeneration:
    """Records the texts it generates outputs for; returns one embedding and a summary, or raises.

    Attributes:
        error: What a generation raises instead of returning, if anything.
        texts: The text of every generation, in order.
    """

    error: Exception | None = None
    texts: list[str] = field(default_factory=list[str])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Activate the embedding and summary legs and make this recorder the executor's generation."""
        monkeypatch.setattr(executor, 'get_embedding_provider', MagicMock)
        monkeypatch.setattr(generation_module, 'get_summary_provider', MagicMock)
        monkeypatch.setattr(executor, 'node_layer_active', lambda: False)
        monkeypatch.setattr(executor, 'run_generation', self)

    async def __call__(
        self, text: str, source: str, *, run_embedding: bool, run_summary: bool, run_nodes: bool,
    ) -> tuple[list[ChunkEmbedding] | None, str | None, None]:
        """Record ``text``, then raise the error or return the outputs of the legs that run."""
        del source, run_nodes
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        chunks = [ChunkEmbedding(embedding=vector_at(0.0), start_index=0, end_index=len(text))] if run_embedding else None
        return chunks, GENERATED_SUMMARY if run_summary else None, None


def text_of(label: str) -> str:
    """Return the text of the entry labeled ``label``."""
    return f'Text of {label}.{_BODY}'


def drain_report(*, reset: int = 0, completed: int = 0, later: int = 0, failed: int = 0, discarded: int = 0) -> DrainReport:
    """Return a drain report with the given counts."""
    return DrainReport(reset=reset, completed=completed, later=later, failed=failed, discarded=discarded)


async def add_entry(db: ScopedDb, label: str) -> str:
    """Store an entry owned by alice with no job and no generated outputs, and return its id."""
    return await db.add_entry(label, owner='alice', visibility='private', text=text_of(label))


async def entry_with_job(db: ScopedDb, label: str, *, created_in_s: int = 0) -> str:
    """Store an entry owned by alice, enqueue its job and date the job ``created_in_s`` seconds from now."""
    context_id = await add_entry(db, label)
    await enqueue(db, context_id)
    await shift_job(db, context_id, column='created_at', seconds=created_in_s)
    return context_id


async def fail_job(db: ScopedDb, context_id: str) -> None:
    """Fail the entry's job after its last attempt, as a worker does."""
    await db.execute(
        "UPDATE generation_jobs SET status = 'failed', attempts = ?, last_error = 'provider down' "
        'WHERE context_entry_id = ?',
        [MAX_ATTEMPTS, context_id],
    )


async def _summary_of(db: ScopedDb, context_id: str) -> object:
    """Read the entry's stored summary."""
    rows = await db.fetch_all('SELECT summary FROM context_entries WHERE id = ?', [context_id])
    return rows[0][0]


async def _embedded(db: ScopedDb, context_id: str) -> bool:
    """Whether the entry stores embeddings."""
    rows = await db.fetch_all('SELECT COUNT(*) FROM embedding_metadata WHERE context_id = ?', [context_id])
    return int(str(rows[0][0])) > 0


async def case_pending_and_failed_jobs_are_completed_oldest_first(db: ScopedDb, generation: DrainGeneration) -> None:
    """A failed job returns to the queue first; then every job runs, oldest first, and stores its outputs."""
    first = await entry_with_job(db, 'first', created_in_s=-30)
    failed = await entry_with_job(db, 'failed', created_in_s=-20)
    last = await entry_with_job(db, 'last', created_in_s=-10)
    await fail_job(db, failed)

    report = await drain_generation_jobs(db.repos, max_attempts=MAX_ATTEMPTS)

    assert report == drain_report(reset=1, completed=3)
    assert generation.texts == [text_of('first'), text_of('failed'), text_of('last')]
    for context_id in (first, failed, last):
        assert await read_job(db, context_id) is None
        assert await _embedded(db, context_id)
        assert await _summary_of(db, context_id) == GENERATED_SUMMARY


async def case_a_job_another_process_holds_is_left_to_it(db: ScopedDb, generation: DrainGeneration) -> None:
    """A job running under another claim's live lease is neither run nor touched."""
    held = await entry_with_job(db, 'held')
    other_claim = await claim(db)
    assert other_claim is not None

    report = await drain_generation_jobs(db.repos, max_attempts=MAX_ATTEMPTS)

    assert report == drain_report()
    assert generation.texts == []
    job = await require_job(db, held)
    assert (job.status, job.attempts, job.claim_token) == ('running', 1, other_claim.claim_token)


async def case_a_failed_attempt_returns_the_job_to_the_queue_and_ends_the_drain(
    db: ScopedDb, generation: DrainGeneration,
) -> None:
    """A failed attempt with attempts left waits out its backoff, so the drain runs it once and ends."""
    generation.error = ToolError('Generation failed after exhausting configured retries: embedding: down')
    context_id = await entry_with_job(db, 'flaky')

    report = await drain_generation_jobs(db.repos, max_attempts=MAX_ATTEMPTS)

    assert report == drain_report(later=1)
    assert generation.texts == [text_of('flaky')]
    job = await require_job(db, context_id)
    assert (job.status, job.attempts) == ('pending', 1)
    assert abs(job.available_in_s - retry_delay_s(1)) <= CLOCK_SLACK_S
    assert job.last_error is not None
    assert 'embedding: down' in job.last_error


async def case_the_last_failed_attempt_fails_the_job(db: ScopedDb, generation: DrainGeneration) -> None:
    """A failed attempt that was the job's last leaves the job failed."""
    generation.error = ToolError('Generation failed after exhausting configured retries: summary: down')
    context_id = await entry_with_job(db, 'broken')

    report = await drain_generation_jobs(db.repos, max_attempts=1)

    assert report == drain_report(failed=1)
    job = await require_job(db, context_id)
    assert (job.status, job.attempts) == ('failed', 1)


async def case_running_jobs_count_as_pending_and_failed_jobs_as_failed(db: ScopedDb, generation: DrainGeneration) -> None:
    """A pending and a running job are pending; a failed job is failed."""
    del generation
    await entry_with_job(db, 'running', created_in_s=-10)
    await entry_with_job(db, 'pending')
    failed = await entry_with_job(db, 'failed')
    assert await claim(db) is not None
    await fail_job(db, failed)

    assert await count_generation_jobs(db.backend) == JobCounts(pending=2, failed=1)


async def case_a_database_without_jobs_counts_none(db: ScopedDb, generation: DrainGeneration) -> None:
    """A database whose entries hold no job counts zero of each."""
    del generation
    assert await count_generation_jobs(db.backend) == JobCounts(pending=0, failed=0)


async def case_the_table_probe_follows_the_table(db: ScopedDb, generation: DrainGeneration) -> None:
    """The probe finds the table a prepared database holds, and not one that was dropped."""
    del generation
    assert await generation_jobs_table_exists(db.backend)

    await db.execute('DROP TABLE generation_jobs')

    assert not await generation_jobs_table_exists(db.backend)


async def case_the_entry_reads_leave_job_holders_out(db: ScopedDb, generation: DrainGeneration) -> None:
    """Both embedding modes' entry reads leave out an entry that holds a job, unless told the table is absent."""
    del generation
    free = await add_entry(db, 'free')
    queued = await entry_with_job(db, 'queued')

    async def read_ids(read: Callable[..., Awaitable[list[tuple[str, str]]]], *, skip: bool) -> set[str]:
        return {normalize_id(entry_id) for entry_id, _text in await read(db.backend, skip_job_holders=skip)}

    for read in (_find_missing_entries, _find_all_entries):
        assert free in await read_ids(read, skip=True)
        assert queued not in await read_ids(read, skip=True)
        assert {free, queued} <= await read_ids(read, skip=False)


DRAIN_CASES: tuple[Callable[[ScopedDb, DrainGeneration], Awaitable[None]], ...] = (
    case_pending_and_failed_jobs_are_completed_oldest_first,
    case_a_job_another_process_holds_is_left_to_it,
    case_a_failed_attempt_returns_the_job_to_the_queue_and_ends_the_drain,
    case_the_last_failed_attempt_fails_the_job,
    case_running_jobs_count_as_pending_and_failed_jobs_as_failed,
    case_a_database_without_jobs_counts_none,
    case_the_table_probe_follows_the_table,
    case_the_entry_reads_leave_job_holders_out,
)
