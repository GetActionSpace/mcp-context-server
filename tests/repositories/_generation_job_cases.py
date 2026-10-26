"""The generation job repository cases both backends run.

Each case takes a prepared, seeded database (:class:`~tests.repositories._access_scope_cases.ScopedDb`)
and drives :class:`~app.repositories.generation_job_repository.GenerationJobRepository`
through its request channel, reading and shifting the job rows as the table owner through
its seed channel. ``tests/repositories/test_generation_job_repository.py`` runs them on
SQLite and ``tests/integration/postgresql/test_generation_job_repository_postgresql.py`` on
PostgreSQL; each entry point adds the cases only its backend has.

Every timestamp a case compares is the database's own: a case reads how many seconds from
now a job becomes available and moves the job's times relative to the database clock.
"""

import re
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass

from app.backends.base import MAINTENANCE
from app.repositories.generation_job_repository import JOB_ABANDONED_ERROR
from app.repositories.generation_job_repository import LAST_ERROR_MAX_LENGTH
from app.repositories.generation_job_repository import ClaimedJob
from app.repositories.generation_job_repository import GenerationJobRepository
from app.repositories.generation_job_repository import RecordedFailure
from tests.repositories._access_scope_cases import ScopedDb

MAX_ATTEMPTS = 3
LEASE_S = 120

# The tolerance, in seconds, of a comparison against the database clock: a statement and the
# read that checks it run a moment apart, and SQLite keeps whole seconds.
CLOCK_SLACK_S = 2

_TOKEN = re.compile(r'[0-9a-f]{32}')


@dataclass(frozen=True, slots=True)
class JobRow:
    """A job row as the table owner reads it.

    Attributes:
        status: The stored status.
        attempts: The claims the job has spent.
        last_error: The stored failure text.
        claim_token: The token of the current claim.
        available_in_s: Seconds from now, by the database clock, at which the job becomes
            claimable or its lease ends; negative when that time has passed.
    """

    status: str
    attempts: int
    last_error: str | None
    claim_token: str | None
    available_in_s: int


def jobs(db: ScopedDb) -> GenerationJobRepository:
    """Return the job repository of the request channel."""
    return db.repos.generation_jobs


async def add_entry(db: ScopedDb, label: str) -> str:
    """Store one entry owned by alice with no job and no generated outputs, and return its id."""
    return await db.add_entry(label, owner='alice', visibility='private')


async def enqueue(db: ScopedDb, *context_ids: str) -> None:
    """Enqueue a job for each entry in one transaction."""
    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        for context_id in context_ids:
            await jobs(db).enqueue(context_id, txn=txn)


async def replace_job(db: ScopedDb, context_id: str) -> None:
    """Replace an entry's job with a fresh one in one transaction, as a text change does."""
    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        await jobs(db).delete_for_entry(context_id, txn=txn)
        await jobs(db).enqueue(context_id, txn=txn)


async def claim(db: ScopedDb, *, max_attempts: int = MAX_ATTEMPTS) -> ClaimedJob | None:
    """Claim the next job under the standard lease."""
    return await jobs(db).claim_next(max_attempts=max_attempts, lease_s=LEASE_S, scope=MAINTENANCE)


async def state_in_transaction(db: ScopedDb, context_id: str) -> str:
    """Return the entry's generation state as a write transaction reads it."""
    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        return await jobs(db).state_for(context_id, txn=txn)


def _seconds_from_now(db: ScopedDb, column: str) -> str:
    """Render the whole seconds from now, by the database clock, to the time in ``column``."""
    if db.seed_backend.backend_type == 'sqlite':
        return f"CAST(strftime('%s', {column}) AS INTEGER) - CAST(strftime('%s', 'now') AS INTEGER)"
    return f'CAST(round(EXTRACT(EPOCH FROM {column} - now())) AS INTEGER)'


def _now_plus(db: ScopedDb, seconds: int) -> tuple[str, object]:
    """Render the database time ``seconds`` from now, with the value it binds."""
    if db.seed_backend.backend_type == 'sqlite':
        return "datetime('now', ?)", f'{seconds:+d} seconds'
    return 'now() + make_interval(secs => ?)', float(seconds)


async def read_job(db: ScopedDb, context_id: str) -> JobRow | None:
    """Read an entry's job row as the table owner, or None when it has none."""
    rows = await db.fetch_all(
        f'SELECT status, attempts, last_error, claim_token, {_seconds_from_now(db, "available_at")} '
        'FROM generation_jobs WHERE context_entry_id = ?',
        [context_id],
    )
    if not rows:
        return None
    status, attempts, last_error, claim_token, available_in_s = rows[0]
    return JobRow(
        status=str(status),
        attempts=int(str(attempts)),
        last_error=None if last_error is None else str(last_error),
        claim_token=None if claim_token is None else str(claim_token),
        available_in_s=int(str(available_in_s)),
    )


async def require_job(db: ScopedDb, context_id: str) -> JobRow:
    """Read an entry's job row, which must exist."""
    job = await read_job(db, context_id)
    assert job is not None, f'entry {context_id} holds no job'
    return job


async def shift_job(db: ScopedDb, context_id: str, *, column: str, seconds: int) -> None:
    """Set the job's ``available_at`` or ``created_at`` to ``seconds`` from now by the database clock."""
    assert column in {'available_at', 'created_at'}
    expression, value = _now_plus(db, seconds)
    await db.execute(f'UPDATE generation_jobs SET {column} = {expression} WHERE context_entry_id = ?', [value, context_id])


async def put_failed_job(db: ScopedDb, context_id: str, *, attempts: int, last_error: str) -> None:
    """Store a failed job for the entry as the table owner, the state a job reaches after its last attempt."""
    await db.execute(
        "INSERT INTO generation_jobs (context_entry_id, status, attempts, last_error) VALUES (?, 'failed', ?, ?)",
        [context_id, attempts, last_error],
    )


def assert_available_in(job: JobRow, seconds: int) -> None:
    """Assert the job becomes available ``seconds`` from now, within the clock slack."""
    assert seconds - CLOCK_SLACK_S <= job.available_in_s <= seconds, job


async def case_enqueue_creates_a_pending_claimable_job(db: ScopedDb) -> None:
    """A new job is pending with no attempt, no error and no claim, claimable at once."""
    queued = await add_entry(db, 'queued')
    untouched = await add_entry(db, 'untouched')

    await enqueue(db, queued)

    job = await require_job(db, queued)
    assert (job.status, job.attempts, job.last_error, job.claim_token) == ('pending', 0, None, None)
    assert_available_in(job, 0)
    assert await state_in_transaction(db, queued) == 'pending'
    assert await state_in_transaction(db, untouched) == 'ready'
    assert await read_job(db, untouched) is None


async def case_delete_for_entry_removes_the_job(db: ScopedDb) -> None:
    """Deleting an entry's job reports whether one existed, and the entry then reads as ready."""
    entry = await add_entry(db, 'superseded')
    await enqueue(db, entry)

    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        assert await jobs(db).delete_for_entry(entry, txn=txn) is True
        assert await jobs(db).state_for(entry, txn=txn) == 'ready'
    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        assert await jobs(db).delete_for_entry(entry, txn=txn) is False

    assert await read_job(db, entry) is None


async def case_states_map_every_requested_id(db: ScopedDb) -> None:
    """A running job reads as pending, a failed one as failed and an entry without a job as ready."""
    running = await add_entry(db, 'running')
    pending = await add_entry(db, 'pending')
    failed = await add_entry(db, 'failed')
    ready = await add_entry(db, 'ready')
    await enqueue(db, running)
    assert await claim(db) is not None
    await enqueue(db, pending)
    await put_failed_job(db, failed, attempts=MAX_ATTEMPTS, last_error='provider down')

    states = await jobs(db).states_for_ids([running, pending, failed, ready], scope=MAINTENANCE)

    assert states == {running: 'pending', pending: 'pending', failed: 'failed', ready: 'ready'}
    assert await jobs(db).states_for_ids([], scope=MAINTENANCE) == {}


async def case_claim_runs_the_job_under_a_lease(db: ScopedDb) -> None:
    """A claim spends one attempt, sets a fresh token and holds the job until its lease ends."""
    entry = await add_entry(db, 'claimed')
    await enqueue(db, entry)

    claimed = await claim(db)

    assert claimed is not None
    assert (claimed.context_id, claimed.attempts) == (entry, 1)
    assert _TOKEN.fullmatch(claimed.claim_token)
    job = await require_job(db, entry)
    assert (job.status, job.attempts, job.claim_token) == ('running', 1, claimed.claim_token)
    assert_available_in(job, LEASE_S)
    assert await claim(db) is None


async def case_claims_follow_creation_order(db: ScopedDb) -> None:
    """Jobs are claimed oldest first, and jobs created at the same time in id order."""
    first, second, third = [await add_entry(db, label) for label in ('first', 'second', 'third')]
    tied = sorted([await add_entry(db, 'tied-a'), await add_entry(db, 'tied-b')])
    await enqueue(db, third, *tied, first, second)
    for entry, seconds in ((first, -30), (second, -20), (third, -10), (tied[0], -40), (tied[1], -40)):
        await shift_job(db, entry, column='created_at', seconds=seconds)

    order = [job.context_id for job in [await claim(db) for _ in range(5)] if job is not None]

    assert order == [*tied, first, second, third]
    assert await claim(db) is None


async def case_jobs_not_yet_available_or_failed_are_not_claimed(db: ScopedDb) -> None:
    """A job waiting out its backoff and a failed job are never claimed."""
    waiting = await add_entry(db, 'waiting')
    failed = await add_entry(db, 'failed-job')
    await enqueue(db, waiting)
    await shift_job(db, waiting, column='available_at', seconds=60)
    await put_failed_job(db, failed, attempts=MAX_ATTEMPTS, last_error='provider down')

    assert await claim(db) is None

    assert (await require_job(db, waiting)).status == 'pending'
    assert (await require_job(db, failed)).status == 'failed'


async def case_an_expired_lease_is_reclaimed(db: ScopedDb) -> None:
    """A running job whose lease ended is claimed again with a new token and another attempt."""
    entry = await add_entry(db, 'abandoned')
    await enqueue(db, entry)
    first = await claim(db)
    assert first is not None
    await shift_job(db, entry, column='available_at', seconds=-1)

    second = await claim(db)

    assert second is not None
    assert (second.context_id, second.attempts) == (entry, 2)
    assert second.claim_token != first.claim_token
    assert await jobs(db).renew_lease(entry, first.claim_token, lease_s=LEASE_S, scope=MAINTENANCE) is False
    assert await jobs(db).renew_lease(entry, second.claim_token, lease_s=LEASE_S, scope=MAINTENANCE) is True


async def case_an_expired_lease_at_the_cap_fails_the_job(db: ScopedDb) -> None:
    """A running job whose lease ended after its last attempt becomes failed instead of being claimed."""
    entry = await add_entry(db, 'exhausted')
    await enqueue(db, entry)
    assert await claim(db, max_attempts=1) is not None
    await shift_job(db, entry, column='available_at', seconds=-1)

    assert await claim(db, max_attempts=1) is None

    job = await require_job(db, entry)
    assert (job.status, job.attempts, job.claim_token, job.last_error) == ('failed', 1, None, JOB_ABANDONED_ERROR)


async def case_renewal_extends_the_current_lease(db: ScopedDb) -> None:
    """Renewing with the current token moves the lease end to a full lease from now."""
    entry = await add_entry(db, 'renewed')
    await enqueue(db, entry)
    claimed = await claim(db)
    assert claimed is not None
    await shift_job(db, entry, column='available_at', seconds=5)

    assert await jobs(db).renew_lease(entry, claimed.claim_token, lease_s=LEASE_S, scope=MAINTENANCE) is True

    job = await require_job(db, entry)
    assert (job.status, job.claim_token) == ('running', claimed.claim_token)
    assert_available_in(job, LEASE_S)


async def case_a_failure_schedules_a_retry_with_backoff(db: ScopedDb) -> None:
    """A failed attempt below the cap returns the job to the queue 30 seconds out, then 120."""
    entry = await add_entry(db, 'retried')
    await enqueue(db, entry)

    for attempt, delay in ((1, 30), (2, 120)):
        claimed = await claim(db)
        assert claimed is not None
        assert claimed.attempts == attempt

        failure = await jobs(db).record_failure(
            entry, claimed.claim_token, f'provider down {attempt}', max_attempts=MAX_ATTEMPTS, scope=MAINTENANCE,
        )

        assert failure == RecordedFailure(attempts=attempt, failed=False, retry_in_s=delay)
        job = await require_job(db, entry)
        assert (job.status, job.attempts, job.claim_token, job.last_error) == (
            'pending', attempt, None, f'provider down {attempt}',
        )
        assert_available_in(job, delay)
        assert await claim(db) is None
        await shift_job(db, entry, column='available_at', seconds=-1)


async def case_a_failure_at_the_cap_fails_the_job(db: ScopedDb) -> None:
    """The last attempt's failure leaves the job failed, available now, with its error cut to the stored length."""
    entry = await add_entry(db, 'given-up')
    await enqueue(db, entry)
    claimed = await claim(db, max_attempts=1)
    assert claimed is not None

    failure = await jobs(db).record_failure(
        entry, claimed.claim_token, 'x' * (LAST_ERROR_MAX_LENGTH + 500), max_attempts=1, scope=MAINTENANCE,
    )

    assert failure == RecordedFailure(attempts=1, failed=True, retry_in_s=None)
    job = await require_job(db, entry)
    assert (job.status, job.attempts, job.claim_token) == ('failed', 1, None)
    assert job.last_error == 'x' * LAST_ERROR_MAX_LENGTH
    assert_available_in(job, 0)
    assert await claim(db, max_attempts=1) is None


async def case_a_replaced_job_ignores_the_old_claim(db: ScopedDb) -> None:
    """Neither a failure nor a renewal under the token of a replaced job touches the new job."""
    entry = await add_entry(db, 'replaced')
    await enqueue(db, entry)
    claimed = await claim(db)
    assert claimed is not None
    await replace_job(db, entry)

    failure = await jobs(db).record_failure(
        entry, claimed.claim_token, 'stale failure', max_attempts=MAX_ATTEMPTS, scope=MAINTENANCE,
    )
    renewed = await jobs(db).renew_lease(entry, claimed.claim_token, lease_s=LEASE_S, scope=MAINTENANCE)

    assert failure is None
    assert renewed is False
    job = await require_job(db, entry)
    assert (job.status, job.attempts, job.claim_token, job.last_error) == ('pending', 0, None, None)


async def case_delete_claimed_requires_the_current_token(db: ScopedDb) -> None:
    """A completion deletes the job only under the token it claimed it with."""
    entry = await add_entry(db, 'completed')
    await enqueue(db, entry)
    claimed = await claim(db)
    assert claimed is not None

    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        assert await jobs(db).delete_claimed(entry, 'f' * 32, txn=txn) is False
    assert (await require_job(db, entry)).status == 'running'
    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        assert await jobs(db).delete_claimed(entry, claimed.claim_token, txn=txn) is True

    assert await read_job(db, entry) is None


async def case_delete_job_requires_the_current_token(db: ScopedDb) -> None:
    """Dropping the job of a vanished entry deletes it only under the token it was claimed with."""
    entry = await add_entry(db, 'orphaned')
    await enqueue(db, entry)
    claimed = await claim(db)
    assert claimed is not None

    assert await jobs(db).delete_job(entry, 'f' * 32, scope=MAINTENANCE) is False
    assert (await require_job(db, entry)).status == 'running'
    assert await jobs(db).delete_job(entry, claimed.claim_token, scope=MAINTENANCE) is True

    assert await read_job(db, entry) is None


async def case_release_returns_claims_without_spending_attempts(db: ScopedDb) -> None:
    """Releasing claims makes each job pending and claimable again with the attempt given back."""
    first, second, queued = [await add_entry(db, label) for label in ('released-a', 'released-b', 'still-queued')]
    await enqueue(db, first, second)
    claims = [await claim(db), await claim(db)]
    await enqueue(db, queued)
    tokens = [job.claim_token for job in claims if job is not None]
    assert len(tokens) == 2

    released = await jobs(db).release_claims([*tokens, 'f' * 32], scope=MAINTENANCE)

    assert released == 2
    for entry in (first, second):
        job = await require_job(db, entry)
        assert (job.status, job.attempts, job.claim_token) == ('pending', 0, None)
        assert_available_in(job, 0)
    assert (await require_job(db, queued)).attempts == 0
    assert await jobs(db).release_claims([], scope=MAINTENANCE) == 0


async def case_reset_failed_requeues_failed_jobs_and_keeps_their_error(db: ScopedDb) -> None:
    """The restart reset makes failed jobs pending with no attempts, keeping the last error, and nothing else."""
    failed = await add_entry(db, 'reset')
    running = await add_entry(db, 'reset-running')
    pending = await add_entry(db, 'reset-pending')
    await put_failed_job(db, failed, attempts=MAX_ATTEMPTS, last_error='provider down')
    await enqueue(db, running)
    claimed = await claim(db)
    assert claimed is not None
    await enqueue(db, pending)
    await shift_job(db, pending, column='available_at', seconds=60)

    reset = await jobs(db).reset_failed(scope=MAINTENANCE)

    assert reset == 1
    job = await require_job(db, failed)
    assert (job.status, job.attempts, job.claim_token, job.last_error) == ('pending', 0, None, 'provider down')
    assert_available_in(job, 0)
    running_job = await require_job(db, running)
    assert (running_job.status, running_job.attempts, running_job.claim_token) == ('running', 1, claimed.claim_token)
    pending_job = await require_job(db, pending)
    assert pending_job.status == 'pending'
    assert_available_in(pending_job, 60)
    reclaimed = await claim(db)
    assert reclaimed is not None
    assert reclaimed.context_id == failed


async def case_deleting_the_entry_deletes_its_job(db: ScopedDb) -> None:
    """A job goes with its entry."""
    entry = await add_entry(db, 'deleted')
    await enqueue(db, entry)

    await db.execute('DELETE FROM context_entries WHERE id = ?', [entry])

    assert await read_job(db, entry) is None


GENERATION_JOB_CASES: tuple[Callable[[ScopedDb], Awaitable[None]], ...] = (
    case_enqueue_creates_a_pending_claimable_job,
    case_delete_for_entry_removes_the_job,
    case_states_map_every_requested_id,
    case_claim_runs_the_job_under_a_lease,
    case_claims_follow_creation_order,
    case_jobs_not_yet_available_or_failed_are_not_claimed,
    case_an_expired_lease_is_reclaimed,
    case_an_expired_lease_at_the_cap_fails_the_job,
    case_renewal_extends_the_current_lease,
    case_a_failure_schedules_a_retry_with_backoff,
    case_a_failure_at_the_cap_fails_the_job,
    case_a_replaced_job_ignores_the_old_claim,
    case_delete_claimed_requires_the_current_token,
    case_delete_job_requires_the_current_token,
    case_release_returns_claims_without_spending_attempts,
    case_reset_failed_requeues_failed_jobs_and_keeps_their_error,
    case_deleting_the_entry_deletes_its_job,
)
