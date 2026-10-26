"""Two-principal cases of the generation job seams (J1-J4).

A generation job is a child row of its entry. The cases that read or delete jobs first give
seven of the eight seed rows a job in one of the three stored states (:data:`JOB_STATUSES`,
written through the seed channel), so every scope reads jobs of more than one state and none
of another principal's private rows; the shared seed itself holds no job.

- J1: ``get_database_statistics`` counts the generation jobs of the entries the caller may
  read, a running job as pending.
- J2: ``GenerationJobRepository.states_for_ids`` reports the state of every entry a READ-scoped
  read returned, as the read tools call it: its job's state, a running job as ``pending``, and
  ``ready`` for an entry without a job.
- J3: ``enqueue`` writes a job on each entry the write gate clears, as the write tools call it:
  the caller's probe inside the transaction, then one enqueue per entry it may modify. Under
  row-level security alone (:data:`RLS_GENERATION_JOB_CASES`), an enqueue that skips the gate
  is refused by the job table's insert policy on every entry the caller may not modify, one it
  may read through a read grant included, while the entries it may modify, through a write
  grant included, get their job.
- J4: ``delete_for_entry`` deletes the job of each entry the write gate clears and reports
  whether it held one. Under row-level security alone, a delete that skips the gate deletes no
  job of an entry the caller may not modify, which the delete policy hides, and deletes those
  of the entries it may modify.
"""

import uuid
from collections.abc import Sequence
from typing import Literal

import asyncpg

from app.access_scope import Scope
from app.backends.base import TransactionContext
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import WRITABLE
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import per_request_scope
from tests.repositories._access_scope_cases import per_scope

type JobStatus = Literal['pending', 'running', 'failed']
type Outcome = tuple[str, object]
type WriteObservable = tuple[tuple[Outcome, ...], tuple[str, ...]]

# The stored job status of each seed row that holds a job in the cases that read or delete jobs.
JOB_STATUSES: dict[str, JobStatus] = {
    'alice_private': 'pending',
    'alice_public': 'failed',
    'alice_private_bob_read': 'running',
    'alice_private_team_read': 'failed',
    'alice_private_bob_write': 'pending',
    'alice_public_carol_write': 'running',
    'bob_private': 'failed',
}
_JOB_LABELS = tuple(label for label in SEED_LABELS if label in JOB_STATUSES)

# The outcome of a target whose job the job table's insert policy refused.
POLICY_REFUSED = 'refused by the insert policy'
ENQUEUED = 'enqueued'
_POLICY_MESSAGE = 'new row violates row-level security policy for table "generation_jobs"'


def _state(label: str) -> str:
    """The generation state a seed row reports: its job's state, a running job as pending, else ready."""
    job = JOB_STATUSES.get(label)
    if job is None:
        return 'ready'
    return 'failed' if job == 'failed' else 'pending'


async def _add_jobs(db: ScopedDb) -> None:
    """Store the job of :data:`JOB_STATUSES` on each seed row through the seed channel; a running job holds a claim."""
    for label, status in JOB_STATUSES.items():
        attempts, claim_token = (1, uuid.uuid4().hex) if status == 'running' else (0, None)
        await db.execute(
            'INSERT INTO generation_jobs (context_entry_id, status, attempts, claim_token) VALUES (?, ?, ?, ?)',
            (db.ids[label], status, attempts, claim_token),
        )


async def _job_labels(db: ScopedDb) -> tuple[str, ...]:
    """Return the labels of the entries holding a job, read through the seed channel, in seed order."""
    rows = await db.fetch_all('SELECT context_entry_id FROM generation_jobs')
    return in_seed_order(db.labels_of(str(row[0]) for row in rows))


async def _generation_counts(db: ScopedDb, scope: Scope) -> dict[str, int]:
    """Return the generation block of ``get_database_statistics`` as the scope."""
    stats = await db.repos.statistics.get_database_statistics(scope=scope)
    return dict(stats['generation'])


def _expected_counts(scope_name: str) -> dict[str, int]:
    """The pending and failed jobs among the rows the scope reads."""
    states = [_state(label) for label in READABLE[scope_name]]
    return {'pending': states.count('pending'), 'failed': states.count('failed')}


async def _states_of_read_entries(db: ScopedDb, scope: Scope) -> tuple[Outcome, ...]:
    """Read every seed row as the scope, then report the generation state of each row it read, in seed order."""
    rows = await db.repos.context.get_by_ids([db.ids[label] for label in SEED_LABELS], scope=scope)
    states = await db.repos.generation_jobs.states_for_ids([str(row['id']) for row in rows], scope=scope)
    by_label = dict(zip(db.labels_of(states), states.values(), strict=True))
    return tuple((label, by_label[label]) for label in in_seed_order(by_label))


def _expected_states(scope_name: str) -> tuple[Outcome, ...]:
    """The generation state of every row the scope reads."""
    return tuple((label, _state(label)) for label in READABLE[scope_name])


async def _cleared_by_the_write_gate(db: ScopedDb, scope: Scope, txn: TransactionContext) -> Sequence[tuple[str, str]]:
    """Return the label and id of every seed row the scope may modify, as its probe inside ``txn`` reports, in seed order."""
    ids = [db.ids[label] for label in SEED_LABELS]
    access = await db.repos.context.probe_ids(ids, scope=scope, txn=txn)
    return [
        (label, context_id)
        for label, context_id in zip(SEED_LABELS, ids, strict=True)
        if context_id in access and access[context_id].can_write
    ]


async def _enqueue_behind_the_write_gate(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
    """Enqueue a job on every seed row the scope's write gate clears, then report the rows holding a job."""
    async with db.backend.begin_transaction(access_scope=scope) as txn:
        for _label, context_id in await _cleared_by_the_write_gate(db, scope, txn):
            await db.repos.generation_jobs.enqueue(context_id, txn=txn)
    return await _job_labels(db)


async def _delete_behind_the_write_gate(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Delete the job of every seed row the scope's write gate clears, then report each deletion and the rows holding a job."""
    outcomes: list[Outcome] = []
    async with db.backend.begin_transaction(access_scope=scope) as txn:
        for label, context_id in await _cleared_by_the_write_gate(db, scope, txn):
            outcomes.append((label, await db.repos.generation_jobs.delete_for_entry(context_id, txn=txn)))
    return tuple(outcomes), await _job_labels(db)


def _expected_gated_delete(scope_name: str) -> WriteObservable:
    """The scope deletes the job of every row it may modify that holds one; the other jobs remain."""
    writable = WRITABLE[scope_name]
    outcomes = tuple((label, label in JOB_STATUSES) for label in writable)
    return outcomes, tuple(label for label in _JOB_LABELS if label not in writable)


def _refusal(error: asyncpg.exceptions.InsufficientPrivilegeError) -> object:
    """Name a refusal by the job table's insert policy; report any other refusal as it is."""
    message = error.args[0] if error.args else ''
    return POLICY_REFUSED if message == _POLICY_MESSAGE else (error.sqlstate, message)


async def _enqueue_without_the_write_gate(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Enqueue a job on every seed row as the scope, each in its own transaction, then report each outcome and the jobs."""
    outcomes: list[Outcome] = []
    for label in SEED_LABELS:
        try:
            async with db.backend.begin_transaction(access_scope=scope) as txn:
                await db.repos.generation_jobs.enqueue(db.ids[label], txn=txn)
        except asyncpg.exceptions.InsufficientPrivilegeError as error:
            outcomes.append((label, _refusal(error)))
        else:
            outcomes.append((label, ENQUEUED))
    return tuple(outcomes), await _job_labels(db)


def _expected_ungated_enqueue(scope_name: str) -> WriteObservable:
    """The insert policy admits a job on exactly the rows the scope may modify and refuses every other."""
    writable = WRITABLE[scope_name]
    outcomes = tuple((label, ENQUEUED if label in writable else POLICY_REFUSED) for label in SEED_LABELS)
    return outcomes, writable


async def _delete_without_the_write_gate(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Delete the job of every seed row as the scope, each in its own transaction, then report each deletion and the jobs."""
    outcomes: list[Outcome] = []
    for label in SEED_LABELS:
        async with db.backend.begin_transaction(access_scope=scope) as txn:
            outcomes.append((label, await db.repos.generation_jobs.delete_for_entry(db.ids[label], txn=txn)))
    return tuple(outcomes), await _job_labels(db)


def _expected_ungated_delete(scope_name: str) -> WriteObservable:
    """The delete policy lets the scope delete the jobs of exactly the rows it may modify."""
    writable = WRITABLE[scope_name]
    outcomes = tuple((label, label in JOB_STATUSES and label in writable) for label in SEED_LABELS)
    return outcomes, tuple(label for label in _JOB_LABELS if label not in writable)


# The cases whose observable the rows each scope may read decide alone.
GENERATION_JOB_READ_CASES: tuple[AccessCase, ...] = (
    AccessCase('J1', _generation_counts, per_scope(_expected_counts), setup=_add_jobs),
    AccessCase('J2', _states_of_read_entries, per_scope(_expected_states), setup=_add_jobs),
)

GENERATION_JOB_WRITE_CASES: tuple[AccessCase, ...] = (
    AccessCase('J3', _enqueue_behind_the_write_gate, per_scope(lambda name: WRITABLE[name])),
    AccessCase('J4', _delete_behind_the_write_gate, per_scope(_expected_gated_delete), setup=_add_jobs),
)

# The cases only row-level security runs: without the write gate in front of them, the job writes
# are limited by the database's policies alone, which no other run enforces.
RLS_GENERATION_JOB_CASES: tuple[AccessCase, ...] = (
    AccessCase(
        'J3', _enqueue_without_the_write_gate, per_request_scope(_expected_ungated_enqueue),
        variant='without-the-write-gate',
    ),
    AccessCase(
        'J4', _delete_without_the_write_gate, per_request_scope(_expected_ungated_delete),
        variant='without-the-write-gate', setup=_add_jobs,
    ),
)
