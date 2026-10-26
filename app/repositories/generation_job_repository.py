"""Generation job repository: the queue of entries whose generated outputs are outstanding.

``generation_jobs`` holds one row per entry whose embeddings, summary or index-tree node
summaries are still to be generated (``pending``, or ``running`` under a worker's claim)
or whose generation gave up (``failed``); an entry without a row is ``ready``. The state
reads report exactly those three values, a running job as ``pending``.

The write paths add and remove an entry's job inside the transaction that writes the
entry, so a job always belongs to the stored text. A worker claims the oldest available
job under a lease, renews the lease while it generates, and then deletes the job with the
outputs it wrote, records the failure of the attempt, or gives the claim back when it
stops. Each of those statements is keyed on the claim's token, so a job that was replaced,
reset or claimed again since is never touched under an older claim.

Every time a statement compares or sets is the database clock's: ``now()`` on PostgreSQL,
and on SQLite ``datetime('now')``, the form ``CURRENT_TIMESTAMP`` stores.
"""

import secrets
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.backends.base import BackendAccess
from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import chunk_ids
from app.types import GenerationStatus

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

# The error a job is failed with when its lease ended after its last attempt.
JOB_ABANDONED_ERROR = 'the process running this job stopped before it finished'

# The longest failure text a job keeps.
LAST_ERROR_MAX_LENGTH = 2000

_RETRY_BASE_DELAY_S = 30
_RETRY_DELAY_FACTOR = 4
_RETRY_MAX_DELAY_S = 480


def retry_delay_s(attempt: int) -> int:
    """Return how long a job waits for its next claim after its ``attempt``-th attempt failed.

    Args:
        attempt: The number of the failed attempt, from 1.

    Returns:
        The wait in seconds: 30 after the first attempt, four times longer after each
        further one, at most 480.
    """
    return min(int(_RETRY_BASE_DELAY_S * _RETRY_DELAY_FACTOR ** (attempt - 1)), _RETRY_MAX_DELAY_S)


def _now_sql(backend_type: str) -> str:
    """Render the database's current time in the form the job timestamps store."""
    return "datetime('now')" if backend_type == 'sqlite' else 'now()'


def _later_sql(backend_type: str, seconds: str) -> str:
    """Render the database time ``seconds`` (an integer expression) from now."""
    if backend_type == 'sqlite':
        return f"datetime('now', '+' || ({seconds}) || ' seconds')"
    return f'now() + make_interval(secs => ({seconds})::integer)'


def _affected(status: str) -> int:
    """Return the row count of a PostgreSQL command status such as ``UPDATE 2``."""
    return int(status.split()[-1]) if status else 0


def _state(status: object) -> GenerationStatus:
    """Return the generation state a stored job status, or a missing job, reports."""
    if status is None:
        return 'ready'
    return 'failed' if status == 'failed' else 'pending'


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    """A job a worker claimed.

    Attributes:
        context_id: The id of the job's entry.
        attempts: The claims the job has spent, this one included.
        claim_token: The token every statement about this claim is keyed on.
    """

    context_id: str
    attempts: int
    claim_token: str


@dataclass(frozen=True, slots=True)
class RecordedFailure:
    """What recording the failure of an attempt did to its job.

    Attributes:
        attempts: The claims the job has spent.
        failed: Whether the job gave up: the failed attempt was its last.
        retry_in_s: The seconds until the job may be claimed again; None when it failed.
    """

    attempts: int
    failed: bool
    retry_in_s: int | None


class GenerationJobRepository(BaseRepository):
    """Repository of the generation jobs of context entries.

    ``enqueue``, ``delete_for_entry`` and ``state_for`` run inside the transaction that
    writes the entry; ``states_for_ids`` reads the state of many entries for a response.
    ``claim_next``, ``renew_lease``, ``record_failure``, ``release_claims``,
    ``reset_failed`` and ``delete_job`` drive the queue for a worker, and
    ``delete_claimed`` removes a completed job inside the transaction that writes its
    outputs.
    """

    async def enqueue(self, context_id: str, *, txn: 'TransactionContext') -> None:
        """Add a pending job, claimable at once, for an entry that holds none.

        Args:
            context_id: The entry's id.
            txn: The transaction that writes the entry.
        """
        sql = f'INSERT INTO generation_jobs (context_entry_id) VALUES ({self._placeholder(1)})'
        await self._run_in_transaction(txn, sql, [context_id])

    async def delete_for_entry(self, context_id: str, *, txn: 'TransactionContext') -> bool:
        """Delete an entry's job, whatever its state.

        Args:
            context_id: The entry's id.
            txn: The transaction that writes the entry.

        Returns:
            Whether the entry held a job.
        """
        sql = f'DELETE FROM generation_jobs WHERE context_entry_id = {self._placeholder(1)}'
        return await self._run_in_transaction(txn, sql, [context_id]) > 0

    def _states_sql(self, count: int) -> str:
        """Render the read of the job status of ``count`` entries."""
        jobs = self._readable('generation_jobs', alias='gj')
        return f'SELECT gj.context_entry_id, gj.status FROM {jobs} WHERE gj.context_entry_id IN ({self._placeholders(count)})'

    async def state_for(self, context_id: str, *, txn: 'TransactionContext') -> GenerationStatus:
        """Return an entry's generation state as the transaction sees it.

        Args:
            context_id: The entry's id.
            txn: The transaction that wrote the entry.

        Returns:
            'ready' when the entry holds no job, 'failed' when its job failed, else 'pending'.
        """
        sql = self._states_sql(1)
        if txn.backend_type == 'sqlite':

            def _read_sqlite(conn: sqlite3.Connection) -> object:
                row = conn.execute(sql, (context_id,)).fetchone()
                return None if row is None else row['status']

            status = await self._run_sqlite_txn(_read_sqlite, cast(sqlite3.Connection, txn.connection))
        else:
            row = await cast('asyncpg.Connection', txn.connection).fetchrow(sql, context_id)
            status = None if row is None else row['status']
        return _state(status)

    async def states_for_ids(self, context_ids: Sequence[str], *, scope: BackendAccess) -> dict[str, GenerationStatus]:
        """Return the generation state of each entry, reading the ids in bounded chunks.

        Args:
            context_ids: Ids of entries the caller read, in canonical form.
            scope: The identity the read runs as.

        Returns:
            The state of every requested id, keyed by it in the order given: 'ready' when
            the entry holds no job, 'failed' when its job failed, else 'pending'.
        """
        if not context_ids:
            return {}
        chunks = chunk_ids(list(context_ids))

        if self.backend.backend_type == 'sqlite':

            def _read_sqlite(conn: sqlite3.Connection) -> dict[str, object]:
                found: dict[str, object] = {}
                for chunk in chunks:
                    for row in conn.execute(self._states_sql(len(chunk)), chunk).fetchall():
                        found[str(row['context_entry_id'])] = row['status']
                return found

            found = await self.backend.execute_read(_read_sqlite, access_scope=scope)
        else:

            async def _read_postgresql(conn: 'asyncpg.Connection') -> dict[str, object]:
                found: dict[str, object] = {}
                for chunk in chunks:
                    for row in await conn.fetch(self._states_sql(len(chunk)), *chunk):
                        found[normalize_id(str(row['context_entry_id']))] = row['status']
                return found

            found = await self.backend.execute_read(cast(Any, _read_postgresql), access_scope=scope)
        return {context_id: _state(found.get(context_id)) for context_id in context_ids}

    async def claim_next(self, *, max_attempts: int, lease_s: int, scope: BackendAccess) -> ClaimedJob | None:
        """Claim the oldest available job under a new lease.

        One write first fails every running job whose lease ended after its last attempt,
        then claims the oldest job that is pending or running under an ended lease, is
        available and has attempts left, spending one attempt and setting a fresh claim
        token, and finally reads the claimed job back by that token. On PostgreSQL the
        claim skips a job another transaction holds locked; on SQLite the first statement
        takes the database's write lock, so no two processes claim the same job.

        Args:
            max_attempts: The attempts a job may spend.
            lease_s: The seconds the claim holds the job before another worker may reclaim it.
            scope: The identity the write runs as.

        Returns:
            The claimed job, or None when no job is claimable.
        """
        backend_type = self.backend.backend_type
        placeholder = self._placeholder
        now = _now_sql(backend_type)
        skip_locked = ' FOR UPDATE SKIP LOCKED' if backend_type == 'postgresql' else ''
        sweep_sql = (
            f"UPDATE generation_jobs SET status = 'failed', claim_token = NULL, last_error = {placeholder(1)} "
            f"WHERE status = 'running' AND available_at <= {now} AND attempts >= {placeholder(2)}"
        )
        claim_sql = (
            f"UPDATE generation_jobs SET status = 'running', attempts = attempts + 1, claim_token = {placeholder(1)}, "
            f'available_at = {_later_sql(backend_type, placeholder(2))} '
            'WHERE context_entry_id = ('
            "SELECT context_entry_id FROM generation_jobs WHERE status IN ('pending', 'running') "
            f'AND available_at <= {now} AND attempts < {placeholder(3)} '
            f'ORDER BY created_at, context_entry_id LIMIT 1{skip_locked})'
        )
        claimed_sql = f'SELECT context_entry_id, attempts FROM generation_jobs WHERE claim_token = {placeholder(1)}'

        if backend_type == 'sqlite':

            def _claim_sqlite(conn: sqlite3.Connection) -> ClaimedJob | None:
                token = secrets.token_hex(16)
                conn.execute(sweep_sql, (JOB_ABANDONED_ERROR, max_attempts))
                conn.execute(claim_sql, (token, lease_s, max_attempts))
                row = conn.execute(claimed_sql, (token,)).fetchone()
                return None if row is None else ClaimedJob(str(row['context_entry_id']), int(row['attempts']), token)

            return await self.backend.execute_write(_claim_sqlite, access_scope=scope)

        async def _claim_postgresql(conn: 'asyncpg.Connection') -> ClaimedJob | None:
            token = secrets.token_hex(16)
            await conn.execute(sweep_sql, JOB_ABANDONED_ERROR, max_attempts)
            await conn.execute(claim_sql, token, lease_s, max_attempts)
            row = await conn.fetchrow(claimed_sql, token)
            if row is None:
                return None
            return ClaimedJob(normalize_id(str(row['context_entry_id'])), int(row['attempts']), token)

        return await self.backend.execute_write(cast(Any, _claim_postgresql), access_scope=scope)

    async def renew_lease(self, context_id: str, claim_token: str, *, lease_s: int, scope: BackendAccess) -> bool:
        """Extend a claim's lease to ``lease_s`` seconds from now.

        Args:
            context_id: The job's entry id.
            claim_token: The token of the claim.
            lease_s: The seconds the renewed lease lasts.
            scope: The identity the write runs as.

        Returns:
            Whether the claim still holds the job; False when the job was replaced, reset,
            deleted or claimed again since.
        """
        placeholder = self._placeholder
        sql = (
            f'UPDATE generation_jobs SET available_at = {_later_sql(self.backend.backend_type, placeholder(1))} '
            f'WHERE context_entry_id = {placeholder(2)} AND claim_token = {placeholder(3)}'
        )
        return await self._write(sql, [lease_s, context_id, claim_token], scope=scope) > 0

    async def record_failure(
        self, context_id: str, claim_token: str, error: str, *, max_attempts: int, scope: BackendAccess,
    ) -> RecordedFailure | None:
        """Record the failure of a claim's attempt.

        A job with attempts left returns to the queue, claimable after :func:`retry_delay_s`
        of its attempts; a job whose last attempt failed becomes failed. Either way the
        claim ends and the job keeps ``error``, cut to :data:`LAST_ERROR_MAX_LENGTH`.

        Args:
            context_id: The job's entry id.
            claim_token: The token of the claim whose attempt failed.
            error: The failure text.
            max_attempts: The attempts a job may spend.
            scope: The identity the write runs as.

        Returns:
            What the failure did to the job, or None when the claim no longer holds it.
        """
        backend_type = self.backend.backend_type
        placeholder = self._placeholder
        claimed_sql = (
            f'SELECT attempts FROM generation_jobs '
            f'WHERE context_entry_id = {placeholder(1)} AND claim_token = {placeholder(2)}'
        )
        last_error = error[:LAST_ERROR_MAX_LENGTH]

        def _outcome(attempts: object) -> RecordedFailure:
            spent = int(str(attempts))
            if spent >= max_attempts:
                return RecordedFailure(attempts=spent, failed=True, retry_in_s=None)
            return RecordedFailure(attempts=spent, failed=False, retry_in_s=retry_delay_s(spent))

        def _update(outcome: RecordedFailure) -> tuple[str, list[object]]:
            # A claim's attempts change only under its own token, so the outcome read
            # under the token holds for the update keyed on it.
            if outcome.retry_in_s is None:
                status, available_at, delay = 'failed', _now_sql(backend_type), []
            else:
                status, available_at, delay = 'pending', _later_sql(backend_type, placeholder(2)), [outcome.retry_in_s]
            start = 2 + len(delay)
            sql = (
                f'UPDATE generation_jobs SET status = {placeholder(1)}, available_at = {available_at}, '
                f'claim_token = NULL, last_error = {placeholder(start)} '
                f'WHERE context_entry_id = {placeholder(start + 1)} AND claim_token = {placeholder(start + 2)}'
            )
            return sql, [status, *delay, last_error, context_id, claim_token]

        if backend_type == 'sqlite':

            def _record_sqlite(conn: sqlite3.Connection) -> RecordedFailure | None:
                row = conn.execute(claimed_sql, (context_id, claim_token)).fetchone()
                if row is None:
                    return None
                outcome = _outcome(row['attempts'])
                sql, params = _update(outcome)
                return outcome if conn.execute(sql, params).rowcount > 0 else None

            return await self.backend.execute_write(_record_sqlite, access_scope=scope)

        async def _record_postgresql(conn: 'asyncpg.Connection') -> RecordedFailure | None:
            attempts = await conn.fetchval(claimed_sql, context_id, claim_token)
            if attempts is None:
                return None
            outcome = _outcome(attempts)
            sql, params = _update(outcome)
            return outcome if _affected(await conn.execute(sql, *params)) > 0 else None

        return await self.backend.execute_write(cast(Any, _record_postgresql), access_scope=scope)

    async def delete_claimed(self, context_id: str, claim_token: str, *, txn: 'TransactionContext') -> bool:
        """Delete a job under the claim that holds it.

        Args:
            context_id: The job's entry id.
            claim_token: The token of the claim.
            txn: The transaction that writes the job's outputs.

        Returns:
            Whether the claim still held the job.
        """
        placeholder = self._placeholder
        sql = f'DELETE FROM generation_jobs WHERE context_entry_id = {placeholder(1)} AND claim_token = {placeholder(2)}'
        return await self._run_in_transaction(txn, sql, [context_id, claim_token]) > 0

    async def delete_job(self, context_id: str, claim_token: str, *, scope: BackendAccess) -> bool:
        """Delete a job under the claim that holds it, in a transaction of its own.

        Args:
            context_id: The job's entry id.
            claim_token: The token of the claim.
            scope: The identity the transaction runs as.

        Returns:
            Whether the claim still held the job.
        """
        async with self.backend.begin_transaction(access_scope=scope) as txn:
            return await self.delete_claimed(context_id, claim_token, txn=txn)

    async def release_claims(self, claim_tokens: Sequence[str], *, scope: BackendAccess) -> int:
        """Give claims back: each job becomes pending, claimable at once, with its attempt returned.

        Args:
            claim_tokens: The tokens of the claims to release.
            scope: The identity the write runs as.

        Returns:
            The number of jobs the claims still held.
        """
        if not claim_tokens:
            return 0
        chunks = chunk_ids(list(claim_tokens))
        now = _now_sql(self.backend.backend_type)

        def _release_sql(count: int) -> str:
            return (
                f"UPDATE generation_jobs SET status = 'pending', attempts = attempts - 1, available_at = {now}, "
                f'claim_token = NULL WHERE claim_token IN ({self._placeholders(count)})'
            )

        if self.backend.backend_type == 'sqlite':

            def _release_sqlite(conn: sqlite3.Connection) -> int:
                return sum(conn.execute(_release_sql(len(chunk)), chunk).rowcount for chunk in chunks)

            return await self.backend.execute_write(_release_sqlite, access_scope=scope)

        async def _release_postgresql(conn: 'asyncpg.Connection') -> int:
            released = 0
            for chunk in chunks:
                released += _affected(await conn.execute(_release_sql(len(chunk)), *chunk))
            return released

        return await self.backend.execute_write(cast(Any, _release_postgresql), access_scope=scope)

    async def reset_failed(self, *, scope: BackendAccess) -> int:
        """Return every failed job to the queue with no attempts spent, claimable at once.

        Each job keeps its last error until a completion deletes it.

        Args:
            scope: The identity the write runs as.

        Returns:
            The number of jobs reset.
        """
        now = _now_sql(self.backend.backend_type)
        sql = (
            f"UPDATE generation_jobs SET status = 'pending', attempts = 0, available_at = {now}, claim_token = NULL "
            "WHERE status = 'failed'"
        )
        return await self._write(sql, [], scope=scope)

    async def _write(self, sql: str, params: list[object], *, scope: BackendAccess) -> int:
        """Run one write statement in a transaction of its own and return the rows it changed."""
        if self.backend.backend_type == 'sqlite':

            def _write_sqlite(conn: sqlite3.Connection) -> int:
                return conn.execute(sql, params).rowcount

            return await self.backend.execute_write(_write_sqlite, access_scope=scope)

        async def _write_postgresql(conn: 'asyncpg.Connection') -> int:
            return _affected(await conn.execute(sql, *params))

        return await self.backend.execute_write(cast(Any, _write_postgresql), access_scope=scope)

    async def _run_in_transaction(self, txn: 'TransactionContext', sql: str, params: list[object]) -> int:
        """Run one write statement on the transaction's connection and return the rows it changed."""
        if txn.backend_type == 'sqlite':

            def _run_sqlite(conn: sqlite3.Connection) -> int:
                return conn.execute(sql, params).rowcount

            return await self._run_sqlite_txn(_run_sqlite, cast(sqlite3.Connection, txn.connection))
        return _affected(await cast('asyncpg.Connection', txn.connection).execute(sql, *params))
