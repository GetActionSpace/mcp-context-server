"""Generation job seeding and reading, state read recording and an in-memory job repository double for the tool tests."""

import sqlite3
from collections.abc import Mapping
from collections.abc import Sequence

import pytest

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.repositories import RepositoryContainer
from app.types import GenerationStatus


async def seed_generation_jobs(backend: StorageBackend, statuses: Mapping[str, str]) -> None:
    """Give each entry a generation job in the stored status mapped to its id, as the table owner.

    Args:
        backend: The SQLite backend holding the entries.
        statuses: The stored job status ('pending', 'running' or 'failed') of each entry id.
    """

    def _insert(conn: sqlite3.Connection) -> None:
        conn.executemany(
            'INSERT INTO generation_jobs (context_entry_id, status) VALUES (?, ?)', list(statuses.items()),
        )

    await backend.execute_write(_insert, access_scope=MAINTENANCE)


async def stored_jobs(backend: StorageBackend) -> dict[str, str]:
    """Read the stored status of every generation job, as the table owner.

    Args:
        backend: The SQLite backend holding the jobs.

    Returns:
        The stored status of each job, keyed by its entry's id.
    """

    def _read(conn: sqlite3.Connection) -> dict[str, str]:
        rows = conn.execute('SELECT context_entry_id, status FROM generation_jobs').fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


def record_state_reads(monkeypatch: pytest.MonkeyPatch, repos: RepositoryContainer) -> list[list[str]]:
    """Record the ids of every generation state read the job repository runs, letting each read through.

    Args:
        monkeypatch: Wraps the repository's state read for the rest of the test.
        repos: The repositories the tools use.

    Returns:
        The ids of each state read, appended in call order.
    """
    reads: list[list[str]] = []
    read_states = repos.generation_jobs.states_for_ids

    async def _recording_read(context_ids: Sequence[str], *, scope: BackendAccess) -> dict[str, GenerationStatus]:
        reads.append(list(context_ids))
        return await read_states(context_ids, scope=scope)

    monkeypatch.setattr(repos.generation_jobs, 'states_for_ids', _recording_read)
    return reads


class InMemoryGenerationJobs:
    """Stand-in for the generation job repository of a faked repository container, holding its jobs in memory.

    It starts without a job, so every entry reads 'ready' until a write enqueues one. A job
    reads as the repository reports it: a 'failed' job as 'failed', any other as 'pending'.

    Attributes:
        jobs: The stored status of each entry's job, keyed by entry id.
        reads: The ids of every response-level state read, in call order.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, str] = {}
        self.reads: list[list[str]] = []

    def _state(self, context_id: str) -> GenerationStatus:
        """Return the state the entry's job reports, 'ready' without one."""
        status = self.jobs.get(context_id)
        if status is None:
            return 'ready'
        return 'failed' if status == 'failed' else 'pending'

    async def states_for_ids(self, context_ids: Sequence[str], *, scope: BackendAccess) -> dict[str, GenerationStatus]:
        """Return the state of every id.

        Args:
            context_ids: The ids the response carries.
            scope: Unused.

        Returns:
            Each id mapped to its state.
        """
        del scope
        self.reads.append(list(context_ids))
        return {context_id: self._state(context_id) for context_id in context_ids}

    async def state_for(self, context_id: str, *, txn: object) -> GenerationStatus:
        """Return the entry's state.

        Args:
            context_id: The entry's id.
            txn: Unused.

        Returns:
            The entry's state.
        """
        del txn
        return self._state(context_id)

    async def enqueue(self, context_id: str, *, txn: object) -> None:
        """Add a pending job for an entry that holds none, as the job table's primary key requires.

        Args:
            context_id: The entry's id.
            txn: Unused.

        Raises:
            ValueError: When the entry already holds a job.
        """
        del txn
        if context_id in self.jobs:
            raise ValueError(f'{context_id} already holds a job')
        self.jobs[context_id] = 'pending'

    async def delete_for_entry(self, context_id: str, *, txn: object) -> bool:
        """Delete the entry's job.

        Args:
            context_id: The entry's id.
            txn: Unused.

        Returns:
            Whether the entry held a job.
        """
        del txn
        return self.jobs.pop(context_id, None) is not None
