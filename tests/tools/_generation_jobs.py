"""Generation job seeding, state read recording and a job repository double, shared by the read tool tests."""

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


class NoGenerationJobs:
    """Stand-in for the generation job repository of a faked repository container: no entry holds a job.

    Every entry it is asked about reads 'ready'; each state read is recorded.

    Attributes:
        reads: The ids of every state read, in call order.
    """

    def __init__(self) -> None:
        self.reads: list[list[str]] = []

    async def states_for_ids(self, context_ids: Sequence[str], *, scope: BackendAccess) -> dict[str, GenerationStatus]:
        """Return 'ready' for every id.

        Args:
            context_ids: The ids the response carries.
            scope: Unused.

        Returns:
            Each id mapped to 'ready'.
        """
        del scope
        self.reads.append(list(context_ids))
        ready: GenerationStatus = 'ready'
        return dict.fromkeys(context_ids, ready)
