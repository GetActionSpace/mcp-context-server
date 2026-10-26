"""Tests for the generation job repository on SQLite.

Every case of ``tests/repositories/_generation_job_cases.py`` runs here on a SQLite database
the server's own preparation built. The SQLite-only tests prove that two operating-system
processes claiming from one database file never claim the same job, that the state read
binds its ids in chunks under the lowest SQLite variable limit, and the backoff between
attempts. ``tests/integration/postgresql/test_generation_job_repository_postgresql.py``
runs the same cases on PostgreSQL.
"""

import asyncio
import json
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.client.stdio import get_default_environment

from app.backends.base import MAINTENANCE
from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories.generation_job_repository import retry_delay_s
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import sqlite_scoped_db
from tests.repositories._generation_job_cases import GENERATION_JOB_CASES
from tests.repositories._generation_job_cases import enqueue
from tests.repositories._generation_job_cases import jobs
from tests.repositories._generation_job_cases import read_job

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Claims every job it can from the database file named by its argument, after printing
# ``ready`` and reading one line from its standard input, and prints the claimed ids.
CLAIMING_PROCESS = '''
import asyncio
import json
import sys

from app.backends import create_backend
from app.backends.base import MAINTENANCE
from app.repositories.generation_job_repository import GenerationJobRepository


async def main(db_path: str) -> None:
    backend = create_backend(backend_type='sqlite', db_path=db_path)
    await backend.initialize()
    repository = GenerationJobRepository(backend)
    claimed = []
    try:
        print('ready', flush=True)
        sys.stdin.readline()
        while (job := await repository.claim_next(max_attempts=3, lease_s=120, scope=MAINTENANCE)) is not None:
            claimed.append(job.context_id)
    finally:
        await backend.shutdown()
    print(json.dumps(claimed), flush=True)


asyncio.run(main(sys.argv[1]))
'''


@pytest_asyncio.fixture
async def scoped_db(async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """A seeded SQLite database prepared as the server prepares one.

    Yields:
        The seeded database.
    """
    async with sqlite_scoped_db(async_db_initialized, 'compressed', monkeypatch) as db:
        yield db


async def _insert_entries(db: ScopedDb, count: int) -> list[str]:
    """Store ``count`` bare entries as the table owner, in one write, and return their ids in order."""
    ids = [generate_id() for _ in range(count)]

    def _insert(conn: sqlite3.Connection) -> None:
        conn.executemany(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
            "VALUES (?, 'generation-jobs', 'agent', 'text', ?, 'alice')",
            [(context_id, f'entry {position}') for position, context_id in enumerate(ids)],
        )

    await db.seed_backend.execute_write(_insert, access_scope=MAINTENANCE)
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize('case', GENERATION_JOB_CASES, ids=lambda case: case.__name__.removeprefix('case_'))
async def test_generation_job_case(scoped_db: ScopedDb, case: Callable[[ScopedDb], Awaitable[None]]) -> None:
    """Each shared case holds on SQLite."""
    await case(scoped_db)


@pytest.mark.asyncio
@pytest.mark.usefixtures('sqlite_999_variables')
async def test_states_bind_ids_in_chunks_under_the_variable_limit(scoped_db: ScopedDb) -> None:
    """More ids than one statement may bind are read in chunks, a job in every chunk found."""
    ids = await _insert_entries(scoped_db, 950)
    await enqueue(scoped_db, ids[0], ids[-1])

    states = await jobs(scoped_db).states_for_ids(ids, scope=MAINTENANCE)

    assert list(states) == ids
    assert {context_id for context_id, state in states.items() if state == 'pending'} == {ids[0], ids[-1]}
    assert {state for context_id, state in states.items() if context_id not in {ids[0], ids[-1]}} == {'ready'}


def _claim_from_two_processes(db_path: Path) -> list[list[str]]:
    """Start two claiming processes on ``db_path``, let them claim at the same moment and return what each claimed."""
    processes = [
        subprocess.Popen(
            [sys.executable, '-c', CLAIMING_PROCESS, str(db_path)],
            cwd=REPOSITORY_ROOT,
            env=get_default_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            assert process.stdout is not None
            first_line = process.stdout.readline().strip()
            assert first_line == 'ready', process.stderr.read() if process.stderr else first_line
        for process in processes:
            assert process.stdin is not None
            process.stdin.write('go\n')
            process.stdin.flush()
        claimed: list[list[str]] = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=180)
            assert process.returncode == 0, stderr
            claimed.append(json.loads(stdout.strip().splitlines()[-1]))
        return claimed
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.asyncio
async def test_two_processes_never_claim_the_same_job(scoped_db: ScopedDb, temp_db_path: Path) -> None:
    """Two processes claiming from one database file together claim every job exactly once."""
    ids = await _insert_entries(scoped_db, 200)
    await enqueue(scoped_db, *ids)

    first, second = await asyncio.to_thread(_claim_from_two_processes, temp_db_path)

    assert first
    assert second
    assert not set(first) & set(second)
    assert sorted(first + second) == sorted(ids)
    for context_id in (first[0], second[0]):
        job = await read_job(scoped_db, context_id)
        assert job is not None
        assert (job.status, job.attempts) == ('running', 1)


def test_retry_delay_grows_fourfold_to_its_ceiling() -> None:
    """The wait before a retry is 30 seconds after the first attempt, then 120, then 480 at most."""
    assert [retry_delay_s(attempt) for attempt in range(1, 6)] == [30, 120, 480, 480, 480]
