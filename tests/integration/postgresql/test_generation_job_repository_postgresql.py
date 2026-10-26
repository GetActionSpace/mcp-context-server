"""The generation job repository against a real PostgreSQL server.

Every case of ``tests/repositories/_generation_job_cases.py`` runs here on an isolated
database the server's own preparation built. The PostgreSQL-only tests prove that a claim
skips a job another transaction holds locked instead of waiting for it, and that claims
made at the same time on several connections never share a job.
``tests/repositories/test_generation_job_repository.py`` runs the same cases on SQLite.
"""

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast

import asyncpg
import pytest
import pytest_asyncio

from app.backends.base import MAINTENANCE
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_scoped_db
from tests.repositories._generation_job_cases import GENERATION_JOB_CASES
from tests.repositories._generation_job_cases import add_entry
from tests.repositories._generation_job_cases import claim
from tests.repositories._generation_job_cases import enqueue
from tests.repositories._generation_job_cases import shift_job

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

# How long a claim may take before the test concludes it waited on a locked job.
_CLAIM_DEADLINE_S = 15.0


@pytest_asyncio.fixture
async def scoped_db(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """A seeded isolated database prepared as the server prepares one.

    Yields:
        The seeded database.
    """
    async with postgresql_scoped_db(pg_test_url, 'compressed', monkeypatch) as db:
        yield db


@pytest.mark.asyncio
@pytest.mark.parametrize('case', GENERATION_JOB_CASES, ids=lambda case: case.__name__.removeprefix('case_'))
async def test_generation_job_case(scoped_db: ScopedDb, case: Callable[[ScopedDb], Awaitable[None]]) -> None:
    """Each shared case holds on PostgreSQL."""
    await case(scoped_db)


@pytest.mark.asyncio
async def test_a_claim_skips_a_job_another_transaction_holds_locked(scoped_db: ScopedDb) -> None:
    """While another transaction holds the oldest job locked, a claim takes the next one at once."""
    locked = await add_entry(scoped_db, 'locked')
    free = await add_entry(scoped_db, 'free')
    await enqueue(scoped_db, locked, free)
    await shift_job(scoped_db, locked, column='created_at', seconds=-10)

    async with scoped_db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        connection = cast(asyncpg.Connection, txn.connection)
        await connection.execute('SELECT 1 FROM generation_jobs WHERE context_entry_id = $1 FOR UPDATE', locked)
        claimed = await asyncio.wait_for(claim(scoped_db), timeout=_CLAIM_DEADLINE_S)

    assert claimed is not None
    assert claimed.context_id == free
    released = await claim(scoped_db)
    assert released is not None
    assert released.context_id == locked


@pytest.mark.asyncio
async def test_concurrent_claims_never_share_a_job(scoped_db: ScopedDb) -> None:
    """Rounds of claims running together on the backend's connections claim every job exactly once."""
    ids = [await add_entry(scoped_db, f'concurrent-{position}') for position in range(24)]
    await enqueue(scoped_db, *ids)

    claimed: list[str] = []
    while True:
        claims = await asyncio.gather(*(claim(scoped_db) for _ in range(8)))
        round_ids = [job.context_id for job in claims if job is not None]
        if not round_ids:
            break
        claimed.extend(round_ids)

    assert len(claimed) == len(set(claimed))
    assert sorted(claimed) == sorted(ids)
