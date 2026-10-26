"""The migration CLI's generation job drain against a real PostgreSQL server.

Every case of ``tests/cli/_generation_drain_cases.py`` runs here on an isolated database the
server's own preparation built: the drain's claims, the job counts, the table probe and the
entry reads of the two embedding modes, in the SQL PostgreSQL runs.
``tests/cli/test_generation_drain.py`` runs the same cases on SQLite.
"""

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable

import pytest
import pytest_asyncio

from tests.cli._generation_drain_cases import DRAIN_CASES
from tests.cli._generation_drain_cases import DrainGeneration
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_scoped_db

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


@pytest_asyncio.fixture
async def scoped_db(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """A seeded isolated database prepared as the server prepares one, with fp32 embeddings.

    Yields:
        The seeded database.
    """
    async with postgresql_scoped_db(pg_test_url, 'fp32', monkeypatch) as db:
        yield db


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch) -> DrainGeneration:
    """Activate the embedding and summary legs and replace the executor's generation with a recorder."""
    recorder = DrainGeneration()
    recorder.install(monkeypatch)
    return recorder


@pytest.mark.asyncio
@pytest.mark.parametrize('case', DRAIN_CASES, ids=lambda case: case.__name__.removeprefix('case_'))
async def test_drain_case(
    scoped_db: ScopedDb, generation: DrainGeneration, case: Callable[[ScopedDb, DrainGeneration], Awaitable[None]],
) -> None:
    """Each shared case holds on PostgreSQL."""
    await case(scoped_db, generation)
