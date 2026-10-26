"""The generation reads and writes of the context repository against a real PostgreSQL server.

Every case of ``tests/repositories/_context_generation_cases.py`` runs here on an isolated
database the server's own preparation built and seeded, among them the summary write that
advances ``updated_at`` through the table's update trigger and leaves ``version`` alone;
``tests/repositories/test_context_generation_mixin.py`` runs the same cases on SQLite.
"""

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable

import pytest
import pytest_asyncio

from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_scoped_db
from tests.repositories._context_generation_cases import CONTEXT_GENERATION_CASES

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


@pytest_asyncio.fixture
async def scoped_db(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """A seeded isolated database prepared as the server prepares one.

    Yields:
        The seeded database.
    """
    async with postgresql_scoped_db(pg_test_url, 'compressed', monkeypatch) as db:
        yield db


@pytest.mark.asyncio
@pytest.mark.parametrize('case', CONTEXT_GENERATION_CASES, ids=lambda case: case.__name__.removeprefix('case_'))
async def test_context_generation_case(scoped_db: ScopedDb, case: Callable[[ScopedDb], Awaitable[None]]) -> None:
    """Each shared case holds on PostgreSQL."""
    await case(scoped_db)
