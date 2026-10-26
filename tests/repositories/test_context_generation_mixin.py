"""Tests for the generation reads and writes of the context repository on SQLite.

Every case of ``tests/repositories/_context_generation_cases.py`` runs here on a SQLite
database the server's own preparation built and seeded;
``tests/integration/postgresql/test_context_generation_mixin_postgresql.py`` runs the same
cases on PostgreSQL.
"""

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable

import pytest
import pytest_asyncio

from app.backends.base import StorageBackend
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import sqlite_scoped_db
from tests.repositories._context_generation_cases import CONTEXT_GENERATION_CASES


@pytest_asyncio.fixture
async def scoped_db(async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """A seeded SQLite database prepared as the server prepares one.

    Yields:
        The seeded database.
    """
    async with sqlite_scoped_db(async_db_initialized, 'compressed', monkeypatch) as db:
        yield db


@pytest.mark.asyncio
@pytest.mark.parametrize('case', CONTEXT_GENERATION_CASES, ids=lambda case: case.__name__.removeprefix('case_'))
async def test_context_generation_case(scoped_db: ScopedDb, case: Callable[[ScopedDb], Awaitable[None]]) -> None:
    """Each shared case holds on SQLite."""
    await case(scoped_db)
