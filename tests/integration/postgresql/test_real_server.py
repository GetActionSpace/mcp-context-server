"""PostgreSQL entry point for the real-server integration suite.

Runs the shared backend-parametrized harness
(:class:`tests.integration._harness.MCPServerIntegrationTest`) against a REAL
MCP server subprocess backed by PostgreSQL, giving PostgreSQL the same
end-to-end tool-surface coverage as SQLite. The identical assertion methods
execute against the asyncpg backend, exercising PG-specific SQL paths that have
no SQLite equivalent: tsvector/tsquery FTS, the pgvector ``<->`` / HNSW /
``DISTINCT ON`` vector search, jsonb metadata operators (``->>`` / ``@>`` /
``jsonb_typeof``), native-UUID dedup interleaving ordering, ``Decimal`` ->
float statistics coercion, the ``jsonb_merge_patch`` deep-merge function, and
the asyncpg UUID type codec.

The run uses an ISOLATED database (``pg_parity_db``) rather than the shared
``mcp_test`` database, so it is independent of the schema mutations and the
seed-locked ``compression_metadata`` singleton that the dedicated compression
round-trip tests apply to ``mcp_test``. The harness runs the fp32 vector layout
(``ENABLE_EMBEDDING_COMPRESSION=false``); the compressed PG path is covered by
``test_compression_round_trip.py`` and ``test_search_compressed_postgresql.py``.

The harness runs twice per protocol era. The superuser run serves its database as
``mcp_test`` with row-level security off, so the application's own conditions alone
decide what each request reaches. The row-level security run serves the ``pg_rls_db``
database with ``ENABLE_POSTGRESQL_RLS``: ``mcp_owner`` migrates and owns the tables,
every request logs in as ``mcp_app``, and the identity each request binds decides what
the policies and the read view let it reach, beside the application's conditions.

Embedding/semantic/hybrid sub-tests self-gate on Ollama model availability
inside the harness, so the suite degrades gracefully (skip-as-pass) when no
model is present; FTS runs unconditionally (native tsvector, no model needed).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from tests.integration._harness import REAL_SERVER_TIMEOUT_S
from tests.integration._harness import MCPServerIntegrationTest
from tests.integration._harness.core import CLIENT_MODES
from tests.integration._harness.core import ClientMode
from tests.integration.postgresql._rls_catalog import assert_served_with_row_level_security
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.integration.postgresql.conftest import RLS_CREDENTIALS

pytestmark = [
    pytest.mark.requires_docker_postgres,
    pytest.mark.integration,
    pytest.mark.timeout(REAL_SERVER_TIMEOUT_S),
]


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database-name path component replaced."""
    parts = urlsplit(pg_url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        f'/{new_db}',
        parts.query,
        parts.fragment,
    ))


@pytest_asyncio.fixture
async def pg_parity_db(pg_test_url: str) -> AsyncIterator[str]:
    """Provision an isolated PostgreSQL database for the full parity run.

    The harness stores roughly a hundred entries across many tool calls. An
    isolated database keeps the run independent of the shared ``mcp_test``
    database that other PG tests mutate (notably the compression round-trip
    test, which seals the seed-locked ``compression_metadata`` singleton and
    swaps ``mcp_test`` to the compressed vector layout). The database is created
    fresh, the pgvector extension installed, and the database dropped (with
    FORCE, terminating any lingering pool connections) at teardown.

    Args:
        pg_test_url: docker-compose pgvector DSN (skips when Docker is absent).

    Yields:
        Connection string pointing at the isolated parity database.
    """
    db_name = 'mcp_real_server_parity'
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {db_name} WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE {db_name}')
    finally:
        await admin.close()

    target_url = _replace_db_name(pg_test_url, db_name)
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await setup.close()

    yield target_url

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {db_name} WITH (FORCE)')
    finally:
        await admin.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_real_server_postgresql(pg_parity_db: str, client_mode: ClientMode) -> None:
    """Run the full real-server harness against PostgreSQL (parity with SQLite).

    Constructs the shared harness in PostgreSQL mode and runs every assertion
    method through a real MCP server subprocess connected to the isolated
    pgvector database, once per protocol era. Mirrors
    ``tests/integration/sqlite/test_real_server.py``::``test_real_server`` so the
    two backends share one source of truth.

    Args:
        pg_parity_db: Isolated pgvector DSN provided by :func:`pg_parity_db`.
        client_mode: Protocol era the harness client negotiates.
    """
    test = MCPServerIntegrationTest(
        backend='postgresql', pg_url=pg_parity_db, pg_admin_url=pg_parity_db, rls=None, client_mode=client_mode,
    )
    success = await test.run_all_tests()
    assert success, f'PostgreSQL integration tests failed ({client_mode} client mode)'


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_real_server_postgresql_row_level_security(pg_rls_db: RlsUrls, client_mode: ClientMode) -> None:
    """Run the full real-server harness against PostgreSQL with row-level security.

    Every server the harness starts serves the ``pg_rls_db`` database as ``mcp_app`` with
    ``ENABLE_POSTGRESQL_RLS`` and ``mcp_owner`` as the migration role, once per protocol
    era, and the rows no tool shows are counted as the superuser. The same assertions as
    the superuser run hold, and the catalog afterwards shows that the tables the requests
    reached carried the policies.

    Args:
        pg_rls_db: The database ``mcp_owner`` owns, one connection string per role.
        client_mode: Protocol era the harness client negotiates.
    """
    test = MCPServerIntegrationTest(
        backend='postgresql', pg_url=pg_rls_db.app, pg_admin_url=pg_rls_db.admin, rls=RLS_CREDENTIALS,
        client_mode=client_mode,
    )
    success = await test.run_all_tests()
    assert success, f'PostgreSQL row-level security integration tests failed ({client_mode} client mode)'
    await assert_served_with_row_level_security(pg_rls_db)
