"""Identity hygiene of the request backend under the row-level security policies.

The request backend of a row-level security deployment binds each transaction's identity
transaction-locally on pooled ``mcp_app`` connections. These tests prove against a real
PostgreSQL server that nothing of one transaction's identity reaches the next transaction
on the same pooled connection, that concurrent transactions never see each other's
identity, that a write retried after a fault binds its identity again, and that the empty
principal reads exactly what no identity reads: the public rows.
"""

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable

import asyncpg
import pytest
import pytest_asyncio

import app.backends.postgresql_backend as pg_module
from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.resilience import RetryConfig
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.settings import get_settings
from tests.helpers import rebind_package_settings
from tests.integration.postgresql._rls_matrix import MATRIX_SCOPES
from tests.integration.postgresql._rls_matrix import MATRIX_VISIBILITY
from tests.integration.postgresql._rls_matrix import PUBLIC_ROWS
from tests.integration.postgresql._rls_matrix import labels
from tests.integration.postgresql._rls_matrix import rls_matrix_db
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import ScopedDb

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

READABLE = MATRIX_VISIBILITY[AccessMode.READ]
ALICE = AccessScope('alice', frozenset())

type AnyConnection = asyncpg.Connection[asyncpg.Record] | asyncpg.pool.PoolConnectionProxy[asyncpg.Record]
type Observation = tuple[str | None, int, tuple[str, ...]]
type BackendFactory = Callable[[int], Awaitable[PostgreSQLBackend]]


@pytest_asyncio.fixture
async def matrix_db(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """The policy matrix database in the fp32 layout.

    Yields:
        The seeded database with the policies installed.
    """
    async with rls_matrix_db(pg_rls_db, layout='fp32', monkeypatch=monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def request_backend(
    matrix_db: ScopedDb, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[BackendFactory]:
    """Yield a factory of initialized ``mcp_app`` request backends with a pool of the given size.

    The backends use the row-level security request profile and retry a failed write after
    a negligible delay. Every backend the factory built is shut down afterwards.

    Yields:
        The factory, taking the pool size.
    """
    _ = matrix_db
    backends: list[PostgreSQLBackend] = []

    async def _build(pool_size: int) -> PostgreSQLBackend:
        monkeypatch.setenv('POSTGRESQL_POOL_MIN', '1')
        monkeypatch.setenv('POSTGRESQL_POOL_MAX', str(pool_size))
        get_settings.cache_clear()
        rebind_package_settings(monkeypatch, pg_module, get_settings())
        backend = PostgreSQLBackend(
            connection_string=pg_rls_db.app,
            retry_config=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.01, jitter=False),
            provision_vector=False,
            profile=RlsRequestProfile(read_views=RLS_READ_VIEWS),
            credentials=NO_CREDENTIAL_OVERRIDE,
        )
        await backend.initialize()
        backends.append(backend)
        return backend

    yield _build
    for backend in backends:
        await backend.shutdown()


def _observer(db: ScopedDb, *, pause_s: float = 0.0) -> Callable[[AnyConnection], Awaitable[Observation]]:
    """Return a callable reading the bound principal, the backend pid and the rows the policies admit."""

    async def _observe(conn: AnyConnection) -> Observation:
        principal = await conn.fetchval("SELECT NULLIF(current_setting('app.principal_id', true), '')")
        if pause_s:
            await conn.execute('SELECT pg_sleep($1)', pause_s)
        rows = await conn.fetch('SELECT id FROM context_entries')
        return principal, int(await conn.fetchval('SELECT pg_backend_pid()')), labels(db, rows)

    return _observe


class TestPooledConnectionHygiene:
    """A committed identity is gone before the next transaction on the same pooled connection."""

    @pytest.mark.asyncio
    async def test_a_transaction_without_a_bind_after_an_alice_transaction_sees_public_rows_only(
        self, matrix_db: ScopedDb, request_backend: BackendFactory,
    ) -> None:
        backend = await request_backend(1)
        observe = _observer(matrix_db)

        alice = await backend.execute_read(observe, access_scope=ALICE)
        unbound: list[tuple[BackendAccess, Observation]] = []
        for access in (SYSTEM_SCOPE, MAINTENANCE):
            read = await backend.execute_read(observe, access_scope=access)
            written = await backend.execute_write(observe, access_scope=access)
            async with backend.begin_transaction(access_scope=access) as txn:
                transactional = await observe(txn.connection)
            unbound.extend([(access, read), (access, written), (access, transactional)])

        assert alice[0] == 'alice'
        assert alice[2] == READABLE['alice']
        for unbound_access, (principal, pid, rows) in unbound:
            assert pid == alice[1], unbound_access
            assert principal is None, unbound_access
            assert rows == PUBLIC_ROWS, unbound_access

    @pytest.mark.asyncio
    async def test_concurrent_identities_never_cross(self, matrix_db: ScopedDb, request_backend: BackendFactory) -> None:
        backend = await request_backend(4)
        observe = _observer(matrix_db, pause_s=0.02)
        named = [(name, scope) for name, scope in MATRIX_SCOPES.items() if scope is not None] * 4

        results = await asyncio.gather(*(backend.execute_read(observe, access_scope=scope) for _, scope in named))

        assert len({pid for _, pid, _ in results}) > 1
        for (name, scope), (principal, _, rows) in zip(named, results, strict=True):
            assert principal == (scope.principal_id or None), name
            assert rows == READABLE[name], name

    @pytest.mark.asyncio
    async def test_a_write_retried_after_a_fault_binds_its_identity_again(
        self, matrix_db: ScopedDb, request_backend: BackendFactory,
    ) -> None:
        backend = await request_backend(1)
        observe = _observer(matrix_db)
        attempts: list[Observation] = []

        async def _fail_once(conn: AnyConnection) -> Observation:
            attempts.append(await observe(conn))
            if len(attempts) == 1:
                raise asyncpg.exceptions.QueryCanceledError('canceling statement due to statement timeout')
            return attempts[-1]

        result = await backend.execute_write(_fail_once, access_scope=ALICE)

        assert len(attempts) == 2
        assert attempts[0][0] == result[0] == 'alice'
        assert result[2] == READABLE['alice']

    @pytest.mark.asyncio
    async def test_the_empty_principal_reads_what_no_identity_reads(
        self, matrix_db: ScopedDb, request_backend: BackendFactory,
    ) -> None:
        backend = await request_backend(1)
        observe = _observer(matrix_db)

        blank = await backend.execute_read(observe, access_scope=AccessScope('', frozenset()))
        unset = await backend.execute_read(observe, access_scope=SYSTEM_SCOPE)

        assert blank[0] is None
        assert unset[0] is None
        assert blank[2] == unset[2] == PUBLIC_ROWS
