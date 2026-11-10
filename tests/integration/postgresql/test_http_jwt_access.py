"""JWT principals reach only the entries their ownership, grants and role allow (PostgreSQL, over HTTP).

Runs the JWT scenarios against real HTTP servers verifying minted JWTs, once per
protocol era: the access scenario of :mod:`tests.integration._http_jwt_access`, where
alice shares her entries with the other principals until the operator CLI hands them to a
new owner, and the anonymous principal's scenarios of
:mod:`tests.integration._http_jwt_anonymous`, for which the test opens the server's
database as the superuser to count the rows the anonymous principal owns. Each
server serves an isolated database on the pgvector container of the ``pg_test_url``
fixture (``@requires_docker_postgres``, skipped cleanly without Docker). The access
scenario and the anonymous principal's reads and refusals run twice: on a database the
superuser serves with row-level security off, and on the ``pg_rls_db`` database served
as ``mcp_app`` with ``ENABLE_POSTGRESQL_RLS``, where the principal and groups of each
token become the identity its transactions bind and the policies decide what each
request reaches. The same scenarios run against SQLite in
``tests/integration/sqlite/test_http_jwt_access.py``.
"""

import contextlib
from collections.abc import AsyncIterator
from collections.abc import Iterator
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from tests.integration._harness.core import CLIENT_MODES
from tests.integration._harness.core import ClientMode
from tests.integration._http_jwt import build_access_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import running_http_server
from tests.integration._http_jwt import serve_jwks
from tests.integration._http_jwt import server_database
from tests.integration._http_jwt_access import verify_jwt_access_scoping
from tests.integration._http_jwt_access import verify_reassigned_owner_manages_grants
from tests.integration._http_jwt_anonymous import JWKS_KID
from tests.integration._http_jwt_anonymous import build_anonymous_env
from tests.integration._http_jwt_anonymous import build_jwks_env
from tests.integration._http_jwt_anonymous import mint_anonymous_token
from tests.integration._http_jwt_anonymous import verify_anonymous_arm
from tests.integration._http_jwt_anonymous import verify_jwks_composition
from tests.integration._http_jwt_anonymous import verify_read_only_setting_fails_open
from tests.integration._http_jwt_anonymous import without_read_only_principal
from tests.integration.postgresql._rls_catalog import assert_served_with_row_level_security
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.integration.postgresql.conftest import RLS_CREDENTIALS

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ACCESS_DATABASE = 'mcp_http_jwt_access'


def _with_database(pg_url: str, database: str) -> str:
    """Return ``pg_url`` pointing at ``database``.

    Returns:
        The connection string of ``database``.
    """
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{database}', parts.query, parts.fragment))


@pytest_asyncio.fixture
async def access_database_url(pg_test_url: str) -> AsyncIterator[str]:
    """Create an empty database for the access server and drop it afterwards.

    Yields:
        The connection string of the database.
    """
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {ACCESS_DATABASE} WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE {ACCESS_DATABASE}')
    finally:
        await admin.close()

    yield _with_database(pg_test_url, ACCESS_DATABASE)

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {ACCESS_DATABASE} WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture
def access_server(access_database_url: str, jwt_key_pair: RSAKeyPair) -> Iterator[str]:
    """Run the JWT access server on ``access_database_url``.

    Yields:
        The server origin URL.
    """
    env = build_access_env(
        backend='postgresql', database=access_database_url, port=free_port(), public_key=jwt_key_pair.public_key,
    )
    with running_http_server(env) as base_url:
        yield base_url


@pytest.fixture
def rls_access_server(pg_rls_db: RlsUrls, jwt_key_pair: RSAKeyPair) -> Iterator[str]:
    """Run the JWT access server on ``pg_rls_db`` as ``mcp_app`` with row-level security.

    Yields:
        The server origin URL.
    """
    env = {
        **build_access_env(
            backend='postgresql', database=pg_rls_db.app, port=free_port(), public_key=jwt_key_pair.public_key,
        ),
        **RLS_CREDENTIALS.environment(),
    }
    with running_http_server(env) as base_url:
        yield base_url


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_jwt_principals_reach_only_what_their_grants_allow(
    access_server: str,
    access_database_url: str,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
) -> None:
    """Shared user grants and group grants decide what bob and carol read, update and delete.

    Once the operator reassigns alice's entries, their new owner manages their grants.
    """
    await verify_jwt_access_scoping(access_server, jwt_key_pair, client_mode)
    await verify_reassigned_owner_manages_grants(access_server, jwt_key_pair, client_mode, access_database_url)


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_jwt_principals_reach_only_what_their_grants_allow_under_row_level_security(
    rls_access_server: str,
    pg_rls_db: RlsUrls,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
) -> None:
    """The same grants decide what bob and carol reach when the database enforces them too.

    Every request runs as ``mcp_app`` under the identity its token carries, so alice's stores
    write their ``author_groups`` read grants through the grants policies, which also decide
    her shares, listings and unshares through their owner arms; bob reaches her entries
    through the read view's group arm, and carol only through the user grants alice gives her.
    The operator CLI then reassigns alice's entries as the table owner, the foreign key
    carries the new owner into each grant's entry owner, and the owner arms answer the new
    owner's sharing calls.
    """
    await verify_jwt_access_scoping(rls_access_server, jwt_key_pair, client_mode)
    await verify_reassigned_owner_manages_grants(rls_access_server, jwt_key_pair, client_mode, pg_rls_db.maintenance)
    await assert_served_with_row_level_security(pg_rls_db)


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_anonymous_principal_reads_public_entries_and_writes_nothing(
    access_database_url: str,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anonymous principal reads only public entries, and its token writes once the setting is unset."""
    env = build_anonymous_env(
        backend='postgresql', database=access_database_url, port=free_port(), public_key=jwt_key_pair.public_key,
    )
    token = mint_anonymous_token(jwt_key_pair)
    with running_http_server(env) as base_url:
        async with server_database('postgresql', access_database_url, monkeypatch) as database:
            await verify_anonymous_arm(base_url, jwt_key_pair, client_mode, database, token)
    with running_http_server(without_read_only_principal(env)) as base_url:
        await verify_read_only_setting_fails_open(base_url, token, client_mode)


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_anonymous_principal_reads_public_entries_and_writes_nothing_under_row_level_security(
    pg_rls_db: RlsUrls,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anonymous principal reaches only public entries when the database enforces access too.

    Its requests run as ``mcp_app`` bound to the anonymous identity, which the read view's
    public arm alone admits; alice's writes pass the database's write check, which the
    anonymous identity's binding would fail.
    """
    env = {
        **build_anonymous_env(
            backend='postgresql', database=pg_rls_db.app, port=free_port(), public_key=jwt_key_pair.public_key,
        ),
        **RLS_CREDENTIALS.environment(),
    }
    token = mint_anonymous_token(jwt_key_pair)
    with running_http_server(env) as base_url:
        async with server_database('postgresql', pg_rls_db.admin, monkeypatch) as database:
            await verify_anonymous_arm(base_url, jwt_key_pair, client_mode, database, token)
    with running_http_server(without_read_only_principal(env)) as base_url:
        await verify_read_only_setting_fails_open(base_url, token, client_mode)
    await assert_served_with_row_level_security(pg_rls_db)


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_jwks_deployment_verifies_the_anonymous_token_with_its_static_key(
    access_database_url: str,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
) -> None:
    """Beside an identity provider's JWKS keys, the operator's static key verifies the anonymous subject only."""
    anonymous_key_pair = RSAKeyPair.generate()
    with serve_jwks(jwt_key_pair, JWKS_KID) as jwks_uri:
        env = build_jwks_env(
            backend='postgresql',
            database=access_database_url,
            port=free_port(),
            jwks_uri=jwks_uri,
            anonymous_public_key=anonymous_key_pair.public_key,
        )
        with running_http_server(env) as base_url:
            await verify_jwks_composition(base_url, jwt_key_pair, anonymous_key_pair, client_mode)
