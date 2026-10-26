"""Docker-compose-based PostgreSQL fixture for integration tests.

The fixture brings up a single pgvector container (image matches the
production deployment) on an ephemeral port, waits for ``pg_isready``,
yields the connection string, then tears the container down with
``docker compose down -v``. When the docker CLI or daemon is unavailable
the fixture skips cleanly so the test session keeps running on machines
without Docker.

Every compose call names a project derived from this checkout's repository
root (:func:`_project_name_for`), so test sessions in two checkouts of the
repository (separate clones or linked working trees) each run their own
container and never bring down each other's. Within one checkout, only the
session that brought the container up removes it at session end, so a
session that runs no PostgreSQL test can end beside one that still uses it.
The compose file's ``name:`` line serves manual ``docker compose`` use only;
``-p`` overrides it.

Credentials (``mcp_test:mcp_test``) are throwaway: the container exposes
no listening interface beyond loopback for the test session lifetime and
its data directory is mounted on tmpfs. The same holds for the login roles
``initdb/02-roles.sql`` creates, whose password is the role name;
:func:`pg_rls_db` hands out one connection string per role.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import os
import shutil
import socket
import subprocess
import time
import warnings
from collections.abc import AsyncIterator
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from tests.helpers import RlsCredentials
from tests.integration.postgresql._rls_urls import RlsUrls

COMPOSE_FILE = Path(__file__).parent / 'docker-compose.test.yml'


def _project_name_for(root: Path) -> str:
    """Return the compose project name for the checkout rooted at ``root``.

    The suffix is the first 8 hex characters of the SHA-256 of the resolved
    root path, so the name is stable for one checkout, differs between
    checkouts, and stays a valid compose project name (lowercase letters,
    digits and dashes).

    Args:
        root: The resolved repository root of the checkout.

    Returns:
        ``mcp-context-server-test-pg-`` followed by the 8-character suffix.
    """
    digest = hashlib.sha256(str(root).encode()).hexdigest()
    return f'mcp-context-server-test-pg-{digest[:8]}'


PROJECT_NAME = _project_name_for(Path(__file__).resolve().parents[3])

_NON_DEFAULT_SCHEMA = 'mcp_test'
NON_DEFAULT_SCHEMA = _NON_DEFAULT_SCHEMA  # public re-export for tests


def _compose_command(*args: str) -> list[str]:
    """Return the ``docker compose`` argv for ``args`` on this checkout's test project.

    Args:
        *args: The compose subcommand and its options.

    Returns:
        The argv naming the checkout's project and the test compose file.
    """
    return ['docker', 'compose', '-p', PROJECT_NAME, '-f', str(COMPOSE_FILE), *args]


@dataclass
class _SessionContainer:
    """Whether this pytest session brought the checkout's container up.

    Attributes:
        brought_up: Set by :func:`pg_test_url` right before ``docker compose up``.
    """

    brought_up: bool = False


SESSION_CONTAINER = _SessionContainer()


def _force_cleanup() -> None:
    """Best-effort idempotent removal of this checkout's postgres test container and volumes.

    Skips silently if docker CLI is not on PATH (CI environments without
    docker, fresh developer machines, etc.). Catches all subprocess errors
    so the helper can be safely chained from atexit, sessionfinish, and
    fixture pre-flight without raising. Stderr from a failing teardown is
    surfaced through warnings.warn so silent failures do not accumulate.
    """
    if shutil.which('docker') is None:
        return
    try:
        result = subprocess.run(
            _compose_command('down', '-v', '--remove-orphans'),
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        warnings.warn(
            f'docker compose down for {PROJECT_NAME} did not complete: {exc!r}',
            stacklevel=2,
        )
        return
    if result.returncode != 0:
        warnings.warn(
            f'docker compose down for {PROJECT_NAME} returned '
            f'exit code {result.returncode}; stderr: {result.stderr.strip()}',
            stacklevel=2,
        )


def _cleanup_brought_up_container() -> None:
    """Remove the checkout's container when this session brought it up.

    A session that never brought the container up leaves it alone, so a
    session that runs no PostgreSQL test (a ``-m "not integration"`` run)
    can end while another session of the same checkout is still using it.
    """
    if SESSION_CONTAINER.brought_up:
        _force_cleanup()


# Belt-and-suspenders cleanup at interpreter shutdown. The primary safety
# nets are the fixture-level try/finally and pytest_sessionfinish; this
# atexit hook adds redundancy for edge cases (interpreter abnormal exit
# paths that bypass pytest's own shutdown). atexit is not invoked on
# Windows for Ctrl-C delivered via the Win32 console control handler, so
# this layer cannot substitute for pytest_sessionfinish on Windows.
atexit.register(_cleanup_brought_up_container)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Remove the container this session brought up, after the full pytest session.

    Called by pytest after all tests complete and before pytest returns
    its exit code. This hook runs even when session-scoped fixture
    finalizers are skipped, such as KeyboardInterrupt during fixture
    setup or pytest.fail before the yield. See pytest issue 4517.

    Args:
        session: The pytest Session object (unused but required by hookspec).
        exitstatus: The exit status pytest will return (unused but required).
    """
    del session, exitstatus  # ARG001: required by pytest hookspec
    _cleanup_brought_up_container()


def _docker_available() -> bool:
    """Return True when the docker CLI is on PATH and the daemon answers."""
    if shutil.which('docker') is None:
        return False
    try:
        result = subprocess.run(
            ['docker', 'info'], capture_output=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _free_port() -> int:
    """Return an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _wait_for_health(timeout_s: float = 60.0) -> None:
    """Poll ``docker compose ps`` until the postgres-test container is healthy.

    Surfaces ``docker compose ps`` ``stderr`` in the final ``TimeoutError``
    so bring-up failures are visible to the developer/CI. Wraps each
    ``subprocess.run`` in a 5s timeout so the polling loop cannot stall on
    a hung Docker daemon (mirrors the established :func:`_force_cleanup`
    pattern at the symmetric teardown boundary).

    Args:
        timeout_s: Maximum wait time in seconds.

    Raises:
        TimeoutError: When the container does not become healthy in time.
            The error message includes the most recent ``docker compose ps``
            stderr and stdout to aid bring-up debugging.
    """
    deadline = time.time() + timeout_s
    last_result: subprocess.CompletedProcess[str] | None = None
    while time.time() < deadline:
        try:
            result = subprocess.run(
                _compose_command('ps', '--format', '{{.Health}}'),
                capture_output=True, text=True, check=False, timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            warnings.warn(
                f'docker compose ps for {PROJECT_NAME} timed out after 5s: {exc!r}',
                stacklevel=2,
            )
            time.sleep(1.0)
            continue
        last_result = result
        if 'healthy' in result.stdout:
            return
        time.sleep(1.0)

    stderr_hint = (
        last_result.stderr.strip()
        if last_result and last_result.stderr
        else '(no stderr captured)'
    )
    stdout_hint = (
        last_result.stdout.strip()
        if last_result and last_result.stdout
        else '(no stdout captured)'
    )
    raise TimeoutError(
        f'PostgreSQL container did not become healthy within {timeout_s}s. '
        f'Last `docker compose ps` stderr: {stderr_hint}; stdout: {stdout_hint}',
    )


@pytest.fixture(scope='session')
def pg_test_url() -> Iterator[str]:
    """Session-scoped fixture providing a pgvector connection string.

    Skips the test when Docker is unavailable.

    Performs a pre-flight cleanup of any stale container left over from a
    prior aborted run before bringing up a fresh one, and runs a hardened
    teardown (timeout + visible stderr + remove-orphans) at session end.

    Yields:
        Connection string pointing at a fresh pgvector container on an
        ephemeral loopback port.
    """
    if not _docker_available():
        pytest.skip(
            'Docker CLI is not available or the Docker daemon is not '
            'running; PostgreSQL integration tests require docker-compose',
        )

    # Pre-flight cleanup: remove any stale container/volume from a prior
    # aborted run so the new bring-up starts from a known empty state.
    _force_cleanup()

    port = _free_port()
    env = {**os.environ, 'MCP_PG_TEST_PORT': str(port)}

    SESSION_CONTAINER.brought_up = True
    up_result = subprocess.run(
        _compose_command('up', '-d', '--wait'),
        env=env, capture_output=True, text=True, check=False,
    )
    if up_result.returncode != 0:
        pytest.fail(
            'docker compose up failed:\n'
            f'stdout: {up_result.stdout}\nstderr: {up_result.stderr}',
        )

    try:
        _wait_for_health()
        yield f'postgresql://mcp_test:mcp_test@localhost:{port}/mcp_test'
    finally:
        _force_cleanup()


def login_url(url: str, role: str) -> str:
    """Return ``url`` logging in as ``role``, whose password is its own name.

    Args:
        url: A connection string of the test container.
        role: A login role of ``initdb/02-roles.sql``, or one a test creates the same way.

    Returns:
        The connection string.
    """
    parts = urlsplit(url)
    host = parts.netloc.rsplit('@', 1)[1]
    return urlunsplit((parts.scheme, f'{role}:{role}@{host}', parts.path, parts.query, parts.fragment))


def _database_url(pg_url: str, database: str, *, role: str | None = None) -> str:
    """Return ``pg_url`` pointing at ``database``, logging in as ``role`` when given.

    Args:
        pg_url: A connection string of the test container.
        database: The database to connect to.
        role: A login role of ``initdb/02-roles.sql``, whose password is its name;
            None keeps the credentials of ``pg_url``.

    Returns:
        The connection string.
    """
    parts = urlsplit(pg_url)
    url = urlunsplit((parts.scheme, parts.netloc, f'/{database}', parts.query, parts.fragment))
    return url if role is None else login_url(url, role)


@pytest_asyncio.fixture
async def pg_non_default_schema_db(pg_test_url: str) -> AsyncIterator[str]:
    """Provision an isolated PG database with POSTGRESQL_SCHEMA=mcp_test.

    The fixture creates a fresh database, installs pgvector inside it,
    creates the ``mcp_test`` schema, and yields the connection string.
    Tests that consume this fixture exercise the production schema
    routing automatically because :class:`PostgreSQLBackend` issues
    ``SET search_path`` on every pool connection from the pool's setup
    callback, each time the connection is acquired. Ad-hoc
    ``asyncpg.connect()`` calls inside the fixture's own setup
    phase (database creation, schema creation) still need explicit
    ``search_path`` configuration; use
    :func:`apply_non_default_search_path` for those callsites.

    The database is dropped at teardown so the fixture is self-cleaning
    and does not interact with the singleton ``compression_metadata``
    invariant from other tests.

    Yields:
        Connection string pointing at the per-test isolated database
        where the ``mcp_test`` schema has been pre-created.
    """
    db_name = 'mcp_non_default_schema_e2e'
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
        await admin.execute(f'CREATE DATABASE {db_name}')
    finally:
        await admin.close()

    target_url = _database_url(pg_test_url, db_name)

    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await setup.execute(
            f'CREATE SCHEMA IF NOT EXISTS {_NON_DEFAULT_SCHEMA}',
        )
    finally:
        await setup.close()

    yield target_url

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
    finally:
        await admin.close()


_RLS_DATABASE = 'mcp_rls'

# What a server on the ``pg_rls_db`` database starts with to run with row-level security:
# ``mcp_owner`` migrates and owns the tables, and the policies name ``mcp_context_request``,
# the role ``mcp_app`` inherits.
RLS_CREDENTIALS = RlsCredentials(migration_user='mcp_owner', migration_password='mcp_owner', policy_role='mcp_context_request')


@pytest_asyncio.fixture
async def pg_rls_db(pg_test_url: str) -> AsyncIterator[RlsUrls]:
    """Provision a fresh database owned by ``mcp_owner`` for the row-level security tests.

    ``mcp_test`` creates the database ``OWNER mcp_owner`` and, inside it, the ``vector``
    extension; the database holds nothing else. Its ``public`` schema belongs to the
    database owner, so ``mcp_owner`` creates the tables there and owns them. The database
    is dropped with every connection still open to it at teardown.

    Yields:
        One connection string per role of ``initdb/02-roles.sql`` and the superuser.
    """
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {_RLS_DATABASE} WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE {_RLS_DATABASE} OWNER mcp_owner')
    finally:
        await admin.close()

    urls = RlsUrls(
        admin=_database_url(pg_test_url, _RLS_DATABASE),
        maintenance=_database_url(pg_test_url, _RLS_DATABASE, role='mcp_owner'),
        app=_database_url(pg_test_url, _RLS_DATABASE, role='mcp_app'),
        outsider=_database_url(pg_test_url, _RLS_DATABASE, role='mcp_outsider'),
    )
    setup = await asyncpg.connect(urls.admin)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await setup.close()

    yield urls

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {_RLS_DATABASE} WITH (FORCE)')
    finally:
        await admin.close()


async def apply_non_default_search_path(conn: asyncpg.Connection) -> None:
    """Set ``search_path`` on ``conn`` to the non-default schema first.

    Helper for ad-hoc ``asyncpg.connect()`` callsites in tests that
    need to operate on tables in the non-default schema BEFORE the
    production pool (whose setup callback sets ``search_path`` on every
    acquired connection) is initialized. Pool-routed connections do
    not need this helper.
    """
    await conn.execute(
        f'SET search_path = {_NON_DEFAULT_SCHEMA}, public',
    )
