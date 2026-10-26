"""The database layouts of the access-scope cases, and the SQLite variable cap.

Each layout is built by the server's own startup preparation (``prepare_database``) under
pinned settings, so it matches what a server with that configuration provisions:

- ``fp32``: compression off; embeddings live in the fp32 ``vec_context_embeddings`` table.
- ``compressed``: compression on from the first start; the compressed table replaces the
  fp32 one, which is never created, and the provenance row is bootstrapped.
- ``fp32_rls`` and ``compressed_rls`` (PostgreSQL only): the same storage with
  ``ENABLE_POSTGRESQL_RLS`` on, prepared by the table owner, which grants the policy role
  the request privileges and installs the policies and the read view.

:func:`sqlite_scoped_db` prepares the SQLite backend of ``async_db_initialized``;
:func:`postgresql_scoped_db` prepares an isolated database on the pgvector test container.
Both serve the request and the seed channel from one backend.
:func:`postgresql_owner_seeded_db` prepares an existing database through a maintenance
backend logged in as its owner, which also serves the seed channel, while the request
channel logs in as another role. :func:`postgresql_rls_scoped_db` does the same for the
row-level security layouts, with the request backend under a given session profile. Each
seeds the database with :func:`~tests.repositories._access_scope_cases.seed_access_rows`.
:func:`configure_postgresql_layout` pins the environment of a layout on PostgreSQL for a
test that prepares the database itself.
:func:`assert_seeded_layout` proves a prepared database holds the seed and the layout's
vector storage, reading through the seed channel. :func:`limit_sqlite_variables` caps
SQLite bind variables for the ``sqlite_999_variables`` fixture of
``tests/repositories/conftest.py``.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Mapping
from contextlib import asynccontextmanager
from types import ModuleType
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import pytest

import app.migrations
import app.startup
from app.backends import StorageBackend
from app.backends import create_backend
from app.backends import create_maintenance_backend
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session_profiles import SessionProfile
from app.compression.factory import reset_cached_compression_provider
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import AppSettings
from app.settings import get_settings
from app.startup.database_setup import prepare_database
from tests.helpers import RlsCredentials
from tests.helpers import read_grants
from tests.helpers import rebind_package_settings
from tests.repositories._access_scope_cases import EMBEDDING_DIM
from tests.repositories._access_scope_cases import SEED_KEYWORD
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import SEED_ROWS
from tests.repositories._access_scope_cases import SEED_SOURCE
from tests.repositories._access_scope_cases import SEED_THREAD
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import seed_access_rows

SQLITE_TEST_VARIABLE_LIMIT = 999


async def _counts_by_label(db: ScopedDb, sql: str) -> dict[str, int]:
    """Run a ``SELECT <entry id>, COUNT(*) ... GROUP BY <entry id>`` and key the counts by label."""
    rows = await db.fetch_all(sql)
    return {db.labels_of([str(row[0])])[0]: int(str(row[1])) for row in rows}


async def _table_exists(db: ScopedDb, table: str) -> bool:
    """Return whether ``table`` exists in the database's working schema."""
    if db.seed_backend.backend_type == 'sqlite':
        sql = "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?"
    else:
        sql = 'SELECT COUNT(*) FROM pg_tables WHERE schemaname = current_schema() AND tablename = ?'
    rows = await db.fetch_all(sql, (table,))
    return int(str(rows[0][0])) == 1


async def assert_seeded_layout(db: ScopedDb) -> None:
    """Assert the database holds :data:`SEED_ROWS` with their children and the layout's vector storage.

    Every statement runs through the seed channel.

    Args:
        db: The seeded database.
    """
    rows = await db.fetch_all('SELECT id, owner_id, visibility, content_type, thread_id, source FROM context_entries')
    assert {db.labels_of([str(row[0])])[0]: row[1:] for row in rows} == {
        row.label: (row.owner, row.visibility, 'multimodal' if row.image else 'text', SEED_THREAD, SEED_SOURCE)
        for row in SEED_ROWS
    }

    grants = {row.label: await read_grants(db.seed_backend, db.ids[row.label]) for row in SEED_ROWS}
    assert grants == {
        row.label: sorted((grant.principal_type, grant.principal_id, grant.permission, row.owner) for grant in row.grants)
        for row in SEED_ROWS
    }

    tag_rows = await db.fetch_all('SELECT context_entry_id, tag FROM tags ORDER BY tag')
    tags: dict[str, list[str]] = {}
    for entry_id, tag in tag_rows:
        tags.setdefault(db.labels_of([str(entry_id)])[0], []).append(str(tag))
    assert tags == {row.label: [f'owner-{row.owner}', 'seed'] for row in SEED_ROWS}

    assert await _counts_by_label(
        db, 'SELECT context_entry_id, COUNT(*) FROM image_attachments GROUP BY context_entry_id',
    ) == {row.label: 1 for row in SEED_ROWS if row.image}
    assert await _counts_by_label(
        db, 'SELECT context_id, COUNT(*) FROM context_index_nodes GROUP BY context_id',
    ) == dict.fromkeys(SEED_LABELS, 1)
    assert await _counts_by_label(
        db, 'SELECT context_id, COUNT(*) FROM embedding_metadata GROUP BY context_id',
    ) == dict.fromkeys(SEED_LABELS, 1)

    if db.seed_backend.backend_type == 'sqlite':
        fts_sql = (
            'SELECT ce.id FROM context_entries_fts JOIN context_entries ce '
            'ON ce.rowid_int = context_entries_fts.rowid WHERE context_entries_fts MATCH ?'
        )
    else:
        fts_sql = "SELECT id FROM context_entries WHERE text_search_vector @@ plainto_tsquery('english', ?)"
    fts_rows = await db.fetch_all(fts_sql, (SEED_KEYWORD,))
    assert in_seed_order(db.labels_of(str(row[0]) for row in fts_rows)) == SEED_LABELS

    vector_table, absent_table = (
        ('vec_context_embeddings', 'vec_context_embeddings_compressed') if db.layout == 'fp32'
        else ('vec_context_embeddings_compressed', 'vec_context_embeddings')
    )
    assert await _table_exists(db, vector_table)
    assert not await _table_exists(db, absent_table)
    assert await db.fetch_all(f'SELECT COUNT(*) FROM {vector_table}') == [(len(SEED_ROWS),)]
    if db.layout == 'compressed':
        assert await db.fetch_all('SELECT dim FROM compression_metadata') == [(EMBEDDING_DIM,)]


def limit_sqlite_variables(monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    """Cap the bind variables of every connection a SQLite backend opens or hands out.

    Readers are opened per read, so capping connection creation reaches them; the writer is
    opened once and reused, so the cap is also applied whenever it is handed out, which
    reaches a writer opened before the cap was installed. SQLite checks the cap when a
    statement is prepared, so a statement the writer prepared and cached before the cap
    keeps running uncapped: install the cap before the statements under test first run.

    Args:
        monkeypatch: Restores both backend methods after the test.
        limit: The maximum number of bind variables per statement.
    """
    from app.backends.sqlite_backend import SQLiteBackend

    create_connection = SQLiteBackend._create_connection
    ensure_writer_connection = SQLiteBackend._ensure_writer_connection

    def _create_capped_connection(self: SQLiteBackend, readonly: bool = False) -> sqlite3.Connection:
        conn = create_connection(self, readonly)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
        return conn

    async def _ensure_capped_writer_connection(self: SQLiteBackend) -> sqlite3.Connection:
        conn = await ensure_writer_connection(self)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
        return conn

    monkeypatch.setattr(SQLiteBackend, '_create_connection', _create_capped_connection)
    monkeypatch.setattr(SQLiteBackend, '_ensure_writer_connection', _ensure_capped_writer_connection)


# Settings every layout pins, so the layout does not depend on the developer's environment.
_LAYOUT_ENVIRONMENT = {
    'EMBEDDING_DIM': str(EMBEDDING_DIM),
    'ENABLE_EMBEDDING_GENERATION': 'true',
    'ENABLE_FTS': 'true',
    'FTS_LANGUAGE': 'english',
    'ENABLE_INDEX_TREE_NODE_SUMMARIES': 'true',
    'COMPRESSION_PROVIDER': 'turboquant',
    'COMPRESSION_BITS': '4',
    'COMPRESSION_VARIANT': 'ip',
    'COMPRESSION_SEED': '0',
}


def _reset_compression_caches() -> None:
    """Drop the cached compression provider and provenance so the next use reads the current settings."""
    reset_cached_compression_provider()
    _reset_compression_cache()


def _configure_layout(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    overrides: Mapping[str, str],
    backend_package: ModuleType | None,
) -> AppSettings:
    """Set the layout's environment and rebind every module-level settings binding the preparation reads.

    Args:
        monkeypatch: Restores the environment and the bindings after the test.
        layout: The layout to configure.
        overrides: Further environment values, such as the PostgreSQL routing.
        backend_package: The backend package whose modules also read settings, if any.

    Returns:
        The settings of the layout.
    """
    environment = {
        **_LAYOUT_ENVIRONMENT,
        'ENABLE_EMBEDDING_COMPRESSION': 'true' if layout == 'compressed' else 'false',
        **overrides,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    rebind_package_settings(monkeypatch, app.migrations, settings)
    rebind_package_settings(monkeypatch, app.startup, settings)
    monkeypatch.setattr(app.startup, 'settings', settings)
    if backend_package is not None:
        rebind_package_settings(monkeypatch, backend_package, settings)
    _reset_compression_caches()
    return settings


def _scoped_db(request_backend: StorageBackend, seed_backend: StorageBackend, layout: Layout) -> ScopedDb:
    """Return an unseeded database with its request channel over one backend and its seed channel over the other.

    Args:
        request_backend: The backend the seams under test run on.
        seed_backend: The backend the seed is written and read through; may be the request backend.
        layout: The database layout.

    Returns:
        The database with its own repository container per channel.
    """
    return ScopedDb(
        backend=request_backend,
        repos=RepositoryContainer(request_backend),
        seed_backend=seed_backend,
        seed=RepositoryContainer(seed_backend),
        layout=layout,
    )


@asynccontextmanager
async def sqlite_scoped_db(
    backend: StorageBackend, layout: Layout, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """Prepare ``backend`` as the server does for ``layout`` and seed it; it serves both channels.

    Args:
        backend: An initialized SQLite backend holding the base schema.
        layout: The layout to build.
        monkeypatch: Restores the layout's environment and bindings after the test.

    Yields:
        The seeded database.
    """
    settings = _configure_layout(monkeypatch, layout, {}, None)
    try:
        await prepare_database(backend, settings, request_role=None)
        db = _scoped_db(backend, backend, layout)
        await seed_access_rows(db)
        yield db
    finally:
        _reset_compression_caches()


def _with_database(pg_url: str, database: str) -> str:
    """Return ``pg_url`` pointing at ``database``."""
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{database}', parts.query, parts.fragment))


async def _recreate_database(pg_url: str, database: str) -> None:
    """Drop ``database`` if it exists, dropping its connections, and create it empty."""
    import asyncpg

    admin = await asyncpg.connect(pg_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {database} WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE {database}')
    finally:
        await admin.close()


async def _drop_database(pg_url: str, database: str) -> None:
    """Drop ``database``, ignoring failures."""
    import asyncpg

    admin = await asyncpg.connect(pg_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {database} WITH (FORCE)')
    finally:
        await admin.close()


def configure_postgresql_layout(
    monkeypatch: pytest.MonkeyPatch, layout: Layout, request_url: str, *, overrides: Mapping[str, str],
) -> AppSettings:
    """Configure ``layout`` with the PostgreSQL routing of ``request_url`` in the ``public`` schema.

    Args:
        monkeypatch: Restores the environment and the bindings after the test.
        layout: The layout to configure.
        request_url: Connection string of the request connection settings.
        overrides: Further environment values, which win over the layout's own, such as a
            feature switched off.

    Returns:
        The settings of the layout.
    """
    import app.backends.postgresql_backend as postgresql_backend_package

    return _configure_layout(
        monkeypatch,
        layout,
        {
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_CONNECTION_STRING': request_url,
            'POSTGRESQL_SCHEMA': 'public',
            **overrides,
        },
        postgresql_backend_package,
    )


@asynccontextmanager
async def _seeded_postgresql_db(
    *,
    request_backend: StorageBackend,
    seed_backend: StorageBackend,
    layout: Layout,
    settings: AppSettings,
    request_role: str | None,
) -> AsyncGenerator[ScopedDb]:
    """Initialize both backends, prepare the database through the seed backend and seed it.

    Args:
        request_backend: The backend the seams under test run on.
        seed_backend: The backend that prepares the database and writes the seed; may be the request backend.
        layout: The layout the settings configure.
        settings: The settings of the layout.
        request_role: The role the request backend logs in as, which the preparation
            grants the request privileges to through the policy role; None grants nothing.

    Yields:
        The seeded database. Both backends are shut down afterwards.
    """
    backends = (seed_backend,) if request_backend is seed_backend else (seed_backend, request_backend)
    try:
        for backend in backends:
            await backend.initialize()
        await prepare_database(seed_backend, settings, request_role=request_role)
        db = _scoped_db(request_backend, seed_backend, layout)
        await seed_access_rows(db)
        yield db
    finally:
        for backend in backends:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)
        _reset_compression_caches()


@asynccontextmanager
async def postgresql_scoped_db(
    pg_url: str, layout: Layout, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """Create an isolated database, prepare it as the server does for ``layout`` and seed it.

    One backend, logged in with the credentials of ``pg_url``, serves both channels.

    Args:
        pg_url: Connection string of the test container's maintenance database.
        layout: The layout to build.
        monkeypatch: Restores the layout's environment and bindings after the test.

    Yields:
        The seeded database.
    """
    database = f'mcp_access_scope_{layout}'
    await _recreate_database(pg_url, database)
    target_url = _with_database(pg_url, database)
    try:
        settings = configure_postgresql_layout(monkeypatch, layout, target_url, overrides={})
        backend = create_backend(backend_type='postgresql', connection_string=target_url)
        async with _seeded_postgresql_db(
            request_backend=backend, seed_backend=backend, layout=layout, settings=settings, request_role=None,
        ) as db:
            yield db
    finally:
        await _drop_database(pg_url, database)


@asynccontextmanager
async def postgresql_owner_seeded_db(
    *, request_url: str, maintenance_url: str, layout: Layout, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """Prepare an existing database for ``layout`` as its owner and seed it through the owner.

    The maintenance backend logs in with the credentials of ``maintenance_url`` (the
    database owner), creates every table and serves the seed channel; the request backend
    logs in with those of ``request_url`` and serves the request channel. The database
    itself belongs to the caller.

    Args:
        request_url: Connection string of the request role.
        maintenance_url: Connection string of the role that owns the database.
        layout: The layout to build.
        monkeypatch: Restores the layout's environment and bindings after the test.

    Yields:
        The seeded database.
    """
    settings = configure_postgresql_layout(monkeypatch, layout, request_url, overrides={})
    async with _seeded_postgresql_db(
        request_backend=create_backend(backend_type='postgresql', connection_string=request_url),
        seed_backend=create_maintenance_backend(connection_string=maintenance_url),
        layout=layout,
        settings=settings,
        request_role=None,
    ) as db:
        yield db


@asynccontextmanager
async def postgresql_rls_scoped_db(
    *,
    request_url: str,
    maintenance_url: str,
    policy_role: str,
    request_profile: SessionProfile,
    layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """Prepare an existing database for ``layout`` with row-level security on, as its owner, and seed it.

    This builds the ``fp32_rls`` and ``compressed_rls`` layouts: ``ENABLE_POSTGRESQL_RLS``
    is on and the migration credentials are those of ``maintenance_url``. The maintenance
    backend logs in as the database owner and runs ``prepare_database`` with the role of
    ``request_url`` as the request role, which grants ``policy_role`` the request privileges
    and installs the policies and the read view; it then writes the seed as the owner, whom
    the policies do not filter, and serves the seed channel. The request backend logs in
    with the credentials of ``request_url`` under ``request_profile`` and serves the request
    channel, so every statement of the seams under test reaches rows through the policies
    and the read view. The database itself belongs to the caller.

    Args:
        request_url: Connection string of the request login role, a member of ``policy_role``.
        maintenance_url: Connection string of the role that owns the database.
        policy_role: The NOLOGIN role the policies name.
        request_profile: The session profile of the request backend.
        layout: The layout to build.
        monkeypatch: Restores the layout's environment and bindings after the test.

    Yields:
        The seeded database.
    """
    request_role = urlsplit(request_url).username
    maintenance = urlsplit(maintenance_url)
    assert request_role is not None
    assert maintenance.username is not None
    assert maintenance.password is not None
    settings = configure_postgresql_layout(
        monkeypatch,
        layout,
        request_url,
        overrides=RlsCredentials(
            migration_user=maintenance.username, migration_password=maintenance.password, policy_role=policy_role,
        ).environment(),
    )
    async with _seeded_postgresql_db(
        request_backend=PostgreSQLBackend(
            connection_string=request_url, profile=request_profile, credentials=NO_CREDENTIAL_OVERRIDE,
        ),
        seed_backend=create_maintenance_backend(connection_string=maintenance_url),
        layout=layout,
        settings=settings,
        request_role=request_role,
    ) as db:
        yield db
