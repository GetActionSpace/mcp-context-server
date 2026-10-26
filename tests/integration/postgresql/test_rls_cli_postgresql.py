"""The migration CLI runs as the table owner, and refuses a role that row-level security filters.

Every test prepares the ``pg_rls_db`` database as a start with ``ENABLE_POSTGRESQL_RLS``
and split roles does, as ``mcp_owner``, and seeds it through the owner with entries only
their owner may read. Each in-place mode completes with the owner's connection string and,
with the request role ``mcp_app``'s, exits 1 naming the role and the filtered tables before
it reads or changes anything: a filtered role would see the private rows as absent, so
``--compress`` and ``--decompress`` would drop payloads it never read. A refusal also holds
with the maintenance session statements suppressed, as behind a pooler that drops them.
The integer-to-UUIDv7 migration provisions full-text search on a pre-existing target as
the owner and refuses ``mcp_app``. ``--disable-rls`` reports in a dry run, removes the
objects in a real run, and refuses while a member of the policy role is connected unless
``--force``; its own pooled connections never count, even as a superuser, and neither does
any superuser session. The startup compression guard still refuses fp32 rows only their
owner reads.

Every command-line subprocess starts from :func:`tests.helpers.postgresql_subprocess_env`.
"""

import asyncio
import contextlib
import subprocess
import sys
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from app.backends.postgresql_backend.session_profiles import MaintenanceProfile
from app.cli.migrate import main as cli_main
from app.errors import ConfigurationError
from app.migrations.rls_ddl import MANAGED_TABLES
from app.repositories import RepositoryContainer
from app.settings import AppSettings
from app.startup.database_setup import prepare_database
from tests.helpers import postgresql_subprocess_env
from tests.integration.postgresql._rls_catalog import configure_rls_layout
from tests.integration.postgresql._rls_catalog import maintenance_backend
from tests.integration.postgresql._rls_catalog import object_oids
from tests.integration.postgresql._rls_catalog import row_security
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
CLI_TIMEOUT_S: Final = 180
SCHEMA_NAME: Final = 'public'
REQUEST_ROLE: Final = 'mcp_app'
ENTRIES: Final = 3
V2_SOURCE_DATABASE: Final = 'mcp_rls_v2_source'
_V2_SOURCE_SQL: Final = '''
CREATE TABLE context_entries (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL,
    content_type TEXT NOT NULL,
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    text_search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text_content, ''))) STORED
);
CREATE TABLE tags (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE,
    tag TEXT NOT NULL
);
CREATE TABLE image_attachments (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE,
    image_data BYTEA NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSONB,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO context_entries (thread_id, source, content_type, text_content)
VALUES ('v2', 'user', 'text', 'first migrated entry'), ('v2', 'agent', 'text', 'second migrated entry');
'''


@dataclass(frozen=True, slots=True)
class Run:
    """How one command-line run ended and what it printed."""

    returncode: int
    stderr: str


def _cli(source_url: str, *args: str, **overrides: str) -> Run:
    """Run ``mcp-context-server-migrate --source-url source_url *args`` in a subprocess.

    Args:
        source_url: The ``--source-url`` value.
        args: Further command-line arguments.
        overrides: Further environment values.

    Returns:
        The exit code and what the command printed to stderr.
    """
    result = subprocess.run(
        [sys.executable, '-m', 'app.cli.migrate', '--source-url', source_url, *args],
        cwd=REPO_ROOT,
        env=postgresql_subprocess_env(source_url, rls=None, **overrides),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_S,
        check=False,
    )
    return Run(returncode=result.returncode, stderr=result.stderr)


async def _run(source_url: str, *args: str, **overrides: str) -> Run:
    """Run :func:`_cli` off the event loop."""
    return await asyncio.to_thread(_cli, source_url, *args, **overrides)


async def _prepare(
    urls: RlsUrls,
    monkeypatch: pytest.MonkeyPatch,
    *,
    layout: Layout,
    embeddings: bool,
    entries: int = ENTRIES,
    overrides: dict[str, str] | None = None,
) -> AppSettings:
    """Prepare ``layout`` with row-level security as the owner and seed entries only their owner reads.

    Args:
        urls: The ``pg_rls_db`` connection strings.
        monkeypatch: Restores the layout's environment and bindings after the test.
        layout: The layout to build.
        embeddings: Whether every entry carries one embedding.
        entries: How many entries to seed.
        overrides: Further environment values of the layout.

    Returns:
        The settings of the start.
    """
    settings = configure_rls_layout(urls, monkeypatch, layout=layout, overrides=overrides or {}, rls=True)
    async with maintenance_backend(urls.maintenance) as backend:
        await prepare_database(backend, settings, request_role=REQUEST_ROLE)
        repos = RepositoryContainer(backend)
        db = ScopedDb(backend=backend, repos=repos, seed_backend=backend, seed=repos, layout=layout)
        for index in range(entries):
            await db.add_entry(
                f'private-{index}', owner='alice', visibility='private', angle=0.1 * index if embeddings else None,
            )
    return settings


@contextlib.asynccontextmanager
async def _connect(url: str) -> AsyncGenerator[asyncpg.Connection]:
    """Yield a raw connection to ``url``, closed afterwards."""
    conn = await asyncpg.connect(url)
    try:
        yield conn
    finally:
        await conn.close()


async def _count(url: str, table: str) -> int | None:
    """Return the rows of ``table`` read as the superuser of ``url``, or None when the table is absent."""
    async with _connect(url) as conn:
        if not await conn.fetchval('SELECT to_regclass($1) IS NOT NULL', f'{SCHEMA_NAME}.{table}'):
            return None
        return int(await conn.fetchval(f'SELECT count(*) FROM {SCHEMA_NAME}.{table}'))


async def _owners(url: str) -> list[str]:
    """Return the owner of every context entry, read as the superuser of ``url``."""
    async with _connect(url) as conn:
        return [str(row[0]) for row in await conn.fetch('SELECT owner_id FROM context_entries ORDER BY id')]


async def _refusal(urls: RlsUrls) -> str:
    """Return the refusal the CLI prints for ``mcp_app``: every managed table with row-level security on."""
    async with _connect(urls.admin) as conn:
        enabled = sorted(
            table
            for table, (on, _forced) in (await row_security(conn, SCHEMA_NAME)).items()
            if on and table in MANAGED_TABLES
        )
    return (
        f"This command must connect with the table owner's credentials: row-level security filters {REQUEST_ROLE} "
        f"on {', '.join(enabled)}. Rerun with the owner's connection string (docs/migration-v2-to-v3.md)."
    )


class TestReassignOwner:
    """``--reassign-owner`` rewrites every private row as the owner and refuses the request role."""

    @pytest.mark.asyncio
    async def test_the_owner_rewrites_every_row(self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=False)

        run = await _run(pg_rls_db.maintenance, '--reassign-owner', 'alice', 'bob')

        assert run.returncode == 0, run.stderr
        assert f"Reassigned {ENTRIES} context entries from 'alice' to 'bob'" in run.stderr
        assert await _owners(pg_rls_db.admin) == ['bob'] * ENTRIES

    @pytest.mark.asyncio
    async def test_the_request_role_is_refused_before_any_statement(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=False)

        run = await _run(pg_rls_db.app, '--reassign-owner', 'alice', 'bob')

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert await _owners(pg_rls_db.admin) == ['alice'] * ENTRIES

    @pytest.mark.asyncio
    async def test_the_refusal_holds_with_the_session_statements_suppressed(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=False)
        refusal = await _refusal(pg_rls_db)
        monkeypatch.setattr(MaintenanceProfile, 'session_statements', lambda _self: [])

        rc = await asyncio.to_thread(cli_main, ['--source-url', pg_rls_db.app, '--reassign-owner', 'alice', 'bob'])

        assert rc == 1
        assert refusal in capsys.readouterr().err
        assert await _owners(pg_rls_db.admin) == ['alice'] * ENTRIES


class TestCompression:
    """``--compress`` and ``--decompress`` keep every private payload as the owner and drop nothing for the request role."""

    @pytest.mark.asyncio
    async def test_compress_completes_as_the_owner(self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=True)

        run = await _run(pg_rls_db.maintenance, '--compress', ENABLE_EMBEDDING_COMPRESSION='true')

        assert run.returncode == 0, run.stderr
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings') is None
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings_compressed') == ENTRIES
        assert await _count(pg_rls_db.admin, 'compression_metadata') == 1

    @pytest.mark.asyncio
    async def test_compress_refuses_the_request_role_and_drops_nothing(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=True)

        run = await _run(pg_rls_db.app, '--compress', ENABLE_EMBEDDING_COMPRESSION='true')

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings') == ENTRIES
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings_compressed') is None

    @pytest.mark.asyncio
    async def test_decompress_completes_as_the_owner(self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='compressed', embeddings=True)

        run = await _run(pg_rls_db.maintenance, '--decompress', ENABLE_EMBEDDING_COMPRESSION='false')

        assert run.returncode == 0, run.stderr
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings') == ENTRIES
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings_compressed') is None
        assert await _count(pg_rls_db.admin, 'compression_metadata') == 0

    @pytest.mark.asyncio
    async def test_decompress_refuses_the_request_role_which_would_read_no_payload(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='compressed', embeddings=True)

        run = await _run(pg_rls_db.app, '--decompress', ENABLE_EMBEDDING_COMPRESSION='false')

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings_compressed') == ENTRIES
        assert await _count(pg_rls_db.admin, 'compression_metadata') == 1

    @pytest.mark.asyncio
    async def test_zero_data_decompress_completes_as_the_owner(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='compressed', embeddings=False)

        run = await _run(pg_rls_db.maintenance, '--decompress', ENABLE_EMBEDDING_COMPRESSION='false')

        assert run.returncode == 0, run.stderr
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings_compressed') is None
        assert await _count(pg_rls_db.admin, 'compression_metadata') == 0

    @pytest.mark.asyncio
    async def test_zero_data_decompress_refuses_the_request_role_and_drops_nothing(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='compressed', embeddings=False)

        run = await _run(pg_rls_db.app, '--decompress', ENABLE_EMBEDDING_COMPRESSION='false')

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert await _count(pg_rls_db.admin, 'vec_context_embeddings_compressed') == 0
        assert await _count(pg_rls_db.admin, 'compression_metadata') == 1

    @pytest.mark.asyncio
    async def test_the_startup_guard_still_refuses_fp32_rows_only_their_owner_reads(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=True)
        compressed = configure_rls_layout(pg_rls_db, monkeypatch, layout='compressed', overrides={}, rls=True)

        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            with pytest.raises(ConfigurationError, match='still holds uncompressed fp32 embeddings'):
                await prepare_database(backend, compressed, request_role=REQUEST_ROLE)

        assert await _count(pg_rls_db.admin, 'vec_context_embeddings') == ENTRIES


class TestEmbeddingModes:
    """``--re-embed`` and ``--embed-missing`` refuse the request role before any provider call."""

    @pytest.mark.asyncio
    async def test_re_embed_refuses_the_request_role(self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=True)

        run = await _run(pg_rls_db.app, '--re-embed')

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert await _count(pg_rls_db.admin, 'embedding_metadata') == ENTRIES

    @pytest.mark.asyncio
    async def test_embed_missing_refuses_the_request_role(
        self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=False)

        run = await _run(pg_rls_db.app, '--embed-missing')

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert await _count(pg_rls_db.admin, 'embedding_metadata') == 0


def _with_database(url: str, database: str) -> str:
    """Return ``url`` pointing at ``database``."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{database}', parts.query, parts.fragment))


@pytest_asyncio.fixture
async def v2_source(pg_test_url: str) -> AsyncGenerator[str]:
    """An integer-keyed PostgreSQL source with full-text search and two entries.

    Yields:
        Its connection string, as the superuser.
    """
    async with _connect(pg_test_url) as admin:
        await admin.execute(f'DROP DATABASE IF EXISTS {V2_SOURCE_DATABASE} WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE {V2_SOURCE_DATABASE}')
    source_url = _with_database(pg_test_url, V2_SOURCE_DATABASE)
    async with _connect(source_url) as conn:
        await conn.execute(_V2_SOURCE_SQL)
    try:
        yield source_url
    finally:
        async with _connect(pg_test_url) as admin:
            await admin.execute(f'DROP DATABASE IF EXISTS {V2_SOURCE_DATABASE} WITH (FORCE)')


async def _has_fts(url: str) -> bool:
    async with _connect(url) as conn:
        return bool(
            await conn.fetchval(
                'SELECT EXISTS (SELECT 1 FROM information_schema.columns '
                "WHERE table_schema = $1 AND table_name = 'context_entries' AND column_name = 'text_search_vector')",
                SCHEMA_NAME,
            ),
        )


class TestUuidMigrationTarget:
    """The full-text search backstop provisions a pre-existing row-level security target as the owner only."""

    @pytest.mark.asyncio
    async def test_the_owner_provisions_full_text_search_and_copies(
        self, pg_rls_db: RlsUrls, v2_source: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(
            pg_rls_db, monkeypatch, layout='fp32', embeddings=False, entries=0, overrides={'ENABLE_FTS': 'false'},
        )

        run = await _run(v2_source, '--target-url', pg_rls_db.maintenance)

        assert run.returncode == 0, run.stderr
        assert await _has_fts(pg_rls_db.admin)
        assert await _count(pg_rls_db.admin, 'context_entries') == 2

    @pytest.mark.asyncio
    async def test_the_request_role_is_refused_before_the_backstop(
        self, pg_rls_db: RlsUrls, v2_source: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _prepare(
            pg_rls_db, monkeypatch, layout='fp32', embeddings=False, entries=0, overrides={'ENABLE_FTS': 'false'},
        )

        run = await _run(v2_source, '--target-url', pg_rls_db.app)

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert not await _has_fts(pg_rls_db.admin)
        assert await _count(pg_rls_db.admin, 'context_entries') == 0


async def _installed_objects(urls: RlsUrls) -> dict[str, int]:
    """Return the OID of every policy, helper, trigger and view of ours, read as the table owner."""
    async with _connect(urls.maintenance) as conn:
        return await object_oids(conn, SCHEMA_NAME)


class TestDisableRls:
    """``--disable-rls`` removes what the server installed, as the owner, while no policy-role member is connected."""

    @pytest_asyncio.fixture
    async def installed(self, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """The fp32 layout prepared with row-level security; returns what it installed."""
        await _prepare(pg_rls_db, monkeypatch, layout='fp32', embeddings=False)
        objects = await _installed_objects(pg_rls_db)
        assert any(name.startswith('policy ') for name in objects)
        return objects

    @pytest.mark.asyncio
    async def test_a_dry_run_reports_and_changes_nothing(self, pg_rls_db: RlsUrls, installed: dict[str, int]) -> None:
        run = await _run(pg_rls_db.maintenance, '--disable-rls', '--dry-run')

        assert run.returncode == 0, run.stderr
        assert '[DRY-RUN] Would issue' in run.stderr
        assert await _installed_objects(pg_rls_db) == installed

    @pytest.mark.asyncio
    async def test_a_real_run_removes_every_object_and_disables_every_table(
        self, pg_rls_db: RlsUrls, installed: dict[str, int],
    ) -> None:
        _ = installed

        run = await _run(pg_rls_db.maintenance, '--disable-rls')

        assert run.returncode == 0, run.stderr
        assert await _installed_objects(pg_rls_db) == {}
        async with _connect(pg_rls_db.admin) as conn:
            enabled = {table for table, (on, _forced) in (await row_security(conn, SCHEMA_NAME)).items() if on}
        assert enabled == set()

    @pytest.mark.asyncio
    async def test_the_command_s_own_superuser_connections_do_not_refuse_it(
        self, pg_rls_db: RlsUrls, installed: dict[str, int],
    ) -> None:
        """A superuser counts as a member of every role; the pooled connections of the command itself never refuse it."""
        _ = installed

        run = await _run(pg_rls_db.admin, '--disable-rls')

        assert run.returncode == 0, run.stderr
        assert await _installed_objects(pg_rls_db) == {}

    @pytest.mark.asyncio
    async def test_a_connected_member_refuses_the_removal(self, pg_rls_db: RlsUrls, installed: dict[str, int]) -> None:
        async with _connect(pg_rls_db.app) as member:
            pid = member.get_server_pid()

            run = await _run(pg_rls_db.maintenance, '--disable-rls')

        assert run.returncode == 1
        assert f'pid {pid} ({REQUEST_ROLE}' in run.stderr
        assert '--force' in run.stderr
        assert await _installed_objects(pg_rls_db) == installed

    @pytest.mark.asyncio
    async def test_a_connected_superuser_session_does_not_refuse_the_removal(
        self, pg_rls_db: RlsUrls, installed: dict[str, int],
    ) -> None:
        """A superuser counts as a member of every role, yet no superuser serves requests with row-level security on."""
        _ = installed
        async with _connect(pg_rls_db.admin) as superuser:
            superuser_pid = superuser.get_server_pid()
            async with _connect(pg_rls_db.app) as member:
                member_pid = member.get_server_pid()

                refused = await _run(pg_rls_db.maintenance, '--disable-rls')

            run = await _run(pg_rls_db.maintenance, '--disable-rls')

        assert refused.returncode == 1
        assert f'pid {member_pid} ({REQUEST_ROLE}' in refused.stderr
        assert f'pid {superuser_pid} (' not in refused.stderr
        assert run.returncode == 0, run.stderr
        assert await _installed_objects(pg_rls_db) == {}

    @pytest.mark.asyncio
    async def test_force_removes_while_a_member_is_connected(self, pg_rls_db: RlsUrls, installed: dict[str, int]) -> None:
        _ = installed
        async with _connect(pg_rls_db.app) as member:
            pid = member.get_server_pid()

            run = await _run(pg_rls_db.maintenance, '--disable-rls', '--force')

        assert run.returncode == 0, run.stderr
        assert f'pid {pid} ({REQUEST_ROLE}' in run.stderr
        assert await _installed_objects(pg_rls_db) == {}

    @pytest.mark.asyncio
    async def test_the_request_role_is_refused(self, pg_rls_db: RlsUrls, installed: dict[str, int]) -> None:
        run = await _run(pg_rls_db.app, '--disable-rls')

        assert run.returncode == 1
        assert await _refusal(pg_rls_db) in run.stderr
        assert await _installed_objects(pg_rls_db) == installed
