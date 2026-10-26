"""The server start checks the PostgreSQL roles before and after it prepares the database.

Server subprocesses start from :func:`tests.helpers.postgresql_subprocess_env` on the
``pg_rls_db`` database, which ``mcp_owner`` owns: with ``ENABLE_POSTGRESQL_RLS`` a
superuser, the table owner and a login role outside the policy role are refused as the
request role with exit 78 before any DDL, and so are request credentials that resolve at
run time to the migration role although the configured names differ. ``mcp_app`` starts,
and with the flag off a split start over row-level security enabled outside the server is
refused. The posture checks (:func:`verify_rls_posture`) run directly on a request
backend after a start: each revoked privilege, foreign policy, forced table and change to
the read view is named, and the next start repairs what the migration owns, revoking
every grant on the read view beyond the policy role's SELECT with a WARNING. The jwt
provider over HTTP without row-level security logs its warning.
"""

import asyncio
import logging
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import asyncpg
import pytest
import pytest_asyncio
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from app.backends import StorageBackend
from app.backends import create_backend
from app.errors import ConfigurationError
from app.settings import AppSettings
from app.startup.database_setup import prepare_database
from app.startup.rls_posture import RequestRole
from app.startup.rls_posture import verify_rls_posture
from tests.helpers import RlsCredentials
from tests.helpers import postgresql_subprocess_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import terminate
from tests.integration._http_jwt import wait_for_health
from tests.integration.postgresql._rls_catalog import configure_rls_layout
from tests.integration.postgresql._rls_catalog import maintenance_backend
from tests.integration.postgresql._rls_catalog import view_acl
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.integration.postgresql.conftest import RLS_CREDENTIALS
from tests.integration.postgresql.conftest import login_url

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
BOOT_TIMEOUT_S: Final = 180
POLICY_ROLE: Final = RLS_CREDENTIALS.policy_role
PR: Final = f'"{POLICY_ROLE}"'
VIEW: Final = '"public".context_entries_readable'
APP_ROLE: Final = RequestRole(
    name='mcp_app', is_superuser=False, bypass_rls=False, policy_role_exists=True, policy_member=True,
)
# Every subsystem that needs an external service is off, so a start needs only the database.
QUIET_START: Final = {
    'MCP_TRANSPORT': 'stdio',
    'MCP_AUTH_PROVIDER': 'none',
    'LOG_LEVEL': 'WARNING',
    'FASTMCP_ENABLE_RICH_LOGGING': 'false',
    'ENABLE_EMBEDDING_GENERATION': 'false',
    'ENABLE_SEMANTIC_SEARCH': 'false',
    'ENABLE_FTS': 'false',
    'ENABLE_HYBRID_SEARCH': 'false',
    'ENABLE_SUMMARY_GENERATION': 'false',
    'ENABLE_RERANKING': 'false',
    'ENABLE_EMBEDDING_COMPRESSION': 'false',
    'ENABLE_INDEX_TREE_NODE_SUMMARIES': 'false',
}
SERVER_COMMAND: Final = [sys.executable, '-c', 'from app.server import main; main()']


@dataclass(frozen=True, slots=True)
class Start:
    """What one server start printed and how it ended."""

    returncode: int
    stderr: str


def _start(request_url: str, *, rls: RlsCredentials | None, **overrides: str) -> Start:
    """Start the server over stdio with stdin closed, so a start that passes serves nothing and exits 0.

    Args:
        request_url: The request connection settings, as a connection string.
        rls: The row-level security credentials, or None to start with the flag off.
        overrides: Further environment values.

    Returns:
        The exit code and what the server logged.
    """
    result = subprocess.run(
        SERVER_COMMAND,
        cwd=REPO_ROOT,
        env=postgresql_subprocess_env(request_url, rls=rls, **{**QUIET_START, **overrides}),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=BOOT_TIMEOUT_S,
        check=False,
    )
    return Start(returncode=result.returncode, stderr=result.stderr)


def _serve_until_healthy(env: dict[str, str], port: int, log_path: Path) -> None:
    """Start the server over HTTP with ``env``, wait for its health endpoint and stop it, logging to ``log_path``."""
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(SERVER_COMMAND, cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=log)
        try:
            wait_for_health(f'http://127.0.0.1:{port}', proc)
        finally:
            terminate(proc)


async def _admin(urls: RlsUrls, *statements: str) -> None:
    """Run ``statements`` in the test database as the superuser."""
    conn = await asyncpg.connect(urls.admin)
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


async def _as_owner(urls: RlsUrls, *statements: str) -> None:
    """Run ``statements`` in the test database as the table owner."""
    conn = await asyncpg.connect(urls.maintenance)
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


class TestRefusedRequestRoles:
    """With the flag on, a request role the policies could not hold never reaches the DDL."""

    @pytest.mark.asyncio
    async def test_the_superuser_is_refused(self, pg_rls_db: RlsUrls) -> None:
        start = await asyncio.to_thread(_start, pg_rls_db.admin, rls=RLS_CREDENTIALS)

        assert start.returncode == ConfigurationError.EXIT_CODE, start.stderr
        assert "'mcp_test'" in start.stderr
        assert 'rolsuper=true' in start.stderr

    @pytest.mark.asyncio
    async def test_the_table_owner_is_refused(self, pg_rls_db: RlsUrls) -> None:
        start = await asyncio.to_thread(_start, pg_rls_db.maintenance, rls=RLS_CREDENTIALS)

        assert start.returncode == ConfigurationError.EXIT_CODE, start.stderr
        assert f'GRANT {PR} TO "mcp_owner";' in start.stderr

    @pytest.mark.asyncio
    async def test_a_login_role_outside_the_policy_role_is_refused(self, pg_rls_db: RlsUrls) -> None:
        start = await asyncio.to_thread(_start, pg_rls_db.outsider, rls=RLS_CREDENTIALS)

        assert start.returncode == ConfigurationError.EXIT_CODE, start.stderr
        assert f'GRANT {PR} TO "mcp_outsider";' in start.stderr

    @pytest.mark.asyncio
    async def test_a_login_role_acting_as_the_owner_is_refused_at_run_time(self, pg_rls_db: RlsUrls) -> None:
        """mcp_alias logs in under its own name, but every session it opens acts as mcp_owner."""
        await _admin(
            pg_rls_db,
            "CREATE ROLE mcp_alias LOGIN PASSWORD 'mcp_alias' NOSUPERUSER NOBYPASSRLS IN ROLE mcp_owner",
            'ALTER ROLE mcp_alias SET role = mcp_owner',
            f'GRANT {POLICY_ROLE} TO mcp_owner',
        )
        try:
            start = await asyncio.to_thread(_start, login_url(pg_rls_db.app, 'mcp_alias'), rls=RLS_CREDENTIALS)
        finally:
            await _admin(pg_rls_db, f'REVOKE {POLICY_ROLE} FROM mcp_owner', 'DROP ROLE mcp_alias')

        assert start.returncode == ConfigurationError.EXIT_CODE, start.stderr
        assert (
            "resolve to the same PostgreSQL role 'mcp_owner' (request current_user 'mcp_owner', "
            "POSTGRESQL_MIGRATION_USER='mcp_owner')"
        ) in start.stderr


class TestRefusedTableOwnerLogin:
    """A refused table-owner login names the migration credentials, not the request login that already succeeded."""

    @pytest.mark.asyncio
    async def test_a_wrong_migration_password_names_the_migration_credentials(self, pg_rls_db: RlsUrls) -> None:
        start = await asyncio.to_thread(
            _start, pg_rls_db.app, rls=RLS_CREDENTIALS, POSTGRESQL_MIGRATION_PASSWORD='not-the-owner-password',
        )

        assert start.returncode == ConfigurationError.EXIT_CODE, start.stderr
        assert 'password authentication failed for user "mcp_owner"' in start.stderr
        remediation = 'Check POSTGRESQL_MIGRATION_USER, POSTGRESQL_MIGRATION_PASSWORD and the server pg_hba.conf rules.'
        assert remediation in start.stderr
        assert 'Check POSTGRESQL_USER' not in start.stderr


class TestStarts:
    """The request role mcp_app starts; a split start with the flag off refuses row-level security it did not install."""

    @pytest.mark.asyncio
    async def test_the_request_role_starts_with_the_flag_on(self, pg_rls_db: RlsUrls) -> None:
        start = await asyncio.to_thread(_start, pg_rls_db.app, rls=RLS_CREDENTIALS)

        assert start.returncode == 0, start.stderr
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            owner_rows = await conn.fetch("SELECT DISTINCT tableowner FROM pg_tables WHERE schemaname = 'public'")
            can_insert = await conn.fetchval("SELECT has_table_privilege('mcp_app', 'public.context_entries', 'INSERT')")
            enabled = await conn.fetchval("SELECT relrowsecurity FROM pg_class WHERE oid = 'public.context_entries'::regclass")
        finally:
            await conn.close()
        assert {str(row[0]) for row in owner_rows} == {'mcp_owner'}
        assert can_insert is True
        assert enabled is True

    @pytest.mark.asyncio
    async def test_row_level_security_enabled_outside_the_server_is_refused_with_the_flag_off(
        self, pg_rls_db: RlsUrls,
    ) -> None:
        split = {'POSTGRESQL_MIGRATION_USER': 'mcp_owner', 'POSTGRESQL_MIGRATION_PASSWORD': 'mcp_owner'}
        first = await asyncio.to_thread(_start, pg_rls_db.app, rls=None, **split)
        assert first.returncode == 0, first.stderr
        await _as_owner(pg_rls_db, 'ALTER TABLE public.tags ENABLE ROW LEVEL SECURITY')

        second = await asyncio.to_thread(_start, pg_rls_db.app, rls=None, **split)

        assert second.returncode == ConfigurationError.EXIT_CODE, second.stderr
        assert 'ALTER TABLE "public".tags DISABLE ROW LEVEL SECURITY' in second.stderr

    @pytest.mark.asyncio
    async def test_a_grant_on_the_read_view_is_revoked_by_the_next_start(self, pg_rls_db: RlsUrls) -> None:
        first = await asyncio.to_thread(_start, pg_rls_db.app, rls=RLS_CREDENTIALS)
        assert first.returncode == 0, first.stderr
        await _as_owner(pg_rls_db, f'GRANT SELECT ON {VIEW} TO mcp_outsider')

        second = await asyncio.to_thread(_start, pg_rls_db.app, rls=RLS_CREDENTIALS)

        assert second.returncode == 0, second.stderr
        assert 'mcp_outsider (SELECT)' in second.stderr
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            assert await view_acl(conn, VIEW) == [(POLICY_ROLE, 'SELECT')]
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_jwt_over_http_without_row_level_security_warns(self, pg_rls_db: RlsUrls, tmp_path: Path) -> None:
        port = free_port()
        env = postgresql_subprocess_env(
            pg_rls_db.app,
            rls=None,
            **{
                **QUIET_START,
                'POSTGRESQL_MIGRATION_USER': 'mcp_owner',
                'POSTGRESQL_MIGRATION_PASSWORD': 'mcp_owner',
                'MCP_TRANSPORT': 'http',
                'FASTMCP_HOST': '127.0.0.1',
                'FASTMCP_PORT': str(port),
                'MCP_AUTH_PROVIDER': 'jwt',
                'MCP_AUTH_JWT_PUBLIC_KEY': RSAKeyPair.generate().public_key,
            },
        )
        log_path = tmp_path / 'server.log'

        await asyncio.to_thread(_serve_until_healthy, env, port, log_path)

        assert (
            'MCP_AUTH_PROVIDER=jwt on PostgreSQL without ENABLE_POSTGRESQL_RLS: '
            'access isolation is enforced in the application layer only.'
        ) in log_path.read_text(encoding='utf-8')


@asynccontextmanager
async def _request_backend(url: str) -> AsyncGenerator[StorageBackend]:
    """Yield an initialized request backend logged in with the credentials of ``url``."""
    backend = create_backend(backend_type='postgresql', connection_string=url)
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.shutdown()


async def _prepare(urls: RlsUrls, settings: AppSettings) -> None:
    """Prepare the database as a start with split roles does, as the table owner."""
    async with maintenance_backend(urls.maintenance) as backend:
        await prepare_database(backend, settings, request_role=APP_ROLE.name)


async def _refusal(urls: RlsUrls, settings: AppSettings, *, request_url: str | None = None) -> str:
    """Run the posture checks for mcp_app and return the refusal."""
    async with _request_backend(request_url or urls.app) as backend:
        with pytest.raises(ConfigurationError) as refusal:
            await verify_rls_posture(backend, settings, request_role=APP_ROLE, maintenance_role='mcp_owner')
    return str(refusal.value)


async def _passes(urls: RlsUrls, settings: AppSettings) -> None:
    """Run the posture checks for mcp_app, which must pass."""
    async with _request_backend(urls.app) as backend:
        await verify_rls_posture(backend, settings, request_role=APP_ROLE, maintenance_role='mcp_owner')


@pytest_asyncio.fixture
async def started(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AppSettings:
    """The ``pg_rls_db`` database prepared as a start with row-level security and split roles prepares it.

    Returns:
        The settings of that start.
    """
    settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides={}, rls=True)
    await _prepare(pg_rls_db, settings)
    return settings


class TestPosture:
    """The posture checks name each departure from what the start installed."""

    @pytest.mark.asyncio
    async def test_a_prepared_database_passes_and_the_request_backend_initializes_with_the_extension_present(
        self, pg_rls_db: RlsUrls, started: AppSettings,
    ) -> None:
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            assert await conn.fetchval("SELECT count(*) FROM pg_extension WHERE extname = 'vector'") == 1
        finally:
            await conn.close()

        await _passes(pg_rls_db, started)

    @pytest.mark.asyncio
    async def test_a_revoked_table_privilege_is_named(self, pg_rls_db: RlsUrls, started: AppSettings) -> None:
        await _as_owner(pg_rls_db, f'REVOKE INSERT ON public.context_entries FROM {POLICY_ROLE}')

        refusal = await _refusal(pg_rls_db, started)

        assert f'GRANT INSERT ON TABLE "public".context_entries TO {PR}' in refusal

    @pytest.mark.asyncio
    async def test_a_revoked_sequence_privilege_is_named(self, pg_rls_db: RlsUrls, started: AppSettings) -> None:
        conn = await asyncpg.connect(pg_rls_db.maintenance)
        try:
            sequence = str(await conn.fetchval("SELECT pg_get_serial_sequence('public.tags', 'id')"))
            await conn.execute(f'REVOKE USAGE ON SEQUENCE {sequence} FROM {POLICY_ROLE}')
        finally:
            await conn.close()

        refusal = await _refusal(pg_rls_db, started)

        assert f'GRANT USAGE ON SEQUENCE "public"."{sequence.removeprefix("public.")}" TO {PR}' in refusal

    @pytest.mark.asyncio
    async def test_a_login_role_outside_the_policy_role_is_named(self, pg_rls_db: RlsUrls, started: AppSettings) -> None:
        await _admin(pg_rls_db, "CREATE ROLE mcp_posture_probe LOGIN PASSWORD 'mcp_posture_probe' NOSUPERUSER NOBYPASSRLS")
        try:
            refusal = await _refusal(pg_rls_db, started, request_url=login_url(pg_rls_db.app, 'mcp_posture_probe'))
        finally:
            await _admin(pg_rls_db, 'DROP ROLE mcp_posture_probe')

        assert f'not a member of the policy role {PR}' in refusal

    @pytest.mark.asyncio
    @pytest.mark.parametrize('grantee', ['PUBLIC', 'mcp_app'])
    async def test_a_foreign_policy_that_applies_to_the_request_role_is_refused(
        self, pg_rls_db: RlsUrls, started: AppSettings, grantee: str,
    ) -> None:
        await _as_owner(pg_rls_db, f'CREATE POLICY tags_extra ON public.tags FOR SELECT TO {grantee} USING (true)')

        refusal = await _refusal(pg_rls_db, started)

        assert 'tags_extra' in refusal
        assert 'DROP POLICY "tags_extra" ON "public".tags' in refusal

    @pytest.mark.asyncio
    async def test_a_foreign_policy_for_another_role_is_not_refused(self, pg_rls_db: RlsUrls, started: AppSettings) -> None:
        await _as_owner(pg_rls_db, 'CREATE POLICY tags_outsider ON public.tags FOR SELECT TO mcp_outsider USING (true)')

        await _passes(pg_rls_db, started)

    @pytest.mark.asyncio
    async def test_forced_row_level_security_is_refused(self, pg_rls_db: RlsUrls, started: AppSettings) -> None:
        await _as_owner(pg_rls_db, 'ALTER TABLE public.tags FORCE ROW LEVEL SECURITY')

        refusal = await _refusal(pg_rls_db, started)

        assert 'ALTER TABLE "public".tags NO FORCE ROW LEVEL SECURITY' in refusal

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('change', 'named'),
        [
            (f'ALTER VIEW {VIEW} SET (security_barrier = true)', 'security_barrier'),
            (f'DROP TRIGGER reject_writes ON {VIEW}', 'reject_writes'),
            (f'CREATE OR REPLACE VIEW {VIEW} AS SELECT ce.* FROM public.context_entries ce', 'context_entries_readable'),
        ],
        ids=['security-barrier', 'missing-trigger', 'no-gate'],
    )
    async def test_a_changed_read_view_is_refused_and_the_next_start_repairs_it(
        self, pg_rls_db: RlsUrls, started: AppSettings, change: str, named: str,
    ) -> None:
        await _as_owner(pg_rls_db, change)

        refusal = await _refusal(pg_rls_db, started)
        await _prepare(pg_rls_db, started)

        assert named in refusal
        await _passes(pg_rls_db, started)

    @pytest.mark.asyncio
    async def test_a_read_view_of_another_owner_is_refused(self, pg_rls_db: RlsUrls, started: AppSettings) -> None:
        await _admin(pg_rls_db, f'ALTER VIEW {VIEW} OWNER TO mcp_outsider')

        refusal = await _refusal(pg_rls_db, started)

        assert 'not owned by the owner of' in refusal

    @pytest.mark.asyncio
    async def test_a_grant_on_the_read_view_after_the_migration_prints_the_revoke(
        self, pg_rls_db: RlsUrls, started: AppSettings,
    ) -> None:
        await _as_owner(pg_rls_db, f'GRANT SELECT ON {VIEW} TO mcp_outsider')

        refusal = await _refusal(pg_rls_db, started)

        assert f'REVOKE ALL ON {VIEW} FROM "mcp_outsider";' in refusal

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('grant', 'named'),
        [
            (f'GRANT SELECT ON {VIEW} TO mcp_outsider', 'mcp_outsider (SELECT)'),
            ('GRANT ALL ON ALL TABLES IN SCHEMA public TO mcp_outsider', 'mcp_outsider ('),
            (f'GRANT INSERT ON {VIEW} TO {POLICY_ROLE}', f'{POLICY_ROLE} (INSERT)'),
        ],
        ids=['view-select', 'schema-wide', 'policy-role-insert'],
    )
    async def test_the_next_start_revokes_a_grant_on_the_read_view_with_a_warning(
        self, pg_rls_db: RlsUrls, started: AppSettings, grant: str, named: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _as_owner(pg_rls_db, grant)
        caplog.set_level(logging.WARNING, logger='app.migrations.rls')

        await _prepare(pg_rls_db, started)

        assert any(named in record.getMessage() for record in caplog.records)
        await _passes(pg_rls_db, started)
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            assert await view_acl(conn, VIEW) == [(POLICY_ROLE, 'SELECT')]
        finally:
            await conn.close()
