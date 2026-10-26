"""Tests for the server lifespan in app/server.py.

Covers startup failure cleanup, migration failure propagation, the shutdown of the
backend and the embedding provider, and the PostgreSQL start: the request role is
discovered and refused before any DDL when the policies could not hold it, the schema is
prepared on a maintenance backend logged in as the table owner, the runtime check
refuses request and migration credentials that resolve to one role with row-level
security on, and the request backend opens afterwards and is checked whenever the roles
are split. The read-only principal is logged right after the row-level security posture
warning, before any tool registers.
"""

import dataclasses
import os
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.startup as app_startup
from app.errors import ConfigurationError
from app.startup.rls_posture import RequestRole
from tests.helpers import patch_database_setup_steps
from tests.helpers import patch_generation_worker_phase


class TestLifespanErrorHandling:
    """Tests for lifespan() error handling."""

    @pytest.mark.asyncio
    async def test_startup_failure_shuts_down_backend(self, tmp_path: Path) -> None:
        """Verify backend shutdown on startup failure."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
            'ENABLE_SEMANTIC_SEARCH': 'false',
            'ENABLE_FTS': 'false',
        }

        with patch.dict(os.environ, env, clear=False):
            # Mock backend that will fail during init_database
            mock_backend = MagicMock()
            mock_backend.initialize = AsyncMock()
            mock_backend.shutdown = AsyncMock()
            mock_backend.backend_type = 'sqlite'

            with (
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.startup.database_setup.init_database', side_effect=RuntimeError('Database init failed')),
            ):
                from app.server import lifespan

                # Mock FastMCP instance
                mock_mcp = MagicMock()

                # Call lifespan and expect it to raise
                with pytest.raises(RuntimeError, match='Database init failed'):
                    async with lifespan(mock_mcp):
                        pass

                # Verify backend was shut down on failure
                mock_backend.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_migration_failure_propagates(self, tmp_path: Path) -> None:
        """Verify migration errors not swallowed."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
            'ENABLE_SEMANTIC_SEARCH': 'false',
            'ENABLE_FTS': 'false',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.initialize = AsyncMock()
            mock_backend.shutdown = AsyncMock()
            mock_backend.backend_type = 'sqlite'

            with (
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.startup.database_setup.init_database', new=AsyncMock()),
                patch('app.startup.database_setup.handle_metadata_indexes', new=AsyncMock()),
                patch('app.startup.database_setup.guard_compression_disable_over_populated', new=AsyncMock()),
                patch(
                    'app.startup.database_setup.apply_semantic_search_migration',
                    side_effect=RuntimeError('Migration failed'),
                ),
            ):
                from app.server import lifespan

                mock_mcp = MagicMock()

                # Migration failure should propagate
                with pytest.raises(RuntimeError, match='Migration failed'):
                    async with lifespan(mock_mcp):
                        pass

    @pytest.mark.asyncio
    async def test_shutdown_logs_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify shutdown errors logged not raised."""
        import logging

        import app.startup
        from app.server import lifespan

        caplog.set_level(logging.ERROR)

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        # Shutdown will raise an error
        mock_backend.shutdown = AsyncMock(side_effect=RuntimeError('Shutdown failed'))
        mock_backend.backend_type = 'sqlite'
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        # Mock settings - disable embedding generation to avoid initialization
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        # register_search_tools reads semantic_search.mode; 'false' forces the tool off.
        mock_settings.semantic_search.mode = 'false'
        mock_settings.semantic_search.enabled = False
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch_database_setup_steps(),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

                # Should NOT raise despite shutdown error
                async with lifespan(mock_mcp):
                    pass

                # Verify error was logged
                assert any('shutdown' in r.message.lower() for r in caplog.records)
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider

    @pytest.mark.asyncio
    async def test_embedding_provider_shutdown_on_exit(self) -> None:
        """Verify embedding provider shutdown called on exit."""
        import app.startup
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        # Create mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.initialize = AsyncMock()
        mock_embedding_provider.shutdown = AsyncMock()
        mock_embedding_provider.is_available = AsyncMock(return_value=True)
        mock_embedding_provider.provider_name = 'test-provider'

        # Mock settings - enable embedding generation
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        # register_search_tools reads semantic_search.mode; 'true' with a provider present
        # registers semantic_search_context.
        mock_settings.semantic_search.mode = 'true'
        mock_settings.semantic_search.enabled = True
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch_database_setup_steps(),
                patch_generation_worker_phase(),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch('app.startup.providers.check_vector_storage_dependencies', new=AsyncMock(return_value=True)),
                patch(
                    'app.startup.providers.check_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None, 'install_instructions': None}),
                ),
                patch('app.startup.providers.create_embedding_provider', return_value=mock_embedding_provider),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

                async with lifespan(mock_mcp):
                    # Verify embedding provider was set
                    assert app.startup._embedding_provider is not None

                # Verify embedding provider shutdown was called
                mock_embedding_provider.shutdown.assert_awaited_once()
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider


APP_ROLE = RequestRole(name='mcp_app', is_superuser=False, bypass_rls=False, policy_role_exists=True, policy_member=True)


class _PostgreSQLStart:
    """Patches the PostgreSQL start of the lifespan and records, in order, every step it takes.

    The discovered request role and the role the maintenance backend logs in as are given;
    the role checks themselves run unpatched, so a refusal is the real one.
    """

    def __init__(self, *, request_role: RequestRole | None, maintenance_role: str, rls: bool) -> None:
        self.request_role = request_role
        self.maintenance_role = maintenance_role
        self.steps: list[str] = []
        self.settings = MagicMock()
        self.settings.storage.backend_type = 'postgresql'
        self.settings.storage.postgresql_migration_user = None if request_role is None else 'mcp_owner'
        self.settings.access_control.postgresql_rls_enabled = rls
        self.settings.access_control.postgresql_rls_policy_role = 'mcp_context_request'
        self.settings.access_control.anonymous_principal = None
        self.settings.auth.provider = 'none'
        self.maintenance = self._backend('maintenance')
        self.request = self._backend('request')
        self.prepare_database = AsyncMock(side_effect=self._record('prepare_database'))
        self.verify_rls_posture = AsyncMock(side_effect=self._record('verify_rls_posture'))

    def _record(self, step: str) -> Callable[..., Awaitable[None]]:
        async def _step(*_args: object, **_kwargs: object) -> None:
            self.steps.append(step)

        return _step

    def _backend(self, name: str) -> MagicMock:
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        backend.initialize = AsyncMock(side_effect=self._record(f'{name}.initialize'))
        backend.shutdown = AsyncMock(side_effect=self._record(f'{name}.shutdown'))
        return backend

    async def _discover(self, _settings: object) -> RequestRole | None:
        self.steps.append('discover_request_role')
        return self.request_role

    async def _read_login_role(self, backend: object) -> str:
        assert backend is self.maintenance
        self.steps.append('read_login_role')
        return self.maintenance_role

    def _create_maintenance_backend(self) -> MagicMock:
        self.steps.append('create_maintenance_backend')
        return self.maintenance

    def _create_backend(self) -> MagicMock:
        self.steps.append('create_backend')
        return self.request

    def _register_core_tools(self, *_args: object) -> None:
        self.steps.append('register_core_tools')

    @contextmanager
    def patched(self) -> Generator[None]:
        """Install the patches, restoring the startup globals afterwards."""
        original = (app_startup._backend, app_startup._repositories)
        try:
            with (
                patch('app.server.settings', self.settings),
                patch('app.server.discover_request_role', side_effect=self._discover),
                patch('app.server.read_login_role', side_effect=self._read_login_role),
                patch('app.server.create_maintenance_backend', side_effect=self._create_maintenance_backend),
                patch('app.server.create_backend', side_effect=self._create_backend),
                patch('app.server.prepare_database', self.prepare_database),
                patch('app.server.verify_rls_posture', self.verify_rls_posture),
                patch('app.server.RepositoryContainer', return_value=MagicMock()),
                patch('app.server.register_core_tools', side_effect=self._register_core_tools),
                patch('app.server.initialize_providers', new=AsyncMock()),
                patch('app.server.register_search_tools', new=AsyncMock()),
                patch('app.server.install_json_string_middleware', new=AsyncMock()),
            ):
                yield
        finally:
            app_startup._backend, app_startup._repositories = original


class TestPostgreSQLMaintenancePhase:
    """On PostgreSQL the schema is prepared as the table owner before the request backend opens."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('rls', [True, False])
    async def test_split_roles_prepare_as_the_owner_then_verify_the_request_role(self, rls: bool) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(request_role=APP_ROLE, maintenance_role='mcp_owner', rls=rls)
        with start.patched():
            async with lifespan(MagicMock()):
                assert app_startup.get_backend() is start.request

        assert start.steps == [
            'discover_request_role',
            'create_maintenance_backend',
            'maintenance.initialize',
            'read_login_role',
            'prepare_database',
            'maintenance.shutdown',
            'create_backend',
            'request.initialize',
            'verify_rls_posture',
            'register_core_tools',
            'request.shutdown',
        ]
        start.prepare_database.assert_awaited_once_with(start.maintenance, start.settings, request_role='mcp_app')
        start.verify_rls_posture.assert_awaited_once_with(
            start.request, start.settings, request_role=APP_ROLE, maintenance_role='mcp_owner',
        )

    @pytest.mark.asyncio
    async def test_one_role_by_configuration_grants_nothing_and_checks_nothing(self) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(request_role=None, maintenance_role='postgres', rls=False)
        with start.patched():
            async with lifespan(MagicMock()):
                pass

        start.prepare_database.assert_awaited_once_with(start.maintenance, start.settings, request_role=None)
        start.verify_rls_posture.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_effective_role_with_the_flag_off_is_not_split(self) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(
            request_role=dataclasses.replace(APP_ROLE, name='postgres'), maintenance_role='postgres', rls=False,
        )
        with start.patched():
            async with lifespan(MagicMock()):
                pass

        start.prepare_database.assert_awaited_once_with(start.maintenance, start.settings, request_role=None)
        start.verify_rls_posture.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_effective_role_with_the_flag_on_is_refused_before_any_ddl(self) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(
            request_role=dataclasses.replace(APP_ROLE, name='mcp_owner'), maintenance_role='mcp_owner', rls=True,
        )
        with start.patched(), pytest.raises(ConfigurationError, match="resolve to the same PostgreSQL role 'mcp_owner'"):
            async with lifespan(MagicMock()):
                pass

        assert start.steps == [
            'discover_request_role',
            'create_maintenance_backend',
            'maintenance.initialize',
            'read_login_role',
            'maintenance.shutdown',
        ]

    @pytest.mark.asyncio
    async def test_an_unsafe_request_role_is_refused_before_the_maintenance_backend_opens(self) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(
            request_role=dataclasses.replace(APP_ROLE, is_superuser=True), maintenance_role='mcp_owner', rls=True,
        )
        with start.patched(), pytest.raises(ConfigurationError, match='rolsuper=true'):
            async with lifespan(MagicMock()):
                pass

        assert start.steps == ['discover_request_role']

    @pytest.mark.asyncio
    async def test_a_failed_preparation_still_shuts_the_maintenance_backend_down(self) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(request_role=APP_ROLE, maintenance_role='mcp_owner', rls=True)
        start.prepare_database.side_effect = RuntimeError('migration failed')
        with start.patched(), pytest.raises(RuntimeError, match='migration failed'):
            async with lifespan(MagicMock()):
                pass

        assert start.steps[-2:] == ['read_login_role', 'maintenance.shutdown']
        assert 'create_backend' not in start.steps

    @pytest.mark.asyncio
    async def test_the_read_only_principal_is_logged_right_after_the_rls_posture_warning(self) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(request_role=APP_ROLE, maintenance_role='mcp_owner', rls=True)

        def _warn(settings: object) -> None:
            assert settings is start.settings
            start.steps.append('warn_jwt_without_rls')

        def _log(access: object) -> None:
            assert access is start.settings.access_control
            start.steps.append('log_read_only_principal')

        with (
            start.patched(),
            patch('app.server.warn_jwt_without_rls', side_effect=_warn),
            patch('app.server.log_read_only_principal', side_effect=_log),
        ):
            async with lifespan(MagicMock()):
                pass

        assert start.steps[start.steps.index('verify_rls_posture'):] == [
            'verify_rls_posture',
            'warn_jwt_without_rls',
            'log_read_only_principal',
            'register_core_tools',
            'request.shutdown',
        ]

    @pytest.mark.asyncio
    async def test_a_refused_posture_shuts_the_request_backend_down(self) -> None:
        from app.server import lifespan

        start = _PostgreSQLStart(request_role=APP_ROLE, maintenance_role='mcp_owner', rls=True)
        start.verify_rls_posture.side_effect = ConfigurationError('posture refused')
        with start.patched(), pytest.raises(ConfigurationError, match='posture refused'):
            async with lifespan(MagicMock()):
                pass

        start.request.shutdown.assert_awaited_once()
