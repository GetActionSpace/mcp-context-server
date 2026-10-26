"""Tests for the tool registration phases of the server lifespan (app/startup/tool_registration.py).

Covers the tri-state search tool registration matrix, the non-fatal full-text search
availability probe, and the signatures of every tool the phases register.
"""

import ast
from pathlib import Path
from typing import get_args
from typing import get_type_hints
from unittest.mock import patch

import pytest
from fastmcp import Context

from tests.helpers import patch_database_setup_steps
from tests.helpers import patch_generation_worker_phase


class TestSearchToolRegistrationMatrix:
    """Tri-state registration matrix for the three search tools.

    Exercises register_search_tools in app/startup/tool_registration.py through
    lifespan() at the unit level: mocked settings plus a mocked (or absent) embedding
    provider drive whether semantic_search_context, fts_search_context, and
    hybrid_search_context are registered. The registration decision reads
    settings.semantic_search.mode (string) for semantic search and the derived
    settings.fts.enabled / settings.hybrid_search.enabled properties for the
    other two tools.
    """

    @staticmethod
    async def _run_lifespan(
        *,
        semantic_mode: str,
        fts_enabled: bool,
        hybrid_enabled: bool,
        provider_present: bool,
        fts_probe_error: BaseException | None = None,
    ) -> set[str]:
        """Run lifespan() with mocked dependencies and capture registered tools.

        Args:
            semantic_mode: Value bound to settings.semantic_search.mode.
            fts_enabled: Value bound to settings.fts.enabled property.
            hybrid_enabled: Value bound to settings.hybrid_search.enabled property.
            provider_present: When True, an embedding provider is created and set
                so get_embedding_provider() returns it; when False, embedding
                generation is disabled and no provider is initialized.
            fts_probe_error: When given, the FTS availability probe raises it instead
                of reporting availability, modeling an operational fault at boot.

        Returns:
            The set of registered tool function names captured from the
            app.startup.tool_registration.register_tool calls made during lifespan startup.
        """
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        import app.startup
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        mock_repos = MagicMock()
        if fts_probe_error is None:
            mock_repos.fts.is_available = AsyncMock(return_value=True)
        else:
            mock_repos.fts.is_available = AsyncMock(side_effect=fts_probe_error)
        mock_repos.context = MagicMock()
        mock_repos.embedding = MagicMock()

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = provider_present
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        mock_settings.semantic_search.mode = semantic_mode
        # The derived .enabled property mirrors mode != 'false'; set it
        # explicitly because the hybrid registration reads the property directly.
        mock_settings.semantic_search.enabled = semantic_mode != 'false'
        mock_settings.fts.enabled = fts_enabled
        mock_settings.fts.language = 'english'
        mock_settings.hybrid_search.enabled = hybrid_enabled
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.initialize = AsyncMock()
        mock_embedding_provider.shutdown = AsyncMock()
        mock_embedding_provider.is_available = AsyncMock(return_value=True)
        mock_embedding_provider.provider_name = 'test-provider'

        registered: set[str] = set()

        def _capture_register_tool(_mcp: object, func: object, *_args: object, **_kwargs: object) -> bool:
            registered.add(getattr(func, '__name__', repr(func)))
            return True

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch_database_setup_steps(),
                patch_generation_worker_phase(),
                patch('app.startup.tool_registration.register_tool', side_effect=_capture_register_tool),
                patch('app.startup.tool_registration.generate_fts_description', return_value='fts description'),
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
                    pass
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider

        return registered

    @pytest.mark.asyncio
    async def test_semantic_auto_with_provider_registers(self) -> None:
        """mode='auto' + provider present -> semantic_search_context registered."""
        registered = await self._run_lifespan(
            semantic_mode='auto',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=True,
        )
        assert 'semantic_search_context' in registered

    @pytest.mark.asyncio
    async def test_semantic_auto_without_provider_not_registered(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """mode='auto' + no provider -> NOT registered, logged at INFO (not WARNING)."""
        import logging

        caplog.set_level(logging.INFO, logger='app.startup.tool_registration')

        registered = await self._run_lifespan(
            semantic_mode='auto',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=False,
        )
        assert 'semantic_search_context' not in registered

        semantic_records = [
            r for r in caplog.records
            if 'semantic_search_context not registered' in r.message
        ]
        assert len(semantic_records) >= 1
        # auto + no provider is an informational skip, never a warning
        assert all(r.levelno == logging.INFO for r in semantic_records)

    @pytest.mark.asyncio
    async def test_semantic_true_without_provider_warns_not_registered(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """mode='true' + no provider -> NOT registered + a WARNING logged."""
        import logging

        caplog.set_level(logging.INFO, logger='app.startup.tool_registration')

        registered = await self._run_lifespan(
            semantic_mode='true',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=False,
        )
        assert 'semantic_search_context' not in registered

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'ENABLE_SEMANTIC_SEARCH=true' in r.message
        ]
        assert len(warnings) >= 1

    @pytest.mark.asyncio
    async def test_semantic_false_with_provider_not_registered(self) -> None:
        """mode='false' -> NOT registered even when an embedding provider exists."""
        registered = await self._run_lifespan(
            semantic_mode='false',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=True,
        )
        assert 'semantic_search_context' not in registered

    @pytest.mark.asyncio
    async def test_fts_and_hybrid_auto_register_by_default(self) -> None:
        """FTS and hybrid auto-register by default (enabled property True)."""
        registered = await self._run_lifespan(
            semantic_mode='auto',
            fts_enabled=True,
            hybrid_enabled=True,
            provider_present=False,
        )
        # FTS uses built-in database capabilities, so it registers with no provider.
        assert 'fts_search_context' in registered
        # Hybrid registers because at least one mode (FTS) is available.
        assert 'hybrid_search_context' in registered


class TestDiagnosticStartupProbesDoNotAbortStartup:
    """A startup probe whose only consumer is a log line cannot take the server down.

    The FTS availability probe reports whether the index is already provisioned; the tool is
    registered either way and re-checks migration status on every call. The probe deliberately
    lets operational faults propagate rather than reporting them as "not migrated", so an
    external VACUUM or backup holding the database lock past the read retry budget at the
    moment the server boots raises here. Inside the startup try that shuts every provider down
    and re-raises, that transient made EVERY tool unavailable; it must degrade to a warning.
    """

    @pytest.mark.asyncio
    async def test_fts_availability_probe_fault_leaves_the_server_running(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A locked database during the probe warns, registers the tool, and starts up."""
        import logging
        import sqlite3

        caplog.set_level(logging.INFO, logger='app.startup.tool_registration')

        registered = await TestSearchToolRegistrationMatrix._run_lifespan(
            semantic_mode='false',
            fts_enabled=True,
            hybrid_enabled=True,
            provider_present=False,
            fts_probe_error=sqlite3.OperationalError('database is locked'),
        )

        # Startup completed: the FTS tool and every other tool are still registered.
        assert 'fts_search_context' in registered
        assert 'hybrid_search_context' in registered
        assert 'store_context' in registered

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'full-text search availability' in r.message
        ]
        assert len(warnings) == 1
        assert 'database is locked' in warnings[0].message


def _declares_context(annotation: object) -> bool:
    """Report whether an annotation is, or wraps, ``fastmcp.Context``.

    Args:
        annotation: A resolved parameter annotation (``Context``, ``Context | None``,
            ``Annotated[Context, ...]``, and so on).

    Returns:
        True when ``Context`` appears anywhere inside the annotation.
    """
    if isinstance(annotation, type) and issubclass(annotation, Context):
        return True
    return any(_declares_context(argument) for argument in get_args(annotation))


class TestRegisteredToolSignatures:
    """The tools the server registers take no ``fastmcp.Context`` parameter.

    MCP client log notifications (what ``Context.info`` and its siblings send) are a
    deprecated protocol capability: every send emits an ``MCPDeprecationWarning``.
    No tool uses its ``Context`` for anything else, so none of them declares one.
    """

    @staticmethod
    def _registered_tool_names() -> set[str]:
        """Names of the functions ``app/startup/tool_registration.py`` passes to ``register_tool``.

        Returns:
            The second positional argument of every ``register_tool(mcp, <tool>)`` call.
        """
        import app.startup.tool_registration

        tree = ast.parse(Path(app.startup.tool_registration.__file__).read_text(encoding='utf-8'))
        return {
            node.args[1].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'register_tool'
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
        }

    def test_registration_calls_cover_every_annotated_tool(self) -> None:
        """The registration scan finds exactly the tools TOOL_ANNOTATIONS describes."""
        from app.tools import TOOL_ANNOTATIONS

        assert self._registered_tool_names() == set(TOOL_ANNOTATIONS)

    def test_no_registered_tool_declares_a_context_parameter(self) -> None:
        """No registered tool function has a parameter annotated with ``fastmcp.Context``."""
        import app.startup.tool_registration

        offenders: dict[str, list[str]] = {}
        for name in sorted(self._registered_tool_names()):
            hints = get_type_hints(getattr(app.startup.tool_registration, name), include_extras=True)
            context_parameters = [
                parameter for parameter, annotation in hints.items() if _declares_context(annotation)
            ]
            if context_parameters:
                offenders[name] = context_parameters

        assert offenders == {}, f'Tools declaring a Context parameter: {offenders}'
