"""
MCP Context Server implementation using FastMCP.

This server provides persistent multimodal context storage capabilities for LLM agents,
enabling shared memory across different conversation threads with support for text and images.
"""

import logging
import sys
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Literal
from typing import cast

from fastmcp import FastMCP
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

# ============================================================================
# CRITICAL: Logger Configuration MUST happen BEFORE importing app modules
# that trigger backend loading (app.backends, app.repositories)
# ============================================================================
from app.errors import ConfigurationError
from app.logger_config import config_logger
from app.settings import get_settings

# This is the FIRST get_settings() call in the import graph (the app module
# imports below bind their own module-level settings from the cache), so a
# settings ValidationError surfaces here -- at import time, before main()'s
# error-classification handler exists. Classify it inline: a validation
# failure (e.g. POSTGRESQL_POOL_MIN above POSTGRESQL_POOL_MAX, an
# out-of-bounds numeric) is a permanent misconfiguration and must exit with
# EX_CONFIG so supervisors do not restart-loop on it, not the generic exit 1
# an unhandled traceback produces.
try:
    settings = get_settings()
except ValidationError as e:
    print(f'Configuration invalid: {e}', file=sys.stderr)
    raise SystemExit(ConfigurationError.EXIT_CODE) from e
config_logger(settings.logging.level)
logger = logging.getLogger(__name__)

# Import auth factory for explicit auth configuration
from app.auth import create_auth_provider
from app.auth.access import log_read_only_principal
from app.backends import StorageBackend
from app.backends import create_backend
from app.backends import create_maintenance_backend
from app.errors import DependencyError
from app.repositories import RepositoryContainer

# Import from startup module - global state and initialization
from app.startup import DB_PATH
from app.startup import get_backend
from app.startup import get_embedding_provider
from app.startup import get_reranking_provider
from app.startup import get_summary_provider
from app.startup import set_backend
from app.startup import set_chunking_service
from app.startup import set_embedding_provider
from app.startup import set_repositories
from app.startup import set_reranking_provider
from app.startup import set_summary_provider

# Import the lifespan startup phases
from app.startup.database_setup import prepare_database
from app.startup.generation_worker import start_generation_worker
from app.startup.generation_worker import stop_generation_worker
from app.startup.providers import initialize_providers
from app.startup.rls_posture import discover_request_role
from app.startup.rls_posture import read_login_role
from app.startup.rls_posture import refuse_unsafe_request_role
from app.startup.rls_posture import split_request_role
from app.startup.rls_posture import verify_rls_posture
from app.startup.rls_posture import warn_jwt_without_rls
from app.startup.tool_registration import install_json_string_middleware
from app.startup.tool_registration import register_core_tools
from app.startup.tool_registration import register_search_tools


def _get_server_version() -> str:
    """Get server version from package metadata or pyproject.toml fallback.

    Returns:
        Version string (e.g., '0.14.0') or 'unknown' if unavailable.
    """
    # Primary: installed package metadata (works for pip, uv, editable installs)
    try:
        return pkg_version('mcp-context-server')
    except PackageNotFoundError:
        pass

    # Fallback: read directly from pyproject.toml (for running from source)
    try:
        pyproject_path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        if pyproject_path.exists():
            with pyproject_path.open('rb') as f:
                data = tomllib.load(f)
            version = data.get('project', {}).get('version')
            if isinstance(version, str):
                return version
    except Exception:
        pass

    return 'unknown'


# Cache version at module load time
SERVER_VERSION = _get_server_version()


async def _start_postgresql() -> StorageBackend:
    """Prepare the PostgreSQL database as the table owner, then open and check the request backend.

    The role requests act as is read first, and with row-level security on a role the
    policies could not hold is refused before any DDL. The schema is prepared on a
    short-lived maintenance backend logged in with the migration credentials; when the two
    backends act as different roles, the preparation grants the request role what it needs
    and the request backend is checked afterward. With row-level security on, request
    and migration credentials that act as one role are refused.

    Returns:
        The initialized request backend, installed as the backend every tool shares.
    """
    request_role = await discover_request_role(settings)
    refuse_unsafe_request_role(request_role, settings)
    maintenance = create_maintenance_backend()
    try:
        await maintenance.initialize()
        maintenance_role = await read_login_role(maintenance)
        split_role = split_request_role(request_role, maintenance_role, settings)
        await prepare_database(maintenance, settings, request_role=None if split_role is None else split_role.name)
    finally:
        try:
            await maintenance.shutdown()
        except Exception as shutdown_error:
            logger.error(f'Error shutting down the maintenance backend: {shutdown_error}')
    backend = create_backend()
    await backend.initialize()
    set_backend(backend)
    if split_role is not None:
        await verify_rls_posture(backend, settings, request_role=split_role, maintenance_role=maintenance_role)
    return backend


# Lifespan context manager for FastMCP
@asynccontextmanager
async def lifespan(mcp: FastMCP[None]) -> AsyncGenerator[None, None]:
    """Manage server lifecycle - initialize on startup, cleanup on shutdown.

    This ensures that the database manager's background tasks run in the
    same event loop as FastMCP, preventing the hanging issue.

    Args:
        mcp: The FastMCP server instance for tool registration

    Yields:
        None: Control is yielded back to FastMCP during server operation
    """
    # Startup
    try:
        if settings.storage.backend_type == 'postgresql':
            backend = await _start_postgresql()
        else:
            # Create backend ONCE at the start - used throughout initialization and runtime
            backend = create_backend(backend_type=None, db_path=DB_PATH)
            await backend.initialize()
            set_backend(backend)
            await prepare_database(backend, settings, request_role=None)
        warn_jwt_without_rls(settings)
        log_read_only_principal(settings.access_control)
        # Initialize repositories with the backend
        repos = RepositoryContainer(backend)
        set_repositories(repos)

        register_core_tools(mcp, settings)
        await initialize_providers(settings, backend.backend_type)
        await register_search_tools(mcp, settings, backend.backend_type, repos)
        await install_json_string_middleware(mcp)
        await start_generation_worker(settings, backend)

        logger.info(f'MCP Context Server initialized (backend: {backend.backend_type})')
    except Exception as e:
        logger.error(f'Failed to initialize server: {e}')
        # Shut down any provider already initialized before the failing step so a
        # startup failure does not leak its HTTP client / thread pools (mirrors
        # the clean-shutdown cleanup below). Each shutdown is isolated so one
        # provider's shutdown error cannot mask the original startup exception.
        for _startup_provider in (
            get_reranking_provider(),
            get_embedding_provider(),
            get_summary_provider(),
        ):
            if _startup_provider is not None:
                try:
                    await _startup_provider.shutdown()
                except Exception as shutdown_error:
                    logger.error(f'Error shutting down provider during startup failure: {shutdown_error}')
        startup_backend = get_backend()
        if startup_backend:
            try:
                await startup_backend.shutdown()
            except Exception as shutdown_error:
                logger.error(f'Error shutting down backend during startup failure: {shutdown_error}')
        raise

    # Yield control to FastMCP
    yield

    # Shutdown
    logger.info('Shutting down MCP Context Server')
    # At this point, startup succeeded and _backend must be set
    shutdown_backend = get_backend()
    assert shutdown_backend is not None
    # The worker gives its claims back through the backend, so it stops first.
    await stop_generation_worker()
    try:
        await shutdown_backend.shutdown()
    except Exception as e:
        logger.error(f'Error during shutdown: {e}')
    finally:
        # Shutdown reranking provider if initialized
        shutdown_reranking_provider = get_reranking_provider()
        if shutdown_reranking_provider is not None:
            try:
                await shutdown_reranking_provider.shutdown()
            except Exception as e:
                logger.error(f'Error shutting down reranking provider: {e}')

        # Shutdown embedding provider if initialized
        shutdown_embedding_provider = get_embedding_provider()
        if shutdown_embedding_provider is not None:
            try:
                await shutdown_embedding_provider.shutdown()
            except Exception as e:
                logger.error(f'Error shutting down embedding provider: {e}')

        # Shutdown summary provider if initialized
        shutdown_summary_provider = get_summary_provider()
        if shutdown_summary_provider is not None:
            try:
                await shutdown_summary_provider.shutdown()
            except Exception as e:
                logger.error(f'Error shutting down summary provider: {e}')

        set_backend(None)
        set_repositories(None)
        set_embedding_provider(None)
        set_reranking_provider(None)
        set_chunking_service(None)
        set_summary_provider(None)
    logger.info('MCP Context Server shutdown complete')


def main() -> None:
    """Main entry point for the MCP Context Server.

    Supports both stdio (default) and HTTP transport modes:
    - stdio: Default for local process spawning (uv run mcp-context-server)
    - http: For Docker/remote deployments (set MCP_TRANSPORT=http)

    Initialization and shutdown are handled by the lifespan context manager.

    Exit codes follow BSD sysexits.h convention for supervisor integration:
    - 0: Normal shutdown
    - 69 (EX_UNAVAILABLE): External dependency unavailable (supervisor may retry with backoff)
    - 78 (EX_CONFIG): Configuration error (supervisor should NOT restart)
    - 1: General error (unknown cause)
    """
    try:
        # Log server version first (before any subsystem messages)
        logger.info(f'MCP Context Server v{SERVER_VERSION}')

        # Determine transport mode early (controls auth and health endpoint)
        transport = settings.transport.transport
        logger.info(f'Transport: {transport.upper()}')

        # Initialize authentication provider only for HTTP transports
        # Auth has no effect on stdio (MCP specification: local process communication)
        if transport == 'stdio':
            if settings.auth.provider != 'none':
                logger.warning(
                    'MCP_AUTH_PROVIDER=%s is configured but has no effect on stdio transport. '
                    'Authentication is only applicable to HTTP transports.',
                    settings.auth.provider,
                )
            auth_provider = None
        else:
            auth_provider = create_auth_provider()

        # Resolve server instructions (env var override or default)
        from app.instructions import resolve_instructions

        instructions_text = resolve_instructions(settings.instructions)

        # Create FastMCP server with lifespan management and explicit auth
        # mask_error_details=False exposes validation errors for LLM autocorrection
        # strict_input_validation=False is pinned explicitly (not left to the
        # FASTMCP_STRICT_INPUT_VALIDATION env fallback): lax scalar coercion is
        # LOAD-BEARING for this server. Real MCP clients intermittently send
        # scalar params as JSON-encoded strings, and the project's
        # JsonStringDeserializerMiddleware deliberately repairs only array and
        # object params -- scalars rely on FastMCP's default coercion, which
        # strict mode disables (rejecting previously working tool calls).
        mcp = FastMCP(
            name='mcp-context-server',
            version=SERVER_VERSION,
            instructions=instructions_text or None,
            lifespan=lifespan,
            mask_error_details=False,
            strict_input_validation=False,
            auth=auth_provider,
        )

        if transport == 'stdio':
            mcp.run(transport=transport, show_banner=False)
        else:
            # Register health check endpoint for container orchestration (HTTP only)
            async def _health_handler(_: Request) -> JSONResponse:
                """Health check endpoint for Docker/Kubernetes liveness probes."""
                return JSONResponse({'status': 'ok'})

            mcp.custom_route('/health', methods=['GET'])(_health_handler)

            host = settings.transport.host
            port = settings.transport.port
            logger.info(f'Server URL: http://{host}:{port}/mcp')

            if transport in ('http', 'streamable-http'):
                stateless_http = settings.transport.stateless_http
                if not stateless_http:
                    logger.warning(
                        'Stateless HTTP mode: disabled. Server-side session tracking is active. '
                        'This requires sticky sessions for horizontal scaling.',
                    )
                mcp.run(
                    transport=cast(Literal['stdio', 'http', 'sse', 'streamable-http'], transport),
                    host=host,
                    port=port,
                    stateless_http=stateless_http,
                    # FastMCP forwards log_level to the uvicorn.Config it builds, whose
                    # __init__ dictConfig-installs the non-propagating uvicorn/uvicorn.access
                    # loggers at run time. Without this, uvicorn falls back to
                    # FASTMCP_LOG_LEVEL (default INFO) and prints its startup banner and one
                    # access line per request regardless of LOG_LEVEL. Passing it makes
                    # LOG_LEVEL govern the uvicorn tree too (uvicorn wants a lowercase name).
                    log_level=settings.logging.level.lower(),
                    show_banner=False,
                )
            else:
                # SSE transport does not support stateless mode. FastMCP raises
                # ValueError when stateless_http is True for transport='sse', and an
                # unset stateless_http is resolved from FASTMCP_STATELESS_HTTP (whose
                # documented default is true), so it MUST be passed explicitly as
                # False here or SSE crashes on startup under the project's own
                # default configuration. Warn when the configured value asked for
                # stateless so the operator knows SSE forces stateful sessions.
                if settings.transport.stateless_http:
                    logger.warning(
                        'Stateless HTTP mode is not supported by the SSE transport; '
                        'forcing stateful sessions. Server-side session tracking is '
                        'active, which requires sticky sessions for horizontal scaling. '
                        'Use the streamable-http transport for stateless mode.',
                    )
                mcp.run(
                    transport=cast(Literal['stdio', 'http', 'sse', 'streamable-http'], transport),
                    host=host,
                    port=port,
                    stateless_http=False,
                    # Pass LOG_LEVEL through so the uvicorn logger tree follows it on the SSE
                    # transport too (see the streamable-http branch above for the rationale).
                    log_level=settings.logging.level.lower(),
                    show_banner=False,
                )

    except KeyboardInterrupt:
        logger.info('Server shutdown requested')
    except ConfigurationError as e:
        # Configuration errors: missing packages, invalid settings, missing API keys
        # Exit code 78 (EX_CONFIG) signals supervisor NOT to restart
        logger.critical(f'Configuration error (will not retry): {e}')
        sys.exit(ConfigurationError.EXIT_CODE)
    except DependencyError as e:
        # Dependency errors: service down, model not pulled, network issues
        # Exit code 69 (EX_UNAVAILABLE) allows supervisor to retry with backoff
        logger.error(f'Dependency unavailable (may retry): {e}')
        sys.exit(DependencyError.EXIT_CODE)
    except Exception as e:
        logger.error(f'Server error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
