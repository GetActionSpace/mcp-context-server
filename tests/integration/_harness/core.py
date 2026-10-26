"""Shared plumbing of the real-server integration harness.

``HarnessCore`` owns the per-run state: the server environment for each
backend, the FastMCP client and the second-server helper, image and result
extraction, the tool call that reports a tool error instead of raising it, and
the result list every check reports into. ``ClientMode``,
``CLIENT_MODES`` and ``ERA_PROTOCOL_VERSIONS`` name the protocol eras a
harness client negotiates.

Every PostgreSQL server the harness starts gets its environment from
:func:`tests.helpers.postgresql_subprocess_env`: it inherits the provider
configuration of the parent environment, never its PostgreSQL routing, and
carries the row-level security credentials of the run when it has them, so
every server on the run's database starts the way the primary did.
"""

import base64
import contextlib
import os
import tempfile
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from typing import Literal

from anyio import Path as AsyncPath
from fastmcp import Client
from fastmcp.client.transports import ClientTransport
from fastmcp.client.transports import PythonStdioTransport
from mcp.client.stdio import get_default_environment
from mcp.types import TextContent
from mcp.types.version import HANDSHAKE_PROTOCOL_VERSIONS
from mcp.types.version import MODERN_PROTOCOL_VERSIONS

from tests.helpers import RlsCredentials
from tests.helpers import postgresql_subprocess_env

# Protocol era a harness client negotiates. 'auto' adopts the sessionless
# 2026-07-28 era (no initialize handshake) over stdio and streamable HTTP;
# 'legacy' forces the initialize handshake that handshake-era clients such as
# Claude Code use. The entry points run the full harness once per mode, and
# ERA_PROTOCOL_VERSIONS lists the protocol versions each mode may negotiate.
type ClientMode = Literal['auto', 'legacy']
CLIENT_MODES: tuple[ClientMode, ...] = ('auto', 'legacy')
ERA_PROTOCOL_VERSIONS: dict[ClientMode, tuple[str, ...]] = {
    'auto': MODERN_PROTOCOL_VERSIONS,
    'legacy': HANDSHAKE_PROTOCOL_VERSIONS,
}

# The result of a tool call made without raising: the extracted content and None,
# or None and the text of the tool error.
type ToolOutcome = tuple[dict[str, Any] | None, str | None]

# Embedding batch size of both primary servers and of every second server: a fixed
# size makes the long-document store check embed its twelve chunks in three
# batches of four on both backends, whatever the provider's speed.
HARNESS_EMBEDDING_BATCH_SIZE = '4'

# Feature settings of the PostgreSQL primary server and of every second server that
# does not share the primary's environment: the full search surface, low-effort
# summaries, the fixed embedding batch size, and the fp32 vector layout. The fp32
# layout exercises the pgvector <-> / HNSW / DISTINCT ON path, which the compressed
# tests leave uncovered, and keeps the run independent of the seed-locked
# compression_metadata singleton that the dedicated compression round-trip tests
# seal.
SERVER_FEATURES: dict[str, str] = {
    'ENABLE_SEMANTIC_SEARCH': 'true',
    'ENABLE_FTS': 'true',
    'ENABLE_HYBRID_SEARCH': 'true',
    'ENABLE_EMBEDDING_COMPRESSION': 'false',
    'EMBEDDING_BATCH_SIZE': HARNESS_EMBEDDING_BATCH_SIZE,
    'SUMMARY_OPENAI_REASONING_EFFORT': 'low',
    'SUMMARY_ANTHROPIC_EFFORT': 'low',
}


class HarnessCore:
    """Server, client and result plumbing shared by every check mixin."""

    def __init__(
        self,
        temp_db_path: Path | None = None,
        *,
        backend: str = 'sqlite',
        pg_url: str | None = None,
        pg_admin_url: str | None = None,
        rls: RlsCredentials | None = None,
        client_mode: ClientMode = 'auto',
    ) -> None:
        """Initialize the integration test suite.

        Args:
            temp_db_path: Optional path to a temporary SQLite database.
            backend: Storage backend to exercise ('sqlite' or 'postgresql').
            pg_url: The request connection string every PostgreSQL server is
                given; required when ``backend='postgresql'``.
            pg_admin_url: A superuser connection string to the same database,
                for the checks that count rows no tool shows; required when
                ``backend='postgresql'``.
            rls: The row-level security credentials every PostgreSQL server
                starts with; None starts them with ENABLE_POSTGRESQL_RLS off.
            client_mode: Protocol era every harness client negotiates
                (see :data:`ClientMode`).

        Raises:
            ValueError: If the PostgreSQL backend lacks either connection string.
        """
        if backend == 'postgresql' and (pg_url is None or pg_admin_url is None):
            raise ValueError('The PostgreSQL harness requires pg_url and pg_admin_url')
        self.client: Client[Any] | None = None
        self.test_results: list[tuple[str, bool, str]] = []
        self.test_thread_id = f'integration_test_{int(time.time())}'
        self.temp_db_path = temp_db_path
        self.backend = backend
        self.pg_url = pg_url
        self.pg_admin_url = pg_admin_url
        self.rls = rls
        self.client_mode: ClientMode = client_mode
        self.registered_tools: frozenset[str] = frozenset()
        # The environment the primary server was started with; a second server that
        # shares the primary's database starts from it (see _second_server).
        self.server_env: dict[str, str] | None = None

    def _postgresql_env(self, **overrides: str) -> dict[str, str]:
        """Return the environment of a server subprocess on the harness's PostgreSQL database.

        Args:
            overrides: Environment values applied last.

        Returns:
            The complete environment, carrying the run's row-level security credentials.
        """
        assert self.pg_url is not None  # __init__ requires it on PostgreSQL
        return postgresql_subprocess_env(self.pg_url, rls=self.rls, **overrides)

    def _new_client(self, transport: ClientTransport) -> Client[Any]:
        """Build a client that negotiates the harness's protocol era.

        Every harness connection, including the short-lived second servers some
        tests start, is built here, so one run exercises the tool surface in
        exactly the era ``client_mode`` selects.

        Args:
            transport: The transport to connect through.

        Returns:
            An unconnected client.
        """
        return Client(transport, mode=self.client_mode)

    async def start_server(self) -> bool:
        """Start the MCP server via FastMCP Client.

        Returns:
            bool: True (server starts automatically with Client).
        """
        print('[OK] Server will be started by FastMCP Client')
        return True

    async def connect_client(self) -> bool:
        """Connect FastMCP client to server.

        Returns:
            bool: True if client connected successfully.

        Raises:
            RuntimeError: If attempting to use default database in test mode.
        """
        try:
            # Use the wrapper script that sets up Python path correctly
            wrapper_script = Path(__file__).parents[2] / 'run_server.py'
            print(f'[INFO] Connecting to server via wrapper: {wrapper_script}')

            if self.backend == 'postgresql':
                # PostgreSQL path: the server auto-initializes its schema on
                # startup, so there is no SQLite pre-init step. The environment
                # MUST be passed explicitly via PythonStdioTransport(env=...)
                # because a bare script path applies the MCP SDK env whitelist
                # that strips app-specific vars. It inherits the parent's
                # provider configuration (EMBEDDING_MODEL / EMBEDDING_DIM in CI)
                # and clears DISABLED_TOOLS, so the full tool surface runs as on
                # SQLite, whose default stdio environment carries none.
                server_env = self._postgresql_env(**SERVER_FEATURES)
                self.server_env = server_env
                transport = PythonStdioTransport(
                    script_path=str(wrapper_script),
                    env=server_env,
                )
                self.client = self._new_client(transport)
                await self.client.__aenter__()
                self.registered_tools = frozenset(tool.name for tool in await self.client.list_tools())
                print(f'[OK] Client connected successfully (postgresql, {self.client_mode} mode)')
                return True

            # SQLite path: the environment is passed explicitly so the server
            # opens this test's own database. It starts from the MCP SDK's
            # default stdio environment, the same clean base a bare script path
            # gets, so nothing else from the developer's shell reaches the server.
            if self.temp_db_path is None:
                raise RuntimeError('The SQLite harness requires temp_db_path')
            default_db = Path.home() / '.mcp' / 'context_storage.db'
            if self.temp_db_path.resolve() == default_db.resolve():
                raise RuntimeError(
                    'CRITICAL: Attempting to use default database in test!\n'
                    f'Default: {default_db}\n'
                    f'Current: {self.temp_db_path}',
                )
            server_env = {
                **get_default_environment(),
                'STORAGE_BACKEND': 'sqlite',
                'DB_PATH': str(self.temp_db_path),
                'MCP_TEST_MODE': '1',
                'ENABLE_SEMANTIC_SEARCH': 'true',
                'ENABLE_FTS': 'true',
                'ENABLE_HYBRID_SEARCH': 'true',
                'EMBEDDING_BATCH_SIZE': HARNESS_EMBEDDING_BATCH_SIZE,
                'SUMMARY_OPENAI_REASONING_EFFORT': 'low',
                'SUMMARY_ANTHROPIC_EFFORT': 'low',
            }
            print(f'[INFO] Using temporary database: {self.temp_db_path}')
            self.server_env = server_env
            transport = PythonStdioTransport(script_path=str(wrapper_script), env=server_env)
            self.client = self._new_client(transport)

            # Connect to server
            await self.client.__aenter__()

            # Confirm the server answers a request in the negotiated era and record
            # the tools it registered; feature-gated checks follow this list
            self.registered_tools = frozenset(tool.name for tool in await self.client.list_tools())

            print(f'[OK] Client connected successfully ({self.client_mode} mode)')
            return True

        except Exception as e:
            print(f'[ERROR] Failed to connect client: {e}')
            import traceback

            traceback.print_exc()
            return False

    def _create_test_image(self) -> str:
        """Create a small test image as base64.

        Returns:
            str: Base64 encoded test image.
        """
        # Create a simple 1x1 pixel PNG image
        png_header = b'\x89PNG\r\n\x1a\n'
        ihdr = b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        idat = b'\x00\x00\x00\x0bIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05W\xbf\xaa\xd4'
        iend = b'\x00\x00\x00\x00IEND\xaeB`\x82'
        png_data = png_header + ihdr + idat + iend
        return base64.b64encode(png_data).decode('utf-8')

    def _extract_content(self, result: object) -> dict[str, Any]:
        """Extract content from FastMCP CallToolResult.

        Args:
            result: CallToolResult object from FastMCP.

        Returns:
            dict: The actual result content.
        """
        # FastMCP CallToolResult has structured_content attribute
        content = getattr(result, 'structured_content', None)
        if content is not None:
            if isinstance(content, dict):
                # Handle wrapped results
                if 'result' in content:
                    if isinstance(content['result'], list):
                        return {'success': True, 'results': content['result']}
                    if isinstance(content['result'], dict):
                        return content['result']
                # Special handling for search responses - return full content as-is
                # (search_context, semantic_search, fts_search, hybrid_search all return 'results' and 'count')
                if 'results' in content and 'count' in content:
                    # Add success=True if not present, preserve all other fields (error, stats, etc.)
                    if 'success' not in content:
                        return {'success': True, **content}
                    return content
                # Special handling for list_threads - it returns threads directly
                if 'threads' in content:
                    return {'success': True, 'threads': content['threads'], 'total_threads': content.get('total_threads', 0)}
                # Special handling for get_statistics - it returns stats directly
                if 'total_entries' in content:
                    return {'success': True, **content}  # Include all statistics fields
                # Direct dict results
                return content
            # List results
            if isinstance(content, list):
                return {'success': True, 'results': content}

        # Should not reach here with current FastMCP, but return error for safety
        return {'success': False, 'error': 'Unable to extract content from result'}

    async def _outcome(self, tool: str, arguments: dict[str, Any], *, client: Client[Any] | None = None) -> ToolOutcome:
        """Call a tool without raising on a tool error.

        Args:
            tool: The tool name.
            arguments: The tool arguments.
            client: The client to call through; the primary client when omitted.

        Returns:
            The extracted content and None, or None and the error text.
        """
        caller = client if client is not None else self.client
        assert caller is not None
        result = await caller.call_tool(tool, arguments, raise_on_error=False)
        if result.is_error:
            return None, ' '.join(block.text for block in result.content if isinstance(block, TextContent))
        return self._extract_content(result), None

    def _record(self, test_name: str, problems: list[str], passed: str) -> bool:
        """Record a check's outcome.

        Args:
            test_name: The check's name.
            problems: The problems found.
            passed: The detail recorded when there are none.

        Returns:
            bool: True when there are no problems.
        """
        self.test_results.append((test_name, not problems, '; '.join(problems) or passed))
        return not problems

    @contextlib.asynccontextmanager
    async def _second_server(
        self, extra_env: dict[str, str], *, share_primary_database: bool = False,
    ) -> AsyncGenerator[Client[Any]]:
        """Run a SECOND server subprocess carrying extra environment overrides.

        Some behavior is decided once, at server startup -- a feature toggle, a pool
        timeout, the indexed-metadata configuration, the default principal -- so it
        cannot be exercised against the long-lived server ``connect_client`` connected
        to. By default the overrides are applied on top of the harness defaults, and
        backend routing matches the harness's own: PostgreSQL reuses the run's
        database with the run's row-level security credentials, while SQLite gets an
        isolated temporary file so the second server never contends with the primary
        client's database.

        With ``share_primary_database`` the second server instead starts from the
        primary server's own environment, so it serves the primary's database on
        both backends (SQLite in WAL mode admits the second process) with the same
        embedding provider, model and dimension and the same compression setting --
        the startup guard refuses a compression setting that disagrees with the
        database -- and differs only by the overrides.

        ``PythonStdioTransport(env=...)`` passes the dict explicitly on both backends,
        because the MCP SDK env whitelist applied to a bare script path strips
        app-specific variables.

        Args:
            extra_env: Environment overrides applied last, so they win over every
                other value.
            share_primary_database: Start from the primary server's environment and
                serve its database.

        Yields:
            A connected client for the second server.

        Raises:
            RuntimeError: If sharing is requested before the primary server started.
        """
        wrapper_script = Path(__file__).parents[2] / 'run_server.py'
        second_db_path: Path | None = None
        if share_primary_database:
            if self.server_env is None:
                raise RuntimeError('The primary server must be started before a second server can share its database')
            server_env = {**self.server_env, **extra_env}
        elif self.backend == 'postgresql':
            server_env = self._postgresql_env(**{**SERVER_FEATURES, **extra_env})
        else:
            second_db_path = Path(tempfile.mkdtemp(prefix='mcp_harness_second_')) / 'second_server.db'
            server_env = {
                **os.environ,
                'MCP_TEST_MODE': '1',
                'DISABLED_TOOLS': '',
                **SERVER_FEATURES,
                'STORAGE_BACKEND': 'sqlite',
                'DB_PATH': str(second_db_path),
                **extra_env,
            }

        transport = PythonStdioTransport(script_path=str(wrapper_script), env=server_env)
        second_client = self._new_client(transport)
        try:
            await second_client.__aenter__()
            await second_client.list_tools()
            yield second_client
        finally:
            with contextlib.suppress(Exception):
                await second_client.__aexit__(None, None, None)
            if second_db_path is not None:
                for suffix in ('-wal', '-shm', ''):
                    candidate = AsyncPath(str(second_db_path) + suffix)
                    with contextlib.suppress(Exception):
                        if await candidate.exists():
                            await candidate.unlink()
                with contextlib.suppress(Exception):
                    await AsyncPath(second_db_path.parent).rmdir()

    async def cleanup(self) -> None:
        """Clean up server and resources."""
        try:
            # Disconnect client (this also stops the server subprocess)
            if self.client:
                await self.client.__aexit__(None, None, None)
                print('[OK] Client disconnected and server stopped')

            # Clean up temporary database file if it exists
            if self.temp_db_path:
                async_temp_db_path = AsyncPath(self.temp_db_path)
                if await async_temp_db_path.exists():
                    try:
                        # Remove WAL and SHM files if they exist
                        wal_file = AsyncPath(str(self.temp_db_path) + '-wal')
                        shm_file = AsyncPath(str(self.temp_db_path) + '-shm')
                        if await wal_file.exists():
                            await wal_file.unlink()
                        if await shm_file.exists():
                            await shm_file.unlink()
                        # Remove main database file
                        await async_temp_db_path.unlink()
                        print(f'[OK] Temporary database cleaned up: {self.temp_db_path}')
                    except Exception as cleanup_err:
                        print(f'[WARNING] Could not clean up temp database: {cleanup_err}')

        except Exception as e:
            print(f'[WARNING] Cleanup error: {e}')
