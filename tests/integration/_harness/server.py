"""Real-server checks for the server surface and startup wiring.

Protocol-era negotiation, the reported server version, tool annotations
delivered by ``list_tools``, the ``ENABLE_FTS=false`` force-off on a second
server, a stand-in Starlette ``/health`` route (the live route is checked in
``tests/integration/sqlite/test_http_transport.py``), and the PostgreSQL-only
session-pooler check staying inert on SQLite.
"""

from typing import Any

from tests.integration._harness.core import ERA_PROTOCOL_VERSIONS
from tests.integration._harness.core import HarnessCore


class ServerMixin(HarnessCore):
    """Checks for server identity, tool registration and startup wiring."""

    async def test_client_negotiates_requested_protocol_era(self) -> bool:
        """Verify the harness client runs in the protocol era ``client_mode`` selects.

        ``'auto'`` must adopt a sessionless (2026-07-28 or later) era with no
        initialize handshake, so the run reaches the tool surface the way current
        FastMCP clients do. ``'legacy'`` must complete the initialize handshake that
        handshake-era clients such as Claude Code use, and a ping -- routed only in
        that era -- must succeed.

        Returns:
            bool: True if the negotiated era matches the requested mode.
        """
        test_name = 'client_negotiates_requested_protocol_era'
        assert self.client is not None  # Type guard for Pyright
        try:
            protocol_version = self.client.protocol_version
            expected_versions = ERA_PROTOCOL_VERSIONS[self.client_mode]
            handshake_completed = self.client.initialize_result is not None
            if protocol_version not in expected_versions or handshake_completed != (self.client_mode == 'legacy'):
                self.test_results.append((
                    test_name,
                    False,
                    (
                        f'{self.client_mode} mode negotiated {protocol_version!r} '
                        f'(handshake completed: {handshake_completed}); expected one of {expected_versions}'
                    ),
                ))
                return False
            if self.client_mode == 'legacy' and not await self.client.ping():
                self.test_results.append((test_name, False, 'ping did not return an empty result'))
                return False

            self.test_results.append(
                (test_name, True, f'{self.client_mode} mode negotiated protocol {protocol_version}'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_server_version_is_project_version(self) -> bool:
        """Test that the negotiated server identity reports the project version.

        Verifies that the server reports its own version (from pyproject.toml)
        rather than the FastMCP framework version or MCP SDK version. The identity
        comes from the initialize handshake in the legacy era and from
        ``server/discover`` in the sessionless era.

        Returns:
            bool: True if server version matches project version.
        """
        test_name = 'server_version_is_project_version'
        assert self.client is not None  # Type guard for Pyright
        try:
            from app.server import SERVER_VERSION

            server_info = self.client.server_info
            assert server_info is not None, 'a connected client always carries the server identity'

            server_version = server_info.version
            if server_version != SERVER_VERSION:
                self.test_results.append((
                    test_name,
                    False,
                    f'Server reports version {server_version!r}, expected {SERVER_VERSION!r}',
                ))
                return False

            self.test_results.append(
                (test_name, True, f'Server version correctly reports {SERVER_VERSION}'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_tool_annotations_exposed_to_client(self) -> bool:
        """Verify tool behavior-hint annotations reach the client via list_tools.

        Clients rely on the readOnlyHint/destructiveHint/idempotentHint wire hints
        for auto-approval and destructive-action confirmation. This asserts the
        hints declared in TOOL_ANNOTATIONS are delivered over the wire (tools
        absent due to DISABLED_TOOLS are skipped, not failed).

        Returns:
            bool: True if test passed.
        """
        test_name = 'tool_annotations_exposed_to_client'
        assert self.client is not None
        try:
            tools = await self.client.list_tools()
            ann_by_name: dict[str, Any] = {t.name: t.annotations for t in tools}

            # (tool, hint attribute, expected value)
            expectations: list[tuple[str, str, bool]] = [
                ('search_context', 'read_only_hint', True),
                ('get_context_by_ids', 'read_only_hint', True),
                ('list_threads', 'read_only_hint', True),
                ('get_statistics', 'read_only_hint', True),
                ('grep_context', 'read_only_hint', True),
                ('navigate_context', 'read_only_hint', True),
                ('read_context_range', 'read_only_hint', True),
                ('plan_context_sync', 'read_only_hint', True),
                ('list_ingest_state', 'read_only_hint', True),
                ('list_context_grants', 'read_only_hint', True),
                ('share_context', 'destructive_hint', False),
                ('share_context', 'idempotent_hint', True),
                ('unshare_context', 'destructive_hint', True),
                ('unshare_context', 'idempotent_hint', True),
                ('store_context', 'read_only_hint', False),
                ('store_context', 'destructive_hint', False),
                ('update_context', 'destructive_hint', True),
                ('update_context', 'idempotent_hint', False),
                ('delete_context', 'destructive_hint', True),
                ('delete_context', 'idempotent_hint', True),
            ]
            checked = 0
            for tool_name, attr, expected in expectations:
                annotations = ann_by_name.get(tool_name)
                if annotations is None:
                    # Tool disabled (DISABLED_TOOLS) or no annotations exposed; skip.
                    continue
                actual = getattr(annotations, attr, None)
                if actual != expected:
                    self.test_results.append((test_name, False,
                        f'{tool_name}.{attr} = {actual!r}, expected {expected!r}'))
                    return False
                checked += 1

            if checked == 0:
                self.test_results.append((test_name, False, 'No tool annotations were exposed to assert against'))
                return False

            self.test_results.append((test_name, True, f'Tool annotations delivered to client ({checked} hints verified)'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_tool_descriptions_describe_the_response(self) -> bool:
        """Verify every served tool description carries its return and error sections.

        A client learns what a tool answers, and how it fails, from the description
        list_tools serves. Each tool's docstring ends with a ``Returns:`` and a
        ``Raises:`` section, and the full-text search tool's generated description
        carries the same two, so both headings must reach the client for every tool,
        in that order, each followed by text.

        Returns:
            bool: True if test passed.
        """
        test_name = 'tool_descriptions_describe_the_response'
        assert self.client is not None
        try:
            tools = await self.client.list_tools()
            problems: list[str] = []
            for tool in tools:
                description = tool.description or ''
                returns, raises = description.find('Returns:'), description.find('Raises:')
                if returns < 0 or raises < 0:
                    missing = ' and '.join(name for name, at in (('Returns:', returns), ('Raises:', raises)) if at < 0)
                    problems.append(f'{tool.name} serves no {missing} section')
                elif not (returns < raises and description[returns + 8:raises].strip() and description[raises + 7:].strip()):
                    problems.append(f'{tool.name} serves an empty or misplaced Returns: or Raises: section')
            if not tools:
                problems.append('the server lists no tools')
            return self._record(test_name, problems, f'{len(tools)} tool descriptions carry their return and error sections')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_force_off_removes_search_tool(self) -> bool:
        """Verify ENABLE_FTS=false force-off removes fts_search_context end to end.

        Opens a SECOND server subprocess (``_second_server``) with ENABLE_FTS
        forced to 'false' while ENABLE_SEMANTIC_SEARCH and ENABLE_HYBRID_SEARCH
        stay enabled. The tri-state toggle should drop fts_search_context from the
        registered tool surface while leaving semantic_search_context and
        hybrid_search_context registered. This proves the force-off path on BOTH
        SQLite and PostgreSQL.

        When no embedding provider is available (no Ollama model in the
        environment), semantic_search_context cannot register and
        hybrid_search_context has no remaining search mode (FTS is forced off),
        so the presence assertions are skipped gracefully; the FTS absence
        assertion -- the core force-off behavior -- always runs.

        Returns:
            bool: True if test passed (or skipped gracefully).
        """
        test_name = 'force_off_removes_search_tool'
        assert self.client is not None
        try:
            async with self._second_server({'ENABLE_FTS': 'false'}) as second_client:
                # Determine whether an embedding provider is available; if not,
                # semantic_search_context cannot register and (with FTS forced off)
                # hybrid_search_context has no search mode, so skip those presence
                # assertions while still asserting FTS absence.
                stats_data = self._extract_content(await second_client.call_tool('get_statistics', {}))
                semantic_info = stats_data.get('semantic_search', {})
                semantic_available = bool(semantic_info.get('available', False))
                tool_names = {t.name for t in await second_client.list_tools()}

            if 'fts_search_context' in tool_names:
                self.test_results.append(
                    (test_name, False, 'fts_search_context still registered despite ENABLE_FTS=false'),
                )
                return False

            if not semantic_available:
                self.test_results.append(
                    (test_name, True, 'FTS force-off verified; semantic/hybrid presence skipped (no embedding provider)'),
                )
                return True

            if 'semantic_search_context' not in tool_names:
                self.test_results.append(
                    (test_name, False, 'semantic_search_context missing while ENABLE_SEMANTIC_SEARCH stayed enabled'),
                )
                return False

            if 'hybrid_search_context' not in tool_names:
                self.test_results.append(
                    (test_name, False, 'hybrid_search_context missing while ENABLE_HYBRID_SEARCH stayed enabled'),
                )
                return False

            self.test_results.append(
                (test_name, True, 'ENABLE_FTS=false removed only fts_search_context; semantic and hybrid remain'),
            )
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_health_endpoint_returns_ok(self) -> bool:
        """Verify a stand-in Starlette /health route returns HTTP 200 with {"status": "ok"}.

        The health endpoint is only available on HTTP transport. Since the harness uses
        stdio transport, this check builds its own Starlette app with a local /health route
        and calls it through TestClient; the live route registered by ``main()`` is checked
        by tests/integration/sqlite/test_http_transport.py.

        Returns:
            bool: True if test passed.
        """
        test_name = 'health_endpoint_returns_ok'
        assert self.client is not None
        try:
            from starlette.applications import Starlette
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            from starlette.routing import Route
            from starlette.testclient import TestClient

            async def health_handler(_: Request) -> JSONResponse:
                return JSONResponse({'status': 'ok'})

            app = Starlette(routes=[Route('/health', health_handler, methods=['GET'])])

            with TestClient(app) as test_client:
                response = test_client.get('/health')

                if response.status_code != 200:
                    self.test_results.append((test_name, False,
                        f'Health endpoint status code: {response.status_code}'))
                    return False

                body = response.json()
                if body.get('status') != 'ok':
                    self.test_results.append((test_name, False,
                        f'Health endpoint body: {body}'))
                    return False

            self.test_results.append((test_name, True,
                'Health endpoint returns 200 with {"status": "ok"}'))
            return True

        except ImportError as e:
            self.test_results.append((test_name, True,
                f'Skipped (missing dependency: {e})'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_session_pooler_validation_noop_on_sqlite(self) -> bool:
        """Verify the session-pooler advisory wiring is a PostgreSQL-only no-op on SQLite.

        ``PostgreSQLBackend._detect_session_mode_pooler()`` (which calls
        ``is_supabase_session_pooler`` from ``app/startup/validation.py``) runs only inside
        the PostgreSQL backend's ``initialize()``; a healthy SQLite server boot proves the
        advisory neither runs nor crashes on the default backend.

        Returns:
            bool: True if test passed.
        """
        test_name = 'session_pooler_validation_noop_on_sqlite'
        assert self.client is not None
        if self.backend != 'sqlite':
            self.test_results.append(
                (test_name, True, 'Skipped on postgresql (SQLite-only no-op assertion)'),
            )
            return True
        try:
            data = self._extract_content(await self.client.call_tool('list_threads', {}))
            if 'threads' not in data:
                self.test_results.append((test_name, False, f'Server not operational: {data}'))
                return False
            self.test_results.append((test_name, True,
                'SQLite server operational; session-pooler advisory is a PG-only no-op'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Unexpected failure: {e}'))
            return False
