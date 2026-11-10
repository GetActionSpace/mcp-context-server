"""Real-server checks for retrieving entries by id.

``get_context_by_ids`` with existing and missing ids, the ``summary`` field
omitted by default and returned under ``GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true``,
and resolution of a short id prefix to the canonical 32-character id.
"""

import contextlib
import sqlite3
import tempfile
from pathlib import Path

from anyio import Path as AsyncPath
from mcp.client.stdio import get_default_environment

from tests.integration._harness.core import HarnessCore
from tests.integration._harness.core import server_transport


class RetrieveMixin(HarnessCore):
    """Checks for retrieving context entries by id."""

    async def test_get_context_by_ids(self) -> bool:
        """Test retrieving specific contexts by IDs.

        Returns:
            bool: True if test passed.
        """
        test_name = 'get_context_by_ids'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Store test data
            result1 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'agent',  # Must be 'user' or 'agent'
                    'text': 'First context for retrieval',
                },
            )

            result2 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Second context with image',
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                },
            )

            data1 = self._extract_content(result1)
            data2 = self._extract_content(result2)

            if not (data1.get('success') and data2.get('success')):
                self.test_results.append((test_name, False, f'Failed to store test contexts: {data1}, {data2}'))
                return False

            context_ids = [data1['context_id'], data2['context_id']]

            # Test retrieval without images
            without_images = await self.client.call_tool(
                'get_context_by_ids',
                {
                    'context_ids': context_ids,
                    'include_images': False,
                },
            )

            without_data = self._extract_content(without_images)

            # get_context_by_ids returns success with results
            if not without_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to retrieve without images: {without_data}'))
                return False

            # Test retrieval with images
            with_images = await self.client.call_tool(
                'get_context_by_ids',
                {
                    'context_ids': context_ids,
                    'include_images': True,
                },
            )

            with_data = self._extract_content(with_images)

            # get_context_by_ids returns success with results
            if not with_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to retrieve with images: {with_data}'))
                return False

            # Verify both retrievals got the correct number of results
            if len(without_data.get('results', [])) == 2 and len(with_data.get('results', [])) == 2:
                self.test_results.append((test_name, True, f'Retrieved {len(context_ids)} contexts'))
                return True
            self.test_results.append((test_name, False, 'Incorrect number of results'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_get_context_by_ids_partial_match(self) -> bool:
        """Test getting mix of existing and non-existing IDs.

        Returns:
            bool: True if test passed.
        """
        test_name = 'Get Context By IDs Partial Match'
        assert self.client is not None
        try:
            # First store a context to get a valid ID
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_partial',
                    'source': 'agent',
                    'text': 'Context for partial match test',
                },
            )

            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store context: {store_data}'))
                return False

            valid_id = store_data.get('context_id')
            if not valid_id:
                self.test_results.append((test_name, False, 'No context_id returned'))
                return False

            # Get by IDs including valid and invalid
            result = await self.client.call_tool(
                'get_context_by_ids',
                {
                    'context_ids': [valid_id, '0' * 32, 'f' * 32],  # One valid, two never-issued UUIDv7 hex
                },
            )

            data = self._extract_content(result)

            # Should return only the valid entry (1 result)
            results = data.get('results', data)
            if isinstance(results, list):
                # Should have exactly 1 result (the valid ID)
                if len(results) == 1:
                    self.test_results.append((test_name, True, 'Partial match returned only valid entries'))
                    return True
                self.test_results.append((test_name, False, f'Expected 1 result, got {len(results)}'))
                return False

            self.test_results.append((test_name, False, f'Unexpected result format: {data}'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_get_context_by_ids_omits_summary_by_default(self) -> bool:
        """Test that get_context_by_ids omits the summary field by default.

        Default configuration (GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY unset) means
        the get_context_by_ids tool MUST NOT return a `summary` key, because
        the tool already returns the full untruncated text_content.

        Returns:
            bool: True if test passed.
        """
        test_name = 'get_context_by_ids_omits_summary_by_default'
        assert self.client is not None  # Type guard for Pyright
        try:
            summary_thread = f'{self.test_thread_id}_summary_field'

            # Store a context entry (no summary provider configured in test server)
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': summary_thread,
                    'source': 'agent',
                    'text': 'Test entry to verify summary field exists in API response',
                    'metadata': {'test_type': 'summary_field'},
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Failed to store context: {store_data}'),
                )
                return False

            context_id = store_data['context_id']

            # Retrieve via get_context_by_ids - should OMIT summary field by default
            get_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])
            if len(results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result, got {len(results)}'),
                )
                return False

            entry = results[0]

            # Verify summary field is OMITTED in the default configuration
            if 'summary' in entry:
                self.test_results.append(
                    (test_name, False,
                     ('summary field unexpectedly present in get_context_by_ids response '
                      '(default config should omit it)')),
                )
                return False

            # Verify the full text is returned (not truncated) in get_context_by_ids
            if entry['text_content'] != 'Test entry to verify summary field exists in API response':
                self.test_results.append(
                    (test_name, False,
                     f'text_content mismatch: {entry["text_content"]!r}'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'summary field correctly omitted from get_context_by_ids response by default'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_get_context_by_ids_includes_summary_when_enabled(self) -> bool:
        """Verify GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true causes end-to-end empty-string normalization.

        Spawns a short-lived secondary server subprocess with the env var set
        and observes the wire payload. The third leg of the tri-state contract
        (verbatim pass-through with a real stored summary) is exercised by the
        in-process unit test
        tests/server/test_server_tools.py::TestGetContextByIds
        ::test_get_context_by_ids_summary_passes_through_when_stored.

        Subprocess environment plumbing:
            The MCP SDK helper mcp.client.stdio.get_default_environment() applies
            an OS-variable whitelist (PATH, SYSTEMROOT, ..., on Windows; HOME,
            LOGNAME, ..., on POSIX) when env=None is passed to the transport.
            Constructing a client from a bare script path delegates to
            this whitelist, so application-specific env vars (DB_PATH, MCP_TEST_MODE,
            GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY) DO NOT reach the subprocess. This
            test builds the env dict explicitly and hands it to ``server_transport``,
            on SQLite from the same whitelist plus the four app-specific vars and on
            PostgreSQL from the harness's PostgreSQL server environment, ensuring
            the subprocess actually runs with the toggle enabled.

        Returns:
            bool: True if test passed.
        """
        test_name = 'get_context_by_ids_includes_summary_when_enabled'

        tmp_dir = Path(tempfile.mkdtemp(prefix='mcp_summary_opt_in_'))
        tmp_db = tmp_dir / 'summary_opt_in.db'

        subprocess_env: dict[str, str]
        if self.backend == 'postgresql':
            # PG server auto-initializes its schema; no SQLite pre-init. It serves
            # the run's database with the run's row-level security credentials and
            # the summary-inclusion toggle on. Neither generation subsystem is
            # needed, and disabling both keeps the NULL-summary -> '' wire
            # normalization deterministic whatever SUMMARY_MIN_CONTENT_LENGTH or
            # provider configuration the parent environment carries.
            subprocess_env = self._postgresql_env(
                GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY='true',
                ENABLE_SUMMARY_GENERATION='false',
                ENABLE_EMBEDDING_GENERATION='false',
            )
        else:
            # Initialize the schema before the secondary server opens the file
            from app.schemas import load_schema
            schema_sql = load_schema('sqlite')
            with sqlite3.connect(str(tmp_db)) as init_conn:
                init_conn.executescript(schema_sql)
                init_conn.commit()

            # The MCP SDK's default stdio environment lets the subprocess locate
            # Python, system DLLs, and temp dirs while keeping the developer's
            # shell out of the run.
            subprocess_env = get_default_environment()
            # Application-specific overrides REQUIRED by the secondary server.
            subprocess_env['DB_PATH'] = str(tmp_db)
            subprocess_env['MCP_TEST_MODE'] = '1'
            subprocess_env['STORAGE_BACKEND'] = 'sqlite'
            subprocess_env['GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY'] = 'true'

        secondary_client = self._new_client(server_transport(subprocess_env))
        try:
            await secondary_client.__aenter__()
            await secondary_client.list_tools()

            store_result = await secondary_client.call_tool(
                'store_context',
                {
                    'thread_id': 'summary_opt_in_thread',
                    'source': 'agent',
                    'text': 'Opt-in summary inclusion test',
                    'metadata': {'test_type': 'summary_opt_in'},
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store: {store_data}'))
                return False
            context_id = store_data['context_id']

            get_result = await secondary_client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])
            if len(results) != 1:
                self.test_results.append((test_name, False, f'Expected 1 result, got {len(results)}'))
                return False

            entry = results[0]
            # Sanity: core fields preserved
            if entry.get('id') != context_id:
                self.test_results.append(
                    (test_name, False, f'Expected id={context_id}, got {entry.get("id")!r}'),
                )
                return False
            if entry.get('text_content') != 'Opt-in summary inclusion test':
                self.test_results.append(
                    (test_name, False, f'text_content mismatch: {entry.get("text_content")!r}'),
                )
                return False

            # Tri-state assertion (second leg): toggle=true + no provider -> '' on the wire.
            if 'summary' not in entry:
                self.test_results.append(
                    (test_name, False,
                     'summary key MUST be present on the wire when GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true'),
                )
                return False
            if entry['summary'] != '':
                self.test_results.append(
                    (test_name, False,
                     f'NULL DB summary must normalize to empty string end-to-end, got {entry["summary"]!r}'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 ('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true: summary key present and == "" on the wire '
                  '(second leg of tri-state contract verified end-to-end).')),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
        finally:
            with contextlib.suppress(Exception):
                await secondary_client.__aexit__(None, None, None)
            # Best-effort cleanup (use anyio.Path for async-safe filesystem ops).
            # The secondary server, stopped above, receives its env through its transport, so
            # os.environ is never modified and needs no restoration.
            async_tmp_db = AsyncPath(tmp_db)
            async_tmp_dir = AsyncPath(tmp_dir)
            with contextlib.suppress(Exception):
                await async_tmp_db.unlink(missing_ok=True)
                await async_tmp_dir.rmdir()

    async def test_prefix_id_resolution_returns_canonical_id(self) -> bool:
        """A short id prefix resolves to the canonical 32-char hex context_id on both backends.

        On PostgreSQL the id column is a native UUID, whose text form is 36-char
        hyphenated; find_ids_by_prefix normalizes every match, so navigate_context
        echoes response['context_id'] from the resolved prefix as the canonical
        32-char lowercase hex on both backends, never a hyphenated/pgproto form.

        Returns:
            bool: True if test passed.
        """
        test_name = 'prefix_id_resolution_returns_canonical_id'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_prefix'
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': '# Prefix\nbody for prefix id resolution\n',
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, f'store failed: {store}'))
                return False
            cid = store['context_id']
            if len(cid) != 32:
                self.test_results.append((test_name, False, f'stored id not 32-char hex: {cid!r}'))
                return False

            prefix = cid[:12]
            nav = self._extract_content(await self.client.call_tool('navigate_context', {'context_id': prefix}))
            echoed = nav.get('context_id')
            if echoed != cid:
                self.test_results.append((
                    test_name, False,
                    f'prefix-resolved context_id not canonical: got {echoed!r} (len {len(str(echoed))}), want {cid!r}',
                ))
                return False

            self.test_results.append((test_name, True, f'prefix resolves to canonical 32-char id on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
