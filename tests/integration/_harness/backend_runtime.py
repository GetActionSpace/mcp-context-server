"""Real-server checks for storage backend runtime behavior.

NUL-bearing input rejected without charging the circuit breaker, the
shared connection-metrics contract of ``get_statistics``, the SQLite write
queue across idle windows and bursts, transactional writes counted in
``total_queries``, and the idle SQLite writer recycled without a failed
call.
"""

import asyncio
from itertools import pairwise
from typing import Any

from fastmcp import Client

from tests.integration._harness.core import HarnessCore


class BackendRuntimeMixin(HarnessCore):
    """Checks for the circuit breaker, connection metrics, write queue and writer lifecycle."""

    async def test_nul_input_does_not_trip_breaker(self) -> bool:
        """A burst of NUL-bearing client input is rejected without opening the circuit breaker.

        A NUL (U+0000) in a string parameter aborts a PostgreSQL bind with a
        non-ControlFlowError, which the circuit breaker charges, so ten such calls reaching
        the database would open it into a process-wide outage, while SQLite would
        silently store the NUL (a cross-backend divergence). The boundary guards reject
        the input as a clean client error on BOTH backends before any bind and leave the
        breaker closed, so healthy traffic still succeeds afterward. On PostgreSQL this
        proves the breaker stays closed; on SQLite it proves the parity rejection (the
        NUL is never stored).

        Returns:
            bool: True if test passed.
        """
        test_name = 'nul_input_breaker_safe'
        assert self.client is not None  # Type guard for Pyright
        nul = '\x00'
        # Four always-registered NUL vectors that would each reach a string bind without
        # the guards: store text, store thread_id, search thread_id, grep thread_id.
        bad_calls: list[tuple[str, dict[str, Any]]] = [
            ('store_context', {'thread_id': 'nul-burst', 'source': 'agent', 'text': f'x{nul}y'}),
            ('store_context', {'thread_id': f'thr{nul}', 'source': 'agent', 'text': 'x'}),
            ('search_context', {'thread_id': f'thr{nul}', 'limit': 5}),
            ('grep_context', {'pattern': 'needle', 'thread_id': f'thr{nul}'}),
        ]
        try:
            # 4 vectors x 4 rounds = 16 failing calls, well past the default breaker
            # threshold of 10.
            for _ in range(4):
                for tool_name, args in bad_calls:
                    try:
                        result = await self.client.call_tool(tool_name, args)
                        data = self._extract_content(result)
                        # A raised ToolError is the expected rejection; a returned dict must
                        # NOT report success (which would mean the NUL was accepted/stored).
                        if data.get('success') is True:
                            self.test_results.append(
                                (test_name, False, f'{tool_name} accepted NUL input: {data}'),
                            )
                            return False
                    except Exception:
                        # A client-side ToolError is the expected rejection outcome.
                        continue
            # The breaker must still be CLOSED: a healthy store must succeed.
            healthy = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_nul_healthy',
                    'source': 'agent',
                    'text': 'healthy store after the NUL burst',
                },
            )
            healthy_data = self._extract_content(healthy)
            if not healthy_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Healthy store failed after NUL burst (breaker open?): {healthy_data}'),
                )
                return False

            self.test_results.append((test_name, True, 'NUL input rejected on both backends; breaker stayed closed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_connection_metrics_cross_backend_contract(self) -> bool:
        """get_statistics publishes the shared connection-metrics contract on BOTH backends.

        ``backend_type`` and ``pool_size`` are the two keys the SQLite and PostgreSQL
        ``get_metrics()`` implementations both guarantee, so a monitoring client can
        identify the backend and read one pool bound without branching on
        backend-specific keys (SQLite's pool bound is the reader-pool size, since writes
        serialize onto a single writer). Nothing else pins that shared shape end to end.

        The three operator-facing failure fields are asserted to move TOGETHER: a nonzero
        ``failed_queries`` REQUIRES both a ``last_error`` message and a
        ``last_error_time``, and a zero count requires neither to be set. A charge that
        leaves the diagnostic fields clean reports a healthy, error-free database while
        the breaker counts an outage; a counter with no message leaves an operator a
        number and no diagnosis, or an OLD message beside fresh failures.

        Returns:
            bool: True if test passed.
        """
        test_name = 'connection_metrics_cross_backend_contract'
        assert self.client is not None
        try:
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': f'{self.test_thread_id}_conn_metrics',
                'source': 'agent',
                'text': 'Entry stored before reading the connection-metrics contract',
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {store}'))
                return False

            stats = self._extract_content(await self.client.call_tool('get_statistics', {}))
            metrics = stats.get('connection_metrics')
            if not isinstance(metrics, dict):
                self.test_results.append((
                    test_name, False, f'connection_metrics missing or not a mapping: {metrics!r}',
                ))
                return False

            backend_type = metrics.get('backend_type')
            if backend_type != self.backend:
                self.test_results.append((
                    test_name, False,
                    f'connection_metrics.backend_type is {backend_type!r}, expected {self.backend!r}',
                ))
                return False

            pool_size = metrics.get('pool_size')
            if not isinstance(pool_size, int) or isinstance(pool_size, bool):
                self.test_results.append((
                    test_name, False, f'connection_metrics.pool_size must be an int, got {pool_size!r}',
                ))
                return False

            failed_queries = metrics.get('failed_queries')
            if not isinstance(failed_queries, int) or isinstance(failed_queries, bool):
                self.test_results.append((
                    test_name, False, f'connection_metrics.failed_queries must be an int, got {failed_queries!r}',
                ))
                return False

            last_error = metrics.get('last_error')
            last_error_time = metrics.get('last_error_time')
            if failed_queries > 0:
                if not isinstance(last_error, str) or not last_error:
                    self.test_results.append((
                        test_name, False,
                        f'failed_queries={failed_queries} but last_error is {last_error!r} (charge without diagnosis)',
                    ))
                    return False
                if not isinstance(last_error_time, (int, float)) or isinstance(last_error_time, bool):
                    self.test_results.append((
                        test_name, False,
                        f'failed_queries={failed_queries} but last_error_time is {last_error_time!r}',
                    ))
                    return False
            elif last_error is not None or last_error_time is not None:
                self.test_results.append((
                    test_name, False,
                    f'failed_queries=0 but last_error={last_error!r}/last_error_time={last_error_time!r}',
                ))
                return False

            self.test_results.append((
                test_name, True,
                (
                    f'connection_metrics: backend_type={backend_type}, pool_size={pool_size}, '
                    f'failed_queries={failed_queries}'
                ),
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_write_queue_survives_idle_windows_and_bursts(self) -> bool:
        """Writes interleaved with reads and idle windows all complete; none is dropped.

        SQLite serializes every write through one processor task whose queue getter
        OUTLIVES an idle timeout. With a getter recreated per loop iteration, a request
        landing in the same scheduling window as a timeout would be dequeued by a getter
        the next iteration never inspects: the write would be silently dropped and
        its caller would await a future nobody resolves, so the tool call would NEVER
        return. This drives the queue over the real transport with writes arriving right
        at the idle boundary (covering both the 1.0 s default and the 0.1 s test-mode
        timeout) plus a gapless burst, and asserts every call returns and every entry is
        retrievable afterwards. Each call carries an explicit timeout so a dropped write
        fails the test instead of hanging the suite. PostgreSQL has no write queue; there
        the same traffic pattern is a plain durability check.

        Returns:
            bool: True if test passed.
        """
        test_name = 'write_queue_survives_idle_windows_and_bursts'
        assert self.client is not None
        thread = f'{self.test_thread_id}_queue_idle'
        # Idle windows straddling BOTH configured queue timeouts, so a write always
        # arrives close to a timeout boundary.
        idle_windows = (1.05, 0.1, 1.0)
        call_timeout_s = 180.0
        stored_ids: list[str] = []
        try:
            for index, idle in enumerate(idle_windows):
                await asyncio.sleep(idle)
                store = self._extract_content(await asyncio.wait_for(
                    self.client.call_tool('store_context', {
                        'thread_id': thread,
                        'source': 'agent',
                        'text': f'write-queue entry after an idle window {index}',
                    }),
                    timeout=call_timeout_s,
                ))
                if not store.get('success'):
                    self.test_results.append((test_name, False, f'Store {index} after idle window failed: {store}'))
                    return False
                stored_ids.append(str(store['context_id']))

                # A read between writes exercises the reader pool while the queue is idle.
                read = self._extract_content(await asyncio.wait_for(
                    self.client.call_tool('search_context', {'thread_id': thread, 'limit': 50}),
                    timeout=call_timeout_s,
                ))
                if len(read.get('results', [])) != len(stored_ids):
                    self.test_results.append((
                        test_name, False,
                        f'After {len(stored_ids)} writes the thread holds {len(read.get("results", []))} entries',
                    ))
                    return False

            # Gapless burst: consecutive writes with no idle window between them.
            for index in range(2):
                store = self._extract_content(await asyncio.wait_for(
                    self.client.call_tool('store_context', {
                        'thread_id': thread,
                        'source': 'agent',
                        'text': f'write-queue burst entry {index}',
                    }),
                    timeout=call_timeout_s,
                ))
                if not store.get('success'):
                    self.test_results.append((test_name, False, f'Burst store {index} failed: {store}'))
                    return False
                stored_ids.append(str(store['context_id']))

            got = self._extract_content(await asyncio.wait_for(
                self.client.call_tool('get_context_by_ids', {'context_ids': stored_ids}),
                timeout=call_timeout_s,
            ))
            returned = {str(row.get('id')) for row in got.get('results', [])}
            if returned != set(stored_ids):
                self.test_results.append((
                    test_name, False,
                    f'Retrieved {sorted(returned)} but stored {sorted(stored_ids)}',
                ))
                return False

            self.test_results.append((
                test_name, True, f'All {len(stored_ids)} interleaved writes completed and are retrievable',
            ))
            return True
        except TimeoutError:
            self.test_results.append((
                test_name, False, 'A tool call never returned within the timeout (dropped write?)',
            ))
            return False
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_transactional_write_moves_total_queries(self) -> bool:
        """A committed transactional write is counted in ``connection_metrics.total_queries``.

        ``total_queries`` and ``failed_queries`` are published side by side, so an
        operator (or an alert rule) reads them as ONE population and computes a failure
        rate from the pair. The store, update and delete write paths run entirely inside
        ``begin_transaction``, whose failure arm charges ``failed_queries``. A success arm
        that counted nothing would leave ``total_queries`` unmoved by a run of deletes, so
        every transactional write would be missing from the denominator; the commit
        therefore counts one completed operation on both backends.

        The probe is three deletes by ids between two statistics reads. A delete by ids
        performs exactly ONE counted operation (its transaction) and no other counted
        read, so three of them move the counter by three more than an idle interval does.
        ``get_statistics`` itself issues a fixed number of counted reads per call, which
        back-to-back calls measure. The generation worker's poll is a counted operation
        too wherever the worker shares the request backend (SQLite, and PostgreSQL
        without row-level security), and it lands in whichever interval is open when it
        fires. The idle cost is therefore the smallest of three idle intervals, and the
        assertion asks for all three deletes, which one poll cannot supply.

        Returns:
            bool: True if test passed.
        """
        test_name = 'transactional_write_moves_total_queries'
        assert self.client is not None
        thread = f'{self.test_thread_id}_query_counter'
        deletes = 3

        async def _total_queries() -> int | None:
            """Read connection_metrics.total_queries, or None when it is unreadable."""
            assert self.client is not None
            stats = self._extract_content(await self.client.call_tool('get_statistics', {}))
            metrics = stats.get('connection_metrics')
            if not isinstance(metrics, dict):
                return None
            value = metrics.get('total_queries')
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            return value

        try:
            entry_ids: list[str] = []
            for number in range(1, deletes + 1):
                stored = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'Entry {number} deleted while measuring the completed-operation counter',
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Store failed: {stored}'))
                    return False
                entry_ids.append(str(stored['context_id']))

            # Warm whatever a first statistics call initializes, so the idle intervals
            # below measure a steady-state call rather than a one-off.
            await self.client.call_tool('get_statistics', {})
            reads = [await _total_queries() for _ in range(4)]
            idle_reads = [value for value in reads if value is not None]
            if len(idle_reads) != len(reads):
                self.test_results.append((test_name, False, 'connection_metrics.total_queries is not an integer'))
                return False
            idle = min(later - earlier for earlier, later in pairwise(idle_reads))

            for entry_id in entry_ids:
                deleted = self._extract_content(await self.client.call_tool('delete_context', {
                    'context_ids': [entry_id],
                }))
                if not deleted.get('success') or deleted.get('deleted_count') != 1:
                    self.test_results.append((test_name, False, f'Delete failed: {deleted}'))
                    return False
            after_delete = await _total_queries()
            if after_delete is None:
                self.test_results.append((test_name, False, 'connection_metrics.total_queries is not an integer'))
                return False
            delete_delta = after_delete - idle_reads[-1]
            if delete_delta - idle < deletes:
                self.test_results.append((
                    test_name, False,
                    (
                        f'{deletes} committed delete transactions moved total_queries by '
                        f'{delete_delta - idle} beyond the {idle}-query idle interval'
                    ),
                ))
                return False

            stored_again = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Entry stored while measuring the completed-operation counter',
            }))
            if not stored_again.get('success'):
                self.test_results.append((test_name, False, f'Second store failed: {stored_again}'))
                return False
            after_store = await _total_queries()
            if after_store is None:
                self.test_results.append((test_name, False, 'connection_metrics.total_queries is not an integer'))
                return False
            store_delta = after_store - after_delete
            if store_delta <= idle:
                self.test_results.append((
                    test_name, False,
                    (
                        f'A committed store moved total_queries by {store_delta - idle} '
                        f'beyond the {idle}-query idle interval'
                    ),
                ))
                return False

            self.test_results.append((
                test_name, True,
                (
                    f'{deletes} deletes and a store moved total_queries by {delete_delta} and {store_delta} '
                    f'against a {idle}-query idle interval'
                ),
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_idle_writer_recycle_is_invisible_to_callers(self) -> bool:
        """Recycling the idle SQLite writer neither fails a call nor charges the breaker.

        SQLite keeps ONE writer connection for the whole process, and the health-check
        loop closes it once it has been idle past ``POOL_IDLE_TIMEOUT_S`` so an idle
        server stops pinning the database file and its -wal/-shm siblings; the next write
        recreates it lazily. Closing a connection a caller might still hold is the risk
        that behavior carries, and it would surface as a failed call, a charged
        ``failed_queries``, or a circuit state other than HEALTHY. A second server with a
        one-second idle timeout and health-check interval makes the recycle happen inside
        a test-sized window: write, idle past the recycle, write again, burst once more,
        then assert every call succeeded, every entry is retrievable, and the failure
        counters never moved.

        Both settings are SQLite-only; on PostgreSQL, whose pool manages its own
        connections, the same traffic is a plain idle-durability check.

        Returns:
            bool: True if test passed.
        """
        test_name = 'idle_writer_recycle_is_invisible_to_callers'
        thread = f'{self.test_thread_id}_writer_recycle'
        # The recycle needs BOTH a one-second idle writer and a health-check tick to
        # observe it. The pool clamps its health-check cadence to five seconds whenever
        # it detects a test environment, whatever the interval setting asks for, so the
        # idle window below is sized to contain one tick with margin rather than to the
        # configured interval.
        recycle_env = {'POOL_IDLE_TIMEOUT_S': '1', 'POOL_HEALTH_CHECK_INTERVAL_S': '1'}
        idle_window_s = 7.0
        try:
            async with self._second_server(recycle_env) as client:

                async def _failure_metrics(active_client: Client[Any]) -> tuple[int, str] | None:
                    """Read (failed_queries, circuit_state), or None when unreadable."""
                    stats = self._extract_content(await active_client.call_tool('get_statistics', {}))
                    metrics = stats.get('connection_metrics')
                    if not isinstance(metrics, dict):
                        return None
                    failed = metrics.get('failed_queries')
                    if not isinstance(failed, int) or isinstance(failed, bool):
                        return None
                    return failed, str(metrics.get('circuit_state'))

                stored_ids: list[str] = []
                before: tuple[int, str] | None = None
                for index, idle in enumerate((0.0, idle_window_s, 0.0)):
                    if idle:
                        await asyncio.sleep(idle)
                    store = self._extract_content(await asyncio.wait_for(
                        client.call_tool('store_context', {
                            'thread_id': thread, 'source': 'agent',
                            'text': f'writer-recycle entry {index}',
                        }),
                        timeout=180.0,
                    ))
                    if not store.get('success'):
                        self.test_results.append((test_name, False, f'Store {index} failed: {store}'))
                        return False
                    stored_ids.append(str(store['context_id']))
                    if index == 0:
                        before = await _failure_metrics(client)

                after = await _failure_metrics(client)
                if before is None or after is None:
                    self.test_results.append((test_name, False, 'connection_metrics did not expose the failure counters'))
                    return False
                if after[0] != before[0]:
                    self.test_results.append((
                        test_name, False,
                        f'failed_queries moved from {before[0]} to {after[0]} across the idle window',
                    ))
                    return False
                if after[1] != 'healthy':
                    self.test_results.append((
                        test_name, False, f'circuit_state is {after[1]!r} after the idle window, expected healthy',
                    ))
                    return False

                got = self._extract_content(await client.call_tool('get_context_by_ids', {'context_ids': stored_ids}))
                returned = {str(row.get('id')) for row in got.get('results', [])}
                if returned != set(stored_ids):
                    self.test_results.append((
                        test_name, False, f'Retrieved {sorted(returned)} but stored {sorted(stored_ids)}',
                    ))
                    return False

            self.test_results.append((
                test_name, True, 'Writes across an idle recycle window all succeeded with no failure charge',
            ))
            return True
        except TimeoutError:
            self.test_results.append((test_name, False, 'A write never returned after the idle window'))
            return False
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
