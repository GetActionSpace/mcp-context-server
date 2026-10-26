"""Tests for the shared transaction pieces in app.tools._transactions: the heartbeat,
connection error classification, the version re-read used by the compare-and-set retry,
the writer of an entry's generated outputs, and the refusal of a readable entry.
"""

import sqlite3
from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import call
from unittest.mock import patch

import asyncpg
import pytest
from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.backends.base import MAINTENANCE
from app.errors import ControlFlowError
from app.repositories.context_repository.records import EntryProbe
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools._transactions import EntryNotAuthorizedError
from app.tools._transactions import is_connection_error
from app.tools._transactions import reread_entry_version
from app.tools._transactions import transaction_heartbeat
from app.tools._transactions import write_generation_outputs
from tests.helpers import LOCAL_SCOPE


class TestTransactionHeartbeat:
    """Test in-transaction heartbeat helper."""

    @pytest.mark.asyncio
    async def test_heartbeat_executes_select_1(self) -> None:
        """Verify transaction_heartbeat sends SELECT 1 for PostgreSQL transactions."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_txn = AsyncMock()
        type(mock_txn).backend_type = PropertyMock(return_value='postgresql')
        type(mock_txn).connection = PropertyMock(return_value=mock_conn)

        await transaction_heartbeat(mock_txn)

        mock_conn.execute.assert_called_once_with('SELECT 1')

    @pytest.mark.asyncio
    async def test_heartbeat_noop_for_sqlite(self) -> None:
        """Verify transaction_heartbeat is a no-op for SQLite transactions."""
        mock_conn = MagicMock()
        mock_txn = MagicMock()
        type(mock_txn).backend_type = PropertyMock(return_value='sqlite')
        type(mock_txn).connection = PropertyMock(return_value=mock_conn)

        await transaction_heartbeat(mock_txn)

        mock_conn.execute.assert_not_called()


class TestConnectionErrorClassification:
    """Test connection error classification for retry logic."""

    def test_connection_errors_classified_correctly(self) -> None:
        """Verify is_connection_error identifies retryable connection errors."""
        assert is_connection_error(asyncpg.InterfaceError('connection closed'))
        assert is_connection_error(ConnectionResetError('reset'))
        assert is_connection_error(OSError('network unreachable'))

    def test_non_connection_errors_not_retried(self) -> None:
        """Verify non-connection errors are not classified as retryable."""
        assert not is_connection_error(ValueError('bad value'))
        assert not is_connection_error(TypeError('wrong type'))
        assert not is_connection_error(RuntimeError('logic error'))

    def test_query_canceled_error_is_retryable(self) -> None:
        """statement_timeout cancel (SQLSTATE 57014) is classified retryable.

        QueryCanceledError is raised when PostgreSQL cancels a statement that
        exceeded statement_timeout. It is a transient lock-wait/timeout error,
        safe to retry because the DB write is idempotent and generation already
        completed outside the transaction.
        """
        assert is_connection_error(asyncpg.exceptions.QueryCanceledError('canceling statement due to statement timeout'))

    def test_query_canceled_error_sqlstate_is_57014(self) -> None:
        """Document the SQLSTATE this classifier now treats as retryable."""
        assert asyncpg.exceptions.QueryCanceledError.sqlstate == '57014'

    def test_transaction_rollback_errors_are_retryable(self) -> None:
        """Server-initiated transaction rollbacks (SQLSTATE class 40) are retryable.

        PostgreSQL aborts one transaction to break a deadlock (40P01) or a
        serialization cycle (40001); the loser is expected to retry and succeeds
        once the competing transaction commits. Classifying the class-40 base
        as a connection-style transient makes the tool layer re-run the
        transaction instead of surfacing routine lock contention to the client.
        """
        assert is_connection_error(asyncpg.exceptions.TransactionRollbackError('rollback'))
        assert is_connection_error(asyncpg.exceptions.DeadlockDetectedError('deadlock detected'))
        assert is_connection_error(asyncpg.exceptions.SerializationError('could not serialize access'))

    def test_transaction_rollback_sqlstates_are_class_40(self) -> None:
        """Document the SQLSTATEs this classifier treats as retryable rollbacks."""
        assert asyncpg.exceptions.TransactionRollbackError.sqlstate == '40000'
        assert asyncpg.exceptions.DeadlockDetectedError.sqlstate == '40P01'
        assert asyncpg.exceptions.SerializationError.sqlstate == '40001'

    def test_sqlite_locked_family_is_retryable(self) -> None:
        """SQLite write contention (SQLITE_BUSY / SQLITE_LOCKED family) is retryable.

        begin_transaction -- the path every store/update transaction site uses --
        bypasses the SQLite write queue and performs no backend-level retry, so
        the tool-layer retry loops must classify a cross-process lock collision
        as transient, mirroring the PostgreSQL class-40 rollback treatment.
        """
        assert is_connection_error(sqlite3.OperationalError('database is locked'))
        assert is_connection_error(sqlite3.OperationalError('database table is locked'))

    def test_generic_sqlite_operational_error_not_retryable(self) -> None:
        """A non-contention sqlite3.OperationalError is NOT classified retryable."""
        assert not is_connection_error(sqlite3.OperationalError('no such table: context_entries'))
        assert not is_connection_error(sqlite3.OperationalError('malformed database schema'))

    def test_pool_saturation_timeout_error_not_retryable(self) -> None:
        """A pool-acquire TimeoutError is NOT retried despite subclassing OSError.

        TimeoutError is an OSError subclass on Python 3.12, so without an explicit
        carve-out it would ride the bare-OSError arm and be retried. The pool-acquire
        TimeoutError begin_transaction re-raises signals a SATURATED connection pool,
        and retrying it re-runs the full acquire wait each time. It must fail fast at
        the tool layer, matching execute_write's handling of the same signal.
        """
        assert isinstance(TimeoutError(), OSError)
        # asyncio.TimeoutError is an alias for the builtin TimeoutError on Python
        # 3.11+, and that builtin is exactly what asyncpg's pool.acquire timeout
        # surfaces, so excluding TimeoutError covers the pool-saturation shape.
        assert not is_connection_error(TimeoutError('pool acquire timed out'))

    def test_connection_reset_error_still_retryable_after_timeout_carveout(self) -> None:
        """The TimeoutError carve-out does not disturb the other OSError members.

        ConnectionResetError is an OSError subclass but not a TimeoutError, so a lost
        connection stays retryable after the saturation-timeout exclusion.
        """
        assert not isinstance(ConnectionResetError('reset'), TimeoutError)
        assert is_connection_error(ConnectionResetError('connection reset by peer'))
        assert is_connection_error(OSError('network unreachable'))


class TestRereadEntryVersion:
    """The version refresh after a compare-and-set conflict retries the READ.

    ``version`` is monotonic, so re-entering the write with the token whose
    compare-and-set just failed matches zero rows by construction. A transient fault
    during the refresh must therefore be retried on the read itself, leaving the
    caller to re-enter the write only once a fresh token is in hand.
    """

    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_and_returns_fresh_version(self) -> None:
        """A dropped connection during the refresh retries and yields the new version."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(
            side_effect=[asyncpg.InterfaceError('connection recycled'), EntryProbe(True, 'agent', 7, 'local', True)],
        )
        with patch('app.tools._transactions.asyncio.sleep', new_callable=AsyncMock):
            probe = await reread_entry_version(repos, '0190abcdef1234567890abcdef123456', scope=LOCAL_SCOPE)
        assert probe == EntryProbe(True, 'agent', 7, 'local', True)
        assert repos.context.check_entry_exists.await_count == 2

    @pytest.mark.asyncio
    async def test_missing_entry_is_reported_not_retried(self) -> None:
        """A deleted row is a clean answer, not a fault to retry."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(False, None, None, None, False))
        probe = await reread_entry_version(repos, '0190abcdef1234567890abcdef123456', scope=LOCAL_SCOPE)
        assert probe.exists is False
        assert probe.version is None
        assert repos.context.check_entry_exists.await_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_propagate(self) -> None:
        """The refresh is bounded; a persistent fault surfaces to the caller."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(
            side_effect=asyncpg.InterfaceError('connection recycled'),
        )
        with (
            patch('app.tools._transactions.asyncio.sleep', new_callable=AsyncMock),
            pytest.raises(asyncpg.InterfaceError),
        ):
            await reread_entry_version(repos, '0190abcdef1234567890abcdef123456', scope=LOCAL_SCOPE, max_retries=1)
        assert repos.context.check_entry_exists.await_count == 2

    @pytest.mark.asyncio
    async def test_logical_error_is_not_retried(self) -> None:
        """A non-transient error fails immediately instead of burning retries."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(side_effect=ValueError('bad id'))
        with pytest.raises(ValueError, match='bad id'):
            await reread_entry_version(repos, '0190abcdef1234567890abcdef123456', scope=LOCAL_SCOPE)
        assert repos.context.check_entry_exists.await_count == 1

    @pytest.mark.asyncio
    async def test_readable_entry_without_write_access_is_reported(self) -> None:
        """The refresh returns the whole probe, so the caller sees an entry it may read but no longer modify."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 4, 'alice', False))
        probe = await reread_entry_version(repos, '0190abcdef1234567890abcdef123456', scope=LOCAL_SCOPE)
        assert (probe.exists, probe.version, probe.can_write) == (True, 4, False)

    @pytest.mark.asyncio
    async def test_probe_runs_under_the_given_scope(self) -> None:
        """The refresh probes the entry as the caller, so an entry it may no longer read reads as gone."""
        bob = AccessScope('bob', frozenset())
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(False, None, None, None, False))
        probe = await reread_entry_version(repos, '0190abcdef1234567890abcdef123456', scope=bob)
        assert (probe.exists, probe.version) == (False, None)
        repos.context.check_entry_exists.assert_awaited_once_with('0190abcdef1234567890abcdef123456', scope=bob)


CONTEXT_ID = '0190abcdef1234567890abcd00000001'


class TestWriteGenerationOutputs:
    """write_generation_outputs writes an entry's generated embeddings and node rows on the given transaction."""

    @pytest.fixture
    def mock_repos(self) -> MagicMock:
        """A repository container whose embedding and node writers record their calls."""
        repos = MagicMock()
        repos.embeddings.delete_all_chunks = AsyncMock(return_value=1)
        repos.embeddings.store_chunked = AsyncMock()
        repos.index_nodes.replace_nodes_for_context = AsyncMock()
        return repos

    @pytest.fixture
    def mock_txn(self) -> MagicMock:
        """A SQLite transaction context."""
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        return txn

    @pytest.mark.asyncio
    async def test_embeddings_replace_the_stored_chunks(self, mock_repos: MagicMock, mock_txn: MagicMock) -> None:
        """Generated embeddings delete the stored chunks first, then store the new ones under the model."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock(), MagicMock()])
        calls = MagicMock()
        calls.attach_mock(mock_repos.embeddings.delete_all_chunks, 'delete_all_chunks')
        calls.attach_mock(mock_repos.embeddings.store_chunked, 'store_chunked')

        written = await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, chunk_embeddings, None, embedding_model='model-a', scope=MAINTENANCE,
        )

        assert written is True
        assert calls.mock_calls == [
            call.delete_all_chunks(CONTEXT_ID, txn=mock_txn, scope=MAINTENANCE),
            call.store_chunked(
                context_id=CONTEXT_ID, chunk_embeddings=chunk_embeddings, model='model-a', txn=mock_txn, scope=MAINTENANCE,
            ),
        ]

    @pytest.mark.asyncio
    async def test_no_embeddings_leaves_the_stored_chunks(self, mock_repos: MagicMock, mock_txn: MagicMock) -> None:
        """Without generated embeddings no chunk is deleted or stored."""
        written = await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, None, None, embedding_model='model-a', scope=LOCAL_SCOPE,
        )

        assert written is False
        mock_repos.embeddings.delete_all_chunks.assert_not_called()
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('nodes', [[], [MagicMock()]], ids=['empty', 'one'])
    async def test_a_node_list_replaces_the_stored_nodes(
        self, mock_repos: MagicMock, mock_txn: MagicMock, nodes: list[IndexNodeRow],
    ) -> None:
        """A node list, empty included, replaces the entry's node rows."""
        await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, None, nodes, embedding_model='model-a', scope=LOCAL_SCOPE,
        )

        mock_repos.index_nodes.replace_nodes_for_context.assert_awaited_once_with(
            CONTEXT_ID, nodes, txn=mock_txn, scope=LOCAL_SCOPE,
        )

    @pytest.mark.asyncio
    async def test_no_node_list_keeps_the_stored_nodes(self, mock_repos: MagicMock, mock_txn: MagicMock) -> None:
        """None for the nodes leaves the entry's node rows as they are."""
        await write_generation_outputs(
            mock_repos, mock_txn, CONTEXT_ID, cast(list[ChunkEmbedding], [MagicMock()]), None,
            embedding_model='model-a', scope=LOCAL_SCOPE,
        )

        mock_repos.index_nodes.replace_nodes_for_context.assert_not_called()


class TestEntryNotAuthorizedError:
    """The denial for a readable entry the caller may not modify or delete."""

    def test_modify_message_names_the_entry(self) -> None:
        """A modify denial reads exactly like the update tool reports it."""
        error = EntryNotAuthorizedError(['0190abcdef1234567890abcd00000001'], action='modify')
        assert str(error) == 'Not authorized to modify context entry with ID 0190abcdef1234567890abcd00000001'
        assert error.context_ids == ('0190abcdef1234567890abcd00000001',)
        assert error.action == 'modify'

    def test_delete_message_lists_the_entries_in_input_order(self) -> None:
        """A delete denial lists every refused entry in the order the caller named them."""
        error = EntryNotAuthorizedError(['b' * 32, 'a' * 32], action='delete')
        assert str(error) == f'Not authorized to delete context entries: {"b" * 32}, {"a" * 32}'
        assert error.context_ids == ('b' * 32, 'a' * 32)

    @pytest.mark.parametrize(
        ('action', 'message'),
        [
            ('share', 'Not authorized to share context entry with ID 0190abcdef1234567890abcd00000001'),
            ('unshare', 'Not authorized to unshare context entry with ID 0190abcdef1234567890abcd00000001'),
            ('list_grants', 'Not authorized to list the grants of context entry with ID 0190abcdef1234567890abcd00000001'),
        ],
    )
    def test_grant_messages_name_the_entry(
        self, action: Literal['share', 'unshare', 'list_grants'], message: str,
    ) -> None:
        """A refused grant change or listing reads exactly like the sharing tools report it."""
        error = EntryNotAuthorizedError(['0190abcdef1234567890abcd00000001'], action=action)
        assert str(error) == message
        assert error.action == action

    def test_is_breaker_exempt_control_flow(self) -> None:
        """A denial is client-input control flow, never a backend fault or a ToolError."""
        error = EntryNotAuthorizedError(['0190abcdef1234567890abcd00000001'], action='modify')
        assert isinstance(error, ControlFlowError)
        assert not isinstance(error, ToolError)
