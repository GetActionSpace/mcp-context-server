"""Transaction retry in store_context and update_context.

A statement-timeout cancellation (SQLSTATE 57014) on the first attempt is retried, the second attempt commits,
and generation runs once.
"""

import contextlib
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import asyncpg
import pytest

from app.backends.base import BackendAccess
from app.repositories.context_repository.records import EntryProbe
from app.tools._store_transaction import StoreOutcome
from app.tools._update_transaction import UpdateOutcome
from app.tools.context.store import store_context
from app.tools.context.update import update_context


class TestStatementTimeoutRetryToolLayer:
    """store_context / update_context retry a simulated QueryCanceledError.

    The tool-layer retry loop wraps backend.begin_transaction(); a single
    SQLSTATE 57014 cancellation on the first attempt must be retried and the
    second attempt must commit. Generation runs OUTSIDE the transaction and must
    be invoked exactly once across the whole call (never re-skipped on retry,
    never duplicated). asyncio.sleep is patched so backoff does not slow tests.
    """

    @pytest.mark.asyncio
    async def test_store_context_retries_statement_timeout_then_succeeds(self) -> None:
        """A single 57014 on the first begin_transaction is retried, then commits once."""
        repos = MagicMock()
        state = {'calls': 0}

        @contextlib.asynccontextmanager
        async def _begin(*, access_scope: BackendAccess) -> AsyncGenerator[MagicMock, None]:
            _ = access_scope
            state['calls'] += 1
            txn = MagicMock()
            txn.backend_type = 'postgresql'
            txn.connection = AsyncMock()
            if state['calls'] == 1:
                # Enter succeeds, body runs, then exit raises -- mirrors a
                # statement_timeout firing mid-transaction with rollback.
                yield txn
                raise asyncpg.exceptions.QueryCanceledError(
                    'canceling statement due to statement timeout',
                )
            yield txn

        repos.context.backend.begin_transaction = _begin

        with (
            patch('app.tools.context.store.ensure_repositories', AsyncMock(return_value=repos)),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=None),
            patch('app.tools._generation.generate_compression_with_timeout', AsyncMock(side_effect=lambda x: x)),
            patch(
                'app.tools.context.store.execute_store_in_transaction',
                AsyncMock(return_value=StoreOutcome('ctx-1', False, False, 'ready')),
            ) as mock_exec,
            patch('asyncio.sleep', AsyncMock()),
        ):
            result = await store_context(thread_id='t', source='user', text='hello world')

        assert result['success'] is True
        assert result['context_id'] == 'ctx-1'
        # begin_transaction entered twice (one canceled, one committed);
        # execute_store_in_transaction ran once per attempt => exactly twice,
        # never producing two committed rows (only the second attempt commits).
        assert state['calls'] == 2
        assert mock_exec.await_count == 2

    @pytest.mark.asyncio
    async def test_store_context_generation_invoked_once_across_retry(self) -> None:
        """Generation runs once OUTSIDE the txn and is NOT re-run on retry."""
        repos = MagicMock()
        repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)
        repos.embeddings.exists = AsyncMock(return_value=False)
        state = {'calls': 0}
        gen_calls = {'n': 0}

        @contextlib.asynccontextmanager
        async def _begin(*, access_scope: BackendAccess) -> AsyncGenerator[MagicMock, None]:
            _ = access_scope
            state['calls'] += 1
            txn = MagicMock()
            txn.backend_type = 'postgresql'
            txn.connection = AsyncMock()
            if state['calls'] == 1:
                yield txn
                raise asyncpg.exceptions.QueryCanceledError(
                    'canceling statement due to statement timeout',
                )
            yield txn

        repos.context.backend.begin_transaction = _begin

        async def _fake_generate_embeddings(_text: str) -> None:
            gen_calls['n'] += 1

        with (
            patch('app.tools.context.store.ensure_repositories', AsyncMock(return_value=repos)),
            patch('app.tools.context.store.get_embedding_provider', return_value=object()),
            patch('app.tools.context.store.get_summary_provider', return_value=None),
            patch(
                'app.tools._generation.generate_embeddings_with_timeout',
                AsyncMock(side_effect=_fake_generate_embeddings),
            ),
            patch('app.tools._generation.generate_compression_with_timeout', AsyncMock(side_effect=lambda x: x)),
            patch(
                'app.tools.context.store.execute_store_in_transaction',
                AsyncMock(return_value=StoreOutcome('ctx-2', False, False, 'ready')),
            ),
            patch('asyncio.sleep', AsyncMock()),
        ):
            result = await store_context(thread_id='t', source='user', text='hello world')

        assert result['success'] is True
        # Despite the transaction retrying once, embedding generation ran exactly
        # once (outside the transaction); the retry re-runs only the DB write.
        assert gen_calls['n'] == 1
        assert state['calls'] == 2

    @pytest.mark.asyncio
    async def test_update_context_retries_statement_timeout_then_succeeds(self) -> None:
        """update_context retries a 57014 cancellation, then commits once."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
        state = {'calls': 0}

        @contextlib.asynccontextmanager
        async def _begin(*, access_scope: BackendAccess) -> AsyncGenerator[MagicMock, None]:
            _ = access_scope
            state['calls'] += 1
            txn = MagicMock()
            txn.backend_type = 'postgresql'
            txn.connection = AsyncMock()
            if state['calls'] == 1:
                yield txn
                raise asyncpg.exceptions.QueryCanceledError(
                    'canceling statement due to statement timeout',
                )
            yield txn

        repos.context.backend.begin_transaction = _begin

        with (
            patch('app.tools.context.update.ensure_repositories', AsyncMock(return_value=repos)),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch(
                'app.tools.context.update.resolve_or_normalize_id',
                AsyncMock(return_value='0190abcdef1234567890abcd00000001'),
            ),
            patch('app.tools._generation.generate_compression_with_timeout', AsyncMock(side_effect=lambda x: x)),
            patch(
                'app.tools.context.update.execute_update_in_transaction',
                AsyncMock(return_value=UpdateOutcome(['text'], False, 'ready')),
            ) as mock_exec,
            patch('asyncio.sleep', AsyncMock()),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000001', text='new text',
            )

        assert result['success'] is True
        assert state['calls'] == 2
        assert mock_exec.await_count == 2
