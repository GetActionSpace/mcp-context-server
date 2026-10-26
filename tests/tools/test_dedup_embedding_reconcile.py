"""Tests for the dedup pre-check / transaction divergence reconciliation.

When the read-only deduplication pre-check skips embedding generation for a
likely duplicate that already has embeddings, a concurrent same-thread write can
flip the in-transaction outcome to a genuine INSERT. Committing then would leave
a row with no embeddings while generation is enabled, violating the
generation-first guarantee. execute_store_in_transaction raises
EmbeddingsReconcileRequiredError in that case; store_context and
store_context_batch catch it, regenerate embeddings OUTSIDE the transaction, and
retry. These tests exercise the catch/regenerate/retry loop wiring, and that a
store with deduplicate=False, which skips no generation leg, never enters it.

Also guards the variant-aware semantic_distance contract wording so the
"L2 Euclidean (LOWER = better)" label with fixed thresholds does not regress
(it is wrong under the default ip compression variant, which returns a negated
inner product).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.repositories.context_repository.records import DuplicateCandidate
from app.tools._store_transaction import StoreOutcome
from app.tools._transactions import EmbeddingsReconcileRequiredError


def _make_mock_txn() -> tuple[MagicMock, object]:
    """Create a mock transaction context manager."""
    mock_txn = MagicMock()
    mock_txn.connection = MagicMock()
    mock_txn.backend_type = 'sqlite'

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        _ = access_scope
        yield mock_txn

    return mock_txn, mock_begin_transaction


def _fake_chunk_embeddings() -> list[tuple[str, list[float]]]:
    """Stand-in embeddings; execute_store_in_transaction is mocked so shape is irrelevant."""
    return [('chunk-0', [0.1, 0.2, 0.3])]


@pytest.mark.usefixtures('initialized_server')
class TestStoreContextReconcile:
    """store_context regenerates skipped embeddings when the store inserts a new row."""

    @pytest.mark.asyncio
    async def test_reconcile_regenerates_and_retries(self) -> None:
        """A reconcile signal triggers one regeneration and a successful retry."""
        from app.tools.context.store import store_context

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.context.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.context.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.store.get_summary_provider', return_value=None),
            patch('app.tools.context.store.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools._generation.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ) as mock_gen_emb,
            patch(
                'app.tools._generation.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Pre-check sees a duplicate WITH embeddings -> generation skipped.
            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(context_id='dup-id', summary=None),
            )
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            # First transaction diverges (INSERT); second succeeds after regeneration.
            mock_exec.side_effect = [
                EmbeddingsReconcileRequiredError('reconcile this entry'),
                StoreOutcome('ctx-1', False, True, 'ready'),
            ]

            result = await store_context(
                thread_id='reconcile-thread',
                source='user',
                text='reconcile this entry',
            )

        assert result['success'] is True
        assert result['context_id'] == 'ctx-1'
        # Exactly one retry of the transaction.
        assert mock_exec.call_count == 2
        # Embeddings were regenerated exactly once (only in the reconcile path,
        # since the pre-check skipped the initial generation).
        assert mock_gen_emb.await_count == 1

    @pytest.mark.asyncio
    async def test_reconcile_regenerates_reused_summary(self) -> None:
        """A divergence INSERT regenerates a summary the pre-check reused.

        The pre-check snapshot pairs the candidate's stored summary with the
        text its hash matched, but a concurrent commit can still change the
        candidate BETWEEN the pre-check and the transaction. When the
        transaction then diverges into a fresh INSERT, the reconcile path must
        regenerate the summary for THIS entry's text instead of persisting the
        candidate's since-stale reuse.
        """
        from app.tools.context.store import store_context

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.context.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=MagicMock()),
            patch('app.tools.context.store.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools.context.store.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='fresh summary for this text',
            ) as mock_gen_summary,
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Pre-check sees a duplicate WITH a stored summary -> reuse, no model call.
            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(
                    context_id='dup-id',
                    summary='stale summary of other text',
                ),
            )

            # First transaction diverges (INSERT); second succeeds after regeneration.
            mock_exec.side_effect = [
                EmbeddingsReconcileRequiredError('x' * 600),
                StoreOutcome('ctx-1', False, False, 'ready'),
            ]

            result = await store_context(
                thread_id='reconcile-thread',
                source='user',
                text='x' * 600,
            )

        assert result['success'] is True
        # The reused (potentially poisoned) summary was replaced by a fresh one.
        assert mock_gen_summary.await_count == 1
        assert mock_exec.call_count == 2
        # The retry passed the regenerated summary, not the reused one.
        retry_kwargs = mock_exec.call_args_list[1].kwargs
        assert retry_kwargs['summary'] == 'fresh summary for this text'
        # The first attempt flagged the reuse so the transaction could abort.
        first_kwargs = mock_exec.call_args_list[0].kwargs
        assert first_kwargs['summary_pending'] is True

    @pytest.mark.asyncio
    async def test_reconcile_does_not_loop_forever(self) -> None:
        """A repeated reconcile signal surfaces a ToolError rather than looping."""
        from app.tools.context.store import store_context

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.context.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.context.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.store.get_summary_provider', return_value=None),
            patch('app.tools.context.store.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools._generation.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ),
            patch(
                'app.tools._generation.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(context_id='dup-id', summary=None),
            )
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            # Always diverges -- the one-shot guard must stop the loop.
            mock_exec.side_effect = EmbeddingsReconcileRequiredError('reconcile this entry')

            with pytest.raises(ToolError, match='Failed to reconcile embeddings'):
                await store_context(
                    thread_id='reconcile-thread',
                    source='user',
                    text='reconcile this entry',
                )


@pytest.mark.usefixtures('initialized_server')
class TestStoreBatchReconcile:
    """store_context_batch regenerates skipped embeddings on dedup divergence."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('atomic', [True, False])
    async def test_batch_reconcile_regenerates_and_retries(self, atomic: bool) -> None:
        """Atomic and non-atomic batch store both reconcile a diverged INSERT."""
        from app.tools.batch.store import store_context_batch

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.batch.store.get_summary_provider', return_value=None),
            patch('app.tools.batch.store.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools.batch.store.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ) as mock_gen_emb,
            patch(
                'app.tools.batch.store.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(context_id='dup-id', summary=None),
            )
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            mock_exec.side_effect = [
                EmbeddingsReconcileRequiredError('batch reconcile entry'),
                StoreOutcome('ctx-1', False, True, 'ready'),
            ]

            result = await store_context_batch(
                entries=[{
                    'thread_id': 'batch-reconcile-thread',
                    'source': 'user',
                    'text': 'batch reconcile entry',
                }],
                atomic=atomic,
            )

        assert result['results'][0]['success'] is True
        assert result['results'][0]['context_id'] == 'ctx-1'
        assert mock_exec.call_count == 2
        assert mock_gen_emb.await_count == 1

    @pytest.mark.asyncio
    async def test_atomic_reconcile_regenerates_summary_once_per_text_source_pair(self) -> None:
        """Identical (text, source) entries share ONE summary regeneration.

        The atomic reconcile's embedding block computes the regenerated
        embedding once and broadcasts it to every matching entry, and the
        summary block does the same: calling the model INSIDE the per-entry
        loop would issue redundant, sequential, identical LLM round-trips when
        several diverged entries share the same text AND source. The prompt
        varies only by source for a fixed text, so one call per distinct
        source must be broadcast to all matching entries.
        """
        from app.tools.batch.store import store_context_batch

        _, mock_begin_transaction = _make_mock_txn()
        # Long enough to clear SUMMARY_MIN_CONTENT_LENGTH so the pre-check
        # actually reserves (reuses) the candidate summary for both entries.
        shared_text = 'y' * 600

        with (
            patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.batch.store.get_summary_provider', return_value=MagicMock()),
            patch('app.tools.batch.store.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools.batch.store.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ),
            patch(
                'app.tools.batch.store.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
            patch(
                'app.tools.batch.store.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='fresh shared summary',
            ) as mock_gen_summary,
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Pre-check: duplicate WITH embeddings and a stored summary, so
            # both entries skip generation and REUSE the candidate summary.
            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(
                    context_id='dup-id',
                    summary='stale summary of other text',
                ),
            )
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            # First transaction attempt diverges on entry 1 (whole atomic txn
            # aborts); the retry stores both entries.
            mock_exec.side_effect = [
                EmbeddingsReconcileRequiredError(shared_text),
                StoreOutcome('ctx-1', False, True, 'ready'),
                StoreOutcome('ctx-2', False, True, 'ready'),
            ]

            result = await store_context_batch(
                entries=[
                    {'thread_id': 'batch-shared-1', 'source': 'user', 'text': shared_text},
                    {'thread_id': 'batch-shared-2', 'source': 'user', 'text': shared_text},
                ],
                atomic=True,
            )

        assert all(r['success'] for r in result['results'])
        assert mock_exec.call_count == 3
        # ONE model call for the shared (text, source) pair, broadcast to both.
        assert mock_gen_summary.await_count == 1
        retry_calls = mock_exec.call_args_list[1:]
        assert all(c.kwargs['summary'] == 'fresh shared summary' for c in retry_calls)

    @pytest.mark.asyncio
    async def test_nonatomic_reconcile_failure_isolates_to_entry(self) -> None:
        """A reconcile-time embedding failure fails only that entry, not the batch.

        Non-atomic mode promises per-entry isolation. When the dedup pre-check
        skipped embeddings for a likely duplicate and the store then INSERTs
        (EmbeddingsReconcileRequiredError), a ToolError from the reconcile-time
        regeneration must be recorded as THAT entry's failure while a sibling
        entry that already succeeded is preserved -- never re-raised batch-wide.
        """
        from app.tools.batch.store import store_context_batch

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.batch.store.get_summary_provider', return_value=None),
            patch('app.tools.batch.store.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools.batch.store.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                side_effect=ToolError('embedding provider unavailable'),
            ) as mock_gen_emb,
            patch(
                'app.tools.batch.store.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Both entries look like duplicates with embeddings -> upfront
            # generation is skipped for both.
            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(context_id='dup-id', summary=None),
            )
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            # Entry 0 stores cleanly; entry 1 diverges (INSERT) so its skipped
            # embeddings must be regenerated -- and that regeneration raises.
            mock_exec.side_effect = [
                StoreOutcome('ctx-good', False, True, 'ready'),
                EmbeddingsReconcileRequiredError('diverged entry'),
            ]

            result = await store_context_batch(
                entries=[
                    {'thread_id': 't', 'source': 'user', 'text': 'good entry'},
                    {'thread_id': 't', 'source': 'user', 'text': 'diverged entry'},
                ],
                atomic=False,
            )

        # No batch-wide ToolError was raised: the sibling result survives.
        assert result['succeeded'] == 1
        assert result['failed'] == 1
        assert result['success'] is False
        by_index = {r['index']: r for r in result['results']}
        assert by_index[0]['success'] is True
        assert by_index[0]['context_id'] == 'ctx-good'
        assert by_index[1]['success'] is False
        assert by_index[1]['context_id'] is None
        assert by_index[1]['error']
        # The reconcile-time regeneration was attempted exactly once (entry 1).
        assert mock_gen_emb.await_count == 1

    @pytest.mark.asyncio
    async def test_nonatomic_reconcile_summary_failure_isolates_to_entry(self) -> None:
        """A reconcile-time summary PROVIDER failure fails only that entry.

        The reconcile-time summary regeneration is isolated with ``except
        ToolError`` so a failure records a per-entry result instead of escaping the
        per-entry loop. That isolation only holds if the summary helper actually
        normalizes provider failures: the retry layer raises SummaryRetryExhaustedError
        and the providers raise bare RuntimeError/ValueError, none of which is a
        ToolError, so a routine summary outage escaped the loop, hit the
        function-level handler, and aborted the whole non-atomic batch -- discarding
        sibling results for entries already committed in their own transactions.
        The provider (not the helper) is faulted here so the normalization itself is
        under test.
        """
        from app.summary.retry import SummaryRetryExhaustedError
        from app.tools.batch.store import store_context_batch

        _, mock_begin_transaction = _make_mock_txn()
        good_text = 'a' * 600
        diverged_text = 'b' * 600
        failing_provider = MagicMock()
        failing_provider.summarize = AsyncMock(
            side_effect=SummaryRetryExhaustedError('summary sidecar unreachable'),
        )

        with (
            patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=failing_provider),
            patch('app.tools._generation.get_summary_provider', return_value=failing_provider),
            patch(
                'app.tools.batch.store.generate_index_nodes_with_timeout',
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch('app.tools.batch.store.execute_store_in_transaction') as mock_exec,
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Both entries look like duplicates WITH a stored summary, so the
            # pre-check reuses it and no upfront model call happens.
            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(
                    context_id='dup-id',
                    summary='stale summary of other text',
                ),
            )

            # Entry 0 stores cleanly; entry 1 diverges into an INSERT, so its
            # reused summary must be regenerated -- and the provider is down.
            mock_exec.side_effect = [
                StoreOutcome('ctx-good', False, True, 'ready'),
                EmbeddingsReconcileRequiredError(diverged_text),
            ]

            result = await store_context_batch(
                entries=[
                    {'thread_id': 't', 'source': 'user', 'text': good_text},
                    {'thread_id': 't', 'source': 'user', 'text': diverged_text},
                ],
                atomic=False,
            )

        # No batch-wide abort: the committed sibling result survives.
        assert result['succeeded'] == 1
        assert result['failed'] == 1
        by_index = {r['index']: r for r in result['results']}
        assert by_index[0]['success'] is True
        assert by_index[0]['context_id'] == 'ctx-good'
        assert by_index[1]['success'] is False
        assert by_index[1]['error']
        assert 'Summary generation failed' in by_index[1]['error']


def _repos_with_a_reusable_duplicate(mock_begin_transaction: object) -> AsyncMock:
    """Mock repositories whose pre-check would offer a duplicate with stored embeddings and summary."""
    mock_repos = AsyncMock()
    mock_backend = MagicMock()
    mock_backend.backend_type = 'sqlite'
    mock_backend.begin_transaction = mock_begin_transaction
    mock_repos.context.backend = mock_backend
    mock_repos.context.check_latest_is_duplicate = AsyncMock(
        return_value=DuplicateCandidate(context_id='dup-id', summary='Stored summary'),
    )
    mock_repos.context.insert_entry = AsyncMock(return_value='fresh-id')
    mock_repos.embeddings.exists = AsyncMock(return_value=True)
    return mock_repos


@pytest.mark.usefixtures('initialized_server')
class TestNoReconcileWithoutDeduplication:
    """A store without deduplication skips no generation leg, so its fresh insert never needs a reconcile."""

    @pytest.mark.asyncio
    async def test_single_store_inserts_once(self) -> None:
        """store_context runs one transaction that inserts; the reconcile regeneration never runs."""
        from app.tools._store_transaction import execute_store_in_transaction
        from app.tools.context.store import store_context

        _, mock_begin_transaction = _make_mock_txn()
        mock_repos = _repos_with_a_reusable_duplicate(mock_begin_transaction)

        async def generation(
            _text: str, _source: str, *, run_embedding: bool, run_summary: bool, run_nodes: bool,
        ) -> tuple[list[tuple[str, list[float]]] | None, str | None, list[object] | None]:
            # Honors the leg flags, so a leg the store skipped yields nothing, as run_generation does.
            return (
                _fake_chunk_embeddings() if run_embedding else None,
                'Fresh summary' if run_summary else None,
                [] if run_nodes else None,
            )

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.store.get_summary_provider', return_value=MagicMock()),
            patch('app.tools.context.store.node_layer_active', return_value=True),
            patch('app.tools.context.store.run_generation', new_callable=AsyncMock, side_effect=generation),
            patch('app.tools.context.store.embed_then_compress', new_callable=AsyncMock) as mock_regenerate,
            patch(
                'app.tools.context.store.execute_store_in_transaction',
                new_callable=AsyncMock,
                side_effect=execute_store_in_transaction,
            ) as mock_exec,
        ):
            result = await store_context(
                thread_id='kb:handbook', source='user', text='z' * 600, deduplicate=False,
            )

        assert result['context_id'] == 'fresh-id'
        assert mock_exec.await_count == 1
        mock_regenerate.assert_not_awaited()
        mock_repos.context.insert_entry.assert_awaited_once()
        mock_repos.context.store_with_deduplication.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('atomic', [True, False])
    async def test_batch_store_inserts_once(self, atomic: bool) -> None:
        """store_context_batch runs one store per entry that inserts; the reconcile regeneration never runs."""
        from app.tools._store_transaction import execute_store_in_transaction
        from app.tools.batch.store import store_context_batch

        _, mock_begin_transaction = _make_mock_txn()
        mock_repos = _repos_with_a_reusable_duplicate(mock_begin_transaction)

        with (
            patch('app.tools.batch.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.batch.store.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.batch.store.get_summary_provider', return_value=MagicMock()),
            patch('app.tools.batch.store.node_layer_active', return_value=True),
            patch(
                'app.tools.batch.store.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ) as mock_gen_emb,
            patch(
                'app.tools.batch.store.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
            patch(
                'app.tools.batch.store.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='Fresh summary',
            ) as mock_gen_summary,
            patch('app.tools.batch.store.generate_index_nodes_with_timeout', new_callable=AsyncMock, return_value=[]),
            patch(
                'app.tools.batch.store.execute_store_in_transaction',
                new_callable=AsyncMock,
                side_effect=execute_store_in_transaction,
            ) as mock_exec,
        ):
            result = await store_context_batch(
                entries=[{'thread_id': 'kb:handbook', 'source': 'user', 'text': 'z' * 600, 'deduplicate': False}],
                atomic=atomic,
            )

        assert result['results'][0]['context_id'] == 'fresh-id'
        assert mock_exec.await_count == 1
        assert mock_gen_emb.await_count == 1
        assert mock_gen_summary.await_count == 1
        mock_repos.context.check_latest_is_duplicate.assert_not_awaited()
        mock_repos.context.insert_entry.assert_awaited_once()


class TestSemanticDistanceContractWording:
    """Guards the variant-aware semantic_distance contract wording."""

    def test_search_tool_drops_fixed_threshold_heuristic(self) -> None:
        """Search tools carry no fixed-band heuristic, and semantic and hybrid state the variant-aware distance."""
        paths = sorted(Path('app/tools/search').glob('*.py'))
        assert paths
        for path in paths:
            assert '<0.5 very similar' not in path.read_text(encoding='utf-8'), path
        for name in ('semantic.py', 'hybrid.py'):
            assert 'negated inner product' in Path('app/tools/search', name).read_text(encoding='utf-8'), name

    def test_types_describe_variant_aware_distance(self) -> None:
        """TypedDicts do not assert an unconditional L2 Euclidean distance."""
        content = Path('app/types.py').read_text(encoding='utf-8')
        assert 'negated inner product' in content
        assert 'L2 Euclidean distance (LOWER = better)' not in content
