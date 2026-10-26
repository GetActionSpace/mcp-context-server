"""Conformance of store_context_batch([single]) with store_context in generation.

Both paths invoke the embedding and summary generation helpers the same number of times, and both skip the
summary for short text.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from app.startup import ensure_repositories
from app.tools.batch.store import store_context_batch
from app.tools.context.store import store_context
from tests.tools._conformance import THREAD_PREFIX


@pytest.mark.usefixtures('initialized_server')
class TestGenerationConformance:
    """Verify embedding/summary generation parity between batch and non-batch."""

    @pytest.mark.asyncio
    async def test_generation_conformance_embeddings_triggered(self) -> None:
        """Both paths call generate_embeddings_with_timeout for the same text."""
        from app.repositories.embedding_repository.records import ChunkEmbedding

        mock_embedding = ChunkEmbedding(
            embedding=[0.1] * 1024,
            start_index=0,
            end_index=26,
        )
        mock_gen_embed = AsyncMock(return_value=[mock_embedding])
        mock_provider = AsyncMock()

        thread_nb = f'{THREAD_PREFIX}_gen_embed_nb'
        thread_b = f'{THREAD_PREFIX}_gen_embed_b'

        repos = await ensure_repositories()

        # Mock embedding storage to avoid missing vec_context_embeddings table
        with (
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.store.get_summary_provider', return_value=None),
            patch('app.tools._generation.generate_embeddings_with_timeout', mock_gen_embed),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.batch.store.get_summary_provider', return_value=None),
            patch('app.tools.batch._store_generation.generate_embeddings_with_timeout', mock_gen_embed),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_summary_provider', return_value=None),
            patch.object(repos.embeddings, 'store_chunked', AsyncMock()),
            patch.object(repos.embeddings, 'exists', AsyncMock(return_value=False)),
        ):
            mock_gen_embed.reset_mock()
            await store_context(
                thread_id=thread_nb, source='user', text='Embedding conformance text',
            )
            nb_call_count = mock_gen_embed.call_count

            mock_gen_embed.reset_mock()
            await store_context_batch(
                entries=[{
                    'thread_id': thread_b, 'source': 'user',
                    'text': 'Embedding conformance text',
                }],
                atomic=True,
            )
            b_call_count = mock_gen_embed.call_count

        assert nb_call_count == b_call_count, (
            f'Embedding generation call count mismatch: '
            f'nonbatch={nb_call_count} vs batch={b_call_count}'
        )
        assert nb_call_count >= 1, 'Embedding generation was not called'

    @pytest.mark.asyncio
    async def test_generation_conformance_summary_triggered(self) -> None:
        """Both paths call generate_summary_with_timeout for long text."""
        mock_gen_summary = AsyncMock(return_value='Mock summary')
        mock_provider = AsyncMock()

        # Text longer than default min_content_length (500)
        long_text = 'Summary conformance test content. ' * 20

        thread_nb = f'{THREAD_PREFIX}_gen_summ_nb'
        thread_b = f'{THREAD_PREFIX}_gen_summ_b'

        with (
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_provider),
            patch('app.tools.batch._store_generation.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
        ):
            mock_gen_summary.reset_mock()
            await store_context(
                thread_id=thread_nb, source='user', text=long_text,
            )
            nb_call_count = mock_gen_summary.call_count

            mock_gen_summary.reset_mock()
            await store_context_batch(
                entries=[{
                    'thread_id': thread_b, 'source': 'user', 'text': long_text,
                }],
                atomic=True,
            )
            b_call_count = mock_gen_summary.call_count

        assert nb_call_count == b_call_count, (
            f'Summary generation call count mismatch: '
            f'nonbatch={nb_call_count} vs batch={b_call_count}'
        )
        assert nb_call_count >= 1, 'Summary generation was not called for non-batch'

    @pytest.mark.asyncio
    async def test_generation_conformance_skip_short_content(self) -> None:
        """Both paths skip summary generation for text shorter than min_content_length."""
        mock_gen_summary = AsyncMock(return_value='Should not be called')
        mock_provider = AsyncMock()

        short_text = 'Short text'

        thread_nb = f'{THREAD_PREFIX}_gen_skip_nb'
        thread_b = f'{THREAD_PREFIX}_gen_skip_b'

        with (
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_provider),
            patch('app.tools._generation.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_provider),
            patch('app.tools.batch._store_generation.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=mock_provider),
        ):
            mock_gen_summary.reset_mock()
            await store_context(
                thread_id=thread_nb, source='user', text=short_text,
            )
            nb_call_count = mock_gen_summary.call_count

            mock_gen_summary.reset_mock()
            await store_context_batch(
                entries=[{
                    'thread_id': thread_b, 'source': 'user', 'text': short_text,
                }],
                atomic=True,
            )
            b_call_count = mock_gen_summary.call_count

        assert nb_call_count == 0, f'Non-batch called summary generation for short text ({nb_call_count} calls)'
        assert b_call_count == 0, f'Batch called summary generation for short text ({b_call_count} calls)'
