"""Tests for SUMMARY_MIN_CONTENT_LENGTH feature.

Tests verify:
- SummarySettings.min_content_length defaults, env var override, and range validation
- store_context skips summary generation for short text
- store_context generates summary for text >= threshold
- store_context boundary behavior (exactly at threshold IS summarized)
- store_context boundary: one char below threshold skips, one char above generates
- store_context with threshold=0 always generates summary
- update_context clears existing summary when text becomes short
- update_context generates summary for long text
- Deduplication preserves existing summary for short-text duplicates via COALESCE
- clear_summary parameter in update_context_entry
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import app.tools
from app.backends.base import BackendAccess
from app.repositories.context_repository.records import EntryProbe
from app.settings.summary import SummarySettings
from app.startup import ensure_repositories
from tests.helpers import LOCAL_SCOPE
from tests.tools._generation_jobs import InMemoryGenerationJobs

store_context = app.tools.store_context
update_context = app.tools.update_context


class TestMinContentLengthSettings:
    """Tests for SummarySettings.min_content_length field."""

    def test_default_is_500(self) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH defaults to 500."""
        settings = SummarySettings()
        assert settings.min_content_length == 500

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH can be set via env var."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '500')
        settings = SummarySettings()
        assert settings.min_content_length == 500

    def test_zero_disables_threshold(self) -> None:
        """Verify min_content_length=0 is valid (disables threshold)."""
        monkeypatch_env = {'SUMMARY_MIN_CONTENT_LENGTH': '0'}
        with patch.dict('os.environ', monkeypatch_env):
            settings = SummarySettings()
            assert settings.min_content_length == 0

    def test_minimum_valid_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify ge=0 constraint allows 0."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '0')
        settings = SummarySettings()
        assert settings.min_content_length == 0

    def test_negative_value_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify ge=0 constraint rejects negative values."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '-1')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_maximum_valid_is_10000(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify le=10000 constraint allows 10000."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '10000')
        settings = SummarySettings()
        assert settings.min_content_length == 10000

    def test_exceeds_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify le=10000 constraint rejects 10001."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '10001')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_accessible_via_app_settings(self) -> None:
        """Verify min_content_length is accessible via AppSettings.summary."""
        from app.settings import AppSettings

        settings = AppSettings()
        assert settings.summary.min_content_length == 500


@asynccontextmanager
async def _mock_begin_transaction(*, access_scope: BackendAccess) -> AsyncGenerator[Mock, None]:
    """Mock async context manager for begin_transaction."""
    _ = access_scope
    txn = Mock()
    txn.backend_type = 'sqlite'
    txn.connection = Mock()
    yield txn


def _make_mock_repos() -> MagicMock:
    """Create mock repository container for tool tests."""
    repos = MagicMock()

    mock_backend = MagicMock()
    mock_backend.begin_transaction = _mock_begin_transaction

    repos.context = MagicMock()
    repos.context.backend = mock_backend
    repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)
    repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
    repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))

    repos.tags = MagicMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    repos.images = MagicMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    repos.embeddings = MagicMock()
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.exists = AsyncMock(return_value=False)
    repos.embeddings.delete_all_chunks = AsyncMock(return_value=None)
    repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)

    repos.index_nodes = MagicMock()
    repos.index_nodes.replace_nodes_for_context = AsyncMock()
    repos.index_nodes.get_nodes_for_context = AsyncMock(return_value={})
    repos.index_nodes.count_all_nodes = AsyncMock(return_value=0)

    repos.generation_jobs = InMemoryGenerationJobs()

    return repos


class TestStoreContextMinContentLength:
    """Tests for min_content_length in store_context."""

    @pytest.mark.asyncio
    async def test_skips_summary_for_short_text(self) -> None:
        """Summary provider should NOT be called when text < min_content_length."""
        mock_repos = _make_mock_repos()

        mock_summary_provider = Mock()
        mock_summary_provider.summarize = AsyncMock(return_value='a summary')

        short_text = 'a' * 499  # Just below default threshold of 500

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_summary_provider),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary_provider),
        ):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=short_text,
            )

            assert result['success'] is True
            # Summary provider's summarize method should NOT have been called
            mock_summary_provider.summarize.assert_not_called()
            # store_with_deduplication should be called with summary=None
            mock_repos.context.store_with_deduplication.assert_called_once()
            call_kwargs = mock_repos.context.store_with_deduplication.call_args
            # summary argument should be None (not generated for short text)
            assert call_kwargs.kwargs.get('summary') is None

    @pytest.mark.asyncio
    async def test_generates_summary_for_long_text(self) -> None:
        """Summary provider SHOULD be called when text >= min_content_length."""
        mock_repos = _make_mock_repos()

        long_text = 'a' * 500  # Exactly at default threshold of 500

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=Mock()),
            patch('app.tools._generation.get_summary_provider', return_value=Mock()),
            patch(
                'app.tools._generation.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='generated summary',
            ),
        ):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=long_text,
            )

            assert result['success'] is True
            # store_with_deduplication should be called with the generated summary
            mock_repos.context.store_with_deduplication.assert_called_once()
            call_kwargs = mock_repos.context.store_with_deduplication.call_args
            assert call_kwargs.kwargs.get('summary') == 'generated summary'

    @pytest.mark.asyncio
    async def test_boundary_exactly_at_threshold(self) -> None:
        """Text at exactly min_content_length IS summarized (strict < comparison)."""
        mock_repos = _make_mock_repos()

        # Text at exactly 500 chars should be summarized
        boundary_text = 'x' * 500

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=Mock()),
            patch('app.tools._generation.get_summary_provider', return_value=Mock()),
            patch(
                'app.tools._generation.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='boundary summary',
            ) as mock_gen_summary,
        ):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=boundary_text,
            )

            assert result['success'] is True
            # generate_summary_with_timeout should have been called
            mock_gen_summary.assert_called_once_with(boundary_text, 'agent')

    @pytest.mark.asyncio
    async def test_one_char_below_threshold_skips(self) -> None:
        """Text at (min_content_length - 1) chars is NOT summarized."""
        mock_repos = _make_mock_repos()

        mock_summary_provider = Mock()
        mock_summary_provider.summarize = AsyncMock(return_value='a summary')

        # 499 chars = one below default threshold of 500
        below_text = 'b' * 499

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_summary_provider),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary_provider),
        ):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=below_text,
            )

            assert result['success'] is True
            mock_summary_provider.summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_one_char_above_threshold_generates(self) -> None:
        """Text at (min_content_length + 1) chars IS summarized."""
        mock_repos = _make_mock_repos()

        # 501 chars = one above default threshold of 500
        above_text = 'c' * 501

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=Mock()),
            patch('app.tools._generation.get_summary_provider', return_value=Mock()),
            patch(
                'app.tools._generation.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='above-boundary summary',
            ) as mock_gen_summary,
        ):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=above_text,
            )

            assert result['success'] is True
            mock_gen_summary.assert_called_once_with(above_text, 'agent')

    @pytest.mark.asyncio
    async def test_zero_threshold_always_generates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With min_content_length=0, even short text gets summary."""
        import app.tools.context.store as context_store_module

        mock_repos = _make_mock_repos()

        short_text = 'Hi'  # Very short text

        # Create a replacement settings with min_content_length=0
        mock_settings = MagicMock()
        mock_settings.summary.min_content_length = 0
        mock_settings.embedding.model = 'test-model'
        mock_settings.chunking.enabled = False

        # Replace module-level settings (frozen Pydantic model cannot be patched in-place)
        monkeypatch.setattr(context_store_module, 'settings', mock_settings)

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.store.get_summary_provider', return_value=Mock()),
            patch('app.tools._generation.get_summary_provider', return_value=Mock()),
            patch(
                'app.tools._generation.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='short summary',
            ) as mock_gen_summary,
        ):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=short_text,
            )

            assert result['success'] is True
            # With threshold=0, summary should be generated even for short text
            mock_gen_summary.assert_called_once_with(short_text, 'agent')


class TestUpdateContextMinContentLength:
    """Tests for min_content_length in update_context."""

    @pytest.mark.asyncio
    async def test_clears_summary_for_short_text(self) -> None:
        """Update with short text should clear existing summary."""
        mock_repos = _make_mock_repos()

        short_text = 'a' * 100  # Well below default threshold of 500

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.update.get_summary_provider', return_value=Mock()),
            patch('app.tools._generation.get_summary_provider', return_value=Mock()),
            patch('app.tools._generation.generate_embeddings_with_timeout', new_callable=AsyncMock, return_value=None),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000001',
                text=short_text,
            )

            assert result['success'] is True
            # update_context_entry should be called with clear_summary=True
            mock_repos.context.update_context_entry.assert_called_once()
            call_kwargs = mock_repos.context.update_context_entry.call_args
            assert call_kwargs.kwargs.get('clear_summary') is True
            # summary should be None (not generated)
            assert call_kwargs.kwargs.get('summary') is None
            # Message should mention summary cleared
            assert 'summary cleared' in result['message']

    @pytest.mark.asyncio
    async def test_generates_summary_for_long_text(self) -> None:
        """Update with long text should regenerate summary."""
        mock_repos = _make_mock_repos()

        mock_repos.context.update_context_entry.return_value = (True, ['text_content', 'summary'])
        long_text = 'a' * 500  # Well above default threshold of 300

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repos),
            patch('app.tools.context.update.get_summary_provider', return_value=Mock()),
            patch('app.tools._generation.get_summary_provider', return_value=Mock()),
            patch(
                'app.tools._generation.generate_summary_with_timeout',
                new_callable=AsyncMock,
                return_value='updated summary',
            ),
            patch('app.tools._generation.generate_embeddings_with_timeout', new_callable=AsyncMock, return_value=None),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000001',
                text=long_text,
            )

            assert result['success'] is True
            # update_context_entry should be called with summary text and clear_summary=False
            mock_repos.context.update_context_entry.assert_called_once()
            call_kwargs = mock_repos.context.update_context_entry.call_args
            assert call_kwargs.kwargs.get('summary') == 'updated summary'
            assert call_kwargs.kwargs.get('clear_summary') is False


@pytest.mark.usefixtures('initialized_server')
class TestClearSummaryRepository:
    """Tests for clear_summary parameter in update_context_entry."""

    @pytest.mark.asyncio
    async def test_clear_summary_sets_null_in_sqlite(self) -> None:
        """Verify clear_summary=True sets summary to NULL in SQLite."""
        repos = await ensure_repositories()

        # First, store an entry with a summary
        entry_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-clear-summary',
            source='agent',
            content_type='text',
            text_content='This is test content for clear summary test',
            metadata='{}',
            summary='Old summary that should be cleared',
        )

        # Verify the summary was stored
        stored_summary = (await repos.context.get_by_ids([entry_id], scope=LOCAL_SCOPE))[0]['summary']
        assert stored_summary == 'Old summary that should be cleared'

        # Update with clear_summary=True
        success, fields = await repos.context.update_context_entry(
            context_id=entry_id,
            text_content='Updated short text',
            clear_summary=True,
            scope=LOCAL_SCOPE,
        )
        assert success is True
        assert 'summary' in fields

        # Verify the summary is now NULL
        cleared_summary = (await repos.context.get_by_ids([entry_id], scope=LOCAL_SCOPE))[0]['summary']
        assert cleared_summary is None

    @pytest.mark.asyncio
    async def test_clear_summary_takes_precedence_over_summary(self) -> None:
        """When both clear_summary and summary are provided, clear_summary wins."""
        repos = await ensure_repositories()

        entry_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-clear-precedence',
            source='agent',
            content_type='text',
            text_content='Test content for precedence test',
            metadata='{}',
            summary='Original summary',
        )

        # Both summary and clear_summary provided - clear_summary should win
        success, fields = await repos.context.update_context_entry(
            context_id=entry_id,
            summary='This should be ignored',
            clear_summary=True,
            scope=LOCAL_SCOPE,
        )
        assert success is True
        assert 'summary' in fields

        result_summary = (await repos.context.get_by_ids([entry_id], scope=LOCAL_SCOPE))[0]['summary']
        assert result_summary is None

    @pytest.mark.asyncio
    async def test_summary_none_preserves_existing(self) -> None:
        """When summary=None and clear_summary=False, existing summary is preserved."""
        repos = await ensure_repositories()

        entry_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-preserve-summary',
            source='agent',
            content_type='text',
            text_content='Test content for preserve test',
            metadata='{}',
            summary='Summary that should be preserved',
        )

        # Update text without touching summary
        success, fields = await repos.context.update_context_entry(
            context_id=entry_id,
            text_content='Updated text content',
            summary=None,
            clear_summary=False,
            scope=LOCAL_SCOPE,
        )
        assert success is True

        # Summary should be preserved
        result_summary = (await repos.context.get_by_ids([entry_id], scope=LOCAL_SCOPE))[0]['summary']
        assert result_summary == 'Summary that should be preserved'


@pytest.mark.usefixtures('initialized_server')
class TestDedupPreservesExistingSummary:
    """Tests for deduplication behavior with short text and existing summary."""

    @pytest.mark.asyncio
    async def test_short_text_duplicate_preserves_summary(self) -> None:
        """Duplicate short text with existing summary preserves it via COALESCE.

        When a short-text entry is stored again as a duplicate:
        - min_content_length check skips summary generation (summary_text=None)
        - store_with_deduplication's COALESCE(NULL, summary) preserves existing
        """
        repos = await ensure_repositories()

        # First, store an entry with a summary (simulates pre-threshold entry)
        entry_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-dedup-preserve',
            source='agent',
            content_type='text',
            text_content='Short duplicate text',
            metadata='{}',
            summary='Pre-existing summary from before threshold',
        )

        # Verify summary exists
        initial_summary = (await repos.context.get_by_ids([entry_id], scope=LOCAL_SCOPE))[0]['summary']
        assert initial_summary == 'Pre-existing summary from before threshold'

        # Now store the same text again as a duplicate, with summary=None
        # (simulating what happens when min_content_length skips generation)
        updated_id, was_dedup = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-dedup-preserve',
            source='agent',
            text_content='Short duplicate text',
            metadata='{}',
            content_type='text',
            summary=None,  # Not generated because text < min_content_length
        )

        # Should be a dedup (same entry updated)
        assert was_dedup is True
        assert updated_id == entry_id

        # Summary should be PRESERVED via COALESCE(NULL, existing_summary)
        preserved_summary = (await repos.context.get_by_ids([entry_id], scope=LOCAL_SCOPE))[0]['summary']
        assert preserved_summary == 'Pre-existing summary from before threshold'
