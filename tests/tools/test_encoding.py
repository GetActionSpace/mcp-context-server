"""
Test UTF-8 encoding support across all languages and character sets.

This module ensures that the MCP Context Server correctly handles text
in various languages and special characters without encoding issues.
"""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.tools
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository.records import EntryProbe
from app.types import MetadataDict
from tests.tools._generation_jobs import InMemoryGenerationJobs

# Get the actual async functions from app.tools
store_context = app.tools.store_context
search_context = app.tools.search_context
update_context = app.tools.update_context


@pytest.mark.usefixtures('mock_server_dependencies')
class TestUTF8Encoding:
    """Test UTF-8 encoding support for various languages and character sets."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures.

        Tools wrap multi-step writes in ``backend.begin_transaction()`` and pass
        a ``txn`` argument to repository methods, so the mock backend must
        expose ``begin_transaction`` as an async context manager.
        """
        from contextlib import asynccontextmanager
        from unittest.mock import Mock

        self.mock_repos = MagicMock(spec=RepositoryContainer)

        # Mock backend exposing begin_transaction() as an async context manager
        mock_backend = Mock()

        @asynccontextmanager
        async def mock_begin_transaction(*, access_scope):
            _ = access_scope
            txn = Mock()
            txn.backend_type = 'sqlite'
            txn.connection = Mock()
            yield txn

        mock_backend.begin_transaction = mock_begin_transaction

        self.mock_repos.context = AsyncMock()
        self.mock_repos.context.backend = mock_backend
        self.mock_repos.tags = AsyncMock()
        self.mock_repos.images = AsyncMock()

        # Mock embeddings repository for generation-first transactional writes
        self.mock_repos.embeddings = AsyncMock()
        self.mock_repos.embeddings.store_chunked = AsyncMock(return_value=None)
        self.mock_repos.embeddings.delete_all_chunks = AsyncMock(return_value=None)

        # Mock index_tree node-summary repository (text-change updates clear stale node rows).
        self.mock_repos.index_nodes = AsyncMock()
        self.mock_repos.index_nodes.replace_nodes_for_context = AsyncMock(return_value=None)
        self.mock_repos.index_nodes.get_nodes_for_context = AsyncMock(return_value={})
        self.mock_repos.generation_jobs = InMemoryGenerationJobs()

    @pytest.mark.asyncio
    async def test_russian_text_encoding(self) -> None:
        """Test Russian text is properly encoded and stored."""
        russian_text = 'кривая ошибка - это тестовый текст на русском языке'

        # Mock repository responses
        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=russian_text,
            )

        assert result['success'] is True

        # Verify the text was passed correctly
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == russian_text

    @pytest.mark.asyncio
    async def test_chinese_text_encoding(self) -> None:
        """Test Chinese text is properly encoded and stored."""
        chinese_text = '这是一个中文测试文本，包含简体字和繁體字'

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=chinese_text,
            )

        assert result['success'] is True

        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == chinese_text

    @pytest.mark.asyncio
    async def test_arabic_hebrew_text_encoding(self) -> None:
        """Test Arabic and Hebrew text (RTL languages) are properly encoded."""
        arabic_text = 'هذا نص اختبار باللغة العربية'
        hebrew_text = 'זהו טקסט בדיקה בעברית'

        # Test Arabic
        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=arabic_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == arabic_text

        # Test Hebrew
        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=hebrew_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == hebrew_text

    @pytest.mark.asyncio
    async def test_emoji_encoding(self) -> None:
        """Test emoji and special Unicode characters are properly encoded."""
        emoji_text = '🎉 Testing emojis! 🚀 Complex ones: 👨‍👩‍👧‍👦 🏳️‍🌈'

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=emoji_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == emoji_text

    @pytest.mark.asyncio
    async def test_mixed_languages_encoding(self) -> None:
        """Test mixed language text with multiple scripts."""
        mixed_text = '''
        English: Hello World!
        Русский: Привет, мир!
        中文：你好，世界！
        日本語：こんにちは、世界！
        한국어: 안녕하세요, 세계!
        العربية: مرحبا بالعالم!
        עברית: שלום עולם!
        Emoji: 🌍🌎🌏
        '''

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=mixed_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        # Text gets stripped of leading/trailing whitespace
        assert call_args[1]['text_content'] == mixed_text.strip()

    @pytest.mark.asyncio
    async def test_special_unicode_characters(self) -> None:
        """Test special Unicode characters including mathematical symbols."""
        special_text = '∀x∈ℝ: x² ≥ 0, ∑ᵢ₌₁ⁿ i = n(n+1)/2, ∫₀^∞ e⁻ˣ dx = 1'

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=special_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == special_text

    @pytest.mark.asyncio
    async def test_update_context_with_unicode(self) -> None:
        """Test updating context with Unicode text."""
        unicode_texts = [
            'Обновленный русский текст',
            '更新的中文文本',
            'نص عربي محدث',
            'טקסט עברי מעודכן',
            '🎯 Updated with emojis! 🏆',
        ]

        for text in unicode_texts:
            self.mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'agent', 0, 'local', True))
            self.mock_repos.context.update_context_entry = AsyncMock(
                return_value=(True, ['text_content']),
            )
            # Also need to mock the count_images_for_context and get_content_type methods
            self.mock_repos.images.count_images_for_context = AsyncMock(return_value=0)
            self.mock_repos.context.get_content_type = AsyncMock(return_value='text')

            with patch('app.tools.context.update.ensure_repositories', return_value=self.mock_repos):
                result = await update_context(
                    context_id=generate_id(),
                    text=text,
                )

            assert result['success'] is True
            call_args = self.mock_repos.context.update_context_entry.call_args
            assert call_args[1]['text_content'] == text

    @pytest.mark.asyncio
    async def test_search_with_unicode_metadata(self) -> None:
        """Test searching with Unicode in metadata filters."""
        # Mock search response with Unicode content
        mock_entries = [
            {
                'id': 1,
                'thread_id': 'test-thread',
                'source': 'user',
                'content_type': 'text',
                'text_content': 'Текст на русском',
                'metadata': json.dumps({'language': 'русский'}),
                'created_at': '2025-01-01 00:00:00',
                'updated_at': '2025-01-01 00:00:00',
                'visibility': 'private',
                'is_owner': 1,
                'tags': ['тест', 'юникод'],
            },
        ]

        # Fix: search_contexts returns a tuple of (rows, stats)
        self.mock_repos.context.search_contexts = AsyncMock(return_value=(mock_entries, {}))
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=['тест', 'юникод'])

        with patch('app.tools.search.browse.ensure_repositories', return_value=self.mock_repos):
            result = await search_context(
                thread_id='test-thread',
                metadata={'language': 'русский'},
                limit=50,
            )

        assert 'results' in result
        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'Текст на русском'
        assert result['results'][0]['tags'] == ['тест', 'юникод']

    @pytest.mark.asyncio
    async def test_metadata_with_unicode(self) -> None:
        """Test storing and retrieving metadata with Unicode values."""
        metadata: MetadataDict = {
            'title': '测试标题',
            'description': 'وصف الاختبار',
            'tags': ['тег1', 'タグ2', '태그3'],
            'notes': '📝 Notes with emojis 🌟',
        }

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='Test content',
                metadata=metadata,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        stored_metadata = json.loads(call_args[1]['metadata'])
        assert stored_metadata == metadata

    @pytest.mark.asyncio
    async def test_tags_with_unicode(self) -> None:
        """Test tags with Unicode characters."""
        unicode_tags = ['русский', '中文', 'العربية', 'עברית', '🏷️']

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))
        self.mock_repos.tags.store_tags = AsyncMock()

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text='Test content',
                tags=unicode_tags,
            )

        assert result['success'] is True
        call_args = self.mock_repos.tags.store_tags.call_args
        assert call_args[0][1] == unicode_tags

    @pytest.mark.asyncio
    async def test_very_long_unicode_text(self) -> None:
        """Test storing very long Unicode text."""
        # Create a long text with various Unicode characters
        long_text = ''.join([
            'Длинный русский текст. ' * 100,
            '很长的中文文本。' * 100,
            'نص عربي طويل. ' * 100,
            'טקסט עברי ארוך. ' * 100,
            '🎉' * 50,
        ])

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=long_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == long_text
        assert len(call_args[1]['text_content']) == len(long_text)

    @pytest.mark.asyncio
    async def test_zero_width_characters(self) -> None:
        """Test handling of zero-width and invisible Unicode characters."""
        # Text with zero-width joiners, non-joiners, and other invisible chars
        tricky_text = 'Hello\u200bWorld\u200cTest\u200d\ufeffInvisible'

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=tricky_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == tricky_text

    @pytest.mark.asyncio
    async def test_combining_characters(self) -> None:
        """Test Unicode combining characters and normalization."""
        # Text with combining diacritics
        combining_text = 'e\u0301'  # é as e + combining acute accent
        normalized_text = 'café'  # Pre-composed character

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=combining_text + ' ' + normalized_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert combining_text in call_args[1]['text_content']
        assert normalized_text in call_args[1]['text_content']

    @pytest.mark.asyncio
    async def test_surrogate_pairs(self) -> None:
        """Test Unicode characters that require surrogate pairs."""
        # Characters outside the BMP (Basic Multilingual Plane)
        surrogate_text = '𝓗𝓮𝓵𝓵𝓸 𝔀𝓸𝓻𝓵𝓭 🎭'  # Mathematical bold script

        self.mock_repos.context.store_with_deduplication = AsyncMock(return_value=(generate_id(), False))

        with patch('app.tools.context.store.ensure_repositories', return_value=self.mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text=surrogate_text,
            )

        assert result['success'] is True
        call_args = self.mock_repos.context.store_with_deduplication.call_args
        assert call_args[1]['text_content'] == surrogate_text
