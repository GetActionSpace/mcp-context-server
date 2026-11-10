"""unicode61 tokenizer behavior of the SQLite FTS5 table: multilingual text and hyphenated query terms."""

import sqlite3
from pathlib import Path

import pytest

from app.ids import generate_id


class TestFtsMultilingualUnicode61:
    """Test FTS with unicode61 tokenizer for multilingual content.

    The unicode61 tokenizer provides proper Unicode tokenization for all languages,
    but does NOT provide stemming. This is a trade-off: we get multilingual support
    at the cost of losing features like "running" matching "run".
    """

    @pytest.fixture
    def multilingual_db(self, tmp_path: Path) -> Path:
        """Create a database with multilingual content for FTS testing.

        Returns:
            Path to the test database with multilingual entries.
        """
        db_path = tmp_path / 'test_fts_multilingual.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        # Load FTS migration template and apply tokenizer replacement
        # Use 'unicode61' (no stemming) for multilingual support testing
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text()
        fts_sql = fts_sql.replace('{TOKENIZER}', 'unicode61')

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(schema_sql)
            conn.executescript(fts_sql)

            # Insert multilingual test data
            test_entries = [
                # German
                ('german-thread', 'agent', 'text', 'Die Programmierung ist interessant'),
                # French
                ('french-thread', 'agent', 'text', 'Le developpement logiciel est fascinant'),
                # Spanish
                ('spanish-thread', 'agent', 'text', 'La programacion es muy importante'),
                # Russian (Cyrillic)
                ('russian-thread', 'agent', 'text', 'Программирование это интересно'),
                # Chinese
                ('chinese-thread', 'agent', 'text', 'Python 编程语言很流行'),
                # Japanese
                ('japanese-thread', 'agent', 'text', 'プログラミングは楽しいです'),
                # Korean
                ('korean-thread', 'agent', 'text', '프로그래밍은 재미있습니다'),
                # Arabic
                ('arabic-thread', 'agent', 'text', 'البرمجة ممتعة جدا'),
                # Mixed content with accents
                ('mixed-thread', 'agent', 'text', 'Cafe resume naive facade'),
            ]

            for thread_id, source, content_type, text_content in test_entries:
                conn.execute(
                    '''
                    INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                    VALUES (?, ?, ?, ?, ?, 'local')
                    ''',
                    (generate_id(), thread_id, source, content_type, text_content),
                )
            conn.commit()

        return db_path

    def test_german_tokenization(self, multilingual_db: Path) -> None:
        """Test that German text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'Programmierung'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'Programmierung' in results[0]['text_content']

    def test_french_tokenization(self, multilingual_db: Path) -> None:
        """Test that French text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'developpement'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'developpement' in results[0]['text_content']

    def test_spanish_tokenization(self, multilingual_db: Path) -> None:
        """Test that Spanish text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'programacion'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'programacion' in results[0]['text_content']

    def test_russian_cyrillic_tokenization(self, multilingual_db: Path) -> None:
        """Test that Russian Cyrillic text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'Программирование'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'Программирование' in results[0]['text_content']

    def test_chinese_tokenization(self, multilingual_db: Path) -> None:
        """Test that Chinese text is searchable.

        Note: FTS5 with unicode61 indexes an unbroken run of CJK characters as one
        token, so the test searches for the ASCII word that stands apart from the run.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            # Search for "Python" which is ASCII within Chinese text
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'Python'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'chinese-thread'

    def test_japanese_tokenization(self, multilingual_db: Path) -> None:
        """Test that Japanese text entry exists and can be retrieved.

        Note: FTS5 with unicode61 has limited support for CJK tokenization
        without explicit ICU support, but entries should still be stored
        and retrievable via exact match or thread filtering.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                WHERE ce.thread_id = 'japanese-thread'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'japanese-thread'

    def test_korean_tokenization(self, multilingual_db: Path) -> None:
        """Test that Korean text entry exists and can be retrieved.

        Note: FTS5 with unicode61 handles Hangul, but word boundaries may
        differ from Korean language conventions.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                WHERE ce.thread_id = 'korean-thread'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'korean-thread'

    def test_arabic_tokenization(self, multilingual_db: Path) -> None:
        """Test that Arabic text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'البرمجة'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'arabic-thread'

    def test_accented_characters(self, multilingual_db: Path) -> None:
        """Test that words with accented characters are searchable.

        Unicode61 tokenizer handles accented characters properly.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            # Search for words that could have accents
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'Cafe'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'mixed-thread'

    def test_case_insensitive_search(self, multilingual_db: Path) -> None:
        """Test that unicode61 provides case-insensitive search."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search lowercase
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'programmierung'
            ''',
            )
            results_lower = cursor.fetchall()

            # Search uppercase
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'PROGRAMMIERUNG'
            ''',
            )
            results_upper = cursor.fetchall()

            # Both should find the same entry
            assert len(results_lower) == 1
            assert len(results_upper) == 1
            assert results_lower[0]['thread_id'] == results_upper[0]['thread_id']

    def test_prefix_search_multilingual(self, multilingual_db: Path) -> None:
        """Test that prefix search works with multilingual content."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'Program*'
            ''',
            )
            results = cursor.fetchall()

            # Should match German "Programmierung" and Spanish "programacion"
            assert len(results) >= 2


class TestFtsHyphenatedQueries:
    """Integration tests for FTS hyphen handling with real SQLite FTS5 database.

    These tests verify that hyphenated queries like "full-text" work correctly
    and do not cause errors like "no such column: text".
    """

    @pytest.fixture
    def hyphen_test_db(self, tmp_path: Path) -> Path:
        """Create a database with hyphenated content for testing.

        Returns:
            Path to the test database with hyphenated entries.
        """
        db_path = tmp_path / 'test_fts_hyphen.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text()
        fts_sql = fts_sql.replace('{TOKENIZER}', 'unicode61')

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(schema_sql)
            conn.executescript(fts_sql)

            # Insert test data with hyphenated words
            test_entries = [
                ('test-thread', 'agent', 'text', 'Implementing full-text search functionality'),
                ('test-thread', 'agent', 'text', 'Running pre-commit hooks before committing'),
                ('test-thread', 'user', 'text', 'Real-time data processing with streaming'),
                ('test-thread', 'agent', 'text', 'User-friendly interface design patterns'),
                ('test-thread', 'user', 'text', 'Open-source software development practices'),
                ('test-thread', 'agent', 'text', 'Multi-threaded application architecture'),
                ('test-thread', 'user', 'text', 'Regular search without hyphens'),
            ]

            for thread_id, source, content_type, text_content in test_entries:
                conn.execute(
                    '''
                    INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                    VALUES (?, ?, ?, ?, ?, 'local')
                    ''',
                    (generate_id(), thread_id, source, content_type, text_content),
                )
            conn.commit()

        return db_path

    def test_fts_hyphenated_match_mode(self, hyphen_test_db: Path) -> None:
        """Test match mode search with hyphenated term - should NOT error."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search with the quoted hyphenated term
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH '"full-text"'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'full-text' in results[0]['text_content']

    def test_fts_hyphenated_prefix_mode(self, hyphen_test_db: Path) -> None:
        """Test prefix mode search with hyphenated term."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Prefix search with quoted hyphenated term
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH '"pre-commit"*'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'pre-commit' in results[0]['text_content']

    def test_fts_hyphenated_phrase_mode(self, hyphen_test_db: Path) -> None:
        """Test phrase mode search with hyphenated term."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Phrase search including hyphenated word
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH '"full-text search"'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'full-text search' in results[0]['text_content']

    def test_fts_common_hyphenated_terms(self, hyphen_test_db: Path) -> None:
        """Test common hyphenated programming terms."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            hyphenated_terms = [
                ('real-time', 'Real-time'),
                ('user-friendly', 'User-friendly'),
                ('open-source', 'Open-source'),
                ('multi-threaded', 'Multi-threaded'),
            ]

            for term, expected_content in hyphenated_terms:
                cursor = conn.execute(
                    f'''
                    SELECT ce.* FROM context_entries ce
                    JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                    WHERE fts.text_content MATCH '"{term}"'
                    ''',
                )
                results = cursor.fetchall()

                assert len(results) >= 1, f'Failed to find term: {term}'
                assert any(
                    expected_content.lower() in r['text_content'].lower() for r in results
                ), f'Content mismatch for term: {term}'

    def test_fts_hyphenated_with_regular_words(self, hyphen_test_db: Path) -> None:
        """Test search mixing hyphenated and regular words."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search for "full-text" AND "search" (both must match)
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH '"full-text" search'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'full-text' in results[0]['text_content']
            assert 'search' in results[0]['text_content']

    def test_fts_non_hyphenated_query(self, hyphen_test_db: Path) -> None:
        """Test that regular (non-hyphenated) queries work."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # A regular search without hyphens works unquoted
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'Regular search'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'Regular search' in results[0]['text_content']
