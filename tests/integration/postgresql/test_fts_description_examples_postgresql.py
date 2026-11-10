"""The examples of the PostgreSQL full-text description hold on a real PostgreSQL.

The description served for ``fts_search_context`` tells a client agent how each search mode
reads a query. Each test builds the query the server builds for a mode (the tsquery function of
``get_tsquery_function`` over the text of ``transform_query_postgresql``) and asserts the outcome
the description states for its example.
"""

from collections.abc import AsyncIterator
from typing import Literal

import asyncpg
import pytest
import pytest_asyncio

from app.repositories.fts_repository.query import ENGLISH_STOPWORD_FTS_LANGUAGES
from app.repositories.fts_repository.query import get_tsquery_function
from app.repositories.fts_repository.query import transform_query_postgresql
from app.settings.search import FtsSettings
from app.tools.descriptions import generate_fts_description

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

Mode = Literal['match', 'prefix', 'phrase', 'boolean']

# The dictionary each text search configuration sends plain words to, per token type.
_WORD_DICTIONARIES = '''
    SELECT c.cfgname, t.alias, d.dictname, d.dictinitoption
    FROM pg_ts_config c
    JOIN pg_ts_config_map m ON m.mapcfg = c.oid
    JOIN pg_ts_dict d ON d.oid = m.mapdict
    JOIN ts_token_type('default') t ON t.tokid = m.maptokentype
    WHERE t.alias IN ('asciiword', 'word')
'''


@pytest_asyncio.fixture
async def conn(pg_test_url: str) -> AsyncIterator[asyncpg.Connection]:
    connection = await asyncpg.connect(pg_test_url)
    try:
        yield connection
    finally:
        await connection.close()


async def _matching(conn: asyncpg.Connection, language: str, mode: Mode, query: str, documents: list[str]) -> list[str]:
    """Return the documents the server's query for this mode matches, in the given order."""
    statement = f"SELECT to_tsvector('{language}', $1) @@ {get_tsquery_function(mode, language)}$2)"
    transformed = transform_query_postgresql(query, mode)
    return [document for document in documents if await conn.fetchval(statement, document, transformed)]


@pytest.mark.asyncio
async def test_the_english_stemming_example(conn: asyncpg.Connection) -> None:
    assert '"running" WILL match "run" and "runs", but not "runner"' in generate_fts_description('postgresql', 'english')
    assert await _matching(conn, 'english', 'match', 'running', ['run', 'runs', 'runner']) == ['run', 'runs']


@pytest.mark.asyncio
async def test_the_english_stop_words_are_ignored(conn: asyncpg.Connection) -> None:
    assert '"the", "is", "a" are REMOVED and ignored in searches' in generate_fts_description('postgresql', 'english')
    assert await _matching(conn, 'english', 'match', 'the python is a', ['python code', 'ruby code']) == ['python code']


@pytest.mark.asyncio
async def test_another_language_follows_its_own_stems_and_stop_words(conn: asyncpg.Connection) -> None:
    """Under german neither English example holds, which is why the description gives them for english only."""
    assert await _matching(conn, 'german', 'match', 'running', ['run', 'running']) == ['running']
    assert await _matching(conn, 'german', 'match', 'the python', ['python code', 'the python code']) == ['the python code']


async def _word_dictionaries(conn: asyncpg.Connection) -> dict[str, dict[str, tuple[str, str]]]:
    """Return, for each language FTS_LANGUAGE accepts, the dictionary and its options per word token type."""
    dictionaries: dict[str, dict[str, tuple[str, str]]] = {}
    for row in await conn.fetch(_WORD_DICTIONARIES):
        try:
            FtsSettings.validate_language(row['cfgname'])
        except ValueError:
            continue
        dictionaries.setdefault(row['cfgname'], {})[row['alias']] = (row['dictname'], row['dictinitoption'] or '')
    return dictionaries


@pytest.mark.asyncio
async def test_a_stop_word_list_is_claimed_exactly_where_postgresql_has_one(conn: asyncpg.Connection) -> None:
    """The description of every accepted language follows the stop word list of its own dictionary."""
    dictionaries = await _word_dictionaries(conn)
    assert len(dictionaries) == 29
    for language, by_token_type in dictionaries.items():
        if language == 'simple':
            continue
        description = generate_fts_description('postgresql', language)
        _, options = by_token_type['word']
        if 'stopwords' in options:
            assert f'Stop words: ENABLED (for {language} (language-specific list)' in description, language
        else:
            assert f'PostgreSQL has no {language} stop word list' in description, language


@pytest.mark.asyncio
async def test_ascii_words_follow_the_english_rules_where_the_description_says(conn: asyncpg.Connection) -> None:
    """Only the configurations the description names send ASCII words to the English stemmer and stop list."""
    dictionaries = await _word_dictionaries(conn)
    for language, by_token_type in dictionaries.items():
        ascii_dictionary, _ = by_token_type['asciiword']
        assert (ascii_dictionary == 'english_stem') == (language in ENGLISH_STOPWORD_FTS_LANGUAGES), language
        description = generate_fts_description('postgresql', language)
        named = 'ASCII words follow the English stemming rules and stop list' in description
        assert named == (language in ENGLISH_STOPWORD_FTS_LANGUAGES - {'english'}), language
        if language not in ENGLISH_STOPWORD_FTS_LANGUAGES and language != 'simple':
            assert ascii_dictionary == by_token_type['word'][0] == f'{language}_stem', language
    assert await _matching(conn, 'hindi', 'match', 'the book', ['a book', 'books']) == ['a book', 'books']
    assert await _matching(conn, 'russian', 'match', 'this is not a book', ['книга', 'a book']) == ['a book']
    assert await _matching(conn, 'russian', 'match', 'и не книга', ['книги', 'a book']) == ['книги']


@pytest.mark.asyncio
async def test_a_language_without_a_stop_word_list_requires_every_word(conn: asyncpg.Connection) -> None:
    assert 'No catalan word is ignored' in generate_fts_description('postgresql', 'catalan')
    assert await _matching(conn, 'catalan', 'match', 'el gat', ['un gat negre', 'el gat negre']) == ['el gat negre']
    assert await _matching(conn, 'french', 'match', 'le chat', ['un chat noir', 'le chat noir']) == [
        'un chat noir', 'le chat noir',
    ]


@pytest.mark.asyncio
async def test_words_of_another_language_follow_the_configuration(conn: asyncpg.Connection) -> None:
    assert '"maison" matches "maisons"' in generate_fts_description('postgresql', 'english')
    assert await _matching(conn, 'english', 'match', 'maison', ['maisons', 'maisonette']) == ['maisons']
    assert await _matching(conn, 'german', 'match', 'tests', ['a test', 'running']) == ['a test']
    assert await _matching(conn, 'simple', 'match', 'tests', ['a test', 'tests']) == ['tests']


@pytest.mark.asyncio
async def test_text_without_spaces_is_indexed_as_one_word(conn: asyncpg.Connection) -> None:
    documents = ['全文検索は便利です', '検索']
    for language in ('english', 'simple'):
        assert await _matching(conn, language, 'match', '検索', documents) == ['検索']
    assert await _matching(conn, 'english', 'prefix', '全文', documents) == ['全文検索は便利です']


@pytest.mark.asyncio
async def test_the_match_and_prefix_examples(conn: asyncpg.Connection) -> None:
    description = ' '.join(generate_fts_description('postgresql', 'english').split())
    assert await _matching(conn, 'english', 'match', 'running tests', ['we run a test', 'a test only']) == ['we run a test']
    assert 'Example: "search*" matches "searching", "searched", "searchable"' in description
    words = ['searching', 'searched', 'searchable', 'research']
    for query in ('search*', 'search'):
        assert await _matching(conn, 'english', 'prefix', query, words) == ['searching', 'searched', 'searchable']
    documents = ['full text', 'the text is fully there', 'full only']
    assert await _matching(conn, 'english', 'prefix', 'full-text', documents) == ['full text', 'the text is fully there']
    documents = ['python async', 'python only', 'asynchronous python']
    assert await _matching(conn, 'simple', 'prefix', 'python asy', documents) == ['python async', 'asynchronous python']


@pytest.mark.asyncio
async def test_a_typed_prefix_is_stemmed_where_stemming_is_on(conn: asyncpg.Connection) -> None:
    description = ' '.join(generate_fts_description('postgresql', 'english').split())
    assert '"runn*" does not match "running" (indexed as "run")' in description
    assert await _matching(conn, 'english', 'prefix', 'runn*', ['running', 'runner']) == ['runner']
    assert await _matching(conn, 'simple', 'prefix', 'runn*', ['running', 'runner']) == ['running', 'runner']


@pytest.mark.asyncio
async def test_the_phrase_examples(conn: asyncpg.Connection) -> None:
    description = generate_fts_description('postgresql', 'english')
    assert '"lord of the rings" matches "lord ... rings" with correct distance' in description
    documents = ['we run test suites', 'test then run']
    assert await _matching(conn, 'english', 'phrase', 'running tests', documents) == ['we run test suites']
    documents = ['the lord of the rings', 'lord and many more rings', 'lord rings']
    assert await _matching(conn, 'english', 'phrase', 'lord of the rings', documents) == ['the lord of the rings']


@pytest.mark.asyncio
async def test_boolean_mode_has_no_grouping(conn: asyncpg.Connection) -> None:
    """Parentheses are ignored and "or" binds looser than AND, as the description says."""
    description = ' '.join(generate_fts_description('postgresql', 'english').split())
    assert '"python async or python await" means (python AND async) OR (python AND await)' in description
    documents = ['python async', 'python await', 'python sync', 'await only']
    assert await _matching(conn, 'english', 'boolean', 'python async or python await', documents) == [
        'python async', 'python await',
    ]
    assert await _matching(conn, 'english', 'boolean', 'python (async or await)', documents) == [
        'python async', 'python await', 'await only',
    ]


@pytest.mark.asyncio
async def test_boolean_mode_negates_one_term_and_quotes_a_phrase(conn: asyncpg.Connection) -> None:
    documents = ['python async', 'python async blocking']
    assert await _matching(conn, 'english', 'boolean', 'python async -blocking', documents) == ['python async']
    documents = ['error handling in python', 'python handling of an error']
    assert await _matching(conn, 'english', 'boolean', '"error handling" python', documents) == ['error handling in python']
