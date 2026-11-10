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

from app.repositories.fts_repository.query import get_tsquery_function
from app.repositories.fts_repository.query import transform_query_postgresql
from app.tools.descriptions import generate_fts_description

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

Mode = Literal['match', 'prefix', 'phrase', 'boolean']


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


@pytest.mark.asyncio
async def test_the_match_and_prefix_examples(conn: asyncpg.Connection) -> None:
    assert await _matching(conn, 'english', 'match', 'running tests', ['we run a test', 'a test only']) == ['we run a test']
    words = ['searching', 'searched', 'search', 'research']
    assert await _matching(conn, 'english', 'prefix', 'search*', words) == ['searching', 'searched', 'search']
    documents = ['full text', 'the text is fully there', 'full only']
    assert await _matching(conn, 'english', 'prefix', 'full-text', documents) == ['full text', 'the text is fully there']


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
