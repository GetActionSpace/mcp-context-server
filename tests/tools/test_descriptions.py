"""Tests for the tool descriptions and schemas the server serves to clients.

Tests verify that generate_fts_description produces correct output
for both SQLite and PostgreSQL backends across various FTS language
configurations, and that structural invariants of the description
template are maintained. The access-scoping tests also pin the
SQLite score note in the description hybrid_search_context serves,
and the read-back tests pin that the full-text description and the
return description of every other entry-carrying search and read tool
name the visibility, is_owner and generation_status their results report,
and that the get_statistics description names its generation block. The write
tool tests pin that every store and update tool names defer_generation and the
generation_status its response reports.
The sync tests pin what the corpus sync tools' served schemas tell a
client: how to compute a manifest item's text digest, and no name of
the server's own modules or constants.
"""

import inspect
import json
import re
import sqlite3
import typing
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Literal
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from fastmcp.tools import Tool

from app import tools as exported_tools
from app.discovery_types import StatisticsResponseDict
from app.ingest import MAX_PLAN_DELETES
from app.repositories.fts_repository.query import desired_sqlite_fts_tokenizer
from app.repositories.fts_repository.query import transform_query_sqlite
from app.tools import TOOL_ANNOTATIONS
from app.tools import plan_context_sync
from app.tools import register_tool
from app.tools import store_context
from app.tools import store_context_batch
from app.tools import update_context
from app.tools import update_context_batch
from app.tools.descriptions import _FTS_DESCRIPTION_TEMPLATE
from app.tools.descriptions import _POSTGRESQL_MODE_DESCRIPTIONS
from app.tools.descriptions import _SQLITE_MODE_DESCRIPTIONS
from app.tools.descriptions import _generate_postgresql_fts_description
from app.tools.descriptions import _generate_sqlite_fts_description
from app.tools.descriptions import generate_fts_description
from app.tools.search.limits import RANKED_SEARCH_DEPTH


async def _served_tool(tool_name: str) -> Tool:
    """Return a tool as a server serves it, registered through register_tool whatever DISABLED_TOOLS holds."""
    mcp: FastMCP[None] = FastMCP(name='served-tools')
    with patch('app.tools.is_tool_disabled', return_value=False):
        assert register_tool(mcp, getattr(exported_tools, tool_name)) is True
    tool = await mcp.get_tool(tool_name)
    assert tool is not None
    return tool


async def _served_description(tool_name: str) -> str:
    """Return the description a server serves for a tool that brings no description of its own."""
    tool = await _served_tool(tool_name)
    assert tool.description is not None
    return tool.description


class TestServedDescription:
    """A tool's served description is its whole docstring unless the registration passes one."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
    async def test_served_description_is_the_whole_docstring(self, tool_name: str) -> None:
        """The description carries every docstring section, the return and error descriptions included."""
        description = await _served_description(tool_name)

        assert description == inspect.getdoc(getattr(exported_tools, tool_name))
        assert description.index('Returns:') < description.index('Raises:')

    @pytest.mark.asyncio
    async def test_a_description_passed_at_registration_is_served_as_given(self) -> None:
        """The full-text search tool is registered with a generated description, which replaces its docstring."""
        from app.tools.search.fts import fts_search_context

        generated = generate_fts_description('sqlite', 'english')
        mcp: FastMCP[None] = FastMCP(name='served-tools')
        with patch('app.tools.is_tool_disabled', return_value=False):
            assert register_tool(mcp, fts_search_context, description=generated) is True
        tool = await mcp.get_tool('fts_search_context')

        assert tool is not None
        assert tool.description == generated
        assert tool.description != inspect.getdoc(fts_search_context)

    @pytest.mark.asyncio
    async def test_list_threads_serves_the_explanation_of_its_fields(self) -> None:
        """Text that follows a docstring section, such as the field list of list_threads, is served too."""
        description = await _served_description('list_threads')

        assert 'Fields explained:' in description
        assert '- source_types: Number of distinct sources' in description


class TestGenerateFtsDescription:
    """Test the public generate_fts_description dispatcher."""

    def test_sqlite_english(self) -> None:
        result = generate_fts_description('sqlite', 'english')
        assert 'SQLite' in result
        assert 'FTS5' in result
        assert 'english' in result
        assert 'Stemming: ENABLED' in result
        assert 'Stop words: DISABLED' in result

    def test_sqlite_german(self) -> None:
        result = generate_fts_description('sqlite', 'german')
        assert 'SQLite' in result
        assert 'FTS5' in result
        assert 'Stemming: DISABLED' in result
        assert 'Stop words: DISABLED' in result

    def test_postgresql_english(self) -> None:
        result = generate_fts_description('postgresql', 'english')
        assert 'PostgreSQL' in result
        assert 'tsvector' in result
        assert 'english' in result
        assert 'Stemming: ENABLED' in result
        assert 'Stop words: ENABLED' in result

    def test_postgresql_german(self) -> None:
        result = generate_fts_description('postgresql', 'german')
        assert 'PostgreSQL' in result
        assert 'tsvector' in result
        assert 'Stemming: ENABLED' in result
        assert 'Stop words: ENABLED' in result
        assert 'Snowball' in result
        assert 'for german text' in result

    def test_postgresql_simple(self) -> None:
        result = generate_fts_description('postgresql', 'simple')
        assert 'Stemming: DISABLED' in result
        assert 'Stop words: DISABLED' in result
        assert 'simple configuration' in result


class TestSqliteFtsDescription:
    """Test SQLite-specific FTS description generation."""

    def test_english_stemming_enabled(self) -> None:
        result = _generate_sqlite_fts_description('english')
        assert 'ENABLED' in result
        assert 'Porter algorithm' in result
        assert '"running" WILL match "run"' in result

    def test_non_english_stemming_disabled(self) -> None:
        result = _generate_sqlite_fts_description('french')
        assert 'Stemming: DISABLED' in result
        assert 'Porter stemmer is English-only' in result
        assert 'does NOT match' in result

    def test_prefix_note_matches_how_a_hyphenated_word_is_searched(self) -> None:
        """The hyphen note describes the prefix query the server builds: each part a prefix, both required."""
        assert transform_query_sqlite('full-text', 'prefix') == '"full"* "text"*'
        result = _generate_sqlite_fts_description('english')
        assert 'Hyphenated words are split: "full-text" becomes "full* AND text*"' in result
        assert 'adjacency' not in result


class TestSqliteDescriptionExamplesOnARealIndex:
    """The examples of the SQLite description hold on an FTS5 table built with the server's tokenizer."""

    @staticmethod
    def _matching(
        language: str, mode: Literal['match', 'prefix', 'phrase', 'boolean'], query: str, documents: list[str],
    ) -> list[str]:
        """Return the documents the server's query for this mode matches, in insertion order."""
        database = sqlite3.connect(':memory:')
        try:
            database.execute(
                f"CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='{desired_sqlite_fts_tokenizer(language)}')",
            )
            database.executemany('INSERT INTO docs(body) VALUES (?)', [(document,) for document in documents])
            expression = transform_query_sqlite(query, mode, language)
            rows = database.execute('SELECT body FROM docs WHERE docs MATCH ? ORDER BY rowid', (expression,))
            return [row[0] for row in rows]
        finally:
            database.close()

    def test_the_english_stemming_example(self) -> None:
        assert '"running" WILL match "run" and "runs", but not "runner"' in _generate_sqlite_fts_description('english')
        assert self._matching('english', 'match', 'running', ['run', 'runs', 'runner']) == ['run', 'runs']

    def test_match_mode_finds_stemmed_forms_only_where_stemming_is_on(self) -> None:
        for language in ('english', 'german'):
            description = _generate_sqlite_fts_description(language)
            assert 'finds docs containing BOTH words (stemmed forms when Stemming is ENABLED)' in description
            assert 'exact words' not in description
        documents = ['python async', 'pythons asyncs', 'python only']
        assert self._matching('english', 'match', 'python async', documents) == ['python async', 'pythons asyncs']
        assert self._matching('german', 'match', 'python async', documents) == ['python async']

    def test_phrase_mode_keeps_the_word_order(self) -> None:
        description = _generate_sqlite_fts_description('english')
        assert '"exact phrase" matches "exact phrase", not "phrase exact"' in description
        assert 'matches only' not in description
        documents = ['exact phrase', 'exact phrases', 'phrase exact', 'an exact and phrase']
        assert self._matching('english', 'phrase', 'exact phrase', documents) == ['exact phrase', 'exact phrases']
        assert self._matching('german', 'phrase', 'exact phrase', documents) == ['exact phrase']

    def test_a_hyphenated_prefix_query_requires_both_parts_anywhere(self) -> None:
        documents = ['full text', 'the text is fully there', 'full only']
        assert self._matching('english', 'prefix', 'full-text', documents) == ['full text', 'the text is fully there']


class TestPostgresqlFtsDescription:
    """Test PostgreSQL-specific FTS description generation."""

    def test_the_english_examples_are_given_for_english_only(self) -> None:
        """Another language has its own stems and stop words, so its description claims no English example."""
        english = _generate_postgresql_fts_description('english')
        assert '"running" WILL match "run" and "runs", but not "runner"' in english
        assert '"the", "is", "a" are REMOVED and ignored in searches' in english
        for language in ('german', 'spanish'):
            other = _generate_postgresql_fts_description(language)
            assert '"running" WILL match' not in other
            assert '"the", "is", "a" are REMOVED' not in other
            assert f'Inflected forms of a {language} word match each other' in other
            assert f'The most common {language} words are REMOVED and ignored in searches' in other

    def test_no_mode_promises_stemming_where_it_is_off(self) -> None:
        simple = _generate_postgresql_fts_description('simple')
        assert 'Stemming: DISABLED' in simple
        assert 'query with stemming' not in simple
        assert 'when Stemming is ENABLED' in simple

    def test_boolean_mode_states_that_parentheses_do_not_group(self) -> None:
        result = _generate_postgresql_fts_description('english')
        assert 'Parentheses for grouping' not in result
        assert 'Grouping: none. Parentheses are ignored' in result
        assert '"python (async or await) -blocking"' not in result

    def test_the_phrase_example_keeps_a_stop_word_inside_the_phrase(self) -> None:
        result = _generate_postgresql_fts_description('english')
        assert '"lord of the rings"' in result
        assert 'over the rainbow' not in result

    def test_non_simple_stemming_enabled(self) -> None:
        result = _generate_postgresql_fts_description('spanish')
        assert 'Stemming: ENABLED' in result
        assert 'Snowball' in result
        assert 'for spanish text' in result
        assert 'Stop words: ENABLED' in result


class TestDescriptionStructure:
    """Test structural invariants of description output."""

    def test_sqlite_description_contains_all_four_modes(self) -> None:
        result = generate_fts_description('sqlite', 'english')
        for mode in ('match', 'prefix', 'phrase', 'boolean'):
            assert mode in result, f'{mode} missing in sqlite description'

    def test_postgresql_description_contains_all_four_modes(self) -> None:
        result = generate_fts_description('postgresql', 'english')
        for mode in ('match', 'prefix', 'phrase', 'boolean'):
            assert mode in result, f'{mode} missing in postgresql description'

    def test_mode_description_strings_contain_all_modes(self) -> None:
        for desc_name, desc in [
            ('_SQLITE_MODE_DESCRIPTIONS', _SQLITE_MODE_DESCRIPTIONS),
            ('_POSTGRESQL_MODE_DESCRIPTIONS', _POSTGRESQL_MODE_DESCRIPTIONS),
        ]:
            assert isinstance(desc, str)
            assert len(desc) > 0
            for mode in ('match', 'prefix', 'phrase', 'boolean'):
                assert mode in desc, f'{mode} missing in {desc_name}'

    def test_template_contains_all_format_fields(self) -> None:
        expected_fields = [
            '{backend}', '{engine}', '{language}',
            '{stemming_status}', '{stemming_detail}', '{stemming_example}',
            '{stopwords_status}', '{stopwords_detail}', '{stopwords_example}',
            '{non_language_behavior}', '{mode_descriptions}', '{fts_score_scope}',
        ]
        for field in expected_fields:
            assert field in _FTS_DESCRIPTION_TEMPLATE, f'{field} missing in template'

    def test_description_lists_both_metadata_filter_params(self) -> None:
        """Both backends advertise the working `metadata` simple-equality filter, not only metadata_filters."""
        for backend in ('sqlite', 'postgresql'):
            result = generate_fts_description(backend, 'english')
            assert '- metadata: Simple key=value equality matching' in result, (
                f'{backend} FTS description omits the simple `metadata` filter param'
            )
            assert 'metadata_filters' in result


class TestDescriptionAccessScoping:
    """The description states the caller's read scope and how each backend's score relates to it."""

    def test_both_backends_limit_results_to_readable_entries(self) -> None:
        for backend in ('sqlite', 'postgresql'):
            result = generate_fts_description(backend, 'english')
            assert 'Only entries the caller may read are matched, returned and counted.' in result, (
                f'{backend} FTS description does not state the caller read scope'
            )

    def test_sqlite_names_the_whole_index_bm25_statistics(self) -> None:
        result = generate_fts_description('sqlite', 'english')
        assert 'statistics of the whole FTS5 index, entries the caller cannot read included' in result
        assert 'the set of matching entries and the number of results per page depend only on what the caller can read' in (
            result
        )

    def test_postgresql_scores_each_entry_on_its_own(self) -> None:
        result = generate_fts_description('postgresql', 'english')
        assert 'ts_rank_cd scores each entry on its own' in result
        assert 'whole FTS5 index' not in result

    @pytest.mark.asyncio
    async def test_hybrid_names_the_whole_index_bm25_statistics(self) -> None:
        """The served hybrid description carries the SQLite score note its fts_score inherits from bm25()."""
        from app.tools.search.hybrid import hybrid_search_context

        mcp: FastMCP[None] = FastMCP(name='hybrid-description')
        assert register_tool(mcp, hybrid_search_context) is True
        tool = await mcp.get_tool('hybrid_search_context')
        assert tool is not None
        assert tool.description is not None
        # The description is the tool docstring, wrapped across lines; a client reads it as prose.
        description = ' '.join(tool.description.split())
        assert 'On SQLite the BM25 score draws on statistics of the whole FTS5 index' in description
        assert 'entries the caller cannot read included' in description
        assert 'the set of matching entries and the number of results per page depend only on what the caller can read' in (
            description
        )
        assert "PostgreSQL's ts_rank_cd scores each entry on its own" in description


class TestDescriptionReadBack:
    """Every entry-carrying search and read tool names the visibility, ownership and generation state its results report."""

    @pytest.mark.parametrize('backend', ['sqlite', 'postgresql'])
    def test_fts_description_names_the_access_fields(self, backend: Literal['sqlite', 'postgresql']) -> None:
        """The served full-text description carries its own return description, which names both fields."""
        description = ' '.join(generate_fts_description(backend, 'english').split())
        assert 'visibility' in description
        assert 'is_owner (whether the caller owns the entry)' in description

    @pytest.mark.parametrize(
        'tool_name',
        ['semantic_search_context', 'hybrid_search_context', 'search_context', 'get_context_by_ids'],
    )
    @pytest.mark.asyncio
    async def test_served_return_description_names_the_access_fields(self, tool_name: str) -> None:
        """The served return description of each docstring-described tool names both fields and never the owner."""
        description = await _served_description(tool_name)
        returns = ' '.join(description[description.index('Returns:'):description.index('Raises:')].split())
        assert 'visibility' in returns
        assert 'is_owner' in returns
        assert 'whether the caller owns the entry' in returns
        assert 'owner_id' not in returns

    @pytest.mark.parametrize('backend', ['sqlite', 'postgresql'])
    def test_fts_description_names_the_generation_state(self, backend: Literal['sqlite', 'postgresql']) -> None:
        """The served full-text description names generation_status and each state it takes."""
        description = ' '.join(generate_fts_description(backend, 'english').split())
        assert 'generation_status' in description
        for state in ("'ready'", "'pending'", "'failed'"):
            assert state in description

    @pytest.mark.parametrize(
        'tool_name',
        [
            'semantic_search_context', 'hybrid_search_context', 'search_context', 'get_context_by_ids',
            'grep_context', 'navigate_context', 'read_context_range',
        ],
    )
    @pytest.mark.asyncio
    async def test_served_return_description_names_the_generation_state(self, tool_name: str) -> None:
        """The served return description of each entry-carrying tool names generation_status and its states."""
        description = await _served_description(tool_name)
        returns = ' '.join(description[description.index('Returns:'):description.index('Raises:')].split())
        assert 'generation_status' in returns
        for state in ("'ready'", "'pending'", "'failed'"):
            assert state in returns

    @pytest.mark.asyncio
    async def test_statistics_return_description_names_the_generation_block(self) -> None:
        """The served get_statistics description names the generation block and counts it per caller."""
        description = ' '.join((await _served_description('get_statistics')).split())
        returns = description[description.index('Returns:'):description.index('Raises:')]
        assert 'generation (dict with pending' in returns
        assert 'and failed' in returns
        assert 'summary, node and generation figures count only the entries the caller may read' in description

    @pytest.mark.asyncio
    async def test_statistics_return_description_names_every_top_level_field(self) -> None:
        """An agent picks a tool by its field list, so the list names every field the response can carry."""
        description = await _served_description('get_statistics')
        top_level = description[description.index('Returns:'):description.index('Raises:')]
        # Drop the parenthesized explanations, so a name inside a nested block does not count for a top-level field.
        while (stripped := re.sub(r'\([^()]*\)', '', top_level)) != top_level:
            top_level = stripped
        names = set(re.findall(r'[a-z_]+', top_level))
        missing = sorted(set(StatisticsResponseDict.__annotations__) - names)
        assert not missing, f'the get_statistics return description omits {missing}'

    @pytest.mark.asyncio
    async def test_statistics_return_description_names_the_fields_of_each_block(self) -> None:
        """Each block described as 'dict with ...' names every field of its type but the reason an unavailable one adds."""
        description = await _served_description('get_statistics')
        returns = description[description.index('Returns:'):description.index('Raises:')]
        checked = 0
        for name, hint in typing.get_type_hints(StatisticsResponseDict).items():
            start = re.search(rf'\b{name} \(dict with\b', ' '.join(returns.split()))
            if start is None:
                continue
            text = ' '.join(returns.split())
            depth, end = 1, start.end()
            while depth:
                depth += {'(': 1, ')': -1}.get(text[end], 0)
                end += 1
            listed = set(re.findall(r'[a-z_]+', text[start.end():end]))
            missing = sorted(set(typing.get_type_hints(hint)) - listed - {'message'})
            assert not missing, f'the {name} block of the get_statistics return description omits {missing}'
            checked += 1
        assert checked == 8


class TestHighlightedIsConditional:
    """The full-text return description says when a result carries a highlighted snippet."""

    @pytest.mark.parametrize('backend', ['sqlite', 'postgresql'])
    def test_fts_description_marks_highlighted_as_conditional(self, backend: Literal['sqlite', 'postgresql']) -> None:
        description = ' '.join(generate_fts_description(backend, 'english').split())
        assert 'highlighted (set only when highlight=True)' in description


class TestRankedPaginationDescription:
    """Each ranked search tool tells a client that a page is a window into one ranking of bounded depth."""

    @staticmethod
    def _assert_states_the_ranked_window(description: str, label: str) -> None:
        prose = ' '.join(description.split())
        assert f'A query has ONE ranking, at most {RANKED_SEARCH_DEPTH} rows deep' in prose, (
            f'{label} does not state the depth of the ranking'
        )
        returns = description[description.index('Returns:'):description.index('Raises:')]
        assert 'rank_depth_limit' in returns, f'the {label} return description omits rank_depth_limit'

    @pytest.mark.parametrize('backend', ['sqlite', 'postgresql'])
    def test_fts_description_states_the_ranked_window(self, backend: Literal['sqlite', 'postgresql']) -> None:
        self._assert_states_the_ranked_window(generate_fts_description(backend, 'english'), f'{backend} full-text')

    @pytest.mark.parametrize('tool_name', ['semantic_search_context', 'hybrid_search_context'])
    @pytest.mark.asyncio
    async def test_served_description_states_the_ranked_window(self, tool_name: str) -> None:
        self._assert_states_the_ranked_window(await _served_description(tool_name), tool_name)


class TestWriteToolDescriptions:
    """Every write tool tells a client how to defer generation and which generation state its response reports."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'tool_function', [store_context, update_context, store_context_batch, update_context_batch],
        ids=lambda function: function.__name__,
    )
    async def test_served_description_names_deferral_and_the_generation_state(
        self, tool_function: Callable[..., Awaitable[object]],
    ) -> None:
        tool = await _served_tool(tool_function.__name__)
        assert tool.description is not None
        description = ' '.join(tool.description.split())
        served = ' '.join(json.dumps({'description': tool.description, 'input': tool.parameters}).split())
        assert 'defer_generation' in served
        assert 'generation_status' in description
        for state in ("'ready'", "'pending'", "'failed'"):
            assert state in description

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool_function', [store_context, update_context], ids=lambda function: function.__name__)
    async def test_the_parameter_states_both_modes(self, tool_function: Callable[..., Awaitable[object]]) -> None:
        """The parameter's served description says what each value does and when pending turns ready or failed."""
        tool = await _served_tool(tool_function.__name__)
        description = ' '.join(tool.parameters['properties']['defer_generation']['description'].split())
        assert 'False (default)' in description
        assert 'True:' in description
        assert "'pending'" in description
        assert "'ready'" in description
        assert "'failed'" in description

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool_name', ['store_context', 'update_context'])
    async def test_the_parameter_names_the_failures_that_refuse_the_write(self, tool_name: str) -> None:
        """A failed section summary never refuses a write, so the parameter names the failures that do."""
        tool = await _served_tool(tool_name)
        description = ' '.join(tool.parameters['properties']['defer_generation']['description'].split())
        assert 'an embedding or summary failure' in description
        assert 'a section summary that fails is left out' in description
        assert 'a generation failure' not in description


_INTERNAL_NAME = re.compile(r'\bapp\.|MAX_PLAN_DELETES|content_hash')


class TestSyncToolSchemas:
    """The corpus sync tools' served schemas tell a client what to send, in the client's terms."""

    def test_manifest_item_description_is_client_facing(self) -> None:
        """The item's served description names no server module and carries no field list of its own."""
        tool = Tool.from_function(plan_context_sync)
        description = tool.parameters['$defs']['SyncManifestItem']['description']
        assert 'app.' not in description
        assert 'Attributes:' not in description
        assert '{source_id, keep: true}' in description

    def test_text_digest_states_the_whitespace_rule(self) -> None:
        """A client learns from the schema alone that the text is hashed after surrounding whitespace is removed."""
        tool = Tool.from_function(plan_context_sync)
        properties = tool.parameters['$defs']['SyncManifestItem']['properties']
        description = properties['text_digest']['description']
        assert 'whitespace' in description
        assert 'SHA-256' in description
        assert 'str.strip()' in description

    def test_every_manifest_item_field_is_described(self) -> None:
        tool = Tool.from_function(plan_context_sync)
        properties = tool.parameters['$defs']['SyncManifestItem']['properties']
        assert set(properties) == {'source_id', 'text_digest', 'meta_digest', 'keep'}
        for name, schema in properties.items():
            assert schema.get('description'), f'{name} carries no served description'

    @pytest.mark.asyncio
    async def test_plan_description_states_the_whitespace_rule(self) -> None:
        # The description is the tool docstring, wrapped across lines; a client reads it as prose.
        description = ' '.join((await _served_description('plan_context_sync')).split())
        assert 'digest of the text it would store after surrounding whitespace is removed' in description

    @pytest.mark.parametrize('tool_name', ['plan_context_sync', 'list_ingest_state'])
    @pytest.mark.asyncio
    async def test_served_schemas_name_no_server_identifier(self, tool_name: str) -> None:
        """Neither the description nor the input or output schema names a server module, constant or column."""
        tool = await _served_tool(tool_name)
        assert tool.output_schema is not None
        served = json.dumps({'description': tool.description, 'input': tool.parameters, 'output': tool.output_schema})
        assert _INTERNAL_NAME.search(served) is None, _INTERNAL_NAME.search(served)

    def test_plan_output_states_the_listed_delete_bound(self) -> None:
        tool = Tool.from_function(plan_context_sync)
        assert tool.output_schema is not None
        description = ' '.join(tool.output_schema['description'].split())
        assert f'The first {MAX_PLAN_DELETES} planned deletes' in description
