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

import json
import re
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Literal

import pytest
from fastmcp import FastMCP
from fastmcp.tools import Tool

from app.ingest import MAX_PLAN_DELETES
from app.tools import get_context_by_ids
from app.tools import get_statistics
from app.tools import list_ingest_state
from app.tools import plan_context_sync
from app.tools import register_tool
from app.tools import search_context
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


class TestPostgresqlFtsDescription:
    """Test PostgreSQL-specific FTS description generation."""

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
    def test_docstring_return_description_names_the_access_fields(self, tool_name: str) -> None:
        """The return description of each docstring-described tool names both fields and never the owner."""
        from app.tools.search.hybrid import hybrid_search_context
        from app.tools.search.semantic import semantic_search_context

        functions: dict[str, Callable[..., Awaitable[object]]] = {
            'semantic_search_context': semantic_search_context,
            'hybrid_search_context': hybrid_search_context,
            'search_context': search_context,
            'get_context_by_ids': get_context_by_ids,
        }
        docstring = functions[tool_name].__doc__
        assert docstring is not None
        returns = ' '.join(docstring[docstring.index('Returns:'):docstring.index('Raises:')].split())
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
    def test_docstring_return_description_names_the_generation_state(self, tool_name: str) -> None:
        """The return description of each docstring-described entry-carrying tool names generation_status and its states."""
        from app.tools.navigation.grep import grep_context
        from app.tools.navigation.navigate import navigate_context
        from app.tools.navigation.read_range import read_context_range
        from app.tools.search.hybrid import hybrid_search_context
        from app.tools.search.semantic import semantic_search_context

        functions: dict[str, Callable[..., Awaitable[object]]] = {
            'semantic_search_context': semantic_search_context,
            'hybrid_search_context': hybrid_search_context,
            'search_context': search_context,
            'get_context_by_ids': get_context_by_ids,
            'grep_context': grep_context,
            'navigate_context': navigate_context,
            'read_context_range': read_context_range,
        }
        docstring = functions[tool_name].__doc__
        assert docstring is not None
        returns = ' '.join(docstring[docstring.index('Returns:'):docstring.index('Raises:')].split())
        assert 'generation_status' in returns
        for state in ("'ready'", "'pending'", "'failed'"):
            assert state in returns

    def test_statistics_return_description_names_the_generation_block(self) -> None:
        """The get_statistics description names the generation block and counts it per caller."""
        docstring = get_statistics.__doc__
        assert docstring is not None
        description = ' '.join(docstring.split())
        returns = description[description.index('Returns:'):description.index('Raises:')]
        assert 'generation (dict with pending' in returns
        assert 'and failed' in returns
        assert 'summary, node and generation figures count only the entries the caller may read' in description


class TestWriteToolDescriptions:
    """Every write tool tells a client how to defer generation and which generation state its response reports."""

    @pytest.mark.parametrize(
        'tool_function', [store_context, update_context, store_context_batch, update_context_batch],
        ids=lambda function: function.__name__,
    )
    def test_served_description_names_deferral_and_the_generation_state(
        self, tool_function: Callable[..., Awaitable[object]],
    ) -> None:
        tool = Tool.from_function(tool_function)
        assert tool.description is not None
        description = ' '.join(tool.description.split())
        served = ' '.join(json.dumps({'description': tool.description, 'input': tool.parameters}).split())
        assert 'defer_generation' in served
        assert 'generation_status' in description
        for state in ("'ready'", "'pending'", "'failed'"):
            assert state in description

    @pytest.mark.parametrize('tool_function', [store_context, update_context], ids=lambda function: function.__name__)
    def test_the_parameter_states_both_modes(self, tool_function: Callable[..., Awaitable[object]]) -> None:
        """The parameter's served description says what each value does and when pending turns ready or failed."""
        tool = Tool.from_function(tool_function)
        description = ' '.join(tool.parameters['properties']['defer_generation']['description'].split())
        assert 'False (default)' in description
        assert 'True:' in description
        assert "'pending'" in description
        assert "'ready'" in description
        assert "'failed'" in description


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

    def test_plan_description_states_the_whitespace_rule(self) -> None:
        tool = Tool.from_function(plan_context_sync)
        assert tool.description is not None
        # The description is the tool docstring, wrapped across lines; a client reads it as prose.
        description = ' '.join(tool.description.split())
        assert 'digest of the text it would store after surrounding whitespace is removed' in description

    @pytest.mark.parametrize('tool_function', [plan_context_sync, list_ingest_state], ids=lambda f: f.__name__)
    def test_served_schemas_name_no_server_identifier(self, tool_function: Callable[..., Awaitable[object]]) -> None:
        """Neither the description nor the input or output schema names a server module, constant or column."""
        tool = Tool.from_function(tool_function)
        assert tool.output_schema is not None
        served = json.dumps({'description': tool.description, 'input': tool.parameters, 'output': tool.output_schema})
        assert _INTERNAL_NAME.search(served) is None, _INTERNAL_NAME.search(served)

    def test_plan_output_states_the_listed_delete_bound(self) -> None:
        tool = Tool.from_function(plan_context_sync)
        assert tool.output_schema is not None
        description = ' '.join(tool.output_schema['description'].split())
        assert f'The first {MAX_PLAN_DELETES} planned deletes' in description
