"""Tests for MCP server instructions support."""

from tests.helpers import env_var


class TestDefaultInstructions:
    """Tests for DEFAULT_INSTRUCTIONS constant in app/instructions.py."""

    def test_default_instructions_is_non_empty_string(self) -> None:
        """DEFAULT_INSTRUCTIONS must be a non-empty string."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert isinstance(DEFAULT_INSTRUCTIONS, str)
        assert len(DEFAULT_INSTRUCTIONS) > 0

    def test_default_instructions_contains_server_purpose(self) -> None:
        """DEFAULT_INSTRUCTIONS must describe what the server does."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'context' in DEFAULT_INSTRUCTIONS.lower()
        assert 'agent' in DEFAULT_INSTRUCTIONS.lower()

    def test_default_instructions_contains_search_tools(self) -> None:
        """DEFAULT_INSTRUCTIONS must mention key search tools."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'hybrid_search_context' in DEFAULT_INSTRUCTIONS
        assert 'search_context' in DEFAULT_INSTRUCTIONS
        assert 'get_context_by_ids' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_contains_store_tool(self) -> None:
        """DEFAULT_INSTRUCTIONS must mention store_context."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'store_context' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_mentions_thread_id(self) -> None:
        """DEFAULT_INSTRUCTIONS must explain thread_id concept."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'thread_id' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_mentions_metadata(self) -> None:
        """DEFAULT_INSTRUCTIONS must explain metadata concept."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'metadata' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_mentions_references(self) -> None:
        """DEFAULT_INSTRUCTIONS must explain references for knowledge graph."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'references' in DEFAULT_INSTRUCTIONS.lower()

    def test_default_instructions_character_budget(self) -> None:
        """DEFAULT_INSTRUCTIONS should be within reasonable character budget (~2000-6300 chars)."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 2000 < len(DEFAULT_INSTRUCTIONS) < 6300

    def test_default_instructions_lists_navigation_tools(self) -> None:
        """DEFAULT_INSTRUCTIONS must list the v3 navigation tools so clients discover them."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'grep_context' in DEFAULT_INSTRUCTIONS
        assert 'navigate_context' in DEFAULT_INSTRUCTIONS
        assert 'read_context_range' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_lists_sync_tools_and_knowledge_bases(self) -> None:
        """DEFAULT_INSTRUCTIONS lists both sync tools and names the kb: thread convention."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert '| `plan_context_sync`       |' in DEFAULT_INSTRUCTIONS
        assert '| `list_ingest_state`       |' in DEFAULT_INSTRUCTIONS
        assert '`kb:<corpus>`' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_list_sharing_tools_and_the_access_concept(self) -> None:
        """DEFAULT_INSTRUCTIONS lists the three sharing tools and states who reads and shares an entry."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert '| `share_context`           | Share your entry with a user or group (read/write)' in DEFAULT_INSTRUCTIONS
        assert '| `unshare_context`         | Remove a grant from your entry' in DEFAULT_INSTRUCTIONS
        assert '| `list_context_grants`     | List the grants on your entry' in DEFAULT_INSTRUCTIONS
        assert (
            "- **Access**: entries are private to their owner unless public or shared; results carry "
            "`visibility` and `is_owner`, and only an entry's owner shares it."
        ) in DEFAULT_INSTRUCTIONS

    def test_default_instructions_explain_deferred_generation(self) -> None:
        """DEFAULT_INSTRUCTIONS tells a storing agent how to defer generation and how to read its state."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert (
            '- Set `defer_generation: true` (per entry in the batch tools) when you need not wait for embeddings '
            'and summaries: the entry is stored at once, readable and full-text searchable, and its '
            '`generation_status` in write and read results stays `pending` until background generation finishes '
            '(`ready`, from then on semantic search finds it) or gives up (`failed`).'
        ) in DEFAULT_INSTRUCTIONS

    def test_default_instructions_tool_rows_share_the_column_widths(self) -> None:
        """Every row of the tool table is as wide as its header row."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        table = [line for line in DEFAULT_INSTRUCTIONS.splitlines() if line.startswith('|')]
        assert len(table) > 2
        assert {len(line) for line in table} == {len(table[0])}

    def test_default_instructions_is_static_constant(self) -> None:
        """DEFAULT_INSTRUCTIONS must be a simple string constant, not dynamically generated."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        # Verify it does not contain dynamic format markers or placeholders
        assert '{field_names}' not in DEFAULT_INSTRUCTIONS
        assert '{0}' not in DEFAULT_INSTRUCTIONS

    def test_default_instructions_mentions_skill_integration(self) -> None:
        """DEFAULT_INSTRUCTIONS must mention Skill integration for context server usage."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        lower = DEFAULT_INSTRUCTIONS.lower()
        assert 'skill' in lower
        assert 'context server' in lower or 'context storage' in lower


class TestInstructionsResolution:
    """Tests for instructions resolution logic (env var override vs default)."""

    def test_resolve_instructions_returns_default_when_no_env(self) -> None:
        """When MCP_SERVER_INSTRUCTIONS is not set, should use DEFAULT_INSTRUCTIONS."""
        from app.instructions import DEFAULT_INSTRUCTIONS
        from app.instructions import resolve_instructions
        from app.settings.server import InstructionsSettings

        settings = InstructionsSettings()
        result = resolve_instructions(settings)
        assert result == DEFAULT_INSTRUCTIONS

    def test_resolve_instructions_returns_env_override(self) -> None:
        """When MCP_SERVER_INSTRUCTIONS is set, should use env var value."""
        from app.instructions import resolve_instructions
        from app.settings.server import InstructionsSettings

        custom = 'Custom instructions text.'
        with env_var('MCP_SERVER_INSTRUCTIONS', custom):
            settings = InstructionsSettings()
            result = resolve_instructions(settings)
            assert result == custom

    def test_resolve_instructions_empty_string_disables(self) -> None:
        """Empty MCP_SERVER_INSTRUCTIONS should return empty string (effectively disables)."""
        from app.instructions import resolve_instructions
        from app.settings.server import InstructionsSettings

        with env_var('MCP_SERVER_INSTRUCTIONS', ''):
            settings = InstructionsSettings()
            result = resolve_instructions(settings)
            assert result == ''
