"""Default server instructions for MCP Context Server.

Provides the DEFAULT_INSTRUCTIONS constant used when MCP_SERVER_INSTRUCTIONS
environment variable is not set. These instructions are sent to MCP clients
during initialization via the MCP protocol's instructions field.
"""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.settings.server import InstructionsSettings

# Default server instructions sent to MCP clients during initialization.
# Override at runtime via MCP_SERVER_INSTRUCTIONS environment variable.
DEFAULT_INSTRUCTIONS: str = ('''
# MCP Context Server

Persistent context storage for LLM agents. Store, search, and retrieve context entries across sessions.

## Tools

| Tool                      | Purpose                                                      |
|---------------------------|--------------------------------------------------------------|
| `store_context`           | Store new entry                                              |
| `get_context_by_ids`      | Retrieve full entries by ID                                  |
| `search_context`          | Browse/filter entries (truncated text + summary)             |
| `hybrid_search_context`   | Combined FTS + semantic (truncated text + summary)           |
| `semantic_search_context` | Vector similarity search (truncated text + summary)          |
| `fts_search_context`      | Full-text linguistic search (truncated text + summary)       |
| `update_context`          | Update existing entry                                        |
| `delete_context`          | Delete entries by ID                                         |
| `list_threads`            | List all thread IDs                                          |
| `get_statistics`          | Database statistics                                          |
| `grep_context`            | Locate literal/regex matches across entries                  |
| `navigate_context`        | Markdown outline (index_tree) of one entry                   |
| `read_context_range`      | Read a char/line/node slice of one entry                     |
| `plan_context_sync`       | Diff a corpus manifest against stored ingest state           |
| `list_ingest_state`       | Page through your stored ingest state                        |
| `share_context`           | Share your entry with a user or group (read/write)           |
| `unshare_context`         | Remove a grant from your entry                               |
| `list_context_grants`     | List the grants on your entry                                |

Batch: `store_context_batch`, `update_context_batch`, `delete_context_batch` (up to 100 entries each).

## Core Concepts

- **thread_id**: Groups related entries (e.g., one per knowledge base, project, session, or task). Scope searches with thread_id when working within a specific thread; omit for cross-thread knowledge discovery.
- **source**: "user" or "agent" — filters by entry creator.
- **text**: Prefer storing context in pure Markdown format.
- **metadata**: JSON object for structured data. Use metadata_filters for advanced operators (gt, lt, contains, exists, etc.). Include descriptive fields (agent_name, task_name, status, project, references) for discoverability. Link entries into a knowledge graph with references. Store cross-references when creating entries: `metadata: {"references": {"context_ids": ["0190abcdef1234567890abcdef123456", "0190abcdef1234567890abcdef987654"]}}`. Retrieve referenced entries with `get_context_by_ids` to follow the knowledge chain.
- **tags**: Lowercase labels for categorization (OR logic).
- **Access**: entries are private to their owner unless public or shared; results carry `visibility` and `is_owner`, and only an entry's owner shares it.
- **Knowledge bases**: corpora live in threads named `kb:<corpus>`; read them with the search and navigation tools scoped by thread_id; the `ingest` metadata block is maintained by the sync client.

## Best Practices

**Searching:**
- All search tools return truncated `text_content` and a `summary` field. Use `get_context_by_ids` to retrieve full content of entries that look relevant based on the truncated preview, summary, and metadata.
- Use `search_context` to browse and list entries within a thread (e.g., fetch recent messages by source filter).
- Use `hybrid_search_context` for knowledge discovery — finding past decisions, patterns, and related work, typically cross-thread (e.g., by project metadata filter or across all projects).
- These tools complement each other: `search_context` for thread-scoped context loading, `hybrid_search_context` for cross-thread knowledge search.
- Use `fts_search_context` for precise linguistic queries only (stemming, boolean operators).
- Use `semantic_search_context` for meaning-based similarity.
- Prefer `hybrid_search_context` over `fts_search_context` and `semantic_search_context` when available.

**Storing:**
- Always include descriptive metadata (agent_name, task_name, status, project, references) for discoverability.
- Add tags for cross-cutting categorization.
- Add `metadata.references.context_ids` to metadata linking to entries your work builds upon.
- Use `metadata_patch` (not `metadata`) in `update_context` to preserve fields you do not change.
- Set `defer_generation: true` (per entry in the batch tools) when you need not wait for embeddings and summaries: the entry is stored at once, readable and full-text searchable, and its `generation_status` in write and read results stays `pending` until background generation finishes (`ready`, from then on semantic search finds it) or gives up (`failed`).

**Retrieving:**
- All search tools return truncated `text_content` with `is_text_content_truncated` flag, plus a `summary` field (empty string when not generated). You can query many entries without hurting the context window. Always call `get_context_by_ids` for full content of relevant entries.
- Check `metadata.references.context_ids` in retrieved entries and follow them with `get_context_by_ids` for deeper context.

## Skill Integration

If you have access to Skills, check their descriptions for any that relate to context storage, context retrieval, context preservation, or working with a context server. Read and apply all matching Skills to extend these default instructions with richer patterns for storing, searching, and retrieving context. If a description is ambiguous, read the full Skill content to determine relevance. Skills are authored by users or teams to customize how you interact with this server — always apply all that are relevant.
''').strip()


def resolve_instructions(instructions_settings: 'InstructionsSettings') -> str:
    """Resolve the effective instructions text.

    If MCP_SERVER_INSTRUCTIONS env var is set (including empty string),
    uses that value. Otherwise, returns DEFAULT_INSTRUCTIONS.

    Args:
        instructions_settings: The InstructionsSettings instance.

    Returns:
        The resolved instructions text.
    """
    if instructions_settings.server_instructions is not None:
        return instructions_settings.server_instructions
    return DEFAULT_INSTRUCTIONS
