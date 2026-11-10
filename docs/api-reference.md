# API Reference

## Introduction

The MCP Context Server exposes 21 MCP tools for context management, organized into core operations, search tools, navigation tools (locate / navigate / extract), sync tools, sharing tools, and batch operations.

**Tool Categories:**
- **Core Operations**: `store_context`, `search_context`, `get_context_by_ids`, `delete_context`, `update_context`, `list_threads`, `get_statistics`
- **Search Tools**: `semantic_search_context`, `fts_search_context`, `hybrid_search_context`
- **Navigation Tools**: `grep_context`, `navigate_context`, `read_context_range`
- **Sync Tools**: `plan_context_sync`, `list_ingest_state`
- **Sharing Tools**: `share_context`, `unshare_context`, `list_context_grants`
- **Batch Operations**: `store_context_batch`, `update_context_batch`, `delete_context_batch`

## Context Identifier Format

Every context entry is identified by a UUIDv7 value. The canonical public form is a 32-character lowercase hexadecimal string with no hyphens (regex `^[0-9a-f]{32}$`).

Example: `0190abcdef1234567890abcdef123456`

**Accepted Input Formats:**

Every tool that accepts a context identifier accepts BOTH of the following at the parameter boundary:

- The canonical 32-character hex form (e.g., `0190abcdef1234567890abcdef123456`)
- The 36-character hyphenated UUID form (e.g., `0190abcd-ef12-3456-7890-abcdef123456`)

Whitespace is stripped and the value is folded to lowercase before validation. Storage canonicalizes to the 32-character lowercase hex form; the `context_id` returned in tool responses is always in canonical form regardless of input format.

**Prefix Lookup:**

Every tool parameter that accepts a context identifier ALSO accepts a hex prefix of 8 to 31 characters, including the bulk and batch list parameters (each element is resolved independently). The server resolves each prefix against the stored entries the caller may read, so an entry the caller cannot read never matches a prefix or makes one ambiguous:

- Exactly one match: resolves to that entry's full ID.
- Zero matches: returns an error `No context entry matches prefix '<prefix>'`.
- More than one match: returns an error `Ambiguous prefix '<prefix>' matches multiple entries`.

Prefixes shorter than 8 characters or containing non-hex characters are rejected at validation.

**Input Format Support by Tool:**

| Parameter Kind                                                                                                                                                                                           | Accepts Full UUID (32-hex / 36-hyphenated) | Accepts Prefix (8-31 hex)                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|-------------------------------------------|
| Single-ID parameters (`update_context.context_id`, `navigate_context.context_id`, `read_context_range.context_id`, and the `context_id` of `share_context`, `unshare_context` and `list_context_grants`) | Yes                                        | Yes                                       |
| Bulk / batch list parameters (`get_context_by_ids.context_ids`, `delete_context.context_ids`, `delete_context_batch.context_ids`, each `update_context_batch` entry's `context_id`)                      | Yes                                        | Yes (each element resolved independently) |

**Bulk Parameter Example:**

```python
get_context_by_ids(context_ids=[
    "0190abcdef1234567890abcdef123456",
    "0190abcd-ef12-3456-7890-abcdef987654"
])
```

## Access Scoping

Every tool call runs as one principal: the verified token subject under `MCP_AUTH_PROVIDER=jwt`, and `ACCESS_CONTROL_DEFAULT_PRINCIPAL` for STDIO, `none` and `simple_token`. Each entry has an owner (the principal that stored it), a `visibility` of `private` (owner plus every grantee) or `public` (everyone), and optional read or write grants for users or groups. The tools apply these rules on SQLite and PostgreSQL alike, and with `ENABLE_POSTGRESQL_RLS=true` PostgreSQL also enforces them in the database; the [Access Model](authentication.md#access-model) explains them in full.

- **Reads**: `get_context_by_ids` omits an ID the caller may not read exactly as it omits a missing one. `search_context`, `semantic_search_context`, `fts_search_context`, `hybrid_search_context` and `grep_context` match, return and count only readable entries, so hidden entries never add to `count`, `total_matches`, `truncated` or the `rows_returned` explain statistic, never consume `grep_context`'s `max_entries_scanned` budget, and never take a slot on a page. With `explain_query=True` on PostgreSQL, the returned query plan still carries row estimates from table-wide planner statistics. With `ENABLE_POSTGRESQL_RLS=true`, these plans also show the read view's copy of the access condition beside the tool's own, the view's role gate in `InitPlan` subplans, and the row-level security policy conditions on the base tables the query reads, such as `tags` (see [Row-level security (PostgreSQL)](database-backends.md#row-level-security-postgresql)). `navigate_context` and `read_context_range` report an unreadable entry as `Context entry not found: {id}`. ID prefixes resolve over readable entries only.
- **Read-back**: every entry a read tool returns carries its `visibility` (`private` or `public`) and `is_owner`, a boolean that is true when the caller owns the entry: each entry of `get_context_by_ids` and of the four search tools, each `grep_context` result, and the top level of the `navigate_context` and `read_context_range` responses. `is_owner` is false for another principal's `public` entry and for an entry reached through a read or write grant, so it tells a client which entries it may share, publish or delete. No response reports the owner's identity, `granted_by` or the grants of an entry, except `list_context_grants` for the owner. The sync tools carry neither field: they list only the caller's own entries, and the visibility is part of each entry's `meta_digest`. The same entries and results also carry `generation_status` (see [Generation Status](#generation-status)), which every principal that may read the entry sees.
- **Discovery**: `list_threads` lists a thread only when the caller may read at least one of its entries, and computes every figure from those entries. `get_statistics` counts only readable entries in every entry, thread, image, tag, embedding, index, summary, node and generation figure; `database_size_mb`, `embeddings_size_mb`, `embeddings_size_estimated`, `connection_metrics` and the configuration fields describe the whole deployment and are the same for every caller.
- **Updates**: the owner and principals holding a write grant may change text, metadata, tags and images; only the owner may change `visibility`. An entry the caller may not read is reported as not found (`Context entry with ID {id} not found`; in `update_context_batch`, `Context entry {id} not found`). An entry the caller may read but not modify (through a read grant, or a `public` entry it does not own) is refused with `Not authorized to modify context entry with ID {id}` (in `update_context_batch`, `Not authorized to modify context entry {id}`) for every change, a visibility change included. A visibility change by a write grantee who is not the owner is refused with `Only the owner may change the visibility of context {id}`. In atomic mode the not-found and not-authorized batch messages end with "at index {i}" and the whole batch fails.
- **Deletes**: only the owner may delete an entry. A delete that names IDs (`delete_context` with `context_ids`, `delete_context_batch` with `context_ids`) is refused with `Not authorized to delete context entries: {ids}` and deletes nothing when a named entry the caller may read is not its own; a named ID the caller may not read is skipped and not counted. A thread or criteria delete (`delete_context` with `thread_id`, `delete_context_batch` without `context_ids`) deletes only the caller's own matching entries and leaves the rest in place without an error. Deleting an entry deletes its grants.
- **Sharing**: only the owner of an entry may share it, unshare it or list its grants, with `share_context`, `unshare_context` and `list_context_grants`; a write grant never allows sharing. An entry the caller may not read is reported as `Context entry with ID {id} not found`, and one it may read but does not own is refused explicitly (see [Sharing Tools](#sharing-tools)). Grants change neither an entry's `updated_at` nor the version `update_context` checks, and no update tool adds or removes grants.
- **Read-only principal**: when `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` names the caller, every tool that writes (`store_context`, `store_context_batch`, `update_context`, `update_context_batch`, `delete_context`, `delete_context_batch`, `share_context`, `unshare_context`) fails with `The anonymous principal is read-only: {tool} is not permitted` before it checks any argument; the read tools serve that principal like any other (see [Anonymous Read Access](authentication.md#anonymous-read-access)).
- **Sync**: `plan_context_sync` plans only the caller's own managed entries. A managed entry another principal owns that the caller may read is never planned for an update or a delete and only adds to `foreign_count`; one the caller may not read is not counted, so a manifest item for its source plans a create. `list_ingest_state` lists only the caller's own managed entries, and another principal's entries never take a slot on a page.
- **Deduplication**: when `deduplicate` is true (the default), `store_context` and `store_context_batch` take as the candidate the latest entry of the same thread and source among the entries the caller may read, and merge into it only when the caller owns it and its text is identical; otherwise they insert a new entry owned by the caller. Only opposite-source entries the caller may read mark a new conversational turn. With `deduplicate` false they look for no candidate and always insert a new entry owned by the caller.
- **Full-text scores on SQLite**: `fts_score` comes from FTS5 `bm25()`, which uses statistics of the whole index, entries the caller cannot read included, so the scores of readable entries and their relative order can shift as other principals' entries change; the matching set and page sizes depend only on readable entries. The same score feeds the full-text leg of `hybrid_search_context`. PostgreSQL's `ts_rank_cd` scores each entry on its own.

## Generation Status

The write tools generate each entry's embeddings, its summary and the summaries of its Markdown sections (see [Semantic Search](semantic-search.md#generation-modes) and [Summary Generation](summary-generation.md)). By default a write generates them before it commits: an embedding or summary failure stores nothing, while a section summary that fails is left out and the entry is stored without it. A write that sets `defer_generation` to true (the parameter of `store_context` and `update_context`, or the key of the same name on an entry of `store_context_batch` or `update_context_batch`) commits the entry at once together with a generation job, and the server's generation worker generates the outputs afterward. You can read the entry as soon as the write returns, and `search_context`, `fts_search_context`, `grep_context` and the full-text leg of `hybrid_search_context` find it from then on; `semantic_search_context` and the semantic leg of `hybrid_search_context` find it once the job has stored its embeddings. A server with neither an embedding nor a summary provider has nothing to generate, so a deferring write there is an ordinary write and reports `ready`.

Every store and update response, every entry the read tools return and every `grep_context` result carries `generation_status`:

| Value     | Meaning                                                                                                                                                                                                                                                |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ready`   | The entry has no outstanding generation job: its outputs are complete, or the server's settings call for none                                                                                                                                          |
| `pending` | The entry's generation job waits or runs; the outputs it generates are absent until it completes                                                                                                                                                       |
| `failed`  | The job used all of its `GENERATION_JOB_MAX_ATTEMPTS` attempts (3 by default). The entry stays readable and full-text searchable, and the job runs again at the server's next start or in `mcp-context-server-migrate --embed-missing` or `--re-embed` |

- A write tool reports the entry's state after its write. A store that deduplicates into an existing entry reports that entry's state, a text update without deferral generates the new text's outputs and reports `ready`, and an update without `text` leaves the state as it was.
- No tool reports why a job failed. The server logs every failed attempt, the last one at ERROR, and keeps the last error with the job (see [Monitoring Deferred Generation](semantic-search.md#monitoring-deferred-generation)).
- A completing job stores its outputs and advances the entry's `updated_at`; the version `update_context` checks stays as it was. A text update made while the job runs supersedes it: the worker discards what it generated for the old text, and a deferring update queues a new job.
- `get_statistics` counts the caller's readable entries whose generation is pending or failed in its `generation` block.

## Core Tools

### store_context

Store a context entry with optional images and flexible metadata.

**Parameters:**
- `thread_id` (str, required): Unique identifier for the conversation/task thread. At most 256 characters.
- `source` (str, required): Either 'user' or 'agent'
- `text` (str, required): Text content to store
- `images` (list, optional): Base64 encoded images with mime_type. Each image's own `metadata` crosses the boundary as a JSON-encoded **string**, not an object, and `get_context_by_ids` returns it verbatim as that same string -- including an empty string, which round-trips unchanged. Only an ABSENT (or null) metadata value omits the key from the returned image.
- `metadata` (dict, optional): Additional structured data - completely flexible JSON object for your use case
- `tags` (list, optional): Tags for organization (automatically normalized). At most 100 tags, each at most 128 characters.
- `visibility` (str, optional): `private` (readable by the owner and every grantee) or `public` (readable by everyone). Omitted uses `ACCESS_CONTROL_DEFAULT_VISIBILITY` (default `private`). Publishing as `public` may require the role named by `ACCESS_CONTROL_PUBLISH_ROLE`. Applies only when the store inserts a new entry: a deduplication update never changes the existing entry's owner or visibility.
- `deduplicate` (bool, optional, default `true`): `true` updates the caller's own latest entry of the same thread and source instead of inserting a duplicate when its text is identical, which protects against client retransmits (see [Access Scoping](#access-scoping)). `false` skips that lookup and always inserts a new entry owned by the caller, with `visibility` applied and every generation step run as for any new entry. Corpus sync clients pass `false`; see [Storing Without Deduplication](knowledge-base.md#storing-without-deduplication).
- `defer_generation` (bool, optional, default `false`): `false` generates the entry's embeddings, summary and section summaries before the entry is stored; an embedding or summary failure stores nothing, while a section summary that fails is left out. `true` stores the entry at once with a generation job that generates them in the background (see [Generation Status](#generation-status)). A deferring store skips the deduplication pre-check that lets a likely duplicate reuse its summary; when it deduplicates into an existing entry, that entry keeps a pending job, gets a fresh job in place of a failed one, and gets a job otherwise only when it lacks embeddings or a summary that the server would generate.

**Write-path length caps:** `thread_id` (256 characters), each tag (128 characters) and the value stored under any *indexed* metadata field (512 characters) are bounded at the tool boundary. Each of those values lands in a PostgreSQL btree index whose index-tuple ceiling would otherwise reject the write inside the store transaction while SQLite accepted the identical value, so the cap is what keeps the two backends accepting and rejecting the same input. Values under non-indexed metadata keys are not length-capped. The indexed-value cap is measured on the text the index actually stores, so a list or object under an indexed field is measured as its whole serialized JSON, not skipped.

**Write-path type validation:** a value stored under a field configured in `METADATA_INDEXED_FIELDS` with an `integer`, `boolean`, or `float` type hint must be a valid value of that declared type, and one that is not is rejected at the tool boundary on BOTH backends -- before any embedding or summary generation runs. For `integer` and `boolean` the PostgreSQL index also applies that type as a SQL cast, so the check also keeps such a value from aborting the PostgreSQL transaction after generation while SQLite stores the same value. Accepted input matches PostgreSQL's own parsers, including its non-decimal integer literals (`0x10`, `0o17`, `0b101`) and `_` digit separators; an integer value must also fit the 32-bit `INTEGER` range.

**Metadata Flexibility:**
The metadata field accepts any JSON-serializable structure, making the server adaptable to various use cases:
- **Task Management**: Store `status`, `priority`, `assignee`, `due_date`, `completed`
- **Agent Coordination**: Track `agent_name`, `task_name`, `execution_time`, `resource_usage`
- **Knowledge Base**: Include `category`, `relevance_score`, `source_url`, `author`
- **Debugging Context**: Save `error_type`, `stack_trace`, `environment`, `version`
- **Analytics**: Record `user_id`, `session_id`, `event_type`, `timestamp`

**Performance Note:** The following metadata fields are indexed by default for faster filtering:
- `status`: State information (e.g., 'pending', 'active', 'done')
- `agent_name`: Specific agent identifier
- `task_name`: Task title for string searches
- `project`: Project name for filtering
- `report_type`: Report categorization (e.g., 'research', 'implementation')
- `references`: Cross-references object (PostgreSQL GIN index only)
- `technologies`: Technology stack array (PostgreSQL GIN index only)

Indexed fields are configurable via `METADATA_INDEXED_FIELDS` environment variable. See [Metadata Guide](metadata-addition-updating-and-filtering.md#environment-variables) for details.

**Returns:** Dictionary with success status, `thread_id`, a message, a 32-character lowercase hex `context_id` identifying the stored entry, and the entry's `generation_status` after the store (see [Generation Status](#generation-status)). The message names `generation pending` or `generation failed` when the entry is in that state. Example: `{"success": true, "context_id": "0190abcdef1234567890abcdef123456", "thread_id": "project-123", "message": "Context stored (generation pending)", "generation_status": "pending"}`

### search_context

Search context entries with powerful filtering including metadata queries and date ranges.

**Parameters:**
- `thread_id` (str, optional): Filter by thread
- `source` (str, optional): Filter by source ('user' or 'agent')
- `tags` (list, optional): Filter by tags (OR logic; at most 100 tags per request)
- `content_type` (str, optional): Filter by type ('text' or 'multimodal')
- `metadata` (dict, optional): Simple metadata filters (key=value equality; at most 100 keys per request)
- `metadata_filters` (list, optional): Advanced metadata filters with operators (at most 100 filters per request)
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `limit` (int, optional): Maximum results to return (1-100, default: 30)
- `offset` (int, optional): Pagination offset (default: 0)
- `include_images` (bool, optional): Include image data in response
- `explain_query` (bool, optional): Include query execution statistics

**Metadata Filtering:** Supports simple key=value equality and advanced filtering with 16 operators. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Date Filtering:** Supports ISO 8601 date filtering. See [Date Filtering](#date-filtering) section below.

**Returns:** List of matching context entries with truncated `text_content`, `summary`, `is_text_content_truncated` flag, `visibility`, `is_owner` (true when the caller owns the entry) and `generation_status` (see [Generation Status](#generation-status)), plus optional query statistics

### get_context_by_ids

Fetch specific context entries by their IDs.

**Parameters:**
- `context_ids` (list[str], required): List of context-entry IDs in canonical 32-character hex or 36-character hyphenated UUID form (at most 100 IDs per call). Both forms are accepted at the tool boundary; storage canonicalizes to 32-character lowercase hex. An 8-31 character hex prefix is also accepted for each ID and resolved independently (zero matches or an ambiguous prefix returns an error).
- `include_images` (bool, optional): Include image data (default: True)

**Returns:** List of the requested entries the caller may read, with full untruncated `text_content`; an ID that does not exist and an ID of an entry the caller may not read are both omitted. Each entry contains `id`, `thread_id`, `source`, `content_type`, `text_content`, `metadata`, `tags`, `created_at`, `updated_at`, `visibility` (`private` or `public`), `is_owner` (true when the caller owns the entry; the owner itself is never returned), and `generation_status` (see [Generation Status](#generation-status)). An entry whose `content_type` is `multimodal` also carries `images` (each with `mime_type`, base64 `data` and any image `metadata`) when `include_images` is true; a text entry, and every entry fetched with `include_images` false, has no `images` key. The `summary` field follows a tri-state contract controlled by the `GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY` environment variable:

- When disabled (the default), the `summary` key is omitted entirely; consumers reading `entry.get('summary')` will receive `None`, which is the conventional Python signal for "feature disabled, no value to surface".
- When enabled and the stored summary is a non-empty string, the value is returned verbatim.
- When enabled but the stored summary is `NULL` or empty (e.g., generation was skipped because text was shorter than `SUMMARY_MIN_CONTENT_LENGTH`, no provider is configured, or the entry's generation job has not stored it yet), the value is normalized to an empty string `''`. This mirrors the search-tool contract (search tools always emit `summary` as a string, never `None`) and provides an explicit "feature on, no data yet" signal distinct from the "feature disabled" `None`.

### delete_context

Delete context entries by IDs or thread.

**Parameters:**
- `context_ids` (list[str], optional): Specific 32-character hex or 36-character hyphenated UUID IDs to delete (at most 100 IDs per call). Both forms are accepted at the tool boundary. An 8-31 character hex prefix is also accepted for each ID and resolved independently (zero matches or an ambiguous prefix returns an error).
- `thread_id` (str, optional): Delete the caller's own entries in a thread

Provide EXACTLY ONE of them. Supplying both is rejected with an error, and supplying neither is rejected as well. Deletion is irreversible, so a request that names two different sets of entries is refused rather than served by silently honoring one of them and reporting success for the whole call.

Note the contrast with `delete_context_batch`, which takes the same two argument names and COMBINES them as criteria: there, `context_ids` plus `thread_id` means "these ids, restricted to that thread".

Only the caller's own entries are deleted. With `context_ids`, an ID of an entry the caller may not read is skipped and not counted, and a named entry the caller may read but does not own refuses the whole call with `Not authorized to delete context entries: {ids}`, deleting nothing. With `thread_id`, the caller's own entries in the thread are deleted and every other entry stays in place without an error. Deleting an entry also deletes its tags, images, embeddings and grants.

**Returns:** Dictionary with deletion count

### list_threads

List threads with statistics. Pagination is optional and backward-compatible: with no arguments every thread holding an entry the caller may read is returned, ordered by most-recent activity first. Every figure counts only the entries the caller may read.

**Parameters:**
- `limit` (int, optional): Maximum threads to return (1-100). Omit (the default) to return all threads with no limit.
- `offset` (int, optional): Number of leading threads to skip for pagination (default 0). Ignored when `limit` is omitted.

**Returns:** Dictionary containing:
- `threads`: List of threads for the requested page, each with thread_id, entry_count, source_types, multimodal_count, first_entry/last_entry timestamps, and last_id (a hint for future keyset pagination).
- `total_threads`: Count of threads in THIS response (the returned page), not the whole database.

Threads are ordered by `last_entry` descending, tie-broken by the latest entry id descending; `limit`/`offset` are applied after this ordering. Keyset (cursor) pagination on `last_id` is a possible future enhancement; today `limit`/`offset` is the supported pagination.

### get_statistics

Get database statistics, usage metrics, and feature status.

The entry, thread, image, tag, embedding, index, summary, node and generation figures count only the entries the caller may read. `database_size_mb`, `embeddings_size_mb`, `embeddings_size_estimated`, `connection_metrics` and the configuration fields describe the whole deployment and are the same for every caller.

**Returns:** Dictionary with:
- Total entries count
- Breakdown by source and content type
- Total images count
- Unique tags count
- Database size in MB (`database_size_mb`): whole database via `pg_database_size(current_database())` on PostgreSQL; on-disk database file size on SQLite (excludes the `-wal`/`-shm` sidecars, so it can transiently under-report under WAL mode). Omitted for in-memory or missing-file SQLite databases.
- Embeddings storage size in MB (`embeddings_size_mb`, with the boolean `embeddings_size_estimated`): size of the active vector payload table (`vec_context_embeddings_compressed` when compression is enabled, otherwise `vec_context_embeddings`). Present when embedding generation or compression is enabled. NOT byte-comparable across backends: on PostgreSQL it is the on-disk relation size including indexes (`pg_total_relation_size`); on SQLite it is the exact compressed payload bytes when compression is enabled, or a deterministic fp32 estimate when it is not. `embeddings_size_estimated` is `true` only for the SQLite fp32 estimate.
- Connection metrics
- Semantic search status (enabled, available, model, dimensions, embedding count, coverage)
- Full-text search status (enabled, available, language, backend, engine, indexed entries, coverage)
- Chunking configuration (enabled, available, chunk size, overlap, aggregation)
- Reranking status (enabled, available, provider, model)
- Summary generation status (enabled, available, provider, model, summary count, coverage, min content length)
- Compression status (enabled, available, provider, bits, variant, seed, dim, max_concurrent): present when `ENABLE_EMBEDDING_COMPRESSION=true`; reduced to `{enabled: false, available: false}` shape when disabled.
- Index-tree node-summary status (`index_tree` with `enabled`, `node_count`): `enabled` reflects `ENABLE_INDEX_TREE_NODE_SUMMARIES`; `node_count` is the total stored per-node summaries (0 when disabled or the table is absent).
- Generation state (`generation` with `pending` and `failed`), always present: the readable entries whose generation job waits or runs, and those whose job used all of its attempts (see [Generation Status](#generation-status)). An entry whose outputs are complete is in neither count.

### update_context

Update specific fields of an existing context entry.

**Parameters:**
- `context_id` (str, required): ID of the context entry to update. Accepts a 32-character hex UUID, a 36-character hyphenated UUID, or an 8-31 character hex prefix that uniquely identifies an entry. Prefixes shorter than 8 characters are rejected. Ambiguous prefixes (multiple matches) return an error. Whitespace is stripped and case is folded to lowercase at the boundary.
- `text` (str, optional): New text content
- `metadata` (dict, optional): New metadata (full replacement)
- `metadata_patch` (dict, optional): Partial metadata update using RFC 7396 JSON Merge Patch
- `tags` (list, optional): New tags (full replacement). At most 100 tags, each at most 128 characters.
- `images` (list, optional): New images (full replacement). Each image's own `metadata` crosses the boundary as a JSON-encoded **string**, not an object.
- `visibility` (str, optional): New visibility, `private` or `public`. Only the entry owner may change it, and publishing as `public` may require the role named by `ACCESS_CONTROL_PUBLISH_ROLE`.
- `defer_generation` (bool, optional, default `false`): applies only when `text` is given; any other update ignores it and generates nothing. `false` regenerates the embeddings, summary and section summaries before the update is applied; an embedding or summary failure changes nothing, while a section summary that fails is left out. `true` stores the new text at once, removes the old text's embeddings, summary and section summaries in the same transaction, and leaves the new ones to a generation job (see [Generation Status](#generation-status)).

The same write-path length caps as `store_context` apply: each tag is limited to 128 characters, and the value under any indexed metadata field to 512 characters, in both the `metadata` and the `metadata_patch` form.

**Metadata Update Options:**

Use `metadata` for full replacement or `metadata_patch` for partial updates. These parameters are mutually exclusive.

RFC 7396 JSON Merge Patch semantics (`metadata_patch`):
- New keys are ADDED to existing metadata
- Existing keys are REPLACED with new values
- Null values DELETE keys

```python
# Update single field while preserving others
update_context(context_id="0190abcdef1234567890abcdef123456", metadata_patch={"status": "completed"})

# Add new field and delete another
update_context(context_id="0190abcdef1234567890abcdef123456", metadata_patch={"reviewer": "alice", "draft": None})
```

**Limitations (RFC 7396):** Null values cannot be stored (null means delete key - use full replacement if needed), arrays are replaced entirely (not merged). See [Metadata Guide](metadata-addition-updating-and-filtering.md#partial-updates-metadata_patch) for details.

**Field Update Rules:**
- **Updatable fields**: text_content, metadata, tags, images (owner or write grant); visibility (owner only)
- **Immutable fields**: id, thread_id, source, created_at, owner (preserved for data integrity)
- **Auto-managed fields**: content_type (recalculated based on image presence), updated_at (set to current timestamp)

**Update Behavior:**
- Only provided fields are updated (selective updates)
- Tags and images use full replacement semantics for consistency
- The entry's grants stay as they are; only `share_context` and `unshare_context` change them
- Content type automatically switches between 'text' and 'multimodal' based on image presence
- At least one updatable field must be provided
- An entry the caller may not read is reported as `Context entry with ID {id} not found`; an entry it may read but not modify is refused with `Not authorized to modify context entry with ID {id}`

**Returns:** Dictionary with:
- Success status
- Context ID
- List of updated fields
- Success/error message, which names `generation pending` or `generation failed` when the entry is in that state
- `generation_status`: the entry's state after the update (see [Generation Status](#generation-status))

## Search Tools

### semantic_search_context

Perform semantic similarity search using vector embeddings.

Note: This tool is available by default (`ENABLE_SEMANTIC_SEARCH=auto`) whenever an embedding provider is present; set `ENABLE_SEMANTIC_SEARCH=false` to force off. The implementation varies by backend:
- **SQLite**: Uses sqlite-vec extension with embedding model via Ollama
- **PostgreSQL**: Uses pgvector extension (pre-installed in pgvector Docker image) with embedding model via Ollama

**Parameters:**
- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic; at most 100 tags per request)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality; at most 100 keys per request)
- `metadata_filters` (list, optional): Advanced metadata filters with operators (at most 100 filters per request)
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string
- List of semantically similar context entries with truncated `text_content`, `summary`, `is_text_content_truncated` flag, `visibility`, `is_owner` (true when the caller owns the entry), `generation_status`, and similarity scores. An entry without embeddings is never a result, so an entry whose deferred generation has not stored its embeddings yet is absent (see [Generation Status](#generation-status))
- Result count
- Model name used for embeddings
- Query execution statistics (only when `explain_query=True`)

**Use Cases:**
- Find related work across different threads based on semantic similarity
- Discover contexts with similar meaning but different wording
- Concept-based retrieval without exact keyword matching
- Find similar content within a specific time period using date filters

**Date Filtering Example:**
```python
# Find similar content from the past week
semantic_search_context(
    query="authentication implementation",
    start_date="2025-11-22",
    end_date="2025-11-29"
)
```

For setup instructions, see the [Semantic Search Guide](semantic-search.md).

### fts_search_context

Perform full-text search with linguistic processing, relevance ranking, and highlighted snippets.

Note: This tool is available by default (`ENABLE_FTS=auto`); set `ENABLE_FTS=false` to force off. The implementation varies by backend:
- **SQLite**: Uses FTS5 with BM25 ranking. Porter stemmer (English) or unicode61 tokenizer (multilingual).
- **PostgreSQL**: Uses tsvector/tsquery with ts_rank_cd ranking. Stems 28 languages; the 29th `FTS_LANGUAGE` value, `simple`, tokenizes without stemming.

**Parameters:**
- `query` (str, required): Search query (1-10000 characters)
- `mode` (str, optional): Search mode - `match` (default), `prefix`, `phrase`, or `boolean`
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic; at most 100 tags per request)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality; at most 100 keys per request)
- `metadata_filters` (list, optional): Advanced metadata filters with operators (at most 100 filters per request)
- `highlight` (bool, optional): Include highlighted snippets in results (default: false)
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Search Modes:**
- `match`: All words required, in their stemmed forms where stemming applies (default)
- `prefix`: Every word is matched as a word beginning
- `phrase`: The words in sequence, in this order
- `boolean`: Operators, in backend-specific syntax: `AND`, `OR`, `NOT` and parentheses on SQLite; a space for AND, `or` and a leading `-` for NOT on PostgreSQL, which has no grouping

The description the server serves for this tool states the exact behavior of the running backend and `FTS_LANGUAGE`: stemming, stop words, and the syntax and examples of each mode.

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string and search mode
- List of matching entries with truncated `text_content`, `summary`, `is_text_content_truncated` flag, `visibility`, `is_owner` (true when the caller owns the entry), `generation_status`, relevance scores, and a highlighted snippet when `highlight` is true. The full-text index covers an entry as soon as it is stored, whatever its generation state
- Result count
- The configured `FTS_LANGUAGE` (`language`)

**Example:**
```python
# Search with prefix matching
fts_search_context(
    query="auth",
    mode="prefix",
    thread_id="project-123"
)

# Boolean search with metadata filter (SQLite syntax; on PostgreSQL write "authentication security")
fts_search_context(
    query="authentication AND security",
    mode="boolean",
    metadata_filters=[{"key": "status", "operator": "eq", "value": "active"}]
)
```

For detailed configuration, see the [Full-Text Search Guide](full-text-search.md).

### hybrid_search_context

Perform hybrid search combining FTS and semantic search with Reciprocal Rank Fusion (RRF).

Note: This tool is available by default (`ENABLE_HYBRID_SEARCH=auto`) when at least one of full-text or semantic search is available; set `ENABLE_HYBRID_SEARCH=false` to force off. The RRF algorithm combines results from available search methods, boosting documents that appear in both.

**Parameters:**
- `query` (str, required): Natural language search query (1-10000 characters)
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `fusion_method` (str, optional): Fusion algorithm - `'rrf'` (default)
- `rrf_k` (int, optional): RRF smoothing constant (1-1000, default from HYBRID_RRF_K env var)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic; at most 100 tags per request)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality; at most 100 keys per request)
- `metadata_filters` (list, optional): Advanced metadata filters with operators (at most 100 filters per request)
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string and fusion method
- List of matching entries with truncated `text_content`, `summary`, `is_text_content_truncated` flag, `visibility`, `is_owner` (true when the caller owns the entry), `generation_status`, combined RRF scores, and individual search rankings. An entry without embeddings yet is found by the full-text leg only
- Result count and counts from each search method
- List of search modes actually used
- Query execution statistics (only when `explain_query=True`)

**Scores Breakdown:**
Each result includes a `scores` object with:
- `rrf`: Combined RRF score (higher = better)
- `fts_rank`: Position in FTS results (1-based), null if not in FTS results
- `semantic_rank`: Position in semantic results (1-based), null if not in semantic results
- `fts_score`: Original FTS relevance score (BM25/ts_rank)
- `semantic_distance`: Original semantic distance, lower = more similar. The underlying metric is Euclidean L2 (>= 0) for uncompressed/`mse` storage, or a negated inner product (~ -1..0 for normalized embeddings, more negative = more similar) when the default `ip` compression variant is active.
- `rerank_score`: Cross-encoder relevance score (higher = better, 0.0-1.0), null if reranking disabled

**Note:** When `ENABLE_RERANKING=true` (default), results are re-ordered by `rerank_score` after initial retrieval. The original scores (`fts_score`, `semantic_distance`) are preserved for debugging but `rerank_score` determines final ordering.

**Graceful Degradation:**
- If only FTS is available, returns FTS results only
- If only semantic search is available, returns semantic results only
- If neither is available, raises an error

**Example:**
```python
# Full hybrid search
hybrid_search_context(
    query="authentication implementation",
    thread_id="project-123"
)

# Hybrid with metadata filtering
hybrid_search_context(
    query="performance optimization",
    metadata={"status": "completed"},
    metadata_filters=[{"key": "priority", "operator": "gte", "value": 7}]
)
```

For detailed configuration and troubleshooting, see the [Hybrid Search Guide](hybrid-search.md).

### Ranked-Search Pagination

`semantic_search_context`, `fts_search_context` and `hybrid_search_context` all decide their final order AFTER the database returns rows -- through cross-encoder reranking, RRF fusion, or both. Their pagination contract follows from that:

- Every page of a query is a window into ONE ordering, computed once over a FIXED candidate depth of 100 rows. The depth does not vary with `limit` or `offset`, so consecutive pages never repeat a row or skip one.
- 100 is also the largest page a single request can ask for, so any one request can be served whole.
- A window that reaches past the depth is served short, and a window that STARTS at or past it returns nothing at all. Either way the response carries a `rank_depth_limit` object saying so explicitly, instead of leaving the caller to guess whether the result set was exhausted:

```json
{
  "rank_depth_limit": {
    "requested_offset": 100,
    "requested_limit": 10,
    "rank_depth": 100
  }
}
```

The key is ABSENT when the whole requested window fits inside the depth. A page that starts at or past the depth is answered directly from these numbers, without running a search or an embedding call.

## Search Tools Response Structure

All search tools return consistent response structures with common fields and tool-specific additions:

| Field               | search_context     | semantic_search_context | fts_search_context | hybrid_search_context |
|---------------------|--------------------|-------------------------|--------------------|-----------------------|
| `results`           | List of entries    | List of entries         | List of entries    | List of entries       |
| `count`             | Yes                | Yes                     | Yes                | Yes                   |
| `query`             | No                 | Yes                     | Yes                | Yes                   |
| `stats`             | explain_query=True | explain_query=True      | explain_query=True | explain_query=True    |
| `model`             | No                 | Yes (embedding model)   | No                 | No                    |
| `mode`              | No                 | No                      | Yes (search mode)  | No                    |
| `language`          | No                 | No                      | Yes (FTS language) | No                    |
| `fusion_method`     | No                 | No                      | No                 | Yes                   |
| `search_modes_used` | No                 | No                      | No                 | Yes                   |
| `fts_count`         | No                 | No                      | No                 | Yes                   |
| `semantic_count`    | No                 | No                      | No                 | Yes                   |

**Entry Fields by Tool:**

| Entry Field                                    | search_context        | semantic_search_context    | fts_search_context         | hybrid_search_context      |
|------------------------------------------------|-----------------------|----------------------------|----------------------------|----------------------------|
| `id`, `thread_id`, `source`, `content_type`    | Yes                   | Yes                        | Yes                        | Yes                        |
| `text_content`                                 | Truncated             | Truncated                  | Truncated                  | Truncated                  |
| `summary`                                      | Yes (string)          | Yes (string)               | Yes (string)               | Yes (string)               |
| `is_text_content_truncated`                    | Yes                   | Yes                        | Yes                        | Yes                        |
| `metadata`, `tags`, `created_at`, `updated_at` | Yes                   | Yes                        | Yes                        | Yes                        |
| `visibility`                                   | Yes                   | Yes                        | Yes                        | Yes                        |
| `is_owner`                                     | Yes                   | Yes                        | Yes                        | Yes                        |
| `generation_status`                            | Yes                   | Yes                        | Yes                        | Yes                        |
| `images`                                       | include_images=True   | include_images=True        | include_images=True        | include_images=True        |
| `scores`                                       | No                    | Yes                        | Yes                        | Yes                        |
| `highlighted`                                  | No                    | No                         | highlight=True             | No                         |

**`summary` field**: Present in all search tool results. Populated by automatic LLM-based summary generation (enabled by default with Ollama). Contains a dense summary (token limit controlled by `SUMMARY_MAX_TOKENS`, default 4000) that is more informative than the truncated `text_content`. Empty string when summary generation is disabled or the summary has not yet been generated, as for an entry whose deferred generation is still `pending`. See [Summary Generation Guide](summary-generation.md) for configuration.

**Scores Object Structure:**

All search tools (except `search_context`) return a unified `scores` object with applicable fields:

| Field               | semantic_search | fts_search | hybrid_search | Polarity        |
|---------------------|-----------------|------------|---------------|-----------------|
| `semantic_distance` | Yes             | No         | Yes           | LOWER = better  |
| `semantic_rank`     | null            | No         | Yes           | LOWER = better  |
| `fts_score`         | No              | Yes        | Yes           | HIGHER = better |
| `fts_rank`          | No              | null       | Yes           | LOWER = better  |
| `rrf`               | No              | No         | Yes           | HIGHER = better |
| `rerank_score`      | Yes*            | Yes*       | Yes*          | HIGHER = better |

*`rerank_score` is present when reranking is enabled (`ENABLE_RERANKING=true`, default).

**Notes:**
- `stats` is only included when `explain_query=True` for all search tools
- All search tools return truncated `text_content` (configurable via `SEARCH_TRUNCATION_LENGTH`, default 300 chars) with `summary` and `is_text_content_truncated` flag; use `get_context_by_ids` for full content
- Every result of every search tool reports the entry's `visibility` (`private` or `public`), `is_owner` (a boolean, true when the caller owns the entry) and `generation_status` (`ready`, `pending` or `failed`; see [Generation Status](#generation-status)); the owner itself is never returned
- `semantic_search_context` and the semantic leg of `hybrid_search_context` return only entries that have embeddings; `search_context`, `fts_search_context` and the full-text leg of `hybrid_search_context` find an entry as soon as it is stored
- For standalone FTS and semantic searches, rank fields are always `null` (no cross-method ranking)

## Navigation Tools (locate / navigate / extract)

These read-only tools complement search: grep locates exact text, navigate orients within a record, and read extracts a span. They share one Unicode code-point character-offset contract, so a `grep_context` match's offsets and a `navigate_context` node's offsets both feed directly into `read_context_range`. See [Grep, Navigation & Partial Reads](grep-navigation-partial-read.md) for when to use each.

### grep_context

Server-side grep: literal or regular-expression, line-oriented, UNRANKED pattern matching over stored `text_content`. Unlike `fts_search_context` (stemmed, ranked) it matches raw characters and returns precise match locations. Matching runs in Python — the stdlib `re` engine for literal patterns and the third-party `regex` engine for user regular expressions — so results are identical on SQLite and PostgreSQL.

**Parameters:**
- `pattern` (str, required): Literal substring (default) or regular expression to match (at most `GREP_MAX_PATTERN_CHARS` characters, default 32768)
- `is_regex` (bool, optional): Treat `pattern` as a Python regular expression (default: False — literal substring, auto-escaped)
- `case_sensitive` (bool, optional): Match case-sensitively (default: False — Unicode-aware case-insensitive)
- `output_mode` (str, optional): `files_with_matches` (default; context_ids + match_count), `content` (each match with line + offsets + context), or `count` (per-entry tally)
- `context_lines` (int, optional): Surrounding lines before/after each match in content mode (0-100; effectively clamped to `GREP_MAX_CONTEXT_LINES`, default 20; like `grep -C`)
- `max_matches` (int, optional): Maximum total matches to return (default 100; clamped to the server cap)
- `max_entries_scanned` (int, optional): Maximum entries the scan visits (clamped to the server cap)
- `thread_id` / `source` / `tags` / `metadata_filters` / `content_type` (optional): Reuse the store's filters to scope the scan (the ripgrep glob/type analog); scoping with `thread_id` is recommended; `tags` accepts at most 100 tags and `metadata_filters` at most 100 filters per request

**Returns:** `{mode, total_matches, truncated, results}`. In `content` mode each result carries `context_id`, `line_number`, `line`, `match_start`/`match_end` (code-point offsets into `text_content`), and `before`/`after` context lines; in `files_with_matches` mode `context_id` + `match_count`; in `count` mode `context_id` + `count`. Every result, each `content` match included, also carries its entry's `visibility` (`private` or `public`), `is_owner` (true when the caller owns the entry) and `generation_status` (see [Generation Status](#generation-status)). Grep reads the stored text, so it finds an entry as soon as it is stored, whatever its generation state. `truncated` is True when matches or the scan were capped.

### navigate_context

Build a navigable Markdown-heading outline (index_tree) for one record, computed on demand from the current text — never stale, works for every entry. The synthetic root spans the whole document and mirrors the entry summary; each node carries char offsets that feed `read_context_range`.

**Parameters:**
- `context_id` (str, required): Context entry id (32/36-char UUID or 8-31 char hex prefix)
- `max_depth` (int, optional): Deepest Markdown heading level to include (1-6, default 6; deeper headings fold into their section)
- `include_node_summaries` (bool, optional): Attach stored per-node LLM summaries to descendant nodes when that layer is enabled (default: False)

**Returns:** `{context_id, total_chars, node_count, root, visibility, is_owner, generation_status}` where `root` is a recursive node `{node_id, level, ordinal, title, char_start, char_end, summary, children}`. `node_id` is a heading-path slug with a sibling ordinal (e.g. `setup/install`, `notes-2`); the root's `summary` mirrors the entry summary by reference and is null while the entry has none. `visibility`, `is_owner` and `generation_status` describe the entry (its visibility, whether the caller owns it, and its [generation state](#generation-status)) and appear at the top level only, never on a node. The outline itself never depends on generation, but after a deferred store or text update the nodes carry no stored summaries until the entry's generation job completes.

### read_context_range

Read part of one record by character range, line range, or outline `node_id`. Slices the full stored `text_content` directly (works for every entry regardless of embeddings).

**Parameters:**
- `context_id` (str, required): Context entry id (32/36-char UUID or 8-31 char hex prefix)
- `start_char` / `end_char` (int, optional): Character range (Unicode code-point offsets; end exclusive)
- `start_line` / `end_line` (int, optional): Line range (1-based, inclusive)
- `node_id` (str, optional): An outline node id from `navigate_context`

Provide exactly ONE addressing mode. Pair with `grep_context` (content mode) by feeding `match_start`/`match_end` into `start_char`/`end_char`, or with `navigate_context` by passing a section's `node_id`.

**Returns:** `{context_id, start_char, end_char, start_line, end_line, text, visibility, is_owner, generation_status}` echoing the RESOLVED span, with the entry's visibility, whether the caller owns it, and its [generation state](#generation-status). Out-of-range offsets are clamped to `[0, len(text)]`, so a stale offset or node_id from a prior turn degrades gracefully.

## Sync Tools

These read-only tools serve clients that keep a document corpus in sync with a thread, such as `mcp-context-ingest`. A managed entry carries the reserved `metadata.ingest` block `{scope, source_id, meta_digest, synced_at}` and belongs to the corpus of thread `T` when its block is valid and its `scope` equals `T`. Both tools manage only the caller's own entries (see [Access Scoping](#access-scoping)). Both register by default; `ENABLE_CONTEXT_SYNC=false` leaves them out. The [Knowledge Base Guide](knowledge-base.md) describes the block, the digests and the sync flow.

### plan_context_sync

Compare a manifest of a corpus's source documents with the caller's managed entries of a thread and answer what to create, update and delete. Writes nothing; apply the plan with `store_context_batch` (each entry with `"deduplicate": false`), `update_context_batch` and `delete_context_batch`.

**Parameters:**
- `thread_id` (str, required): The corpus thread, such as `kb:handbook`. At most 256 characters.
- `manifest` (list, required): At most 1,000 items, each `source_id` at most once. An ordinary item is `{source_id, text_digest, meta_digest}`, where `text_digest` is the SHA-256 hex digest of the text the client would store (after surrounding whitespace is removed) and `meta_digest` the 64-character hex digest the client writes into the block. A keep item `{source_id, keep: true}` carries no digest and marks a source the client could not read in this run: the plan creates, updates and deletes nothing for it. An item accepts no other key, and `source_id` is 1 to 256 characters.
- `source_id_after` (str, optional): Exclusive lower bound of the planned source ids; omitted means no lower bound.
- `source_id_through` (str, optional): Inclusive upper bound of the planned source ids; omitted means no upper bound. Source ids compare by code point, and every manifest item must lie in `(source_id_after, source_id_through]`.
- `allow_empty_manifest` (bool, optional, default `false`): Confirm an empty manifest, which plans the deletion of every managed entry in the range.

**Returns:** `{thread_id, creates, updates, deletes, deletes_total, unchanged_count, kept_count, foreign_count}`:
- `creates`: manifest indices of the ordinary items that have no stored entry.
- `updates`: `{index, context_id, change}` per ordinary item whose stored entry differs; `change` is `text` (the item's `text_digest` differs from the stored content hash, or the entry has none), `metadata` (the `meta_digest` differs) or `both`.
- `deletes`: `{context_id, source_id, reason}`, at most 100, ordered by `source_id` and then `context_id`. `reason` is `absent` (the manifest does not name the source) or `duplicate` (another entry of the same source survives: the entry whose content hash matches the item, then whose `meta_digest` matches, then the newest).
- `deletes_total`: every planned delete of the range. Delete the listed entries and plan again until it is 0.
- `unchanged_count`, `kept_count`: ordinary items already in line, and keep items.
- `foreign_count`: readable managed entries of the range that another principal owns, which a sync never changes.

Invalid arguments are refused before any read: a `source_id_after` that does not sort before `source_id_through`, an empty manifest without `allow_empty_manifest`, repeated source ids, an item outside the range, more than 1,000 items, a malformed digest, a keep item with a digest, or an ordinary item without both. A client planning a corpus of more than 1,000 documents sorts its manifest by `source_id` and plans consecutive ranges; see [Sync Planning](knowledge-base.md#sync-planning).

### list_ingest_state

List one page of the caller's own managed entries of a thread, ordered by `source_id` and then `context_id` in code-point order.

**Parameters:**
- `thread_id` (str, required): The corpus thread. At most 256 characters.
- `limit` (int, optional, default 100): Entries per page, 1 to 200.
- `cursor` (str, optional): The `next_cursor` of the previous page; omitted reads the first page. A malformed cursor is refused with `invalid cursor`.

**Returns:** `{thread_id, entries, count, next_cursor}`. Each entry is `{context_id, source_id, text_digest, meta_digest, synced_at}`, where `text_digest` is the stored content hash. `next_cursor` is null on the last page, including a page that ends exactly at the last entry. See [Listing Ingest State](knowledge-base.md#listing-ingest-state).

## Sharing Tools

The owner of an entry uses these tools to give other principals access to it, take that access away, and see who holds it. Only the entry's owner may call them on it. An entry the caller may not read gets `Context entry with ID {id} not found`, exactly like an ID no entry carries; an entry the caller may read but does not own (through a read or write grant, or because it is `public`) gets an explicit refusal. A write grant lets its holder edit an entry but never share it. Each call names one entry by its full ID or by an 8-31 character hex prefix, which resolves over the entries the caller may read. All three tools register by default, and `DISABLED_TOOLS` can leave any of them out. [Sharing Entries](authentication.md#sharing-entries) describes how grants combine with visibility.

### share_context

Grant one user or one group `read` or `write` permission on an entry the caller owns.

**Parameters:**
- `context_id` (str, required): The entry's 32-character hex or 36-character hyphenated ID, or an 8-31 character hex prefix.
- `principal_type` (str, required): `user` for one principal (a token subject) or `group` for every member of a group.
- `principal_id` (str, required): The user or group ID, matched verbatim against the token's subject or groups: case-sensitive, any characters, 1 to 256 characters, no leading or trailing whitespace. The server does not check that the user or group exists.
- `permission` (str, required): `read`, or `write`, which also allows reading and lets the grantee change text, metadata, tags and images (but not visibility, deletion or sharing).

Sharing is idempotent: a grant the entry already holds stays as it is and the response reports `created: false`, a group read grant written by `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups` included. `read` and `write` are separate grants, so sharing `read` with a principal that holds `write` stores a second grant. A group grant reaches every member of the group. Grants leave the entry's `updated_at` unchanged and never conflict with a concurrent `update_context`.

**Returns:** `{success, context_id, principal_type, principal_id, permission, created, message}`. `context_id` is the resolved full ID. `message` is `Granted {permission} on context entry {id} to {principal_type} {principal_id}`, or `{principal_type} {principal_id} already holds {permission} on context entry {id}` when `created` is false.

**Errors:**
- `Context entry with ID {id} not found`: the caller may not read the entry, or no entry has the ID.
- `Not authorized to share context entry with ID {id}`: the caller may read the entry but does not own it.
- `Cannot share context entry with ID {id} with its owner`: a `user` grant naming the owner. A group grant to a group the owner belongs to is allowed.
- `Cannot share context entry with ID {id} with the anonymous principal; set its visibility to 'public' to publish it`: a `user` grant naming the principal that `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` names. The server refuses it before looking the ID up, and `{id}` is the value the caller passed.
- `principal_id cannot be empty or whitespace`, `principal_id cannot have leading or trailing whitespace`, `principal_id exceeds 256 characters`, or a `principal_id:` error for text the database cannot store.
- `The anonymous principal is read-only: share_context is not permitted`.

### unshare_context

Remove exactly one grant from an entry the caller owns.

**Parameters:**
- `context_id` (str, required): The entry's full ID or an 8-31 character hex prefix.
- `principal_type` (str, required): `user` or `group`.
- `principal_id` (str, required): The grantee's ID, under the same rules as for `share_context`.
- `permission` (str, required): The permission of the grant to remove, `read` or `write`.

The call removes only the named grant: removing `read` from a principal that also holds `write` leaves the `write` grant, which still allows reading. Removing a grant the entry does not hold changes nothing and reports `removed: false`. The owner can remove any grant the entry holds, including one naming the owner or the anonymous principal that reached the database without `share_context`. Grants leave the entry's `updated_at` unchanged.

**Returns:** `{success, context_id, principal_type, principal_id, permission, removed, remaining_permissions, message}`. `remaining_permissions` lists the permissions the same principal type and ID still hold on the entry after the call, read in the same transaction; a group grant can still give that principal access through its membership. `message` is `Removed {permission} on context entry {id} from {principal_type} {principal_id}`, or `No {permission} grant on context entry {id} for {principal_type} {principal_id}` when `removed` is false.

**Errors:** `Context entry with ID {id} not found`; `Not authorized to unshare context entry with ID {id}`; the `principal_id` validation errors of `share_context`; `The anonymous principal is read-only: unshare_context is not permitted`.

### list_context_grants

List every grant of an entry the caller owns, with no paging and no cap.

**Parameters:**
- `context_id` (str, required): The entry's full ID or an 8-31 character hex prefix.

**Returns:** `{context_id, grants, count}`. Each grant is `{principal_type, principal_id, permission, granted_by, created_at}`, ordered by `principal_type`, then `principal_id`, then `permission`. `granted_by` names the principal that stored the grant, and `created_at` is a UTC timestamp such as `2026-01-01T12:00:00Z`.

**Errors:** `Context entry with ID {id} not found`; `Not authorized to list the grants of context entry with ID {id}`.

## Batch Operations

The following tools enable efficient batch processing of context entries.

### store_context_batch

Store multiple context entries in a single batch operation.

**Parameters:**
- `entries` (list, required): List of context entries (max 100). Each entry has:
  - `thread_id` (str, required), `source` (str, required), `text` (str, required)
  - `metadata` (dict, optional), `tags` (list, optional), `images` (list, optional)
  - `visibility` (str, optional): `private` or `public`, as for `store_context`
  - `deduplicate` (bool, optional, default `true`): as for `store_context`, per entry. It accepts the values the `store_context` parameter accepts (`true`/`false`, `1`/`0`, and strings such as `"false"` or `"off"`); any other value fails the entry with `deduplicate must be a boolean`.
  - `defer_generation` (bool, optional, default `false`): as for `store_context`, per entry, accepting the same values as `deduplicate`; any other value fails the entry with `defer_generation must be a boolean`. A deferring entry skips the batch's generation pass and is stored with a generation job; in a batch that mixes both kinds, the other entries generate as usual.
- `atomic` (bool, optional): If true, all succeed or all fail (default: true)

Entries are stored in order, so the deduplication candidate of a deduplicating entry can be an entry stored earlier in the same batch: an entry with `deduplicate` true that follows an identical `deduplicate=false` entry of the same thread and source merges into the entry just inserted.

Each entry is subject to the same write-path length caps as `store_context`: `thread_id` at most 256 characters, at most 100 tags of at most 128 characters each, and at most 512 characters in the value under any indexed metadata field. A breach is reported as a per-entry validation error, so `atomic=false` still stores the remaining entries.

**Returns:** Dictionary with success, total, succeeded, failed, results array, message. Each result carries `index`, `success`, `context_id`, `error` and `generation_status`, the stored entry's state after the store (null for an entry that was not stored; see [Generation Status](#generation-status)). The message counts the stored entries whose generation is pending or failed, as in `generation pending for 2 entries`.

### update_context_batch

Update multiple context entries in a single batch operation.

**Parameters:**
- `updates` (list, required): List of update operations (max 100). Each update has:
  - `context_id` (str, required): 32-character hex or 36-character hyphenated UUID, or an 8-31 character hex prefix resolved against stored entries (zero matches or an ambiguous prefix returns an error).
  - `text` (str, optional), `metadata` (dict, optional), `metadata_patch` (dict, optional)
  - `tags` (list, optional), `images` (list, optional)
  - `visibility` (str, optional): `private` or `public`; owner-only, as for `update_context`
  - `defer_generation` (bool, optional, default `false`): as for `update_context`, per update, applying only to an update that sends `text`; it accepts the values the `update_context` parameter accepts, and any other value fails the update with `defer_generation must be a boolean`
- `atomic` (bool, optional): If true, all succeed or all fail (default: true)

Each update is subject to the same write-path length caps as `update_context`: at most 100 tags of at most 128 characters each, and at most 512 characters in the value under any indexed metadata field (in both the `metadata` and the `metadata_patch` form). A breach is reported as a per-entry validation error.

**Note:** `metadata_patch` uses RFC 7396 JSON Merge Patch semantics. See [Metadata Guide](metadata-addition-updating-and-filtering.md#partial-updates-metadata_patch) for details.

An entry the caller may not read is reported as `Context entry {id} not found`, and one it may read but not modify as `Not authorized to modify context entry {id}`; atomic mode ends the message with "at index {i}" and fails the whole batch, while non-atomic mode records the message in that entry's result.

**Returns:** Dictionary with success, total, succeeded, failed, results array, message. Each result carries `index`, `context_id`, `success`, `updated_fields`, `error` and `generation_status`, the entry's state after the update (null for an update that failed; see [Generation Status](#generation-status)). The message counts the updated entries whose generation is pending or failed.

### delete_context_batch

Delete multiple context entries by various criteria. **IRREVERSIBLE.**

**Parameters:**
- `context_ids` (list[str], optional): Specific 32-character hex or 36-character hyphenated UUID context IDs to delete, or 8-31 character hex prefixes resolved independently per element (zero matches or an ambiguous prefix returns an error). At most 100 IDs per call.
- `thread_ids` (list, optional): Delete the caller's own entries in these threads (at most 100 thread IDs per call)
- `source` (str, optional): Filter by source ('user' or 'agent') - must combine with another criterion
- `older_than_days` (int, optional): Delete entries older than N days - must combine with another criterion

At least one criterion must be provided. `source` and `older_than_days` are each insufficient on their own: on any database older than the requested window, either one matches essentially every row, so a single scalar would irreversibly reach the whole table. Combine them with each other, with `thread_ids`, or with `context_ids` (a retention purge is expressible as `older_than_days` plus `source`). Cascading delete removes associated tags, images, embeddings, and grants.

Only the caller's own entries are deleted, and an entry the caller may not read is never matched or counted. With `context_ids`, a named entry that matches the other criteria and that the caller may read but does not own refuses the whole call with `Not authorized to delete context entries: {ids}`, deleting nothing. Without `context_ids`, only the caller's own matching entries are deleted and every other matching entry stays in place without an error.

**Returns:** Dictionary with success, deleted_count, criteria_used, message

## Filtering Reference

The following filtering options apply to `search_context`, `semantic_search_context`, `fts_search_context`, and `hybrid_search_context` tools.

### Metadata Filtering

*Simple filtering* (exact match):
```python
metadata={'status': 'active', 'priority': 5}
```

*Advanced filtering* with operators:
```python
metadata_filters=[
    {'key': 'priority', 'operator': 'gt', 'value': 3},
    {'key': 'status', 'operator': 'in', 'value': ['active', 'pending']},
    {'key': 'agent_name', 'operator': 'starts_with', 'value': 'gpt'},
    {'key': 'completed', 'operator': 'eq', 'value': False}
]
```

**Supported Operators:**
- `eq`: Equals (case-insensitive for strings by default)
- `ne`: Not equals
- `gt`, `gte`, `lt`, `lte`: Numeric comparisons
- `in`, `not_in`: List membership (value lists accept at most 100 members)
- `exists`, `not_exists`: Field presence
- `contains`, `starts_with`, `ends_with`: String operations
- `is_null`, `is_not_null`: Null checks
- `array_contains`: Check if array field contains element

All string operators support `case_sensitive: true/false` option.

For comprehensive documentation on metadata filtering including real-world use cases, operator examples, nested JSON paths, and performance optimization, see the [Metadata Guide](metadata-addition-updating-and-filtering.md).

### Date Filtering

Filter entries by creation timestamp using ISO 8601 format:
```python
# Find entries from a specific day
search_context(thread_id="project-123", start_date="2025-11-29", end_date="2025-11-29")

# Find entries from a date range
search_context(thread_id="project-123", start_date="2025-11-01", end_date="2025-11-30")

# Find entries with precise timestamp
search_context(thread_id="project-123", start_date="2025-11-29T10:00:00")
```

Supported ISO 8601 formats:
- Date-only: `2025-11-29`
- DateTime: `2025-11-29T10:00:00`
- UTC (Z suffix): `2025-11-29T10:00:00Z`
- Timezone offset, whole-minute only: `2025-11-29T10:00:00+02:00` (or `+05:30`)

**Note:** Date-only `end_date` values automatically expand to end-of-day (`T23:59:59.999999`) for intuitive "entire day" behavior. Naive datetime (without timezone) is interpreted as UTC. A timezone offset must be whole-minute (`[+-]HH:MM`); an offset carrying sub-minute precision (seconds or a fractional part, such as `+05:30:15`) is rejected, because SQLite's `datetime()` evaluates such an offset to NULL and would silently match zero rows while PostgreSQL accepts it.

## Additional Resources

### Related Documentation

- **Summary Generation**: [Summary Generation Guide](summary-generation.md) - LLM-based summary generation setup
- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search setup
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata operators
- **Knowledge Base**: [Knowledge Base Guide](knowledge-base.md) - document corpora, corpus sync and `mcp-context-ingest`
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication, the access model, sharing entries and anonymous read access
- **Main Documentation**: [README.md](../README.md) - overview and quick start
