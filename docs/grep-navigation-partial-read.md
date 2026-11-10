# Grep, Navigation & Partial Reads

This guide covers the three READ_ONLY navigation tools — `grep_context`, `navigate_context`, and `read_context_range` — that let an agent work with stored context the way it works with a codebase: locate exact text, orient within a record, then read only the relevant span. They are complementary to the search tools, not a replacement.

## When to use which

Pick the tool by what you know and what you need back:

| Need                                                             | Tool                      | Why                                                               |
|------------------------------------------------------------------|---------------------------|-------------------------------------------------------------------|
| Find an EXACT string, identifier, ID, date, code token, or regex | `grep_context`            | Literal/regex, line-oriented, unranked, returns precise locations |
| Find content by MEANING (paraphrase, synonym, concept)           | `semantic_search_context` | Vector similarity over embeddings                                 |
| Find content by WORDS with stemming/ranking                      | `fts_search_context`      | Linguistic full-text search (BM25 / ts_rank)                      |
| Combine linguistic + semantic recall                             | `hybrid_search_context`   | Reciprocal Rank Fusion of FTS + semantic                          |
| Orient in a long record (table of contents)                      | `navigate_context`        | Code-derived Markdown heading tree (index_tree)                   |
| Read only PART of a record                                       | `read_context_range`      | Slice by char range, line range, or outline node                  |

## grep is not full-text search

`grep_context` and `fts_search_context` look similar but differ at the match unit, and the difference is the whole point:

- Full-text search matches normalized, stemmed lexemes and returns whole entries ranked by relevance. It cannot match `cat` inside `concatenate`, cannot do arbitrary substrings or regular expressions, cannot match case-sensitively or on punctuation/operators, and does not emit line-level locations.
- Grep matches raw characters (or a regular expression) over lines, is unranked, preserves case and punctuation, and returns LOCATIONS (line number plus start/end character offsets).

Use full-text/semantic search for recall ("find documents about X"); use grep for precise witnesses ("where exactly does this exact token appear"). They coexist as separate tools on purpose — folding ranking or stemming into grep would destroy the precise-locate guarantee.

## The locate -> navigate -> extract workflow

All three tools share ONE Unicode code-point character-offset contract, so their offsets compose directly:

1. `grep_context` (in `content` mode) returns each match with `match_start`/`match_end`.
2. Feed those into `read_context_range` (`start_char`/`end_char`) to read the hit with surrounding context.
3. Alternatively, call `navigate_context` to get a section's `node_id`, then `read_context_range(node_id=...)` to read that whole section.

Because grep returns locations rather than whole records, an agent can grep a large thread, then pull back only the spans it needs — instead of retrieving full entries.

## grep_context details

Matching runs in Python, not in backend SQL: the stdlib `re` engine for literal patterns and the third-party `regex` engine for user regular expressions (whose `timeout` bounds catastrophic backtracking). Because matching is identical Python code on both backends, SQLite and PostgreSQL return byte-identical results. SQLite has no built-in `REGEXP` and its `LIKE` is ASCII-only; PostgreSQL's POSIX regex dialect differs from Python's — running matching in Python is the only backend-parity-safe choice. SQL is used only as an optional coarse pure-ASCII substring pre-filter to narrow the scan.

The tool follows the ripgrep / Claude Code Grep model: `output_mode` is `files_with_matches` (default, just locators), `content` (lines with offsets and `-C`-style context), or `count`. `is_regex` defaults to False (literal substring, auto-escaped); `case_sensitive` defaults to False (Unicode-aware, so it correctly matches Cyrillic/accented case variants that an ASCII `LIKE` would miss). Output is always bounded by `max_matches`, and the scan by `max_entries_scanned`; the response carries a `truncated` flag. Scope the scan with `thread_id`/`source`/`tags`/`metadata_filters`/`content_type` whenever possible — an unscoped pattern scans entries sequentially. Regular-expression matching anchors `^`/`$` per line (the line-oriented / ripgrep model, via `MULTILINE`; `.` still does not cross newlines), and is guarded by BOTH a per-entry timeout and an aggregate scan budget: a pathological pattern fails gracefully — the offending entry is skipped, and once the aggregate budget is exhausted the scan stops with `truncated=true` — rather than hanging.

## navigate_context and the index_tree

`navigate_context` builds a per-record table of contents from the record's Markdown headings — the structure agents are already encouraged to write. The tree is code-derived (no LLM), parsed fresh on every call, so it is never stale and works for every entry, including those stored without an embedding provider. Heading detection is fenced-code-block aware (a `#` inside a code fence is ignored). A record with no headings collapses to a single synthetic root node spanning the whole document.

Each node carries a stable-ish `node_id` (a heading-path slug with a sibling ordinal, e.g. `setup`, `setup/install`, `notes-2`), its `level`, `title`, and `char_start`/`char_end` offsets. The synthetic root node's `summary` mirrors the record's stored entry summary by reference, which makes the summary the top-level node of the index_tree without overloading the flat summary column; it is null while the entry has no summary, such as an entry whose deferred generation job has not completed yet. Node ids are regenerated wholesale on every parse and are not stable across heading inserts or renames, nor across upgrades that change the slug algorithm itself (for example the per-segment length cap): stored per-node summaries whose persisted node_id no longer matches the recomputed one re-attach by exact char span, a node_id hit is trusted only when the stored row's span matches the recomputed node's span (a stale id can collide with a DIFFERENT section's current id after a slug-rule change), and a node_id captured before such an upgrade may no longer resolve in `read_context_range`.

### Optional per-node LLM summaries

When `ENABLE_INDEX_TREE_NODE_SUMMARIES` is on (the default), each substantial heading section also gets a short LLM-written abstract, stored in the `context_index_nodes` table. These summaries are surfaced by `navigate_context(include_node_summaries=True)`. They are an ADDITIVE layer, generated as a never-raise leg that starts right after the flat document summary completes and overlaps the embedding/compression work, so the abort-mandatory flat summary keeps precedence on the shared summary-model budget (governed by `SUMMARY_MAX_CONCURRENT`, which bounds the flat summary and every per-node summary together). By default the leg runs inside the store or update request; after a write with `defer_generation=true`, the entry's generation job runs it in the background, and until the job completes `navigate_context(include_node_summaries=True)` returns the outline without node summaries (a deferred text update removes the old text's node summaries when it stores the new text). It reuses the configured summary provider with a dedicated short prompt, bounded by a per-node timeout. A node-summary failure or timeout simply omits that node's summary and NEVER aborts a store or fails a generation job (unlike embeddings/summary/compression, which are abort-mandatory when enabled); on an abort-mandatory failure the in-flight node leg is canceled. Set `ENABLE_INDEX_TREE_NODE_SUMMARIES=false` to keep navigation purely code-derived with no per-store LLM cost and no node table.

## read_context_range details

`read_context_range` returns a slice of one record, addressed by exactly one of: a character range (`start_char`/`end_char`), a line range (`start_line`/`end_line`, 1-based inclusive), or an outline `node_id`. It slices the full stored `text_content` directly, so it is generation-independent and works for every entry. Out-of-range offsets are clamped to `[0, len(text)]` and the response echoes the resolved span, so an offset or node id captured in a prior turn degrades gracefully if the record was edited since.

## Generation state

The three tools read the stored text, so they serve an entry as soon as a write stores it, whatever the state of its embeddings and summaries. Each `grep_context` result (every row in `files_with_matches` and `count` mode, every match in `content` mode) and the top level of the `navigate_context` and `read_context_range` responses carry the entry's `generation_status`: `ready`, `pending` while a generation job waits or runs, or `failed` once the job used all of its attempts (see [Generation Status](api-reference.md#generation-status)). An entry stored with `defer_generation=true` can therefore be grepped and read at once, while its node summaries, and the entry summary its outline's root mirrors, appear once its job completes.

## Access scope

All three tools work only on the entries the caller may read (see the [Access Model](authentication.md#access-model)). `grep_context` scans, matches and counts readable entries alone, so an entry the caller cannot read never consumes the `max_entries_scanned` budget, never sets `truncated`, and never appears in the results. `navigate_context` and `read_context_range` report an unreadable entry as `Context entry not found: {id}`, exactly like an entry that does not exist, and an ID prefix resolves over readable entries only. Per-node summaries are read only for an entry the caller may read.

## Backend parity

Every behavior is identical on SQLite and PostgreSQL: all matching, line splitting, offset arithmetic, Markdown parsing, and slicing run in Python over `text_content` (which both backends already return). Offsets are Unicode code-point indices throughout, matching the chunking subsystem's discipline, so chained reads stay correct on multibyte text. The `context_index_nodes` table (created only when per-node summaries are enabled) uses a TEXT foreign key on SQLite and a UUID foreign key on PostgreSQL, both with `ON DELETE CASCADE`.

## Configuration

| Variable                                     | Default                 | Purpose                                                           |
|----------------------------------------------|-------------------------|-------------------------------------------------------------------|
| `ENABLE_GREP_CONTEXT`                        | `auto`                  | Register `grep_context` (auto/true/false)                         |
| `ENABLE_CONTEXT_NAVIGATION`                  | `auto`                  | Register `navigate_context`                                       |
| `ENABLE_CONTEXT_RANGE`                       | `auto`                  | Register `read_context_range`                                     |
| `GREP_MAX_MATCHES_CAP`                       | `1000`                  | Hard ceiling on `max_matches`                                     |
| `GREP_MAX_CONTEXT_LINES`                     | `20`                    | Hard ceiling on `context_lines`                                   |
| `GREP_MAX_ENTRIES_SCANNED`                   | `1000`                  | Hard ceiling on entries scanned                                   |
| `GREP_AGGREGATE_BYTES_BUDGET`                | `67108864`              | Approximate resident-memory cap for a scan                        |
| `GREP_REGEX_TIMEOUT_S`                       | `5.0`                   | Per-entry timeout for `is_regex=true` (ReDoS guard)               |
| `GREP_REGEX_TOTAL_TIMEOUT_S`                 | `30.0`                  | Aggregate budget for one whole `is_regex=true` scan               |
| `ENABLE_INDEX_TREE_NODE_SUMMARIES`           | `true`                  | Generate per-node LLM summaries + provision `context_index_nodes` |
| `INDEX_TREE_NODE_SUMMARY_PROMPT`             | (built-in short prompt) | Override the per-node summary system prompt                       |
| `INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH` | `500`                   | Skip node summaries for sections shorter than this                |
| `INDEX_TREE_NODE_SUMMARY_TIMEOUT_S`          | `240.0`                 | Per-node summary timeout                                          |
| `INDEX_TREE_NODE_SUMMARY_MAX_TOKENS`         | `120`                   | Output budget per node call; a cut-off answer drops the node      |
| `INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT`     | `min(cpu_count, 4)`     | Node-task fan-out cap; budget = `SUMMARY_MAX_CONCURRENT`          |

See [Environment Variables](environment-variables.md) for the full reference and [API Reference](api-reference.md) for exact parameters and return shapes.
