# Full-Text Search Guide

## Introduction

Full-Text Search (FTS) enables linguistic search with stemming, ranking, and boolean queries. Unlike semantic search which finds content by meaning, FTS performs traditional text search with advanced features:

- Finding exact phrases within context entries
- Prefix matching for partial word searches
- Boolean operators (AND, OR, NOT) for complex queries
- Language-specific stemming (e.g., "running" matches "run")
- BM25/ts_rank relevance scoring
- Highlighted snippets in search results

Full-text search is **auto-enabled by default** and can be used alongside or independently of semantic search. Because it relies only on built-in database capabilities, it needs no extra dependencies.

## Prerequisites

Full-Text Search has minimal requirements as it uses built-in database capabilities:

- **SQLite**: Version 3.35+ with FTS5 extension (included by default in Python 3.9+)
- **PostgreSQL**: Version 10+ with built-in tsvector support (no extensions required)
- **Python**: 3.12+ (already required by MCP Context Server)

No additional Python packages or external services are required.

## Installation

FTS functionality is built into the core MCP Context Server - no additional installation steps are needed. It is auto-enabled by default; no environment variable is required to turn it on.

## Configuration

### Environment Variables

Full-text search is controlled by the following environment variables in your MCP configuration:

#### ENABLE_FTS (Optional)

- **Type**: Tri-state
- **Default**: `auto`
- **Description**: Controls registration of the `fts_search_context` tool. `auto` (default) registers it automatically (full-text search uses built-in database capabilities, so no extra dependencies are needed). `true` forces it on. `false` forces it off, for the minimal tool surface. The boolean spellings `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off` are also accepted (`true` maps to force-on, `false` to force-off).
- **Example**: `"ENABLE_FTS": "false"` (only needed to disable; default already registers the tool)

#### FTS_LANGUAGE (Optional)

- **Type**: String
- **Default**: `english`
- **Description**: Language for stemming and text search
- **Example**: `"FTS_LANGUAGE": "german"`

**Supported Languages** (29 total):

```text
arabic, armenian, basque, catalan, danish, dutch, english, finnish, french,
german, greek, hindi, hungarian, indonesian, irish, italian, lithuanian,
nepali, norwegian, portuguese, romanian, russian, serbian, simple, spanish,
swedish, tamil, turkish, yiddish
```

**Important Notes**:
- PostgreSQL uses this language for full stemming support (all 29 languages)
- SQLite uses `porter unicode61` tokenizer for English (with stemming) or `unicode61` for other languages (no stemming, but proper Unicode tokenization)
- Invalid language values are rejected at startup with a clear error message

### MCP Configuration Example

Full-text search is registered by default, so no `ENABLE_FTS` entry is required. Add to your `.mcp.json` file to customize the stemming language (or set `"ENABLE_FTS": "false"` to disable the tool entirely):

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "mcp-context-server"
      ],
      "env": {
        "FTS_LANGUAGE": "english"
      }
    }
  }
}
```

## Backend-Specific Implementation

### SQLite (FTS5)

SQLite uses the FTS5 extension with external content mode for efficient full-text search.

**Features**:
- **BM25 ranking**: Industry-standard relevance scoring algorithm
- **Tokenizer selection**: Porter stemmer for English, unicode61 for other languages
- **External content**: FTS index references main table (no data duplication)
- **Auto-sync triggers**: Index automatically updates on INSERT/UPDATE/DELETE

**Tokenizer Behavior**:

| FTS_LANGUAGE    | Tokenizer          | Stemming | "running" matches "run" |
|-----------------|--------------------|----------|-------------------------|
| `english`       | `porter unicode61` | Yes      | Yes                     |
| Other languages | `unicode61`        | No       | No                      |

**SQLite Stemming Limitation**: SQLite FTS5 only supports English stemming via the Porter stemmer. For non-English languages, use PostgreSQL for full stemming support, or use prefix mode (`mode='prefix'`) to match word beginnings.

### PostgreSQL (tsvector)

PostgreSQL uses native tsvector/tsquery functionality with GIN indexing.

**Features**:
- **ts_rank_cd scoring**: PostgreSQL's built-in relevance ranking with document coverage normalization
- **29 language support**: Full linguistic stemming for all supported languages
- **GIN index**: Optimized for fast full-text searches
- **Generated column**: tsvector automatically computed and stored

**PostgreSQL Advantages**:
- Full stemming support for all 29 languages
- More sophisticated query parsing with `websearch_to_tsquery`
- Native phrase search support with `phraseto_tsquery`

## Usage

### New Tool: fts_search_context

When FTS is enabled, a new MCP tool becomes available.

**Parameters**:
- `query` (str, required): Search query string
- `mode` (str, optional): Search mode - `match` (default), `prefix`, `phrase`, or `boolean`
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `highlight` (bool, optional): Include highlighted snippets in results (default: false)
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering**: The `metadata` and `metadata_filters` parameters work identically to `search_context`. For comprehensive documentation on operators, nested paths, and best practices, see the [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns**:
```json
{
  "query": "original search query",
  "mode": "match",
  "results": [
    {
      "id": "0190abcdef1234567890abcdef123456",
      "thread_id": "thread-abc",
      "source": "agent",
      "content_type": "text",
      "text_content": "matching context content...",
      "summary": "Full summary of the matching context content with key entities and decisions.",
      "is_text_content_truncated": true,
      "visibility": "private",
      "is_owner": true,
      "metadata": {"status": "completed"},
      "created_at": "2025-12-01T10:00:00Z",
      "updated_at": "2025-12-01T10:00:00Z",
      "tags": ["tag1", "tag2"],
      "scores": {
        "fts_score": 2.45,
        "fts_rank": null,
        "rerank_score": 0.87
      },
      "highlighted": "matching <mark>context</mark> content"
    }
  ],
  "count": 1,
  "language": "english",
  "stats": {
    "execution_time_ms": 12.34,
    "filters_applied": 2,
    "rows_returned": 1,
    "backend": "sqlite",
    "query_plan": "..."
  }
}
```

**Note:** The `stats` field is only included when `explain_query=True`. Every key of the block is then present, including `backend` (the active storage backend type, e.g. `"sqlite"` or `"postgresql"`) and `query_plan`. A request that fails parameter validation (for example an invalid `metadata_filters` operator) returns the same keys with the counters zeroed and `query_plan` set to `null`, since no query ran, so a client can read `stats["query_plan"]` unconditionally and only has to handle the null.

**Scores Object**:
- `fts_score`: BM25/ts_rank relevance score (HIGHER = better match)
- `fts_rank`: Always null for standalone FTS (no ranking)
- `rerank_score`: Cross-encoder relevance score (HIGHER = better, 0.0-1.0), present when reranking is enabled

### Search Modes

#### Match Mode (Default)

Standard word matching with implicit AND between terms.

```python
fts_search_context(query="error handling", mode="match")
```

Finds entries containing both "error" AND "handling" (with stemming applied).

#### Prefix Mode

Matches words starting with the query terms. Useful for autocomplete-style searches.

```python
fts_search_context(query="implement", mode="prefix")
```

Finds entries containing words starting with "implement" (implements, implementation, implementing).

**SQLite**: Transforms query to `implement*`
**PostgreSQL**: Transforms query to `implement:*`

#### Phrase Mode

Matches exact phrases in order.

```python
fts_search_context(query="connection refused", mode="phrase")
```

Finds entries containing the exact phrase "connection refused" (words must appear adjacent and in order).

#### Boolean Mode

Supports complex queries with AND, OR, NOT operators.

```python
fts_search_context(query="python OR javascript NOT typescript", mode="boolean")
```

**SQLite syntax**: `word1 AND word2`, `word1 OR word2`, `NOT word`
**PostgreSQL syntax**: Uses `websearch_to_tsquery` which accepts Google-like syntax with quotes, `-` for NOT, and `OR`

### Example Use Cases

1. **Simple search**: Find entries mentioning specific terms
   ```python
   fts_search_context(query="authentication", mode="match")
   ```

2. **Prefix autocomplete**: Find entries with partial word matches
   ```python
   fts_search_context(query="auth", mode="prefix")
   ```

3. **Exact phrase**: Find specific error messages
   ```python
   fts_search_context(query="connection timed out", mode="phrase")
   ```

4. **Boolean query**: Complex search with operators
   ```python
   fts_search_context(query="database AND (postgres OR sqlite) NOT memory", mode="boolean")
   ```

5. **Filtered search**: Combine FTS with metadata filtering
   ```python
   fts_search_context(
       query="performance",
       thread_id="current-task",
       metadata={"status": "completed"},
       highlight=True
   )
   ```

6. **Time-bounded search**: Find recent content matching terms
   ```python
   fts_search_context(query="deployment", start_date="2025-11-01", end_date="2025-11-30")
   ```

7. **Highlighted snippets**: Get marked-up results for display
   ```python
   fts_search_context(query="error handling", highlight=True)
   ```
   Returns `highlighted` field with `<mark>` tags around matched terms.

## Cross-Encoder Reranking

When reranking is enabled (default), FTS results are refined using a cross-encoder model for improved precision.

### How FTS Reranking Works

1. **Candidate retrieval**: FTS retrieves the full ranked depth of candidates
2. **Passage Extraction**: For each candidate, a passage around the FTS match is extracted
3. **Cross-Encoder Scoring**: FlashRank scores query-passage pairs
4. **Re-ordering**: Results are sorted by cross-encoder score
5. **Final Selection**: The requested page is cut from that ordering

Every ranked search serves its page from ONE ordering of at most 100 candidates (the ranked depth, which equals the largest page a single request can ask for), so the candidate pool is the same for every page of a query and does not scale with the requested `limit` or `offset`. A page whose window reaches past that depth is served short, and the response says so through the `rank_depth_limit` key.

### Configuration

| Variable                 | Default                   | Description                                    |
|--------------------------|---------------------------|------------------------------------------------|
| `ENABLE_RERANKING`       | `true`                    | Enable cross-encoder reranking                 |
| `RERANKING_PROVIDER`     | `flashrank`               | Reranking provider                             |
| `RERANKING_MODEL`        | `ms-marco-MiniLM-L-12-v2` | Model (~34MB, downloads on startup)            |
| `FTS_RERANK_WINDOW_SIZE` | `750`                     | Characters around match for passage extraction |
| `FTS_RERANK_GAP_MERGE`   | `100`                     | Gap threshold for merging adjacent highlights  |

### FTS-Specific Passage Extraction

Unlike semantic search which uses full text, FTS reranking uses intelligent passage extraction:

- **Window Size**: Expands around FTS match highlights by `FTS_RERANK_WINDOW_SIZE` characters
- **Boundary Alignment**: Aligns to sentence/paragraph boundaries when possible
- **Gap Merging**: Merges nearby highlights if gap is less than `FTS_RERANK_GAP_MERGE` characters

This ensures the cross-encoder sees the most relevant context around each match.

### When to Disable FTS Reranking

Set `ENABLE_RERANKING=false` if:
- Search latency is critical (reranking adds ~50-100ms)
- Running on resource-constrained systems
- BM25/ts_rank ranking is sufficient for your use case

For more details, see [Semantic Search - Cross-Encoder Reranking](semantic-search.md#cross-encoder-reranking).

### Performance Characteristics

- **Index Updates**: Automatic via triggers (SQLite) or generated column (PostgreSQL)
- **Search Speed**: O(log n) with proper indexing
- **Storage Impact**: ~10-20% of text_content size for index
- **Concurrent Access**: Full support via WAL (SQLite) or MVCC (PostgreSQL)

## Verification

### Complete Setup Checklist

1. **Confirm FTS is not force-disabled**: FTS registers by default (`ENABLE_FTS=auto`), so no action is needed unless you previously set `ENABLE_FTS=false`. To check:
   ```bash
   echo $ENABLE_FTS  # Linux/macOS
   echo %ENABLE_FTS% # Windows
   # Empty or "auto" (default), or "true" forces it on; "false" disables it
   ```

2. **Start the server** (FTS auto-registers; no flag required):
   ```bash
   uv run mcp-context-server
   ```

3. **Check server logs** for:
   ```text
   fts_search_context registered
   ```

4. **Verify MCP client** - List available tools and confirm `fts_search_context` is present

5. **Test functionality**:
   ```python
   fts_search_context(query="test", mode="match")
   ```

### Verify FTS Index Status

Call `get_statistics` to check FTS availability:

```json
{
  "fts": {
    "available": true,
    "indexed_entries": 1000,
    "total_entries": 1000,
    "coverage_percentage": 100.0,
    "backend": "sqlite",
    "engine": "fts5"
  }
}
```

## Troubleshooting

### Issue 1: fts_search_context Not Available

**Error**: `fts_search_context not available` or tool not listed

**Diagnostic Steps**:

1. **Check the environment variable**:
   ```bash
   echo $ENABLE_FTS  # If it shows "false", FTS has been force-disabled
   ```

2. **Check server logs** for FTS initialization messages

3. **Call `get_statistics` tool**: Check `fts.available` field in response

**Solution**: FTS registers by default. If the tool is missing, ensure `ENABLE_FTS` is not set to `false` (remove it to fall back to `auto`, or set `ENABLE_FTS=true` to force registration).

### Issue 2: Invalid FTS_LANGUAGE Error

**Error**: `ValueError: FTS_LANGUAGE='xyz' is not a valid PostgreSQL text search configuration`

**Cause**: Invalid language name specified

**Solution**: Use one of the 29 supported language names (see Configuration section above).

### Issue 3: No Stemming with Non-English Language (SQLite)

**Symptom**: Search for "running" doesn't match "run" when using German, French, etc.

**Cause**: SQLite FTS5 only supports English stemming via Porter stemmer

**Solutions**:

1. **Use prefix mode**: Query with `mode='prefix'` to match word beginnings
   ```python
   fts_search_context(query="runn", mode="prefix")
   ```

2. **Switch to PostgreSQL**: PostgreSQL supports full stemming for all 29 languages

3. **Accept limitation**: Unicode61 tokenizer still provides proper Unicode tokenization for all languages

### Issue 4: Empty Search Results

**Possible Causes**:
- Query terms don't exist in stored context
- Wrong search mode for query syntax
- Date filters excluding all results

**Diagnostic Steps**:

1. Try broader search without filters:
   ```python
   fts_search_context(query="the", mode="match", limit=10)
   ```

2. Check if data exists:
   ```python
   search_context(limit=10)
   ```

3. Verify FTS index coverage via `get_statistics`

### Issue 5: Boolean Query Syntax Errors

**Symptom**: Boolean queries not working as expected

**Solution**: Match syntax to backend:

**SQLite** uses FTS5 syntax:
- AND: `word1 AND word2`
- OR: `word1 OR word2`
- NOT: `NOT word`
- Grouping: `(word1 OR word2) AND word3`

**PostgreSQL** uses websearch_to_tsquery syntax:
- OR: `word1 or word2`
- NOT: `-word` (dash prefix)
- Phrase: `"exact phrase"`
- Default is AND between terms
- Grouping: none. Parentheses are ignored and `or` binds looser than AND, so `python async or python await` means `(python AND async) OR (python AND await)`: repeat a shared term on each side of `or`

### Common Error Messages

| Error Message                       | Cause                           | Solution                                                     |
|-------------------------------------|---------------------------------|--------------------------------------------------------------|
| `fts_search_context not available`  | FTS force-disabled              | Unset `ENABLE_FTS` (default `auto`) or set `ENABLE_FTS=true` |
| `Invalid FTS_LANGUAGE`              | Unknown language                | Use valid language name                                      |
| `FTS5 table not found`              | Migration not applied           | Restart server to apply migration                            |
| `text_search_vector column missing` | PostgreSQL migration incomplete | Restart server to apply migration                            |

## Changing FTS Language

**Important**: Changing `FTS_LANGUAGE` after initial setup requires re-indexing.

### Migration Behavior

When you change `FTS_LANGUAGE`:

1. **SQLite**: FTS5 virtual table is dropped and recreated with new tokenizer
2. **PostgreSQL**: Generated column is dropped and recreated with new language

Both operations automatically rebuild the index from existing data.

### Steps to Change Language

1. **Stop the MCP server**

2. **Update environment variable**:
   ```bash
   export FTS_LANGUAGE=german  # New language
   ```

3. **Restart the server** - migration runs automatically

4. **Verify via logs**:
   ```text
   [INFO] FTS language changed from 'english' to 'german'
   [INFO] Rebuilding FTS index...
   [OK] FTS index rebuilt successfully
   ```

### Data Safety

- Context entries are **never deleted** during language migration
- Only the FTS index is rebuilt
- Queries may return empty results during rebuild

## Comparison: FTS vs Semantic Search

| Feature                   | Full-Text Search           | Semantic Search                    |
|---------------------------|----------------------------|------------------------------------|
| **Query Type**            | Keywords/phrases           | Natural language meaning           |
| **Stemming**              | Yes (language-specific)    | N/A                                |
| **Synonym Support**       | No                         | Yes (via embeddings)               |
| **Phrase Matching**       | Yes                        | Approximate                        |
| **Boolean Operators**     | Yes                        | No                                 |
| **External Dependencies** | None                       | Ollama + embedding model           |
| **Storage Overhead**      | ~10-20%                    | ~3KB per entry                     |
| **Best For**              | Exact matches, known terms | Concept discovery, similar content |

**Recommendation**: Enable both for maximum flexibility. Use FTS when you know the exact terms, semantic search when exploring related concepts.

## Additional Resources

### Official Documentation

- **SQLite FTS5**: [sqlite.org/fts5.html](https://www.sqlite.org/fts5.html)
- **PostgreSQL Full Text Search**: [postgresql.org/docs/current/textsearch.html](https://www.postgresql.org/docs/current/textsearch.html)

### Algorithm References

- **BM25**: [en.wikipedia.org/wiki/Okapi_BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- **Porter Stemmer**: [tartarus.org/martin/PorterStemmer](https://tartarus.org/martin/PorterStemmer/)

### Related Documentation

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation
- **Summary Generation**: [Summary Generation Guide](summary-generation.md) - LLM-based automatic summarization for search results
- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - meaning-based search with embeddings
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search with RRF fusion
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - filtering results with metadata operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication
- **Main Documentation**: [README.md](../README.md) - overview and quick start
