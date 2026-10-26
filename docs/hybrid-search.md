# Hybrid Search Guide

## Introduction

Hybrid Search combines the strengths of Full-Text Search (FTS) and Semantic Search using Reciprocal Rank Fusion (RRF), a proven algorithm used by Elasticsearch, Weaviate, and other search platforms. This approach leverages:

- **Full-Text Search**: Exact keyword matching, stemming, phrase search, and boolean queries
- **Semantic Search**: Meaning-based similarity using vector embeddings
- **RRF Fusion**: Rank-based algorithm that combines results without score normalization

This combination is particularly powerful for:

- Finding content that matches both keywords AND meaning
- Boosting results that appear in both search methods (high confidence matches)
- Graceful fallback when one search method is unavailable
- Cross-domain queries where exact terms and related concepts matter

Hybrid search is **auto-enabled by default**: it registers automatically whenever at least one underlying search method (full-text or semantic) is available.

## Prerequisites

Hybrid Search requires at least one underlying search method to be available. Both are auto-enabled by default:

- **Full-Text Search (FTS)**: `ENABLE_FTS=auto` (default; no additional dependencies, so it is normally available)
- **Semantic Search**: `ENABLE_SEMANTIC_SEARCH=auto` (default; available whenever an embedding provider is present, which it is when embedding generation is on by default)

With the defaults, hybrid search works out of the box. The explicit `true` form remains available (for the prerequisite-gated toggles it registers the tool only when the prerequisites are present, logging a warning otherwise):

```bash
ENABLE_FTS=true
ENABLE_SEMANTIC_SEARCH=true
ENABLE_HYBRID_SEARCH=true
```

**Dependencies by Search Mode:**

| Search Mode        | Dependencies                                 | Setup Guide                                   |
|--------------------|----------------------------------------------|-----------------------------------------------|
| FTS only           | None (built-in)                              | [Full-Text Search Guide](full-text-search.md) |
| Semantic only      | Ollama, embedding model, sqlite-vec/pgvector | [Semantic Search Guide](semantic-search.md)   |
| Both (recommended) | All above                                    | Both guides                                   |

## Installation

Hybrid Search uses existing FTS and Semantic Search infrastructure. No additional installation is required beyond the dependencies for your chosen search modes.

For full hybrid search capability:

```bash
# Install embedding provider and reranking dependencies (e.g., Ollama)
uv sync --extra embeddings-ollama --extra reranking

# Or use another provider: embeddings-openai, embeddings-azure, embeddings-huggingface, embeddings-voyage

# Pull embedding model (for Ollama)
ollama pull qwen3-embedding:0.6b
```

**Note:** The `--extra reranking` is necessary to enable reranking.

## Configuration

### Environment Variables

Hybrid search is controlled by the following environment variables in your MCP configuration:

#### ENABLE_HYBRID_SEARCH (Optional)

- **Type**: Tri-state
- **Default**: `auto`
- **Description**: Controls registration of the `hybrid_search_context` tool. `auto` (default) and `true` both register it when at least one underlying search method is available, otherwise skip it with a warning (hybrid has no underlying mode to fuse, so `true` cannot force the tool on with neither); `false` forces it off, for the minimal tool surface. The boolean spellings `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off` are also accepted (they map to `true`/`false`).
- **Example**: `"ENABLE_HYBRID_SEARCH": "false"` (only needed to disable; default already registers the tool)

**Note**: Hybrid search still requires at least one underlying search method to be available. With the defaults (`ENABLE_FTS=auto`, `ENABLE_SEMANTIC_SEARCH=auto`), this condition is normally satisfied; if you force both underlying methods off, the hybrid tool is skipped even when `ENABLE_HYBRID_SEARCH=true`.

#### HYBRID_RRF_K (Optional)

- **Type**: Integer
- **Default**: `60`
- **Range**: 1-1000
- **Description**: RRF smoothing constant controlling how much emphasis is placed on top-ranked vs. lower-ranked documents
- **Example**: `"HYBRID_RRF_K": "60"`

**Understanding RRF k Parameter:**

| k Value       | Behavior                                                                      |
|---------------|-------------------------------------------------------------------------------|
| Lower (10-30) | More emphasis on top-ranked documents; larger score differences between ranks |
| Default (60)  | Balanced approach; industry standard used by Elasticsearch                    |
| Higher (100+) | More uniform treatment across all ranks; smaller score differences            |

### MCP Configuration Example

With the defaults, FTS, semantic, and hybrid search all auto-register, so no `ENABLE_*` entries are required. Installing the embedding provider extra (here, Ollama) is what makes semantic search available, and hybrid follows automatically. Add to your `.mcp.json` file (the `HYBRID_RRF_K` entry below is optional tuning):

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--with",
        "mcp-context-server[embeddings-ollama,reranking]",
        "mcp-context-server"
      ],
      "env": {
        "HYBRID_RRF_K": "60"
      }
    }
  }
}
```

**Note:** The `--extra reranking` is necessary to enable reranking.

### Adaptive FTS Mode

By default, hybrid search uses `match` mode (AND logic) for FTS queries, requiring all terms to appear in a document. For long queries typical of LLM agents (4+ terms), this often returns zero FTS results, degrading hybrid search to semantic-only.

The adaptive FTS mode automatically switches long queries to OR logic:

| Query Length          | FTS Mode  | Logic                    | Example                            |
|-----------------------|-----------|--------------------------|------------------------------------|
| 1-3 significant terms | `match`   | AND (all terms required) | "python async"                     |
| 4+ significant terms  | `boolean` | OR (any term matches)    | "DRY extraction embedding helper"  |

**Configuration:**

```bash
HYBRID_FTS_OR_THRESHOLD=4  # Default: switch to OR at 4+ terms
```

**How it works:**
1. The query is split into words; single-character words are excluded from the count
2. If significant word count >= threshold, terms are joined with OR keywords
3. Hyphens are replaced with spaces to prevent NOT operator interpretation
4. The transformed query uses `boolean` mode (`websearch_to_tsquery` on PostgreSQL, FTS5 boolean on SQLite)

**Tuning:** Use `explain_query=True` to see the `adaptive_fts_mode` field in stats. If too many irrelevant results appear, increase the threshold. If FTS still returns zero results frequently, decrease it.

## How RRF Fusion Works

### The RRF Algorithm

Reciprocal Rank Fusion combines results from multiple search methods using a simple yet effective formula:

```text
RRF_score(d) = sum(1 / (k + rank_i(d))) for each search method i
```

Where:
- `d` is a document
- `k` is the smoothing constant (default: 60)
- `rank_i(d)` is the rank of document `d` in result list `i` (1-based)

### Why RRF Works

**1. Rank-based, not score-based:** RRF uses positions rather than raw scores, avoiding the need to normalize different scoring systems (BM25 vs L2 distance).

**2. Documents in both lists score higher:** A document ranked #1 in FTS and #1 in semantic search gets:
```text
RRF = 1/(60+1) + 1/(60+1) = 0.0328
```
While a document ranked #1 in only one list gets:
```text
RRF = 1/(60+1) = 0.0164
```

**3. Graceful handling of unique results:** Documents appearing in only one search method still receive a score and can rank highly if their single-source rank is good.

### Visual Example

```text
FTS Results:           Semantic Results:      After RRF Fusion:
1. Doc A (score 2.5)   1. Doc B (dist 0.15)   1. Doc B (rrf 0.0328) [in both]
2. Doc B (score 2.1)   2. Doc C (dist 0.22)   2. Doc A (rrf 0.0164) [FTS only]
3. Doc D (score 1.8)   3. Doc A (dist 0.35)   3. Doc C (rrf 0.0164) [semantic only]
                                              4. Doc D (rrf 0.0159) [FTS only]
```

Doc B appears in both lists (FTS rank 2, semantic rank 1), so it scores highest after fusion.

## Reranking Integration

When both hybrid search and reranking are enabled (both are enabled by default), reranking is applied AFTER RRF fusion:

1. FTS and semantic search run in parallel
2. RRF fusion combines results
3. Cross-encoder reranking refines final ordering

This ensures documents found by both methods rank highest, then reranking optimizes relevance.

### Candidate Depth

The candidate pool is FIXED and page-independent: it does not grow with the requested `limit` or `offset`. Both RRF and the cross-encoder are reordering stages, so a pool sized from the requested page would produce a different ordering for every page, and adjacent pages would repeat rows while other rows were never returned at all.

```text
User requests: limit=5, offset=0
    |
    v
Each leg fetches: 100 (ranked depth) * 2 (HYBRID_RRF_OVERFETCH) = 200 candidates
    |
    v
RRF fuses both legs into one ordering, kept to the ranked depth: 100 documents
    |
    v
Cross-encoder scores that window and reorders it
    |
    v
The requested page is cut from the result: 5 rows
```

A page whose window reaches past the ranked depth is served short, and the response says so through the `rank_depth_limit` key.

### Configuration

Reranking is controlled by these environment variables (see [Semantic Search Guide](semantic-search.md#cross-encoder-reranking) for details):

| Variable                 | Default   | Description                                         |
|--------------------------|-----------|-----------------------------------------------------|
| `ENABLE_RERANKING`       | `true`    | Enable cross-encoder reranking                      |
| `HYBRID_RRF_OVERFETCH`   | `2`       | Multiplier applied to the ranked depth for each leg |

## Usage

### New Tool: hybrid_search_context

When hybrid search is enabled and at least one underlying search method is available, a new MCP tool becomes available.

**Parameters:**

- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `fusion_method` (str, optional): Fusion algorithm - currently only `'rrf'` supported
- `rrf_k` (int, optional): RRF smoothing constant (1-1000, default from settings)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic; at most 100 tags per request)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators (`in`/`not_in` value lists accept at most 100 members)
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering**: The `metadata` and `metadata_filters` parameters work identically to `search_context`. For comprehensive documentation on operators, nested paths, and best practices, see the [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:**

```json
{
  "query": "authentication implementation",
  "results": [
    {
      "id": "0190abcdef1234567890abcdef123456",
      "thread_id": "project-alpha",
      "source": "agent",
      "content_type": "text",
      "text_content": "Implemented JWT authentication...",
      "summary": "JWT authentication implementation with token validation, role-based access control, and session management for the project-alpha backend API.",
      "is_text_content_truncated": true,
      "visibility": "private",
      "is_owner": true,
      "generation_status": "ready",
      "metadata": {"status": "completed", "priority": 8},
      "created_at": "2025-12-01T10:00:00Z",
      "updated_at": "2025-12-01T10:00:00Z",
      "tags": ["auth", "backend"],
      "scores": {
        "rrf": 0.0328,
        "fts_rank": 2,
        "semantic_rank": 1,
        "fts_score": 2.45,
        "semantic_distance": 0.234,
        "rerank_score": 0.95
      }
    }
  ],
  "count": 15,
  "fusion_method": "rrf",
  "search_modes_used": ["fts", "semantic"],
  "fts_count": 12,
  "semantic_count": 10,
  "stats": {
    "execution_time_ms": 125.5,
    "fts_stats": {
      "execution_time_ms": 15.2,
      "filters_applied": 2,
      "rows_returned": 12,
      "backend": "sqlite",
      "query_plan": "..."
    },
    "semantic_stats": {
      "execution_time_ms": 85.3,
      "embedding_generation_ms": 45.1,
      "filters_applied": 2,
      "rows_returned": 10,
      "backend": "sqlite",
      "query_plan": "..."
    },
    "fusion_stats": {
      "rrf_k": 60,
      "total_unique_documents": 15,
      "documents_in_both": 7,
      "documents_fts_only": 5,
      "documents_semantic_only": 3
    },
    "adaptive_fts_mode": "match"
  }
}
```

**Note:** The `stats` field is only included when `explain_query=True`.

Each result reports `visibility`, `is_owner` (true when the caller owns the entry) and `generation_status` (`ready`, `pending` or `failed`; see [Generation Status](api-reference.md#generation-status)).

### Understanding the Stats Field

When `explain_query=True`, the response includes a `stats` object with detailed execution statistics:

| Field               | Type           | Description                                                        |
|---------------------|----------------|--------------------------------------------------------------------|
| `execution_time_ms` | float          | Total hybrid search execution time                                 |
| `fts_stats`         | object or null | FTS search statistics (null if FTS not used)                       |
| `semantic_stats`    | object or null | Semantic search statistics (null if semantic not used)             |
| `fusion_stats`      | object         | RRF fusion statistics                                              |
| `adaptive_fts_mode` | string         | FTS mode selected by the adaptive AND/OR switch (match or boolean) |

**FTS Stats:**

`fts_stats` itself is `null` (not an object) when the FTS leg was rejected by parameter validation and never ran; the fields below only exist when the leg actually ran:
- `execution_time_ms`: FTS search execution time
- `filters_applied`: Total number of filter conditions applied (thread, source, content type, date bounds and the tag subquery, plus every metadata condition)
- `rows_returned`: Number of FTS results before fusion
- `backend`: Active storage backend ('sqlite' or 'postgresql'), always present
- `query_plan`: Backend query execution plan

**Semantic Stats:**

`semantic_stats` itself is `null` (not an object) when the semantic leg was rejected by parameter validation and never ran; the fields below only exist when the leg actually ran:
- `execution_time_ms`: Semantic search execution time
- `embedding_generation_ms`: Time spent generating the query embedding via the configured embedding provider
- `filters_applied`: Total number of filter conditions applied (thread, source, content type, date bounds and the tag subquery, plus every metadata condition)
- `rows_returned`: Number of semantic results before fusion
- `backend`: Active storage backend ('sqlite' or 'postgresql'), always present
- `query_plan`: Backend query execution plan

**Fusion Stats:**
- `rrf_k`: RRF smoothing constant used
- `total_unique_documents`: Total unique documents after fusion
- `documents_in_both`: Documents found by both FTS and semantic search (high confidence)
- `documents_fts_only`: Documents found only by FTS
- `documents_semantic_only`: Documents found only by semantic search

### Understanding the Scores Field

Each result includes a `scores` object with detailed breakdown:

| Field               | Type          | Description                                                                                                                                                            |
|---------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rrf`               | float         | Combined RRF score (higher = better)                                                                                                                                   |
| `fts_rank`          | int or null   | Position in FTS results (1-based), null if not in FTS results                                                                                                          |
| `semantic_rank`     | int or null   | Position in semantic results (1-based), null if not in semantic results                                                                                                |
| `fts_score`         | float or null | Original FTS relevance score (BM25/ts_rank)                                                                                                                            |
| `semantic_distance` | float or null | Original semantic distance, lower = more similar (Euclidean L2 for uncompressed/`mse` storage; negated inner product ~ -1..0 for the default `ip` compression variant) |
| `rerank_score`      | float or null | Cross-encoder relevance score (higher = better, 0.0-1.0), null if reranking disabled                                                                                   |

**Interpreting null values:**

- `fts_rank: null, semantic_rank: 3` - Document found only via semantic search
- `fts_rank: 1, semantic_rank: null` - Document found only via FTS
- Both non-null - Document found by both methods (high confidence match)

### Graceful Degradation

Hybrid search automatically adapts when search methods are unavailable:

| FTS Available | Semantic Available | Behavior                                     |
|---------------|--------------------|----------------------------------------------|
| Yes           | Yes                | Full hybrid search with RRF fusion           |
| Yes           | No                 | FTS results only (semantic_rank always null) |
| No            | Yes                | Semantic results only (fts_rank always null) |
| No            | No                 | Error: "No search modes available"           |

**Common scenarios for partial availability:**

- Semantic unavailable: Ollama not running, embedding model not pulled
- FTS unavailable: FTS migration in progress, database corruption

The `search_modes_used` field in the response indicates which modes were actually executed.

### Entries Whose Generation Is Pending

With both modes available, a single entry can still be visible to only one of them. The full-text index covers an entry as soon as a write stores it, while the semantic leg ranks only entries that have embeddings. An entry stored with `defer_generation=true`, or whose text a deferred update replaced, therefore appears with `semantic_rank: null` and a rank from the full-text leg alone until its generation job stores the new embeddings, and so does an entry whose job `failed`. A deferred text update deletes the old text's embeddings in the same transaction, so the semantic leg never ranks the new text through the old vectors. See [Deferred Generation](semantic-search.md#deferred-generation).

### Example Use Cases

**1. High-confidence document discovery:**
Find documents matching both keywords AND meaning:
```python
hybrid_search_context(query="authentication token validation")
# Results with both fts_rank and semantic_rank are high-confidence matches
```

**2. Fallback to available search:**
Use hybrid even when uncertain which search methods are running:
```python
hybrid_search_context(query="error handling patterns")
# Works with whatever is available
```

**3. Filtered hybrid search:**
Combine hybrid search with metadata filtering:
```python
hybrid_search_context(
    query="performance optimization",
    thread_id="project-alpha",
    metadata={"status": "completed"},
    metadata_filters=[{"key": "priority", "operator": "gte", "value": 7}]
)
```

**4. Time-bounded search:**
Find matching content within a specific date range:
```python
hybrid_search_context(
    query="deployment issues",
    start_date="2025-11-01",
    end_date="2025-11-30"
)
```

**5. Tuning RRF for your use case:**
Adjust k parameter for different ranking behaviors:
```python
# Emphasize top results more (lower k)
hybrid_search_context(query="critical bug", rrf_k=30)

# More uniform treatment of all ranks (higher k)
hybrid_search_context(query="general information", rrf_k=100)
```

### Performance Characteristics

Hybrid search executes FTS and semantic search in parallel for optimal performance:

| Operation         | SQLite   | PostgreSQL |
|-------------------|----------|------------|
| FTS search        | 10-50ms  | 5-30ms     |
| Semantic search   | 50-200ms | 30-100ms   |
| Hybrid (parallel) | 55-220ms | 35-115ms   |
| RRF fusion        | <1ms     | <1ms       |

**Performance notes:**

- Searches run in parallel via `asyncio.gather`
- Over-fetch strategy (limit * 2) improves fusion quality with minimal overhead
- RRF fusion is extremely fast (simple arithmetic operations)
- Total time dominated by slower search method (usually semantic)

## Verification

### Complete Setup Checklist

1. **Confirm FTS is not force-disabled:** FTS registers by default (`ENABLE_FTS=auto`).
   ```bash
   echo $ENABLE_FTS  # Empty/"auto"/"true" is fine; "false" disables it
   ```

2. **Confirm semantic search prerequisites (if using semantic):**
   ```bash
   echo $ENABLE_SEMANTIC_SEARCH  # Empty/"auto"/"true" is fine; "false" disables it
   curl http://localhost:11434   # Should return: Ollama is running
   ollama list                   # Should show your embedding model
   ```

3. **Confirm hybrid search is not force-disabled:** Hybrid registers by default (`ENABLE_HYBRID_SEARCH=auto`) when at least one underlying method is available.
   ```bash
   echo $ENABLE_HYBRID_SEARCH  # Empty/"auto"/"true" is fine; "false" disables it
   ```

4. **Start server and check logs:**
   ```bash
   uv run mcp-context-server
   ```
   Look for:
   ```text
   hybrid_search_context modes available: ['fts', 'semantic']
   ```

5. **Verify MCP client** - List available tools and confirm `hybrid_search_context` is present

6. **Test functionality:**
   ```python
   hybrid_search_context(query="test query", limit=5)
   ```

### Verify via get_statistics

Call `get_statistics` to check hybrid search availability:

```json
{
  "fts": {
    "available": true,
    "indexed_entries": 1000
  },
  "semantic_search": {
    "available": true,
    "indexed_entries": 1000
  }
}
```

Both FTS and semantic search should show as available for full hybrid functionality.

## Troubleshooting

### Issue 1: hybrid_search_context Not Available

**Error**: Tool not listed or "Hybrid search is not available"

**Diagnostic Steps:**

1. **Check environment variables** (any of these set to `false` force-disables the corresponding feature):
   ```bash
   echo $ENABLE_HYBRID_SEARCH   # "false" disables hybrid
   echo $ENABLE_FTS             # "false" disables FTS
   echo $ENABLE_SEMANTIC_SEARCH # "false" disables semantic
   ```

2. **Check server logs** for initialization messages

3. **Call `get_statistics` tool** to verify underlying search methods

**Solution**: Hybrid registers by default when at least one underlying method is available. If the tool is missing, ensure none of `ENABLE_HYBRID_SEARCH`, `ENABLE_FTS`, or `ENABLE_SEMANTIC_SEARCH` is set to `false`, and that at least one underlying method (FTS or semantic) is actually available.

### Issue 2: Only FTS or Only Semantic Results

**Symptom**: Results have `semantic_rank: null` for all entries or `fts_rank: null` for all entries

**Cause**: One search method is unavailable

**Diagnostic Steps:**

1. Check `search_modes_used` in response - shows which modes actually executed
2. Check `fts_count` and `semantic_count` in response

**For missing semantic search:**
- Verify Ollama is running: `curl http://localhost:11434`
- Verify model is available: `ollama list`
- Ensure `ENABLE_SEMANTIC_SEARCH` is not set to `false` (it defaults to `auto`)

**For missing FTS:**
- Ensure `ENABLE_FTS` is not set to `false` (it defaults to `auto`)
- Verify FTS migration completed (check server logs)

### Issue 3: Poor Fusion Quality

**Symptom**: Results don't seem to combine well, or single-source results dominate

**Possible Causes:**

1. **Very different result sets**: FTS and semantic may return completely different documents
2. **One search returning few results**: Limited overlap for fusion
3. **Inappropriate k value**: May need tuning

**Solutions:**

1. **Check overlap**: Look at results - are any entries in both search methods?
2. **Increase limit**: More results = more potential overlap
3. **Tune k parameter**:
   - Lower k (30) for more top-heavy ranking
   - Higher k (100) for more uniform treatment
4. **Verify data has embeddings**: Check `get_statistics` for embedding coverage and its `generation` block for entries whose embeddings are still pending or failed

### Issue 4: Slow Hybrid Search

**Symptom**: Searches taking longer than expected

**Cause**: Usually semantic search is the bottleneck

**Solutions:**

1. **Check Ollama performance**: Ensure model is loaded in memory
   ```bash
   ollama ps  # Shows loaded models
   ```

2. **Increase Ollama keep-alive:**
   ```bash
   export OLLAMA_KEEP_ALIVE=3600  # Keep model loaded
   ```

3. **Use dedicated search tools** if only keywords needed:
   ```python
   fts_search_context(query="exact match")
   ```

### Common Error Messages

| Error Message                    | Cause                                               | Solution                                                          |
|----------------------------------|-----------------------------------------------------|-------------------------------------------------------------------|
| `Hybrid search is not available` | Feature force-disabled                              | Unset `ENABLE_HYBRID_SEARCH` (default `auto`) or set it to `true` |
| `No search modes available`      | Both FTS and semantic force-disabled or unavailable | Keep at least one search method available                         |
| `FTS requires ENABLE_FTS=true`   | FTS force-disabled                                  | Unset `ENABLE_FTS` (default `auto`) or set `ENABLE_FTS=true`      |
| `Semantic search requires...`    | Semantic dependencies missing                       | Set up Ollama and semantic search                                 |
| `All search modes failed`        | Both FTS and semantic errored                       | Check individual search method status                             |

## Comparison: Hybrid vs Individual Search Methods

| Feature                  | FTS                        | Semantic                 | Hybrid                         |
|--------------------------|----------------------------|--------------------------|--------------------------------|
| **Query Type**           | Keywords/phrases           | Natural language meaning | Both                           |
| **Result Ranking**       | BM25/ts_rank score         | L2 / negated-IP distance | RRF combined score             |
| **Best For**             | Exact matches, known terms | Concept discovery        | High-confidence matches        |
| **Performance**          | Fastest                    | Slower                   | Similar to semantic (parallel) |
| **Dependencies**         | None                       | Ollama + model           | At least one method            |
| **Graceful Degradation** | N/A                        | N/A                      | Falls back to available method |

**When to use each:**

- **FTS**: Known exact terms, phrase matching, boolean queries
- **Semantic**: Exploring related concepts, meaning-based retrieval
- **Hybrid**: Best of both, high-confidence discovery, uncertain query type

## Additional Resources

### Related Documentation

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation
- **Summary Generation**: [Summary Generation Guide](summary-generation.md) - LLM-based automatic summarization for search results
- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - semantic search setup with Ollama
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata filtering with operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication
- **Main Documentation**: [README.md](../README.md) - overview and quick start

### Algorithm References

- **Reciprocal Rank Fusion**: [Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- **Elasticsearch RRF**: [Elastic documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)
- **Weaviate Hybrid Search**: [Weaviate documentation](https://weaviate.io/developers/weaviate/search/hybrid)

### Implementation Files

- [`app/fusion.py`](../app/fusion.py) - RRF fusion algorithm implementation
- [`app/tools/search/hybrid.py`](../app/tools/search/hybrid.py) - hybrid_search_context tool definition
- [`app/settings/search.py`](../app/settings/search.py) - Search, hybrid and reranking settings
- [`app/types.py`](../app/types.py) - TypedDict definitions for hybrid search
