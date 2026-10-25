# Semantic Search Guide

## Introduction

Semantic search enables finding context entries based on meaning rather than exact keyword matching. Using vector embeddings from multiple providers, the MCP Context Server can understand semantic similarity between queries and stored context, making it powerful for:

- Finding related work across different threads
- Discovering similar contexts without shared keywords
- Concept-based retrieval from large context collections
- Cross-agent knowledge discovery

Semantic search is **auto-enabled by default** and supports multiple embedding providers via LangChain integration.

## Enablement

The `semantic_search_context` tool is controlled by the tri-state `ENABLE_SEMANTIC_SEARCH` variable:

- `auto` (default): the tool registers automatically whenever an embedding provider is available. When installed with an embedding provider extra (as in the examples below), a provider is present and semantic search registers automatically. With the bare package and no provider, set `ENABLE_EMBEDDING_GENERATION=false` (otherwise the server exits because the configured provider is unavailable).
- `true`: register the tool when an embedding provider is available. If none is, the server logs a warning and the tool is NOT registered (`true` cannot force it on without a provider).
- `false`: force the tool off, for the minimal tool surface.

The boolean spellings `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off` are also accepted (they map to `true`/`false`).

You still need the provider prerequisites below (an embedding provider plus its dependencies, and `sqlite-vec`/`pgvector` for vector storage) for semantic search to function; `auto` only governs whether the tool is exposed once those prerequisites are met. Embedding storage (the vector tables and chunk columns) is provisioned from `ENABLE_EMBEDDING_GENERATION`, decoupled from this toggle, so turning semantic search on later never requires re-embedding.

## Embedding Providers

The server supports 5 embedding providers:

| Provider             | Default Model                          | Dimensions | Cost            | Best For                       |
|----------------------|----------------------------------------|------------|-----------------|--------------------------------|
| **Ollama** (default) | qwen3-embedding:0.6b                   | 1024       | Free (local)    | Development, privacy-focused   |
| **OpenAI**           | text-embedding-3-small                 | 1536       | $0.02/1M tokens | Production, high quality       |
| **Azure OpenAI**     | text-embedding-ada-002                 | 1536       | Pay-as-you-go   | Enterprise, compliance         |
| **HuggingFace**      | sentence-transformers/all-MiniLM-L6-v2 | 384        | Free (API)      | Open source, experimentation   |
| **Voyage AI**        | voyage-3                               | 1024       | $0.06/1M tokens | RAG optimization, long context |

Select a provider via the `EMBEDDING_PROVIDER` environment variable.

## Prerequisites

- **Python**: 3.12+ (already required by MCP Context Server)
- **SQLite**: 3.35+ minimum, 3.41+ recommended (for SQLite backend)
- **PostgreSQL**: 14+ with pgvector extension (for PostgreSQL backend)
- **RAM**: 4GB minimum, 8GB recommended for local models

### Provider-Specific Requirements

| Provider     | Requirements                                     |
|--------------|--------------------------------------------------|
| Ollama       | Ollama installed locally, ~1GB storage per model |
| OpenAI       | OpenAI API key, internet access                  |
| Azure OpenAI | Azure subscription, deployed embedding model     |
| HuggingFace  | HuggingFace API token (optional for some models) |
| Voyage AI    | Voyage AI API key                                |

## Installation

### Step 1: Install Provider Dependencies

Each provider has its own optional dependencies:

```bash
# Ollama provider (default)
uv sync --extra embeddings-ollama

# OpenAI provider
uv sync --extra embeddings-openai

# Azure OpenAI provider
uv sync --extra embeddings-azure

# HuggingFace provider
uv sync --extra embeddings-huggingface

# Voyage AI provider
uv sync --extra embeddings-voyage

# All providers
uv sync --extra embeddings-all
```

### Step 2: Provider-Specific Setup

See the provider-specific sections below for detailed setup instructions.

## Provider Configuration

### Ollama (Default)

Ollama runs embedding models locally with no API costs.

#### Setup

1. **Install Ollama** from [ollama.com/download](https://ollama.com/download)

   **Windows**:
   - Download and run the .exe installer
   - Verify Windows Defender allows port 11434

   **macOS**:
   - Download the .dmg from [ollama.com/download/mac](https://ollama.com/download/mac)
   - Drag Ollama to Applications and launch

   **Important macOS Note**: The default macOS Python lacks SQLite extension support. You must use Homebrew Python:
   ```bash
   brew install python
   /opt/homebrew/bin/python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Linux**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull embedding model**:
   ```bash
   ollama pull qwen3-embedding:0.6b
   ```

3. **Verify installation**:
   ```bash
   ollama list
   curl http://localhost:11434
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,reranking]", "mcp-context-server"],
      "env": {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
        "EMBEDDING_DIM": "1024",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

**Note:** The `--extra reranking` is necessary to enable reranking.

#### Environment Variables

| Variable                       | Default                  | Description                                                                                          |
|--------------------------------|--------------------------|------------------------------------------------------------------------------------------------------|
| `EMBEDDING_PROVIDER`           | `ollama`                 | Set to `ollama`                                                                                      |
| `EMBEDDING_MODEL`              | `qwen3-embedding:0.6b`   | Ollama model name                                                                                    |
| `EMBEDDING_DIM`                | `1024`                   | Vector dimensions                                                                                    |
| `OLLAMA_HOST`                  | `http://localhost:11434` | Ollama API URL                                                                                       |
| `EMBEDDING_OLLAMA_TRUNCATE`    | `false`                  | Truncation mode: false (default) returns error when context exceeded, true enables silent truncation |
| `EMBEDDING_OLLAMA_NUM_CTX`     | `4096`                   | Context window size in tokens (range: 512-2097152)                                                   |

**Docker Networking**: Use `host.docker.internal:11434` (Windows/macOS) or `172.17.0.1:11434` (Linux) when running in containers.

**Instructed queries**: the default `qwen3-embedding:0.6b` model is instruct-aware — its model card prescribes an instruction prefix on the query side and omitting it costs roughly 1-5% retrieval quality. Set `EMBEDDING_QUERY_INSTRUCTION` to enable the prefix for search queries; the recommended value and the multi-line quoting mechanics are documented in the [Environment Variables Reference](environment-variables.md#embedding-settings).

#### Alternative Ollama Models

| Model                | Dimensions | Notes                                              |
|----------------------|------------|----------------------------------------------------|
| qwen3-embedding:0.6b | 1024       | Default, MTEB-R 61.82, 32K context, 100+ languages |
| nomic-embed-text     | 768        | Strong performance, English-focused                |
| mxbai-embed-large    | 1024       | Higher quality, slower                             |
| all-minilm           | 384        | Very fast, good for large-scale                    |

### OpenAI

OpenAI provides high-quality embeddings via API.

#### Setup

1. **Get API key** from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

2. **Install dependencies**:
   ```bash
   uv sync --extra embeddings-openai
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-openai,reranking]", "mcp-context-server"],
      "env": {
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIM": "1536",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

**Note:** The `--extra reranking` is necessary to enable reranking.

#### Environment Variables

| Variable              | Default                  | Description                  |
|-----------------------|--------------------------|------------------------------|
| `EMBEDDING_PROVIDER`  | -                        | Set to `openai`              |
| `EMBEDDING_MODEL`     | `text-embedding-3-small` | OpenAI model name            |
| `EMBEDDING_DIM`       | `1536`                   | Vector dimensions            |
| `OPENAI_API_KEY`      | -                        | **Required**: OpenAI API key |
| `OPENAI_API_BASE`     | -                        | Custom base URL (optional)   |
| `OPENAI_ORGANIZATION` | -                        | Organization ID (optional)   |

#### Available OpenAI Models

| Model                  | Dimensions | Price    | Notes                       |
|------------------------|------------|----------|-----------------------------|
| text-embedding-3-small | 1536       | $0.02/1M | Recommended, cost-effective |
| text-embedding-3-large | 3072       | $0.13/1M | Higher quality              |
| text-embedding-ada-002 | 1536       | $0.10/1M | Legacy model                |

### Azure OpenAI

Azure OpenAI provides enterprise-grade embeddings with compliance features.

#### Setup

1. **Create Azure OpenAI resource** in [Azure Portal](https://portal.azure.com)

2. **Deploy embedding model** in Azure OpenAI Studio

3. **Get credentials**:
   - Endpoint URL
   - API key
   - Deployment name

4. **Install dependencies**:
   ```bash
   uv sync --extra embeddings-azure
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-azure,reranking]", "mcp-context-server"],
      "env": {
        "EMBEDDING_PROVIDER": "azure",
        "EMBEDDING_MODEL": "text-embedding-ada-002",
        "EMBEDDING_DIM": "1536",
        "AZURE_OPENAI_API_KEY": "${AZURE_OPENAI_API_KEY}",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "your-deployment-name",
        "AZURE_OPENAI_API_VERSION": "2024-02-01"
      }
    }
  }
}
```

**Note:** The `--extra reranking` is necessary to enable reranking.

#### Environment Variables

| Variable                                 | Default      | Description                      |
|------------------------------------------|--------------|----------------------------------|
| `EMBEDDING_PROVIDER`                     | -            | Set to `azure`                   |
| `EMBEDDING_MODEL`                        | -            | Model name                       |
| `EMBEDDING_DIM`                          | `1536`       | Vector dimensions                |
| `AZURE_OPENAI_API_KEY`                   | -            | **Required**: Azure API key      |
| `AZURE_OPENAI_ENDPOINT`                  | -            | **Required**: Azure endpoint URL |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | -            | **Required**: Deployment name    |
| `AZURE_OPENAI_API_VERSION`               | `2024-02-01` | API version                      |

### HuggingFace

HuggingFace provides access to open-source embedding models.

#### Setup

1. **Get API token** (optional) from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Install dependencies**:
   ```bash
   uv sync --extra embeddings-huggingface
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-huggingface,reranking]", "mcp-context-server"],
      "env": {
        "EMBEDDING_PROVIDER": "huggingface",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "EMBEDDING_DIM": "384",
        "HUGGINGFACEHUB_API_TOKEN": "${HUGGINGFACEHUB_API_TOKEN}"
      }
    }
  }
}
```

**Note:** The `--extra reranking` is necessary to enable reranking.

#### Environment Variables

| Variable                   | Default                                  | Description                                      |
|----------------------------|------------------------------------------|--------------------------------------------------|
| `EMBEDDING_PROVIDER`       | -                                        | Set to `huggingface`                             |
| `EMBEDDING_MODEL`          | `sentence-transformers/all-MiniLM-L6-v2` | Model identifier                                 |
| `EMBEDDING_DIM`            | `384`                                    | Vector dimensions                                |
| `HUGGINGFACEHUB_API_TOKEN` | -                                        | HuggingFace API token (optional for some models) |

#### Recommended HuggingFace Models

| Model                                   | Dimensions | Notes                         |
|-----------------------------------------|------------|-------------------------------|
| sentence-transformers/all-MiniLM-L6-v2  | 384        | Fast, good quality            |
| sentence-transformers/all-mpnet-base-v2 | 768        | Higher quality                |
| BAAI/bge-small-en-v1.5                  | 384        | BGE series, excellent quality |
| BAAI/bge-base-en-v1.5                   | 768        | BGE series, balanced          |

### Voyage AI

Voyage AI specializes in RAG-optimized embeddings with long context support.

#### Setup

1. **Get API key** from [dash.voyageai.com](https://dash.voyageai.com)

2. **Install dependencies**:
   ```bash
   uv sync --extra embeddings-voyage
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-voyage,reranking]", "mcp-context-server"],
      "env": {
        "EMBEDDING_PROVIDER": "voyage",
        "EMBEDDING_MODEL": "voyage-3",
        "EMBEDDING_DIM": "1024",
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}"
      }
    }
  }
}
```

**Note:** The `--extra reranking` is necessary to enable reranking.

#### Environment Variables

| Variable             | Default    | Description                                                                                          |
|----------------------|------------|------------------------------------------------------------------------------------------------------|
| `EMBEDDING_PROVIDER` | -          | Set to `voyage`                                                                                      |
| `EMBEDDING_MODEL`    | `voyage-3` | Voyage model name                                                                                    |
| `EMBEDDING_DIM`      | `1024`     | Vector dimensions                                                                                    |
| `VOYAGE_API_KEY`     | -          | **Required**: Voyage AI API key                                                                      |
| `VOYAGE_TRUNCATION`  | `false`    | Truncation mode: false (default) returns error when context exceeded, true enables silent truncation |
| `VOYAGE_BATCH_SIZE`  | `7`        | Texts per API call (1-128)                                                                           |

#### Available Voyage Models

| Model            | Dimensions | Context | Notes               |
|------------------|------------|---------|---------------------|
| voyage-3         | 1024       | 32K     | Latest, recommended |
| voyage-3-lite    | 512        | 32K     | Faster, lower cost  |
| voyage-code-3    | 1024       | 32K     | Optimized for code  |
| voyage-finance-2 | 1024       | 32K     | Financial domain    |
| voyage-law-2     | 1024       | 32K     | Legal domain        |

## LangSmith Tracing

LangSmith provides observability for embedding operations, including:

- Cost tracking per request
- Latency monitoring
- Error debugging
- Usage analytics

### Installation

LangSmith tracing requires the optional `langsmith` dependency:

```bash
# For local development
uv sync --extra langsmith

# For uvx (combined with embeddings, e.g., Ollama)
uvx --python 3.12 --with "mcp-context-server[embeddings-ollama,langsmith,reranking]" mcp-context-server
```

**Graceful Degradation**: If `LANGSMITH_TRACING=true` but the `langsmith` package is not installed, the server will log a warning once and continue operating without tracing.

### Setup

1. **Create account** at [smith.langchain.com](https://smith.langchain.com)

2. **Get API key** from settings

3. **Install dependencies**: `uv sync --extra langsmith`

4. **Enable tracing** in your configuration:

```json
{
  "env": {
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_API_KEY": "${LANGSMITH_API_KEY}",
    "LANGSMITH_PROJECT": "mcp-context-server"
  }
}
```

### Environment Variables

| Variable             | Default                           | Description                      |
|----------------------|-----------------------------------|----------------------------------|
| `LANGSMITH_TRACING`  | `false`                           | Enable/disable tracing           |
| `LANGSMITH_API_KEY`  | -                                 | LangSmith API key                |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | API endpoint                     |
| `LANGSMITH_PROJECT`  | `mcp-context-server`              | Project name for grouping traces |

### Viewing Traces

1. Navigate to [smith.langchain.com](https://smith.langchain.com)
2. Select your project
3. View embedding operations with timing and cost data

## Text Chunking

Text chunking splits long context entries into smaller chunks for embedding generation. This improves semantic search quality for long documents by ensuring each chunk fits within the embedding model's optimal context window.

**Key point:** Chunking is enabled by default. No configuration is required.

### Configuration

| Variable                | Default | Description                                                                       |
|-------------------------|---------|-----------------------------------------------------------------------------------|
| `ENABLE_CHUNKING`       | `true`  | Enable text chunking for embedding generation                                     |
| `CHUNK_SIZE`            | `1500`  | Target chunk size in characters                                                   |
| `CHUNK_OVERLAP`         | `150`   | Overlap between consecutive chunks                                                |
| `CHUNK_AGGREGATION`     | `max`   | How to aggregate chunk scores (only 'max' supported; avg, sum planned for future) |

### How It Works

1. **Text Splitting**: Long text is split using RecursiveCharacterTextSplitter with separators: `['\n\n', '\n', '. ', ' ', '']`
2. **Embedding Generation**: Each chunk is embedded independently; the chunks reach the provider in batches (see [Batched Document Embedding](#batched-document-embedding))
3. **Storage**: Multiple embeddings stored per context entry (1:N relationship)
4. **Search Deduplication**: Results are deduplicated by context_id using the configured aggregation method

### Aggregation Methods

Currently only `max` aggregation is supported. The `avg` and `sum` methods will be added in a future release.

### When to Adjust Settings

- **Increase CHUNK_SIZE**: For documents with long cohesive sections
- **Increase CHUNK_OVERLAP**: To preserve context across chunk boundaries

## Cross-Encoder Reranking

Reranking uses a cross-encoder model (FlashRank) to improve search result precision by re-scoring candidates using the full query-document context.

**Key point:** Reranking is enabled by default. The FlashRank model (~34MB) downloads automatically during server startup.

### Installation

Reranking requires the `reranking` extra which contains the `flashrank` package:

```bash
# For local development
uv sync --extra reranking

# For uvx (combined with embeddings, e.g., Ollama)
uvx --python 3.12 --with "mcp-context-server[embeddings-ollama,reranking]" mcp-context-server
```

**Important:** Since `ENABLE_RERANKING=true` by default, you must install the `reranking` extra for the server to start successfully. To run without reranking, set `ENABLE_RERANKING=false`.

### Configuration

| Variable               | Default                   | Description                                   |
|------------------------|---------------------------|-----------------------------------------------|
| `ENABLE_RERANKING`     | `true`                    | Enable cross-encoder reranking                |
| `RERANKING_PROVIDER`   | `flashrank`               | Reranking provider name                       |
| `RERANKING_MODEL`      | `ms-marco-MiniLM-L-12-v2` | FlashRank model (~34MB)                       |
| `RERANKING_MAX_LENGTH` | `512`                     | Maximum input length in tokens                |
| `RERANKING_CACHE_DIR`  | None                      | Custom cache directory for model files        |

### How It Works

1. **Candidate retrieval**: Search retrieves the full ranked depth of candidates
2. **Cross-Encoder Scoring**: FlashRank scores each candidate against the query
3. **Re-ordering**: Results sorted by cross-encoder score (higher = more relevant)
4. **Paging**: The requested page is cut from that ordering

Every ranked search serves its page from ONE ordering of at most 100 candidates (the ranked depth, which equals the largest page a single request can ask for), so the candidate pool is the same for every page of a query and does not scale with the requested `limit` or `offset`. A page whose window reaches past that depth is served short, and the response says so through the `rank_depth_limit` key.

### Available Models

| Model                               | Size | Notes                          |
|-------------------------------------|------|--------------------------------|
| `ms-marco-MiniLM-L-12-v2` (default) | 34MB | Fast, good quality             |
| `ms-marco-MiniLM-L-6-v2`            | 23MB | Faster, slightly lower quality |
| `rank_zephyr_7b_v1_full`            | 14GB | Highest quality, requires GPU  |

### Performance Considerations

- **Startup**: Model downloads during server initialization (~34MB for default model)
- **Latency**: Adds 20-100ms per search depending on candidate count
- **Memory**: ~200MB RAM for default model
- **Cache**: Model cached in `~/.cache/flashrank/` by default

## Context Length and Truncation Control

Embedding models have maximum context windows (measured in tokens). Text exceeding this limit must either be truncated or rejected with an error.

### Truncation Behavior by Provider

| Provider        | Truncation Control | Default Behavior                        | Configuration                         |
|-----------------|--------------------|-----------------------------------------|---------------------------------------|
| **Ollama**      | Configurable       | Error on context exceed                 | `EMBEDDING_OLLAMA_TRUNCATE=false`     |
| **Voyage**      | Configurable       | Error on context exceed                 | `VOYAGE_TRUNCATION=false`             |
| **OpenAI**      | Always error       | Returns error if input exceeds limit    | N/A                                   |
| **Azure**       | Always error       | Returns error if input exceeds limit    | N/A                                   |
| **HuggingFace** | Always truncate    | Silently truncates (cannot be disabled) | N/A                                   |

### Recommended Configuration

**For production use**, keep truncation **disabled** (`false`) and enable **chunking** (`ENABLE_CHUNKING=true`, default):

```bash
# Recommended: Chunking handles long documents, errors on unexpected overflows
ENABLE_CHUNKING=true        # Default
EMBEDDING_OLLAMA_TRUNCATE=false       # Default - errors prevent silent quality degradation
```

**Silent truncation** (`EMBEDDING_OLLAMA_TRUNCATE=true`) is only recommended when:
- You accept potential embedding quality degradation for very long documents
- Chunking is disabled and you want graceful handling of edge cases

### How Pre-Validation Works

When truncation is disabled (`EMBEDDING_OLLAMA_TRUNCATE=false` or `VOYAGE_TRUNCATION=false`):

1. **Before calling the embedding API**, text length is estimated using heuristic: 1 token ~ 3 characters
2. If estimated tokens exceed the model's context limit, an error is raised **before** the API call
3. This provides fail-fast behavior with clear error messages

Example error:
```text
ValueError: Text length (5000 chars, ~1666 estimated tokens) may exceed context window
(1000 tokens from EMBEDDING_OLLAMA_NUM_CTX) for model qwen3-embedding:0.6b.
Options: 1) Enable chunking (ENABLE_CHUNKING=true, default),
         2) Increase EMBEDDING_OLLAMA_NUM_CTX,
         3) Set EMBEDDING_OLLAMA_TRUNCATE=true to allow silent truncation.
```

### Context Limits by Model

Common embedding models and their context limits:

| Model                  | Provider    | Max Tokens | Notes                                 |
|------------------------|-------------|------------|---------------------------------------|
| qwen3-embedding:0.6b   | Ollama      | 32,000     | Default model                         |
| nomic-embed-text       | Ollama      | 8,192      | Limited by EMBEDDING_OLLAMA_NUM_CTX   |
| text-embedding-3-small | OpenAI      | 8,191      | Always error on exceed                |
| text-embedding-3-large | OpenAI      | 8,191      | Always error on exceed                |
| voyage-3               | Voyage AI   | 32,000     | Configurable via truncation           |
| all-MiniLM-L6-v2       | HuggingFace | 256        | Always silently truncates             |

**Note**: For Ollama models, the effective context limit is `min(model_max_tokens, EMBEDDING_OLLAMA_NUM_CTX)`.

### Startup Validation

On server startup, the universal validator checks:

1. **When chunking enabled**: Warns if `CHUNK_SIZE` may exceed model's context limit
2. **When chunking disabled**: Warns about potential issues with large documents
3. **Unknown models**: Uses provider default and logs a warning

Example startup warning:
```text
[EMBEDDING CONFIG] CHUNK_SIZE (100000 chars, ~33333 tokens estimate) exceeds
model context limit (32000 tokens from model spec for qwen3-embedding:0.6b).
Chunks will cause embedding errors.
Recommendation: Reduce CHUNK_SIZE to ~76800 chars (80% of context window).
```

## Common Configuration Settings

### Timeout and Retry Settings

All providers support these settings:

| Variable                       | Default | Description                                                                                               |
|--------------------------------|---------|-----------------------------------------------------------------------------------------------------------|
| `EMBEDDING_TIMEOUT_S`          | `240.0` | Timeout in seconds for one attempt of one request (one batch of a document's chunks, or one search query) |
| `EMBEDDING_RETRY_MAX_ATTEMPTS` | `5`     | Max retry attempts                                                                                        |
| `EMBEDDING_RETRY_BASE_DELAY_S` | `1.0`   | Base delay between retries                                                                                |
| `EMBEDDING_BATCH_SIZE`         | `auto`  | Chunks per batch: `auto` (sized to `EMBEDDING_BATCH_TARGET_S`) or a fixed 1-512                           |
| `EMBEDDING_BATCH_TARGET_S`     | `5.0`   | Wall time in seconds one batch should take in `auto` mode                                                 |

### Full Configuration Example

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-openai,reranking]", "mcp-context-server"],
      "env": {
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIM": "1536",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "EMBEDDING_TIMEOUT_S": "240",
        "EMBEDDING_RETRY_MAX_ATTEMPTS": "5",
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "${LANGSMITH_API_KEY}"
      }
    }
  }
}
```

## PostgreSQL Backend Setup

When using PostgreSQL backend (including Supabase), ensure the pgvector extension is enabled.

### Docker PostgreSQL (pgvector/pgvector image)

The `pgvector/pgvector` Docker image has pgvector pre-installed:

```bash
docker run --name pgvector18 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mcp_context \
  -p 5432:5432 \
  -d pgvector/pgvector:pg18-trixie

# Configure and run (semantic search auto-registers once the provider is ready)
export STORAGE_BACKEND=postgresql
uv run mcp-context-server
```

### Supabase Setup

Supabase provides pgvector but requires manual enablement:

**Method A: Via Dashboard**
1. Navigate to **Database > Extensions** in Supabase Dashboard
2. Search for "vector" and enable it

**Method B: Via SQL**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Other PostgreSQL Deployments

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT * FROM pg_available_extensions WHERE name = 'vector';
```

## Changing Embedding Dimensions

**IMPORTANT**: Changing embedding dimensions requires database migration and loss of existing embeddings.

### Common Dimensions by Model

| Model                  | Dimension |
|------------------------|-----------|
| qwen3-embedding:0.6b   | 1024      |
| text-embedding-3-small | 1536      |
| text-embedding-3-large | 3072      |
| all-MiniLM-L6-v2       | 384       |
| voyage-3               | 1024      |

### Migration Procedure

When you change `EMBEDDING_DIM` and restart, you'll see:

```text
RuntimeError: Embedding dimension mismatch detected!
  Existing database dimension: 1024
  Configured EMBEDDING_DIM: 1536
```

**Steps**:

1. **Back up database**:
   ```bash
   # Windows
   copy %USERPROFILE%\.mcp\context_storage.db %USERPROFILE%\.mcp\context_storage.backup.db

   # Linux/macOS
   cp ~/.mcp/context_storage.db ~/.mcp/context_storage.backup.db
   ```

2. **Update configuration** with new dimension and model

3. **Delete database**:
   ```bash
   # Windows
   del %USERPROFILE%\.mcp\context_storage.db

   # Linux/macOS
   rm ~/.mcp/context_storage.db
   ```

4. **Restart server** - new tables created automatically

5. **Re-store data** - embeddings are generated at `store_context` and `store_context_batch` time (and regenerated when text changes via `update_context` or `update_context_batch`)

## Usage

### semantic_search_context Tool

When registered (by default, whenever an embedding provider is available), the `semantic_search_context` MCP tool becomes available.

**Parameters**:
- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Filter by thread
- `source` (str, optional): Filter by source ('user' or 'agent')
- `tags` (list, optional): Filter by tags (OR logic; at most 100 tags per request)
- `content_type` (str, optional): Filter by content type
- `start_date` (str, optional): Filter by creation date (ISO 8601)
- `end_date` (str, optional): Filter by creation date (ISO 8601)
- `metadata` (dict, optional): Simple metadata filters
- `metadata_filters` (list, optional): Advanced metadata filters (`in`/`not_in` value lists accept at most 100 members)
- `include_images` (bool, optional): Include image data (default: false)
- `explain_query` (bool, optional): Include statistics (default: false)

**Metadata Filtering**: See the [Metadata Guide](metadata-addition-updating-and-filtering.md) for operators and examples.

**Returns**:
```json
{
  "query": "original search query",
  "results": [
    {
      "id": "0190abcdef1234567890abcdef123456",
      "thread_id": "thread-abc",
      "text_content": "matching context...",
      "summary": "Summary of the matching context entry with key semantic information.",
      "is_text_content_truncated": true,
      "visibility": "private",
      "is_owner": true,
      "scores": {
        "semantic_distance": 0.234,
        "semantic_rank": null,
        "rerank_score": 0.92
      },
      "tags": ["tag1", "tag2"]
    }
  ],
  "count": 5,
  "model": "text-embedding-3-small",
  "stats": {
    "execution_time_ms": 85.3,
    "embedding_generation_ms": 45.1,
    "filters_applied": 2,
    "rows_returned": 5,
    "backend": "sqlite",
    "query_plan": "..."
  }
}
```

**Note:** The `stats` field is only included when `explain_query=True`. Every key of the block is then present, including `backend` (the active storage backend type) and `query_plan`. A request that fails parameter validation (for example an invalid `metadata_filters` operator) returns the same keys with the counters zeroed and `query_plan` set to `null`, since no query ran, so a client can read `stats["query_plan"]` unconditionally and only has to handle the null.

**Scores Object**:
- `semantic_distance`: LOWER = more similar. The underlying metric depends on embedding storage: Euclidean L2 (>= 0) for uncompressed/`mse` storage, or a negated inner product (~ -1..0 for normalized embeddings, where more negative = more similar) when the default `ip` compression variant is active. Compare values within a single result set rather than against fixed thresholds, since the numeric range differs by storage variant.
- `semantic_rank`: Always null for standalone semantic search (no ranking)
- `rerank_score`: Cross-encoder relevance score (HIGHER = better, 0.0-1.0), present when reranking is enabled

### Automatic Embedding Generation

Embeddings are generated automatically:

- **On `store_context`**: Embeddings generated before the entry is written
- **On `store_context_batch`**: Embeddings generated for each entry
- **On `update_context`**: Embeddings regenerated when text changes
- **On `update_context_batch`**: Embeddings regenerated for entries with text changes
- **On `delete_context`**: Embeddings cascade deleted

### Batched Document Embedding

The server embeds a chunked document in batches, one provider request per batch, instead of sending every chunk in one request. The retry policy and the timeouts apply to each batch:

- `EMBEDDING_TIMEOUT_S` bounds one attempt of one batch, and `EMBEDDING_RETRY_MAX_ATTEMPTS` retries re-send only the batch that failed, never the chunks already embedded.
- The total timeout derived from those settings (every attempt plus the backoff between them, with a 10% margin) also applies to each batch. A document's size sets how many batches it takes, not how long any single request may run, so a large document needs no larger timeout.

`EMBEDDING_BATCH_SIZE` chooses the batch size. With the default `auto`, the first batch carries 4 chunks; after each batch the server folds the measured seconds per chunk into an exponentially weighted estimate and sizes the next batch to take about `EMBEDDING_BATCH_TARGET_S` (default 5 seconds), between 1 and 64 chunks. The estimate comes from the provider's observed speed, queueing at the provider included, so it needs no setting tied to the CPU, GPU, or hosted service behind the provider, and it carries over from one document to the next. An integer from 1 to 512 fixes the size instead.

Each batch holds one of the `EMBEDDING_MAX_CONCURRENT` slots and releases it between batches, and waiting callers receive slots in arrival order. When every slot is taken, a store or update that arrives gets the next slot a finishing batch frees, instead of waiting for the rest of a running document.

When a batch fails or runs out of time, nothing is saved (see [Embedding-First Transactional Integrity](#embedding-first-transactional-integrity)) and the error names the batch and the progress made:

```text
Embedding generation failed: <provider message> (batch 3 of 40, chunks 9-15 of 280; 8 chunks embedded in 12.3 s)
Embedding generation exceeded total timeout (1341s) on batch 3 of 40 (chunks 9-15 of 280; 8 chunks embedded in 12.3 s). This may indicate the embedding provider is overloaded or unreachable.
```

In `auto` mode the batch count is projected from the size of the failing batch. The server logs the start of each document's embedding (text length, chunk count, batch mode) and its completion (chunks, batches, elapsed time) at INFO, and every batch at DEBUG.

## Embedding-First Transactional Integrity

**CRITICAL:** When `ENABLE_EMBEDDING_GENERATION=true` (default when semantic search is enabled), embedding generation follows an "embedding-first" pattern:

1. Embeddings are generated FIRST (outside any database transaction)
2. If embedding generation fails, **NO data is saved** (operation returns error immediately)
3. If embedding generation succeeds, ALL database operations occur in a SINGLE atomic transaction

This ensures data consistency: you never have context entries without their corresponding embeddings when embedding generation is enabled.

**Key Implications:**
- Embedding failures prevent data storage (not graceful degradation)
- Network issues with embedding provider will cause store_context to fail
- A batch that still fails after its retries fails the whole store; the batches already embedded are discarded with it
- Consider retry logic in client code for transient embedding failures

### Example Use Cases

1. **Cross-thread discovery**:
   ```python
   semantic_search_context(query="authentication implementation", limit=10)
   ```

2. **Agent collaboration**:
   ```python
   semantic_search_context(query="API rate limiting solutions", source="agent")
   ```

3. **Metadata-filtered search**:
   ```python
   semantic_search_context(
       query="performance optimization",
       metadata={"status": "completed"},
       metadata_filters=[{"key": "priority", "operator": "gte", "value": 7}]
   )
   ```

4. **Time-bounded search**:
   ```python
   semantic_search_context(query="database optimization", start_date="2025-11-01")
   ```

### Performance Characteristics

- **Embedding Generation**: 50-150ms per text (varies by provider)
- **Similarity Search**: O(n * d) where n = filtered entries, d = dimensions
- **Acceptable Scale**: <100K context entries
- **Storage Impact**: ~4 bytes per dimension per entry

## Verification

### Setup Checklist

1. **Verify provider installation**:
   ```bash
   # For Ollama
   ollama list
   curl http://localhost:11434

   # For API providers, check credentials are set
   echo $OPENAI_API_KEY
   ```

2. **Verify dependencies**:
   ```bash
   # Check LangChain packages installed
   python -c "from langchain_openai import OpenAIEmbeddings; print('OK')"
   ```

3. **Start server**:
   ```bash
   # ENABLE_SEMANTIC_SEARCH defaults to auto; the tool registers automatically
   # once the provider is available. Set it explicitly only to force on/off.
   export EMBEDDING_PROVIDER=openai
   export OPENAI_API_KEY=sk-...
   uv run mcp-context-server
   ```

4. **Check logs** for:
   ```text
   Embedding generation enabled with provider: openai (model: text-embedding-3-small)
   semantic_search_context registered
   ```

5. **Test functionality**:
   ```python
   semantic_search_context(query="test", limit=5)
   ```

## Troubleshooting

### Startup Error Classification

The server classifies startup errors into two categories with different exit codes:

**Configuration Errors (Exit Code 78)**

These errors require human intervention to fix. The server will NOT recover by restarting:

| Error Type                  | Example                                          | Fix                                                      |
|-----------------------------|--------------------------------------------------|----------------------------------------------------------|
| Missing API key             | `OPENAI_API_KEY environment variable is not set` | Set the required environment variable                    |
| Package not installed       | `langchain-openai package not installed`         | Run `uv sync --extra embeddings-openai`                  |
| Invalid provider            | `Unknown provider: 'invalid'`                    | Use valid EMBEDDING_PROVIDER value                       |
| Vector storage dependencies | `vector storage dependencies not available`      | Install required packages                                |
| Model not found             | `model 'model' not found (status code: 404)`     | Pull model: `ollama pull <model>` or fix EMBEDDING_MODEL |

**Dependency Errors (Exit Code 69)**

These errors may resolve with time. Supervisor may retry with backoff:

| Error Type               | Example                                             | Fix                            |
|--------------------------|-----------------------------------------------------|--------------------------------|
| Service not running      | `Ollama service not accessible: Connection refused` | Start Ollama: `ollama serve`   |
| Network issues           | `Connection timed out`                              | Check network connectivity     |
| Temporary unavailability | `Service returned status 503`                       | Wait and retry                 |

### Log Messages

The server logs startup errors with clear indicators:

```text
# Configuration Error (will NOT retry)
[FATAL] Configuration error (will not retry): ENABLE_EMBEDDING_GENERATION=true but langchain-openai package not installed...

# Dependency Error (may retry)
[ERROR] Dependency unavailable (may retry): ENABLE_EMBEDDING_GENERATION=true but ollama is not available (service may be down)...
```

### Provider Connection Errors

**Error**: `Failed to connect to Ollama: Connection refused`

**Solutions**:
- Start Ollama: `ollama serve`
- Check firewall allows port 11434
- For Docker: use `OLLAMA_HOST=http://host.docker.internal:11434`

**Error**: `OpenAI API key not found`

**Solutions**:
- Set `OPENAI_API_KEY` environment variable
- Verify key is valid at [platform.openai.com](https://platform.openai.com)

### Dimension Mismatch

**Error**: `Embedding dimension mismatch: expected 1536, got 1024`

**Cause**: Model produces different dimensions than configured

**Solution**: Update `EMBEDDING_DIM` to match model output, or use correct model

### Model Not Found

**Error**: `Model 'qwen3-embedding:0.6b' not found`

**Solution** (Ollama):
```bash
ollama pull qwen3-embedding:0.6b
```

### Dependencies Not Installed

**Error**: `ImportError: langchain_openai not installed`

**Solution**:
```bash
uv sync --extra embeddings-openai
```

### macOS SQLite Extension Error

**Error**: `AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'`

**Cause**: Default macOS Python lacks SQLite extension support

**Solution**:
```bash
brew install python
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
uv sync --extra embeddings-ollama
```

### Common Error Messages

| Error                 | Cause                | Solution                              |
|-----------------------|----------------------|---------------------------------------|
| `Connection refused`  | Provider not running | Start Ollama or check API credentials |
| `API key invalid`     | Wrong credentials    | Verify API key is correct             |
| `Model not found`     | Model not available  | Pull model or check model name        |
| `Dimension mismatch`  | Wrong EMBEDDING_DIM  | Match dimension to model output       |
| `Rate limit exceeded` | API rate limiting    | Reduce request rate or upgrade plan   |

## Performance Optimization

### Provider Selection by Use Case

| Use Case             | Recommended Provider      |
|----------------------|---------------------------|
| Development          | Ollama (free, local)      |
| Production (general) | OpenAI (balanced)         |
| Enterprise           | Azure OpenAI (compliance) |
| Cost-sensitive       | HuggingFace (free API)    |
| RAG optimization     | Voyage AI (specialized)   |

### Batch Size (Voyage AI)

Adjust batch size for throughput vs latency:
```bash
VOYAGE_BATCH_SIZE=20  # Higher for throughput, lower for latency
```

### Timeouts

`EMBEDDING_TIMEOUT_S` bounds one attempt of one request, a single batch of chunks, so a larger document needs no larger value. Raise it when single requests are slow, for example a provider behind a slow connection:
```bash
EMBEDDING_TIMEOUT_S=600
```

To keep each request short instead, lower `EMBEDDING_BATCH_TARGET_S` or set a small fixed `EMBEDDING_BATCH_SIZE`.

## Additional Resources

### Official Documentation

- **LangChain Embeddings**: [python.langchain.com/docs/integrations/text_embedding](https://python.langchain.com/docs/integrations/text_embedding)
- **Ollama**: [ollama.com](https://ollama.com)
- **OpenAI Embeddings**: [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
- **Azure OpenAI**: [learn.microsoft.com/azure/ai-services/openai](https://learn.microsoft.com/azure/ai-services/openai)
- **Voyage AI**: [docs.voyageai.com](https://docs.voyageai.com)
- **LangSmith**: [smith.langchain.com](https://smith.langchain.com)

### Related Documentation

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation
- **Summary Generation**: [Summary Generation Guide](summary-generation.md) - LLM-based automatic summarization for search results
- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - linguistic search with stemming
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - filtering with operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication
- **Main Documentation**: [README.md](../README.md) - overview and quick start
