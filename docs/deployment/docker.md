# Docker Deployment Guide

## Introduction

This guide covers deploying the MCP Context Server using Docker and Docker Compose. Docker deployment enables HTTP transport mode for remote client connections, making the server accessible to MCP clients across networks.

**Key Features:**
- HTTP transport mode for remote access (vs. stdio for local)
- Fifteen pre-configured deployment options (9 standard + 6 local-build alternatives)
- Support for Ollama (local), OpenAI (cloud), and mixed-provider (Ollama embeddings + OpenAI summary) configurations
- Health checks and container orchestration support
- Shared Ollama model volume across Ollama configurations

## Prerequisites

- **Docker Engine**: 20.10+ or Docker Desktop
- **Docker Compose**: V2 (included with Docker Desktop)
- **Storage**: ~2GB for images and models (~600MB for qwen3-embedding:0.6b with Ollama)
- **Network**: Port 8000 available for HTTP transport
- **OpenAI API Key**: Required for OpenAI embedding configurations

## Deployment Options

Fifteen Docker Compose configurations are provided, organized by provider combination:

### Ollama Embeddings (Local, Self-Hosted)

| Configuration                | File                                            | Database               | Use Case                          |
|------------------------------|-------------------------------------------------|------------------------|-----------------------------------|
| SQLite + Ollama              | `docker-compose.sqlite.ollama.yml`              | Local SQLite           | Single-user, testing, development |
| PostgreSQL + Ollama          | `docker-compose.postgresql.ollama.yml`          | Internal PostgreSQL    | Multi-user, production            |
| External PostgreSQL + Ollama | `docker-compose.postgresql-external.ollama.yml` | Supabase, corporate DB | Existing database infrastructure  |

### OpenAI Embeddings (Cloud API)

| Configuration                | File                                            | Database               | Use Case                         |
|------------------------------|-------------------------------------------------|------------------------|----------------------------------|
| SQLite + OpenAI              | `docker-compose.sqlite.openai.yml`              | Local SQLite           | Single-user, cloud embeddings    |
| PostgreSQL + OpenAI          | `docker-compose.postgresql.openai.yml`          | Internal PostgreSQL    | Multi-user, cloud embeddings     |
| External PostgreSQL + OpenAI | `docker-compose.postgresql-external.openai.yml` | Supabase, corporate DB | Enterprise with cloud embeddings |

### Mixed Providers (Ollama Embeddings + OpenAI Summary)

| Configuration                       | File                                                   | Database               | Use Case                                    |
|-------------------------------------|--------------------------------------------------------|------------------------|---------------------------------------------|
| SQLite + Ollama/OpenAI              | `docker-compose.sqlite.ollama-openai.yml`              | Local SQLite           | Local embeddings, cloud summaries           |
| PostgreSQL + Ollama/OpenAI          | `docker-compose.postgresql.ollama-openai.yml`          | Internal PostgreSQL    | Production with local embeddings            |
| External PostgreSQL + Ollama/OpenAI | `docker-compose.postgresql-external.ollama-openai.yml` | Supabase, corporate DB | Enterprise with local embeddings            |

**Choosing Between Ollama and OpenAI:**

| Factor      | Ollama                          | OpenAI                            |
|-------------|---------------------------------|-----------------------------------|
| Cost        | Free (self-hosted)              | Pay-per-use API                   |
| Privacy     | Data stays local                | Data sent to OpenAI               |
| Setup       | Automatic model download        | Requires API key                  |
| Performance | Depends on hardware             | Consistent cloud performance      |
| Model       | qwen3-embedding:0.6b (1024 dim) | text-embedding-3-small (1536 dim) |

### Local Build Alternatives (Ollama)

For users who want to build the MCP Context Server image from local source code instead of pulling from GHCR, `.local` variants are available for the Ollama-only configurations:

| Configuration                        | File                                                  | Database               | Use Case                                     |
|--------------------------------------|-------------------------------------------------------|------------------------|----------------------------------------------|
| SQLite + Ollama (local build)        | `docker-compose.sqlite.ollama.local.yml`              | Local SQLite           | Development, testing with local code changes |
| PostgreSQL + Ollama (local build)    | `docker-compose.postgresql.ollama.local.yml`          | Internal PostgreSQL    | Testing PostgreSQL with local code           |
| External PostgreSQL + Ollama (local) | `docker-compose.postgresql-external.ollama.local.yml` | Supabase, corporate DB | Testing external DB with local code          |

### Local Build Alternatives (Ollama + OpenAI Summary)

`.local` variants are also available for the mixed-provider (Ollama embeddings + OpenAI summary) configurations:

| Configuration                                     | File                                                         | Database               | Use Case                                     |
|---------------------------------------------------|--------------------------------------------------------------|------------------------|----------------------------------------------|
| SQLite + Ollama/OpenAI (local build)              | `docker-compose.sqlite.ollama-openai.local.yml`              | Local SQLite           | Development, testing with local code changes |
| PostgreSQL + Ollama/OpenAI (local build)          | `docker-compose.postgresql.ollama-openai.local.yml`          | Internal PostgreSQL    | Testing PostgreSQL with local code           |
| External PostgreSQL + Ollama/OpenAI (local build) | `docker-compose.postgresql-external.ollama-openai.local.yml` | Supabase, corporate DB | Testing external DB with local code          |

These `.local` files build the same Docker image as the published GHCR image but from local source code. Use them when you want to test local changes before pushing. The OpenAI-only compose files always build from source (no GHCR image exists for those configurations).

## Quick Start

### Ollama Deployments

#### SQLite + Ollama (Simplest)

```bash
# Build and start
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml up -d

# Wait for embedding model download (first run only, ~2-3 minutes)
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml logs -f ollama

# Verify server is ready
curl http://localhost:8000/health
```

#### PostgreSQL + Ollama (Recommended for Production)

```bash
# Build and start (includes PostgreSQL with pgvector)
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml up -d

# Wait for all services to be healthy
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml ps

# Verify server is ready
curl http://localhost:8000/health
```

#### External PostgreSQL + Ollama (Supabase, Corporate)

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql-external.ollama.example deploy/docker/.env

# 2. Edit .env with your PostgreSQL connection details
# See "External PostgreSQL Configuration" section below

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql-external.ollama.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

### OpenAI Deployments

All OpenAI configurations **require** an `.env` file with your OpenAI API key.

#### SQLite + OpenAI

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-sqlite.openai.example deploy/docker/.env

# 2. Edit .env and add your OPENAI_API_KEY
# OPENAI_API_KEY=sk-your-openai-api-key-here

# 3. Build and start
docker compose -f deploy/docker/docker-compose.sqlite.openai.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

#### PostgreSQL + OpenAI

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql.openai.example deploy/docker/.env

# 2. Edit .env and add your OPENAI_API_KEY
# OPENAI_API_KEY=sk-your-openai-api-key-here

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql.openai.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

#### External PostgreSQL + OpenAI (Supabase, Corporate)

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql-external.openai.example deploy/docker/.env

# 2. Edit .env with your OpenAI API key AND PostgreSQL connection details
# See "External PostgreSQL Configuration" section below

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql-external.openai.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

### Mixed Provider Deployments (Ollama Embeddings + OpenAI Summary)

Mixed provider configurations use Ollama for local embedding generation and OpenAI for cloud-based summary generation. These configurations pull the pre-built `ghcr.io/getactionspace/mcp-context-server:latest-ollama-openai` image from GHCR. To build from local source code instead, use the `.local` variants (e.g., `docker-compose.sqlite.ollama-openai.local.yml`). All configurations **require** an `.env` file with your OpenAI API key.

#### SQLite + Ollama Embeddings + OpenAI Summary

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-sqlite.ollama-openai.example deploy/docker/.env

# 2. Edit .env and add your OPENAI_API_KEY
# OPENAI_API_KEY=sk-your-openai-api-key-here

# 3. Build and start
docker compose -f deploy/docker/docker-compose.sqlite.ollama-openai.yml up -d

# Wait for embedding model download (first run only, ~2-3 minutes)
docker compose -f deploy/docker/docker-compose.sqlite.ollama-openai.yml logs -f ollama

# Verify server is ready
curl http://localhost:8000/health
```

#### PostgreSQL + Ollama Embeddings + OpenAI Summary

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql.ollama-openai.example deploy/docker/.env

# 2. Edit .env and add your OPENAI_API_KEY
# OPENAI_API_KEY=sk-your-openai-api-key-here

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql.ollama-openai.yml up -d

# Wait for all services to be healthy
docker compose -f deploy/docker/docker-compose.postgresql.ollama-openai.yml ps

# Verify server is ready
curl http://localhost:8000/health
```

#### External PostgreSQL + Ollama Embeddings + OpenAI Summary

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql-external.ollama-openai.example deploy/docker/.env

# 2. Edit .env with your OpenAI API key AND PostgreSQL connection details
# See "External PostgreSQL Configuration" section below

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql-external.ollama-openai.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

## Building from Source

### When to Build from Source

Two GHCR-published images are available: `ghcr.io/getactionspace/mcp-context-server:latest` (Ollama embeddings + Ollama summary) and `ghcr.io/getactionspace/mcp-context-server:latest-ollama-openai` (Ollama embeddings + OpenAI summary). Until October 2026 the images were published at `ghcr.io/alex-feel/mcp-context-server`; that path keeps the tags published before the move and receives no new ones, so a deployment that pulls it stays on the last release published there.

Build from source when:

- **Testing local code changes** -- use the `.local` variants of compose files
- **Using OpenAI embeddings** -- the OpenAI compose files build with `EMBEDDING_EXTRA=embeddings-openai`

### Image Source Categories

| Category                         | Files                       | `image:`                                                         | `pull_policy` | `build:` block                                  |
|----------------------------------|-----------------------------|------------------------------------------------------------------|---------------|-------------------------------------------------|
| GHCR pull                        | `*.ollama.yml`              | `ghcr.io/getactionspace/mcp-context-server:latest`               | `always`      | None                                            |
| GHCR pull (variant)              | `*.ollama-openai.yml`       | `ghcr.io/getactionspace/mcp-context-server:latest-ollama-openai` | `always`      | None                                            |
| Local build (GHCR equivalent)    | `*.ollama.local.yml`        | `mcp-context-server`                                             | `build`       | Yes (no custom args)                            |
| Local build (variant equivalent) | `*.ollama-openai.local.yml` | `mcp-context-server`                                             | `build`       | Yes (with SUMMARY_EXTRA=summary-openai)         |
| Local build (provider-specific)  | `*.openai.yml`              | `mcp-context-server`                                             | `build`       | Yes (with EMBEDDING_EXTRA / SUMMARY_EXTRA args) |

## Client Connection

Once deployed, connect MCP clients via HTTP transport:

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Claude Code CLI

```bash
claude mcp add --transport http context-server http://localhost:8000/mcp
```

### Remote Access

Replace `localhost` with your server's IP or hostname for remote connections:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "http",
      "url": "http://your-server.example.com:8000/mcp"
    }
  }
}
```

### Running the Ingest CLI from the Image

The image installs `mcp-context-ingest`, the command that keeps a directory of documents in sync with a `kb:<corpus>` thread, beside the server. Run it in a one-off container of the stack's server service: on the stack's network it reaches the server by the service name `mcp-context-server` and the port the server listens on, 8000 in every file under [Docker Compose Files](#docker-compose-files) (the test stacks `docker-compose.sqlite.ollama.test.yml` and `docker-compose.postgresql.ollama.test.yml` listen on 8001 and 8002).

```bash
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml run --rm --no-deps \
  -v /path/to/handbook:/corpus:ro \
  --entrypoint mcp-context-ingest mcp-context-server \
  --server http://mcp-context-server:8000/mcp --corpus handbook --source-dir /corpus
```

`--no-deps` starts no other service, `--entrypoint` replaces the server's entrypoint in that container only, and `--rm` removes the container when the run ends; the running server is untouched. Use the `-f` file of your stack. For a server that requires a bearer token, mount the directory that holds a token file; the command reads the file before every request, so a refresher can rotate the token during a long run:

```bash
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml run --rm --no-deps \
  -v /path/to/handbook:/corpus:ro -v /path/to/secrets:/run/secrets/ingest:ro \
  --entrypoint mcp-context-ingest mcp-context-server \
  --server http://mcp-context-server:8000/mcp --corpus handbook --source-dir /corpus \
  --token-file /run/secrets/ingest/token
```

Mount the directory rather than the file. A bind mount of a single file keeps the file that existed when the container started, so a refresher that writes a new file and renames it over the old one, the usual way to replace a file atomically, never reaches the running command; a rename inside a mounted directory is visible in the container. Have the refresher replace `token` inside that directory. The container runs as UID 10001 (`appuser`), which must be able to read the file: make UID 10001 its owner, or GID 10001 its group with read permission for the group, or make it readable to every user; or run the container as the file's owner with `--user "$(id -u):$(id -g)"` before `--entrypoint`, which works because the command writes no file unless `--report` is given. A token file the command cannot read ends the run with exit 2 at the first request.

Alternatively pass a token exported in the shell with `-e INGEST_TOKEN` before `--entrypoint` and `--token-env INGEST_TOKEN` at the end. The same user reads the corpus: a mounted file it cannot read is kept as stored, and a directory it cannot list makes the run withhold its deletes, so neither ever deletes an entry; make the tree readable to that user. From the host, point `--server` at the published port (`http://localhost:8000/mcp`) and run the command with `uvx --from mcp-context-server mcp-context-ingest` instead. Flags, exit codes and scheduling are in [The mcp-context-ingest CLI](../knowledge-base.md#the-mcp-context-ingest-cli).

## Architecture

### Services Overview

**Ollama configurations** define two or three services:

**mcp-context-server:**
- Production MCP server image
- HTTP transport on port 8000
- Health check endpoint at `/health`
- Non-root user (UID 10001) for security

**ollama:**
- Stock `ollama/ollama:latest` image with automatic model pulling via inline entrypoint
- Downloads `qwen3-embedding:0.6b` on first start
- Health check waits for model availability

**postgres (PostgreSQL configurations only):**
- pgvector/pgvector:pg18-trixie image
- Pre-installed pgvector extension for semantic search
- Persistent volume for data

**OpenAI configurations** are simpler - no Ollama service required:

**mcp-context-server:**
- Same as above, but uses OpenAI API for embeddings
- Requires `OPENAI_API_KEY` in `.env` file

**postgres (PostgreSQL configurations only):**
- Same as Ollama configurations

### Volume Management

| Volume                               | Purpose                           | Used By                   |
|--------------------------------------|-----------------------------------|---------------------------|
| `mcp-context-sqlite-ollama-data`     | SQLite database                   | SQLite + Ollama           |
| `mcp-context-sqlite-openai-data`     | SQLite database                   | SQLite + OpenAI           |
| `mcp-context-postgresql-ollama-data` | PostgreSQL data                   | PostgreSQL + Ollama       |
| `mcp-context-postgresql-openai-data` | PostgreSQL data                   | PostgreSQL + OpenAI       |
| `ollama-models`                      | Embedding models (~600MB)         | All Ollama configurations |
| `flashrank-model-cache`              | FlashRank reranking model (~34MB) | All configurations        |

The `ollama-models` volume is shared across all Ollama configurations, so switching between SQLite and PostgreSQL does not re-download the embedding model. The `flashrank-model-cache` volume persists the FlashRank reranking model across container restarts.

### Host Access to the Ollama Sidecar

Every Ollama-bearing Docker Compose file publishes the sidecar's TCP port 11434 to the Docker host at the loopback address `127.0.0.1:11434`. Native Ollama clients, scripts, debugging tools, or observability tooling running on the same host can reach the sidecar directly at `http://127.0.0.1:11434` (for example, `curl http://127.0.0.1:11434/api/tags` lists the locally cached models).

The binding is intentionally loopback-only (`127.0.0.1`), not all-interfaces (`0.0.0.0`). The sidecar is therefore NOT reachable from other machines on the host's network. Users who deliberately want to expose Ollama to a LAN can widen the host-ip portion of the mapping (for example, `"0.0.0.0:11434:11434"` to publish on every interface, or `"192.0.2.10:11434:11434"` to publish on one specific interface); the project does not recommend doing so without a deliberate security review.

If port 11434 is already in use on the host (for example, a native Ollama install is already running), `docker compose up` fails with a port-binding error. Either stop the conflicting process or remap the host-side port in the `ports:` block (for example, `"127.0.0.1:11500:11434"`) and update any host-side clients to use the new host port.

### Automatic Model Download (Ollama Only)

Each Ollama-based Docker Compose file declares an inline `entrypoint:` block on the `ollama` service that:

1. Starts the Ollama server in the background.
2. Waits for the server to accept requests.
3. Checks the configured embedding model (`EMBEDDING_MODEL`) and pulls it if not already cached.
4. For Ollama-only configurations (where `SUMMARY_PROVIDER` is `ollama`), checks the summary model (`SUMMARY_MODEL`) and pulls it if missing and different from the embedding model. For mixed-provider configurations (`ollama-openai`), the summary model is not pulled because OpenAI handles summary generation.
5. Hands off to the running Ollama server for normal request handling.

The check-then-pull logic is idempotent: subsequent restarts skip the download when models are already cached in the `ollama-models` Docker volume. Models persist across container restarts and across all Ollama configurations sharing this volume.

## Configuration

### Environment Variables

All Docker Compose files use environment variables for configuration. Key settings:

**Transport Settings:**

| Variable                  | Default   | Description                                              |
|---------------------------|-----------|----------------------------------------------------------|
| `MCP_TRANSPORT`           | `http`    | Transport mode (set to `http` for Docker)                |
| `FASTMCP_HOST`            | `0.0.0.0` | HTTP bind address                                        |
| `FASTMCP_PORT`            | `8000`    | HTTP port                                                |
| `FASTMCP_STATELESS_HTTP`  | `true`    | Stateless HTTP mode (enabled by default)                 |

**Summary Generation Settings:**

| Variable                     | Default      | Description                                                     |
|------------------------------|--------------|-----------------------------------------------------------------|
| `ENABLE_SUMMARY_GENERATION`  | `true`       | Enable automatic LLM-based summary generation                   |
| `SUMMARY_PROVIDER`           | `ollama`     | Summary provider: `ollama`, `openai`, or `anthropic`            |
| `SUMMARY_MODEL`              | `qwen3:0.6b` | Summary model name (provider-specific)                          |
| `SUMMARY_MAX_TOKENS`         | `4000`       | Maximum output tokens for summary generation (50-16384)         |
| `SUMMARY_TIMEOUT_S`          | `240.0`      | Timeout in seconds for summary generation API calls             |
| `SUMMARY_RETRY_MAX_ATTEMPTS` | `5`          | Maximum retry attempts on transient errors                      |
| `SUMMARY_RETRY_BASE_DELAY_S` | `3.0`        | Base delay in seconds between retries (exponential backoff)     |
| `SUMMARY_MAX_CONCURRENT`     | `2`          | Maximum concurrent summary generation operations (1-20)         |
| `SUMMARY_PROMPT`             | (built-in)   | Custom system prompt; unset uses the optimized built-in default |

**Search Features:**

| Variable                 | Default | Description                                                                                                                                                                                                      |
|--------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_SEMANTIC_SEARCH` | `auto`  | Register vector similarity search tool — auto: register when an embedding provider is available; true: same as auto, but a missing provider logs a warning (the tool is still skipped); false: force off         |
| `ENABLE_FTS`             | `auto`  | Register full-text search tool — auto: register (built-in, no extra deps); true: force on; false: force off                                                                                                      |
| `ENABLE_HYBRID_SEARCH`   | `auto`  | Register combined FTS + semantic search tool — auto: register when >=1 of FTS/semantic is available; true: same as auto, but a missing prerequisite logs a warning (the tool is still skipped); false: force off |
| `ENABLE_CHUNKING`        | `true`  | Enable text chunking for embeddings                                                                                                                                                                              |
| `ENABLE_RERANKING`       | `true`  | Enable cross-encoder reranking                                                                                                                                                                                   |

**Embedding Settings (Ollama):**

| Variable             | Default                 | Description                 |
|----------------------|-------------------------|-----------------------------|
| `EMBEDDING_PROVIDER` | `ollama`                | Embedding provider          |
| `EMBEDDING_MODEL`    | `qwen3-embedding:0.6b`  | Ollama embedding model      |
| `EMBEDDING_DIM`      | `1024`                  | Embedding vector dimensions |
| `OLLAMA_HOST`        | `http://ollama:11434`   | Ollama API endpoint         |

**Embedding Settings (OpenAI):**

| Variable             | Default                  | Description                     |
|----------------------|--------------------------|---------------------------------|
| `EMBEDDING_PROVIDER` | `openai`                 | Embedding provider              |
| `EMBEDDING_MODEL`    | `text-embedding-3-small` | OpenAI embedding model          |
| `EMBEDDING_DIM`      | `1536`                   | Embedding vector dimensions     |
| `OPENAI_API_KEY`     | (required)               | OpenAI API key (from .env file) |

**Storage Settings (SQLite):**

| Variable          | Default                    | Description                    |
|-------------------|----------------------------|--------------------------------|
| `STORAGE_BACKEND` | `sqlite`                   | Database backend               |
| `DB_PATH`         | `/data/context_storage.db` | Database path inside container |

**Storage Settings (PostgreSQL):**

| Variable              | Default       | Description              |
|-----------------------|---------------|--------------------------|
| `STORAGE_BACKEND`     | `postgresql`  | Database backend         |
| `POSTGRESQL_HOST`     | `postgres`    | PostgreSQL hostname      |
| `POSTGRESQL_PORT`     | `5432`        | PostgreSQL port          |
| `POSTGRESQL_USER`     | `postgres`    | PostgreSQL username      |
| `POSTGRESQL_PASSWORD` | `postgres`    | PostgreSQL password      |
| `POSTGRESQL_DATABASE` | `mcp_context` | PostgreSQL database name |

The bundled PostgreSQL files serve requests as the sidecar's bootstrap role with row-level security off. `ENABLE_POSTGRESQL_RLS`, `POSTGRESQL_MIGRATION_USER` and `POSTGRESQL_MIGRATION_PASSWORD` are unset there; see [Enabling row-level security](#enabling-row-level-security) to turn it on.

**Metadata Indexing Settings:**

| Variable                   | Default                 | Description                                              |
|----------------------------|-------------------------|----------------------------------------------------------|
| `METADATA_INDEXED_FIELDS`  | `status,agent_name,...` | Comma-separated fields to index with optional type hints |
| `METADATA_INDEX_SYNC_MODE` | `additive`              | Index sync mode: `strict`, `auto`, `warn`, `additive`    |

See the [Metadata Guide](../metadata-addition-updating-and-filtering.md#environment-variables) for full details on configurable metadata indexing.

**Chunking and Reranking:**

Text chunking and cross-encoder reranking are enabled by default to improve search quality. These features work automatically without configuration.

| Variable           | Default                   | Description                                     |
|--------------------|---------------------------|-------------------------------------------------|
| `ENABLE_CHUNKING`  | `true`                    | Text chunking for long document embedding       |
| `CHUNK_SIZE`       | `1500`                    | Target chunk size in characters                 |
| `ENABLE_RERANKING` | `true`                    | Cross-encoder result reranking                  |
| `RERANKING_MODEL`  | `ms-marco-MiniLM-L-12-v2` | FlashRank model (~34MB, downloads on startup)   |

To disable these features, set the environment variables to `false` in your Docker Compose file:

```yaml
environment:
  - ENABLE_CHUNKING=false
  - ENABLE_RERANKING=false
```

For more configuration options, see [Text Chunking](../semantic-search.md#text-chunking) and [Cross-Encoder Reranking](../semantic-search.md#cross-encoder-reranking).

### External PostgreSQL Configuration

For external PostgreSQL (Supabase, corporate databases), copy the appropriate `.env.example` file to `.env` and configure:

**For Ollama:**
```bash
cp deploy/docker/.env-postgresql-external.ollama.example deploy/docker/.env
```

**For OpenAI:**
```bash
cp deploy/docker/.env-postgresql-external.openai.example deploy/docker/.env
```

Then configure PostgreSQL connection:

```bash
# Option A: Individual variables (recommended)
POSTGRESQL_HOST=your-db-host.com
POSTGRESQL_PORT=5432
POSTGRESQL_USER=postgres
POSTGRESQL_PASSWORD='your-secure-password'
POSTGRESQL_DATABASE=mcp_context
POSTGRESQL_SSL_MODE=require

# Option B: Connection string
POSTGRESQL_CONNECTION_STRING=postgresql://user:password@host:5432/database?sslmode=require
```

**Important: Special Characters in Passwords**

If your password contains special characters (`$`, `#`, `&`, `*`, etc.), wrap it in **single quotes** (not double quotes):

```bash
# WRONG - $ will be interpreted as a variable
POSTGRESQL_PASSWORD="pass$word"

# CORRECT - single quotes prevent variable interpolation
POSTGRESQL_PASSWORD='pass$word'
```

### Supabase Configuration

For Supabase, use the Session Pooler connection (supports IPv4):

```bash
POSTGRESQL_HOST=aws-0-us-east-1.pooler.supabase.com
POSTGRESQL_PORT=5432
POSTGRESQL_USER=postgres.your-project-ref
POSTGRESQL_PASSWORD='your-database-password'
POSTGRESQL_DATABASE=postgres
POSTGRESQL_SSL_MODE=require
```

See the [Using with Supabase section in the Database Backends Guide](../database-backends.md#using-with-supabase) for detailed connection setup.

### Enabling row-level security

With [row-level security](../database-backends.md#row-level-security-postgresql), PostgreSQL enforces the access model in addition to the server. It needs requests to connect as a login role that does not own the tables, while the role that owns them keeps running the migrations.

**Bundled PostgreSQL files** (`docker-compose.postgresql.*.yml`): the app connects as the sidecar's bootstrap role, which owns the tables, and row-level security stays off. To enable it on a running deployment:

1. Create the NOLOGIN policy role and a login role that is its member, as the bootstrap role (`postgres`, unless you set another `POSTGRESQL_USER` before the first start):

   ```bash
   docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml exec postgres \
     psql -U postgres -d mcp_context \
     -c "CREATE ROLE mcp_context_request NOLOGIN; CREATE ROLE mcp_app LOGIN PASSWORD 'choose-a-password' NOSUPERUSER NOBYPASSRLS; GRANT mcp_context_request TO mcp_app;"
   ```

2. Create `deploy/docker/docker-compose.override.yml`, which points the app's requests at the login role and keeps the bootstrap role for migrations:

   ```yaml
   services:
     mcp-context-server:
       environment:
         - POSTGRESQL_USER=mcp_app
         - POSTGRESQL_PASSWORD=choose-a-password
         - POSTGRESQL_MIGRATION_USER=postgres
         - POSTGRESQL_MIGRATION_PASSWORD=postgres
         - ENABLE_POSTGRESQL_RLS=true
   ```

3. Start with both files. Compose reads the override only when it is named with a second `-f`, because the stack is started with an explicit `-f`:

   ```bash
   docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml -f deploy/docker/docker-compose.override.yml up -d
   ```

The first start installs the policies and the read view as `postgres`; later starts change nothing unless the configuration changed. The sidecar bootstraps its superuser from `POSTGRESQL_MIGRATION_USER` and `POSTGRESQL_MIGRATION_PASSWORD` when they are set and from `POSTGRESQL_USER` and `POSTGRESQL_PASSWORD` otherwise, so the same five variables can also go into `deploy/docker/.env`, which every bundled PostgreSQL file loads, without changing the role that owns the tables.

**External PostgreSQL files** (`docker-compose.postgresql-external.*.yml`): create the two roles in your database as a role administrator, then set the same variables in `deploy/docker/.env`: `POSTGRESQL_USER`/`POSTGRESQL_PASSWORD` (or the user in `POSTGRESQL_CONNECTION_STRING`) for the login role, `POSTGRESQL_MIGRATION_USER`/`POSTGRESQL_MIGRATION_PASSWORD` for the role that owns the tables, and `ENABLE_POSTGRESQL_RLS=true`. For Supabase user names and grants, see [Row-Level Security on Supabase](../database-backends.md#row-level-security-on-supabase).

A misconfigured role setup halts the container with exit code 78 and a log message naming the fix. To turn row-level security off again, remove what the server installed before starting with the flag off:

1. Stop the app, but not the sidecar: `docker compose -f <file> -f deploy/docker/docker-compose.override.yml stop mcp-context-server`. With an external database, run `docker compose -f <file> down`.
2. Remove the policies and the read view as the table owner, here from a one-off container on the stack's network (with an external database, pass its table owner's connection string instead):

   ```bash
   docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml run --rm --no-deps \
     --entrypoint mcp-context-server-migrate mcp-context-server \
     --disable-rls --source-url postgresql://postgres:postgres@postgres:5432/mcp_context
   ```

3. Delete the override, or set `ENABLE_POSTGRESQL_RLS=false` in it (or in `deploy/docker/.env` when you enabled it there), and start the stack again with `up -d`.

The full procedure, including the order other deployments follow, is in [Disabling row-level security](../database-backends.md#disabling-row-level-security).

### Optional .env for Ollama Configurations

Ollama configurations work without any `.env` file. However, you can optionally create one for additional settings like LangSmith tracing:

```bash
# Copy the optional template
cp deploy/docker/.env-sqlite.ollama.example deploy/docker/.env
# or
cp deploy/docker/.env-postgresql.ollama.example deploy/docker/.env

# Edit .env to enable optional features
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_your-langsmith-api-key
```

## Verification

### Health Check

```bash
# Check server health
curl http://localhost:8000/health

# Expected response
{"status": "ok"}
```

### Container Status

```bash
# Ollama deployment
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml ps

# OpenAI deployment
docker compose -f deploy/docker/docker-compose.sqlite.openai.yml ps

# Expected output: all services "healthy"
NAME                                    STATUS
context-server-mcp-context-server-1     Up (healthy)
context-server-ollama-1                 Up (healthy)   # Ollama only
context-server-postgres-1               Up (healthy)   # PostgreSQL only
```

### Model Availability (Ollama Only)

```bash
# Check if embedding model is loaded
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama list

# Expected output includes:
# qwen3-embedding:0.6b    ~600 MB
```

### Test MCP Connection

```bash
# List available tools via MCP
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

## Troubleshooting

### Issue 1: Ollama Health Check Failing

**Symptom:** `ollama` container stays in "starting" state

**Cause:** Embedding model download takes longer than expected (slow network)

**Solution:**
```bash
# Check download progress
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml logs -f ollama

# The entrypoint shows: "Pulling model: qwen3-embedding:0.6b..."
# Wait for: "Model pulled successfully!"
```

### Issue 2: PostgreSQL Connection Refused

**Symptom:** Server fails to connect to PostgreSQL

**Causes:**
- PostgreSQL container not yet ready
- Incorrect credentials in `.env`

**Solutions:**
```bash
# Check PostgreSQL logs
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml logs postgres

# Verify PostgreSQL is accepting connections
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml exec postgres pg_isready

# Test credentials manually
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml exec postgres \
  psql -U postgres -d mcp_context -c "SELECT 1"
```

### Issue 3: External PostgreSQL "getaddrinfo failed"

**Symptom:** Cannot connect to Supabase Direct Connection

**Cause:** IPv6 not available on your system

**Solution:** Use Session Pooler connection instead (see Supabase Configuration above)

### Issue 4: Port 8000 Already in Use

**Symptom:** Container fails to start, port binding error

**Solution:** Modify the port mapping in docker-compose file:
```yaml
ports:
  - "8001:8000"  # Use port 8001 on host
```

Then connect clients to `http://localhost:8001/mcp`

### Issue 5: Port 11434 Already in Use

**Symptom:** The `ollama` container fails to start with a port-binding error (for example, "Bind for 127.0.0.1:11434 failed: port is already allocated").

**Cause:** Another process on the host is already listening on TCP port 11434, most commonly a native Ollama installation running outside Docker.

**Solutions:**

1. Stop the conflicting host process (for example, stop the native Ollama service) and run `docker compose up -d` again.

2. Or remap the host-side port to a free port while keeping the container-side port at 11434:

```yaml
ollama:
  ports:
    - "127.0.0.1:11500:11434"  # Use port 11500 on the host loopback
```

Then update any host-side clients to reach the sidecar at `http://127.0.0.1:11500` instead of `http://127.0.0.1:11434`. Containers inside the same Compose project continue to reach the sidecar via the internal `http://ollama:11434` URL regardless of the host-side mapping.

### Issue 6: Semantic Search Not Available (Ollama)

**Symptom:** `semantic_search_context` tool not listed

**Causes:**
- Ollama not healthy yet
- Embedding model not downloaded

**Solutions:**
```bash
# Verify Ollama is healthy
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml ps ollama

# Check if model exists
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama list

# If model missing, trigger download
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama pull qwen3-embedding:0.6b
```

### Issue 7: OpenAI API Key Missing

**Symptom:** Server fails to start with OpenAI configuration

**Cause:** `.env` file missing or `OPENAI_API_KEY` not set

**Solution:**
```bash
# Ensure .env exists
cp deploy/docker/.env-sqlite.openai.example deploy/docker/.env

# Verify OPENAI_API_KEY is set
grep OPENAI_API_KEY deploy/docker/.env
```

### Issue 8: Summary Generation Not Working (Ollama)

**Symptom:** `summary` field is an empty string in search results

**Causes:**
- Summary model not pulled into the Ollama container
- `ENABLE_SUMMARY_GENERATION` set to `false`

**Solutions:**
```bash
# Pull the summary model
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama pull qwen3:0.6b

# Verify model is available
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama list

# Check server logs for summary errors
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml logs mcp-context-server | grep -i summary
```

**Note:** Summaries are generated before the database transaction during `store_context` and `update_context` operations (in parallel with embedding generation). The summary is available immediately in the response and in subsequent search results.

### Issue 9: Mixed Provider (Ollama Embeddings + OpenAI Summary) Infinite Restart

**Symptom:** Ollama container enters infinite restart loop when using `SUMMARY_PROVIDER=openai` with an Ollama compose file

**Cause:** Existing Ollama compose files (e.g., `docker-compose.sqlite.ollama.yml`) pass `SUMMARY_MODEL` to the Ollama sidecar. When the summary model is an OpenAI model name (e.g., `gpt-5.4-nano`), the entrypoint fails trying to pull it from Ollama. Additionally, the healthcheck requires both embedding and summary models to be present in Ollama.

**Solution:** Use the dedicated mixed-provider compose files instead:

```bash
# SQLite + Ollama Embeddings + OpenAI Summary
docker compose -f deploy/docker/docker-compose.sqlite.ollama-openai.yml up -d

# PostgreSQL + Ollama Embeddings + OpenAI Summary
docker compose -f deploy/docker/docker-compose.postgresql.ollama-openai.yml up -d

# External PostgreSQL + Ollama Embeddings + OpenAI Summary
docker compose -f deploy/docker/docker-compose.postgresql-external.ollama-openai.yml up -d
```

These files are specifically designed for mixed providers: the Ollama sidecar only handles embedding models, and the OpenAI summary provider is configured on the MCP server service.

### Common Error Messages

| Error                                  | Cause                          | Solution                                                                                                 |
|----------------------------------------|--------------------------------|----------------------------------------------------------------------------------------------------------|
| `connection refused`                   | Service not running            | Check container status with `docker compose ps`                                                          |
| `model not found`                      | Embedding model not pulled     | Wait for automatic download or pull manually                                                             |
| `permission denied`                    | Volume permission issue        | Check volume ownership matches UID 10001                                                                 |
| `database is locked`                   | SQLite concurrent access       | Expected for SQLite; use PostgreSQL for concurrency                                                      |
| `Invalid API Key`                      | OpenAI key incorrect           | Verify OPENAI_API_KEY in .env file                                                                       |
| `password authentication failed`       | Wrong PostgreSQL password      | Fix POSTGRESQL_PASSWORD, or POSTGRESQL_MIGRATION_PASSWORD when the error names it, and restart (exit 78) |
| `database "..." does not exist`        | PostgreSQL database not found  | Create database or fix POSTGRESQL_DATABASE (exit 78)                                                     |
| `pgvector extension is not installed`  | pgvector not enabled           | Enable via dashboard or CREATE EXTENSION (exit 78)                                                       |
| `insufficient privileges`              | Cannot create pgvector         | Grant permissions via dashboard (exit 78)                                                                |
| `[Errno 111] Connection refused`       | PostgreSQL not running         | Wait for PostgreSQL to start (exit 69, will retry)                                                       |
| `summary field is empty`               | Summary model not pulled       | Run `ollama pull qwen3:0.6b` in the ollama container                                                     |

## Advanced Configuration

### Custom Embedding Model (Ollama)

To use a different Ollama embedding model, update both services:

```yaml
# In docker-compose file
services:
  mcp-context-server:
    environment:
      - EMBEDDING_MODEL=nomic-embed-text
      - EMBEDDING_DIM=1024

  ollama:
    environment:
      - EMBEDDING_MODEL=nomic-embed-text
```

### Custom Embedding Model (OpenAI)

To use a different OpenAI embedding model:

```yaml
# In docker-compose file
services:
  mcp-context-server:
    environment:
      - EMBEDDING_MODEL=text-embedding-3-large
      - EMBEDDING_DIM=3072  # Adjust for model
```

### GPU Acceleration (Ollama Only)

GPU acceleration enables faster embedding generation and summary inference. Docker Compose files include commented GPU configuration sections for NVIDIA, AMD (ROCm), and Intel (Vulkan) GPUs.

For complete setup instructions, troubleshooting, and vendor-specific details, see the [GPU Acceleration Guide](gpu-acceleration.md).

**Quick start (NVIDIA example):**

Uncomment the NVIDIA GPU section in your Docker Compose file:

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**AMD ROCm:**

Override the Ollama image to the ROCm-tagged variant and uncomment the AMD GPU section:

```yaml
services:
  ollama:
    image: ollama/ollama:rocm
```

See the [GPU Acceleration Guide](gpu-acceleration.md) for AMD and Intel/Vulkan configuration.

### Building Images

The MCP Context Server image is built from the repository root `Dockerfile`:

```bash
docker build -t mcp-context-server .
```

The Ollama service uses the published `ollama/ollama:latest` image directly (or `ollama/ollama:rocm` for AMD GPUs); no local Ollama image build is required.

### Building with Different Embedding Providers

The Dockerfile supports building with different embedding providers via build arguments:

```bash
# Build with Ollama embeddings (default)
docker build -t mcp-context-server .

# Build with OpenAI embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-openai -t mcp-context-server-openai .

# Build with Azure OpenAI embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-azure -t mcp-context-server-azure .

# Build with HuggingFace embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-huggingface -t mcp-context-server-huggingface .

# Build with Voyage embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-voyage -t mcp-context-server-voyage .

# Build with all embedding providers
docker build --build-arg EMBEDDING_EXTRA=embeddings-all -t mcp-context-server-all .
```

**Available Embedding Extras:**

| Extra                    | Provider         | Package               |
|--------------------------|------------------|-----------------------|
| `embeddings-ollama`      | Ollama (default) | langchain-ollama      |
| `embeddings-openai`      | OpenAI           | langchain-openai      |
| `embeddings-azure`       | Azure OpenAI     | langchain-openai      |
| `embeddings-huggingface` | HuggingFace      | langchain-huggingface |
| `embeddings-voyage`      | Voyage AI        | langchain-voyageai    |
| `embeddings-all`         | All providers    | All packages          |

**Note:** The OpenAI and mixed-provider Docker Compose configurations automatically pass the correct build arguments. If using Ollama configurations, no build argument is needed (default is `embeddings-ollama`).

### Building with Different Summary Providers

The Dockerfile supports building with different summary providers via the `SUMMARY_EXTRA` build argument:

```bash
# Build with Ollama summaries (default)
docker build -t mcp-context-server .

# Build with OpenAI summaries
docker build --build-arg SUMMARY_EXTRA=summary-openai -t mcp-context-server-openai-summary .

# Build with Anthropic summaries
docker build --build-arg SUMMARY_EXTRA=summary-anthropic -t mcp-context-server-anthropic-summary .

# Combine embedding and summary build args
docker build \
  --build-arg EMBEDDING_EXTRA=embeddings-openai \
  --build-arg SUMMARY_EXTRA=summary-openai \
  -t mcp-context-server-openai .

# Mixed provider: Ollama embeddings + OpenAI summaries
docker build \
  --build-arg EMBEDDING_EXTRA=embeddings-ollama \
  --build-arg SUMMARY_EXTRA=summary-openai \
  -t mcp-context-server-ollama-openai .
```

**Available Summary Extras:**

| Extra               | Provider          | Package               | Default Model     |
|---------------------|-------------------|-----------------------|-------------------|
| `summary-ollama`    | Ollama (default)  | langchain-ollama      | qwen3:0.6b        |
| `summary-openai`    | OpenAI            | langchain-openai      | gpt-5.4-nano      |
| `summary-anthropic` | Anthropic         | langchain-anthropic   | claude-haiku-4-5  |

To disable summary generation entirely (no `SUMMARY_EXTRA` needed):

```yaml
environment:
  - ENABLE_SUMMARY_GENERATION=false
```

For detailed summary generation configuration, see the [Summary Generation Guide](../summary-generation.md).

### Horizontal Scaling with Docker Compose

For horizontal scaling with Docker Compose, use a PostgreSQL configuration. Stateless HTTP mode is enabled by default:

```yaml
services:
  mcp-context-server:
    deploy:
      replicas: 3
```

**Requirements:**
- PostgreSQL backend (`STORAGE_BACKEND=postgresql`) -- SQLite does not support multiple writers
- Stateless HTTP mode is enabled by default; no additional configuration needed

For production horizontal scaling, Kubernetes with Helm is recommended. See the [Kubernetes Deployment Guide](kubernetes.md#horizontal-scaling).

### Restart Policies and Exit Codes

The MCP Context Server uses BSD sysexits.h exit codes to signal different failure types to container supervisors:

| Exit Code | Meaning                  | Docker Restart Behavior                   |
|-----------|--------------------------|-------------------------------------------|
| 0         | Normal shutdown          | No restart                                |
| 1         | General error            | Restart if policy allows                  |
| 69        | Dependency unavailable   | Restart with backoff (may recover)        |
| 78        | Configuration error      | Container halted (prevents restart loops) |

**Exit Code 69 (EX_UNAVAILABLE)** indicates external dependencies are temporarily unavailable:
- Ollama service not running (may start later)
- Database temporarily unreachable (PostgreSQL connection refused, network timeout)
- Network connectivity issues

**Exit Code 78 (EX_CONFIG)** indicates configuration problems requiring human intervention:
- Missing API keys (OPENAI_API_KEY, VOYAGE_API_KEY, etc.)
- Missing required packages
- Invalid configuration values
- Embedding model not found (requires `ollama pull` or fix EMBEDDING_MODEL)
- PostgreSQL authentication errors (wrong password, database doesn't exist, missing pgvector extension)
- PostgreSQL row-level security misconfiguration (`ENABLE_POSTGRESQL_RLS`, `POSTGRESQL_MIGRATION_USER`, `POSTGRESQL_RLS_POLICY_ROLE`, or the request role's membership and privileges)
- `ENABLE_POSTGRESQL_RLS=false` over installed policies (run `mcp-context-server-migrate --disable-rls` first)

**For detailed PostgreSQL error scenarios,** see [PostgreSQL Backend Error Scenarios](#postgresql-backend-error-scenarios) below.

### Entrypoint Wrapper Script

The Docker image includes an entrypoint wrapper script (`docker-entrypoint.sh`) that intercepts exit codes to prevent infinite restart loops:

**How it works:**

```text
Docker starts container
    |
    v
docker-entrypoint.sh runs Python server
    |
    v
Server exits with code (0, 69, 78, or other)
    |
    v
Entrypoint interprets exit code:
    - Code 0: Normal exit
    - Code 78: Configuration error -> exec sleep infinity (container stays running but idle)
    - Code 69: Dependency error -> exit 69 (Docker may restart)
    - Other: Pass through exit code
```

**Configuration errors (exit code 78) - Container halted:**

When the server encounters a configuration error (e.g., embedding model not found), the entrypoint:

1. Prints a clear error message with troubleshooting steps
2. Replaces itself with `sleep infinity`
3. Container remains "Running" (but idle) instead of entering a restart loop

This allows you to:
- Inspect logs with `docker logs <container>`
- Fix the configuration
- Stop and restart manually: `docker stop <container> && docker compose up -d`

**Dependency errors (exit code 69) - May retry:**

When the server encounters a dependency error (e.g., Ollama not running yet), the entrypoint:

1. Prints an informational message
2. Exits with code 69
3. Docker's `on-failure:5` policy may restart the container (up to 5 times)

**Default restart policy:**

All Docker Compose configurations use `restart: "on-failure:5"` for the MCP Context Server:

```yaml
services:
  mcp-context-server:
    restart: "on-failure:5"  # Restart up to 5 times on failure
```

This provides defense in depth:
- Entrypoint handles configuration errors (code 78) by halting
- Restart policy limits dependency error retries (code 69) to 5 attempts

For production, the built-in health check is already configured:

```yaml
services:
  mcp-context-server:
    restart: "on-failure:5"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"]
      interval: 30s
      timeout: 5s
      start_period: 10s
      retries: 3
```

### PostgreSQL Backend Error Scenarios

When using the PostgreSQL backend, the server classifies initialization errors to control container restart behavior. This prevents infinite restart loops on configuration errors while allowing retry for transient connectivity issues.

**Error Classification:**

| Error Scenario            | Exception Type                   | Exit Code               | Container Behavior    | Fix Required                                 |
|---------------------------|----------------------------------|-------------------------|-----------------------|----------------------------------------------|
| PostgreSQL not running    | Connection refused (Errno 111)   | 69 (DependencyError)    | Retry with backoff    | Wait for PostgreSQL to start                 |
| Network timeout           | `TimeoutError`, `InterfaceError` | 69 (DependencyError)    | Retry with backoff    | Check network connectivity                   |
| Too many connections      | `TooManyConnectionsError`        | 69 (DependencyError)    | Retry with backoff    | Increase `max_connections` or wait           |
| Wrong password            | `InvalidPasswordError`           | 78 (ConfigurationError) | Halt (sleep infinity) | Fix the password variable the error names    |
| Database does not exist   | `InvalidCatalogNameError`        | 78 (ConfigurationError) | Halt (sleep infinity) | Create database or fix `POSTGRESQL_DATABASE` |
| pgvector not installed    | Extension check failure          | 78 (ConfigurationError) | Halt (sleep infinity) | Enable pgvector extension                    |
| Insufficient privileges   | `InsufficientPrivilegeError`     | 78 (ConfigurationError) | Halt (sleep infinity) | Grant permissions via dashboard              |
| Codec registration failed | pgvector codec error             | 78 (ConfigurationError) | Halt (sleep infinity) | Check pgvector installation                  |

**Examples:**

**Scenario 1: PostgreSQL Not Running (Exit 69)**
```text
[ERROR] Failed to connect to PostgreSQL: [Errno 111] Connection refused
[docker-entrypoint] Server exited with code 69
```
- **Container behavior:** Restarts up to 5 times (restart policy)
- **Why retry makes sense:** PostgreSQL service may start soon
- **Fix:** Ensure PostgreSQL container is running and healthy

**Scenario 2: Wrong Database Password (Exit 78)**
```text
[ERROR] PostgreSQL authentication failed: password authentication failed
[docker-entrypoint] CONFIGURATION ERROR - CONTAINER HALTED
[docker-entrypoint] Server exited with code 78
```
- **Container behavior:** Halts (remains running but idle with `sleep infinity`)
- **Why halt:** Password won't fix itself - requires human intervention
- **Fix:** Update the password variable the error names in the `.env` file (`POSTGRESQL_PASSWORD`, or `POSTGRESQL_MIGRATION_PASSWORD` for the role that owns the tables) and restart the container

**Scenario 3: pgvector Extension Not Installed (Exit 78)**
```text
[ERROR] pgvector extension is not installed
[docker-entrypoint] CONFIGURATION ERROR - CONTAINER HALTED
[docker-entrypoint] Server exited with code 78
```
- **Container behavior:** Halts (remains running but idle)
- **Why halt:** Extension must be manually enabled
- **Fix:** Enable pgvector via Supabase Dashboard → Extensions or `CREATE EXTENSION vector;`

**Scenario 4: Database Does Not Exist (Exit 78)**
```text
[ERROR] PostgreSQL database does not exist: database "mcp_context" does not exist
[docker-entrypoint] CONFIGURATION ERROR - CONTAINER HALTED
[docker-entrypoint] Server exited with code 78
```
- **Container behavior:** Halts (remains running but idle)
- **Why halt:** Database must be created manually
- **Fix:** Create database or fix `POSTGRESQL_DATABASE` environment variable

**Recovery Steps for Configuration Errors (Exit 78):**

1. **Check container logs** to identify specific error:
   ```bash
   docker logs <container_name>
   ```

2. **Fix the configuration** based on error message:
   - Wrong password: Update the password variable the error names in the `.env` file (`POSTGRESQL_PASSWORD`, or `POSTGRESQL_MIGRATION_PASSWORD` for the role that owns the tables)
   - Missing database: Create database or fix `POSTGRESQL_DATABASE`
   - Missing extension: Enable pgvector via dashboard or SQL
   - Permission denied: Grant required permissions (Supabase: use dashboard)

3. **Stop and restart** the container:
   ```bash
   docker stop <container_name>
   docker compose -f <your-compose-file>.yml up -d
   ```

**Understanding Exit Code 69 vs 78:**

- **Exit 69 (Dependency Error):** External dependency temporarily unavailable
  - Examples: PostgreSQL starting up, network hiccup, connection pool exhausted
  - Container restarts automatically (may succeed when dependency becomes available)
  - No human intervention needed

- **Exit 78 (Configuration Error):** Configuration problem requiring human fix
  - Examples: Wrong password, missing database, missing extension
  - Container halts to prevent infinite restart loop
  - Requires manual fix and restart

### Troubleshooting Container Halted State

If your container shows "Running" but the server is not responding:

1. **Check container logs:**
   ```bash
   docker logs <container_name>
   ```

2. **Look for "CONFIGURATION ERROR - CONTAINER HALTED" message:**
   ```text
   ==============================================
   [FATAL] CONFIGURATION ERROR - CONTAINER HALTED
   ==============================================
   ```

3. **Fix the configuration** based on the error message (e.g., pull missing model, set API key)

4. **Restart the container:**
   ```bash
   docker stop <container_name>
   docker compose -f <your-compose-file>.yml up -d
   ```

### Production Considerations

1. **Change default PostgreSQL password** in production deployments
2. **Serve requests through a dedicated login role** with [row-level security](#enabling-row-level-security) when more than one principal uses the server
3. **Use named volumes** for data persistence (already configured)
4. **Configure resource limits** for container memory/CPU
5. **Set up monitoring** using the `/health` endpoint
6. **Use reverse proxy** (nginx, traefik) for TLS termination, with a request body limit above the server's own 4 MiB and read and send timeouts long enough for batch calls with generation on; for nginx, `client_max_body_size 5m; proxy_read_timeout 3600s; proxy_send_timeout 3600s;` covers `mcp-context-ingest` at its defaults (see [Limits](../knowledge-base.md#limits))
7. **Secure OpenAI API key** - never commit `.env` files to version control
8. **Set restart policies** with limited retry count to prevent infinite loops

## Files Reference

### Docker Compose Files

| File                                                         | Description                                                      | Image Source |
|--------------------------------------------------------------|------------------------------------------------------------------|--------------|
| `docker-compose.sqlite.ollama.yml`                           | SQLite + Ollama embeddings                                       | GHCR         |
| `docker-compose.sqlite.ollama.local.yml`                     | SQLite + Ollama embeddings (local build)                         | Local        |
| `docker-compose.postgresql.ollama.yml`                       | PostgreSQL + Ollama embeddings                                   | GHCR         |
| `docker-compose.postgresql.ollama.local.yml`                 | PostgreSQL + Ollama embeddings (local build)                     | Local        |
| `docker-compose.postgresql-external.ollama.yml`              | External PostgreSQL + Ollama embeddings                          | GHCR         |
| `docker-compose.postgresql-external.ollama.local.yml`        | External PostgreSQL + Ollama embeddings (local build)            | Local        |
| `docker-compose.sqlite.openai.yml`                           | SQLite + OpenAI embeddings                                       | Local        |
| `docker-compose.postgresql.openai.yml`                       | PostgreSQL + OpenAI embeddings                                   | Local        |
| `docker-compose.postgresql-external.openai.yml`              | External PostgreSQL + OpenAI embeddings                          | Local        |
| `docker-compose.sqlite.ollama-openai.yml`                    | SQLite + Ollama embeddings + OpenAI summary                      | GHCR         |
| `docker-compose.sqlite.ollama-openai.local.yml`              | SQLite + Ollama embeddings + OpenAI summary (local build)        | Local        |
| `docker-compose.postgresql.ollama-openai.yml`                | PostgreSQL + Ollama embeddings + OpenAI summary                  | GHCR         |
| `docker-compose.postgresql.ollama-openai.local.yml`          | PostgreSQL + Ollama embeddings + OpenAI summary (local)          | Local        |
| `docker-compose.postgresql-external.ollama-openai.yml`       | External PostgreSQL + Ollama embeddings + OpenAI summary         | GHCR         |
| `docker-compose.postgresql-external.ollama-openai.local.yml` | External PostgreSQL + Ollama embeddings + OpenAI summary (local) | Local        |

### Environment Templates

| File                                             | Description                                         | Required |
|--------------------------------------------------|-----------------------------------------------------|----------|
| `.env-sqlite.ollama.example`                     | Optional config for SQLite + Ollama                 | No       |
| `.env-postgresql.ollama.example`                 | Optional config for PostgreSQL + Ollama             | No       |
| `.env-postgresql-external.ollama.example`        | PostgreSQL connection for external DB + Ollama      | Yes      |
| `.env-sqlite.openai.example`                     | OpenAI API key for SQLite + OpenAI                  | Yes      |
| `.env-postgresql.openai.example`                 | OpenAI API key for PostgreSQL + OpenAI              | Yes      |
| `.env-postgresql-external.openai.example`        | OpenAI + PostgreSQL for external DB                 | Yes      |
| `.env-sqlite.ollama-openai.example`              | OpenAI API key for SQLite + Ollama/OpenAI           | Yes      |
| `.env-postgresql.ollama-openai.example`          | OpenAI API key for PostgreSQL + Ollama/OpenAI       | Yes      |
| `.env-postgresql-external.ollama-openai.example` | OpenAI + PostgreSQL for external DB + Ollama/OpenAI | Yes      |

### Other Files

| File                                 | Description                                       |
|--------------------------------------|---------------------------------------------------|
| `Dockerfile`                         | Multi-stage MCP server image (repository root)    |
| `deploy/docker/docker-entrypoint.sh` | Exit code handler to prevent infinite restart     |
| `.dockerignore`                      | Build context optimization (repository root)      |

## Additional Resources

### Related Documentation

- **API Reference**: [API Reference](../api-reference.md) - complete tool documentation
- **Database Backends**: [Database Backends Guide](../database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](../semantic-search.md) - vector similarity search configuration
- **Full-Text Search**: [Full-Text Search Guide](../full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](../hybrid-search.md) - combined search with RRF fusion
- **Metadata Filtering**: [Metadata Guide](../metadata-addition-updating-and-filtering.md) - metadata filtering with operators
- **Authentication**: [Authentication Guide](../authentication.md) - bearer token authentication
- **Knowledge Base**: [Knowledge Base Guide](../knowledge-base.md) - document corpora and `mcp-context-ingest`
- **Main Documentation**: [README.md](../../README.md) - overview and quick start

### GPU Acceleration

- **GPU Setup**: [GPU Acceleration Guide](gpu-acceleration.md) - NVIDIA, AMD, and Intel GPU setup

### Kubernetes Deployment

For Kubernetes deployments, see:
- **Helm Chart**: [Helm Deployment Guide](helm.md)
- **Raw Manifests**: [Kubernetes Guide](kubernetes.md)
