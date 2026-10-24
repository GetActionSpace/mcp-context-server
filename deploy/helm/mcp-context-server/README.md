# MCP Context Server Helm Chart

A Helm chart for deploying MCP Context Server on Kubernetes.

## Prerequisites

- Kubernetes 1.21+
- Helm 3.8+
- PV provisioner support (for SQLite persistence)

## Installation

### Quick Start (SQLite)

```bash
helm install mcp ./deploy/helm/mcp-context-server
```

### SQLite with Persistence

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-sqlite.yaml
```

### PostgreSQL

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-postgresql.yaml \
  --set storage.postgresql.host=your-postgres-host \
  --set storage.postgresql.password=your-password
```

### With Semantic Search (Ollama Sidecar)

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set search.semantic.enabled=true \
  --set ollama.enabled=true
```

### With Summary Generation (Ollama Sidecar)

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set search.summary.enabled=true \
  --set ollama.enabled=true
```

## Configuration

### Key Values

| Parameter                           | Description                                                      | Default                                     |
|-------------------------------------|------------------------------------------------------------------|---------------------------------------------|
| `image.repository`                  | Image repository                                                 | `ghcr.io/getactionspace/mcp-context-server` |
| `image.tag`                         | Image tag                                                        | `Chart.appVersion`                          |
| `replicaCount`                      | Number of replicas                                               | `1`                                         |
| `env.FASTMCP_STATELESS_HTTP`        | Stateless HTTP mode (enabled by default)                         | `"true"`                                    |
| `env.FASTMCP_ENABLE_RICH_LOGGING`   | Rich log formatting (disable for containers)                     | `"false"`                                   |
| `service.type`                      | Kubernetes service type                                          | `ClusterIP`                                 |
| `service.port`                      | Service port                                                     | `8000`                                      |
| `storage.backend`                   | Storage backend (sqlite/postgresql)                              | `sqlite`                                    |
| `storage.postgresql.migration.user` | Role that owns the tables and runs migrations                    | `""` (request user)                         |
| `storage.postgresql.rls.enabled`    | PostgreSQL row-level security (`ENABLE_POSTGRESQL_RLS`)          | `false`                                     |
| `storage.postgresql.rls.policyRole` | NOLOGIN role the policies name                                   | `""` (`mcp_context_request`)                |
| `auth.provider`                     | HTTP authentication (`none`, `simple_token` or `jwt`)            | `""` (`none`)                               |
| `auth.jwt.anonymousPublicKey`       | Anonymous verifier key, used beside `auth.jwt.jwksUri`           | `""` (no anonymous verifier)                |
| `auth.jwt.anonymousAlgorithm`       | Signing algorithm of the anonymous verifier                      | `""` (`RS256`)                              |
| `accessControl.publishRole`         | Role required to publish entries as `public`                     | `""` (any owner may publish)                |
| `accessControl.anonymousPrincipal`  | Read-only jwt subject every write tool refuses                   | `""` (no principal is read-only)            |
| `search.fts.enabled`                | Full-text search tool (`true`=auto-register, `false`=off)        | `true`                                      |
| `search.semantic.enabled`           | Semantic search + embedding subsystem (`true`=auto, `false`=off) | `false`                                     |
| `search.hybrid.enabled`             | Hybrid search tool (`true`=auto-register, `false`=off)           | `true`                                      |
| `search.chunking.enabled`           | Enable text chunking for embeddings                              | `true`                                      |
| `search.chunking.size`              | Chunk size in characters                                         | `1500`                                      |
| `search.reranking.enabled`          | Enable cross-encoder reranking                                   | `true`                                      |
| `search.reranking.model`            | Reranking model name                                             | `ms-marco-MiniLM-L-12-v2`                   |
| `search.summary.enabled`            | Enable LLM-based summary generation (opt-in; needs a provider)   | `false`                                     |
| `search.summary.provider`           | Summary provider (ollama/openai/anthropic)                       | `"ollama"`                                  |
| `search.summary.model`              | Summary generation model                                         | `"qwen3:0.6b"`                              |
| `ollama.enabled`                    | Enable Ollama sidecar                                            | `false`                                     |

### Storage Configuration

#### SQLite

```yaml
storage:
  backend: sqlite
  sqlite:
    enabled: true
    path: /data/context_storage.db
    persistence:
      enabled: true
      size: 1Gi
      storageClassName: ""
```

#### PostgreSQL

```yaml
storage:
  backend: postgresql
  postgresql:
    enabled: true
    host: "postgresql-host"
    port: "5432"
    user: "postgres"
    password: "your-password"
    database: "mcp_context"
    sslMode: "prefer"
```

For production, use an existing secret:

```yaml
storage:
  postgresql:
    existingSecret: "my-postgres-secret"
    existingSecretKey: "password"
```

#### Row-Level Security

PostgreSQL can enforce the access model through row-level security as well. Requests then connect as a login role that is a member of a NOLOGIN policy role and is neither superuser nor `BYPASSRLS`, while the role that owns the tables runs the migrations inside the server pod at startup. Create the roles first:

```sql
CREATE ROLE mcp_context_request NOLOGIN;
CREATE ROLE mcp_app LOGIN PASSWORD 'choose-a-password' NOSUPERUSER NOBYPASSRLS;
GRANT mcp_context_request TO mcp_app;
```

```yaml
storage:
  backend: postgresql
  postgresql:
    enabled: true
    user: "mcp_app"
    existingSecret: "mcp-postgresql"   # key postgresql-password
    migration:
      user: "postgres"
      existingSecret: "mcp-postgresql" # key postgresql-migration-password
    rls:
      enabled: true
```

Rendering fails when `rls.enabled` is set without `migration.user` or with the SQLite backend. To rotate the login role, create a new one, grant it the policy role, change `storage.postgresql.user` and upgrade; no DDL runs. To disable, set `rls.enabled: false` in your values, scale the Deployment to zero, run `mcp-context-server-migrate --disable-rls` from a one-off pod with the table owner's credentials, then upgrade and scale back up. The [Helm Deployment Guide](../../../docs/deployment/helm.md#row-level-security) has the full procedure, and the `auth` and `accessControl` values are listed there too.

### Search Features

Enable all search features:

```yaml
search:
  fts:
    enabled: true
    language: "english"
  semantic:
    enabled: true
    model: "qwen3-embedding:0.6b"
    dim: 1024
    # Optional query-side instruction for instruct-aware models such as qwen3-embedding
    # (double-quoted scalar so \n becomes a real newline; applies to search queries only)
    queryInstruction: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
  hybrid:
    enabled: true
    rrfK: 60
  chunking:
    enabled: true
    size: 1500
    overlap: 150
  reranking:
    enabled: true
    model: "ms-marco-MiniLM-L-12-v2"

ollama:
  enabled: true
```

### Text Chunking

Text chunking splits long documents into smaller chunks for embedding generation, improving semantic search quality. Enabled by default.

```yaml
search:
  chunking:
    enabled: true
    size: 1500       # Chunk size in characters
    overlap: 150     # Overlap between chunks
    aggregation: max # Score aggregation (only 'max' supported)
```

### Reranking

Cross-encoder reranking improves search precision by re-scoring results. Enabled by default with FlashRank.

```yaml
search:
  reranking:
    enabled: true
    provider: flashrank
    model: ms-marco-MiniLM-L-12-v2  # ~34MB, downloads on startup
    maxLength: 512
    overfetch: 4
```

### Summary Generation

LLM-based summary generation creates concise summaries of stored context entries. Opt-in in the Helm chart (default `search.summary.enabled: false`): enable it together with a summary provider -- set `ollama.enabled: true` for `provider: ollama`, or configure an API key under `summarySecrets` for openai/anthropic. Enabling it without a reachable provider fails startup, mirroring `search.semantic.enabled`.

```yaml
search:
  summary:
    enabled: false           # opt-in; set true together with a provider (and ollama.enabled)
    provider: ollama         # ollama, openai, or anthropic
    model: "qwen3:0.6b"     # Summary model
    maxTokens: 4000          # Max output tokens (50-16384)
    minContentLength: 500    # Min chars to trigger summary (0=always)
```

For OpenAI or Anthropic providers:

```yaml
summarySecrets:
  openaiApiKey: "sk-..."       # For provider=openai
  anthropicApiKey: "sk-ant-..."  # For provider=anthropic
  existingSecret: ""           # Use pre-existing secret
  existingSecretKey: ""        # Override the key name inside existingSecret
```

When `existingSecret` is set, the chart reads the API key from a fixed data key inside that secret unless you override it with `existingSecretKey`. The default key name matches the active provider: `openai-api-key`, `azure-openai-api-key`, `huggingface-api-token`, or `voyage-api-key` for `embeddingSecrets`, and `openai-api-key` or `anthropic-api-key` for `summarySecrets`. Set `existingSecretKey` when your secret stores the key under a different name (otherwise the pod fails to start with `CreateContainerConfigError`). The same `existingSecret` / `existingSecretKey` pair is available under `embeddingSecrets` for the semantic-search provider keys.

### Ingress

Enable ingress with TLS:

```yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: mcp.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mcp-tls
      hosts:
        - mcp.example.com
```

## Upgrading

```bash
helm upgrade mcp ./deploy/helm/mcp-context-server
```

## Uninstalling

```bash
helm uninstall mcp
```

Note: PersistentVolumeClaims are not deleted automatically. To remove data:

```bash
kubectl delete pvc -l app.kubernetes.io/instance=mcp
```

## Resources

- [MCP Context Server Documentation](https://github.com/GetActionSpace/mcp-context-server)
- [Helm Documentation](https://helm.sh/docs/)
