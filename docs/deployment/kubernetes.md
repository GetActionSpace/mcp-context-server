# Kubernetes Deployment Guide

This guide covers deploying the MCP Context Server on Kubernetes.

## Deployment Options

There are two ways to deploy MCP Context Server on Kubernetes:

### 1. Helm Chart (Recommended)

The Helm chart provides a complete, configurable deployment with:
- Customizable values for different environments
- Pre-configured health checks and probes
- Support for SQLite and PostgreSQL backends
- Optional Ollama sidecar for semantic search
- Ingress configuration
- Service account management

See the [Helm Deployment Guide](helm.md) for detailed instructions.

### 2. Raw Kubernetes Manifests

For users who prefer `kubectl apply` over Helm, you can generate manifests from the Helm chart:

```bash
# Generate manifests with default values
helm template mcp ./deploy/helm/mcp-context-server > manifests.yaml

# Apply to cluster
kubectl apply -f manifests.yaml
```

See `deploy/kubernetes/README.md` for more details.

## Quick Start with Helm

### SQLite (Simple)

```bash
helm install mcp ./deploy/helm/mcp-context-server
```

### PostgreSQL (Production)

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-postgresql.yaml \
  --set storage.postgresql.host=your-postgres-host \
  --set storage.postgresql.password=your-password
```

### With Semantic Search

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set search.semantic.enabled=true \
  --set ollama.enabled=true
```

## Architecture Overview

### Components

```text
+-------------------+     +------------------+     +------------------+
|   MCP Client      |---->|  Ingress/Service |---->| MCP Context      |
| (Claude, etc.)    |     |                  |     | Server Pod       |
+-------------------+     +------------------+     +------------------+
                                                          |
                          +------------------+            |
                          |  Ollama Sidecar  |<-----------+ (optional)
                          |  (embeddings)    |
                          +------------------+
                                                          |
                          +------------------+            |
                          |  SQLite PVC or   |<-----------+
                          |  PostgreSQL      |
                          +------------------+
```

### Resource Requirements

**Minimum:**
- CPU: 100m
- Memory: 256Mi

**Recommended (with semantic search):**
- CPU: 500m-2000m
- Memory: 2Gi-4Gi (Ollama requires significant memory)

### Storage

**SQLite:**
- Requires PersistentVolumeClaim
- Single replica only (SQLite doesn't support concurrent writes)
- Good for development and single-user deployments

**PostgreSQL:**
- External PostgreSQL required (not included in chart)
- Supports multiple replicas
- Recommended for production

## Horizontal Scaling

MCP Context Server supports horizontal scaling with multiple replicas when configured correctly.

### Requirements

Horizontal scaling requires ALL of the following:

| Requirement         | Setting                       | Why                                   |
|---------------------|-------------------------------|---------------------------------------|
| HTTP transport      | `MCP_TRANSPORT=http`          | stdio does not support network access |
| Stateless HTTP mode | Enabled by default            | Eliminates server-side session state  |
| PostgreSQL backend  | `STORAGE_BACKEND=postgresql`  | SQLite is file-based, single-writer   |

**Why stateless HTTP mode matters:**

Stateless HTTP mode is enabled by default in MCP Context Server because it uses no stateful MCP features. Each request creates a fresh transport context, eliminating the need for session affinity (sticky sessions). If needed, you can disable it by setting `FASTMCP_STATELESS_HTTP=false`. The setting applies to clients on handshake-era protocol versions; clients that negotiate the sessionless `2026-07-28` protocol never create server sessions.

### Helm Configuration

```yaml
# values-horizontal-scaling.yaml
replicaCount: 3

env:
  MCP_TRANSPORT: "http"
  # FASTMCP_STATELESS_HTTP defaults to "true" -- no override needed

storage:
  backend: postgresql
  sqlite:
    enabled: false
  postgresql:
    enabled: true
    host: "postgresql-host"
    password: ""  # Set via --set or existingSecret
```

Apply with:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-postgresql.yaml \
  -f values-horizontal-scaling.yaml \
  --set storage.postgresql.host=your-postgres-host \
  --set storage.postgresql.password=your-password
```

### Important Notes

- **SQLite does NOT support horizontal scaling** -- it uses a single file on disk with single-writer limitation
- Each replica maintains its own connection pool to PostgreSQL
- Rolling updates work seamlessly since each request is independent

## Configuration

### Environment Variables

The deployment uses a ConfigMap for non-sensitive configuration:

| Variable                    | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `MCP_TRANSPORT`             | Transport mode (http for Kubernetes)                         |
| `FASTMCP_HOST`              | HTTP bind address                                            |
| `FASTMCP_PORT`              | HTTP port                                                    |
| `FASTMCP_STATELESS_HTTP`    | Stateless HTTP mode (enabled by default)                     |
| `LOG_LEVEL`                 | Logging level                                                |
| `STORAGE_BACKEND`           | sqlite or postgresql                                         |
| `ENABLE_SUMMARY_GENERATION` | Enable automatic LLM-based summary generation                |
| `SUMMARY_PROVIDER`          | Summary provider: ollama, openai, or anthropic               |
| `SUMMARY_MODEL`             | Summary model name (default: qwen3:0.6b for Ollama)          |
| `SUMMARY_MAX_TOKENS`        | Maximum output tokens for summary generation (default: 4000) |
| `ENABLE_FTS`                | Full-text search                                             |
| `ENABLE_SEMANTIC_SEARCH`    | Semantic search                                              |
| `ENABLE_HYBRID_SEARCH`      | Hybrid search                                                |

### Secrets

PostgreSQL credentials are stored in a Kubernetes Secret:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mcp-context-server
type: Opaque
data:
  postgresql-password: <base64-encoded-password>
```

Or use an existing secret:

```yaml
storage:
  postgresql:
    existingSecret: "my-postgres-secret"
    existingSecretKey: "password"
```

## Networking

### Service

The chart creates a ClusterIP service by default. To expose externally:

**LoadBalancer:**
```yaml
service:
  type: LoadBalancer
  port: 8000
```

**NodePort:**
```yaml
service:
  type: NodePort
  port: 8000
```

### Ingress

Enable ingress for external access:

```yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "5m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
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

The annotations raise ingress-nginx's defaults of a 1 MiB request body and 60-second read and send timeouts. The server accepts request bodies up to 4 MiB, and a batch call that generates before it returns, such as one `mcp-context-ingest` sends to sync a corpus without `--defer-generation`, can run far longer than a minute, so keep the body limit above 4 MiB and the timeouts at least the client's `--call-timeout` (see [Limits](../knowledge-base.md#limits)). Other ingress controllers take the same limits under their own annotations.

## Health Checks and Probes

The chart configures three probes:

**Liveness Probe:**
- Path: `/health`
- Restarts container if unhealthy
- Recommended: `failureThreshold: 3`, `periodSeconds: 10`

**Readiness Probe:**
- Path: `/health`
- Removes from service if not ready
- Recommended: `failureThreshold: 1`, `periodSeconds: 5`

**Startup Probe:**
- Path: `/health`
- Allows slow startup (e.g., model loading, embedding initialization)
- Recommended: `failureThreshold: 30`, `periodSeconds: 10` (5 minutes max startup)

### Exit Codes and Restart Behavior

The MCP Context Server uses BSD sysexits.h exit codes:

| Exit Code | Meaning                | Kubernetes Behavior                |
|-----------|------------------------|------------------------------------|
| 0         | Normal shutdown        | Pod terminates successfully        |
| 1         | General error          | CrashLoopBackOff with backoff      |
| 69        | Dependency unavailable | CrashLoopBackOff (may recover)     |
| 78        | Configuration error    | CrashLoopBackOff (requires fix)    |

**Exit Code 69 (EX_UNAVAILABLE)** - transient dependency issues:
- Ollama service not yet running
- Embedding model not pulled
- Database temporarily unreachable

**Exit Code 78 (EX_CONFIG)** - configuration problems:
- Missing API keys (OPENAI_API_KEY, etc.)
- Missing required packages
- Invalid EMBEDDING_PROVIDER

### Recommended Probe Configuration

```yaml
spec:
  containers:
  - name: mcp-context-server
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
      failureThreshold: 1
    startupProbe:
      httpGet:
        path: /health
        port: 8000
      failureThreshold: 30    # 30 * 10s = 5 minutes max startup
      periodSeconds: 10
```

**Note:** The startup probe allows time for:
- Ollama to start (when using sidecar)
- Embedding model to download (~2-3 minutes for qwen3-embedding:0.6b)
- Database migrations to complete

## Monitoring

### Health Endpoint

```bash
kubectl port-forward svc/mcp-context-server 8000:8000
curl http://localhost:8000/health
```

### Logs

```bash
kubectl logs -l app.kubernetes.io/name=mcp-context-server -f
```

### Health Monitoring

Use the `/health` endpoint to monitor server status. The server does not expose a `/metrics` endpoint.

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=mcp-context-server

# Check pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

### Database Connection Issues

```bash
# Check ConfigMap
kubectl get configmap mcp-context-server -o yaml

# Check Secret
kubectl get secret mcp-context-server -o yaml

# Test connectivity from pod
kubectl exec -it <pod-name> -- nc -zv postgres-host 5432
```

### Persistent Volume Issues

```bash
# Check PVC status
kubectl get pvc

# Check PV
kubectl get pv

# Describe PVC for events
kubectl describe pvc mcp-context-server-data
```

## Additional Resources

### Related Documentation

- [Helm Deployment Guide](helm.md) - Detailed Helm configuration
- [Docker Deployment Guide](docker.md) - Alternative Docker Compose deployment
- [Summary Generation Guide](../summary-generation.md) - LLM-based summary configuration
- [Database Backends](../database-backends.md) - Database configuration
- [API Reference](../api-reference.md) - MCP tools documentation
