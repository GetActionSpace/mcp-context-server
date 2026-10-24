# Helm Deployment Guide

This guide covers deploying the MCP Context Server using the official Helm chart.

## Prerequisites

- Kubernetes 1.21+
- Helm 3.8+
- PV provisioner support (for SQLite persistence)
- Optional: PostgreSQL database for production deployments

## Installation

### Install from Local Chart

```bash
# Clone the repository
git clone https://github.com/GetActionSpace/mcp-context-server.git
cd mcp-context-server

# Install with default values (SQLite)
helm install mcp ./deploy/helm/mcp-context-server

# Install with custom values
helm install mcp ./deploy/helm/mcp-context-server -f my-values.yaml
```

## Configuration Profiles

### SQLite (Development)

Best for single-user deployments and development:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-sqlite.yaml
```

### PostgreSQL (Production)

Best for multi-user and production deployments:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-postgresql.yaml \
  --set storage.postgresql.host=your-postgres-host \
  --set storage.postgresql.password=your-password
```

### With Semantic Search

Enable AI-powered semantic search with Ollama sidecar:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set search.semantic.enabled=true \
  --set ollama.enabled=true
```

### With Summary Generation

Enable automatic LLM-based summarization with Ollama sidecar:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set ollama.enabled=true \
  --set search.summary.enabled=true \
  --set search.summary.provider=ollama \
  --set search.summary.model=qwen3:0.6b
```

**Note:** When using the Ollama sidecar, models must be pulled manually before use:

```bash
# Pull the summary model
kubectl exec -it <pod-name> -c ollama -- ollama pull qwen3:0.6b
# Pull the embedding model (if semantic search is also enabled)
kubectl exec -it <pod-name> -c ollama -- ollama pull qwen3-embedding:0.6b
```

### Full-Featured Production

All features enabled with PostgreSQL:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set storage.backend=postgresql \
  --set storage.postgresql.enabled=true \
  --set storage.postgresql.host=postgres.example.com \
  --set storage.postgresql.password=secure-password \
  --set search.fts.enabled=true \
  --set search.semantic.enabled=true \
  --set search.hybrid.enabled=true \
  --set search.summary.enabled=true \
  --set ollama.enabled=true \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=mcp.example.com
```

## Values Reference

### Image Configuration

```yaml
image:
  repository: ghcr.io/getactionspace/mcp-context-server
  tag: ""  # Defaults to Chart.appVersion
  pullPolicy: IfNotPresent

imagePullSecrets: []
```

### Service Configuration

```yaml
service:
  type: ClusterIP  # or LoadBalancer, NodePort
  port: 8000
```

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
      storageClassName: ""  # Use default
      accessModes:
        - ReadWriteOnce
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
    password: ""  # Set via --set or secret
    database: "mcp_context"
    sslMode: "prefer"
    # Use existing secret instead of password
    existingSecret: ""
    existingSecretKey: "postgresql-password"
    # Role that owns the tables and runs migrations (POSTGRESQL_MIGRATION_USER and
    # POSTGRESQL_MIGRATION_PASSWORD); empty keeps the request credentials above
    migration:
      user: ""
      password: ""
      existingSecret: ""
      existingSecretKey: "postgresql-migration-password"
    # Row-level security (ENABLE_POSTGRESQL_RLS, POSTGRESQL_RLS_POLICY_ROLE)
    rls:
      enabled: false
      policyRole: ""  # empty keeps the server default, mcp_context_request
```

### Row-Level Security

With `storage.postgresql.rls.enabled: true`, PostgreSQL enforces the access model through row-level security in addition to the server (see [Row-level security (PostgreSQL)](../database-backends.md#row-level-security-postgresql)). Requests then connect as `storage.postgresql.user`, a login role that is a member of the policy role and is neither superuser nor `BYPASSRLS`, while `storage.postgresql.migration.user` owns the tables and runs the migrations inside the server pod at startup; the chart adds no hook Job. Rendering fails when `rls.enabled` is set with `storage.backend` other than `postgresql`, or without `migration.user`, mirroring the checks the server applies at startup. Each migration value renders only when it is set.

**Enabling:**

1. As a role administrator, create the NOLOGIN policy role and a login role that is its member:

   ```sql
   CREATE ROLE mcp_context_request NOLOGIN;
   CREATE ROLE mcp_app LOGIN PASSWORD 'choose-a-password' NOSUPERUSER NOBYPASSRLS;
   GRANT mcp_context_request TO mcp_app;
   ```

2. Point the request credentials at the login role, keep the current user (which owns the tables) as the migration user, and turn the flag on:

   ```yaml
   storage:
     backend: postgresql
     postgresql:
       enabled: true
       user: "mcp_app"
       existingSecret: "mcp-postgresql"   # key postgresql-password: the login role's password
       migration:
         user: "postgres"
         existingSecret: "mcp-postgresql" # key postgresql-migration-password: the table owner's password
       rls:
         enabled: true
   ```

3. Apply the values with `helm upgrade`. The first pod installs the policies and the read view; pods still running the old configuration keep working during the rollout when they connect as the table owner. Pods that already connect as the login role with the flag off, for example after the disabling steps below, read only `public` entries and can no longer store, update or delete any entry once the first pod with the flag on has started, so in that case scale the Deployment to zero before `helm upgrade` (`kubectl scale deployment -l app.kubernetes.io/instance=mcp --replicas=0`), as the disabling steps do, and scale it back to its replica count afterwards.

**Rotating the login role:** create the new login role and grant it the policy role (`CREATE ROLE mcp_app_2 LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS; GRANT mcp_context_request TO mcp_app_2;`), change `storage.postgresql.user` and its password, and `helm upgrade`. The policies name the policy role, so no DDL runs.

**Disabling:** the server refuses to start with the flag off while the policies are installed, and a pod started with the flag on installs them again, so the steps run in this order:

1. Set `storage.postgresql.rls.enabled: false` in your values file, without applying it yet.
2. Stop every pod. The Deployment uses the default rolling update, which starts new pods before old ones stop, so scale it to zero: `kubectl scale deployment -l app.kubernetes.io/instance=mcp --replicas=0`.
3. Remove the policies and the read view from a one-off pod, with the table owner's credentials (add `--env POSTGRESQL_SCHEMA=<schema>` when the tables live outside `public`):

   ```bash
   kubectl run mcp-disable-rls --rm -it --restart=Never \
     --image=ghcr.io/getactionspace/mcp-context-server:<tag> \
     --command -- mcp-context-server-migrate --disable-rls \
     --source-url 'postgresql://postgres:<password>@<host>:5432/mcp_context'
   ```

4. Apply the values with `helm upgrade`, then scale the Deployment back to its replica count.

The removal refuses while any non-superuser session of a member of the policy role is connected, which is what step 2 prevents; see [Disabling row-level security](../database-backends.md#disabling-row-level-security).

### Authentication and Access Control

The chart renders each of these values into the server's environment only when it is set, so the server defaults apply otherwise. See the [Authentication Guide](../authentication.md) for what each setting does.

```yaml
auth:
  provider: ""          # MCP_AUTH_PROVIDER: none (server default), simple_token or jwt
  clientId: ""          # MCP_AUTH_CLIENT_ID
  token: ""             # MCP_AUTH_TOKEN, stored in the chart secret
  existingSecret: ""    # holds the token (simple_token), the public key (jwt without jwksUri) or the anonymous key (jwt with jwksUri)
  tokenSecretKey: "mcp-auth-token"
  jwt:
    publicKey: ""       # MCP_AUTH_JWT_PUBLIC_KEY: PEM key or HS* shared secret; set this or jwksUri
    publicKeySecretKey: "mcp-auth-jwt-public-key"
    jwksUri: ""         # MCP_AUTH_JWT_JWKS_URI
    issuer: ""          # MCP_AUTH_JWT_ISSUER
    audience: ""        # MCP_AUTH_JWT_AUDIENCE
    algorithm: ""       # MCP_AUTH_JWT_ALGORITHM
    anonymousPublicKey: ""  # MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY: the anonymous verifier's PEM key or HS* secret
    anonymousPublicKeySecretKey: "mcp-auth-anonymous-jwt-public-key"
    anonymousAlgorithm: ""  # MCP_AUTH_ANONYMOUS_JWT_ALGORITHM
  groupsClaim: ""       # MCP_AUTH_GROUPS_CLAIM
  rolesClaim: ""        # MCP_AUTH_ROLES_CLAIM

accessControl:
  defaultPrincipal: ""    # ACCESS_CONTROL_DEFAULT_PRINCIPAL
  defaultVisibility: ""   # ACCESS_CONTROL_DEFAULT_VISIBILITY: private or public
  defaultGroupGrants: ""  # ACCESS_CONTROL_DEFAULT_GROUP_GRANTS: none or author_groups
  publishRole: ""         # ACCESS_CONTROL_PUBLISH_ROLE
  anonymousPrincipal: ""  # ACCESS_CONTROL_ANONYMOUS_PRINCIPAL: the jwt subject every write tool refuses
```

`accessControl.anonymousPrincipal` makes one jwt subject read-only. The server matches it verbatim against the token's `sub`, so a value that differs from the subject you minted leaves that token writable. The server logs `Read-only principal: <id>` at INFO level on startup, and a `store_context` call with the token must answer `The anonymous principal is read-only: store_context is not permitted` before you hand the token out.

On a JWKS deployment your identity provider signs every token the server trusts, so you cannot mint the anonymous token with its key. Set `auth.jwt.anonymousPublicKey` to the public key (or the HS* shared secret) of a key pair you hold, and `auth.jwt.anonymousAlgorithm` when it is not `RS256`: the server then accepts tokens signed with that key for `accessControl.anonymousPrincipal` and for no other subject, under the same `auth.jwt.issuer` and `auth.jwt.audience`. The chart renders the key only with `auth.provider: jwt` and `auth.jwt.jwksUri` set. With `auth.existingSecret`, the key comes from that secret under `auth.jwt.anonymousPublicKeySecretKey` as an optional reference, so a secret without the key starts the server without the anonymous verifier. On a static-key deployment, sign the anonymous token with the private key behind `auth.jwt.publicKey` instead; the chart ignores `auth.jwt.anonymousPublicKey` there. [Anonymous Read Access](../authentication.md#anonymous-read-access) shows how to mint the token and check it.

### Search Configuration

```yaml
search:
  fts:
    enabled: true
    language: "english"
  semantic:
    enabled: false
    model: "qwen3-embedding:0.6b"
    dim: 1024
  hybrid:
    enabled: true
    rrfK: 60
```

### Summary Generation

```yaml
search:
  summary:
    enabled: false            # ENABLE_SUMMARY_GENERATION (opt-in; needs a provider)
    provider: "ollama"        # SUMMARY_PROVIDER: ollama, openai, or anthropic
    model: "qwen3:0.6b"      # SUMMARY_MODEL
    maxTokens: 4000           # SUMMARY_MAX_TOKENS (tokens, 50-16384)
    minContentLength: 500     # SUMMARY_MIN_CONTENT_LENGTH (chars, 0=always)
    prompt: ""                # SUMMARY_PROMPT: empty uses built-in default
    timeout: 240.0            # SUMMARY_TIMEOUT_S
    retryMaxAttempts: 5       # SUMMARY_RETRY_MAX_ATTEMPTS
    retryBaseDelay: 3.0       # SUMMARY_RETRY_BASE_DELAY_S
    maxConcurrent: 2          # SUMMARY_MAX_CONCURRENT (1-20)
```

For OpenAI or Anthropic providers, configure API keys:

```yaml
summarySecrets:
  openaiApiKey: ""        # For summary.provider=openai
  anthropicApiKey: ""     # For summary.provider=anthropic
  existingSecret: ""      # Use pre-existing Kubernetes secret
```

### Ollama Sidecar

```yaml
ollama:
  enabled: false
  image:
    repository: ollama/ollama
    tag: "latest"
    pullPolicy: IfNotPresent
  resources:
    requests:
      cpu: "500m"
      memory: "2Gi"
    limits:
      cpu: "2000m"
      memory: "4Gi"
  persistence:
    enabled: true
    size: 5Gi
    storageClassName: ""
```

### GPU Acceleration

The Ollama sidecar supports GPU acceleration for faster inference. Configure GPU resources in `values.yaml`:

**NVIDIA GPU:**

```yaml
ollama:
  enabled: true
  resources:
    limits:
      nvidia.com/gpu: "1"
```

Requires the [NVIDIA device plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin).

**AMD GPU:**

```yaml
ollama:
  enabled: true
  image:
    tag: "rocm"
  resources:
    limits:
      amd.com/gpu: "1"
```

Requires the [AMD GPU device plugin for Kubernetes](https://github.com/ROCm/k8s-device-plugin).

For complete GPU setup instructions, troubleshooting, and Intel/Vulkan considerations, see the [GPU Acceleration Guide](gpu-acceleration.md).

### Ingress Configuration

```yaml
ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: mcp-context-server.local
      paths:
        - path: /
          pathType: Prefix
  tls: []
```

`annotations` passes through to the Ingress unchanged and is empty by default, because annotations are specific to the ingress controller. With ingress-nginx, set the body-size and timeout annotations shown in [External PostgreSQL with Existing Secret](#external-postgresql-with-existing-secret); its defaults, a 1 MiB request body and 60-second timeouts, refuse or cut off the larger and longer requests the server accepts.

### Resource Limits

```yaml
resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```

### Health Probes

```yaml
probes:
  liveness:
    enabled: true
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 30
    failureThreshold: 3
  readiness:
    enabled: true
    path: /health
    initialDelaySeconds: 10
    periodSeconds: 10
    failureThreshold: 3
  startup:
    enabled: true
    path: /health
    initialDelaySeconds: 5
    periodSeconds: 10
    failureThreshold: 30
```

### Security Context

```yaml
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  fsGroup: 10001

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: false
  capabilities:
    drop:
      - ALL
```

### Service Account

```yaml
serviceAccount:
  create: true
  name: ""
  annotations: {}
```

## Common Operations

### Upgrade

```bash
helm upgrade mcp ./deploy/helm/mcp-context-server
```

### Rollback

```bash
helm rollback mcp 1
```

### Uninstall

```bash
helm uninstall mcp
```

**Note:** PersistentVolumeClaims are not deleted automatically. To remove data:

```bash
kubectl delete pvc -l app.kubernetes.io/instance=mcp
```

### View Rendered Templates

```bash
helm template mcp ./deploy/helm/mcp-context-server
```

### Debug Installation

```bash
helm install mcp ./deploy/helm/mcp-context-server --debug --dry-run
```

## Examples

### External PostgreSQL with Existing Secret

```yaml
# values-production.yaml
storage:
  backend: postgresql
  sqlite:
    enabled: false
  postgresql:
    enabled: true
    host: "postgres.production.svc.cluster.local"
    port: "5432"
    user: "mcp_user"
    database: "mcp_context"
    sslMode: "require"
    existingSecret: "postgres-credentials"
    existingSecretKey: "password"

search:
  fts:
    enabled: true
  semantic:
    enabled: true
  hybrid:
    enabled: true

ollama:
  enabled: true
  resources:
    requests:
      memory: "4Gi"
    limits:
      memory: "8Gi"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
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

The three `nginx.ingress.kubernetes.io` annotations raise ingress-nginx's defaults of a 1 MiB request body and 60-second timeouts: the server accepts request bodies up to 4 MiB, and a batch call with generation on, such as one `mcp-context-ingest` sends, can run far longer than a minute, so the timeouts should be at least the client's `--call-timeout` (see [Limits](../knowledge-base.md#limits)).

Install with:

```bash
helm install mcp ./deploy/helm/mcp-context-server -f values-production.yaml
```

### Development with Port Forward

```bash
# Install minimal setup
helm install mcp ./deploy/helm/mcp-context-server

# Port forward to local machine
kubectl port-forward svc/mcp 8000:8000

# Test connection
curl http://localhost:8000/health
```

## Troubleshooting

### Pod Stuck in Pending

Check PersistentVolumeClaim:
```bash
kubectl get pvc
kubectl describe pvc mcp-data
```

### Ollama Out of Memory

Increase resources:
```yaml
ollama:
  resources:
    limits:
      memory: "8Gi"
```

### Health Check Failing

Check logs:
```bash
kubectl logs -l app.kubernetes.io/name=mcp-context-server
```

## Additional Resources

### Related Documentation

- [GPU Acceleration Guide](gpu-acceleration.md) - NVIDIA, AMD, and Intel GPU setup for Docker and Kubernetes
- [Kubernetes Deployment Guide](kubernetes.md) - General Kubernetes deployment
- [Docker Deployment Guide](docker.md) - Alternative Docker Compose deployment
- [Summary Generation Guide](../summary-generation.md) - LLM-based summary configuration
- [Database Backends](../database-backends.md) - Database configuration details, including row-level security
- [Authentication Guide](../authentication.md) - HTTP authentication and the access model
- [Semantic Search](../semantic-search.md) - Ollama and embedding configuration
