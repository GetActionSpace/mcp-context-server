# Production Dockerfile for MCP Context Server
# Supports HTTP transport for remote client connections

# ============================================
# BUILDER STAGE
# ============================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Build argument for embedding provider (default: ollama for GHCR)
ARG EMBEDDING_EXTRA=embeddings-ollama
# Build argument for summary provider (default: ollama, shares langchain-ollama with embeddings)
ARG SUMMARY_EXTRA=summary-ollama

# uv build optimization settings
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Install dependencies first (better layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --extra ${EMBEDDING_EXTRA} --extra ${SUMMARY_EXTRA} --extra reranking --no-dev

# Copy application code
COPY app/ ./app/
COPY pyproject.toml uv.lock README.md ./

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --extra ${EMBEDDING_EXTRA} --extra ${SUMMARY_EXTRA} --extra reranking --no-dev

# ============================================
# RUNTIME STAGE
# ============================================
FROM python:3.12-slim-bookworm AS runtime

# Security: Create non-root user (UID 10001 avoids collisions with host and system UIDs)
RUN groupadd --gid 10001 appuser \
    && useradd --no-log-init --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Runtime environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTMCP_ENABLE_RICH_LOGGING=false

# Create data directory for SQLite
RUN mkdir -p /data && chown appuser:appuser /data

# Create cache directory for FlashRank reranking model
RUN mkdir -p /cache/flashrank && chown appuser:appuser /cache/flashrank

# Switch to non-root user
USER appuser

# OCI image metadata (static labels for manual builds; CI overrides via docker/metadata-action)
LABEL org.opencontainers.image.source="https://github.com/GetActionSpace/mcp-context-server" \
      org.opencontainers.image.documentation="https://github.com/GetActionSpace/mcp-context-server#readme" \
      org.opencontainers.image.licenses="Elastic-2.0" \
      org.opencontainers.image.vendor="ActionSpace" \
      org.opencontainers.image.title="mcp-context-server" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12-slim-bookworm"

# Expose HTTP port
EXPOSE 8000

# Health check
# Probe the actual configured port: FASTMCP_PORT drives the HTTP bind (default 8000),
# so deployments that override it (e.g. the test stacks on 8001/8002) stay reportable as healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('FASTMCP_PORT', '8000') + '/health').read()" || exit 1

# Copy entrypoint script that handles exit codes to prevent infinite restart loops
COPY --chown=appuser:appuser deploy/docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Run the server via entrypoint wrapper
ENTRYPOINT ["/docker-entrypoint.sh"]
