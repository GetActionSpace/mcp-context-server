"""
Server initialization and lifecycle management for mcp-context-server.

This package contains:
- Database initialization (init_database)
- Global state management (_backend, _repositories, _embedding_provider, _reranking_provider, _summary_provider,
  _generation_worker)
- Lazy initialization helpers (ensure_backend, ensure_repositories)
- Configuration constants (DB_PATH, MAX_IMAGE_SIZE_MB, MAX_TOTAL_SIZE_MB)
- The server lifespan's startup phases, in submodules this package never imports
  (tool_registration imports app.tools, and generation_worker the executor, which imports app.tools;
  app.tools imports this package): database_setup, rls_posture, providers, tool_registration,
  generation_worker

Global State Architecture:
    The _backend, _repositories, _embedding_provider, _reranking_provider, _summary_provider and
    _generation_worker variables are module-level singletons that are initialized once during
    server lifespan and accessed by MCP tool functions. Direct mutation is allowed from the
    server lifespan (app/server.py), its provider phase (app/startup/providers.py) and its
    generation worker phase (app/startup/generation_worker.py).

Usage:
    # In the server lifespan (app/server.py), on SQLite:
    from app.startup import DB_PATH, set_backend, set_repositories
    from app.startup.database_setup import prepare_database

    backend = create_backend(backend_type=None, db_path=DB_PATH)
    await backend.initialize()  # Connection pool (no schema)
    set_backend(backend)
    await prepare_database(backend, settings, request_role=None)  # Schema (init_database) and migrations
    set_repositories(RepositoryContainer(backend))

    # On PostgreSQL the lifespan prepares the schema on a short-lived maintenance backend
    # logged in as the table owner (create_maintenance_backend), passing the request role
    # when requests log in as another role, and only then creates and checks the request
    # backend it shares with every tool (app.startup.rls_posture).

    # In MCP tools (app/tools/*.py):
    from app.startup import ensure_repositories, get_reranking_provider
    repos = await ensure_repositories()
    reranking_provider = get_reranking_provider()
"""

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends import create_backend
from app.backends.base import MAINTENANCE
from app.embeddings import EmbeddingProvider
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.repositories import RepositoryContainer
from app.reranking import RerankingProvider
from app.services import ChunkingService
from app.settings import get_settings
from app.startup.compression_validator import validate_compression_provenance
from app.summary import SummaryProvider

if TYPE_CHECKING:
    from app.generation_jobs.worker import GenerationWorker

logger = logging.getLogger(__name__)
settings = get_settings()

# Configuration constants from settings
DB_PATH = settings.storage.db_path
MAX_IMAGE_SIZE_MB = settings.storage.max_image_size_mb
MAX_TOTAL_SIZE_MB = settings.storage.max_total_size_mb

# Global state - initialized during server lifespan
_backend: StorageBackend | None = None
_repositories: RepositoryContainer | None = None
_embedding_provider: EmbeddingProvider | None = None
_reranking_provider: RerankingProvider | None = None
_chunking_service: ChunkingService | None = None
_summary_provider: SummaryProvider | None = None
_generation_worker: 'GenerationWorker | None' = None


def set_backend(backend: StorageBackend | None) -> None:
    """Set the global backend instance.

    Called from the server lifespan (app/server.py) during startup/shutdown.
    """
    global _backend
    _backend = backend


def set_repositories(repos: RepositoryContainer | None) -> None:
    """Set the global repositories instance.

    Called from the server lifespan (app/server.py) during startup/shutdown.
    """
    global _repositories
    _repositories = repos


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    """Set the global embedding provider instance.

    Called from app.startup.providers during startup and from the server lifespan
    (app/server.py) during shutdown.
    """
    global _embedding_provider
    _embedding_provider = provider


def set_reranking_provider(provider: RerankingProvider | None) -> None:
    """Set the global reranking provider instance.

    Called from app.startup.providers during startup and from the server lifespan
    (app/server.py) during shutdown.
    """
    global _reranking_provider
    _reranking_provider = provider


def get_backend() -> StorageBackend | None:
    """Get the current backend instance (read-only access)."""
    return _backend


def get_repositories() -> RepositoryContainer | None:
    """Get the current repositories instance (read-only access)."""
    return _repositories


def get_embedding_provider() -> EmbeddingProvider | None:
    """Get the current embedding provider instance (read-only access)."""
    return _embedding_provider


def get_reranking_provider() -> RerankingProvider | None:
    """Get the current reranking provider instance (read-only access)."""
    return _reranking_provider


def set_chunking_service(service: ChunkingService | None) -> None:
    """Set the global chunking service instance.

    Called from app.startup.providers during startup and from the server lifespan
    (app/server.py) during shutdown.
    """
    global _chunking_service
    _chunking_service = service


def get_chunking_service() -> ChunkingService | None:
    """Get the current chunking service instance (read-only access)."""
    return _chunking_service


def set_summary_provider(provider: SummaryProvider | None) -> None:
    """Set the global summary provider instance.

    Called from app.startup.providers during startup and from the server lifespan
    (app/server.py) during shutdown.
    """
    global _summary_provider
    _summary_provider = provider


def get_summary_provider() -> SummaryProvider | None:
    """Get the current summary provider instance (read-only access)."""
    return _summary_provider


def set_generation_worker(worker: 'GenerationWorker | None') -> None:
    """Set the global generation worker instance.

    Called from app.startup.generation_worker when the worker starts and when it stops.
    """
    global _generation_worker
    _generation_worker = worker


def get_generation_worker() -> 'GenerationWorker | None':
    """Get the running generation worker, or None when none runs (read-only access)."""
    return _generation_worker


def notify_generation_worker() -> None:
    """Wake the generation worker after a write enqueued a job; without a running worker nothing happens."""
    if _generation_worker is not None:
        _generation_worker.notify()


async def init_database(backend: StorageBackend) -> None:
    """Initialize the base database schema through ``backend``.

    Args:
        backend: The initialized backend to create the schema through: on PostgreSQL the
            maintenance backend, logged in as the role that owns the tables.

    Raises:
        RuntimeError: If no schema file found or backend initialization fails.
    """
    try:
        # The database file is provisioned only for the file-based SQLite backend.
        backend_type = backend.backend_type

        # Ensure the database path exists, but ONLY for the file-based SQLite
        # backend. A PostgreSQL deployment still carries a DB_PATH default, so
        # gating on truthiness alone would create a stray SQLite file the backend
        # never opens -- and, worse, crash at boot under a read-only root
        # filesystem (EROFS) where that path is not writable, even though
        # PostgreSQL never touches it.
        if backend_type == 'sqlite' and DB_PATH:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not DB_PATH.exists():
                DB_PATH.touch()

        # Select schema file based on backend type
        schema_filename = 'postgresql_schema.sql' if backend_type == 'postgresql' else 'sqlite_schema.sql'

        schema_path = Path(__file__).parent.parent / 'schemas' / schema_filename

        # Read schema from file
        if schema_path.exists():
            schema_sql_template = schema_path.read_text(encoding='utf-8')
        else:
            raise RuntimeError(f'Schema file not found: {schema_path}')

        # Template the schema SQL with configured schema name (PostgreSQL only).
        # {SCHEMA} appears only in schema-qualified function DDL, so it MUST be
        # quoted the same way the connection search_path and the migrate-CLI
        # CREATE SCHEMA quote it -- an unquoted mixed-case or reserved-word
        # schema folds to lowercase at parse time and crashes boot with
        # 'schema "..." does not exist'.
        if backend_type == 'postgresql':
            from app.backends.postgresql_backend.session import quote_pg_identifier
            schema_sql = schema_sql_template.replace(
                '{SCHEMA}', quote_pg_identifier(settings.storage.postgresql_schema),
            )
        else:
            schema_sql = schema_sql_template

        # Apply the schema, the way each backend takes it
        if backend.backend_type == 'sqlite':

            def _init_schema_sqlite(conn: sqlite3.Connection) -> None:
                # Single executescript to create all objects atomically
                conn.executescript(schema_sql)

            await backend.execute_write(_init_schema_sqlite, access_scope=MAINTENANCE)
        else:  # postgresql
            migration_timeout_s = settings.storage.postgresql_migration_timeout_s

            async def _init_schema_postgresql(conn: asyncpg.Connection) -> None:
                # Raise the transaction-scoped statement_timeout to the migration budget
                # and take the shared schema-init advisory lock UNDER it, serializing DDL
                # across concurrent Kubernetes pods. A multi-pod peer running the first-time
                # schema build can hold the lock longer than the pool's regular command
                # timeout; acquiring it under the raised client- and server-side deadlines
                # waits the peer out instead of being cancelled client-side and crash-looping
                # the booting pod. The SET LOCAL and the transaction-scoped lock both
                # auto-revert/release on COMMIT/ROLLBACK (no explicit unlock needed).
                await begin_migration(conn, migration_timeout_s)

                # PostgreSQL: parse and execute statements individually
                statements: list[str] = []
                current_stmt: list[str] = []
                in_function = False

                for line in schema_sql.split('\n'):
                    stripped = line.strip()
                    # Skip comment-only lines
                    if stripped.startswith('--'):
                        continue
                    # Track dollar-quoted strings (function bodies)
                    if '$$' in stripped:
                        in_function = not in_function
                    if stripped:
                        current_stmt.append(line)
                    # End of statement: semicolon when not in dollar quotes
                    if stripped.endswith(';') and not in_function:
                        statements.append('\n'.join(current_stmt))
                        current_stmt = []

                # Add any remaining statement
                if current_stmt:
                    statements.append('\n'.join(current_stmt))

                # Execute each statement under the migration budget. Route through
                # execute_migration_ddl so asyncpg's client-side per-call deadline
                # matches the raised server-side statement_timeout; a bare
                # conn.execute would inherit the pool's shorter command_timeout (60s)
                # and be cancelled client-side despite the raised migration budget,
                # crash-looping the boot on any first-time schema statement that runs
                # longer than 60s (large instance, slow disk, multi-pod contention).
                for i, stmt in enumerate(statements):
                    stmt = stmt.strip()
                    if stmt and not stmt.startswith('--'):
                        try:
                            await execute_migration_ddl(conn, stmt, migration_timeout_s)
                            logger.debug('Schema statement %d/%d: SUCCESS', i + 1, len(statements))
                        except Exception as e:
                            logger.error('Schema statement %d/%d FAILED: %s', i + 1, len(statements), e)
                            logger.error('Statement: %s...', stmt[:200])
                            raise

            await backend.execute_write(cast(Any, _init_schema_postgresql), access_scope=MAINTENANCE)
        logger.info(f'Database schema initialized successfully ({backend.backend_type})')
    except Exception as e:
        logger.error(f'Failed to initialize database: {e}')
        raise


async def ensure_backend() -> StorageBackend:
    """Ensure a connection manager exists and is initialized.

    In tests, FastMCP lifespan isn't running, so tools need a lazy
    initializer to operate directly.

    Note:
        This initializes the connection pool only (no schema).
        For full database setup including schema, call init_database(backend)
        after ensure_backend().

    Returns:
        Initialized ``StorageBackend`` singleton to use for DB ops.
    """
    global _backend
    if _backend is None:
        manager = create_backend(backend_type=None, db_path=DB_PATH)
        await manager.initialize()
        _backend = manager
    return _backend


async def ensure_repositories() -> RepositoryContainer:
    """Ensure repositories are initialized.

    Returns:
        Initialized repository container.
    """
    global _repositories
    if _repositories is None:
        manager = await ensure_backend()
        _repositories = RepositoryContainer(manager)
    return _repositories


def propagate_langsmith_settings() -> None:
    """Propagate LangSmith settings to os.environ for SDK auto-detection.

    LangSmith SDK reads environment variables directly, not from Pydantic settings.
    This function bridges the gap when users configure via .env file or settings.

    Environment variables set (when tracing enabled):
    - LANGSMITH_TRACING: 'true'
    - LANGSMITH_API_KEY: API key (only if provided)
    - LANGSMITH_ENDPOINT: API endpoint URL
    - LANGSMITH_PROJECT: Project name for grouping traces

    Graceful degradation:
    - Does nothing if tracing disabled
    - Does nothing if langsmith package not installed (decorator handles this)
    """
    import os

    langsmith_settings = settings.langsmith
    if not langsmith_settings.tracing:
        return  # Tracing disabled, no propagation needed

    # Propagate ALL FOUR variables
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_ENDPOINT'] = langsmith_settings.endpoint
    os.environ['LANGSMITH_PROJECT'] = langsmith_settings.project

    if langsmith_settings.api_key:
        os.environ['LANGSMITH_API_KEY'] = langsmith_settings.api_key.get_secret_value()

    logger.info(
        f'LangSmith tracing enabled: project={langsmith_settings.project}, '
        f'endpoint={langsmith_settings.endpoint}',
    )


async def prewarm_ollama_models() -> None:
    """Pre-warm Ollama models by triggering model loading into memory.

    Sends lightweight requests to Ollama to force model loading from disk
    into RAM/VRAM. Combined with OLLAMA_KEEP_ALIVE=-1 on the Ollama server,
    models remain loaded indefinitely after warming.

    Checks which providers are configured as 'ollama' and sends appropriate
    warm-up requests:
    - Embedding models: POST /api/embed with minimal input
    - Summary models: POST /api/chat with empty messages and matching num_ctx

    Deduplicates when the same model is used for both embedding and summary.

    Failures are logged as warnings -- the server continues startup normally,
    and models will load on the first real request instead.
    """
    import httpx

    ollama_host = settings.ollama.host

    models_to_warm: list[tuple[str, str]] = []  # (model_name, warm_type)

    if settings.embedding.generation_enabled and settings.embedding.provider == 'ollama':
        models_to_warm.append((settings.embedding.model, 'embedding'))

    if settings.summary.generation_enabled and settings.summary.provider == 'ollama':
        models_to_warm.append((settings.summary.model, 'summary'))

    if not models_to_warm:
        return

    # Deduplicate if embedding and summary use the same model
    seen: set[str] = set()
    unique_models: list[tuple[str, str]] = []
    for model_name, warm_type in models_to_warm:
        if model_name not in seen:
            seen.add(model_name)
            unique_models.append((model_name, warm_type))

    async with httpx.AsyncClient(base_url=ollama_host, timeout=120.0) as client:
        for model_name, warm_type in unique_models:
            try:
                logger.info(f'Pre-warming {warm_type} model: {model_name}')
                if warm_type == 'embedding':
                    response = await client.post(
                        '/api/embed',
                        json={'model': model_name, 'input': 'warmup'},
                    )
                else:
                    response = await client.post(
                        '/api/chat',
                        json={
                            'model': model_name,
                            'messages': [],
                            'stream': False,
                            'options': {'num_ctx': settings.summary.ollama_num_ctx},
                        },
                    )
                response.raise_for_status()
                logger.info(f'Model {model_name} pre-warmed successfully')
            except Exception as e:
                logger.warning(
                    f'Failed to pre-warm model {model_name}: {e}. '
                    'Model will load on first request instead.',
                )


__all__ = [
    # Configuration constants
    'DB_PATH',
    'MAX_IMAGE_SIZE_MB',
    'MAX_TOTAL_SIZE_MB',
    # Global state (direct access - for lifespan read/write via setters)
    '_backend',
    '_repositories',
    '_embedding_provider',
    '_reranking_provider',
    '_chunking_service',
    '_summary_provider',
    '_generation_worker',
    # Setters (for lifespan initialization)
    'set_backend',
    'set_repositories',
    'set_embedding_provider',
    'set_reranking_provider',
    'set_chunking_service',
    'set_summary_provider',
    'set_generation_worker',
    # Getters (read-only access)
    'get_backend',
    'get_repositories',
    'get_embedding_provider',
    'get_reranking_provider',
    'get_chunking_service',
    'get_summary_provider',
    'get_generation_worker',
    'notify_generation_worker',
    # Initialization functions
    'init_database',
    'ensure_backend',
    'ensure_repositories',
    'propagate_langsmith_settings',
    'prewarm_ollama_models',
    # Re-exported types
    'RerankingProvider',
    'ChunkingService',
    'SummaryProvider',
    # Compression validator
    'validate_compression_provenance',
]
