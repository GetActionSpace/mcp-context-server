"""Database preparation phase of the server lifespan.

Runs before any repository or tool uses the database: on PostgreSQL on the maintenance
backend, logged in as the table owner, and on SQLite on the backend the server shares.
It ensures the base schema, refuses a start whose configuration contradicts the data
already stored, applies every idempotent migration, validates the connection-pool
acquire timeout and the compression provenance, announces the active compression
configuration, grants the request role what it needs when requests log in as another
role, and finally brings the PostgreSQL row-level security objects up to date.
"""

import logging

from app.backends import StorageBackend
from app.compression.provenance import read_compression_metadata
from app.migrations import apply_access_control_migration
from app.migrations import apply_chunking_migration
from app.migrations import apply_compression_migration
from app.migrations import apply_content_hash_migration
from app.migrations import apply_fts_migration
from app.migrations import apply_function_search_path_migration
from app.migrations import apply_grant_owner_migration
from app.migrations import apply_index_tree_migration
from app.migrations import apply_ingest_index_migration
from app.migrations import apply_jsonb_merge_patch_migration
from app.migrations import apply_request_role_grants
from app.migrations import apply_rls
from app.migrations import apply_semantic_search_migration
from app.migrations import apply_summary_migration
from app.migrations import apply_tag_uniqueness_migration
from app.migrations import apply_version_migration
from app.migrations import guard_rls_disable_over_enabled
from app.migrations import handle_metadata_indexes
from app.settings import AppSettings
from app.startup import init_database
from app.startup.compression_validator import guard_compression_disable_over_populated
from app.startup.compression_validator import validate_compression_provenance
from app.startup.validation import validate_pool_acquire_timeout

logger = logging.getLogger(__name__)


async def prepare_database(backend: StorageBackend, settings: AppSettings, *, request_role: str | None) -> None:
    """Bring the database schema up to date and validate it before the server serves.

    Args:
        backend: The initialized backend to prepare through: on PostgreSQL the maintenance
            backend, logged in as the table owner; on SQLite the backend the server lifespan
            shares with every tool.
        settings: The application settings the server lifespan runs with.
        request_role: The PostgreSQL role requests log in as, as the database reports it,
            when it differs from the role this backend logs in as; None when requests use
            the same role, and always on SQLite.
    """
    # Ensure the base schema exists
    await init_database(backend=backend)
    # Handle metadata field indexing (configurable via METADATA_INDEXED_FIELDS)
    await handle_metadata_indexes(backend=backend)
    # Refuse a bare compression-off flip on a database that holds
    # compressed data BEFORE the provisioning migrations run. With
    # compression off the semantic/chunking migrations recreate the full
    # fp32 vector layout, so deferring this refusal to the post-migration
    # validate_compression_provenance call would leave a stray empty fp32
    # table + HNSW index that a later compression re-enable never drops.
    # Refusing up front means no stray schema is ever created on either
    # backend.
    await guard_compression_disable_over_populated(backend=backend)
    # Refuse a PostgreSQL start with ENABLE_POSTGRESQL_RLS off over the row-level
    # security objects a start with it on installed, before anything is provisioned.
    await guard_rls_disable_over_enabled(backend, settings)
    # Apply the semantic search migration (embedding storage)
    await apply_semantic_search_migration(backend=backend)
    # Apply jsonb_merge_patch migration for PostgreSQL (required for metadata_patch)
    await apply_jsonb_merge_patch_migration(backend=backend)
    # Apply function search_path security fix for PostgreSQL
    await apply_function_search_path_migration(backend=backend)
    # Apply FTS migration if enabled
    await apply_fts_migration(backend=backend)
    # Apply chunking migration (1:N embedding relationship)
    await apply_chunking_migration(backend=backend)
    # Apply summary column migration (always runs, column required for search display)
    await apply_summary_migration(backend=backend)
    # Apply context_entries column migrations (idempotent ADD COLUMN on
    # in-place upgrades; a no-op on fresh DBs whose base schema already has
    # them): content_hash (deduplication optimization) and version (the
    # optimistic-concurrency token used by the update_context /
    # update_context_batch compare-and-set and bumped by the dedup-store UPDATE).
    await apply_content_hash_migration(backend=backend)
    await apply_version_migration(backend=backend)
    # Access-control columns (owner_id/visibility), the context_entry_grants
    # table, and their lookup indexes. Must run before any write path stamps
    # the columns.
    await apply_access_control_migration(backend=backend)
    # The ingest index's SQLite key names the access columns, so it follows their migration.
    await apply_ingest_index_migration(backend=backend)
    # Each grant's entry owner, kept exact by a foreign key onto (id, owner_id), which names
    # the owner column the access-control migration adds; PostgreSQL only.
    await apply_grant_owner_migration(backend=backend)
    # Repair duplicate tag rows and install the unique index that prevents new
    # ones. The delete is the only thing that reaches entries already stored with
    # a repeated label, which every reader would otherwise keep returning twice.
    await apply_tag_uniqueness_migration(backend=backend)
    # Apply compression migration: REPLACES fp32 vec table with compressed
    # storage when ENABLE_EMBEDDING_COMPRESSION=true; no-op when disabled.
    # Must run before validate_compression_provenance because the validator
    # reads from the compression_metadata table created here.
    await apply_compression_migration(backend=backend)
    # Apply index_tree migration: provisions context_index_nodes when
    # per-node summaries are enabled (ENABLE_INDEX_TREE_NODE_SUMMARIES=true);
    # no-op otherwise. The on-demand outline itself never needs the table.
    await apply_index_tree_migration(backend=backend)
    # Validate the connection-pool acquire timeout (PostgreSQL only)
    if backend.backend_type == 'postgresql':
        validate_pool_acquire_timeout()
    # Validate compression provenance: bootstrap-or-validate the
    # singleton seed/bits/variant/provider/dim row. Raises
    # ConfigurationError(exit 78) on bootstrap-missing-seed or env/DB
    # mismatch; the supervisor does not auto-restart.
    await validate_compression_provenance(backend=backend)
    await _announce_compression(backend, settings)
    # The request role's grants and row-level security come last: every table and column
    # this start provisions exists by now, so the grants and the policies cover every
    # managed table and the read view lists every column. A database whose grants and
    # objects are current takes no lock and no DDL here.
    await apply_request_role_grants(backend, settings, request_role=request_role)
    await apply_rls(backend, settings)


async def _announce_compression(backend: StorageBackend, settings: AppSettings) -> None:
    """Log the active embedding compression configuration.

    Args:
        backend: The backend holding the compression_metadata provenance row.
        settings: The application settings the server lifespan runs with.
    """
    # Announce compression configuration at INFO level so
    # operators can verify the active configuration at a glance when
    # LOG_LEVEL=INFO. Mirrors the sibling feature announcements for
    # embedding generation, reranking, chunking, and summary in
    # app.startup.providers.
    # Values come from the singleton compression_metadata row
    # (DB-truth), not the raw env vars, because the validator may have
    # adopted an inherited row in a multi-pod race scenario.
    if settings.compression.enabled:
        # This read feeds a log line and nothing else: validate_compression_provenance
        # already read the same singleton row for validation and aborts on a genuine
        # provenance problem. A transient operational fault on a logging-only read (an external
        # writer holding the database lock past the read retry budget at the moment
        # the server boots) must therefore not abort startup and take every tool down
        # with it; announce the failure and continue.
        try:
            db_meta = await read_compression_metadata(backend)
        except Exception as compression_meta_error:
            logger.warning(
                f'Could not read the compression configuration to announce it: {compression_meta_error}',
            )
        else:
            if db_meta is not None:
                logger.info(
                    f'Embedding compression enabled with provider: {db_meta.provider} '
                    f'(bits={db_meta.bits}, variant={db_meta.variant}, '
                    f'dim={db_meta.dim}, seed={db_meta.seed}, '
                    f'max_concurrent={settings.compression.max_concurrent})',
                )
            elif not settings.embedding.generation_enabled:
                # Embedding storage is provisioned from
                # ENABLE_EMBEDDING_GENERATION; with generation off and nothing
                # previously compressed, no provenance row exists by design
                # (the validator skips seeding, and the migration provisions
                # the schema only when embedding infrastructure exists), so
                # the absent row is the expected idle state, not an error. A
                # populated fp32 store cannot reach this branch: the
                # enable-direction guard exits 78 in the migration first.
                logger.info(
                    'Embedding compression enabled but idle: embedding '
                    'generation is disabled and no compressed data exists',
                )
            else:
                # Defensive: validate_compression_provenance ran before this
                # announcement and would have raised ConfigurationError if the
                # singleton row were missing. Surface the inconsistency loudly rather
                # than silently.
                logger.info(
                    'Embedding compression enabled but provenance row missing '
                    '(validator should have raised; check startup order)',
                )
    else:
        logger.info('Embedding compression disabled (ENABLE_EMBEDDING_COMPRESSION=false)')
