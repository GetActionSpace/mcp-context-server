"""
Database migration functions for mcp-context-server.

This package contains migration orchestration and all migration implementations:
- dependencies.py: Provider and vector storage dependency checking
- semantic.py: Semantic search migrations (vector tables, jsonb_merge_patch)
- fts.py: Full-text search migrations
- metadata.py: Metadata field index management
- chunking.py: 1:N embedding relationship migration
- summary.py: Summary column migration
- content_hash.py: Content hash column migration for deduplication optimization
- version.py: Optimistic-concurrency version column migration
- access_control.py: Access-control columns, the grants table and the access lookup indexes
- ingest.py: The partial index serving the reads of ingest-managed entries
- grant_owner.py: Each PostgreSQL grant's entry owner column, kept exact by a composite foreign key onto the entry
- tag_uniqueness.py: Duplicate tag repair and the unique tag index
- index_tree.py: The per-node summary table of the index tree
- compression.py: Embedding compression migration (vec_context_embeddings_compressed + compression_metadata)
- rls_ddl.py: Rendering of the PostgreSQL row-level security helpers, policies, guard trigger and read view
- rls_state.py: Reading the installed PostgreSQL row-level security objects and planning what differs
- rls.py: Keeping the PostgreSQL row-level security objects and the read view installed and current
- rls_removal.py: Removing the PostgreSQL row-level security objects, and refusing a start with them switched off
- request_role_grants.py: The privileges the request role needs, granted to the policy role

SQL Files (resources):
- add_semantic_search_*.sql: Vector table schemas
- add_fts_*.sql: FTS table schemas
- add_chunking_*.sql: 1:N embedding schema modifications
- add_compression_*.sql: Compressed-payload table schemas
- add_jsonb_merge_patch_postgresql.sql: PostgreSQL merge function
- fix_function_search_path_postgresql.sql: Security fix

PostgreSQL DDL convention: all TABLE and INDEX DDL in
``add_*_postgresql.sql`` uses BARE table names. Operators with a
non-default ``POSTGRESQL_SCHEMA`` value must configure ``search_path``
on every connection so the migration creates tables in the intended
schema. FUNCTION DDL (and ``ALTER FUNCTION`` targets, plus trigger
``EXECUTE FUNCTION`` references) remains schema-qualified for
CVE-2018-1058 mitigation; this is a distinct concern from table-name
resolution. Idempotency-check filters against ``information_schema``,
``pg_indexes``, and ``pg_namespace`` use ``current_schema()`` so the
check inspects whatever schema ``search_path`` resolves to. See
``docs/embedding-compression.md`` for the full operator contract.
"""

from app.migrations.access_control import apply_access_control_migration
from app.migrations.chunking import apply_chunking_migration
from app.migrations.compression import apply_compression_migration
from app.migrations.content_hash import apply_content_hash_migration
from app.migrations.dependencies import ProviderCheckResult
from app.migrations.dependencies import check_provider_dependencies
from app.migrations.dependencies import check_summary_provider_dependencies
from app.migrations.dependencies import check_vector_storage_dependencies
from app.migrations.fts import FtsMigrationStatus
from app.migrations.fts import apply_fts_migration
from app.migrations.fts import estimate_migration_time
from app.migrations.fts import get_fts_migration_status
from app.migrations.fts import reset_fts_migration_status
from app.migrations.grant_owner import apply_grant_owner_migration
from app.migrations.index_tree import apply_index_tree_migration
from app.migrations.ingest import apply_ingest_index_migration
from app.migrations.metadata import handle_metadata_indexes
from app.migrations.request_role_grants import GrantReport
from app.migrations.request_role_grants import apply_request_role_grants
from app.migrations.rls import apply_rls
from app.migrations.rls import create_read_view
from app.migrations.rls_removal import RlsRemovalReport
from app.migrations.rls_removal import guard_rls_disable_over_enabled
from app.migrations.rls_removal import remove_rls
from app.migrations.semantic import apply_function_search_path_migration
from app.migrations.semantic import apply_jsonb_merge_patch_migration
from app.migrations.semantic import apply_semantic_search_migration
from app.migrations.summary import apply_summary_migration
from app.migrations.tag_uniqueness import apply_tag_uniqueness_migration
from app.migrations.version import apply_version_migration

__all__ = [
    # Dependencies
    'ProviderCheckResult',
    'check_vector_storage_dependencies',
    'check_provider_dependencies',
    'check_summary_provider_dependencies',
    # Semantic
    'apply_semantic_search_migration',
    'apply_jsonb_merge_patch_migration',
    'apply_function_search_path_migration',
    # FTS
    'FtsMigrationStatus',
    'apply_fts_migration',
    'estimate_migration_time',
    'get_fts_migration_status',
    'reset_fts_migration_status',
    # Metadata
    'handle_metadata_indexes',
    # Chunking
    'apply_chunking_migration',
    # Index tree
    'apply_index_tree_migration',
    # Summary
    'apply_summary_migration',
    'check_summary_provider_dependencies',
    # Content hash
    'apply_content_hash_migration',
    # Version (optimistic concurrency)
    'apply_version_migration',
    # Access control
    'apply_access_control_migration',
    # Ingest index
    'apply_ingest_index_migration',
    # Grant owner
    'apply_grant_owner_migration',
    # Tag uniqueness
    'apply_tag_uniqueness_migration',
    # Compression
    'apply_compression_migration',
    # Row-level security
    'apply_rls',
    'create_read_view',
    'RlsRemovalReport',
    'guard_rls_disable_over_enabled',
    'remove_rls',
    'GrantReport',
    'apply_request_role_grants',
]
