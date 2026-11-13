"""SQLite-to-SQLite migration runner."""

import logging
import sqlite3

from app.cli._database_url import parse_backend_url
from app.cli.migrate_uuid.id_mapping import build_id_mapping
from app.cli.migrate_uuid.id_mapping import created_at_for_id
from app.cli.migrate_uuid.metadata_ids import MetadataIdRewriter
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.sqlite_copy import copy_context_entries
from app.cli.migrate_uuid.sqlite_copy import copy_embedding_chunks
from app.cli.migrate_uuid.sqlite_copy import copy_embedding_metadata
from app.cli.migrate_uuid.sqlite_copy import copy_image_attachments
from app.cli.migrate_uuid.sqlite_copy import copy_tags
from app.cli.migrate_uuid.sqlite_copy import copy_vec_embeddings_sqlite
from app.cli.migrate_uuid.sqlite_source import detect_optional_tables
from app.cli.migrate_uuid.sqlite_source import detect_source_embedding_dim
from app.cli.migrate_uuid.sqlite_source import detect_source_id_kind
from app.cli.migrate_uuid.sqlite_source import open_source_sqlite
from app.cli.migrate_uuid.sqlite_target import initialize_target_sqlite
from app.cli.migrate_uuid.sqlite_target import load_sqlite_vec_extension
from app.cli.migrate_uuid.sqlite_target import open_target_sqlite
from app.cli.migrate_uuid.sqlite_target import rebuild_fts_sqlite
from app.cli.migrate_uuid.sqlite_target import target_already_has_data_sqlite
from app.cli.migrate_uuid.sqlite_target import target_sqlite_is_compressed

logger = logging.getLogger(__name__)


def run_migration_sqlite_to_sqlite(options: MigrationOptions) -> MigrationStats:
    """Drive a SQLite-to-SQLite migration.

    Opens the source read-only, inspects its schema, initializes the
    target schema, builds the ID mapping, and copies rows table by
    table.

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """
    stats = MigrationStats()
    stats.start_metadata_counts(options)

    _, source_address = parse_backend_url(options.source_url)
    _, target_address = parse_backend_url(options.target_url)

    if target_already_has_data_sqlite(target_address):
        stats.errors.append(
            f'target database already contains context_entries rows: {target_address}. '
            f'Recovery: if a prior run was interrupted, delete the target file and rerun; '
            f'the source database is unchanged. See the Recovering From an Interrupted Migration '
            f'section of docs/migration-v2-to-v3.md.',
        )
        return stats

    source = open_source_sqlite(source_address)
    target: sqlite3.Connection | None = None
    try:
        id_kind = detect_source_id_kind(source)
        if id_kind != 'integer':
            stats.warnings.append(
                f'source database id column is {id_kind!r}; nothing to migrate',
            )
            return stats

        optional_tables = detect_optional_tables(source)
        embedding_dim = detect_source_embedding_dim(source)

        # Defensive backstop (never silently drop embeddings), symmetric with the
        # PostgreSQL runner's pre-existing-target check. On PostgreSQL a compressed
        # target manifests as a MISSING vec_context_embeddings table, which that
        # runner detects directly. On SQLite the same condition is MASKED:
        # initialize_target_sqlite re-executes add_semantic_search_sqlite.sql, whose
        # CREATE VIRTUAL TABLE IF NOT EXISTS re-creates the fp32 vec0 table, so the
        # 'target lacks vec_context_embeddings' warning below can never fire and the
        # migrated fp32 vectors land in a database that already carries compression
        # provenance. The next server start then applies the compression migration,
        # whose leading DROP TABLE IF EXISTS vec_context_embeddings destroys every
        # migrated vector -- and --compress in between is a no-op, because it
        # early-returns on the existing provenance row. Probe the REAL target file
        # (the dry-run handle is an in-memory database that would see nothing) and
        # refuse BEFORE initialize_target_sqlite masks the condition.
        source_has_embeddings = bool(
            optional_tables.get('embedding_metadata') or optional_tables.get('vec_context_embeddings'),
        )
        if source_has_embeddings and target_sqlite_is_compressed(target_address):
            message = (
                'source has embeddings but the target database is already configured '
                'for compressed embeddings (it carries a compression_metadata '
                'provenance row and/or a vec_context_embeddings_compressed table). '
                'The fp32 vectors this migration copies would be destroyed the next '
                'time the server applies the compression migration, and --compress '
                'would not encode them (it is a no-op while a provenance row exists). '
                'Use an empty target file (this CLI initializes it), or run '
                'mcp-context-server-migrate --decompress against the target first to '
                'clear its compression provenance, then rerun the migration and '
                'finish with --compress.'
            )
            if options.dry_run:
                stats.warnings.append(f'{message} (a real run would abort)')
            else:
                stats.errors.append(f'{message} Aborting to avoid silently dropping embeddings.')
                return stats

        cursor = source.execute(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        source_rows = cursor.fetchall()

        if source_rows:
            first_created_at = created_at_for_id(source_rows[0]['created_at'])
            if first_created_at.microsecond == 0:
                logger.info(
                    'source created_at precision appears to be seconds; '
                    'sub-second ordering will use UUIDv7 random tails',
                )

        id_mapping = build_id_mapping(source_rows)
        rewriter = MetadataIdRewriter(options.id_paths, id_mapping, legacy_id_key=options.legacy_id_key)

        from app.repositories.fts_repository.query import desired_sqlite_fts_tokenizer
        from app.settings import get_settings

        target = open_target_sqlite(target_address, options.dry_run)
        initialize_target_sqlite(
            target,
            optional_tables,
            embedding_dim,
            # Derive the FTS tokenizer from FTS_LANGUAGE via the shared source of truth so a
            # CLI-migrated target matches what the server would build (a non-English language
            # gets plain unicode61, not the English Porter stemmer).
            fts_tokenizer=desired_sqlite_fts_tokenizer(get_settings().fts.language),
            stats=stats,
        )

        target.execute('BEGIN')
        try:
            copy_context_entries(source, target, id_mapping, rewriter, stats, options.dry_run)
            if optional_tables.get('tags'):
                copy_tags(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('image_attachments'):
                copy_image_attachments(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('embedding_metadata'):
                em_cursor = target.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_metadata'",
                )
                if em_cursor.fetchone() is not None:
                    copy_embedding_metadata(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('embedding_chunks'):
                ec_cursor = target.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_chunks'",
                )
                if ec_cursor.fetchone() is not None:
                    copy_embedding_chunks(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('vec_context_embeddings'):
                vec_cursor = target.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='vec_context_embeddings'",
                )
                if vec_cursor.fetchone() is not None:
                    if load_sqlite_vec_extension(source):
                        copy_vec_embeddings_sqlite(source, target, stats, options.dry_run)
                    else:
                        stats.warnings.append(
                            'sqlite-vec extension could not be loaded on source; vec rows not copied',
                        )
                else:
                    stats.warnings.append(
                        'target lacks vec_context_embeddings; vec rows not copied',
                    )
            if options.dry_run:
                target.rollback()
            else:
                target.commit()
        except Exception:
            target.rollback()
            raise

        if optional_tables.get('context_entries_fts'):
            rebuild_fts_sqlite(target, stats, options.dry_run)
            if not options.dry_run:
                target.commit()
    finally:
        source.close()
        if target is not None:
            target.close()
    return stats
