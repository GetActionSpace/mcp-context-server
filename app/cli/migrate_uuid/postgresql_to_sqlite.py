"""PostgreSQL-to-SQLite migration runner."""

import logging
import sqlite3

from app.cli._database_url import parse_backend_url
from app.cli.migrate_uuid.id_mapping import NULL_CREATED_AT_ANCHOR
from app.cli.migrate_uuid.id_mapping import created_at_for_id
from app.cli.migrate_uuid.id_mapping import sqlite_timestamp
from app.cli.migrate_uuid.metadata_ids import MetadataIdRewriter
from app.cli.migrate_uuid.pg_connection import pg_column_exists
from app.cli.migrate_uuid.pg_connection import pg_connect
from app.cli.migrate_uuid.pg_connection import pg_table_exists
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.sqlite_target import initialize_target_sqlite
from app.cli.migrate_uuid.sqlite_target import open_target_sqlite
from app.cli.migrate_uuid.sqlite_target import rebuild_fts_sqlite
from app.cli.migrate_uuid.sqlite_target import target_already_has_data_sqlite
from app.cli.migrate_uuid.target_rows import SELECT_DISTINCT_TAGS_SQL
from app.cli.migrate_uuid.target_rows import access_backfill_values
from app.ids import generate_id_with_timestamp

logger = logging.getLogger(__name__)


async def run_migration_mixed_postgresql_to_sqlite(options: MigrationOptions) -> MigrationStats:
    """Migrate from a PostgreSQL source to a SQLite target.

    Mirrors :func:`run_migration_mixed_sqlite_to_postgresql` with the backends
    swapped. Vector embeddings are not transferred (re-embed afterward), but
    context_entries, tags, and image attachments are copied, and the SQLite
    target's FTS5 index is rebuilt from the copied rows so full-text search works
    even though FTS is not portable from PostgreSQL.

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """

    stats = MigrationStats()
    stats.start_metadata_counts(options)
    stats.warnings.append(
        'cross-backend migration drops vector embeddings; re-embed the target after migration',
    )

    _, target_address = parse_backend_url(options.target_url)
    if target_already_has_data_sqlite(target_address):
        stats.errors.append(
            f'target database already contains context_entries rows: {target_address}. '
            f'Recovery: if a prior run was interrupted, delete the target file and rerun; '
            f'the source database is unchanged. See the Recovering From an Interrupted Migration '
            f'section of docs/migration-v2-to-v3.md.',
        )
        return stats

    source_conn = await pg_connect(options.source_url)
    target: sqlite3.Connection | None = None
    try:
        await source_conn.execute('BEGIN TRANSACTION READ ONLY')

        id_column_type = await source_conn.fetchval(
            'SELECT data_type FROM information_schema.columns '
            "WHERE table_schema = current_schema() AND table_name = 'context_entries' AND column_name = 'id'",
        )
        if id_column_type is None:
            stats.errors.append("source PostgreSQL database lacks 'context_entries.id' column")
            return stats
        if str(id_column_type).lower() in ('uuid', 'text', 'character varying'):
            stats.warnings.append(
                f'source PostgreSQL id column is {id_column_type!r}; nothing to migrate',
            )
            return stats

        source_rows = await source_conn.fetch(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        id_mapping: dict[int, str] = {}
        null_created_at = 0
        for row in source_rows:
            if row['created_at'] is None:
                null_created_at += 1
            id_mapping[int(row['id'])] = generate_id_with_timestamp(created_at_for_id(row['created_at']))
        if null_created_at:
            logger.warning(
                '%d source context_entries row(s) had NULL created_at; their ids '
                'were anchored to %s (the stored created_at is preserved as NULL)',
                null_created_at,
                NULL_CREATED_AT_ANCHOR.isoformat(),
            )
        rewriter = MetadataIdRewriter(options.id_paths, id_mapping, legacy_id_key=options.legacy_id_key)

        # Detect which optional tables the PostgreSQL source carries so the
        # SQLite target is shaped to match. FTS is offered on the target (it is
        # not portable from PostgreSQL, but the SQLite target supports it and
        # the index is rebuilt locally from the copied rows below).
        source_has_tags = await pg_table_exists(source_conn, 'tags')
        source_has_images = await pg_table_exists(source_conn, 'image_attachments')
        from app.repositories.fts_repository.query import desired_sqlite_fts_tokenizer
        from app.settings import get_settings

        target = open_target_sqlite(target_address, options.dry_run)
        initialize_target_sqlite(
            target,
            optional_tables={
                'tags': True,
                'image_attachments': True,
                'context_entries_fts': True,
            },
            embedding_dim=None,
            # Derive the FTS tokenizer from FTS_LANGUAGE via the shared source of truth so the
            # PostgreSQL->SQLite target's rebuilt FTS index matches what the server would build
            # for the configured language, instead of a hardcoded English tokenizer.
            fts_tokenizer=desired_sqlite_fts_tokenizer(get_settings().fts.language),
            stats=stats,
        )

        # Guard summary / content_hash on the PostgreSQL source (a v2 DB predating
        # those ALTER-TABLE columns lacks them), mirroring copy_context_entries so
        # all four migration directions tolerate their absence identically. NULL
        # substitution keeps the row keys and the INSERT below unchanged.
        summary_col_src = (
            'summary' if await pg_column_exists(source_conn, 'context_entries', 'summary')
            else 'NULL AS summary'
        )
        content_hash_col_src = (
            'content_hash' if await pg_column_exists(source_conn, 'context_entries', 'content_hash')
            else 'NULL AS content_hash'
        )
        entry_rows = await source_conn.fetch(
            f'SELECT id, thread_id, source, content_type, text_content, '
            f'metadata::text AS metadata, {summary_col_src}, {content_hash_col_src}, created_at, updated_at '
            f'FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        target.execute('BEGIN')
        try:
            for row in entry_rows:
                source_id = int(row['id'])
                new_id = id_mapping[source_id]
                outcome = rewriter.rewrite(row['metadata'], source_id)
                stats.record_metadata(outcome, migrated=True)
                if not options.dry_run:
                    target.execute(
                        'INSERT INTO context_entries '
                        '(id, thread_id, source, content_type, text_content, metadata, summary, '
                        'content_hash, owner_id, visibility, created_at, updated_at) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (
                            new_id,
                            row['thread_id'],
                            row['source'],
                            row['content_type'],
                            row['text_content'],
                            outcome.metadata,
                            row['summary'],
                            row['content_hash'],
                            *access_backfill_values(),
                            sqlite_timestamp(row['created_at']),
                            sqlite_timestamp(row['updated_at']),
                        ),
                    )
                stats.rows_migrated += 1

            # Copy tags and image attachments from the PostgreSQL source into
            # the SQLite target (portable: tags are TEXT, image payloads are
            # BYTEA->BLOB; image_metadata is cast to text for the SQLite TEXT
            # column; timestamps are rendered ISO-8601). Only embeddings are
            # dropped cross-backend. Reads guarded by source table presence.
            if source_has_tags:
                tag_rows = await source_conn.fetch(
                    SELECT_DISTINCT_TAGS_SQL,
                )
                for tag_row in tag_rows:
                    sid = int(tag_row['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'tags row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    if not options.dry_run:
                        target.execute(
                            'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                            (mapped, tag_row['tag']),
                        )
                    stats.tags_migrated += 1

            if source_has_images:
                image_rows = await source_conn.fetch(
                    'SELECT context_entry_id, image_data, mime_type, '
                    'image_metadata::text AS image_metadata, position, created_at '
                    'FROM image_attachments ORDER BY id ASC',
                )
                for img in image_rows:
                    sid = int(img['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'image_attachments row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    # Preserve a schema-legal NULL created_at as NULL, and render a
                    # present timestamp in SQLite's canonical space form (NOT isoformat's
                    # 'T'/offset, which mis-sorts under SQLite TEXT date comparison).
                    img_created_at = sqlite_timestamp(img['created_at'])
                    if not options.dry_run:
                        target.execute(
                            'INSERT INTO image_attachments '
                            '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
                            'VALUES (?, ?, ?, ?, ?, ?)',
                            (
                                mapped,
                                img['image_data'],
                                img['mime_type'],
                                img['image_metadata'],
                                img['position'],
                                img_created_at,
                            ),
                        )
                    stats.images_migrated += 1

            if options.dry_run:
                target.rollback()
            else:
                target.commit()
        except Exception:
            target.rollback()
            raise

        # Rebuild the SQLite FTS5 index from the copied rows, outside the data
        # transaction (mirrors the SQLite->SQLite path), so the SQLite target
        # has working full-text search even though FTS is not portable from
        # PostgreSQL.
        rebuild_fts_sqlite(target, stats, options.dry_run)
        if not options.dry_run:
            target.commit()
    finally:
        await source_conn.close()
        if target is not None:
            target.close()
    return stats
