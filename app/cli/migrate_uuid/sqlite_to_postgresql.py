"""SQLite-to-PostgreSQL migration runner."""

from app.cli._database_url import parse_backend_url
from app.cli.migrate_uuid.id_mapping import build_id_mapping
from app.cli.migrate_uuid.id_mapping import stored_datetime_or_none
from app.cli.migrate_uuid.metadata_ids import MetadataIdRewriter
from app.cli.migrate_uuid.pg_connection import pg_connect
from app.cli.migrate_uuid.pg_connection import pg_table_exists
from app.cli.migrate_uuid.pg_connection import target_pg_has_data
from app.cli.migrate_uuid.pg_prechecks import PG_MAX_INDEXED_THREAD_ID_BYTES
from app.cli.migrate_uuid.pg_prechecks import PG_MAX_INDEXED_VALUE_BYTES
from app.cli.migrate_uuid.pg_prechecks import first_pg_unindexable_column
from app.cli.migrate_uuid.pg_prechecks import first_pg_unindexable_metadata_field
from app.cli.migrate_uuid.pg_prechecks import first_pg_unstorable_column
from app.cli.migrate_uuid.pg_target import ensure_target_pg_fts
from app.cli.migrate_uuid.pg_target import initialize_target_postgresql
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.sqlite_source import detect_optional_tables
from app.cli.migrate_uuid.sqlite_source import detect_source_id_kind
from app.cli.migrate_uuid.sqlite_source import open_source_sqlite
from app.cli.migrate_uuid.sqlite_source import table_has_column
from app.cli.migrate_uuid.target_rows import SELECT_DISTINCT_TAGS_SQL
from app.cli.migrate_uuid.target_rows import access_backfill_values


async def run_migration_mixed_sqlite_to_postgresql(options: MigrationOptions) -> MigrationStats:
    """Migrate from a SQLite source to a PostgreSQL target.

    Vector embeddings are dropped (their on-disk binary formats are not portable
    between the two backends; a warning is emitted) -- re-embed the target
    afterward. All other data is copied: context_entries, tags, and image
    attachments. The target schema is auto-initialized when absent (base layout
    without the vector tables, which the server creates at the configured
    EMBEDDING_DIM on re-embed).

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """
    import asyncpg

    stats = MigrationStats()
    stats.start_metadata_counts(options)
    stats.warnings.append(
        'cross-backend migration drops vector embeddings; re-embed the target after migration',
    )

    _, source_address = parse_backend_url(options.source_url)
    source = open_source_sqlite(source_address)
    # Open the target connection INSIDE the try so a failed target connect closes
    # the already-open SQLite source via the finally instead of leaking it.
    target_conn: asyncpg.Connection | None = None
    try:
        target_conn = await pg_connect(options.target_url)
        id_kind = detect_source_id_kind(source)
        if id_kind != 'integer':
            stats.warnings.append(
                f'source database id column is {id_kind!r}; nothing to migrate',
            )
            return stats

        optional_tables = detect_optional_tables(source)

        # Bind the configured POSTGRESQL_SCHEMA EXPLICITLY for every TARGET probe (see
        # target_pg_has_data and run_migration_postgresql): current_schema() would fall
        # back to public before a non-default target schema exists.
        from app.settings import get_settings
        target_schema = get_settings().storage.postgresql_schema

        if await target_pg_has_data(target_conn, schema=target_schema):
            stats.errors.append(
                'target PostgreSQL database already contains context_entries rows. '
                'Recovery: if a prior run was interrupted, drop and recreate the target database '
                '(or pass a different --target-url) and rerun; the source database is unchanged. '
                'See the Recovering From an Interrupted Migration section of docs/migration-v2-to-v3.md.',
            )
            return stats

        cursor = source.execute(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        source_rows = cursor.fetchall()
        id_mapping = build_id_mapping(source_rows)
        rewriter = MetadataIdRewriter(options.id_paths, id_mapping, legacy_id_key=options.legacy_id_key)

        # Auto-initialize the target schema when absent (mirrors the SQLite
        # path and the PG->PG path). Cross-backend migration drops vector
        # embeddings, so the target is initialized WITHOUT the semantic/chunking
        # layout (with_semantic=False); the server creates the vector tables at
        # the operator's configured dimension when re-embedding later.
        target_initialized = await pg_table_exists(target_conn, 'context_entries', schema=target_schema)
        if not target_initialized:
            if options.dry_run:
                stats.warnings.append(
                    'target PostgreSQL database has no context_entries table; '
                    'it would be auto-initialized on a real run',
                )
            else:
                await initialize_target_postgresql(
                    options.target_url,
                    embedding_dim=None,
                    with_semantic=False,
                    source_has_fts=optional_tables.get('context_entries_fts', False),
                    stats=stats,
                )

        # FTS backstop for a PRE-EXISTING target (mirrors the PG->PG path):
        # the auto-init above runs only when context_entries is absent.
        if target_initialized:
            await ensure_target_pg_fts(
                options.target_url,
                target_conn,
                target_schema=target_schema,
                source_has_fts=optional_tables.get('context_entries_fts', False),
                dry_run=options.dry_run,
                stats=stats,
            )

        if not options.dry_run:
            await target_conn.execute('BEGIN')
        try:
            # Guard summary / content_hash on the SQLite source (a v2 DB predating
            # those ALTER-TABLE columns lacks them), mirroring copy_context_entries
            # and the PostgreSQL source paths so all four directions tolerate their
            # absence identically. NULL substitution keeps the row keys and INSERT below.
            summary_col_src = (
                'summary' if table_has_column(source, 'context_entries', 'summary') else 'NULL AS summary'
            )
            content_hash_col_src = (
                'content_hash' if table_has_column(source, 'context_entries', 'content_hash')
                else 'NULL AS content_hash'
            )
            entry_cursor = source.execute(
                f'SELECT id, thread_id, source, content_type, text_content, metadata, '
                f'{summary_col_src}, {content_hash_col_src}, created_at, updated_at FROM context_entries '
                f'ORDER BY created_at ASC, id ASC',
            )
            # Source ids of context_entries rows skipped for a PostgreSQL-unstorable
            # value. Their tags/image_attachments children must be skipped too: the
            # parent id stays in id_mapping (so the orphan-FK check would not catch
            # them), and inserting a child that references a never-inserted parent
            # would raise an FK violation on PostgreSQL -- relocating the very abort
            # this guard prevents.
            skipped_context_ids: set[int] = set()
            for row in entry_cursor:
                source_id = int(row['id'])
                new_id = id_mapping[source_id]
                outcome = rewriter.rewrite(row['metadata'], source_id)
                rewritten_metadata = outcome.metadata
                # A NUL (U+0000) or unpaired UTF-16 surrogate is legal in a SQLite
                # TEXT value but fatal on the PostgreSQL target: without this check
                # the asyncpg bind raises mid-transaction, ROLLBACKs the whole run,
                # and reports only the raw driver error with no row identification.
                # Checked UNCONDITIONALLY (not behind dry_run) so --dry-run surfaces
                # every affected row before a real run; the offending row is
                # identified and skipped, mirroring the orphan-FK skip-and-warn.
                row_thread_id = row['thread_id']
                unstorable = first_pg_unstorable_column(
                    (
                        ('thread_id', row_thread_id, False),
                        ('text_content', row['text_content'], False),
                        ('summary', row['summary'], False),
                        ('content_hash', row['content_hash'], False),
                        ('metadata', rewritten_metadata, True),
                    ),
                )
                if unstorable is None:
                    # Same skip-and-warn shape for the OTHER SQLite-accepts /
                    # PostgreSQL-rejects class on this path: a value SQLite indexed
                    # happily but the target's btree cannot hold. Only the columns the
                    # target schema itself indexes are checked; text_content and
                    # summary are unindexed and may be arbitrarily large. The metadata
                    # bound into the target is inspected too: every
                    # METADATA_INDEXED_FIELDS key is indexed by the expression index
                    # idx_metadata_<field>, under the same btree ceiling for a
                    # string-typed field, and a typed one holds only values of its
                    # declared type (a hard SQL cast for integer and boolean).
                    unstorable = first_pg_unindexable_column(
                        (('thread_id', row_thread_id, PG_MAX_INDEXED_THREAD_ID_BYTES),),
                    ) or first_pg_unindexable_metadata_field(rewritten_metadata)
                # The counts of a skipped row are not added: none of its values reach the target.
                stats.record_metadata(outcome, migrated=unstorable is None)
                if unstorable is not None:
                    column, reason = unstorable
                    stats.errors.append(
                        f'context_entries row id={source_id} thread_id={row_thread_id!r} '
                        f'column {column!r} skipped: {reason}',
                    )
                    skipped_context_ids.add(source_id)
                    continue
                if not options.dry_run:
                    owner_id_backfill, visibility_backfill = access_backfill_values()
                    await target_conn.execute(
                        'INSERT INTO context_entries '
                        '(id, thread_id, source, content_type, text_content, metadata, summary, '
                        'content_hash, owner_id, visibility, created_at, updated_at) '
                        'VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12)',
                        new_id,
                        row['thread_id'],
                        row['source'],
                        row['content_type'],
                        row['text_content'],
                        rewritten_metadata,
                        row['summary'],
                        row['content_hash'],
                        owner_id_backfill,
                        visibility_backfill,
                        stored_datetime_or_none(row['created_at']),
                        stored_datetime_or_none(row['updated_at']),
                    )
                stats.rows_migrated += 1

            # Copy tags and image attachments (portable across backends: tags
            # are TEXT, image payloads are BYTEA<->BLOB). Only the embedding
            # vectors are dropped cross-backend. Reads are guarded by source
            # table presence.
            if optional_tables.get('tags'):
                tag_cursor = source.execute(SELECT_DISTINCT_TAGS_SQL)
                for tag_row in tag_cursor:
                    sid = int(tag_row['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'tags row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    if sid in skipped_context_ids:
                        stats.warnings.append(
                            f'tags row context_entry_id={sid} skipped: parent context_entries row was skipped',
                        )
                        continue
                    # Skip a tag carrying a PostgreSQL-unstorable NUL/surrogate
                    # (see the context_entries guard above), unconditionally so
                    # --dry-run surfaces it too.
                    tag_unstorable = first_pg_unstorable_column(
                        (('tag', tag_row['tag'], False),),
                    ) or first_pg_unindexable_column(
                        # idx_tags_tag is a btree: a legacy tag longer than its
                        # index-tuple budget aborts the INSERT mid-transaction. The tag
                        # is indexed on its own, so it gets the full single-column budget.
                        (('tag', tag_row['tag'], PG_MAX_INDEXED_VALUE_BYTES),),
                    )
                    if tag_unstorable is not None:
                        column, reason = tag_unstorable
                        stats.errors.append(
                            f'tags row context_entry_id={sid} column {column!r} skipped: {reason}',
                        )
                        continue
                    if not options.dry_run:
                        await target_conn.execute(
                            'INSERT INTO tags (context_entry_id, tag) VALUES ($1::uuid, $2)',
                            mapped,
                            tag_row['tag'],
                        )
                    stats.tags_migrated += 1

            if optional_tables.get('image_attachments'):
                image_cursor = source.execute(
                    'SELECT context_entry_id, image_data, mime_type, image_metadata, position, created_at '
                    'FROM image_attachments ORDER BY id ASC',
                )
                for img in image_cursor:
                    sid = int(img['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'image_attachments row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    if sid in skipped_context_ids:
                        stats.warnings.append(
                            f'image_attachments row context_entry_id={sid} skipped: '
                            f'parent context_entries row was skipped',
                        )
                        continue
                    # Skip an attachment whose mime_type or image_metadata carries a
                    # PostgreSQL-unstorable NUL/surrogate (see the context_entries
                    # guard above), unconditionally so --dry-run surfaces it too.
                    img_unstorable = first_pg_unstorable_column(
                        (
                            ('mime_type', img['mime_type'], False),
                            ('image_metadata', img['image_metadata'], True),
                        ),
                    )
                    if img_unstorable is not None:
                        column, reason = img_unstorable
                        stats.errors.append(
                            f'image_attachments row context_entry_id={sid} column {column!r} skipped: {reason}',
                        )
                        continue
                    # A NULL or malformed created_at is preserved as NULL rather
                    # than crashing _coerce_datetime (the migration must not invent
                    # data for, nor abort on, arbitrary non-app source databases).
                    img_created_at = stored_datetime_or_none(img['created_at'])
                    if not options.dry_run:
                        await target_conn.execute(
                            'INSERT INTO image_attachments '
                            '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
                            'VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6)',
                            mapped,
                            img['image_data'],
                            img['mime_type'],
                            img['image_metadata'],
                            img['position'],
                            img_created_at,
                        )
                    stats.images_migrated += 1

            if not options.dry_run:
                await target_conn.execute('COMMIT')
        except Exception:
            if not options.dry_run:
                await target_conn.execute('ROLLBACK')
            raise
    finally:
        source.close()
        if target_conn is not None:
            await target_conn.close()
    return stats
