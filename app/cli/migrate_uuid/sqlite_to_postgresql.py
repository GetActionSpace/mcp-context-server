"""SQLite-to-PostgreSQL migration runner."""

import sqlite3

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


def _pg_skip_reason(row: sqlite3.Row, metadata: str | None) -> tuple[str, str] | None:
    """Return why a source entry row is skipped on its way to the PostgreSQL target, or None.

    SQLite accepts five classes of value that this check keeps from the PostgreSQL target,
    and the row is skipped on the first one found:

    1. A NUL (U+0000) or unpaired UTF-16 surrogate in any column.
    2. jsonb metadata that is not valid JSON, that decodes to a NUL or an unpaired
       surrogate through an escape such as ``\\u0000``, or that holds a ``NaN``,
       ``Infinity`` or ``-Infinity`` token.
    3. jsonb metadata holding a number literal too large for a double, such as
       ``1e400``. The rewriter cannot re-encode it and so keeps the text verbatim.
    4. A thread_id wider than :data:`PG_MAX_INDEXED_THREAD_ID_BYTES`, the budget for
       idx_context_entries_dedup_hash, the btree it leads. text_content and summary feed
       no btree index, so this check sets no width for them.
    5. A value under a METADATA_INDEXED_FIELDS key that fails its field's check: a
       string-typed value wider than :data:`PG_MAX_INDEXED_VALUE_BYTES`, the budget for
       the field's expression index idx_metadata_<field>, or an integer-, boolean- or
       float-typed value that does not parse as the declared type. Array- and
       object-typed fields build no expression index and are exempt.

    Unchecked, a value of class 1 or 2 makes the INSERT raise mid-transaction on every
    target, because the column types and the jsonb cast come with the base schema: the
    whole run rolls back and reports only the raw driver error, naming no row. A class 3
    value passes the jsonb cast, which stores it as a JSON number, so the INSERT succeeds;
    the row is skipped because the server refuses a non-finite number at its store
    boundary (:func:`app.metadata_types.non_finite_metadata_error`). A class 5 value the
    hard SQL cast of an integer or boolean index rejects makes the INSERT raise in the
    same way only on a target that already carries idx_metadata_<field>, which the server
    creates when it starts. On a target this command initialized, the INSERT succeeds and
    the target's first server start fails to build that index with exit 78.
    A float-typed field's index reads JSON numbers only, through a cast that never fails,
    so it refuses no value; a non-number under it is skipped because the server refuses a
    value of another type than the field declares.

    The check runs on a dry run too, so it surfaces every affected row before a real run.

    Args:
        row: The source ``context_entries`` row.
        metadata: The metadata text the target would store for it.

    Returns:
        The column and the reason of the first value that skips the row, or None.
    """
    thread_id = row['thread_id']
    return first_pg_unstorable_column(
        (
            ('thread_id', thread_id, False),
            ('text_content', row['text_content'], False),
            ('summary', row['summary'], False),
            ('content_hash', row['content_hash'], False),
            ('metadata', metadata, True),
        ),
    ) or first_pg_unindexable_column(
        (('thread_id', thread_id, PG_MAX_INDEXED_THREAD_ID_BYTES),),
    ) or first_pg_unindexable_metadata_field(metadata)


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
            entry_sql = (
                f'SELECT id, thread_id, source, content_type, text_content, metadata, '
                f'{summary_col_src}, {content_hash_col_src}, created_at, updated_at FROM context_entries '
                f'ORDER BY created_at ASC, id ASC'
            )
            # Every skip is decided before any row is stored, so a value citing a skipped
            # entry is kept as written whether the citing entry is older or newer than it:
            # the rewriter of the storing pass treats every skipped id as not migrated.
            # The source is read twice, one row at a time.
            skip_reasons: dict[int, tuple[str, str]] = {}
            for row in source.execute(entry_sql):
                source_id = int(row['id'])
                skip_reason = _pg_skip_reason(row, rewriter.rewrite(row['metadata'], source_id).metadata)
                if skip_reason is not None:
                    skip_reasons[source_id] = skip_reason
            storing_rewriter = rewriter.excluding(skip_reasons.keys())
            # Source ids of the skipped context_entries rows. Their tags/image_attachments
            # children must be skipped too: the parent id stays in id_mapping (so the
            # orphan-FK check would not catch them), and inserting a child that references
            # a never-inserted parent would raise an FK violation on PostgreSQL.
            skipped_context_ids = set(skip_reasons)
            # The metadata stored below differs from the metadata checked above only where
            # a value cites a skipped entry and stays as written instead of becoming a
            # 32-character id. Such a value passes every check the id passed: an id fails
            # the integer and boolean checks, and fails the float check unless it reads as a
            # decimal number (digits only, or digits around one 'e'), so a passing one sits
            # under an untyped top-level key, under an array- or object-typed one, which is
            # exempt and has no check, under a string-typed one, where the shorter original
            # fits too, or under a float-typed one, where the original passes too. So each
            # row is stored as checked, and its skip is taken from the first pass, never
            # decided again.
            for row in source.execute(entry_sql):
                source_id = int(row['id'])
                new_id = id_mapping[source_id]
                outcome = storing_rewriter.rewrite(row['metadata'], source_id)
                rewritten_metadata = outcome.metadata
                unstorable = skip_reasons.get(source_id)
                # The counts of a skipped row are not added: none of its values reach the target.
                stats.record_metadata(outcome, migrated=unstorable is None)
                if unstorable is not None:
                    column, reason = unstorable
                    stats.errors.append(
                        f'context_entries row id={source_id} thread_id={row["thread_id"]!r} '
                        f'column {column!r} skipped: {reason}',
                    )
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
                        # idx_tags_tag and the unique idx_tags_entry_tag
                        # (context_entry_id, tag) index the tag. It gets the budget of a
                        # string-typed metadata value, which reserves no room for
                        # context_entry_id.
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
