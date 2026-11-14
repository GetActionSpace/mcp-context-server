"""PostgreSQL-to-PostgreSQL migration runner."""

import logging

from app.cli.migrate_uuid.id_mapping import NULL_CREATED_AT_ANCHOR
from app.cli.migrate_uuid.id_mapping import created_at_for_id
from app.cli.migrate_uuid.metadata_ids import MetadataIdRewriter
from app.cli.migrate_uuid.pg_connection import detect_source_embedding_dim_pg
from app.cli.migrate_uuid.pg_connection import pg_column_exists
from app.cli.migrate_uuid.pg_connection import pg_connect
from app.cli.migrate_uuid.pg_connection import pg_table_exists
from app.cli.migrate_uuid.pg_connection import target_pg_has_data
from app.cli.migrate_uuid.pg_embeddings import copy_embedding_metadata_pg
from app.cli.migrate_uuid.pg_embeddings import copy_vec_embeddings_pg
from app.cli.migrate_uuid.pg_prechecks import first_pg_unindexable_metadata_field
from app.cli.migrate_uuid.pg_target import ensure_target_pg_fts
from app.cli.migrate_uuid.pg_target import initialize_target_postgresql
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.target_rows import SELECT_DISTINCT_TAGS_SQL
from app.cli.migrate_uuid.target_rows import access_backfill_values
from app.ids import generate_id_with_timestamp
from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.pgvector_limits import exceeds_pgvector_index_dim_limit

logger = logging.getLogger(__name__)


async def run_migration_postgresql(options: MigrationOptions) -> MigrationStats:
    """Drive a PostgreSQL-to-PostgreSQL migration.

    The target PostgreSQL database must already exist (the CLI does not run
    ``CREATE DATABASE``), but its schema is auto-initialized when absent via
    :func:`initialize_target_postgresql`, so the server need not be started
    against the target first to create the schema. When the source carries
    embeddings, the target is built with the fp32 vector layout (compression is
    never enabled here); enable compression afterward with the separate
    ``--compress`` step. If a pre-existing target lacks the fp32
    ``vec_context_embeddings`` table while the source has embeddings (for example
    a target initialized with compression enabled), the migration aborts with a
    recorded error rather than silently dropping the vectors.

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """
    import asyncpg

    stats = MigrationStats()
    stats.start_metadata_counts(options)
    source_conn = await pg_connect(options.source_url)
    # Open the target connection INSIDE the try so a failed target connect
    # (unreachable host, bad credentials, role/connection limit, SSL) closes the
    # already-open source connection via the finally instead of leaking it.
    target_conn: asyncpg.Connection | None = None
    try:
        target_conn = await pg_connect(options.target_url)
        await source_conn.execute('BEGIN TRANSACTION READ ONLY')

        # Bind the configured POSTGRESQL_SCHEMA EXPLICITLY for the two TARGET probes that can
        # run before that schema exists, the data check and the context_entries probe, so
        # they inspect the schema the migration will WRITE to -- current_schema() would fall
        # back to public and mis-resolve a non-default schema (false abort, or silent
        # wrong-schema copy). See target_pg_has_data. The FTS probe binds it too. The vec
        # table backstop and the embedding copies probe through current_schema(), and only
        # after context_entries was found in, or created in, the configured schema.
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

        # Detect what the SOURCE carries so the target can be shaped to match.
        source_has_embeddings = await pg_table_exists(source_conn, 'embedding_metadata')
        # The vector table is detected INDEPENDENTLY of embedding_metadata: a source can
        # carry the metadata table without a vec_context_embeddings table (e.g. semantic
        # search was provisioned but never populated, or the vec table was dropped). The
        # vector copy must gate on THIS, not on embedding_metadata,
        # or `SELECT ... FROM vec_context_embeddings` crashes the whole migration -- the
        # SQLite path already detects the vec table separately.
        source_has_vec = await pg_table_exists(source_conn, 'vec_context_embeddings')
        source_dim = await detect_source_embedding_dim_pg(source_conn)
        # FTS on PostgreSQL is the text_search_vector generated column on
        # context_entries (no separate table); detect it so the auto-init
        # provisions target FTS iff the source had it, mirroring the SQLite path.
        source_has_fts = await pg_column_exists(source_conn, 'context_entries', 'text_search_vector')

        # Auto-initialize the target schema when it has no context_entries table,
        # mirroring the SQLite path (initialize_target_sqlite). The target is built
        # with the fp32 layout (the compression migration is never run here), so
        # copy_vec_embeddings_pg has a vec_context_embeddings table to write into.
        # Compression is a separate --compress step afterward.
        target_initialized = await pg_table_exists(target_conn, 'context_entries', schema=target_schema)
        if not target_initialized:
            # fp32 capability pre-flight for the auto-init: an embedding-carrying
            # source whose dimension exceeds the pgvector index cap cannot be
            # copied into the fp32 vector layout the auto-init would force-build
            # (the HNSW CREATE INDEX would fail mid-pipeline and leave a
            # partially initialized target). Refuse HERE, before any target DDL,
            # with a recorded error (clean exit 1) instead of letting
            # initialize_target_postgresql's own guard surface as an unhandled
            # exception (exit 2); a dry run reports the same refusal as a plan
            # warning, mirroring the pre-existing-target embeddings backstop.
            #
            # source_dim is None when the source embedding_metadata table exists
            # but is empty; the auto-init's semantic migration then falls back to
            # settings.embedding.dim, so the value the target DDL actually
            # templates is that fallback. Resolve it here (mirroring
            # initialize_target_postgresql's own pre-flight) so an over-limit
            # fallback is refused before any DDL and surfaced under --dry-run,
            # instead of slipping past the source_dim-only check to crash the
            # real run mid-index.
            effective_source_dim = source_dim
            fallback_dim_used = False
            if source_has_embeddings and effective_source_dim is None:
                from app.settings import get_settings

                effective_source_dim = get_settings().embedding.dim
                fallback_dim_used = True
            if (
                source_has_embeddings
                and effective_source_dim is not None
                and exceeds_pgvector_index_dim_limit(effective_source_dim)
            ):
                dim_clause = (
                    f'configured EMBEDDING_DIM ({effective_source_dim}, the fallback used '
                    'because the source embedding_metadata table is empty)'
                    if fallback_dim_used
                    else f'source embedding dimension ({effective_source_dim})'
                )
                message = (
                    f'{dim_clause} exceeds the pgvector index limit of '
                    f'{PGVECTOR_INDEX_DIM_LIMIT} dimensions for fp32 vectors: '
                    'auto-initializing the target would fail at the HNSW '
                    'CREATE INDEX and leave the target schema partially initialized. '
                    'Recovery: migrate from a source copy whose embedding tables '
                    '(embedding_metadata, vec_context_embeddings) are dropped so the '
                    'target initializes without the fp32 vector layout, then start the '
                    'target server with ENABLE_EMBEDDING_COMPRESSION=true (compressed '
                    'payloads have no pgvector dimension cap) and re-embed with '
                    '--embed-missing.'
                )
                if options.dry_run:
                    stats.warnings.append(f'{message} (a real run would abort)')
                else:
                    stats.errors.append(f'{message} Aborting before any target DDL.')
                    return stats
            elif options.dry_run:
                stats.warnings.append(
                    'target PostgreSQL database has no context_entries table; '
                    'it would be auto-initialized on a real run',
                )
            else:
                await initialize_target_postgresql(
                    options.target_url,
                    embedding_dim=source_dim,
                    with_semantic=source_has_embeddings,
                    source_has_fts=source_has_fts,
                    stats=stats,
                )

        # Defensive backstop (never silently drop embeddings): a PRE-EXISTING
        # target that has context_entries but lacks the fp32 vec_context_embeddings
        # table (e.g. initialized with compression enabled or embedding
        # generation disabled) cannot receive the source's embeddings -- refuse rather than
        # discard them. Skipped when the target was just auto-initialized
        # (target_initialized is False): a real run already created the fp32 vec
        # table via initialize_target_postgresql, and a dry run reports the
        # auto-init plan instead. A dry run that records the backstop's warning
        # counts the vector rows against the layout the warning's remedy builds.
        vec_backstop_warned = False
        if target_initialized and source_has_embeddings:
            target_has_vec = await pg_table_exists(target_conn, 'vec_context_embeddings')
            if not target_has_vec:
                message = (
                    'source has embeddings but the target lacks the fp32 '
                    'vec_context_embeddings table (the target was likely '
                    'initialized with ENABLE_EMBEDDING_COMPRESSION=true or '
                    'ENABLE_EMBEDDING_GENERATION=false). Re-create the target with '
                    'ENABLE_EMBEDDING_GENERATION=true and ENABLE_EMBEDDING_COMPRESSION=false '
                    '(or let this CLI auto-initialize an empty target), run the '
                    'migration, then run --compress to enable compression.'
                )
                if options.dry_run:
                    stats.warnings.append(f'{message} (a real run would abort)')
                    vec_backstop_warned = True
                else:
                    stats.errors.append(f'{message} Aborting to avoid silently dropping embeddings.')
                    return stats

        # FTS backstop for a PRE-EXISTING target: initialize_target_postgresql
        # runs only when context_entries is absent, so a target bootstrapped by
        # other means would silently lose the source's full-text search.
        if target_initialized:
            await ensure_target_pg_fts(
                options.target_url,
                target_conn,
                target_schema=target_schema,
                source_has_fts=source_has_fts,
                dry_run=options.dry_run,
                stats=stats,
            )

        if not options.dry_run:
            await target_conn.execute('BEGIN')
        try:
            # Guard summary / content_hash: a v2 PostgreSQL source predating those
            # ALTER-TABLE columns (and never re-run against the server, so the
            # auto-migrations never fired) lacks them. The source is read-only and is
            # never auto-migrated here, so naming the columns unconditionally would
            # raise UndefinedColumnError and abort the whole migration -- whereas the
            # SQLite source path (copy_context_entries) already guards via
            # table_has_column. Substitute NULL when absent to keep the row keys and
            # the INSERT below unchanged, giving every direction identical tolerance.
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
            # The metadata expression indexes are deliberately absent from the PostgreSQL
            # base schema (a database this CLI initialized has none until its first server
            # startup), so a PostgreSQL SOURCE can hold an indexed metadata value the cast of
            # an integer or boolean idx_metadata_<field> rejects. Unchecked, such a value
            # aborts the INSERT mid-transaction and rolls the whole run back with a raw
            # driver error naming no source row when the target already carries
            # idx_metadata_<field>; on a target without it, the INSERT succeeds and the
            # target's first server start fails to build that index. The same check skips a
            # string-typed value over the pre-check's budget, PG_MAX_INDEXED_VALUE_BYTES, and
            # a value that does not parse as a float-typed field's declared type, which the
            # server refuses to store.
            # thread_id and tags get no equivalent check here: their indexes ARE in the
            # base schema, so the source already holds those values in btree indexes.
            # Checked unconditionally so --dry-run surfaces the row too.
            #
            # Every context_entries skip is decided before any row is stored, so a stored
            # value citing a skipped entry is kept as written whether the citing entry is
            # older or newer than it: the rewriter of the storing pass treats every skipped
            # id as not migrated.
            skip_reasons: dict[int, tuple[str, str]] = {}
            for entry in entry_rows:
                source_id = int(entry['id'])
                skip_reason = first_pg_unindexable_metadata_field(rewriter.rewrite(entry['metadata'], source_id).metadata)
                if skip_reason is not None:
                    skip_reasons[source_id] = skip_reason
            storing_rewriter = rewriter.excluding(skip_reasons.keys())
            # Source ids whose context_entries row was skipped. Their children must be
            # skipped too: a tag, attachment or embedding row pointing at an id the
            # target never received would violate the foreign key and abort the run --
            # replacing one skipped row with a total failure.
            pg_skipped_context_ids = set(skip_reasons)
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
            for entry in entry_rows:
                source_id = int(entry['id'])
                new_id = id_mapping[source_id]
                outcome = storing_rewriter.rewrite(entry['metadata'], source_id)
                rewritten_metadata = outcome.metadata
                unindexable = skip_reasons.get(source_id)
                # The counts of a skipped row are not added: none of its values reach the target.
                stats.record_metadata(outcome, migrated=unindexable is None)
                if unindexable is not None:
                    column, reason = unindexable
                    stats.errors.append(
                        f'context_entries row id={source_id} thread_id={entry["thread_id"]!r} '
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
                        entry['thread_id'],
                        entry['source'],
                        entry['content_type'],
                        entry['text_content'],
                        rewritten_metadata,
                        entry['summary'],
                        entry['content_hash'],
                        owner_id_backfill,
                        visibility_backfill,
                        entry['created_at'],
                        entry['updated_at'],
                    )
                stats.rows_migrated += 1

            tag_rows = (
                await source_conn.fetch(SELECT_DISTINCT_TAGS_SQL)
                if await pg_table_exists(source_conn, 'tags')
                else []
            )
            for tag_row in tag_rows:
                source_id = int(tag_row['context_entry_id'])
                tag_new_id: str | None = id_mapping.get(source_id)
                if tag_new_id is None:
                    stats.warnings.append(
                        f'tags row references missing context_entry_id={source_id}; skipped',
                    )
                    continue
                if source_id in pg_skipped_context_ids:
                    stats.warnings.append(
                        f'tags row context_entry_id={source_id} skipped: parent context_entries row was skipped',
                    )
                    continue
                if not options.dry_run:
                    await target_conn.execute(
                        'INSERT INTO tags (context_entry_id, tag) VALUES ($1::uuid, $2)',
                        tag_new_id,
                        tag_row['tag'],
                    )
                stats.tags_migrated += 1

            image_rows = (
                await source_conn.fetch(
                    'SELECT context_entry_id, image_data, mime_type, image_metadata, position, created_at '
                    'FROM image_attachments ORDER BY id ASC',
                )
                if await pg_table_exists(source_conn, 'image_attachments')
                else []
            )
            for img in image_rows:
                source_id = int(img['context_entry_id'])
                img_new_id: str | None = id_mapping.get(source_id)
                if img_new_id is None:
                    stats.warnings.append(
                        f'image_attachments row references missing context_entry_id={source_id}; skipped',
                    )
                    continue
                if source_id in pg_skipped_context_ids:
                    stats.warnings.append(
                        f'image_attachments row context_entry_id={source_id} skipped: '
                        f'parent context_entries row was skipped',
                    )
                    continue
                if not options.dry_run:
                    await target_conn.execute(
                        'INSERT INTO image_attachments '
                        '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
                        'VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6)',
                        img_new_id,
                        img['image_data'],
                        img['mime_type'],
                        img['image_metadata'],
                        img['position'],
                        img['created_at'],
                    )
                stats.images_migrated += 1

            # Copy embedding_metadata and vec_context_embeddings, remapping context_id,
            # so the target keeps the source embeddings. PostgreSQL has
            # no embedding_chunks table; the 1:N relationship lives in
            # vec_context_embeddings.id (BIGSERIAL PK) plus context_id
            # (UUID FK). Guarded by source table existence so a v2 source
            # that never enabled semantic search (no embedding_metadata
            # table) does not crash the migration.
            if source_has_embeddings and not source_has_vec:
                stats.warnings.append(
                    'source PostgreSQL database has an embedding_metadata table but no '
                    'vec_context_embeddings table; vector rows not copied (re-embed the '
                    'target afterward). Metadata rows are still migrated.',
                )
            if source_has_embeddings:
                # A dry run against a target the real run auto-initializes counts
                # through the same mapping, so it reports the rows, skips and warnings
                # of that run; the copies then take the target as the auto-init builds
                # it instead of probing tables that do not exist yet. A dry run whose
                # vector backstop warned takes the vector table the same way.
                probe_target = target_initialized or not options.dry_run
                await copy_embedding_metadata_pg(
                    source_conn, target_conn, id_mapping, stats, options.dry_run,
                    skipped_ids=pg_skipped_context_ids,
                    probe_target=probe_target,
                )
                # Gate the vector copy on the SOURCE vec table (copy_vec_embeddings_pg
                # reads FROM vec_context_embeddings, which would crash if absent).
                if source_has_vec:
                    await copy_vec_embeddings_pg(
                        source_conn, target_conn, id_mapping, stats, options.dry_run,
                        skipped_ids=pg_skipped_context_ids,
                        probe_target=probe_target and not vec_backstop_warned,
                    )

            if not options.dry_run:
                await target_conn.execute('COMMIT')
        except Exception:
            if not options.dry_run:
                await target_conn.execute('ROLLBACK')
            raise
    finally:
        await source_conn.close()
        if target_conn is not None:
            await target_conn.close()
    return stats
