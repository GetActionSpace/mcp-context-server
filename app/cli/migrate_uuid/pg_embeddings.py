"""Copy of embedding metadata and fp32 vectors between PostgreSQL databases."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING

from app.cli.migrate_uuid.records import MigrationStats

if TYPE_CHECKING:
    import asyncpg


async def copy_embedding_metadata_pg(
    source: 'asyncpg.Connection[asyncpg.Record]',
    target: 'asyncpg.Connection[asyncpg.Record]',
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
    *,
    skipped_ids: AbstractSet[int],
    probe_target: bool,
) -> None:
    """Copy ``embedding_metadata`` rows from a PostgreSQL source to a PostgreSQL target.

    Mirrors :func:`copy_embedding_metadata` (the SQLite path) but uses
    asyncpg placeholders, native UUID binding, and asyncpg's ``fetch``
    cursor. The source ``context_id`` is a BIGINT (integer-keyed v2
    schema); the target ``context_id`` is a UUID (v3 schema). Mapping
    is applied via ``id_mapping``. A row whose entry the run skipped, or
    whose entry is absent from the source, is left out with a warning
    naming which, since the target holds no entry for it to reference.

    Args:
        source: asyncpg connection to the PostgreSQL source database.
        target: asyncpg connection to the PostgreSQL target database.
        id_mapping: BIGINT-to-UUID mapping built from the source
            ``context_entries.id`` -> ``created_at`` rows.
        stats: Mutated to record ``embedding_metadata_migrated`` and
            warnings.
        dry_run: When True, skip INSERTs (counters still increment).
        skipped_ids: Old ids of the source entries the run skips.
        probe_target: False for a dry run whose target has no schema yet. The
            target is then not probed: the real run builds it with
            :func:`~app.cli.migrate_uuid.pg_target.initialize_target_postgresql`,
            whose ``embedding_metadata`` carries ``chunk_count``.
    """
    has_chunk_count_src = await source.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'embedding_metadata' AND column_name = 'chunk_count'
        )
        ''',
    )
    src_columns = ['context_id', 'model_name', 'dimensions', 'created_at', 'updated_at']
    if has_chunk_count_src:
        src_columns.append('chunk_count')
    select_sql = f'SELECT {", ".join(src_columns)} FROM embedding_metadata ORDER BY context_id ASC'
    rows = await source.fetch(select_sql)

    has_chunk_count_tgt = not probe_target or await target.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'embedding_metadata' AND column_name = 'chunk_count'
        )
        ''',
    )
    tgt_columns = ['context_id', 'model_name', 'dimensions', 'created_at', 'updated_at']
    if has_chunk_count_tgt:
        tgt_columns.append('chunk_count')
    # First column is cast to ::uuid; remaining columns use unadorned $N.
    placeholders = ['$1::uuid'] + [f'${i + 2}' for i in range(len(tgt_columns) - 1)]
    insert_sql = (
        f'INSERT INTO embedding_metadata ({", ".join(tgt_columns)}) '
        f'VALUES ({", ".join(placeholders)})'
    )

    inserted = 0
    for row in rows:
        source_id = int(row['context_id'])
        if source_id in skipped_ids:
            stats.warnings.append(
                f'embedding_metadata row context_id={source_id} skipped: parent context_entries row was skipped',
            )
            continue
        mapped = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'embedding_metadata row references missing context_id={source_id}; skipped',
            )
            continue
        params: list[object] = [
            mapped,
            row['model_name'],
            row['dimensions'],
            row['created_at'],
            row['updated_at'],
        ]
        if has_chunk_count_tgt:
            params.append(row['chunk_count'] if has_chunk_count_src else 1)
        if not dry_run:
            await target.execute(insert_sql, *params)
        inserted += 1
    stats.embedding_metadata_migrated = inserted


async def copy_vec_embeddings_pg(
    source: 'asyncpg.Connection[asyncpg.Record]',
    target: 'asyncpg.Connection[asyncpg.Record]',
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
    *,
    skipped_ids: AbstractSet[int],
    probe_target: bool,
) -> None:
    """Copy ``vec_context_embeddings`` rows from a PostgreSQL source to a PostgreSQL target.

    ``context_id`` is remapped (BIGINT -> UUID), and the target assigns each
    row a new ``id``. The ``embedding`` pgvector column is copied verbatim;
    the source must have pgvector installed and the target must have
    ``vec_context_embeddings`` initialized. Probes both source and target for the chunking
    migration's ``start_index``/``end_index`` columns (added by
    ``add_chunking_postgresql.sql``); when present on both sides, the
    columns are copied through. A row whose entry the run skipped, or
    whose entry is absent from the source, is left out with a warning
    naming which, since the target holds no entry for it to reference.

    Args:
        source: asyncpg connection to the PostgreSQL source database.
        target: asyncpg connection to the PostgreSQL target database.
        id_mapping: BIGINT-to-UUID mapping built from the source
            ``context_entries.id`` -> ``created_at`` rows.
        stats: Mutated to record ``vec_rows_migrated`` and warnings.
        dry_run: When True, skip INSERTs (counters still increment).
        skipped_ids: Old ids of the source entries the run skips.
        probe_target: False for a dry run whose target has no schema yet, or whose
            target lacks the ``vec_context_embeddings`` table the real run stops on.
            The target is then not probed and is taken as
            :func:`~app.cli.migrate_uuid.pg_target.initialize_target_postgresql`
            builds it, whose ``vec_context_embeddings`` carries the chunk boundary columns.
    """
    target_table_exists = not probe_target or await target.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = 'vec_context_embeddings'
        )
        ''',
    )
    if not target_table_exists:
        stats.warnings.append(
            'target PostgreSQL database has no vec_context_embeddings table; '
            'fp32 vec rows not copied (initialize the target schema first)',
        )
        return

    has_boundaries_src = await source.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'vec_context_embeddings' AND column_name = 'start_index'
        )
        ''',
    )
    has_boundaries_tgt = not probe_target or await target.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'vec_context_embeddings' AND column_name = 'start_index'
        )
        ''',
    )

    if has_boundaries_src and has_boundaries_tgt:
        select_sql = (
            'SELECT context_id, embedding, start_index, end_index '
            'FROM vec_context_embeddings ORDER BY context_id ASC'
        )
        insert_sql = (
            'INSERT INTO vec_context_embeddings '
            '(context_id, embedding, start_index, end_index) '
            'VALUES ($1::uuid, $2, $3, $4)'
        )
    else:
        select_sql = (
            'SELECT context_id, embedding FROM vec_context_embeddings '
            'ORDER BY context_id ASC'
        )
        insert_sql = (
            'INSERT INTO vec_context_embeddings (context_id, embedding) '
            'VALUES ($1::uuid, $2)'
        )
        if has_boundaries_src and not has_boundaries_tgt:
            stats.warnings.append(
                'source has start_index/end_index columns but target does not; '
                'chunk boundaries not copied (run the chunking migration on the target first)',
            )

    rows = await source.fetch(select_sql)
    inserted = 0
    for row in rows:
        source_id = int(row['context_id'])
        if source_id in skipped_ids:
            stats.warnings.append(
                f'vec_context_embeddings row context_id={source_id} skipped: parent context_entries row was skipped',
            )
            continue
        mapped = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'vec_context_embeddings row references missing context_id={source_id}; skipped',
            )
            continue
        if not dry_run:
            if has_boundaries_src and has_boundaries_tgt:
                await target.execute(
                    insert_sql,
                    mapped, row['embedding'], row['start_index'], row['end_index'],
                )
            else:
                await target.execute(insert_sql, mapped, row['embedding'])
        inserted += 1
    stats.vec_rows_migrated = inserted
