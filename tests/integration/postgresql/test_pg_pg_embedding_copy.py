"""Integration tests for the PG->PG embedding copy.

``run_migration_postgresql`` copies the ``embedding_metadata`` and
``vec_context_embeddings`` rows along with the entries. This module exercises
an integer-keyed v2 PostgreSQL source with seeded embeddings, runs the
migration into a UUID-keyed v3 PostgreSQL target, and asserts that the
embeddings are preserved.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Generator

import asyncpg
import pytest
import pytest_asyncio

from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql
from app.cli.migrate_uuid.records import MigrationOptions
from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import get_settings
from tests.integration.postgresql._v2_sources import drop_isolated_db
from tests.integration.postgresql._v2_sources import make_isolated_db

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

DIM = 1024


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


_V2_SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS context_entries (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL,
    content_type TEXT NOT NULL,
    text_content TEXT,
    metadata JSONB,
    summary TEXT,
    content_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tags (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE,
    tag TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS image_attachments (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE,
    image_data BYTEA NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSONB,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS embedding_metadata (
    context_id BIGINT NOT NULL PRIMARY KEY,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
'''


def _v2_vec_table_with_boundaries() -> str:
    """SQL creating the v2 vec_context_embeddings WITH start_index/end_index columns."""
    return f'''
    CREATE TABLE IF NOT EXISTS vec_context_embeddings (
        id BIGSERIAL PRIMARY KEY,
        context_id BIGINT NOT NULL,
        embedding vector({DIM}),
        start_index INTEGER NOT NULL DEFAULT 0,
        end_index INTEGER NOT NULL DEFAULT 0
    );
    '''


def _v2_vec_table_without_boundaries() -> str:
    """SQL creating the v2 vec_context_embeddings WITHOUT chunk boundaries."""
    return f'''
    CREATE TABLE IF NOT EXISTS vec_context_embeddings (
        id BIGSERIAL PRIMARY KEY,
        context_id BIGINT NOT NULL,
        embedding vector({DIM})
    );
    '''


@pytest_asyncio.fixture
async def isolated_pg_v2_source_db(pg_test_url: str) -> AsyncIterator[str]:
    """Create an integer-keyed v2 PostgreSQL source database with chunk boundaries."""
    db_name = 'mcp_pg_pg_v2_source'
    target_url = await make_isolated_db(pg_test_url, db_name)
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await setup.execute(_V2_SCHEMA_SQL)
        await setup.execute(_v2_vec_table_with_boundaries())
    finally:
        await setup.close()
    try:
        yield target_url
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest_asyncio.fixture
async def isolated_pg_v2_source_db_no_boundaries(
    pg_test_url: str,
) -> AsyncIterator[str]:
    """Create an integer-keyed v2 PostgreSQL source database without chunk boundaries."""
    db_name = 'mcp_pg_pg_v2_source_nb'
    target_url = await make_isolated_db(pg_test_url, db_name)
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await setup.execute(_V2_SCHEMA_SQL)
        await setup.execute(_v2_vec_table_without_boundaries())
    finally:
        await setup.close()
    try:
        yield target_url
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest_asyncio.fixture
async def isolated_pg_v3_target_db(pg_test_url: str) -> AsyncIterator[str]:
    """Create a UUID-keyed v3 PostgreSQL target database (production schema)."""
    db_name = 'mcp_pg_pg_v3_target'
    target_url = await make_isolated_db(pg_test_url, db_name)

    # Build the production v3 schema using the live init_database + migrations.
    from app.backends import create_backend
    from app.migrations.chunking import apply_chunking_migration
    from app.migrations.semantic import apply_semantic_search_migration
    from app.startup import init_database

    backend = create_backend(
        backend_type='postgresql', connection_string=target_url,
    )
    await backend.initialize()
    try:
        await init_database(backend=backend)
        # Build the fp32 migration-target layout exactly as the CLI's
        # initialize_target_postgresql does: force=True so the fp32
        # vec_context_embeddings table is created regardless of the ambient
        # ENABLE_EMBEDDING_COMPRESSION (which defaults on and would otherwise make
        # the server-style migration skip the fp32 table). A migration target is
        # always the fp32 layout; compression is a separate --compress step.
        await apply_semantic_search_migration(backend=backend, force=True, embedding_dim=DIM)
        await apply_chunking_migration(backend=backend, force=True)
    finally:
        await backend.shutdown()

    try:
        yield target_url
    finally:
        await drop_isolated_db(pg_test_url, db_name)


async def _connect_with_pgvector(pg_url: str) -> asyncpg.Connection:
    """Open an asyncpg connection with pgvector types registered."""
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(pg_url)
    schema_row = await conn.fetchrow(
        '''
        SELECT n.nspname FROM pg_extension e
        JOIN pg_namespace n ON e.extnamespace = n.oid
        WHERE e.extname = 'vector'
        ''',
    )
    schema = schema_row['nspname'] if schema_row else 'public'
    await register_vector(conn, schema=schema)
    return conn


async def _seed_source_with_embeddings(
    pg_url: str, n_docs: int = 3, *, with_boundaries: bool = True,
) -> list[int]:
    """Seed the v2 source with ``n_docs`` context_entries + embeddings.

    Args:
        pg_url: Source PostgreSQL URL.
        n_docs: Number of documents to seed.
        with_boundaries: Whether to populate start_index/end_index columns.

    Returns:
        List of source ``context_entries.id`` values in insertion order.
    """
    src = await _connect_with_pgvector(pg_url)
    ids: list[int] = []
    try:
        for i in range(n_docs):
            row_id = await src.fetchval(
                "INSERT INTO context_entries "
                '(thread_id, source, content_type, text_content) '
                "VALUES ($1, 'user', 'text', $2) RETURNING id",
                f'thread-{i}', f'text content {i}',
            )
            cid = int(row_id)
            ids.append(cid)
            await src.execute(
                "INSERT INTO embedding_metadata "
                '(context_id, model_name, dimensions, chunk_count) '
                "VALUES ($1, 'fake-model', $2, 1)",
                cid, DIM,
            )
            embedding = [0.1] * DIM
            if with_boundaries:
                await src.execute(
                    'INSERT INTO vec_context_embeddings '
                    '(context_id, embedding, start_index, end_index) '
                    'VALUES ($1, $2, $3, $4)',
                    cid, embedding, 0, len(f'text content {i}'),
                )
            else:
                await src.execute(
                    'INSERT INTO vec_context_embeddings '
                    '(context_id, embedding) VALUES ($1, $2)',
                    cid, embedding,
                )
    finally:
        await src.close()
    return ids


@pytest.mark.asyncio
async def test_pg_pg_migration_copies_embedding_metadata(
    isolated_pg_v2_source_db: str,
    isolated_pg_v3_target_db: str,
) -> None:
    """Embeddings are copied from the v2 source to the v3 target."""
    ids = await _seed_source_with_embeddings(
        isolated_pg_v2_source_db, n_docs=3,
    )
    assert len(ids) == 3

    options = MigrationOptions(
        source_url=isolated_pg_v2_source_db,
        target_url=isolated_pg_v3_target_db,
        dry_run=False,
        report_path=None,
    )
    stats = await run_migration_postgresql(options)

    assert stats.rows_migrated == 3
    assert stats.embedding_metadata_migrated == 3
    assert stats.vec_rows_migrated == 3
    tgt = await asyncpg.connect(isolated_pg_v3_target_db)
    try:
        em_count = await tgt.fetchval('SELECT COUNT(*) FROM embedding_metadata')
        vec_count = await tgt.fetchval('SELECT COUNT(*) FROM vec_context_embeddings')
        assert em_count == 3, (
            f'embedding_metadata bug REGRESSED: target has {em_count} rows'
        )
        assert vec_count == 3, (
            f'vec_context_embeddings bug REGRESSED: target has {vec_count} rows'
        )
    finally:
        await tgt.close()


@pytest.mark.asyncio
async def test_pg_pg_migration_preserves_chunk_boundaries(
    isolated_pg_v2_source_db: str,
    isolated_pg_v3_target_db: str,
) -> None:
    """start_index/end_index are copied when both source and target have them."""
    ids = await _seed_source_with_embeddings(
        isolated_pg_v2_source_db, n_docs=2, with_boundaries=True,
    )

    options = MigrationOptions(
        source_url=isolated_pg_v2_source_db,
        target_url=isolated_pg_v3_target_db,
        dry_run=False,
        report_path=None,
    )
    await run_migration_postgresql(options)

    tgt = await asyncpg.connect(isolated_pg_v3_target_db)
    try:
        rows = await tgt.fetch(
            'SELECT start_index, end_index FROM vec_context_embeddings '
            'ORDER BY id ASC',
        )
        # Each text was "text content {i}" so end_index == len(...) > 0.
        assert len(rows) == len(ids)
        for r in rows:
            assert r['start_index'] == 0
            assert r['end_index'] > 0
    finally:
        await tgt.close()


@pytest.mark.asyncio
async def test_pg_pg_migration_handles_missing_embedding_metadata_gracefully(
    pg_test_url: str,
    isolated_pg_v3_target_db: str,
) -> None:
    """Migration completes with zero embedding rows when source has no embeddings."""
    db_name = 'mcp_pg_pg_v2_no_embeddings'
    src_url = await make_isolated_db(pg_test_url, db_name)
    try:
        setup = await asyncpg.connect(src_url)
        try:
            await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
            await setup.execute(_V2_SCHEMA_SQL)
            await setup.execute(_v2_vec_table_with_boundaries())
            # Seed 2 context_entries but NO embedding_metadata / vec rows.
            await setup.execute(
                "INSERT INTO context_entries "
                '(thread_id, source, content_type, text_content) '
                "VALUES ('t', 'user', 'text', 'a'), ('t', 'user', 'text', 'b')",
            )
        finally:
            await setup.close()

        options = MigrationOptions(
            source_url=src_url,
            target_url=isolated_pg_v3_target_db,
            dry_run=False,
            report_path=None,
        )
        stats = await run_migration_postgresql(options)
        assert stats.rows_migrated == 2
        assert stats.embedding_metadata_migrated == 0
        assert stats.vec_rows_migrated == 0
        assert not stats.errors
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest.mark.asyncio
async def test_pg_pg_migration_remaps_context_id_to_uuid(
    isolated_pg_v2_source_db: str,
    isolated_pg_v3_target_db: str,
) -> None:
    """Copied embedding_metadata.context_id uses the UUIDv7 mapping."""
    ids = await _seed_source_with_embeddings(
        isolated_pg_v2_source_db, n_docs=2,
    )

    options = MigrationOptions(
        source_url=isolated_pg_v2_source_db,
        target_url=isolated_pg_v3_target_db,
        dry_run=False,
        report_path=None,
    )
    await run_migration_postgresql(options)

    # Compare: count the context_entries on target and ensure each one
    # has a matching embedding_metadata row by UUID.
    tgt = await asyncpg.connect(isolated_pg_v3_target_db)
    try:
        ce_uuids = await tgt.fetch('SELECT id FROM context_entries ORDER BY created_at ASC')
        em_uuids = await tgt.fetch('SELECT context_id FROM embedding_metadata ORDER BY context_id ASC')
        assert len(ce_uuids) == len(ids)
        assert len(em_uuids) == len(ids)
        ce_set = {str(r['id']) for r in ce_uuids}
        em_set = {str(r['context_id']) for r in em_uuids}
        # Every embedding_metadata.context_id refers to an existing
        # context_entries.id on the target (UUID-keyed).
        assert em_set.issubset(ce_set), (
            f'embedding_metadata UUIDs {em_set - ce_set} have no matching context_entries row'
        )
    finally:
        await tgt.close()


@pytest.mark.asyncio
async def test_pg_pg_migration_dry_run_does_not_insert(
    isolated_pg_v2_source_db: str,
    isolated_pg_v3_target_db: str,
) -> None:
    """Dry-run leaves the target empty even when source has embeddings."""
    await _seed_source_with_embeddings(
        isolated_pg_v2_source_db, n_docs=3,
    )

    options = MigrationOptions(
        source_url=isolated_pg_v2_source_db,
        target_url=isolated_pg_v3_target_db,
        dry_run=True,
        report_path=None,
    )
    stats = await run_migration_postgresql(options)
    # Counters still report what WOULD have happened.
    assert stats.rows_migrated == 3
    assert stats.embedding_metadata_migrated == 3
    assert stats.vec_rows_migrated == 3

    # Target tables remain empty.
    tgt = await asyncpg.connect(isolated_pg_v3_target_db)
    try:
        ce_count = await tgt.fetchval('SELECT COUNT(*) FROM context_entries')
        em_count = await tgt.fetchval('SELECT COUNT(*) FROM embedding_metadata')
        vec_count = await tgt.fetchval('SELECT COUNT(*) FROM vec_context_embeddings')
        assert ce_count == 0
        assert em_count == 0
        assert vec_count == 0
    finally:
        await tgt.close()


@pytest.mark.asyncio
async def test_pg_pg_migration_auto_inits_empty_target(
    isolated_pg_v2_source_db: str,
    pg_test_url: str,
) -> None:
    """A bare (schema-less) target is auto-initialized and receives data + embeddings.

    The CLI mirrors the SQLite path and creates the fp32 target schema itself,
    so the operator never has to pre-initialize the target. It MUST NOT create
    the compressed layout (compression is a separate --compress step).
    """
    await _seed_source_with_embeddings(isolated_pg_v2_source_db, n_docs=3)
    db_name = 'mcp_pg_pg_v3_autoinit'
    target_url = await make_isolated_db(pg_test_url, db_name)
    try:
        options = MigrationOptions(
            source_url=isolated_pg_v2_source_db,
            target_url=target_url,
            dry_run=False,
            report_path=None,
        )
        stats = await run_migration_postgresql(options)

        assert not stats.errors
        assert stats.rows_migrated == 3
        assert stats.embedding_metadata_migrated == 3
        assert stats.vec_rows_migrated == 3

        tgt = await asyncpg.connect(target_url)
        try:
            assert await tgt.fetchval('SELECT COUNT(*) FROM context_entries') == 3
            assert await tgt.fetchval('SELECT COUNT(*) FROM vec_context_embeddings') == 3
            assert await tgt.fetchval('SELECT COUNT(*) FROM embedding_metadata') == 3
            # Auto-init must produce the fp32 layout, never the compressed one.
            comp_table = await tgt.fetchval("SELECT to_regclass('compression_metadata')")
            assert comp_table is None, 'auto-init must NOT create the compressed layout'
        finally:
            await tgt.close()
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest.mark.asyncio
async def test_pg_pg_migration_source_without_embedding_metadata_table(
    pg_test_url: str,
    isolated_pg_v3_target_db: str,
) -> None:
    """A v2 source that never enabled semantic search (no embedding_metadata TABLE) does not crash.

    Distinct from test_..._missing_embedding_metadata_gracefully, which keeps the
    table but seeds zero rows. Here the table is ABSENT entirely; without the
    guard, an unconditional ``SELECT ... FROM embedding_metadata`` would crash
    the whole migration.
    """
    db_name = 'mcp_pg_pg_v2_no_em_table'
    src_url = await make_isolated_db(pg_test_url, db_name)
    try:
        setup = await asyncpg.connect(src_url)
        try:
            # Base v2 schema WITHOUT embedding_metadata / vec_context_embeddings.
            await setup.execute(
                'CREATE TABLE context_entries ('
                '  id BIGSERIAL PRIMARY KEY, thread_id TEXT NOT NULL, source TEXT NOT NULL,'
                '  content_type TEXT NOT NULL, text_content TEXT, metadata JSONB, summary TEXT,'
                '  content_hash TEXT, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,'
                '  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);'
                'CREATE TABLE tags (id BIGSERIAL PRIMARY KEY,'
                '  context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE, tag TEXT NOT NULL);'
                'CREATE TABLE image_attachments (id BIGSERIAL PRIMARY KEY,'
                '  context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE,'
                '  image_data BYTEA NOT NULL, mime_type TEXT NOT NULL, image_metadata JSONB,'
                '  position INTEGER DEFAULT 0, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);',
            )
            await setup.execute(
                "INSERT INTO context_entries (thread_id, source, content_type, text_content) "
                "VALUES ('t', 'user', 'text', 'a'), ('t', 'user', 'text', 'b')",
            )
        finally:
            await setup.close()

        options = MigrationOptions(
            source_url=src_url,
            target_url=isolated_pg_v3_target_db,
            dry_run=False,
            report_path=None,
        )
        # Must complete without raising (the unguarded read would have crashed).
        stats = await run_migration_postgresql(options)
        assert stats.rows_migrated == 2
        assert stats.embedding_metadata_migrated == 0
        assert stats.vec_rows_migrated == 0
        assert not stats.errors
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest.mark.asyncio
async def test_pg_pg_migration_aborts_when_target_lacks_fp32_vec_table(
    isolated_pg_v2_source_db: str,
    pg_test_url: str,
) -> None:
    """Defensive backstop: source has embeddings but target lacks the fp32 vec table.

    Simulates a pre-existing target initialized compression-on (or semantic-off):
    the migration MUST record an error and abort before writing any rows, never
    silently dropping the embeddings at exit 0.
    """
    await _seed_source_with_embeddings(isolated_pg_v2_source_db, n_docs=2)
    db_name = 'mcp_pg_pg_v3_no_fp32_vec'
    target_url = await make_isolated_db(pg_test_url, db_name)
    try:
        setup = await asyncpg.connect(target_url)
        try:
            # UUID-keyed target WITH context_entries + embedding_metadata but NO
            # fp32 vec_context_embeddings table (the compression-on shape).
            await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
            await setup.execute(
                'CREATE TABLE context_entries ('
                '  id UUID PRIMARY KEY, thread_id TEXT NOT NULL, source TEXT NOT NULL,'
                '  content_type TEXT NOT NULL, text_content TEXT, metadata JSONB, summary TEXT,'
                '  content_hash TEXT, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,'
                '  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);'
                'CREATE TABLE embedding_metadata ('
                '  context_id UUID PRIMARY KEY, model_name TEXT NOT NULL, dimensions INTEGER NOT NULL,'
                '  chunk_count INTEGER NOT NULL DEFAULT 1,'
                '  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,'
                '  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);',
            )
        finally:
            await setup.close()

        options = MigrationOptions(
            source_url=isolated_pg_v2_source_db,
            target_url=target_url,
            dry_run=False,
            report_path=None,
        )
        stats = await run_migration_postgresql(options)

        assert stats.errors, 'migration must record an error rather than silently drop embeddings'
        assert any('vec_context_embeddings' in e for e in stats.errors)
        assert stats.rows_migrated == 0

        tgt = await asyncpg.connect(target_url)
        try:
            # Aborted before BEGIN: nothing was written.
            assert await tgt.fetchval('SELECT COUNT(*) FROM context_entries') == 0
        finally:
            await tgt.close()
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest.mark.asyncio
async def test_pg_pg_dry_run_on_empty_target_reports_symmetric_counts(
    isolated_pg_v2_source_db: str,
    pg_test_url: str,
) -> None:
    """Dry-run against a bare (auto-init-able) target previews symmetric counts.

    The embedding counter and the vec counter both report N, with no
    contradictory 'initialize the target schema first' warning, because a real
    run auto-initializes and copies everything. The preview must be accurate
    and write nothing.
    """
    await _seed_source_with_embeddings(isolated_pg_v2_source_db, n_docs=3)
    db_name = 'mcp_pg_pg_dryrun_empty'
    target_url = await make_isolated_db(pg_test_url, db_name)
    try:
        options = MigrationOptions(
            source_url=isolated_pg_v2_source_db,
            target_url=target_url,
            dry_run=True,
            report_path=None,
        )
        stats = await run_migration_postgresql(options)

        assert not stats.errors
        assert stats.rows_migrated == 3
        # Symmetric would-migrate counts, NOT embedding=3 / vec=0.
        assert stats.embedding_metadata_migrated == 3
        assert stats.vec_rows_migrated == 3
        assert not any('initialize the target schema first' in w for w in stats.warnings)

        # Dry-run wrote nothing: the bare target still has no context_entries table.
        tgt = await asyncpg.connect(target_url)
        try:
            assert await tgt.fetchval("SELECT to_regclass('context_entries')") is None
        finally:
            await tgt.close()
    finally:
        await drop_isolated_db(pg_test_url, db_name)
