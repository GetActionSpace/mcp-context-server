"""Cross-backend migration regression tests for tags / images / FTS.

The documented contract of the cross-backend migration paths is that only
vector embeddings are dropped: ``tags`` and ``image_attachments`` travel with
``context_entries``, and the PostgreSQL->SQLite path rebuilds the SQLite FTS
index.

These tests exercise both directions against a live pgvector Docker container
(behind ``requires_docker_postgres``) and assert that tags and image
attachments survive, and that the SQLite target's FTS index is populated.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from collections.abc import Generator
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

from app.cli.migrate_uuid.postgresql_to_sqlite import run_migration_mixed_postgresql_to_sqlite
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.sqlite_to_postgresql import run_migration_mixed_sqlite_to_postgresql
from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import get_settings
from tests.integration.postgresql._v2_sources import PG_V2_SCHEMA
from tests.integration.postgresql._v2_sources import drop_isolated_db
from tests.integration.postgresql._v2_sources import make_isolated_db

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


_SQLITE_V2_SCHEMA = '''
CREATE TABLE context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL,
    content_type TEXT NOT NULL,
    text_content TEXT,
    metadata TEXT,
    summary TEXT,
    content_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER,
    tag TEXT NOT NULL
);
CREATE TABLE image_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER,
    image_data BLOB NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata TEXT,
    position INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
'''


def _seed_sqlite_v2_source(path: Path) -> None:
    """Create and seed an integer-keyed SQLite v2 source with tags + images."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SQLITE_V2_SCHEMA)
        conn.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, text_content) "
            "VALUES (1, 'thread-a', 'user', 'text', 'alpha beta gamma')",
        )
        conn.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, text_content) "
            "VALUES (2, 'thread-a', 'agent', 'text', 'delta epsilon')",
        )
        conn.execute("INSERT INTO tags (context_entry_id, tag) VALUES (1, 'red'), (1, 'blue'), (2, 'green')")
        conn.execute(
            "INSERT INTO image_attachments "
            '(context_entry_id, image_data, mime_type, image_metadata, position) '
            "VALUES (1, ?, 'image/png', '{}', 0)",
            (b'\x89PNG\r\n\x1a\n',),
        )
        conn.commit()
    finally:
        conn.close()


@pytest_asyncio.fixture
async def isolated_pg_v2_source_with_tags_images(pg_test_url: str) -> AsyncIterator[str]:
    """Integer-keyed PG v2 source seeded with tags + images (no embeddings)."""
    db_name = 'mcp_cross_v2_src'
    url = await make_isolated_db(pg_test_url, db_name)
    setup = await asyncpg.connect(url)
    try:
        await setup.execute(PG_V2_SCHEMA)
        await setup.execute(
            "INSERT INTO context_entries (thread_id, source, content_type, text_content) "
            "VALUES ('thread-a', 'user', 'text', 'alpha beta gamma'), "
            "('thread-a', 'agent', 'text', 'delta epsilon')",
        )
        await setup.execute("INSERT INTO tags (context_entry_id, tag) VALUES (1, 'red'), (1, 'blue'), (2, 'green')")
        await setup.execute(
            "INSERT INTO image_attachments "
            '(context_entry_id, image_data, mime_type, image_metadata, position) '
            "VALUES (1, $1, 'image/png', '{}'::jsonb, 0)",
            b'\x89PNG\r\n\x1a\n',
        )
    finally:
        await setup.close()
    try:
        yield url
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest.mark.asyncio
async def test_sqlite_to_pg_copies_tags_and_images(
    tmp_path: Path,
    pg_test_url: str,
) -> None:
    """SQLite->PG migration copies tags + image attachments (Defect A)."""
    source_path = tmp_path / 'v2_source.db'
    _seed_sqlite_v2_source(source_path)

    db_name = 'mcp_cross_sqlite_to_pg_tgt'
    target_url = await make_isolated_db(pg_test_url, db_name)
    try:
        options = MigrationOptions(
            source_url=str(source_path),
            target_url=target_url,
            dry_run=False,
            report_path=None,
        )
        stats = await run_migration_mixed_sqlite_to_postgresql(options)

        assert not stats.errors
        assert stats.rows_migrated == 2
        assert stats.tags_migrated == 3
        assert stats.images_migrated == 1

        tgt = await asyncpg.connect(target_url)
        try:
            assert await tgt.fetchval('SELECT COUNT(*) FROM tags') == 3
            assert await tgt.fetchval('SELECT COUNT(*) FROM image_attachments') == 1
        finally:
            await tgt.close()
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest.mark.asyncio
async def test_pg_to_sqlite_copies_tags_images_and_rebuilds_fts(
    isolated_pg_v2_source_with_tags_images: str,
    tmp_path: Path,
) -> None:
    """PG->SQLite migration copies tags + images and rebuilds the FTS index (Defects B + C)."""
    target_path = tmp_path / 'v3_target.db'

    options = MigrationOptions(
        source_url=isolated_pg_v2_source_with_tags_images,
        target_url=str(target_path),
        dry_run=False,
        report_path=None,
    )
    stats = await run_migration_mixed_postgresql_to_sqlite(options)

    assert not stats.errors
    assert stats.rows_migrated == 2
    assert stats.tags_migrated == 3
    assert stats.images_migrated == 1
    assert stats.fts_rebuilt is True

    conn = sqlite3.connect(str(target_path))
    try:
        assert conn.execute('SELECT COUNT(*) FROM tags').fetchone()[0] == 3
        assert conn.execute('SELECT COUNT(*) FROM image_attachments').fetchone()[0] == 1
        # FTS index is populated: a token from a copied row matches.
        fts_hits = conn.execute(
            "SELECT COUNT(*) FROM context_entries_fts WHERE context_entries_fts MATCH 'alpha'",
        ).fetchone()[0]
        assert fts_hits >= 1
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_pg_to_sqlite_image_null_created_at_does_not_crash(
    pg_test_url: str,
    tmp_path: Path,
) -> None:
    """A schema-legal NULL image created_at is preserved as NULL, not a crash (Defect: None.isoformat())."""
    db_name = 'mcp_cross_pg_null_ts'
    url = await make_isolated_db(pg_test_url, db_name)
    try:
        setup = await asyncpg.connect(url)
        try:
            await setup.execute(PG_V2_SCHEMA)
            await setup.execute(
                "INSERT INTO context_entries (thread_id, source, content_type, text_content) "
                "VALUES ('t', 'user', 'text', 'x')",
            )
            await setup.execute(
                "INSERT INTO image_attachments "
                '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
                "VALUES (1, $1, 'image/png', '{}'::jsonb, 0, NULL)",
                b'\x89PNG\r\n\x1a\n',
            )
        finally:
            await setup.close()

        target_path = tmp_path / 'null_ts_target.db'
        options = MigrationOptions(
            source_url=url,
            target_url=str(target_path),
            dry_run=False,
            report_path=None,
        )
        stats = await run_migration_mixed_postgresql_to_sqlite(options)

        assert not stats.errors
        assert stats.images_migrated == 1
        conn = sqlite3.connect(str(target_path))
        try:
            assert conn.execute('SELECT created_at FROM image_attachments').fetchone()[0] is None
        finally:
            conn.close()
    finally:
        await drop_isolated_db(pg_test_url, db_name)


@pytest.mark.asyncio
async def test_sqlite_to_pg_image_null_created_at_does_not_crash(
    tmp_path: Path,
    pg_test_url: str,
) -> None:
    """A schema-legal NULL image created_at is preserved as NULL, not a crash (_coerce_datetime(None))."""
    source_path = tmp_path / 'null_ts_source.db'
    conn = sqlite3.connect(str(source_path))
    try:
        conn.executescript(_SQLITE_V2_SCHEMA)
        conn.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, text_content) "
            "VALUES (1, 't', 'user', 'text', 'x')",
        )
        conn.execute(
            "INSERT INTO image_attachments "
            '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
            "VALUES (1, ?, 'image/png', '{}', 0, NULL)",
            (b'\x89PNG\r\n\x1a\n',),
        )
        conn.commit()
    finally:
        conn.close()

    db_name = 'mcp_cross_sqlite_null_ts'
    target_url = await make_isolated_db(pg_test_url, db_name)
    try:
        options = MigrationOptions(
            source_url=str(source_path),
            target_url=target_url,
            dry_run=False,
            report_path=None,
        )
        stats = await run_migration_mixed_sqlite_to_postgresql(options)

        assert not stats.errors
        assert stats.images_migrated == 1
        tgt = await asyncpg.connect(target_url)
        try:
            assert await tgt.fetchval('SELECT created_at FROM image_attachments') is None
        finally:
            await tgt.close()
    finally:
        await drop_isolated_db(pg_test_url, db_name)
