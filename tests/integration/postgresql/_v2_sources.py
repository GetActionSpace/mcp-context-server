"""Integer-keyed v2 PostgreSQL databases for the migration tests.

A migration test gets databases of its own on the test container, created under a name
the test chooses and dropped when it ends, so it never sees another test's rows.
"""

import contextlib
import json
from datetime import UTC
from datetime import datetime
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg

# The v2 layout of the three tables every migration direction copies: integer ids and no
# access-control columns.
PG_V2_SCHEMA = '''
CREATE TABLE context_entries (
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
CREATE TABLE tags (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE,
    tag TEXT NOT NULL
);
CREATE TABLE image_attachments (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT REFERENCES context_entries(id) ON DELETE CASCADE,
    image_data BYTEA NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSONB,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
'''


def database_url(pg_url: str, db_name: str) -> str:
    """Return ``pg_url`` with its database name replaced by ``db_name``."""
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{db_name}', parts.query, parts.fragment))


async def make_isolated_db(pg_test_url: str, db_name: str) -> str:
    """Create, or recreate empty, the database ``db_name`` and return its connection URL."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
        await admin.execute(f'CREATE DATABASE {db_name}')
    finally:
        await admin.close()
    return database_url(pg_test_url, db_name)


async def drop_isolated_db(pg_test_url: str, db_name: str) -> None:
    """Drop the database ``db_name``, ignoring a failure."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
    finally:
        await admin.close()


async def seed_pg_v2_source(url: str, rows: list[dict[str, object]]) -> None:
    """Create the v2 tables in the database at ``url`` and insert ``rows`` under their ids.

    Each row has the keys ``seed_source_db`` takes for a SQLite source: ``id``, ``thread_id``, ``source``,
    ``content_type``, ``text_content``, ``metadata`` (a dict or list stored as JSON, or None) and ``created_at``
    (an ISO text read as UTC). A v2 PostgreSQL source keeps metadata in a JSONB column, so unlike a SQLite
    source it never holds metadata text that is not JSON.

    Args:
        url: The connection URL of an empty database.
        rows: The source rows.
    """
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(PG_V2_SCHEMA)
        for row in rows:
            metadata = row.get('metadata')
            created_at = datetime.fromisoformat(str(row['created_at'])).replace(tzinfo=UTC)
            await conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, metadata, created_at, updated_at) '
                'VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $7)',
                row['id'],
                row['thread_id'],
                row['source'],
                row['content_type'],
                row['text_content'],
                json.dumps(metadata) if metadata is not None else None,
                created_at,
            )
    finally:
        await conn.close()
