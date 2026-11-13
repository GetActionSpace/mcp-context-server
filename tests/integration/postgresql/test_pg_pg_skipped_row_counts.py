"""The metadata counts of a PostgreSQL-to-PostgreSQL migration describe the rows the target stores.

A row the target cannot index is skipped, so none of the values its metadata holds reach
the target and none of them is counted: the per-path counts and the stamp counts cover
the stored rows only.
"""

import contextlib
from collections.abc import Generator
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import LegacyIdCounts
from app.cli.migrate_uuid.records import MigrationOptions
from app.settings import get_settings

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

_SOURCE_DB = 'mcp_pg_pg_skip_counts_src'
_TARGET_DB = 'mcp_pg_pg_skip_counts_tgt'

_V2_ENTRIES_SQL = (
    'CREATE TABLE context_entries ('
    '  id BIGSERIAL PRIMARY KEY, thread_id TEXT NOT NULL, source TEXT NOT NULL,'
    '  content_type TEXT NOT NULL, text_content TEXT, metadata JSONB, summary TEXT,'
    '  content_hash TEXT, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,'
    '  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)'
)


@pytest.fixture
def status_is_the_indexed_field(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Index only ``status`` as a string, whatever the developer environment sets."""
    monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'status')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _database_url(pg_test_url: str, name: str) -> str:
    """Return ``pg_test_url`` naming the database ``name``."""
    parts = urlsplit(pg_test_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{name}', parts.query, parts.fragment))


async def _recreate_databases(pg_test_url: str) -> None:
    """Create empty source and target databases for this module."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        for name in (_SOURCE_DB, _TARGET_DB):
            await admin.execute(f'DROP DATABASE IF EXISTS {name}')
            await admin.execute(f'CREATE DATABASE {name}')
    finally:
        await admin.close()


async def _drop_databases(pg_test_url: str) -> None:
    """Drop this module's databases."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        for name in (_SOURCE_DB, _TARGET_DB):
            with contextlib.suppress(Exception):
                await admin.execute(f'DROP DATABASE IF EXISTS {name}')
    finally:
        await admin.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures('status_is_the_indexed_field')
async def test_a_skipped_row_adds_none_of_its_counts(pg_test_url: str) -> None:
    """The values and the stamp of a row the target skips are not counted."""
    await _recreate_databases(pg_test_url)
    try:
        source_url = _database_url(pg_test_url, _SOURCE_DB)
        source = await asyncpg.connect(source_url)
        try:
            await source.execute(_V2_ENTRIES_SQL)
            await source.execute(
                "INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, created_at) "
                "VALUES (1, 't', 'user', 'text', 'kept', NULL, '2024-01-01T10:00:00Z'), "
                "(2, 't', 'agent', 'text', 'skipped', $1::jsonb, '2024-02-01T10:00:00Z')",
                '{"status": "' + 'x' * 3000 + '", "references": {"context_ids": [1, 1]}}',
            )
        finally:
            await source.close()

        stats = await run_migration_postgresql(
            MigrationOptions(
                source_url=source_url,
                target_url=_database_url(pg_test_url, _TARGET_DB),
                legacy_id_key='legacy_id',
            ),
        )

        assert stats.rows_migrated == 1
        assert len(stats.errors) == 1
        assert stats.errors[0].startswith("context_entries row id=2 thread_id='t' column 'metadata.status' skipped")
        assert stats.id_paths == {DEFAULT_ID_PATH: IdPathCounts()}
        assert stats.legacy_id == LegacyIdCounts(key='legacy_id', stamped=1)
    finally:
        await _drop_databases(pg_test_url)
