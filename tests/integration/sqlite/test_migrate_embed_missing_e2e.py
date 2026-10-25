"""End-to-end migration CLI tests on SQLite for --embed-missing.

Exercises ``mcp-context-server-migrate --embed-missing`` (standalone and
composed with ``--compress``) against a freshly-created SQLite database
that contains ``context_entries`` rows lacking ``embedding_metadata``.

Mirrors the structural pattern of ``test_migrate_compress_e2e.py``.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from app.backends import create_backend
from app.backends.base import MAINTENANCE
from app.cli.migrate_compression.compress import run_compress
from app.cli.migrate_embeddings import run_embed_missing
from app.embeddings.base import EmbeddingProvider
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import get_settings
from tests.helpers import LOCAL_SCOPE

DIM = 1024

_ALLOWED_COUNT_TABLES = {
    'vec_context_embeddings',
    'vec_context_embeddings_compressed',
    'embedding_chunks',
    'embedding_metadata',
    'compression_metadata',
    'context_entries',
}


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


class _FakeEmbeddingProvider:
    """Deterministic stub embedding provider returning unit float32 vectors.

    Mirrors the :class:`app.embeddings.base.EmbeddingProvider` protocol so
    the live ``generate_embeddings_with_timeout`` pipeline accepts it.
    """

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim
        self._rng = np.random.default_rng(seed=0)

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def embed_query(self, text: str) -> list[float]:
        """Return a deterministic unit vector derived from the text hash."""
        # Seed per-text so identical inputs produce identical embeddings.
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed=seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        return vec.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(t) for t in texts]

    async def is_available(self) -> bool:
        return True

    def get_dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return 'fake'


def _seed_missing_database(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_docs: int = 3,
) -> list[str]:
    """Create a SQLite database with ``n_docs`` context_entries lacking embeddings.

    Args:
        db_path: Path the SQLite database will be created at.
        monkeypatch: Test-scope env manipulator.
        n_docs: Number of context_entries rows to seed.

    Returns:
        The list of generated context IDs in insertion order. Each row
        has ``text_content`` set and NO corresponding embedding_metadata
        entry.
    """
    monkeypatch.setenv('DB_PATH', str(db_path))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()

    async def _setup() -> list[str]:
        from app.schemas import load_schema

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(load_schema('sqlite'))
            # Create the embedding tables manually. Mirror the
            # ``test_migrate_compress_e2e._seed_fp32_database`` pattern,
            # using plain physical tables so we avoid loading sqlite-vec
            # in this process.
            conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS vec_context_embeddings (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id TEXT NOT NULL,
                    vec_rowid INTEGER NOT NULL,
                    start_index INTEGER NOT NULL DEFAULT 0,
                    end_index INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    context_id TEXT NOT NULL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                );
                ''',
            )
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            repos = RepositoryContainer(backend)
            ids: list[str] = []
            for i in range(n_docs):
                cid, _ = await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='embed-missing-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'document number {i}',
                    metadata=None,
                )
                ids.append(cid)
            return ids
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    return asyncio.run(_setup())


def _table_exists_sqlite(db_path: Path, table_name: str) -> bool:
    """Return True when ``table_name`` exists in the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def _count_table(db_path: Path, table_name: str) -> int:
    """Return COUNT(*) for ``table_name``; restricted to the test allow-list."""
    if table_name not in _ALLOWED_COUNT_TABLES:
        raise ValueError(f'table {table_name!r} not in count allow-list')
    conn = sqlite3.connect(str(db_path))
    try:
        return int(
            conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0],
        )
    finally:
        conn.close()


@pytest.mark.integration
def test_embed_missing_fp32_standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--embed-missing`` standalone backfills into the fp32 vec table."""
    db = tmp_path / 'embed_fp32.db'
    ids = _seed_missing_database(db, monkeypatch, n_docs=3)
    assert _count_table(db, 'embedding_metadata') == 0

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    # Disable chunking so single-shot embed_query is used (avoids depending
    # on chunking service configuration in test).
    monkeypatch.setenv('ENABLE_CHUNKING', 'false')
    get_settings.cache_clear()

    provider = cast(EmbeddingProvider, _FakeEmbeddingProvider())
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc = run_embed_missing(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    # All three context_entries got an embedding_metadata row.
    assert _count_table(db, 'embedding_metadata') == len(ids)
    # The fp32 path wrote into vec_context_embeddings + embedding_chunks.
    assert _count_table(db, 'embedding_chunks') == len(ids)
    assert _count_table(db, 'vec_context_embeddings') == len(ids)
    # Compressed table NOT used in fp32 path.
    assert not _table_exists_sqlite(db, 'vec_context_embeddings_compressed')


@pytest.mark.integration
def test_embed_missing_compressed_standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--embed-missing`` with compression on writes into the compressed table.

    Requires the compression schema migration to have run first so the
    ``vec_context_embeddings_compressed`` table and singleton
    ``compression_metadata`` row exist.
    """
    db = tmp_path / 'embed_compressed.db'
    ids = _seed_missing_database(db, monkeypatch, n_docs=3)
    assert _count_table(db, 'embedding_metadata') == 0
    # Drop the unused fp32 table; the compressed path uses its own.
    conn = sqlite3.connect(str(db))
    try:
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '0')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.setenv('ENABLE_CHUNKING', 'false')
    get_settings.cache_clear()
    _reset_compression_cache()
    # Refresh the module-level ``settings`` binding the autouse
    # ``force_compression_off`` fixture set to compression=off so
    # generate_compression_with_timeout sees compression=true at runtime.
    import app.migrations.compression as _compression_migration_module
    import app.tools._generation as _generation_module
    monkeypatch.setattr(_generation_module, 'settings', get_settings())
    monkeypatch.setattr(
        _compression_migration_module, 'settings', get_settings(),
    )

    # Apply the compression schema migration so the target tables exist.
    async def _apply_compression_schema() -> None:
        from app.migrations.compression import apply_compression_migration
        backend = create_backend(backend_type='sqlite', db_path=str(db))
        await backend.initialize()
        try:
            await apply_compression_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_apply_compression_schema())

    provider = cast(EmbeddingProvider, _FakeEmbeddingProvider())
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc = run_embed_missing(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    assert _count_table(db, 'embedding_metadata') == len(ids)
    # Compressed write path landed in the compressed table.
    assert _table_exists_sqlite(db, 'vec_context_embeddings_compressed')
    assert _count_table(db, 'vec_context_embeddings_compressed') == len(ids)


@pytest.mark.integration
def test_embed_missing_dry_run_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--embed-missing --dry-run`` reports the count without writing rows."""
    db = tmp_path / 'embed_dryrun.db'
    ids = _seed_missing_database(db, monkeypatch, n_docs=3)
    assert _count_table(db, 'embedding_metadata') == 0

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()

    rc = run_embed_missing(f'sqlite:///{db}', dry_run=True)

    assert rc == 0
    err = capsys.readouterr().err
    assert f'Found {len(ids)} entries with missing embeddings' in err
    assert '[DRY-RUN]' in err
    # No new rows.
    assert _count_table(db, 'embedding_metadata') == 0
    assert _count_table(db, 'vec_context_embeddings') == 0


@pytest.mark.integration
def test_embed_missing_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second --embed-missing run finds zero missing and exits cleanly."""
    db = tmp_path / 'embed_idem.db'
    ids = _seed_missing_database(db, monkeypatch, n_docs=3)

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('ENABLE_CHUNKING', 'false')
    get_settings.cache_clear()

    provider = cast(EmbeddingProvider, _FakeEmbeddingProvider())
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc1 = run_embed_missing(f'sqlite:///{db}', dry_run=False)
    assert rc1 == 0
    assert _count_table(db, 'embedding_metadata') == len(ids)

    # Drain capture from the first run.
    capsys.readouterr()

    # Second run should find ZERO missing.
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc2 = run_embed_missing(f'sqlite:///{db}', dry_run=False)
    assert rc2 == 0
    err2 = capsys.readouterr().err
    assert 'Found 0 entries with missing embeddings' in err2
    # Row counts unchanged.
    assert _count_table(db, 'embedding_metadata') == len(ids)


@pytest.mark.integration
def test_compress_then_embed_missing_composed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--compress`` followed by ``--embed-missing`` lands backfilled rows compressed.

    Validates how the two modes compose: pre-existing fp32 embeddings are
    compressed first, then missing entries are embedded directly into the
    compressed table.
    """
    db = tmp_path / 'compose.db'
    # Reuse the fp32 seed pattern from test_migrate_compress_e2e to plant
    # 2 fp32 rows AND 2 missing entries on top.
    import struct
    monkeypatch.setenv('DB_PATH', str(db))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()

    async def _setup() -> tuple[list[str], list[str]]:
        from app.schemas import load_schema

        conn = sqlite3.connect(str(db))
        try:
            conn.executescript(load_schema('sqlite'))
            conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS vec_context_embeddings (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id TEXT NOT NULL,
                    vec_rowid INTEGER NOT NULL,
                    start_index INTEGER NOT NULL DEFAULT 0,
                    end_index INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    context_id TEXT NOT NULL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                );
                ''',
            )
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db))
        await backend.initialize()
        try:
            repos = RepositoryContainer(backend)
            rng = np.random.default_rng(42)
            embedded: list[str] = []
            for i in range(2):
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec /= np.linalg.norm(vec)
                cid, _ = await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='compose-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'embedded-doc-{i}',
                    metadata=None,
                )
                embedded.append(cid)
                blob = struct.pack(f'<{DIM}f', *vec.tolist())

                def _write_chunk(
                    cn: sqlite3.Connection,
                    *,
                    context_id: str = cid,
                    payload: bytes = blob,
                ) -> None:
                    cur = cn.execute(
                        'INSERT INTO vec_context_embeddings (embedding) VALUES (?)',
                        (payload,),
                    )
                    vec_rowid = cur.lastrowid
                    cn.execute(
                        'INSERT INTO embedding_chunks '
                        '(context_id, vec_rowid, start_index, end_index) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, vec_rowid, 0, DIM),
                    )
                    cn.execute(
                        'INSERT INTO embedding_metadata '
                        '(context_id, model_name, dimensions, chunk_count) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, 'test-model', DIM, 1),
                    )

                await backend.execute_write(_write_chunk, access_scope=MAINTENANCE)

            missing: list[str] = []
            for i in range(2):
                cid, _ = await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='compose-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'missing-doc-{i}',
                    metadata=None,
                )
                missing.append(cid)
            return embedded, missing
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    embedded_ids, missing_ids = asyncio.run(_setup())
    assert _count_table(db, 'embedding_metadata') == 2
    assert _count_table(db, 'vec_context_embeddings') == 2

    # Phase 1: --compress (fp32 -> compressed).
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '0')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()
    _reset_compression_cache()

    rc1 = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc1 == 0
    assert _table_exists_sqlite(db, 'vec_context_embeddings_compressed')
    assert not _table_exists_sqlite(db, 'vec_context_embeddings')
    assert _count_table(db, 'vec_context_embeddings_compressed') == len(embedded_ids)

    # Phase 2: --embed-missing against the compressed layout.
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_CHUNKING', 'false')
    # The seeded fp32 rows recorded model_name='test-model'; configure the
    # backfill to the SAME model so the embed-missing model-consistency
    # pre-flight does not (correctly) refuse a mixed-model backfill.
    monkeypatch.setenv('EMBEDDING_MODEL', 'test-model')
    get_settings.cache_clear()
    _reset_compression_cache()
    # Refresh the module-level ``settings`` binding so
    # generate_compression_with_timeout observes compression=true.
    import app.tools._generation as _generation_module
    monkeypatch.setattr(_generation_module, 'settings', get_settings())

    provider = cast(EmbeddingProvider, _FakeEmbeddingProvider())
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc2 = run_embed_missing(f'sqlite:///{db}', dry_run=False)

    assert rc2 == 0
    # All 4 entries now have embedding_metadata rows (2 originally + 2 backfilled).
    assert _count_table(db, 'embedding_metadata') == len(embedded_ids) + len(missing_ids)
    # The compressed table now holds all 4 rows.
    assert _count_table(db, 'vec_context_embeddings_compressed') == len(embedded_ids) + len(missing_ids)
