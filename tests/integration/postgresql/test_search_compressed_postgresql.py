"""End-to-end search_compressed integration test on PostgreSQL.

Mirror of ``tests/integration/sqlite/test_search_compressed.py`` but
against the pgvector docker-compose container. Provisions an isolated
PostgreSQL database, seeds compressed embeddings via the production
write path, and checks that ``EmbeddingRepository.search_compressed``
returns the planted top-K.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from collections.abc import Generator
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import numpy as np
import pytest
import pytest_asyncio

from app.backends import create_backend
from app.compression.provenance import insert_compression_metadata
from app.compression.providers.turboquant.encoder import encode
from app.compression.types import CompressionMetadata
from app.migrations.chunking import apply_chunking_migration
from app.migrations.compression import apply_compression_migration
from app.migrations.semantic import apply_semantic_search_migration
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.settings import get_settings
from app.startup import init_database
from tests.helpers import LOCAL_SCOPE

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

DIM = 1024
N_QUERIES = 5
N_BACKGROUND = 100
TOP_K = 3
PLANT_NOISE = 0.10
SEED = 42
OVERLAP_FLOOR = 0.85


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database name replaced by ``new_db``."""
    parts = urlsplit(pg_url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        f'/{new_db}',
        parts.query,
        parts.fragment,
    ))


@pytest_asyncio.fixture
async def isolated_pg_db_search(
    pg_test_url: str,
) -> AsyncIterator[str]:
    """Create an isolated database, yield its connection string, drop after."""
    db_name = 'mcp_search_compressed_e2e'
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
        await admin.execute(f'CREATE DATABASE {db_name}')
    finally:
        await admin.close()

    target_url = _replace_db_name(pg_test_url, db_name)
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await setup.close()

    yield target_url

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
    finally:
        await admin.close()


@pytest.mark.asyncio
async def test_search_compressed_recall_postgresql(
    isolated_pg_db_search: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_compressed on PG returns planted top-K with >= OVERLAP_FLOOR overlap."""
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', isolated_pg_db_search)
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', str(SEED))
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()
    import app.migrations.chunking as chunking_module
    import app.migrations.compression as compression_module
    import app.migrations.semantic as semantic_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())
    monkeypatch.setattr(semantic_module, 'settings', get_settings())
    monkeypatch.setattr(chunking_module, 'settings', get_settings())

    backend = create_backend(
        backend_type='postgresql',
        connection_string=isolated_pg_db_search,
    )
    await backend.initialize()
    try:
        await init_database(backend=backend)
        # Semantic-search migration creates the embedding_metadata table
        # _store_compressed_pg writes and, with compression on, leaves out the fp32
        # vec_context_embeddings table. With compression on, the chunking migration
        # only adds embedding_metadata.chunk_count, because its vec_context_embeddings
        # statements are skipped. The compression migration creates
        # vec_context_embeddings_compressed and compression_metadata.
        await apply_semantic_search_migration(backend=backend)
        await apply_chunking_migration(backend=backend)
        await apply_compression_migration(backend=backend)
        await insert_compression_metadata(
            backend,
            CompressionMetadata(
                provider='turboquant', bits=4, variant='ip',
                seed=SEED, dim=DIM,
            ),
        )

        rng = np.random.default_rng(SEED)
        queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        repos = RepositoryContainer(backend)
        repo = EmbeddingRepository(backend)
        ground_truth: dict[int, set[str]] = {
            q: set() for q in range(N_QUERIES)
        }

        # Planted neighbors.
        doc_counter = 0
        for q_idx in range(N_QUERIES):
            q = queries[q_idx]
            for _ in range(TOP_K):
                noise = rng.standard_normal(DIM).astype(np.float32)
                doc = q + PLANT_NOISE * noise
                doc /= np.linalg.norm(doc)
                doc = doc.astype(np.float32)
                cid, _ = await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='pg-search-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'planted-q{q_idx}-{doc_counter}',
                    metadata=None,
                )
                doc_counter += 1
                ground_truth[q_idx].add(cid)
                payload = encode(
                    doc[None, :], bits=4, variant='ip', seed=SEED,
                ).to_bytes()
                await repo.store_chunked(
                    cid,
                    [
                        ChunkEmbedding(
                            embedding=doc.tolist(),
                            start_index=0,
                            end_index=10,
                            payload=payload,
                        ),
                    ],
                    model='test-model',
                    scope=LOCAL_SCOPE,
                )

        # Background documents.
        for bg_idx in range(N_BACKGROUND):
            bg = rng.standard_normal(DIM).astype(np.float32)
            bg /= np.linalg.norm(bg)
            bg = bg.astype(np.float32)
            cid, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='pg-search-e2e',
                source='user',
                content_type='text',
                text_content=f'background-{bg_idx}',
                metadata=None,
            )
            payload = encode(
                bg[None, :], bits=4, variant='ip', seed=SEED,
            ).to_bytes()
            await repo.store_chunked(
                cid,
                [
                    ChunkEmbedding(
                        embedding=bg.tolist(),
                        start_index=0,
                        end_index=10,
                        payload=payload,
                    ),
                ],
                model='test-model',
                scope=LOCAL_SCOPE,
            )

        overlaps: list[float] = []
        for q_idx in range(N_QUERIES):
            q = queries[q_idx]
            results, stats = await repo.search_compressed(
                query_embedding=q.tolist(),
                limit=TOP_K,
                thread_id='pg-search-e2e',
                scope=LOCAL_SCOPE,
            )
            assert stats['backend'] == 'postgresql'
            assert len(results) == TOP_K
            returned_ids = {r['id'] for r in results}
            truth = ground_truth[q_idx]
            overlap = len(returned_ids & truth) / TOP_K
            overlaps.append(overlap)

        min_overlap = min(overlaps)
        mean_overlap = float(np.mean(overlaps))
        print(
            f'\n[search_compressed pg e2e] min={min_overlap:.3f} '
            f'mean={mean_overlap:.3f}',
        )
        assert min_overlap >= OVERLAP_FLOOR, (
            f'PG recall regression: min {min_overlap:.3f} < {OVERLAP_FLOOR}; '
            f'per-query overlaps={overlaps}'
        )
    finally:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=10.0)
