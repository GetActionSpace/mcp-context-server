"""Tests for the compression sub-block in ``get_statistics``.

``app.tools.discovery.get_statistics`` returns a ``compression`` sub-block
alongside ``semantic_search``, ``fts``, ``chunking``, ``reranking``, and
``summary``. The block surfaces the provider, bits, variant, seed, dim,
and max_concurrent values so MCP clients can verify the active compression
configuration at runtime.

When compression is disabled the block reduces to
``{enabled: False, available: False}`` mirroring the existing
``semantic_search`` disabled-shape convention.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Generator
from pathlib import Path
from typing import Any
from typing import cast

import pytest
import pytest_asyncio

import app.tools.discovery as discovery_module
from app.backends import StorageBackend
from app.backends import create_backend
from app.compression.provenance import insert_compression_metadata
from app.compression.types import CompressionMetadata
from app.migrations.compression import apply_compression_migration
from app.settings import get_settings
from app.startup import set_backend
from app.startup import set_repositories


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings singleton + module bindings between tests.

    Compression env-var flips would otherwise leak into unrelated tests
    in the same pytest session because ``get_settings()`` is cached at
    process scope.

    Yields:
        Control to the test body; teardown clears the singleton and the
        startup-layer globals.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    # Wipe the startup-layer singletons so a subsequent test starts with
    # a clean slate (the discovery tool reads from these via
    # ensure_backend()/ensure_repositories()).
    set_backend(None)
    set_repositories(None)


def _configure_compression_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    seed: str = '42',
    bits: str = '4',
    variant: str = 'ip',
    dim: str = '1024',
) -> None:
    """Set compression env vars and refresh module-level settings bindings."""
    for var in (
        'ENABLE_EMBEDDING_COMPRESSION', 'COMPRESSION_SEED',
        'COMPRESSION_BITS', 'COMPRESSION_VARIANT', 'COMPRESSION_PROVIDER',
        'EMBEDDING_DIM',
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv(
        'ENABLE_EMBEDDING_COMPRESSION', 'true' if enabled else 'false',
    )
    monkeypatch.setenv('COMPRESSION_SEED', seed)
    monkeypatch.setenv('COMPRESSION_BITS', bits)
    monkeypatch.setenv('COMPRESSION_VARIANT', variant)
    monkeypatch.setenv('EMBEDDING_DIM', dim)
    get_settings.cache_clear()

    # discovery.py caches settings at import time -- refresh the binding so
    # the new env values are seen by the function under test.
    monkeypatch.setattr(discovery_module, 'settings', get_settings())
    # The compression migration module also caches settings; refresh it so
    # the migration sees the new toggle.
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


@pytest_asyncio.fixture
async def backend_with_compression_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with compression migration applied + provenance row.

    Yields:
        Initialized backend ready for ``get_statistics`` to query the
        compressed metadata block. Teardown shuts the backend down.
    """
    _configure_compression_env(monkeypatch, enabled=True)

    db_path = tmp_path / 'test_discovery_compression_enabled.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    await apply_compression_migration(backend=backend)

    # Bootstrap the provenance row directly (the validator normally does this
    # during lifespan; this fixture shortcuts that step so the test focuses on
    # the compression sub-block in the discovery response).
    await insert_compression_metadata(
        backend,
        CompressionMetadata(
            provider='turboquant', bits=4, variant='ip', seed=42, dim=1024,
        ),
    )

    set_backend(backend)

    try:
        yield backend
    finally:
        set_backend(None)
        set_repositories(None)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=5.0)


@pytest_asyncio.fixture
async def backend_with_compression_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with compression DISABLED at env level.

    Yields:
        Initialized backend ready for ``get_statistics`` to verify the
        disabled-shape compression block. Teardown shuts the backend down.
    """
    _configure_compression_env(monkeypatch, enabled=False)

    db_path = tmp_path / 'test_discovery_compression_disabled.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    # Disabled: migration is a no-op, validator never runs.
    await apply_compression_migration(backend=backend)

    set_backend(backend)

    try:
        yield backend
    finally:
        set_backend(None)
        set_repositories(None)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=5.0)


@pytest.mark.asyncio
async def test_get_statistics_includes_compression_block_when_enabled(
    backend_with_compression_enabled: StorageBackend,
) -> None:
    """When compression is enabled, get_statistics returns the full block.

    The block sources provider/bits/variant/seed/dim from the singleton
    ``compression_metadata`` row (DB-truth) and max_concurrent from
    runtime settings.
    """
    del backend_with_compression_enabled  # Implicit: set via set_backend

    stats = await discovery_module.get_statistics()

    assert 'compression' in stats, (
        'get_statistics response is missing the compression sub-block. '
        'Expected sibling to semantic_search/fts/chunking/reranking/summary '
        'blocks.'
    )
    # CompressionStatsDict declares all fields total=False; cast to a plain
    # dict so per-key indexing reads as a runtime structural assertion
    # rather than a strict-typed access pattern.
    block = cast(dict[str, Any], stats['compression'])
    assert block['enabled'] is True
    assert block['available'] is True
    assert block['provider'] == 'turboquant'
    assert block['bits'] == 4
    assert block['variant'] == 'ip'
    assert block['seed'] == 42
    assert block['dim'] == 1024
    assert isinstance(block['max_concurrent'], int)
    assert 1 <= block['max_concurrent'] <= 32


@pytest.mark.asyncio
async def test_get_statistics_compression_block_when_disabled(
    backend_with_compression_disabled: StorageBackend,
) -> None:
    """When compression is disabled, the block reduces to enabled/available=False."""
    del backend_with_compression_disabled  # Implicit: set via set_backend

    stats = await discovery_module.get_statistics()

    assert 'compression' in stats
    block = cast(dict[str, Any], stats['compression'])
    assert block['enabled'] is False
    assert block['available'] is False
    # Provider/bits/variant/seed/dim must NOT be present in disabled shape
    # to match the existing semantic_search disabled-block convention.
    assert 'provider' not in block
    assert 'bits' not in block
