"""Fixtures for the generation job executor and worker tests.

``db`` is a seeded SQLite database prepared as the server prepares one; its request channel
is the repository container the executor and the worker run on. ``active_legs`` makes every
generation leg active, and ``generation`` replaces the generation the executor runs with a
:class:`FakeGeneration` that records its calls.
"""

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Protocol
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio

import app.generation_jobs.executor as executor
import app.tools._generation as generation_module
from app.backends.base import StorageBackend
from app.compression.factory import get_cached_compression_provider
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import vector_at
from tests.repositories._access_scope_layouts import sqlite_scoped_db

type Outputs = tuple[list[ChunkEmbedding] | None, str | None, list[IndexNodeRow] | None]


@dataclass
class FakeGeneration:
    """A stand-in for the generation of an entry's outputs.

    Attributes:
        outputs: What a generation returns.
        error: What a generation raises instead, if anything.
        during: Awaited at the start of each generation, before it returns or raises.
        block: Whether a generation waits until it is cancelled.
        delay_s: How long a generation takes.
        calls: The text, source and legs of every generation, in order.
        started: Set once a generation started.
        cancelled: Whether a generation was cancelled.
    """

    outputs: Outputs = (None, None, None)
    error: Exception | None = None
    during: Callable[[], Awaitable[None]] | None = None
    block: bool = False
    delay_s: float = 0.0
    calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False

    async def __call__(self, text: str, source: str, *, run_embedding: bool, run_summary: bool, run_nodes: bool) -> Outputs:
        """Record the call, then return the outputs or raise the error."""
        self.calls.append(
            {'text': text, 'source': source, 'embedding': run_embedding, 'summary': run_summary, 'nodes': run_nodes},
        )
        self.started.set()
        try:
            if self.during is not None:
                await self.during()
            if self.block:
                await asyncio.Event().wait()
            await asyncio.sleep(self.delay_s)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        return self.outputs


class InstallGeneration(Protocol):
    """Installs a :class:`FakeGeneration` as the generation the executor runs."""

    def __call__(
        self,
        *,
        outputs: Outputs = ...,
        error: Exception | None = ...,
        during: Callable[[], Awaitable[None]] | None = ...,
        block: bool = ...,
        delay_s: float = ...,
    ) -> FakeGeneration:
        """Install and return a fake generation with the given behavior."""
        ...


@pytest_asyncio.fixture
async def db(async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """A seeded SQLite database prepared as the server prepares one, with compressed embeddings.

    Yields:
        The seeded database.
    """
    async with sqlite_scoped_db(async_db_initialized, 'compressed', monkeypatch) as scoped:
        yield scoped


@pytest.fixture
def active_legs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the embedding provider, the summary provider and the node-summary layer active."""
    monkeypatch.setattr(executor, 'get_embedding_provider', MagicMock)
    monkeypatch.setattr(generation_module, 'get_summary_provider', MagicMock)
    monkeypatch.setattr(executor, 'node_layer_active', lambda: True)


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch) -> InstallGeneration:
    """Return a function that installs a :class:`FakeGeneration` built from its keyword arguments."""

    def install(
        *,
        outputs: Outputs = (None, None, None),
        error: Exception | None = None,
        during: Callable[[], Awaitable[None]] | None = None,
        block: bool = False,
        delay_s: float = 0.0,
    ) -> FakeGeneration:
        fake = FakeGeneration(outputs=outputs, error=error, during=during, block=block, delay_s=delay_s)
        monkeypatch.setattr(executor, 'run_generation', fake)
        return fake

    return install


async def chunk_for(db: ScopedDb, text: str) -> ChunkEmbedding:
    """Return one chunk embedding of ``text`` in the form the database's layout stores."""
    vector = vector_at(0.0)
    payload: bytes | None = None
    if db.layout == 'compressed':
        provider = await get_cached_compression_provider()
        payload = provider.encode_sync(np.asarray([vector], dtype=np.float32))
    return ChunkEmbedding(embedding=vector, start_index=0, end_index=len(text), payload=payload)


def node_for(text: str) -> IndexNodeRow:
    """Return one index-tree node row spanning ``text``."""
    return IndexNodeRow(
        node_id='section', level=1, ordinal=1, title='Section', node_summary='About the section.',
        char_start=0, char_end=len(text),
    )
