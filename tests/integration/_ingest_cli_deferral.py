"""The deferred-generation scenario of the mcp-context-ingest command against a live HTTP server.

The SQLite and PostgreSQL entry points (``tests/integration/sqlite/test_ingest_cli.py``
and ``tests/integration/postgresql/test_ingest_cli.py``) run this scenario against a
server that, unlike the servers of the other scenarios in
:mod:`tests.integration._ingest_cli`, generates embeddings: ``stub_generation_env``
points it at an embedding endpoint the test process serves
(:mod:`tests.integration._ollama_stub`). The command runs and the corpus is read back
through the helpers of :mod:`tests.integration._ingest_cli`.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from fastmcp import Client

from tests.integration._ingest_cli import by_source
from tests.integration._ingest_cli import ingest
from tests.integration._ingest_cli import make_directory
from tests.integration._ingest_cli import managed_state
from tests.integration._ingest_cli import open_client
from tests.integration._ingest_cli import page
from tests.integration._ingest_cli import structured
from tests.integration._ingest_cli import write_files
from tests.integration._ollama_stub import OllamaStub

# The length of the vectors the stub embedding endpoint returns, a multiple of 64.
STUB_EMBEDDING_DIM = 64
# How long the scenario waits for the server's worker to finish the generation a run deferred.
GENERATION_TIMEOUT_S = 60
GENERATION_POLL_S = 0.2


async def generation_states(client: Client[Any], context_ids: list[str]) -> dict[str, str]:
    """Return the generation state of some entries the calling principal can read.

    Args:
        client: The principal's client.
        context_ids: The entries.

    Returns:
        Each entry's ``generation_status`` by context id.
    """
    rows = (await structured(client, 'get_context_by_ids', {'context_ids': context_ids}))['result']
    return {str(row['id']): str(row['generation_status']) for row in rows}


async def await_generated(base_url: str, context_ids: list[str]) -> None:
    """Wait until the server's worker has finished the generation of every entry.

    Args:
        base_url: Origin of a server that requires no authentication.
        context_ids: The entries whose generation a run deferred.
    """
    deadline = time.monotonic() + GENERATION_TIMEOUT_S
    async with open_client(base_url) as client:
        while (states := await generation_states(client, context_ids)) != dict.fromkeys(context_ids, 'ready'):
            assert time.monotonic() < deadline, f'generation did not finish within {GENERATION_TIMEOUT_S} s: {states}'
            await asyncio.sleep(GENERATION_POLL_S)


async def verify_deferred_generation(base_url: str, embedder: OllamaStub, work: Path) -> None:
    """Store and update a corpus with deferred generation, converge at once, and let the worker generate.

    A deferring run stores every document before its embeddings exist: it converges with
    exit 0 and counts the pending entries, the server's worker then embeds them through
    the stub endpoint and reports them ready, and a text edit goes the same way. A run
    without the flag that only changes metadata leaves the ready entries ready.

    Args:
        base_url: Origin of a server that requires no authentication, started with
            ``stub_generation_env`` over ``embedder``.
        embedder: The stub endpoint the server embeds through.
        work: The test's directory.
    """
    corpus, thread = make_directory(work / 'corpus'), 'kb:deferred'
    write_files(corpus, {'one.md': page('Deferred one', 600), 'two.md': page('Deferred two', 600)})
    embedded_before = len(embedder.inputs)

    bootstrap = await ingest(base_url, 'deferred', corpus, '--defer-generation', work=work)
    bootstrap.expect(0, created=2, generation_pending=2)
    assert 'generation pending 2' in bootstrap.stdout, bootstrap.stdout
    async with open_client(base_url) as client:
        ids = by_source(await managed_state(client, thread))
    await await_generated(base_url, sorted(ids.values()))
    embedded = embedder.inputs[embedded_before:]
    for title in ('# Deferred one', '# Deferred two'):
        assert any(text.startswith(title) for text in embedded), f'the worker never embedded {title!r}: {embedded}'

    write_files(corpus, {'one.md': page('Deferred one, edited', 600)})
    edited = await ingest(base_url, 'deferred', corpus, '--defer-generation', work=work)
    edited.expect(0, updated=1, unchanged=1, generation_pending=1)
    await await_generated(base_url, sorted(ids.values()))
    assert any(text.startswith('# Deferred one, edited') for text in embedder.inputs), 'the edited text was never embedded'

    (await ingest(base_url, 'deferred', corpus, '--tag', 'guide', work=work)).expect(0, updated=2)
    (await ingest(base_url, 'deferred', corpus, '--tag', 'guide', '--defer-generation', work=work)).expect(0, unchanged=2)
    async with open_client(base_url) as client:
        assert by_source(await managed_state(client, thread)) == ids, 'a deferred update must keep the entry'
