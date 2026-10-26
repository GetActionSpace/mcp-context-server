"""The mcp-context-ingest command end to end against live SQLite servers (over HTTP).

Runs the scenarios of :mod:`tests.integration._ingest_cli` against real HTTP servers,
each serving a temporary SQLite file with generation off: a server that requires no
authentication, one that verifies minted JWTs, and one that does not offer corpus sync;
plus a server that generates embeddings through a stub endpoint, for deferred generation.
The same scenarios run against PostgreSQL in
``tests/integration/postgresql/test_ingest_cli.py``; the unreachable-server scenario
needs no server and runs here only.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from tests.integration._http_jwt import build_env
from tests.integration._http_jwt import build_jwt_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import running_http_server
from tests.integration._ingest_cli import STUB_EMBEDDING_DIM
from tests.integration._ingest_cli import SYNC_DISABLED
from tests.integration._ingest_cli import requires_denying_modes
from tests.integration._ingest_cli import verify_corpus_above_the_request_limit
from tests.integration._ingest_cli import verify_corpus_lifecycle
from tests.integration._ingest_cli import verify_deferred_generation
from tests.integration._ingest_cli import verify_principals_manage_their_own_copy
from tests.integration._ingest_cli import verify_sync_disabled_server
from tests.integration._ingest_cli import verify_token_rotation
from tests.integration._ingest_cli import verify_unlistable_directory
from tests.integration._ingest_cli import verify_unreachable_server
from tests.integration._ingest_cli import verify_unusable_files_are_kept
from tests.integration._ollama_stub import OllamaStub
from tests.integration._ollama_stub import serve_ollama_embeddings
from tests.integration._ollama_stub import stub_generation_env

pytestmark = pytest.mark.integration


@pytest.fixture
def open_server(tmp_path: Path) -> Iterator[str]:
    """Run a server that requires no authentication.

    Yields:
        The server origin URL.
    """
    env = build_env(backend='sqlite', database=str(tmp_path / 'open.db'), port=free_port(), auth=False)
    with running_http_server(env) as base_url:
        yield base_url


@pytest.fixture
def jwt_server(tmp_path: Path, jwt_key_pair: RSAKeyPair) -> Iterator[str]:
    """Run a server that verifies JWTs minted with ``jwt_key_pair``.

    Yields:
        The server origin URL.
    """
    env = build_jwt_env(
        backend='sqlite', database=str(tmp_path / 'jwt.db'), port=free_port(), public_key=jwt_key_pair.public_key,
    )
    with running_http_server(env) as base_url:
        yield base_url


@pytest.fixture
def sync_disabled_server(tmp_path: Path) -> Iterator[str]:
    """Run a server started with ``ENABLE_CONTEXT_SYNC=false``.

    Yields:
        The server origin URL.
    """
    database = str(tmp_path / 'disabled.db')
    env = {**build_env(backend='sqlite', database=database, port=free_port(), auth=False), **SYNC_DISABLED}
    with running_http_server(env) as base_url:
        yield base_url


class GeneratingServer(NamedTuple):
    """A server that generates embeddings, and the stub endpoint it embeds through."""

    base_url: str
    embedder: OllamaStub


@pytest.fixture
def generating_server(tmp_path: Path) -> Iterator[GeneratingServer]:
    """Run a server that requires no authentication and generates embeddings through a stub endpoint.

    Yields:
        The server origin URL and the stub.
    """
    with serve_ollama_embeddings(STUB_EMBEDDING_DIM) as embedder:
        env = {
            **build_env(backend='sqlite', database=str(tmp_path / 'generating.db'), port=free_port(), auth=False),
            **stub_generation_env(embedder),
        }
        with running_http_server(env) as base_url:
            yield GeneratingServer(base_url, embedder)


@pytest.mark.asyncio
async def test_corpus_lifecycle(open_server: str, tmp_path: Path) -> None:
    """Bootstrap, rerun, dry run, edit, removal, tag change, duplicate and emptying converge as planned."""
    await verify_corpus_lifecycle(open_server, tmp_path)


@pytest.mark.asyncio
async def test_unusable_files_keep_their_stored_entries(open_server: str, tmp_path: Path) -> None:
    """Oversized, undecodable and unreadable files exit 1 and never change or delete a stored entry."""
    await verify_unusable_files_are_kept(open_server, tmp_path)


@pytest.mark.asyncio
async def test_corpus_above_the_request_limit_converges(open_server: str, tmp_path: Path) -> None:
    """A corpus larger than one HTTP request converges through packed creates and packed updates."""
    await verify_corpus_above_the_request_limit(open_server, tmp_path)


@pytest.mark.asyncio
async def test_server_without_corpus_sync_exits_2(sync_disabled_server: str, tmp_path: Path) -> None:
    """A server with ENABLE_CONTEXT_SYNC=false stops the run at preflight."""
    await verify_sync_disabled_server(sync_disabled_server, tmp_path)


@pytest.mark.asyncio
async def test_deferred_generation_converges_before_the_worker_generates(
    generating_server: GeneratingServer, tmp_path: Path,
) -> None:
    """A deferring run exits 0 with its entries pending, and the server's worker then generates them."""
    await verify_deferred_generation(generating_server.base_url, generating_server.embedder, tmp_path)


@pytest.mark.asyncio
async def test_unreachable_server_exits_69(tmp_path: Path) -> None:
    """A server URL nothing listens at is a transient failure."""
    await verify_unreachable_server(tmp_path)


@requires_denying_modes
@pytest.mark.asyncio
async def test_unlistable_directory_withholds_every_delete(open_server: str, tmp_path: Path) -> None:
    """A directory that cannot be listed keeps its entries until a rerun can list it."""
    await verify_unlistable_directory(open_server, tmp_path)


@pytest.mark.asyncio
async def test_principals_manage_only_their_own_copy(jwt_server: str, jwt_key_pair: RSAKeyPair, tmp_path: Path) -> None:
    """Missing and forged tokens exit 2; a second principal copies a public corpus and leaves it untouched."""
    await verify_principals_manage_their_own_copy(jwt_server, jwt_key_pair, tmp_path)


@pytest.mark.asyncio
async def test_expired_token_is_retried_once_after_rotation(
    jwt_server: str, jwt_key_pair: RSAKeyPair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token that expires mid-run is refused once and the request is retried with the rotated token."""
    await verify_token_rotation(jwt_server, jwt_key_pair, tmp_path, monkeypatch)
