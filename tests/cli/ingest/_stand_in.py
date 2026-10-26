"""A stand-in context server the ingest client's tests talk Streamable HTTP to.

The stand-in is a FastMCP server offering the tools a sync checks and calls,
served in process through ``httpx2.MockTransport``. The transport's handler passes
each request to the server's ASGI app unless a test scripted another answer for
it, so the client meets the HTTP statuses and transport errors a server or a proxy
in front of it produces, in both the negotiation and the tool calls.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import override

import httpx2
import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from app.cli.ingest.client import IngestClient
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import TokenFileAuth
from app.cli.ingest.client import build_auth

SERVER_URL = 'http://127.0.0.1:8765/mcp'
PLAN_TOOL = 'plan_context_sync'
PLAN_ARGUMENTS: dict[str, object] = {'thread_id': 'kb:handbook', 'manifest': []}
CONTEXT_ID = '0190abcdef1234567890abcdef123456'
TOKEN_A = 'ingest-token-a-5f1c2e'
TOKEN_B = 'ingest-token-b-93d07a'
TOKEN_VARIABLE = 'INGEST_TEST_TOKEN'
LIMITS = "docs/knowledge-base.md, 'Limits'"
PLAN_RESULT: dict[str, object] = {
    'thread_id': 'kb:handbook',
    'creates': [],
    'updates': [],
    'deletes': [],
    'deletes_total': 0,
    'unchanged_count': 0,
    'kept_count': 0,
    'foreign_count': 0,
}
REFUSED_PLAN = 'manifest item 1 repeats source_id a.md'

type Responder = Callable[[httpx2.Request], Awaitable[httpx2.Response]]


def context_server(
    *, sync: bool = True, batch: bool = True, deduplicate: bool = True, defer_generation: bool = True,
) -> FastMCP:
    """Return a stand-in context server offering the tools a sync checks and calls.

    Returns:
        The server. Without ``deduplicate`` its store_context merges and cannot defer
        generation either; without ``defer_generation`` it inserts without merging but
        always generates at once.
    """
    server = FastMCP('context-server')

    def store_context(
        thread_id: str, source: str, text: str, deduplicate: bool = True, defer_generation: bool = False,
    ) -> dict[str, object]:
        """Store one entry."""
        return {
            'context_id': CONTEXT_ID,
            'thread_id': thread_id,
            'source': source,
            'length': len(text),
            'deduplicate': deduplicate,
            'generation_status': 'pending' if defer_generation else 'ready',
        }

    def store_context_generating(thread_id: str, source: str, text: str, deduplicate: bool = True) -> dict[str, object]:
        """Store one entry, generating its embeddings and summary before it returns."""
        return {
            'context_id': CONTEXT_ID,
            'thread_id': thread_id,
            'source': source,
            'length': len(text),
            'deduplicate': deduplicate,
        }

    def store_context_merging(thread_id: str, source: str, text: str) -> dict[str, object]:
        """Store one entry, merging it into an identical latest entry."""
        return {'context_id': CONTEXT_ID, 'thread_id': thread_id, 'source': source, 'length': len(text)}

    def store_context_batch(entries: list[dict[str, object]], atomic: bool = True) -> dict[str, object]:
        """Store entries."""
        return {'success': True, 'total': len(entries), 'succeeded': len(entries), 'failed': 0, 'atomic': atomic}

    def update_context_batch(updates: list[dict[str, object]], atomic: bool = True) -> dict[str, object]:
        """Update entries."""
        return {'success': True, 'total': len(updates), 'succeeded': len(updates), 'failed': 0, 'atomic': atomic}

    def delete_context_batch(context_ids: list[str]) -> dict[str, object]:
        """Delete entries."""
        return {'success': True, 'deleted_count': len(context_ids)}

    def plan_context_sync(thread_id: str, manifest: list[dict[str, object]]) -> dict[str, object]:
        """Plan a sync of a corpus."""
        if thread_id == 'kb:refused':
            raise ToolError(REFUSED_PLAN)
        return {**PLAN_RESULT, 'thread_id': thread_id, 'creates': list(range(len(manifest)))}

    if not deduplicate:
        server.tool(store_context_merging, name='store_context')
    elif not defer_generation:
        server.tool(store_context_generating, name='store_context')
    else:
        server.tool(store_context)
    if batch:
        server.tool(store_context_batch)
        server.tool(update_context_batch)
        server.tool(delete_context_batch)
    if sync:
        server.tool(plan_context_sync)
    return server


@dataclass
class Seen:
    """One request the stand-in server received."""

    method: str | None
    tool: str | None
    authorization: str | None


@dataclass
class Exchange:
    """The HTTP side of a stand-in server: records every request and answers the scripted ones itself."""

    asgi: httpx2.ASGITransport
    scripts: dict[str, list[Responder]] = field(default_factory=dict)
    every_request: Responder | None = None
    seen: list[Seen] = field(default_factory=list)

    def script(self, tool: str, *responders: Responder) -> None:
        """Answer the next calls of a tool with the given responders, in order."""
        self.scripts.setdefault(tool, []).extend(responders)

    def calls_of(self, tool: str) -> list[Seen]:
        """Return the requests that called a tool."""
        return [seen for seen in self.seen if seen.tool == tool]

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Record a request, then answer it from its script or pass it to the server."""
        message = json.loads(request.content) if request.content else {}
        method = message.get('method')
        tool = (message.get('params') or {}).get('name') if method == 'tools/call' else None
        self.seen.append(Seen(method=method, tool=tool, authorization=request.headers.get('Authorization')))
        if self.every_request is not None:
            return await self.every_request(request)
        queue = self.scripts.get(tool) if tool is not None else None
        if queue:
            return await queue.pop(0)(request)
        return await self.asgi.handle_async_request(request)


def status(code: int, *, headers: dict[str, str] | None = None) -> Responder:
    """Return a responder answering with an HTTP status and a non-JSON-RPC body, as a proxy or auth layer does."""

    async def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(code, json={'error': 'refused'}, headers=headers, request=request)

    return respond


def failing(error: type[httpx2.TransportError]) -> Responder:
    """Return a responder raising a transport error instead of answering."""

    async def respond(request: httpx2.Request) -> httpx2.Response:
        raise error('scripted failure', request=request)

    return respond


class _EndingStream(httpx2.AsyncByteStream):
    """An SSE body that sends one keep-alive comment and ends before any message."""

    def __init__(self, request: httpx2.Request, error: type[httpx2.TransportError] | None) -> None:
        self._request = request
        self._error = error

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b': ping\r\n\r\n'
        if self._error is not None:
            raise self._error('scripted failure', request=self._request)


def stream_ending(error: type[httpx2.TransportError] | None) -> Responder:
    """Return a responder that accepts a call with an SSE stream that ends before the call's result.

    Args:
        error: The transport error the stream breaks off with, as it does when the
            server restarts or a proxy drops the connection; None ends it cleanly.

    Returns:
        The responder.
    """

    async def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={'content-type': 'text/event-stream'},
            stream=_EndingStream(request, error),
            request=request,
        )

    return respond


def rotating(token_file: Path, token: str, code: int = 401) -> Responder:
    """Return a responder that refuses the request after a refresher wrote a new token to the file."""

    async def respond(request: httpx2.Request) -> httpx2.Response:
        await asyncio.to_thread(token_file.write_text, token + '\n', encoding='utf-8')
        return httpx2.Response(code, json={'error': 'invalid_token'}, request=request)

    return respond


def delayed(exchange: Exchange, seconds: float) -> Responder:
    """Return a responder that passes the request to the server after a delay."""

    async def respond(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(seconds)
        return await exchange.asgi.handle_async_request(request)

    return respond


@asynccontextmanager
async def serving(server: FastMCP) -> AsyncGenerator[Exchange]:
    """Run a stand-in server's HTTP app for the duration of a test.

    The app's lifespan opens a task group that only the task which opened it may
    close, while pytest-asyncio sets a fixture up and tears it down in different
    tasks, so the lifespan runs in a task of its own.

    Args:
        server: The stand-in server.

    Yields:
        The exchange that serves the app's requests.
    """
    app = server.http_app(path='/mcp', stateless_http=True)
    started = asyncio.Event()
    stopping = asyncio.Event()

    async def run() -> None:
        async with app.lifespan(app):
            started.set()
            await stopping.wait()

    lifespan = asyncio.create_task(run())
    ready = asyncio.create_task(started.wait())
    await asyncio.wait({lifespan, ready}, return_when=asyncio.FIRST_COMPLETED)
    if lifespan.done():
        ready.cancel()
        lifespan.result()
    try:
        yield Exchange(asgi=httpx2.ASGITransport(app=app))
    finally:
        stopping.set()
        await lifespan


def ingest_client(
    exchange: Exchange,
    *,
    auth: httpx2.Auth | None = None,
    call_timeout: float = 30.0,
    server: str = SERVER_URL,
) -> IngestClient:
    """Return an ingest client whose requests reach the stand-in server."""
    return IngestClient(server, auth=auth, call_timeout=call_timeout, transport=httpx2.MockTransport(exchange.handle))


def write_token_file(tmp_path: Path, token: str) -> Path:
    """Write a token file the way a refresher does, with a trailing newline."""
    path = tmp_path / 'token'
    path.write_text(token + '\n', encoding='utf-8')
    return path


def file_auth(path: Path) -> TokenFileAuth:
    """Build the auth for a token file through the command's own builder."""
    auth = build_auth(token_file=path, token_env=None)
    assert isinstance(auth, TokenFileAuth)
    return auth


async def exit_of_plan_call(client: IngestClient) -> IngestExitError:
    """Call the plan tool and return the exit it ends the run with."""
    with pytest.raises(IngestExitError) as raised:
        await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)
    return raised.value
