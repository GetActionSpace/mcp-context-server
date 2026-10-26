"""An Ollama embedding endpoint the test process serves, so a live server generates embeddings without a model.

A server started with the overrides of :func:`stub_generation_env` (``OLLAMA_HOST`` set to
the stub's URL, ``EMBEDDING_PROVIDER=ollama`` and ``EMBEDDING_DIM`` equal to the stub's
dimension) passes its Ollama dependency check, its embedding availability check and its
prewarm, and embeds every text it is given at once, so a test can watch generation
complete in seconds. The stub answers what those checks and the embedding calls ask:
``GET /`` (the service is running), ``POST /api/show`` (every model exists) and
``POST /api/embed``; it records every text it embedded and answers any other request
with 404.

Two modes let a test decide when and whether generation succeeds:

- ``refuse_embeddings`` answers every embed request with HTTP 500, except the texts a
  server embeds while it starts (:data:`PROBE_TEXTS`), so the server starts with an
  active embedding provider whose every document embedding then fails;
- ``hold_marker`` keeps every embed request that carries the marker in one of its texts
  waiting until :meth:`OllamaStub.release` is called, so the entries whose text carries
  it stay pending for exactly as long as the test needs, while every other text is
  embedded at once.
"""

import json
import math
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any
from typing import override

EMBED_PATH = '/api/embed'
SHOW_PATH = '/api/show'
# The model a server embedding through the stub names; the stub serves every model name.
STUB_EMBEDDING_MODEL = 'stub-embedding'
# The texts a server embeds while it starts: the embedding availability check embeds
# ``test`` and the startup prewarm embeds ``warmup``.
PROBE_TEXTS = frozenset({'test', 'warmup'})
# The longest a held embed request waits for its release, so a check that fails before
# it releases the stub never leaves a server's generation waiting until its own timeout.
HOLD_TIMEOUT_S = 120.0


@dataclass
class OllamaStub:
    """A running stub endpoint.

    Attributes:
        dimension: The length of every vector the endpoint returns.
        refuse_embeddings: Answer every embed request with HTTP 500, except one that
            embeds only :data:`PROBE_TEXTS`.
        hold_marker: Keep every embed request one of whose texts contains this string
            waiting until :meth:`release`; None holds nothing.
        url: The endpoint's origin, the value a server's ``OLLAMA_HOST`` takes.
        inputs: Every text the endpoint embedded, in the order the answers were sent.
        released: Set once :meth:`release` was called.
    """

    dimension: int
    refuse_embeddings: bool = False
    hold_marker: str | None = None
    url: str = ''
    inputs: list[str] = field(default_factory=list[str])
    released: threading.Event = field(default_factory=threading.Event)

    def release(self) -> None:
        """Answer every held embed request, and every later one carrying the marker, at once."""
        self.released.set()


def stub_generation_env(stub: OllamaStub) -> dict[str, str]:
    """Return the environment overrides of a server that generates its embeddings through the stub.

    Args:
        stub: The running stub.

    Returns:
        The overrides: embedding generation on, the Ollama provider at the stub's URL, and
        the stub's model name and dimension.
    """
    return {
        'ENABLE_EMBEDDING_GENERATION': 'true',
        'EMBEDDING_PROVIDER': 'ollama',
        'EMBEDDING_MODEL': STUB_EMBEDDING_MODEL,
        'EMBEDDING_DIM': str(stub.dimension),
        'OLLAMA_HOST': stub.url,
    }


class _EmbedHandler(BaseHTTPRequestHandler):
    """Answer the service check, the model check and embed requests, one fixed unit vector per input text."""

    def __init__(self, *args: Any, stub: OllamaStub, **kwargs: Any) -> None:
        self.stub = stub
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Say that the service is running."""
        if self.path != '/':
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._answer(b'Ollama is running', 'text/plain')

    def do_POST(self) -> None:
        """Describe any model, or embed the request's input, a text or a list of texts, as the stub's mode allows."""
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        if self.path == SHOW_PATH:
            self._answer(json.dumps({'model_info': {}, 'capabilities': ['embedding']}).encode('utf-8'))
            return
        if self.path != EMBED_PATH:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        texts = [body['input']] if isinstance(body['input'], str) else list(body['input'])
        if self.stub.refuse_embeddings and not PROBE_TEXTS.issuperset(texts):
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, 'embedding refused by the test endpoint')
            return
        marker = self.stub.hold_marker
        if marker is not None and any(marker in text for text in texts):
            self.stub.released.wait(HOLD_TIMEOUT_S)
        self.stub.inputs.extend(texts)
        vector = [1.0 / math.sqrt(self.stub.dimension)] * self.stub.dimension
        self._answer(json.dumps({'model': body['model'], 'embeddings': [vector for _ in texts]}).encode('utf-8'))

    def _answer(self, payload: bytes, content_type: str = 'application/json') -> None:
        """Send a 200 response with a body."""
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Log nothing."""


@contextmanager
def serve_ollama_embeddings(
    dimension: int, *, refuse_embeddings: bool = False, hold_marker: str | None = None,
) -> Generator[OllamaStub]:
    """Serve the stub on a loopback port in a daemon thread until the block exits.

    Leaving the block releases every held request before the endpoint stops.

    Args:
        dimension: The length of every vector the stub returns; a server using the stub
            sets ``EMBEDDING_DIM`` to it.
        refuse_embeddings: Answer every embed request but the startup ones with HTTP 500.
        hold_marker: Hold every embed request carrying this string until the stub is released.

    Yields:
        The running stub.
    """
    stub = OllamaStub(dimension=dimension, refuse_embeddings=refuse_embeddings, hold_marker=hold_marker)
    endpoint = ThreadingHTTPServer(('127.0.0.1', 0), partial(_EmbedHandler, stub=stub))
    stub.url = f'http://127.0.0.1:{endpoint.server_port}'
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        stub.release()
        endpoint.shutdown()
        endpoint.server_close()
        thread.join(timeout=10)
