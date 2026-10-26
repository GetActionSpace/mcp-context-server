"""An Ollama embedding endpoint the test process serves, so a live server generates embeddings without a model.

A server started with ``OLLAMA_HOST`` set to the stub's URL, ``EMBEDDING_PROVIDER=ollama``
and ``EMBEDDING_DIM`` equal to the stub's dimension passes its Ollama dependency check,
its embedding availability check and its prewarm, and embeds every text it is given at
once, so a test can watch generation complete in seconds. The stub answers what those
checks and the embedding calls ask: ``GET /`` (the service is running), ``POST /api/show``
(every model exists) and ``POST /api/embed``; it records every text it embedded and
answers any other request with 404.
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


@dataclass
class OllamaStub:
    """A running stub endpoint.

    Attributes:
        dimension: The length of every vector the endpoint returns.
        url: The endpoint's origin, the value a server's ``OLLAMA_HOST`` takes.
        inputs: Every text the endpoint embedded, in the order the requests arrived.
    """

    dimension: int
    url: str = ''
    inputs: list[str] = field(default_factory=list[str])


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
        """Describe any model, or embed the request's input, a text or a list of texts."""
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        if self.path == SHOW_PATH:
            self._answer(json.dumps({'model_info': {}, 'capabilities': ['embedding']}).encode('utf-8'))
            return
        if self.path != EMBED_PATH:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        texts = [body['input']] if isinstance(body['input'], str) else list(body['input'])
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
def serve_ollama_embeddings(dimension: int) -> Generator[OllamaStub]:
    """Serve the stub on a loopback port in a daemon thread until the block exits.

    Args:
        dimension: The length of every vector the stub returns; a server using the stub
            sets ``EMBEDDING_DIM`` to it.

    Yields:
        The running stub.
    """
    stub = OllamaStub(dimension=dimension)
    endpoint = ThreadingHTTPServer(('127.0.0.1', 0), partial(_EmbedHandler, stub=stub))
    stub.url = f'http://127.0.0.1:{endpoint.server_port}'
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        thread.join(timeout=10)
