"""How long the real HTTP server keeps an idle kept-alive connection open (SQLite backend).

The server is the actual ``main()`` started as an HTTP server through the helpers of
:mod:`tests.integration._http_jwt`, from the MCP SDK's default environment plus explicit
overrides. A raw socket sends ``GET /health`` over HTTP/1.1, reads the complete
response, and then stays idle: with ``HTTP_KEEP_ALIVE_TIMEOUT_S=2`` the server closes the
connection about two seconds later, and with the default the connection still serves a
second request after seven seconds, longer than uvicorn's own default of five.
"""

import socket
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.integration._http_jwt import build_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import running_http_server

pytestmark = pytest.mark.integration

HEALTH_REQUEST = b'GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n'
SHORT_TIMEOUT_S = 2
DEFAULT_IDLE_PROBE_S = 7.0


def _server_address(base_url: str) -> tuple[str, int]:
    """Split a server origin into the host and port a socket connects to.

    Args:
        base_url: The server origin, e.g. ``http://127.0.0.1:8123``.

    Returns:
        The host and the port.
    """
    host, port = base_url.removeprefix('http://').rsplit(':', 1)
    return host, int(port)


def _read_response(sock: socket.socket) -> tuple[bytes, bytes]:
    """Read one complete HTTP response from ``sock``, failing the test if the connection closes first.

    Args:
        sock: A connected socket whose request has been sent.

    Returns:
        The status line with the headers, and the body.
    """
    data = b''
    while b'\r\n\r\n' not in data:
        chunk = sock.recv(4096)
        assert chunk, f'connection closed before the response headers arrived: {data!r}'
        data += chunk
    head, body = data.split(b'\r\n\r\n', 1)
    length = next(
        int(line.split(b':', 1)[1])
        for line in head.split(b'\r\n')[1:]
        if line.lower().startswith(b'content-length:')
    )
    while len(body) < length:
        chunk = sock.recv(4096)
        assert chunk, f'connection closed before the response body arrived: {body!r}'
        body += chunk
    return head, body


def _get_health(sock: socket.socket) -> None:
    """Send ``GET /health`` on ``sock`` and check that the server answers it and keeps the connection.

    Args:
        sock: A connected socket.
    """
    sock.sendall(HEALTH_REQUEST)
    head, body = _read_response(sock)
    assert head.startswith(b'HTTP/1.1 200'), f'unexpected /health response: {head!r}'
    assert body == b'{"status":"ok"}', f'unexpected /health body: {body!r}'
    assert b'connection: close' not in head.lower(), f'the server did not keep the connection alive: {head!r}'


@pytest.fixture
def short_timeout_server(tmp_path: Path) -> Iterator[str]:
    """Launch a real HTTP server with a two-second keep-alive timeout; yield its base URL.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    env = {
        **build_env(backend='sqlite', database=str(tmp_path / 'keep_alive_short.db'), port=free_port(), auth=False),
        'HTTP_KEEP_ALIVE_TIMEOUT_S': str(SHORT_TIMEOUT_S),
    }
    with running_http_server(env) as base_url:
        yield base_url


@pytest.fixture
def default_timeout_server(tmp_path: Path) -> Iterator[str]:
    """Launch a real HTTP server with the default keep-alive timeout; yield its base URL.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    env = build_env(backend='sqlite', database=str(tmp_path / 'keep_alive_default.db'), port=free_port(), auth=False)
    assert 'HTTP_KEEP_ALIVE_TIMEOUT_S' not in env
    with running_http_server(env) as base_url:
        yield base_url


def test_idle_connection_closed_after_configured_timeout(short_timeout_server: str) -> None:
    """An idle connection is closed by the server once HTTP_KEEP_ALIVE_TIMEOUT_S has passed."""
    with socket.create_connection(_server_address(short_timeout_server), timeout=10.0) as sock:
        _get_health(sock)
        idle_since = time.monotonic()
        tail = sock.recv(1)
        idle_s = time.monotonic() - idle_since

    assert tail == b'', f'expected end of stream, got {tail!r}'
    assert SHORT_TIMEOUT_S - 0.5 <= idle_s <= SHORT_TIMEOUT_S + 2.0, (
        f'the server closed the idle connection after {idle_s:.2f}s, expected about {SHORT_TIMEOUT_S}s'
    )


def test_idle_connection_kept_open_past_uvicorn_default(default_timeout_server: str) -> None:
    """With the default timeout, a connection idle for seven seconds still serves a request."""
    with socket.create_connection(_server_address(default_timeout_server), timeout=10.0) as sock:
        _get_health(sock)
        time.sleep(DEFAULT_IDLE_PROBE_S)
        try:
            _get_health(sock)
        except ConnectionError as exc:
            pytest.fail(f'the server closed the connection within {DEFAULT_IDLE_PROBE_S}s of idling: {exc!r}')
