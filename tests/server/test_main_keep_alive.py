"""Tests for the HTTP keep-alive timeout that main() hands to the uvicorn server.

Each HTTP-family transport passes ``HTTP_KEEP_ALIVE_TIMEOUT_S`` to ``mcp.run`` as
``uvicorn_config={'timeout_keep_alive': ...}``; the stdio transport starts no HTTP
server and passes no uvicorn configuration.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.server as server_module
from app.settings import get_settings


def _run_main(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> dict[str, object]:
    """Run main() under ``env`` with FastMCP mocked and return the keyword arguments of ``mcp.run``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        env: Environment variables to set before the settings are rebuilt.

    Returns:
        The keyword arguments main() passed to ``mcp.run``.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    monkeypatch.setattr(server_module, 'settings', get_settings())

    mock_fastmcp_cls = MagicMock()
    mock_instance = mock_fastmcp_cls.return_value
    mock_instance.run = MagicMock(side_effect=SystemExit(0))
    mock_instance.custom_route = MagicMock(return_value=lambda f: f)

    with (
        patch('app.server.FastMCP', mock_fastmcp_cls),
        patch('app.server.create_auth_provider', return_value=None),
        pytest.raises(SystemExit),
    ):
        server_module.main()

    mock_instance.run.assert_called_once()
    return dict(mock_instance.run.call_args.kwargs)


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    """Rebuild the settings singleton after each test, so no test's environment leaks into the next.

    Yields:
        Nothing; the cache is cleared after the test.
    """
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize('transport', ['http', 'streamable-http', 'sse'])
def test_http_transports_pass_configured_keep_alive_timeout(monkeypatch: pytest.MonkeyPatch, transport: str) -> None:
    """Every HTTP-family transport hands the configured timeout to uvicorn."""
    run_kwargs = _run_main(monkeypatch, {'MCP_TRANSPORT': transport, 'HTTP_KEEP_ALIVE_TIMEOUT_S': '42'})

    assert run_kwargs['transport'] == transport
    assert run_kwargs['uvicorn_config'] == {'timeout_keep_alive': 42}


def test_http_transport_passes_default_keep_alive_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """With HTTP_KEEP_ALIVE_TIMEOUT_S unset, uvicorn gets the 75-second default."""
    monkeypatch.delenv('HTTP_KEEP_ALIVE_TIMEOUT_S', raising=False)
    run_kwargs = _run_main(monkeypatch, {'MCP_TRANSPORT': 'http'})

    assert run_kwargs['uvicorn_config'] == {'timeout_keep_alive': 75}


def test_stdio_transport_passes_no_uvicorn_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stdio transport starts no HTTP server, so it passes no uvicorn configuration."""
    run_kwargs = _run_main(monkeypatch, {'MCP_TRANSPORT': 'stdio', 'HTTP_KEEP_ALIVE_TIMEOUT_S': '42'})

    assert run_kwargs['transport'] == 'stdio'
    assert 'uvicorn_config' not in run_kwargs
