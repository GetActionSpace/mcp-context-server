"""Tests for the generation worker in the server lifespan (app/server.py).

The lifespan starts the worker as its last startup step, after the JSON-string middleware is
installed, handing it the settings and the request backend. On shutdown it stops the worker
before the backend and the providers shut down, so the worker gives its claims back while
the database is still reachable. A worker that fails to start fails the startup, which then
shuts the providers and the backend down as for any other startup failure.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.startup as app_startup


class _Lifespan:
    """Patches every phase of the SQLite lifespan and records, in order, each step it takes."""

    def __init__(self) -> None:
        self.steps: list[str] = []
        self.settings = MagicMock()
        self.settings.storage.backend_type = 'sqlite'
        self.settings.access_control.postgresql_rls_enabled = False
        self.settings.access_control.anonymous_principal = None
        self.settings.auth.provider = 'none'
        self.backend = MagicMock()
        self.backend.backend_type = 'sqlite'
        self.backend.initialize = AsyncMock()
        self.backend.shutdown = AsyncMock(side_effect=self.record('backend.shutdown'))
        self.providers = {name: self._provider(name) for name in ('reranking', 'embedding', 'summary')}
        self.start_generation_worker = AsyncMock(side_effect=self.record('start_generation_worker'))
        self.stop_generation_worker = AsyncMock(side_effect=self.record('stop_generation_worker'))

    def record(self, step: str) -> Callable[..., Awaitable[None]]:
        """Return a step that records its name when awaited."""

        async def _step(*_args: object, **_kwargs: object) -> None:
            self.steps.append(step)

        return _step

    def _provider(self, name: str) -> MagicMock:
        provider = MagicMock()
        provider.shutdown = AsyncMock(side_effect=self.record(f'{name}.shutdown'))
        return provider

    @contextmanager
    def patched(self) -> Generator[None]:
        """Install the patches, restoring the startup globals afterward."""
        original = (app_startup._backend, app_startup._repositories)
        try:
            with (
                patch('app.server.settings', self.settings),
                patch('app.server.create_backend', return_value=self.backend),
                patch('app.server.prepare_database', new=AsyncMock()),
                patch('app.server.RepositoryContainer', return_value=MagicMock()),
                patch('app.server.register_core_tools', side_effect=lambda *_args: self.steps.append('register_core_tools')),
                patch('app.server.initialize_providers', new=AsyncMock(side_effect=self.record('initialize_providers'))),
                patch('app.server.register_search_tools', new=AsyncMock(side_effect=self.record('register_search_tools'))),
                patch(
                    'app.server.install_json_string_middleware',
                    new=AsyncMock(side_effect=self.record('install_json_string_middleware')),
                ),
                patch('app.server.start_generation_worker', self.start_generation_worker),
                patch('app.server.stop_generation_worker', self.stop_generation_worker),
                patch('app.server.get_reranking_provider', return_value=self.providers['reranking']),
                patch('app.server.get_embedding_provider', return_value=self.providers['embedding']),
                patch('app.server.get_summary_provider', return_value=self.providers['summary']),
            ):
                yield
        finally:
            app_startup._backend, app_startup._repositories = original


@pytest.fixture
def lifespan_run() -> _Lifespan:
    """A lifespan whose phases are recorded."""
    return _Lifespan()


@pytest.mark.asyncio
async def test_the_worker_starts_after_the_middleware_install_with_the_request_backend(lifespan_run: _Lifespan) -> None:
    """The worker start is the last startup step and receives the settings and the request backend."""
    from app.server import lifespan

    with lifespan_run.patched():
        async with lifespan(MagicMock()):
            assert lifespan_run.steps == [
                'register_core_tools',
                'initialize_providers',
                'register_search_tools',
                'install_json_string_middleware',
                'start_generation_worker',
            ]

    lifespan_run.start_generation_worker.assert_awaited_once_with(lifespan_run.settings, lifespan_run.backend)


@pytest.mark.asyncio
async def test_the_worker_stops_before_the_backend_and_the_providers(lifespan_run: _Lifespan) -> None:
    """On shutdown the worker stops first, while the backend and the providers still run."""
    from app.server import lifespan

    with lifespan_run.patched():
        async with lifespan(MagicMock()):
            lifespan_run.steps.clear()

    assert lifespan_run.steps == [
        'stop_generation_worker',
        'backend.shutdown',
        'reranking.shutdown',
        'embedding.shutdown',
        'summary.shutdown',
    ]


@pytest.mark.asyncio
async def test_a_failed_worker_start_fails_the_startup_and_shuts_everything_down(lifespan_run: _Lifespan) -> None:
    """A worker that cannot start fails the startup, which shuts the providers and the backend down."""
    from app.server import lifespan

    lifespan_run.start_generation_worker.side_effect = RuntimeError('worker start failed')

    with lifespan_run.patched(), pytest.raises(RuntimeError, match='worker start failed'):
        async with lifespan(MagicMock()):
            pytest.fail('the lifespan must not yield after a failed start')

    assert lifespan_run.steps[-4:] == ['reranking.shutdown', 'embedding.shutdown', 'summary.shutdown', 'backend.shutdown']
    lifespan_run.stop_generation_worker.assert_not_awaited()
