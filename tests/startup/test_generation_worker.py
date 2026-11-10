"""Tests for the generation worker startup phase (app/startup/generation_worker.py) and its registry.

The worker starts only when an embedding or a summary provider is active. On SQLite and on
PostgreSQL with row-level security off it runs on the process's own backend; on PostgreSQL
with row-level security on it runs on a maintenance backend it opens and keeps open until
it stops. Every worker runs as the maintenance identity with the concurrency of the
embedding provider, or of the summary provider when no embedding provider is active.
``app.startup`` registers the running worker and wakes it on request.
"""

from collections.abc import Generator
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.startup.generation_worker as phase
import app.tools._generation as generation_module
from app.backends.base import MAINTENANCE
from app.repositories import RepositoryContainer
from app.startup import get_generation_worker
from app.startup import notify_generation_worker
from app.startup import set_generation_worker
from app.startup.generation_worker import start_generation_worker
from app.startup.generation_worker import stop_generation_worker


class FakeWorker:
    """Records how the phase built, started and stopped it.

    Attributes:
        instances: Every worker built since the test began.
        start_error: What every start raises, if anything.
    """

    instances: ClassVar[list['FakeWorker']] = []
    start_error: ClassVar[Exception | None] = None

    def __init__(self, repos: RepositoryContainer, *, access: object, concurrency: int, max_attempts: int) -> None:
        self.repos = repos
        self.access = access
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.calls: list[str] = []
        FakeWorker.instances.append(self)

    async def start(self) -> None:
        """Record the start, or fail it."""
        self.calls.append('start')
        if FakeWorker.start_error is not None:
            raise FakeWorker.start_error

    async def stop(self) -> None:
        """Record the stop."""
        self.calls.append('stop')

    def notify(self) -> None:
        """Record the notification."""
        self.calls.append('notify')


def _settings(*, rls: bool = False) -> MagicMock:
    """Settings with distinct concurrency values and attempt cap."""
    settings = MagicMock()
    settings.embedding.max_concurrent = 3
    settings.summary.max_concurrent = 2
    settings.generation_jobs.max_attempts = 4
    settings.access_control.postgresql_rls_enabled = rls
    return settings


def _backend(backend_type: str) -> MagicMock:
    """A backend of the given type whose lifecycle calls are awaitable."""
    backend = MagicMock()
    backend.backend_type = backend_type
    backend.initialize = AsyncMock()
    backend.shutdown = AsyncMock()
    return backend


@pytest.fixture(autouse=True)
def isolated_worker() -> Generator[None]:
    """Start every test without a registered worker and leave none behind."""
    FakeWorker.instances = []
    FakeWorker.start_error = None
    set_generation_worker(None)
    with patch.object(phase, 'GenerationWorker', FakeWorker):
        yield
    set_generation_worker(None)


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> dict[str, object | None]:
    """Make both providers active; a test switches one off by setting its entry to None."""
    active: dict[str, object | None] = {'embedding': object(), 'summary': object()}
    monkeypatch.setattr(generation_module, 'get_embedding_provider', lambda: active['embedding'])
    monkeypatch.setattr(generation_module, 'get_summary_provider', lambda: active['summary'])
    monkeypatch.setattr(phase, 'get_embedding_provider', lambda: active['embedding'])
    return active


class TestGate:
    """The worker starts only when a generation leg can run."""

    @pytest.mark.asyncio
    async def test_no_provider_starts_no_worker(self, providers: dict[str, object | None]) -> None:
        """Without an embedding or a summary provider nothing is started or registered."""
        providers['embedding'] = providers['summary'] = None

        await start_generation_worker(_settings(), _backend('sqlite'))

        assert FakeWorker.instances == []
        assert get_generation_worker() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('embedding', 'concurrency'),
        [pytest.param(True, 3, id='embedding-provider'), pytest.param(False, 2, id='summary-provider-only')],
    )
    async def test_the_concurrency_follows_the_active_provider(
        self, providers: dict[str, object | None], embedding: bool, concurrency: int,
    ) -> None:
        """The embedding provider's concurrency applies when it is active, else the summary provider's."""
        if not embedding:
            providers['embedding'] = None

        await start_generation_worker(_settings(), _backend('sqlite'))

        (worker,) = FakeWorker.instances
        assert (worker.concurrency, worker.max_attempts) == (concurrency, 4)


@pytest.mark.usefixtures('providers')
class TestBackendChoice:
    """The worker runs on the process backend, or on a maintenance backend with row-level security on."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('backend_type', 'rls'),
        [pytest.param('sqlite', False, id='sqlite'), pytest.param('postgresql', False, id='postgresql-rls-off')],
    )
    async def test_the_worker_runs_on_the_process_backend(self, backend_type: str, rls: bool) -> None:
        """No backend is opened: the worker's repositories run on the backend the server uses."""
        backend = _backend(backend_type)
        create_maintenance = MagicMock()

        with patch.object(phase, 'create_maintenance_backend', create_maintenance):
            await start_generation_worker(_settings(rls=rls), backend)
            await stop_generation_worker()

        (worker,) = FakeWorker.instances
        assert worker.repos.context.backend is backend
        assert worker.access is MAINTENANCE
        assert worker.calls == ['start', 'stop']
        create_maintenance.assert_not_called()
        backend.shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_with_row_level_security_the_worker_keeps_a_maintenance_backend_open(self) -> None:
        """The worker runs on an opened maintenance backend, which the stop shuts down after the worker."""
        steps: list[str] = []
        maintenance = _backend('postgresql')
        maintenance.initialize.side_effect = lambda: steps.append('initialize')
        maintenance.shutdown.side_effect = lambda: steps.append('shutdown')

        with patch.object(phase, 'create_maintenance_backend', return_value=maintenance):
            await start_generation_worker(_settings(rls=True), _backend('postgresql'))

        (worker,) = FakeWorker.instances
        assert worker.repos.context.backend is maintenance
        assert worker.access is MAINTENANCE
        registered: object = get_generation_worker()
        assert registered is worker
        assert steps == ['initialize']

        await stop_generation_worker()

        assert worker.calls == ['start', 'stop']
        assert steps == ['initialize', 'shutdown']
        assert get_generation_worker() is None

    @pytest.mark.asyncio
    async def test_a_failed_start_shuts_the_maintenance_backend_down(self) -> None:
        """A start that fails leaves no backend open and no worker registered."""
        maintenance = _backend('postgresql')
        FakeWorker.start_error = RuntimeError('reset failed')

        with (
            patch.object(phase, 'create_maintenance_backend', return_value=maintenance),
            pytest.raises(RuntimeError, match='reset failed'),
        ):
            await start_generation_worker(_settings(rls=True), _backend('postgresql'))

        maintenance.shutdown.assert_awaited_once()
        assert get_generation_worker() is None

        await stop_generation_worker()

        maintenance.shutdown.assert_awaited_once()


class TestRegistry:
    """The startup package holds the running worker and wakes it on request."""

    @pytest.mark.asyncio
    async def test_stop_without_a_worker_does_nothing(self) -> None:
        """Stopping when no worker started is a no-op."""
        await stop_generation_worker()

        assert get_generation_worker() is None

    def test_notify_without_a_worker_does_nothing(self) -> None:
        """A notification with no worker running is dropped."""
        notify_generation_worker()

        assert get_generation_worker() is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('providers')
    async def test_notify_wakes_the_running_worker(self) -> None:
        """A notification reaches the registered worker."""
        await start_generation_worker(_settings(), _backend('sqlite'))

        notify_generation_worker()

        (worker,) = FakeWorker.instances
        assert worker.calls == ['start', 'notify']
