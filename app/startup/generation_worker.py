"""Startup phase that starts the generation worker, and the stop that ends it.

The worker runs deferred generation jobs as the maintenance identity: a job was authorized
when the write that enqueued it ran, and its outputs are written for whichever entry holds
it. On SQLite it runs on the process's own backend, so its writes pass through the one
write queue; on PostgreSQL with row-level security off it runs on the request backend, which
binds no identity; on PostgreSQL with row-level security on it runs on a maintenance backend
logged in as the table owner, opened here and kept open until the worker stops.

The worker starts only when an embedding or a summary provider is active: a process that
can generate nothing never claims a job. Node summaries need the summary provider too.
"""

import logging

from app.backends import StorageBackend
from app.backends import create_maintenance_backend
from app.backends.base import MAINTENANCE
from app.generation_jobs.worker import GenerationWorker
from app.repositories import RepositoryContainer
from app.settings import AppSettings
from app.startup import get_embedding_provider
from app.startup import get_generation_worker
from app.startup import get_summary_provider
from app.startup import set_generation_worker

logger = logging.getLogger(__name__)

# The maintenance backend the running worker opened, shut down when the worker stops.
_dedicated_backend: StorageBackend | None = None


async def start_generation_worker(settings: AppSettings, backend: StorageBackend) -> None:
    """Start the generation worker and register it, when a generation provider is active.

    Args:
        settings: The application settings.
        backend: The backend the server's requests run on.
    """
    global _dedicated_backend
    embedding_active = get_embedding_provider() is not None
    if not embedding_active and get_summary_provider() is None:
        logger.info('Generation worker not started: neither an embedding nor a summary provider is active')
        return

    dedicated: StorageBackend | None = None
    if backend.backend_type == 'postgresql' and settings.access_control.postgresql_rls_enabled:
        dedicated = create_maintenance_backend()
    worker_backend = dedicated or backend
    concurrency = settings.embedding.max_concurrent if embedding_active else settings.summary.max_concurrent
    worker = GenerationWorker(
        RepositoryContainer(worker_backend),
        access=MAINTENANCE,
        concurrency=concurrency,
        max_attempts=settings.generation_jobs.max_attempts,
    )
    try:
        if dedicated is not None:
            await dedicated.initialize()
        await worker.start()
    except Exception:
        if dedicated is not None:
            await _shut_down(dedicated)
        raise
    _dedicated_backend = dedicated
    set_generation_worker(worker)


async def stop_generation_worker() -> None:
    """Stop the registered worker and shut down the backend it opened; nothing happens without one.

    A failure to shut the backend down is logged, never raised, so the rest of the server
    shutdown proceeds.
    """
    global _dedicated_backend
    worker = get_generation_worker()
    dedicated, _dedicated_backend = _dedicated_backend, None
    set_generation_worker(None)
    if worker is not None:
        await worker.stop()
    if dedicated is not None:
        await _shut_down(dedicated)


async def _shut_down(backend: StorageBackend) -> None:
    """Shut down the worker's maintenance backend, logging a failure instead of raising it."""
    try:
        await backend.shutdown()
    except Exception as error:
        logger.error(f'Error shutting down the generation worker backend: {error}')
