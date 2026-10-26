"""Summary-provider runtime setup for the migration CLI.

The ``mcp-context-server-migrate`` CLI runs as a standalone process and does NOT execute
the server lifespan (``app/server.py``), so the global summary provider -- read by the
generation pipeline through ``app.startup.get_summary_provider`` -- is unset by default.
Completing the generation jobs a deferred write left outstanding (``--embed-missing``,
``--re-embed``) generates their summaries and index-tree node summaries, so the CLI
registers the configured provider first, as the server lifespan does: without it the jobs
would complete without the summaries the server would have generated.
"""

import contextlib
import logging

from app.settings import get_settings
from app.startup import set_summary_provider
from app.summary.base import SummaryProvider
from app.summary.factory import create_summary_provider

logger = logging.getLogger(__name__)


class SummaryProviderUnavailableError(RuntimeError):
    """Raised when summary generation is enabled and its provider cannot be used.

    Signals an operator-actionable condition (provider extra not installed, service down,
    misconfiguration) so the CLI can exit with a clear message and a non-zero status.
    """


async def initialize_cli_summary_provider() -> SummaryProvider | None:
    """Create and register the summary provider when summary generation is enabled.

    Mirrors the summary provider initialization of the server lifespan
    (``app/startup/providers.py``): create the provider through the configured
    ``SUMMARY_PROVIDER`` factory, initialize it, verify that it is available, and register
    it through :func:`app.startup.set_summary_provider`.

    Returns:
        The registered provider, or None when ``ENABLE_SUMMARY_GENERATION`` is false, in
        which case no provider is registered. The caller owns the provider's lifetime and
        passes it to :func:`shutdown_cli_summary_provider` when done.

    Raises:
        SummaryProviderUnavailableError: If the provider extra is not installed or the
            provider reports that it is not available.
    """
    settings = get_settings()
    if not settings.summary.generation_enabled:
        set_summary_provider(None)
        return None

    try:
        provider = create_summary_provider()
        await provider.initialize()
    except ImportError as exc:
        raise SummaryProviderUnavailableError(
            f'Summary provider import failed: {exc}. Install the provider extra '
            f'(e.g. uv sync --extra summary-{settings.summary.provider}) and re-run, '
            'or set ENABLE_SUMMARY_GENERATION=false.',
        ) from exc

    if not await provider.is_available():
        with contextlib.suppress(Exception):
            await provider.shutdown()
        raise SummaryProviderUnavailableError(
            f'Summary provider {provider.provider_name!r} is not available. Ensure the summary '
            'service is running and reachable and that SUMMARY_PROVIDER / SUMMARY_MODEL are '
            'configured correctly, or set ENABLE_SUMMARY_GENERATION=false.',
        )

    set_summary_provider(provider)
    logger.info('CLI summary provider ready: provider=%s model=%s', provider.provider_name, settings.summary.model)
    return provider


async def shutdown_cli_summary_provider(provider: SummaryProvider | None) -> None:
    """Unregister the summary provider and shut it down.

    Shutdown failures are suppressed so teardown never masks the primary result.

    Args:
        provider: The provider :func:`initialize_cli_summary_provider` returned, or None.
    """
    set_summary_provider(None)
    if provider is not None:
        with contextlib.suppress(Exception):
            await provider.shutdown()


__all__ = [
    'SummaryProviderUnavailableError',
    'initialize_cli_summary_provider',
    'shutdown_cli_summary_provider',
]
