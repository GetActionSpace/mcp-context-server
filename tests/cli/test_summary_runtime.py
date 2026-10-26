"""Tests for the migration CLI's summary-provider setup in app/cli/_summary_runtime.py.

The CLI registers the configured summary provider the way the server lifespan does when
summary generation is enabled, refuses a provider that cannot be imported or is not
available, and clears the registration again when it is done.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.cli._summary_runtime import SummaryProviderUnavailableError
from app.cli._summary_runtime import initialize_cli_summary_provider
from app.cli._summary_runtime import shutdown_cli_summary_provider
from app.settings import get_settings
from app.startup import get_summary_provider
from app.startup import set_summary_provider

_FACTORY = 'app.cli._summary_runtime.create_summary_provider'


@pytest.fixture(autouse=True)
def no_registered_provider() -> Generator[None]:
    """Start and end every test with no summary provider registered."""
    set_summary_provider(None)
    yield
    set_summary_provider(None)


def _summary_generation(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    """Set ENABLE_SUMMARY_GENERATION and rebuild the settings."""
    monkeypatch.setenv('ENABLE_SUMMARY_GENERATION', 'true' if enabled else 'false')
    get_settings.cache_clear()


def _provider(*, available: bool = True) -> MagicMock:
    """Return a summary provider double that reports ``available``."""
    provider = MagicMock()
    provider.provider_name = 'fake'
    provider.initialize = AsyncMock()
    provider.is_available = AsyncMock(return_value=available)
    provider.shutdown = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_disabled_summary_generation_registers_no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ENABLE_SUMMARY_GENERATION=false no provider is created and none is registered."""
    _summary_generation(monkeypatch, enabled=False)
    set_summary_provider(_provider())

    with patch(_FACTORY) as factory:
        assert await initialize_cli_summary_provider() is None

    factory.assert_not_called()
    assert get_summary_provider() is None


@pytest.mark.asyncio
async def test_an_available_provider_is_initialized_and_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """An enabled, available provider is initialized, registered and returned."""
    _summary_generation(monkeypatch, enabled=True)
    provider = _provider()

    with patch(_FACTORY, return_value=provider):
        assert await initialize_cli_summary_provider() is provider

    provider.initialize.assert_awaited_once()
    provider.shutdown.assert_not_awaited()
    assert get_summary_provider() is provider


@pytest.mark.asyncio
async def test_an_unavailable_provider_is_shut_down_and_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider that reports itself unavailable is shut down, not registered, and refused."""
    _summary_generation(monkeypatch, enabled=True)
    provider = _provider(available=False)

    with patch(_FACTORY, return_value=provider), pytest.raises(SummaryProviderUnavailableError) as refused:
        await initialize_cli_summary_provider()

    provider.shutdown.assert_awaited_once()
    assert get_summary_provider() is None
    assert "'fake' is not available" in str(refused.value)
    assert 'ENABLE_SUMMARY_GENERATION=false' in str(refused.value)


@pytest.mark.asyncio
async def test_a_missing_provider_extra_is_refused_with_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider whose extra is not installed is refused with the extra to install."""
    _summary_generation(monkeypatch, enabled=True)
    monkeypatch.setenv('SUMMARY_PROVIDER', 'openai')
    get_settings.cache_clear()

    with (
        patch(_FACTORY, side_effect=ImportError('No module named langchain_openai')),
        pytest.raises(SummaryProviderUnavailableError) as refused,
    ):
        await initialize_cli_summary_provider()

    assert 'No module named langchain_openai' in str(refused.value)
    assert 'uv sync --extra summary-openai' in str(refused.value)
    assert get_summary_provider() is None


@pytest.mark.asyncio
async def test_shutdown_clears_the_registration_and_shuts_the_provider_down() -> None:
    """Shutting down unregisters the provider and shuts it down."""
    provider = _provider()
    set_summary_provider(provider)

    await shutdown_cli_summary_provider(provider)

    provider.shutdown.assert_awaited_once()
    assert get_summary_provider() is None


@pytest.mark.asyncio
async def test_shutdown_tolerates_no_provider_and_a_failing_shutdown() -> None:
    """A provider that was never created, or whose shutdown fails, still ends unregistered."""
    await shutdown_cli_summary_provider(None)

    provider = _provider()
    provider.shutdown = AsyncMock(side_effect=RuntimeError('connection reset'))
    set_summary_provider(provider)

    await shutdown_cli_summary_provider(provider)

    assert get_summary_provider() is None
