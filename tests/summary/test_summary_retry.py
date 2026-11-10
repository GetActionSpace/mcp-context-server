"""Tests for the summary retry wrapper using tenacity.

Tests verify:
- Successful first attempt (no retry needed)
- Retry on transient errors (ConnectionError, OSError)
- Timeout triggers retry
- All retries exhausted raises SummaryRetryExhaustedError
- All timeouts exhausted raises SummaryTimeoutError
- Non-retryable errors are not retried
- Operation name appears in error messages
- compute_summary_total_timeout() returns correct value
"""

import asyncio
import re
import warnings
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from app.summary.retry import SummaryRetryExhaustedError
from app.summary.retry import SummaryTimeoutError
from app.summary.retry import compute_summary_total_timeout
from app.summary.retry import with_summary_retry_and_timeout


@pytest.fixture
def mock_summary_settings():
    """Mock summary settings for fast tests."""
    with patch('app.summary.retry.get_settings') as mock:
        mock.return_value.summary.timeout_s = 1.0
        mock.return_value.summary.retry_max_attempts = 5
        mock.return_value.summary.retry_base_delay_s = 0.01  # Fast retries for testing
        yield mock


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_successful_first_attempt() -> None:
    """Test successful execution on first attempt - no retry needed."""
    mock_func = AsyncMock(return_value='This is a summary.')

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'This is a summary.'
    assert mock_func.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_retry_on_connection_error() -> None:
    """Test retry on transient ConnectionError."""
    mock_func = AsyncMock(side_effect=[ConnectionError('Network error'), 'Summary text'])

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Summary text'
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_retry_on_os_error() -> None:
    """Test retry on transient OSError."""
    mock_func = AsyncMock(side_effect=[OSError('IO error'), 'Summary text'])

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Summary text'
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_timeout_triggers_retry() -> None:
    """Test that timeout triggers retry and eventually succeeds."""
    call_count = 0

    async def slow_then_fast() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(10)  # Will timeout (settings have 1.0s timeout)
        return 'Summary text'

    result = await with_summary_retry_and_timeout(slow_then_fast, 'test_operation')

    assert result == 'Summary text'
    assert call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_exhausted_retries_raises_error() -> None:
    """Test all retries exhausted raises SummaryRetryExhaustedError."""
    mock_func = AsyncMock(side_effect=ConnectionError('Network error'))

    with pytest.raises(SummaryRetryExhaustedError) as exc_info:
        await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert 'failed after 5 attempts' in str(exc_info.value)
    assert mock_func.call_count == 5


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_timeout_exhausted_raises_timeout_error() -> None:
    """Test all timeouts exhausted raises SummaryTimeoutError."""

    async def always_slow() -> str:
        await asyncio.sleep(10)  # Will always timeout
        return 'Never reached'

    with pytest.raises(SummaryTimeoutError) as exc_info:
        await with_summary_retry_and_timeout(always_slow, 'test_operation')

    assert 'timed out after 1.0s' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_non_retryable_error_not_retried() -> None:
    """Test that non-retryable errors are not retried."""
    mock_func = AsyncMock(side_effect=ValueError('Bad input'))

    with pytest.raises(ValueError, match='Bad input'):
        await with_summary_retry_and_timeout(mock_func, 'test_operation')

    # ValueError is not in retry list, so only 1 attempt
    assert mock_func.call_count == 1


def _provider_error(module: str, name: str, *bases: type[Exception]) -> type[Exception]:
    """Return an exception class that reports the given module and name, as a provider package's class does."""
    return type(name, bases or (Exception,), {'__module__': module})


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
@pytest.mark.parametrize(
    ('module', 'name'),
    [
        ('openai', 'RateLimitError'),
        ('openai', 'InternalServerError'),
        ('openai._exceptions', 'APITimeoutError'),
        ('anthropic', 'RateLimitError'),
        ('anthropic', 'OverloadedError'),
        ('anthropic._exceptions', 'APIConnectionError'),
        ('ollama', 'ResponseError'),
    ],
)
async def test_retry_on_a_transient_provider_error(module: str, name: str) -> None:
    """A provider package's transient error is retried, whichever of its modules defines the class."""
    error = _provider_error(module, name)
    mock_func = AsyncMock(side_effect=[error('transient'), 'Summary text'])

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Summary text'
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_retry_on_an_integration_subclass_of_a_transient_provider_error() -> None:
    """An integration's own subclass of a transient provider error is retried like the error it extends."""
    sdk_error = _provider_error('openai', 'RateLimitError')
    wrapped = _provider_error('langchain_openai.chat_models.base', 'OpenAIRateLimitError', sdk_error)
    mock_func = AsyncMock(side_effect=[wrapped('rate limited'), 'Summary text'])

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Summary text'
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_a_permanent_provider_error_and_its_subclasses_are_not_retried() -> None:
    """A provider error outside the transient set is raised at once, and so is an integration's subclass of it."""
    sdk_error = _provider_error('openai', 'BadRequestError')
    wrapped = _provider_error('langchain_openai.chat_models.base', 'OpenAIInvalidRequestError', sdk_error)

    for error in (sdk_error, wrapped):
        mock_func = AsyncMock(side_effect=error('bad request'))
        with pytest.raises(error, match='bad request'):
            await with_summary_retry_and_timeout(mock_func, 'test_operation')
        assert mock_func.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_operation_name_in_error_message() -> None:
    """Test that operation name appears in error messages."""
    mock_func = AsyncMock(side_effect=ConnectionError('Network error'))

    with pytest.raises(SummaryRetryExhaustedError) as exc_info:
        await with_summary_retry_and_timeout(mock_func, 'ollama_summarize')

    assert 'ollama_summarize' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_operation_name_in_timeout_message() -> None:
    """Test that operation name appears in timeout messages."""

    async def always_slow() -> str:
        await asyncio.sleep(10)
        return 'Never reached'

    with pytest.raises(SummaryTimeoutError) as exc_info:
        await with_summary_retry_and_timeout(always_slow, 'openai_summarize')

    assert 'openai_summarize' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_multiple_failures_then_success() -> None:
    """Test recovery after multiple failures."""
    mock_func = AsyncMock(
        side_effect=[
            ConnectionError('Attempt 1 failed'),
            OSError('Attempt 2 failed'),
            'Final summary text',  # Attempt 3 succeeds
        ],
    )

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Final summary text'
    assert mock_func.call_count == 3


def test_compute_summary_total_timeout() -> None:
    """Test compute_summary_total_timeout() returns a positive value."""
    with patch('app.summary.retry.get_settings') as mock:
        mock.return_value.summary.timeout_s = 240.0
        mock.return_value.summary.retry_max_attempts = 5
        mock.return_value.summary.retry_base_delay_s = 1.0

        total = compute_summary_total_timeout()

        # Formula: (max_attempts * timeout_s + total_backoff) * 1.1
        # = (5 * 240 + (min(1*1+1, 60) + min(1*2+1, 60) + min(1*4+1, 60) + min(1*8+1, 60))) * 1.1
        # = (1200 + (2 + 3 + 5 + 9)) * 1.1
        # = 1219 * 1.1
        # = 1340.9
        assert total > 0
        assert total == pytest.approx(1340.9)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_retry_log_shows_operation_name(caplog: pytest.LogCaptureFixture) -> None:
    """Verify retry log message shows operation name, not <unknown>."""
    call_count = 0

    async def flaky_func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError('transient failure')
        return 'summary result'

    with caplog.at_level('WARNING'):
        result = await with_summary_retry_and_timeout(flaky_func, 'test_summarize_operation')

    assert result == 'summary result'
    assert call_count == 2

    retry_messages = [r.message for r in caplog.records if 'Retrying' in r.message]
    assert len(retry_messages) >= 1
    assert 'test_summarize_operation' in retry_messages[0]
    assert '<unknown>' not in retry_messages[0]


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_retry_emits_no_deprecation_warning() -> None:
    """Building and running the retry loop uses only current tenacity parameters."""
    mock_func = AsyncMock(side_effect=[ConnectionError('Network error'), 'summary'])

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'summary'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_first_retry_wait_spans_base_delay_jitter(caplog: pytest.LogCaptureFixture) -> None:
    """The first backoff is the base delay plus a jitter spread across one more base delay."""
    base_delay_s = 0.01
    waits: list[float] = []
    for _ in range(20):
        caplog.clear()
        mock_func = AsyncMock(side_effect=[ConnectionError('Network error'), 'summary'])
        with caplog.at_level('WARNING'):
            await with_summary_retry_and_timeout(mock_func, 'test_operation')
        messages = [r.message for r in caplog.records if r.message.startswith('Retrying test_operation in ')]
        assert len(messages) == 1
        match = re.search(r' in ([0-9.e-]+) seconds ', messages[0])
        assert match is not None
        waits.append(float(match.group(1)))

    assert all(base_delay_s - 1e-4 <= wait <= 2 * base_delay_s + 1e-4 for wait in waits)
    assert max(waits) - min(waits) > 0.3 * base_delay_s
