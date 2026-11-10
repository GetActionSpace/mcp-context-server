"""The OpenAI and Anthropic summary providers retry the transient failures their APIs answer with.

Each test runs the real provider against a stand-in for the provider's HTTP API served from
the test process, so the chat model raises exactly the errors it raises against the real
service: a rate limit, a server error, then a normal answer.
"""

import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any
from typing import override
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from app.summary.retry import SummaryRetryExhaustedError

MAX_ATTEMPTS = 3
TEXT = 'A paragraph of a document that is long enough to be worth a summary. ' * 10
SUMMARY = 'A short summary.'

type Answer = tuple[int, dict[str, object]]

OPENAI_RATE_LIMIT: Answer = (
    429,
    {'error': {'message': 'Rate limit reached', 'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}},
)
OPENAI_SERVER_ERROR: Answer = (500, {'error': {'message': 'The server had an error', 'type': 'server_error', 'code': None}})
OPENAI_ANSWER: Answer = (
    200,
    {
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 1,
        'model': 'gpt-4o-mini',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': SUMMARY}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
    },
)
ANTHROPIC_RATE_LIMIT: Answer = (429, {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'Rate limited'}})
ANTHROPIC_OVERLOADED: Answer = (529, {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}})
ANTHROPIC_ANSWER: Answer = (
    200,
    {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-haiku-4-5',
        'content': [{'type': 'text', 'text': SUMMARY}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 10, 'output_tokens': 4},
    },
)


@dataclass
class ApiStub:
    """A stand-in for a provider API.

    Attributes:
        base_url: The URL the stand-in listens on, without a path.
        paths: The path of every POST request received, in order.
    """

    base_url: str
    paths: list[str] = field(default_factory=list)


@contextmanager
def serve_api(answers: list[Answer]) -> Generator[ApiStub]:
    """Answer POST requests with the scripted answers in order, the last one repeating.

    Args:
        answers: The status and JSON body of each answer.

    Yields:
        The stand-in, which records the path of each request.
    """
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            with lock:
                status, body = answers[min(len(stub.paths), len(answers) - 1)]
                stub.paths.append(self.path)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        @override
        def log_message(self, format: str, *args: Any) -> None:
            """Log nothing."""

    endpoint = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    stub = ApiStub(f'http://127.0.0.1:{endpoint.server_port}')
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        thread.join()


@pytest.fixture
def fast_retries() -> Generator[None]:
    """Give the retry wrapper three attempts with a negligible wait between them."""
    with patch('app.summary.retry.get_settings') as settings:
        settings.return_value.summary.timeout_s = 30.0
        settings.return_value.summary.retry_max_attempts = MAX_ATTEMPTS
        settings.return_value.summary.retry_base_delay_s = 0.01
        yield


@contextmanager
def openai_provider(base_url: str) -> Generator[Any]:
    """Yield an uninitialized OpenAI summary provider configured for the stand-in."""
    from app.summary.providers.langchain_openai import OpenAISummaryProvider

    with patch('app.summary.providers.langchain_openai.get_settings') as settings:
        settings.return_value.summary.model = 'gpt-4o-mini'
        settings.return_value.summary.max_tokens = 500
        settings.return_value.index_tree.max_tokens = 120
        settings.return_value.summary.openai_reasoning_effort = None
        settings.return_value.summary.openai_api_key = SecretStr('sk-test-key')
        settings.return_value.summary.openai_api_base = f'{base_url}/v1'
        yield OpenAISummaryProvider()


@contextmanager
def anthropic_provider(base_url: str, monkeypatch: pytest.MonkeyPatch) -> Generator[Any]:
    """Yield an uninitialized Anthropic summary provider whose client talks to the stand-in."""
    from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

    monkeypatch.setenv('ANTHROPIC_API_URL', base_url)
    monkeypatch.setenv('ANTHROPIC_BASE_URL', base_url)
    with patch('app.summary.providers.langchain_anthropic.get_settings') as settings:
        settings.return_value.summary.model = 'claude-haiku-4-5'
        settings.return_value.summary.max_tokens = 500
        settings.return_value.index_tree.max_tokens = 120
        settings.return_value.summary.anthropic_effort = None
        settings.return_value.summary.anthropic_api_key = SecretStr('sk-ant-test-key')
        yield AnthropicSummaryProvider()


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_retries')
async def test_openai_summary_survives_a_rate_limit_and_a_server_error() -> None:
    """A 429 and a 500 from the OpenAI API are retried, and the third attempt's answer is the summary."""
    answers = [OPENAI_RATE_LIMIT, OPENAI_SERVER_ERROR, OPENAI_ANSWER]
    with serve_api(answers) as stub, openai_provider(stub.base_url) as provider:
        await provider.initialize()
        summary = await provider.summarize(TEXT, 'agent')

    assert summary == SUMMARY
    assert stub.paths == ['/v1/chat/completions'] * 3


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_retries')
async def test_openai_summary_gives_up_after_the_configured_attempts() -> None:
    """A rate limit on every attempt spends exactly the configured attempts, then reports exhaustion."""
    with serve_api([OPENAI_RATE_LIMIT]) as stub, openai_provider(stub.base_url) as provider:
        await provider.initialize()
        with pytest.raises(SummaryRetryExhaustedError):
            await provider.summarize(TEXT, 'agent')

    assert len(stub.paths) == MAX_ATTEMPTS


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_retries')
async def test_openai_node_summary_survives_a_rate_limit() -> None:
    """The section-summary call shares the retry: a 429 is retried there too."""
    with serve_api([OPENAI_RATE_LIMIT, OPENAI_ANSWER]) as stub, openai_provider(stub.base_url) as provider:
        await provider.initialize()
        summary = await provider.summarize_with_prompt(TEXT, 'Summarize the section in one sentence.')

    assert summary == SUMMARY
    assert len(stub.paths) == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_retries')
async def test_anthropic_summary_survives_a_rate_limit_and_an_overload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 and a 529 from the Anthropic API are retried, and the third attempt's answer is the summary."""
    answers = [ANTHROPIC_RATE_LIMIT, ANTHROPIC_OVERLOADED, ANTHROPIC_ANSWER]
    with serve_api(answers) as stub, anthropic_provider(stub.base_url, monkeypatch) as provider:
        await provider.initialize()
        summary = await provider.summarize(TEXT, 'agent')

    assert summary == SUMMARY
    assert stub.paths == ['/v1/messages'] * 3


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_retries')
async def test_anthropic_summary_gives_up_after_the_configured_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """An overload on every attempt spends exactly the configured attempts, then reports exhaustion."""
    with serve_api([ANTHROPIC_OVERLOADED]) as stub, anthropic_provider(stub.base_url, monkeypatch) as provider:
        await provider.initialize()
        with pytest.raises(SummaryRetryExhaustedError):
            await provider.summarize(TEXT, 'agent')

    assert len(stub.paths) == MAX_ATTEMPTS
