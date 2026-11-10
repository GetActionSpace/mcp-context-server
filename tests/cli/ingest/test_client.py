"""Tests for the ingest client's transport: the status table, transport errors, the recorder and the preflight."""

from typing import override

import httpx2
import pytest

from app.cli.ingest import client as client_module
from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_TRANSIENT
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import StatusRecorder
from app.cli.ingest.client import ToolFailure
from app.cli.ingest.client import http_client_factory
from tests.cli.ingest._stand_in import LIMITS
from tests.cli.ingest._stand_in import PLAN_ARGUMENTS
from tests.cli.ingest._stand_in import PLAN_RESULT
from tests.cli.ingest._stand_in import PLAN_TOOL
from tests.cli.ingest._stand_in import REFUSED_PLAN
from tests.cli.ingest._stand_in import SERVER_URL
from tests.cli.ingest._stand_in import Exchange
from tests.cli.ingest._stand_in import context_server
from tests.cli.ingest._stand_in import delayed
from tests.cli.ingest._stand_in import exit_of_plan_call
from tests.cli.ingest._stand_in import failing
from tests.cli.ingest._stand_in import ingest_client
from tests.cli.ingest._stand_in import serving
from tests.cli.ingest._stand_in import status
from tests.cli.ingest._stand_in import stream_ending


class TestHttpClientFactory:
    """The factory builds the client the transport talks through, with the status-recording hook."""

    @pytest.mark.asyncio
    async def test_factory_takes_the_transport_keywords_and_keeps_redirects_off(self) -> None:
        """The transport passes follow_redirects; the client leaves redirects to the transport and applies the timeouts."""
        factory = http_client_factory(StatusRecorder(), read_timeout=12.5)

        async with factory(headers={'X-Extra': '1'}, auth=None, follow_redirects=True) as client:
            assert client.follow_redirects is False
            assert client.timeout == httpx2.Timeout(30, read=12.5)
            assert client.headers['X-Extra'] == '1'
        explicit = httpx2.Timeout(5, read=7)
        async with factory(timeout=explicit) as client:
            assert client.timeout == explicit

    @pytest.mark.asyncio
    async def test_recorder_keeps_the_last_status_and_retry_after_until_reset(self) -> None:
        """Every response's status and Retry-After header are recorded, and a reset clears them."""
        recorder = StatusRecorder()

        async def answer(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(429, headers={'Retry-After': '12'}, request=request)

        factory = http_client_factory(recorder, read_timeout=5.0, transport=httpx2.MockTransport(answer))
        async with factory() as client:
            await client.post(SERVER_URL, json={})

        assert (recorder.last_status, recorder.retry_after) == (429, '12')
        recorder.reset()
        assert (recorder.last_status, recorder.retry_after) == (None, None)


class TestCalls:
    """A successful call returns the structured result; a tool's own error is a per-call failure."""

    @pytest.mark.asyncio
    async def test_call_returns_the_structured_result(self, exchange: Exchange) -> None:
        """The client connects, passes the preflight and returns the tool's structured content."""
        async with ingest_client(exchange) as client:
            await client.preflight()
            result = await client.call_tool(PLAN_TOOL, {'thread_id': 'kb:handbook', 'manifest': [{'source_id': 'a.md'}]})

        assert result == {**PLAN_RESULT, 'creates': [0]}

    @pytest.mark.asyncio
    async def test_tool_error_is_a_tool_failure(self, exchange: Exchange) -> None:
        """A ToolError raised by the tool reaches the caller as a ToolFailure carrying its text."""
        async with ingest_client(exchange) as client:
            result = await client.call_tool(PLAN_TOOL, {'thread_id': 'kb:refused', 'manifest': []})

        assert result == ToolFailure(REFUSED_PLAN)

    @pytest.mark.asyncio
    async def test_recorder_class_is_read_from_the_module_at_construction(
        self, exchange: Exchange, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A recorder subclass set on the module is the one the client's HTTP hook calls."""
        statuses: list[int] = []

        class ListeningRecorder(StatusRecorder):
            @override
            async def on_response(self, response: httpx2.Response) -> None:
                statuses.append(response.status_code)
                await super().on_response(response)

        monkeypatch.setattr(client_module, 'StatusRecorder', ListeningRecorder)

        async with ingest_client(exchange) as client:
            await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)

        assert isinstance(client.recorder, ListeningRecorder)
        assert statuses
        assert set(statuses) == {200}

    @pytest.mark.asyncio
    async def test_recorder_is_reset_before_every_call(self, exchange: Exchange, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each call starts from an empty record, so a status never outlives the call that received it."""
        events: list[str] = []

        class TracingRecorder(StatusRecorder):
            @override
            def reset(self) -> None:
                events.append('reset')
                super().reset()

            @override
            async def on_response(self, response: httpx2.Response) -> None:
                events.append(str(response.status_code))
                await super().on_response(response)

        monkeypatch.setattr(client_module, 'StatusRecorder', TracingRecorder)
        exchange.script(PLAN_TOOL, status(413))

        async with ingest_client(exchange) as client:
            events.clear()
            refused = await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)
            answered = await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)

        assert isinstance(refused, ToolFailure)
        assert answered == PLAN_RESULT
        assert events[:3] == ['reset', '413', 'reset']
        assert set(events[3:]) == {'200'}


class TestStatusTable:
    """Each HTTP status a server or proxy answers a call with maps to its exit code and message."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('code', 'exit_code', 'message'),
        [
            (401, EXIT_CONFIG, 'the server requires a bearer token (HTTP 401): pass --token-file or --token-env'),
            (403, EXIT_CONFIG, 'the server requires a bearer token (HTTP 403): pass --token-file or --token-env'),
            (
                404,
                EXIT_CONFIG,
                (
                    'the server answered HTTP 404 at http://127.0.0.1:8765/mcp: '
                    '--server must name the MCP endpoint, such as http://localhost:8000/mcp'
                ),
            ),
            (429, EXIT_TRANSIENT, 'the server is limiting requests (HTTP 429): rerun later'),
            (502, EXIT_TRANSIENT, 'the server or a proxy in front of it is unavailable (HTTP 502): rerun later'),
            (503, EXIT_TRANSIENT, 'the server or a proxy in front of it is unavailable (HTTP 503): rerun later'),
            (
                504,
                EXIT_TRANSIENT,
                (
                    'a proxy in front of the server timed out (HTTP 504): raise the proxy read timeout '
                    f'to at least --call-timeout, or lower --batch-size; see {LIMITS}'
                ),
            ),
            (
                500,
                EXIT_CONFIG,
                'the call to plan_context_sync failed with HTTP 500: Server returned an error response',
            ),
        ],
    )
    async def test_status_on_a_call_ends_the_run(
        self, exchange: Exchange, code: int, exit_code: int, message: str,
    ) -> None:
        """A refused call raises the exit its status calls for, with the exact message."""
        exchange.script(PLAN_TOOL, status(code))

        async with ingest_client(exchange) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (exit_code, message)

    @pytest.mark.asyncio
    async def test_a_200_that_is_no_mcp_response_ends_the_run(self, exchange: Exchange) -> None:
        """A web page answered with 200 at the --server URL is a configuration error, not a lost stream."""

        async def web_page(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, headers={'content-type': 'text/html'}, text='<html></html>', request=request)

        exchange.script(PLAN_TOOL, web_page)

        async with ingest_client(exchange) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (
            EXIT_CONFIG,
            'the call to plan_context_sync failed: Unexpected content type: text/html',
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool', [PLAN_TOOL, 'store_context_batch'])
    async def test_413_is_a_per_call_failure_naming_the_budget_knobs(self, exchange: Exchange, tool: str) -> None:
        """An oversized request is a failure of that call, naming the flags, the proxy limit and the guide's Limits."""
        exchange.script(tool, status(413))

        async with ingest_client(exchange) as client:
            result = await client.call_tool(tool, {'entries': []} if tool != PLAN_TOOL else PLAN_ARGUMENTS)

        assert result == ToolFailure(
            'the request was too large for the server or a proxy in front of it (HTTP 413): '
            f'lower --max-request-bytes or --batch-size, or raise the proxy body limit; see {LIMITS}',
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('code', 'retry_after', 'message'),
        [
            (429, '30', 'the server is limiting requests (HTTP 429): rerun later; Retry-After: 30'),
            (
                503,
                'Wed, 21 Oct 2026 07:28:00 GMT',
                (
                    'the server or a proxy in front of it is unavailable (HTTP 503): rerun later; '
                    'Retry-After: Wed, 21 Oct 2026 07:28:00 GMT'
                ),
            ),
        ],
    )
    async def test_retry_after_is_reported(
        self, exchange: Exchange, code: int, retry_after: str, message: str,
    ) -> None:
        """A Retry-After header on a transient refusal is printed verbatim."""
        exchange.script(PLAN_TOOL, status(code, headers={'Retry-After': retry_after}))

        async with ingest_client(exchange) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (EXIT_TRANSIENT, message)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('code', 'exit_code', 'message'),
        [
            (
                404,
                EXIT_CONFIG,
                (
                    'the server answered HTTP 404 at http://127.0.0.1:8765/mcp: '
                    '--server must name the MCP endpoint, such as http://localhost:8000/mcp'
                ),
            ),
            (503, EXIT_TRANSIENT, 'the server or a proxy in front of it is unavailable (HTTP 503): rerun later'),
        ],
    )
    async def test_status_while_connecting_ends_the_run(
        self, exchange: Exchange, code: int, exit_code: int, message: str,
    ) -> None:
        """A status answered to the protocol negotiation is classified like one answered to a call."""
        exchange.every_request = status(code)

        with pytest.raises(IngestExitError) as raised:
            async with ingest_client(exchange):
                pass

        assert (raised.value.code, raised.value.message) == (exit_code, message)

    @pytest.mark.asyncio
    async def test_unanswered_error_is_transient(self, exchange: Exchange, monkeypatch: pytest.MonkeyPatch) -> None:
        """An MCP error with no recorded status means no server answered: exit 69."""

        class DeafRecorder(StatusRecorder):
            @override
            async def on_response(self, response: httpx2.Response) -> None:
                return None

        monkeypatch.setattr(client_module, 'StatusRecorder', DeafRecorder)
        exchange.script(PLAN_TOOL, status(503))

        async with ingest_client(exchange) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (
            EXIT_TRANSIENT,
            'the call to plan_context_sync failed before the server answered: Server returned an error response',
        )


class TestTransportErrors:
    """A server that cannot be reached, does not answer in time or drops the connection is a transient failure."""

    @pytest.mark.asyncio
    async def test_connect_error_while_connecting(self, exchange: Exchange) -> None:
        """A refused connection before the session exists exits 69."""
        exchange.every_request = failing(httpx2.ConnectError)

        with pytest.raises(IngestExitError) as raised:
            async with ingest_client(exchange):
                pass

        assert (raised.value.code, raised.value.message) == (
            EXIT_TRANSIENT,
            'cannot reach http://127.0.0.1:8765/mcp: scripted failure',
        )

    @pytest.mark.asyncio
    async def test_connect_error_on_a_call(self, exchange: Exchange) -> None:
        """A connection refused in the middle of a run exits 69."""
        exchange.script(PLAN_TOOL, failing(httpx2.ConnectError))

        async with ingest_client(exchange) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (EXIT_TRANSIENT, 'cannot reach http://127.0.0.1:8765/mcp: scripted failure')

    @pytest.mark.asyncio
    async def test_http_timeout_on_a_call(self, exchange: Exchange) -> None:
        """An HTTP timeout exits 69 and names --call-timeout."""
        exchange.script(PLAN_TOOL, failing(httpx2.ReadTimeout))

        async with ingest_client(exchange) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (
            EXIT_TRANSIENT,
            (
                'the call to plan_context_sync timed out talking to http://127.0.0.1:8765/mcp (ReadTimeout): '
                'rerun the sync, or raise --call-timeout for batches that take longer'
            ),
        )

    @pytest.mark.asyncio
    async def test_call_timeout_of_the_client(self, exchange: Exchange) -> None:
        """A call the server does not answer within --call-timeout exits 69."""
        exchange.script(PLAN_TOOL, delayed(exchange, 5.0))

        async with ingest_client(exchange, call_timeout=1.0) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (
            EXIT_TRANSIENT,
            (
                'the call to plan_context_sync got no answer within --call-timeout (1 s): '
                'rerun the sync, which plans from what the server stored'
            ),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error', [httpx2.ReadError, httpx2.WriteError, httpx2.CloseError, httpx2.RemoteProtocolError],
    )
    @pytest.mark.parametrize(('tool', 'arguments'), [(PLAN_TOOL, PLAN_ARGUMENTS), ('store_context_batch', {'entries': []})])
    async def test_a_connection_lost_mid_call_is_transient(
        self, exchange: Exchange, error: type[httpx2.TransportError], tool: str, arguments: dict[str, object],
    ) -> None:
        """A connection the server or a proxy drops during a call exits 69: the call's outcome is unknown."""
        exchange.script(tool, failing(error))

        async with ingest_client(exchange) as client:
            with pytest.raises(IngestExitError) as raised:
                await client.call_tool(tool, arguments)

        assert (raised.value.code, raised.value.message) == (
            EXIT_TRANSIENT,
            (
                f'the call to {tool} lost its connection to http://127.0.0.1:8765/mcp ({error.__name__}): '
                "the call's outcome is unknown; rerun the sync, which plans from what the server stored"
            ),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('error', [httpx2.ReadError, httpx2.RemoteProtocolError, None])
    @pytest.mark.parametrize(('tool', 'arguments'), [(PLAN_TOOL, PLAN_ARGUMENTS), ('store_context_batch', {'entries': []})])
    async def test_a_response_stream_ending_before_the_result_is_transient(
        self, exchange: Exchange, error: type[httpx2.TransportError] | None, tool: str, arguments: dict[str, object],
    ) -> None:
        """A call the server accepted whose SSE stream ends before the result exits 69: the call's outcome is unknown."""
        exchange.script(tool, stream_ending(error))

        async with ingest_client(exchange) as client:
            with pytest.raises(IngestExitError) as raised:
                await client.call_tool(tool, arguments)

        assert (raised.value.code, raised.value.message) == (
            EXIT_TRANSIENT,
            (
                f'the call to {tool} lost its response stream from http://127.0.0.1:8765/mcp after the server accepted it '
                "(SSE stream ended without a response): the call's outcome is unknown; rerun the sync, which plans "
                'from what the server stored; when calls keep losing their stream after the same time, raise the '
                f'response timeout of the proxy in front of the server to at least --call-timeout; see {LIMITS}'
            ),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('error', [httpx2.LocalProtocolError, httpx2.ProxyError])
    async def test_a_client_side_transport_error_is_not_transient(
        self, exchange: Exchange, error: type[httpx2.TransportError],
    ) -> None:
        """A request the client itself could not form, or a proxy the client could not use, is no transient failure."""
        exchange.script(PLAN_TOOL, failing(error))

        async with ingest_client(exchange) as client:
            with pytest.raises(error):
                await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)


class TestPreflight:
    """The preflight refuses a server that lacks a sync tool or would merge what a sync inserts."""

    @pytest.mark.asyncio
    async def test_server_without_sync_tools(self) -> None:
        """A server with ENABLE_CONTEXT_SYNC=false lacks plan_context_sync, and the refusal says so."""
        async with serving(context_server(sync=False)) as served, ingest_client(served) as client:
            with pytest.raises(IngestExitError) as raised:
                await client.preflight()

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            (
                'the server does not offer plan_context_sync, which a sync needs; '
                'a server offers plan_context_sync unless ENABLE_CONTEXT_SYNC=false'
            ),
        )

    @pytest.mark.asyncio
    async def test_server_without_batch_tools(self) -> None:
        """Every missing tool is named."""
        async with serving(context_server(batch=False)) as served, ingest_client(served) as client:
            with pytest.raises(IngestExitError) as raised:
                await client.preflight()

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            (
                'the server does not offer store_context_batch, update_context_batch, delete_context_batch, '
                'which a sync needs'
            ),
        )

    @pytest.mark.asyncio
    async def test_server_without_the_deduplicate_parameter(self) -> None:
        """A server whose store_context lacks deduplicate would silently merge, so the run stops."""
        async with serving(context_server(deduplicate=False)) as served, ingest_client(served) as client:
            with pytest.raises(IngestExitError) as raised:
                await client.preflight()

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            (
                "the server's store_context has no deduplicate parameter, so it would merge the documents "
                'a sync inserts: upgrade the server'
            ),
        )

    @pytest.mark.asyncio
    async def test_a_deferring_run_needs_the_defer_generation_parameter(self) -> None:
        """A server whose store_context cannot defer generation stops a deferring run."""
        async with serving(context_server(defer_generation=False)) as served, ingest_client(served) as client:
            with pytest.raises(IngestExitError) as raised:
                await client.preflight(defer_generation=True)

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            (
                "the server's store_context has no defer_generation parameter, so it cannot defer generation: "
                'upgrade the server or run without --defer-generation'
            ),
        )

    @pytest.mark.asyncio
    async def test_a_deferring_run_passes_on_a_server_that_defers(self, exchange: Exchange) -> None:
        """A server whose store_context has the parameter passes a deferring run's preflight."""
        async with ingest_client(exchange) as client:
            await client.preflight(defer_generation=True)

    @pytest.mark.asyncio
    async def test_a_run_without_deferral_needs_no_defer_generation_parameter(self) -> None:
        """Only a deferring run checks the parameter, so a server that always generates still serves a plain run."""
        async with serving(context_server(defer_generation=False)) as served, ingest_client(served) as client:
            await client.preflight()
