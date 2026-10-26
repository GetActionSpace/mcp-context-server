"""The ingest client's session with a context server: transport, bearer-token auth, failure classes and preflight.

The client speaks MCP over Streamable HTTP. Its bearer token comes from a file read
before every request, so a refresher can rotate the file during a long run, or once
from an environment variable the operator names; a token never travels on the
command line.

The MCP transport reports every refused HTTP request as the same opaque JSON-RPC
error, so the client records the status and ``Retry-After`` header of each response
with an HTTP event hook and classifies a failed exchange by what it recorded: a
refused token or a wrong endpoint ends the run with exit code 2, an overloaded or
unreachable server, or a connection or response stream lost during a call, with 69,
and a request too large for the server or a proxy in front of it is a failure of
that one call. A 401 after the server accepted the token earlier in the run is
retried once when the token file now holds another token, which is safe because
the server's authentication refuses a request before any tool runs.
"""

import os
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from types import TracebackType
from typing import Protocol
from typing import Self
from typing import override
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import httpx2
from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.client.transports import StreamableHttpTransport
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED
from mcp.types import REQUEST_TIMEOUT
from mcp.types import TextContent

EXIT_CONVERGED = 0
EXIT_PARTIAL = 1
EXIT_CONFIG = 2
EXIT_TRANSIENT = 69

# The tools a sync calls. store_context is listed for its input schema: a server whose
# store_context has a deduplicate or defer_generation parameter honors the per-entry key
# of that name in the batch tools.
SYNC_TOOLS = ('plan_context_sync', 'store_context', 'store_context_batch', 'update_context_batch', 'delete_context_batch')
DEDUPLICATE_PARAMETER = 'deduplicate'
DEFER_GENERATION_PARAMETER = 'defer_generation'
LIMITS_SECTION = "docs/knowledge-base.md, 'Limits'"
# Connect, write and pool timeout, the MCP SDK's own default.
_CONNECT_TIMEOUT_SECONDS = 30.0
_ENDPOINT_EXAMPLE = 'http://localhost:8000/mcp'
_AUTH_STATUSES = frozenset({HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN})
_UNAVAILABLE_STATUSES = frozenset({HTTPStatus.BAD_GATEWAY, HTTPStatus.SERVICE_UNAVAILABLE})


class IngestExitError(Exception):
    """A condition that ends the run with one of the command's exit codes.

    Attributes:
        code: The exit code of the command.
        message: What happened and what the operator can change, without secrets.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolFailure:
    """A call that failed on its own; the caller decides what the failure costs the run.

    Attributes:
        message: The tool's error text, or why the request was refused.
    """

    message: str


def strip_userinfo(url: str) -> str:
    """Return a URL without the userinfo its authority may carry.

    Args:
        url: An absolute URL, possibly with ``user:password@`` before its host.

    Returns:
        The same URL with the userinfo removed.
    """
    parts = urlsplit(url)
    host = parts.netloc.rpartition('@')[2]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


class TokenFileAuth(httpx2.Auth):
    """Bearer auth that reads its token from a file before every request.

    Attributes:
        path: The token file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._sent: str | None = None

    def read_token(self) -> str:
        """Return the token the file holds now.

        Returns:
            The file's content without surrounding whitespace.

        Raises:
            IngestExitError: The file cannot be read or holds no token.
        """
        try:
            token = self.path.read_text(encoding='utf-8').strip()
        except OSError as error:
            reason = error.strerror or str(error)
            raise IngestExitError(EXIT_CONFIG, f'the token file cannot be read: {self.path}: {reason}') from error
        except UnicodeDecodeError as error:
            raise IngestExitError(EXIT_CONFIG, f'the token file cannot be read: {self.path}: {error.reason}') from error
        if not token:
            raise IngestExitError(EXIT_CONFIG, f'the token file is empty: {self.path}')
        return token

    def rotated(self) -> bool:
        """Return whether the file now holds another token than the latest request sent."""
        return self.read_token() != self._sent

    @override
    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response]:
        token = self.read_token()
        request.headers['Authorization'] = f'Bearer {token}'
        self._sent = token
        yield request


def build_auth(*, token_file: Path | None, token_env: str | None) -> httpx2.Auth | None:
    """Return the auth for the token source the operator named, or None when there is none.

    Args:
        token_file: The file ``--token-file`` names, read before every request.
        token_env: The environment variable ``--token-env`` names, read once.

    Returns:
        The auth that sets the bearer token, or None to send no Authorization header.

    Raises:
        IngestExitError: Both sources are named, or the named one holds no usable token.
    """
    if token_file is not None and token_env is not None:
        raise IngestExitError(EXIT_CONFIG, '--token-file and --token-env cannot be combined')
    if token_file is not None:
        auth = TokenFileAuth(token_file)
        # An unusable file is refused before the run starts, not at its first request.
        auth.read_token()
        return auth
    if token_env is None:
        return None
    value = os.environ.get(token_env)
    if value is None:
        raise IngestExitError(EXIT_CONFIG, f'the environment variable {token_env} named by --token-env is not set')
    if not value.strip():
        raise IngestExitError(EXIT_CONFIG, f'the environment variable {token_env} named by --token-env is empty')
    return BearerAuth(value.strip())


class StatusRecorder:
    """Records the HTTP status and ``Retry-After`` header of the latest response.

    Attributes:
        last_status: The status code of the latest response, or None before any.
        retry_after: The latest response's ``Retry-After`` value, or None without one.
    """

    def __init__(self) -> None:
        self.last_status: int | None = None
        self.retry_after: str | None = None

    def reset(self) -> None:
        """Forget the latest response, before the next exchange starts."""
        self.last_status = None
        self.retry_after = None

    async def on_response(self, response: httpx2.Response) -> None:
        """Record a response; the HTTP client calls this for every response it receives."""
        self.last_status = response.status_code
        self.retry_after = response.headers.get('Retry-After')


class HttpClientFactory(Protocol):
    """The HTTP client factory the Streamable HTTP transport calls, taking the follow_redirects keyword it passes."""

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
        follow_redirects: bool = False,
    ) -> httpx2.AsyncClient: ...


def http_client_factory(
    recorder: StatusRecorder,
    *,
    read_timeout: float,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> HttpClientFactory:
    """Return the factory the Streamable HTTP transport builds its HTTP client with.

    Args:
        recorder: The recorder every response is reported to.
        read_timeout: The read timeout when the transport passes no timeout of its own.
        transport: The HTTP transport requests go through, or None for the network.

    Returns:
        A factory accepting the keywords the transport passes, follow_redirects included.
    """

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
        follow_redirects: bool = False,
    ) -> httpx2.AsyncClient:
        # Redirects stay off as in the SDK's own client: the transport follows the
        # redirects that keep to the endpoint's origin itself.
        del follow_redirects
        return httpx2.AsyncClient(
            headers=headers,
            timeout=timeout if timeout is not None else httpx2.Timeout(_CONNECT_TIMEOUT_SECONDS, read=read_timeout),
            auth=auth,
            event_hooks={'response': [recorder.on_response]},
            transport=transport,
        )

    return factory


def _carried(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and every exception it carries as its cause or context."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _carried_exit(error: BaseException) -> IngestExitError | None:
    """Return the IngestExitError an exception carries, such as one the token file's auth raised mid-request."""
    return next((carried for carried in _carried(error) if isinstance(carried, IngestExitError)), None)


class IngestClient:
    """The MCP session of one sync run, used as an async context manager.

    Attributes:
        server: The server URL without userinfo, as messages name it.
        recorder: The recorder the HTTP client reports every response to.
    """

    def __init__(
        self,
        server: str,
        *,
        auth: httpx2.Auth | None,
        call_timeout: float,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.server = strip_userinfo(server)
        # The recorder class is looked up when the client is built, so a caller can
        # substitute a subclass through this module.
        self.recorder = StatusRecorder()
        self._auth = auth
        self._call_timeout = call_timeout
        # Whether the server has answered an exchange of this run, which makes a later
        # 401 an expired token rather than a wrong one.
        self._accepted = False
        factory = http_client_factory(self.recorder, read_timeout=call_timeout, transport=transport)
        self._client = Client(StreamableHttpTransport(server, auth=auth, httpx_client_factory=factory), timeout=call_timeout)
        self._session = AsyncExitStack()

    async def __aenter__(self) -> Self:
        connected = await self._exchange('the connection', lambda: self._session.enter_async_context(self._client))
        if isinstance(connected, ToolFailure):
            raise IngestExitError(EXIT_CONFIG, connected.message)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self._session.__aexit__(exc_type, exc_value, traceback)
        except Exception as error:
            # An exception already propagating is the one the run reports.
            if exc_value is not None:
                return
            failure = self._settle(error, 'the connection')
            raise IngestExitError(EXIT_CONFIG, failure.message) from error

    async def preflight(self, *, defer_generation: bool = False) -> None:
        """Check that the server offers every tool a sync calls, inserts without merging and, when asked, defers.

        Args:
            defer_generation: Whether the run asks the server to defer generation, which
                needs a ``defer_generation`` parameter on ``store_context``.

        Raises:
            IngestExitError: A tool is missing, ``store_context`` has no ``deduplicate``
                parameter or, for a deferring run, no ``defer_generation`` parameter, or the
                tool list could not be read.
        """
        listed = await self._exchange('the tool list', self._client.list_tools)
        if isinstance(listed, ToolFailure):
            raise IngestExitError(EXIT_CONFIG, listed.message)
        tools = {tool.name: tool for tool in listed}
        missing = [name for name in SYNC_TOOLS if name not in tools]
        if missing:
            message = f'the server does not offer {", ".join(missing)}, which a sync needs'
            if 'plan_context_sync' in missing:
                message += '; a server offers plan_context_sync unless ENABLE_CONTEXT_SYNC=false'
            raise IngestExitError(EXIT_CONFIG, message)
        parameters = tools['store_context'].input_schema.get('properties', {})
        if DEDUPLICATE_PARAMETER not in parameters:
            raise IngestExitError(
                EXIT_CONFIG,
                "the server's store_context has no deduplicate parameter, so it would merge the documents "
                'a sync inserts: upgrade the server',
            )
        if defer_generation and DEFER_GENERATION_PARAMETER not in parameters:
            raise IngestExitError(
                EXIT_CONFIG,
                "the server's store_context has no defer_generation parameter, so it cannot defer generation: "
                'upgrade the server or run without --defer-generation',
            )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object] | ToolFailure:
        """Call a tool and return its structured result.

        Args:
            name: The tool to call.
            arguments: The tool's arguments.

        Returns:
            The structured result, or a ToolFailure for the tool's own error or for a
            request too large for the server or a proxy in front of it.

        Raises:
            IngestExitError: The exchange failed in a way that ends the run.
        """
        result = await self._exchange(
            f'the call to {name}',
            lambda: self._client.call_tool(name, arguments, raise_on_error=False),
        )
        if isinstance(result, ToolFailure):
            return result
        if result.is_error:
            text = '\n'.join(block.text for block in result.content if isinstance(block, TextContent))
            return ToolFailure(text or f'{name} reported an error without a message')
        if result.structured_content is None:
            raise IngestExitError(EXIT_CONFIG, f'the server answered {name} without a structured result')
        return result.structured_content

    async def _exchange[T](self, what: str, operation: Callable[[], Awaitable[T]]) -> T | ToolFailure:
        """Run one MCP exchange, retrying a 401 once after the token file was rotated.

        Args:
            what: The exchange as messages name it, such as ``the call to plan_context_sync``.
            operation: The exchange to run.

        Returns:
            The exchange's result, or a ToolFailure for a request too large to send. A
            failure that ends the run is raised as an IngestExitError.
        """
        retried = False
        while True:
            self.recorder.reset()
            try:
                result = await operation()
            except Exception as error:
                if not retried and self._token_rotated(error):
                    retried = True
                    continue
                return self._settle(await self._ending_failure(error), what)
            self._accepted = True
            return result

    async def _ending_failure(self, error: Exception) -> Exception:
        """Return the failure that ended the session when an exchange failed because it ended.

        A request whose session died fails either with the transport's own error or
        with a closed connection, whichever the client notices first, while closing the
        dead session raises the transport's error every time. Closing it here makes the
        classification the same on every run and leaves nothing to raise on exit.

        Args:
            error: The exchange's error.

        Returns:
            The exception closing the session raised, or the exchange's own error when
            the session is still open or closed without one.
        """
        if isinstance(error, MCPError) and error.code != CONNECTION_CLOSED:
            return error
        try:
            await self._session.aclose()
        except Exception as cause:
            return cause
        return error

    def _settle(self, error: Exception, what: str) -> ToolFailure:
        """Return the per-call failure a failed exchange is, or raise the exit that ends the run."""
        carried = _carried_exit(error)
        if carried is error:
            raise error
        if carried is not None:
            raise carried from error
        classified = self._classify(error, what)
        if classified is None:
            raise error
        if isinstance(classified, ToolFailure):
            return classified
        raise classified from error

    def _token_rotated(self, error: Exception) -> bool:
        """Return whether a failed exchange was a 401 to a token the file has since replaced."""
        return (
            isinstance(error, MCPError)
            and error.code != REQUEST_TIMEOUT
            and self.recorder.last_status == HTTPStatus.UNAUTHORIZED
            and self._accepted
            and isinstance(self._auth, TokenFileAuth)
            and self._auth.rotated()
        )

    def _classify(self, error: Exception, what: str) -> IngestExitError | ToolFailure | None:
        """Return what a failed exchange means for the run, or None for an error no class covers."""
        if isinstance(error, MCPError):
            if error.code == REQUEST_TIMEOUT:
                return IngestExitError(
                    EXIT_TRANSIENT,
                    f'{what} got no answer within --call-timeout ({self._call_timeout:g} s): '
                    'rerun the sync, which plans from what the server stored',
                )
            if self.recorder.last_status is None:
                return IngestExitError(EXIT_TRANSIENT, f'{what} failed before the server answered: {error.message}')
            # The server answers a call with its SSE headers before the tool runs, so a
            # restart or a proxy cutting the response afterwards closes an accepted call
            # whose result never arrives; the server may or may not have applied it.
            if error.code == CONNECTION_CLOSED:
                return IngestExitError(
                    EXIT_TRANSIENT,
                    f'{what} lost its response stream from {self.server} after the server accepted it '
                    f"({error.message}): the call's outcome is unknown; rerun the sync, which plans from what "
                    'the server stored; when calls keep losing their stream after the same time, raise the '
                    f'response timeout of the proxy in front of the server to at least --call-timeout; see {LIMITS_SECTION}',
                )
            return self._classify_status(self.recorder.last_status, error.message, what)
        for carried in _carried(error):
            if isinstance(carried, httpx2.ConnectError):
                return IngestExitError(EXIT_TRANSIENT, f'cannot reach {self.server}: {carried}')
            if isinstance(carried, httpx2.TimeoutException):
                return IngestExitError(
                    EXIT_TRANSIENT,
                    f'{what} timed out talking to {self.server} ({type(carried).__name__}): '
                    'rerun the sync, or raise --call-timeout for batches that take longer',
                )
            # The server or a proxy closed the connection mid-exchange, as a restart or a
            # reset connection does; the server may or may not have applied the call.
            if isinstance(carried, httpx2.NetworkError | httpx2.RemoteProtocolError):
                return IngestExitError(
                    EXIT_TRANSIENT,
                    f'{what} lost its connection to {self.server} ({type(carried).__name__}): '
                    "the call's outcome is unknown; rerun the sync, which plans from what the server stored",
                )
        return None

    def _classify_status(self, status: int, detail: str, what: str) -> IngestExitError | ToolFailure:
        """Return what an exchange refused with an HTTP status means for the run."""
        if status in _AUTH_STATUSES:
            return IngestExitError(EXIT_CONFIG, self._refusal(status))
        if status == HTTPStatus.NOT_FOUND:
            return IngestExitError(
                EXIT_CONFIG,
                f'the server answered HTTP 404 at {self.server}: '
                f'--server must name the MCP endpoint, such as {_ENDPOINT_EXAMPLE}',
            )
        if status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            return ToolFailure(
                'the request was too large for the server or a proxy in front of it (HTTP 413): '
                f'lower --max-request-bytes or --batch-size, or raise the proxy body limit; see {LIMITS_SECTION}',
            )
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            return self._transient('the server is limiting requests (HTTP 429): rerun later')
        if status in _UNAVAILABLE_STATUSES:
            return self._transient(f'the server or a proxy in front of it is unavailable (HTTP {status}): rerun later')
        if status == HTTPStatus.GATEWAY_TIMEOUT:
            return self._transient(
                'a proxy in front of the server timed out (HTTP 504): raise the proxy read timeout '
                f'to at least --call-timeout, or lower --batch-size; see {LIMITS_SECTION}',
            )
        if HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            return IngestExitError(EXIT_CONFIG, f'{what} failed: {detail}')
        return IngestExitError(EXIT_CONFIG, f'{what} failed with HTTP {status}: {detail}')

    def _refusal(self, status: int) -> str:
        """Return the message for a refused token, naming expiry when the server accepted it earlier."""
        if self._auth is None:
            return f'the server requires a bearer token (HTTP {status}): pass --token-file or --token-env'
        if not self._accepted:
            return f'the server refused the bearer token (HTTP {status}): the token is wrong or not valid for this server'
        remedy = (
            f'write a fresh token to {self._auth.path} and rerun'
            if isinstance(self._auth, TokenFileAuth)
            else 'rerun with a fresh token'
        )
        return (
            f'the server refused the bearer token (HTTP {status}) after accepting it earlier in this run: '
            f'the token has expired; {remedy}'
        )

    def _transient(self, message: str) -> IngestExitError:
        """Return a transient exit, with the server's Retry-After value when it sent one."""
        if self.recorder.retry_after is not None:
            message += f'; Retry-After: {self.recorder.retry_after}'
        return IngestExitError(EXIT_TRANSIENT, message)
