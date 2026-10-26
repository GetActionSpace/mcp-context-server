"""The sessions of one sync run, and how its batch legs keep several calls in flight over them.

A run opens one session to plan, delete and verify. A create or update leg that
keeps more than one batch call in flight opens a further session for each extra
call, because one session carries one call at a time: the client tells what a
failed exchange means from the HTTP status of the session's latest response, and
closes the session when its connection is lost.
"""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from typing import Protocol

import httpx2

from app.cli.ingest.client import TokenFileAuth
from app.cli.ingest.client import ToolFailure


class SyncSession(Protocol):
    """The MCP session a run calls the server through."""

    async def preflight(self, *, defer_generation: bool = False) -> None:
        """Check that the server offers what a sync needs, deferring generation when the run defers it."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object] | ToolFailure:
        """Call a tool and return its structured result or the failure of that one call."""
        ...


def session_auth(auth: httpx2.Auth | None) -> httpx2.Auth | None:
    """Return the auth of a further session of a run, given the auth of its first session.

    A token file gets a reader of its own, because the reader tracks the token its
    session sent last to tell a rotated token from a refused one; a token read once
    from the environment, or no token, is shared as it is.

    Args:
        auth: The auth of the run's first session.

    Returns:
        The auth for the further session.
    """
    return TokenFileAuth(auth.path) if isinstance(auth, TokenFileAuth) else auth


class RunSessions:
    """The sessions of one run: the first, which also plans and deletes, and the further ones the batch legs open.

    A further session opens when a batch leg first keeps more batches in flight than
    the run holds sessions, and stays open until the run's exit stack closes.

    Attributes:
        first: The session the run opened first.
    """

    def __init__(
        self,
        first: SyncSession,
        *,
        open_session: Callable[[], AbstractAsyncContextManager[SyncSession]],
        stack: AsyncExitStack,
    ) -> None:
        self.first = first
        self._open_session = open_session
        self._stack = stack
        self._held = [first]

    async def take(self, count: int) -> list[SyncSession]:
        """Return as many sessions as a leg keeps batches in flight, opening the ones the run does not hold yet.

        Args:
            count: How many batches the leg keeps in flight at once.

        Returns:
            The sessions, the first one included.
        """
        while len(self._held) < count:
            self._held.append(await self._stack.enter_async_context(self._open_session()))
        return self._held[:count]


async def send_over(
    sessions: Sequence[SyncSession],
    *,
    count: int,
    send: Callable[[SyncSession, int], Awaitable[None]],
) -> None:
    """Send a leg's batches over its sessions, one batch in flight on each, starting the batches in order.

    A failed send keeps every batch not yet started from starting, while the batches
    already in flight finish; no failure cancels another batch.

    Args:
        sessions: The leg's sessions.
        count: How many batches the leg sends.
        send: Sends the batch at a position, from 0, over a session.

    Raises:
        ExceptionGroup: Several sends failed, in the order they failed; a single failure
            is raised as itself. Either is raised once every batch in flight has finished.
    """
    positions = iter(range(count))
    failures: list[Exception] = []

    async def work(session: SyncSession) -> None:
        while not failures and (position := next(positions, None)) is not None:
            try:
                await send(session, position)
            except Exception as error:
                failures.append(error)

    async with asyncio.TaskGroup() as group:
        for session in sessions:
            group.create_task(work(session))
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ExceptionGroup('several batch calls failed', failures)
