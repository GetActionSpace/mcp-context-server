"""Tests for the sessions of a sync run: the further sessions a batch leg opens and how it sends batches over them."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import pytest

from app.cli.ingest.client import TokenFileAuth
from app.cli.ingest.client import ToolFailure
from app.cli.ingest.client import build_auth
from app.cli.ingest.sessions import RunSessions
from app.cli.ingest.sessions import SyncSession
from app.cli.ingest.sessions import send_over
from app.cli.ingest.sessions import session_auth


@dataclass
class _Session:
    """A session that only the scheduler handles; a test never calls a tool through it."""

    name: str
    closed: bool = False

    async def preflight(self, *, defer_generation: bool = False) -> None:
        """Never called by these tests."""
        raise AssertionError(f'preflight is not part of sending batches: {defer_generation}')

    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object] | ToolFailure:
        """Never called by these tests."""
        raise AssertionError(f'{name} is sent by the leg, not the scheduler: {arguments}')


@dataclass
class _Sends:
    """Records the batches a leg sent, and over which session, with scripted holds and failures."""

    holds: dict[int, int] = field(default_factory=dict[int, int])
    failures: dict[int, Exception] = field(default_factory=dict[int, Exception])
    started: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
    finished: list[int] = field(default_factory=list[int])
    busy: set[str] = field(default_factory=set[str])

    async def send(self, session: SyncSession, position: int) -> None:
        """Send one batch: hold it, then fail it or finish it."""
        assert isinstance(session, _Session)
        assert session.name not in self.busy, f'session {session.name} carried two batches at once'
        self.busy.add(session.name)
        self.started.append((session.name, position))
        try:
            for _ in range(self.holds.get(position, 1)):
                await asyncio.sleep(0)
            if position in self.failures:
                raise self.failures[position]
            self.finished.append(position)
        finally:
            self.busy.discard(session.name)


def _sessions(count: int) -> list[SyncSession]:
    """Return sessions named a, b, c and so on."""
    return [_Session(chr(ord('a') + number)) for number in range(count)]


class TestSendOver:
    """Batches start in order, one in flight per session, and no failure cancels a batch in flight."""

    @pytest.mark.asyncio
    async def test_every_batch_is_sent_once_in_order_with_one_in_flight_per_session(self) -> None:
        """Seven batches over three sessions start in position order and all finish."""
        sends = _Sends(holds={0: 4, 1: 1, 2: 2})

        await send_over(_sessions(3), count=7, send=sends.send)

        assert [position for _, position in sends.started] == list(range(7))
        assert [name for name, _ in sends.started[:3]] == ['a', 'b', 'c']
        assert sorted(sends.finished) == list(range(7))
        assert sends.finished[0] == 1

    @pytest.mark.asyncio
    async def test_a_failure_stops_the_batches_not_started_and_waits_for_the_running_ones(self) -> None:
        """The failure is raised as itself after the batch beside it finished, and no later batch starts."""
        failure = RuntimeError('the call ended the run')
        sends = _Sends(holds={0: 1, 1: 3}, failures={0: failure})

        with pytest.raises(RuntimeError) as raised:
            await send_over(_sessions(2), count=5, send=sends.send)

        assert raised.value is failure
        assert [position for _, position in sends.started] == [0, 1]
        assert sends.finished == [1]

    @pytest.mark.asyncio
    async def test_several_failures_are_raised_together_in_the_order_they_happened(self) -> None:
        """Two batches failing while in flight together raise one group, the earlier failure first."""
        late, early = RuntimeError('position 0'), RuntimeError('position 2')
        sends = _Sends(holds={0: 3, 1: 2, 2: 1}, failures={0: late, 2: early})

        with pytest.raises(ExceptionGroup) as raised:
            await send_over(_sessions(3), count=4, send=sends.send)

        assert list(raised.value.exceptions) == [early, late]
        assert sends.finished == [1]
        assert [position for _, position in sends.started] == [0, 1, 2]


class TestRunSessions:
    """A leg opens only the sessions the run does not hold yet, and they close with the run."""

    @pytest.mark.asyncio
    async def test_take_opens_only_the_missing_sessions_and_the_stack_closes_them(self) -> None:
        """The first session is always included, further ones are opened once and closed when the run ends."""
        opened: list[_Session] = []

        @asynccontextmanager
        async def open_session() -> AsyncGenerator[SyncSession]:
            session = _Session(f'further-{len(opened)}')
            opened.append(session)
            try:
                yield session
            finally:
                session.closed = True

        first = _Session('first')
        async with AsyncExitStack() as stack:
            sessions = RunSessions(first, open_session=open_session, stack=stack)

            assert await sessions.take(1) == [first]
            assert opened == []
            three = await sessions.take(3)
            assert three == [first, *opened]
            assert len(opened) == 2
            assert await sessions.take(2) == three[:2]
            assert len(opened) == 2
            assert not any(session.closed for session in opened)

        assert all(session.closed for session in opened)
        assert sessions.first is first


class TestSessionAuth:
    """A further session reads a token file with a reader of its own and shares any other auth."""

    def test_a_token_file_gets_a_reader_of_its_own(self, tmp_path: Path) -> None:
        """The further session's reader reads the same file but tracks its own last token."""
        token = tmp_path / 'token'
        token.write_text('ingest-token\n', encoding='utf-8')
        auth = build_auth(token_file=token, token_env=None)

        further = session_auth(auth)

        assert isinstance(further, TokenFileAuth)
        assert further is not auth
        assert further.path == token

    def test_a_token_read_once_and_no_token_are_shared(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A token from the environment is not read again, and a run without a token sends none."""
        monkeypatch.setenv('INGEST_TEST_TOKEN', 'ingest-token')
        auth = build_auth(token_file=None, token_env='INGEST_TEST_TOKEN')

        assert session_auth(auth) is auth
        assert session_auth(None) is None
