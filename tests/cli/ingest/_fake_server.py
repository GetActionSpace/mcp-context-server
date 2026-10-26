"""An in-memory context server the sync legs of the ingest command run against.

The fake answers the tools a sync calls the way the server does for the caller's
principal: it plans with the server's own diff over the managed entries the
caller can read, stores every batch entry as a fresh entry owned by the caller,
replaces the fields an update names and deletes only the caller's own entries.
It records every call with the number of calls in flight when it arrived, and a
test can make a call fail or end the run, make one entry fail, keep an entry from
being deleted, hold calls so they finish in a chosen order, or answer a batch's
per-entry results in reverse order, so each leg meets the failures and the
timing a real server produces.

Every connection is a session of its own that carries one call at a time, as the
command's client does, and a session that receives a second call while the first
is still running fails the test.
"""

import asyncio
import copy
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import cast

import httpx2
from pydantic import TypeAdapter
from pydantic import ValidationError

from app.cli.ingest.apply import RunConfig
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import ToolFailure
from app.digests import content_hash
from app.ingest import INGEST_KEY
from app.ingest import MAX_PLAN_DELETES
from app.ingest import SyncManifestItem
from app.ingest import ingest_block_error
from app.tools._sync_plan import StoredMember
from app.tools._sync_plan import compute_sync_plan
from app.tools._sync_plan import manifest_error
from app.tools._sync_plan import stored_member

CALLER = 'local'
OTHER = 'bob'
_MANIFEST = TypeAdapter(list[SyncManifestItem])

type Json = dict[str, object]


@dataclass
class Stored:
    """One stored entry of the fake server."""

    context_id: str
    thread_id: str
    text: str
    metadata: Json
    tags: list[str]
    visibility: str
    owner: str = CALLER

    @property
    def block(self) -> Json | None:
        """Return the entry's ingest block, when it has one."""
        block = self.metadata.get(INGEST_KEY)
        return cast('Json', block) if isinstance(block, dict) else None

    @property
    def source_id(self) -> str | None:
        """Return the source id of the entry's ingest block, when it has one."""
        block = self.block
        source_id = block.get('source_id') if block is not None else None
        return source_id if isinstance(source_id, str) else None


@dataclass
class Call:
    """One call the fake server received.

    Attributes:
        tool: The tool it called, or ``preflight``.
        arguments: The tool arguments.
        in_flight: The calls in flight when it arrived, itself included.
        session: The session it came over, numbered from 1 in the order the server opened them.
    """

    tool: str
    arguments: Json
    in_flight: int
    session: int


@dataclass
class _Scripted:
    """What the next call of a tool does instead of, or after, its normal answer."""

    outcome: ToolFailure | IngestExitError | Json
    commit: bool


@dataclass
class FakeServer:
    """An in-memory context server for one caller, with the failures a test arranges."""

    entries: dict[str, Stored] = field(default_factory=dict[str, Stored])
    calls: list[Call] = field(default_factory=list[Call])
    connections: list[httpx2.Auth | None] = field(default_factory=list[httpx2.Auth | None])
    failing_sources: dict[str, str] = field(default_factory=dict[str, str])
    failing_updates: dict[str, str] = field(default_factory=dict[str, str])
    undeletable: set[str] = field(default_factory=set[str])
    preflight_exit: IngestExitError | None = None
    reverse_results: bool = False
    _scripts: dict[str, list[_Scripted]] = field(default_factory=dict[str, list[_Scripted]])
    _holds: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    _issued: int = 0
    _in_flight: int = 0
    _opened: int = 0

    @asynccontextmanager
    async def connect(self, config: RunConfig, auth: httpx2.Auth | None) -> AsyncGenerator['FakeSession']:
        """Open a session the way the command's connector does, recording the auth the run built for it."""
        del config
        self.connections.append(auth)
        self._opened += 1
        yield FakeSession(self, number=self._opened)

    def fail_next(self, tool: str, outcome: ToolFailure | IngestExitError, *, commit: bool = False) -> None:
        """Make the next call of a tool fail or end the run, after applying the call when commit is set."""
        self._scripts.setdefault(tool, []).append(_Scripted(outcome=outcome, commit=commit))

    def answer_next(self, tool: str, result: Json) -> None:
        """Make the next call of a tool answer with a given result, as a server of another version might."""
        self._scripts.setdefault(tool, []).append(_Scripted(outcome=result, commit=False))

    def hold_next(self, tool: str, *turns: int) -> None:
        """Make the next calls of a tool yield to the event loop the given number of times before they answer.

        Every other call yields once, so calls in flight at once overlap; a call held
        longer finishes after the ones sent beside it.
        """
        self._holds.setdefault(tool, []).extend(turns)

    def seed(
        self,
        *,
        thread_id: str,
        text: str,
        metadata: Json,
        owner: str = CALLER,
        visibility: str = 'private',
        tags: list[str] | None = None,
    ) -> str:
        """Store an entry directly, as another client or an earlier run did, and return its id."""
        self._issued += 1
        context_id = f'0190{self._issued:028x}'
        self.entries[context_id] = Stored(
            context_id=context_id,
            thread_id=thread_id,
            text=text,
            metadata=copy.deepcopy(metadata),
            tags=list(tags or []),
            visibility=visibility,
            owner=owner,
        )
        return context_id

    def duplicate(self, context_id: str) -> str:
        """Store a newer copy of an entry, as a batch that committed after its client gave up does."""
        original = self.entries[context_id]
        return self.seed(
            thread_id=original.thread_id,
            text=original.text,
            metadata=original.metadata,
            owner=original.owner,
            visibility=original.visibility,
            tags=original.tags,
        )

    def calls_to(self, tool: str) -> list[Call]:
        """Return the calls of one tool, in order."""
        return [call for call in self.calls if call.tool == tool]

    def by_source(self, thread_id: str) -> dict[str, list[Stored]]:
        """Return the managed entries of a thread grouped by source id."""
        grouped: dict[str, list[Stored]] = {}
        for entry in self.entries.values():
            if entry.thread_id == thread_id and entry.source_id is not None:
                grouped.setdefault(entry.source_id, []).append(entry)
        return grouped

    async def preflight(self, *, session: int) -> None:
        """Pass the preflight unless a test made it end the run."""
        self.calls.append(Call(tool='preflight', arguments={}, in_flight=self._in_flight + 1, session=session))
        if self.preflight_exit is not None:
            raise self.preflight_exit

    async def call_tool(self, name: str, arguments: Json, *, session: int) -> Json | ToolFailure:
        """Answer a tool call as the server does, or as a test scripted it, after the call's hold.

        A script and a hold belong to the calls of a tool in the order the calls
        arrive, whichever of them finishes first.

        Returns:
            The tool's result or the call's failure.
        """
        self._in_flight += 1
        self.calls.append(Call(tool=name, arguments=copy.deepcopy(arguments), in_flight=self._in_flight, session=session))
        scripted = self._scripts.get(name)
        step = scripted.pop(0) if scripted else None
        holds = self._holds.get(name)
        turns = holds.pop(0) if holds else 1
        try:
            for _ in range(turns):
                await asyncio.sleep(0)
            if step is None:
                return self._answer(name, arguments)
            if step.commit:
                self._answer(name, arguments)
            if isinstance(step.outcome, IngestExitError):
                raise step.outcome
            return step.outcome
        finally:
            self._in_flight -= 1

    def _answer(self, name: str, arguments: Json) -> Json | ToolFailure:
        """Run the tool a call names."""
        if name == 'plan_context_sync':
            return self._plan(arguments)
        if name == 'store_context_batch':
            return self._store(cast('list[Json]', arguments['entries']))
        if name == 'update_context_batch':
            return self._update(cast('list[Json]', arguments['updates']))
        if name == 'delete_context_batch':
            return self._delete(cast('list[str]', arguments['context_ids']))
        raise AssertionError(f'a sync never calls {name}')

    def _plan(self, arguments: Json) -> Json | ToolFailure:
        """Plan with the server's diff over the caller's readable managed entries of the range."""
        thread_id = cast('str', arguments['thread_id'])
        after = cast('str | None', arguments.get('source_id_after'))
        through = cast('str | None', arguments.get('source_id_through'))
        try:
            manifest = _MANIFEST.validate_python(arguments['manifest'])
        except ValidationError as error:
            return ToolFailure(str(error))
        problem = manifest_error(
            manifest,
            source_id_after=after,
            source_id_through=through,
            allow_empty=arguments.get('allow_empty_manifest') is True,
        )
        if problem is not None:
            return ToolFailure(problem)
        members = [member for entry in self.entries.values() if (member := self._member(entry, thread_id, after, through))]
        plan = compute_sync_plan(manifest, members)
        return {
            'thread_id': thread_id,
            'creates': plan.creates,
            'updates': list(plan.updates),
            'deletes': list(plan.deletes[:MAX_PLAN_DELETES]),
            'deletes_total': len(plan.deletes),
            'unchanged_count': plan.unchanged_count,
            'kept_count': plan.kept_count,
            'foreign_count': plan.foreign_count,
        }

    @staticmethod
    def _member(entry: Stored, thread_id: str, after: str | None, through: str | None) -> StoredMember | None:
        """Return the member an entry is in a planned range, or None when it is not one."""
        readable = entry.owner == CALLER or entry.visibility == 'public'
        block = entry.block
        if entry.thread_id != thread_id or not readable or block is None or block.get('scope') != thread_id:
            return None
        member = stored_member(
            block,
            context_id=entry.context_id,
            content_hash=content_hash(entry.text),
            is_owner=entry.owner == CALLER,
        )
        if member is None:
            return None
        if (after is not None and member.source_id <= after) or (through is not None and member.source_id > through):
            return None
        return member

    def _store(self, entries: list[Json]) -> Json:
        """Store every entry as a fresh entry owned by the caller."""
        results: list[Json] = []
        for index, entry in enumerate(entries):
            metadata = cast('Json', entry['metadata'])
            block = metadata.get(INGEST_KEY)
            error = ingest_block_error(block)
            if error is None:
                error = self.failing_sources.get(cast('str', cast('Json', block)['source_id']))
            if error is not None:
                results.append({
                    'index': index, 'success': False, 'context_id': None, 'error': error, 'generation_status': None,
                })
                continue
            context_id = self.seed(
                thread_id=cast('str', entry['thread_id']),
                text=cast('str', entry['text']),
                metadata=metadata,
                visibility=cast('str', entry['visibility']),
                tags=cast('list[str]', entry['tags']),
            )
            results.append({
                'index': index, 'success': True, 'context_id': context_id, 'error': None, 'generation_status': 'ready',
            })
        return _batch_response(results, 'Stored', reverse=self.reverse_results)

    def _update(self, updates: list[Json]) -> Json:
        """Replace the fields each update names on the caller's own entries."""
        results: list[Json] = []
        for index, update in enumerate(updates):
            context_id = cast('str', update['context_id'])
            entry = self.entries.get(context_id)
            if entry is None or entry.owner != CALLER:
                error: str | None = f'Context entry {context_id} not found'
            else:
                error = self.failing_updates.get(context_id)
            if entry is None or error is not None:
                results.append({
                    'index': index, 'context_id': context_id, 'success': False, 'updated_fields': None, 'error': error,
                    'generation_status': None,
                })
                continue
            if 'text' in update:
                entry.text = cast('str', update['text'])
            if 'metadata' in update:
                entry.metadata = copy.deepcopy(cast('Json', update['metadata']))
            if 'tags' in update:
                entry.tags = list(cast('list[str]', update['tags']))
            if 'visibility' in update:
                entry.visibility = cast('str', update['visibility'])
            fields = [name for name in ('text', 'metadata', 'tags', 'visibility') if name in update]
            results.append({
                'index': index, 'context_id': context_id, 'success': True, 'updated_fields': fields, 'error': None,
                'generation_status': 'ready',
            })
        return _batch_response(results, 'Updated', reverse=self.reverse_results)

    def _delete(self, context_ids: list[str]) -> Json:
        """Delete the caller's own entries among the named ones."""
        deleted = 0
        for context_id in context_ids:
            entry = self.entries.get(context_id)
            if entry is not None and entry.owner == CALLER and context_id not in self.undeletable:
                del self.entries[context_id]
                deleted += 1
        return {'success': True, 'deleted_count': deleted, 'criteria_used': ['context_ids'], 'message': f'Deleted {deleted}'}


@dataclass
class FakeSession:
    """One session with the fake server, carrying one call at a time as the command's client does.

    Attributes:
        server: The server the session talks to.
        number: The session's number, from 1 in the order the server opened its sessions.
        busy: Whether the session carries a call now.
    """

    server: FakeServer
    number: int
    busy: bool = False

    async def preflight(self) -> None:
        """Run the preflight on the server."""
        await self.server.preflight(session=self.number)

    async def call_tool(self, name: str, arguments: Json) -> Json | ToolFailure:
        """Pass a call to the server; a second call while one is running fails the test."""
        assert not self.busy, f'a session received a call to {name} while it still carried another call'
        self.busy = True
        try:
            return await self.server.call_tool(name, arguments, session=self.number)
        finally:
            self.busy = False


def _batch_response(results: list[Json], verb: str, *, reverse: bool) -> Json:
    """Return a batch tool's response for its per-entry results, listed in reverse order when asked."""
    succeeded = sum(1 for result in results if result['success'])
    return {
        'success': succeeded == len(results),
        'total': len(results),
        'succeeded': succeeded,
        'failed': len(results) - succeeded,
        'results': results[::-1] if reverse else results,
        'message': f'{verb} {succeeded} of {len(results)}',
    }
