"""One sync run: plan the corpus, store, update and delete, then verify that the thread matches.

A run reads the corpus directory, sends its manifest to ``plan_context_sync`` in
source-id ranges and applies each plan with the batch tools: creates are stored
as fresh entries that are never merged into another one, updates replace the
metadata in full, and deletes go by entry id after every create and update, so a
run that stops early never removed anything before the new content landed. A
final plan of every range verifies the result.

The create and update legs keep up to ``--concurrency`` batch calls in flight, each
on a session of its own, and record the outcomes in batch order, so a run reports
what a sequential run reports however its batches finish. Every document goes into
exactly one batch and every result is matched to its entry by index, so the order
in which the server works through the entries of a batch never matters. Plans and
deletes go over the run's first session, one call at a time.

A run that defers generation asks the server to store every create and update at
once and generate its embeddings and summaries in the background; the report counts
the written entries the server answered as pending. The final plan compares what the
write committed, so a run converges while generation is still pending.

A run keeps no state of its own. Every run derives its plan from what the server
stores, so a rerun converges after any failure, and a call whose answer never
came is never sent again within the run: the next run's plan sees whatever it
committed. A directory the walk could not list leaves its files out of the
manifest, so the plan calls their stored entries absent; such a run still
creates and updates, executes no delete at all and reports every planned delete
as withheld.
"""

import itertools
import sys
import time
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Literal
from typing import NotRequired
from typing import TypedDict
from typing import cast

import httpx2
from pydantic import TypeAdapter
from pydantic import ValidationError

from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import IngestClient
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import ToolFailure
from app.cli.ingest.client import build_auth
from app.cli.ingest.corpus import Corpus
from app.cli.ingest.corpus import Document
from app.cli.ingest.corpus import Visibility
from app.cli.ingest.corpus import build_corpus
from app.cli.ingest.corpus import format_synced_at
from app.cli.ingest.manifest import ManifestSlice
from app.cli.ingest.manifest import manifest_items
from app.cli.ingest.manifest import pack
from app.cli.ingest.manifest import slice_manifest
from app.cli.ingest.report import DocumentOutcome
from app.cli.ingest.report import RunReport
from app.cli.ingest.sessions import RunSessions
from app.cli.ingest.sessions import SyncSession
from app.cli.ingest.sessions import send_over
from app.cli.ingest.sessions import session_auth
from app.ingest import PlanContextSyncDict
from app.ingest import SyncDeleteDict
from app.types import GenerationStatus

THREAD_PREFIX = 'kb:'
DEFAULT_INCLUDES = ('**/*.md',)
DEFAULT_BATCH_SIZE = 10
# The batch tools take at most this many entries per call.
MAX_BATCH_SIZE = 100
DEFAULT_MAX_REQUEST_BYTES = 3_145_728
# The MCP SDK refuses an HTTP request body above 4 MiB, and the limit is not configurable.
MAX_REQUEST_BYTES = 4_194_304
DEFAULT_CALL_TIMEOUT = 3600.0
DEFAULT_CONCURRENCY = 1
# The server's embedding and summary concurrency settings go up to 20, so more batches
# in flight would only wait for the server.
MAX_CONCURRENCY = 20

PLAN_TOOL = 'plan_context_sync'
STORE_TOOL = 'store_context_batch'
UPDATE_TOOL = 'update_context_batch'
DELETE_TOOL = 'delete_context_batch'


class _BatchResultItem(TypedDict):
    """One per-entry result of ``store_context_batch`` or ``update_context_batch``, as far as a sync reads it.

    A server that cannot defer generation answers without ``generation_status``;
    the field is None for an entry the batch did not write.
    """

    index: int
    success: bool
    context_id: str | None
    error: str | None
    generation_status: NotRequired[GenerationStatus | None]


class _BatchResponse(TypedDict):
    """The response of ``store_context_batch`` or ``update_context_batch``, as far as a sync reads it."""

    results: list[_BatchResultItem]


_PLAN = TypeAdapter(PlanContextSyncDict)
_BATCH_RESPONSE = TypeAdapter(_BatchResponse)
# How many validation problems a message about an unexpected response names.
_NAMED_PROBLEMS = 3
_NOT_DELETED = 'the server kept the entry when asked to delete it'


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """Every setting of one sync run, resolved from the command line.

    Attributes:
        server: The server's Streamable HTTP endpoint.
        corpus: The corpus name; the corpus lives in the thread ``kb:<corpus>``.
        source_dir: The corpus root, walked recursively.
        includes: Patterns of which at least one must match a file.
        excludes: Patterns of which none may match a file.
        tags: Tags applied to every document.
        metadata: Metadata applied to every document.
        visibility: The visibility sent with every create and update.
        defer_generation: Whether every create and update asks the server to defer generation.
        token_file: The file the bearer token is read from before every request.
        token_env: The environment variable the bearer token is read from once.
        dry_run: Whether the run only plans.
        allow_empty_corpus: Whether a corpus without documents may delete every
            managed entry of its thread.
        report: The file the JSON report is written to.
        batch_size: The most entries one batch call carries.
        max_request_bytes: The request budget of one call.
        call_timeout: How long one call may take, in seconds.
        concurrency: The most create or update batch calls in flight at once.
    """

    server: str
    corpus: str
    source_dir: Path
    includes: tuple[str, ...] = DEFAULT_INCLUDES
    excludes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])
    visibility: Visibility = 'private'
    defer_generation: bool = False
    token_file: Path | None = None
    token_env: str | None = None
    dry_run: bool = False
    allow_empty_corpus: bool = False
    report: Path | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    call_timeout: float = DEFAULT_CALL_TIMEOUT
    concurrency: int = DEFAULT_CONCURRENCY

    @property
    def thread_id(self) -> str:
        """Return the corpus thread, which is also the scope of every managed entry's ingest block."""
        return THREAD_PREFIX + self.corpus


type Connect = Callable[[RunConfig, httpx2.Auth | None], AbstractAsyncContextManager[SyncSession]]


def connect_client(config: RunConfig, auth: httpx2.Auth | None) -> IngestClient:
    """Return the Streamable HTTP session of a run with the configured server.

    Args:
        config: The run's settings.
        auth: The bearer-token auth, or None to send no Authorization header.

    Returns:
        The session, to be entered as an async context manager.
    """
    return IngestClient(config.server, auth=auth, call_timeout=config.call_timeout)


async def run_sync(config: RunConfig, *, connect: Connect = connect_client) -> RunReport:
    """Synchronize a corpus directory into its thread.

    Args:
        config: The run's settings.
        connect: Opens each session of the run, the first and every further one; the
            default speaks Streamable HTTP to the configured server.

    Returns:
        The run's report; its exit code is the command's. Every distinct failure that
        ended the run is among its errors, the batches in flight at once possibly
        ending it together.
    """
    report = RunReport(server=config.server, thread_id=config.thread_id, dry_run=config.dry_run)
    synced_at = format_synced_at(datetime.now(UTC))
    try:
        auth = build_auth(token_file=config.token_file, token_env=config.token_env)
        prepared = _prepare(config, report)
        if prepared is not None:
            async with AsyncExitStack() as stack:
                session = await stack.enter_async_context(connect(config, auth))
                await session.preflight(defer_generation=config.defer_generation)
                sessions = RunSessions(session, open_session=lambda: connect(config, session_auth(auth)), stack=stack)
                await _Run(sessions=sessions, config=config, report=report, prepared=prepared, synced_at=synced_at).run()
    except* IngestExitError as ended:
        for error in _distinct(ended):
            report.fail(error.message, exit_code=error.code)
    return report


def _distinct(ended: BaseExceptionGroup[IngestExitError]) -> list[IngestExitError]:
    """Return each distinct failure that ended the run once, in the order the failures happened."""
    distinct: dict[tuple[int, str], IngestExitError] = {}
    for error in ended.exceptions:
        for leaf in _distinct(error) if isinstance(error, BaseExceptionGroup) else [error]:
            distinct.setdefault((leaf.code, leaf.message), leaf)
    return list(distinct.values())


@dataclass(frozen=True)
class _Prepared:
    """What a run read from its directory before contacting the server.

    Attributes:
        documents: The corpus documents by source id.
        slices: The manifest in the source-id ranges the run plans one at a time.
        withhold_deletes: Whether a directory could not be listed, so no delete may run.
    """

    documents: dict[str, Document]
    slices: list[ManifestSlice]
    withhold_deletes: bool


def _prepare(config: RunConfig, report: RunReport) -> _Prepared | None:
    """Build the corpus and its manifest, record the files and directories the run cannot use, and slice the manifest.

    Returns:
        What the run sends, or None when the run cannot start, which the report records.
    """
    corpus = build_corpus(
        config.source_dir,
        thread_id=config.thread_id,
        includes=config.includes,
        excludes=config.excludes,
        static_metadata=config.metadata,
        tags=config.tags,
        visibility=config.visibility,
        defer_generation=config.defer_generation,
        max_request_bytes=config.max_request_bytes,
    )
    _record_unusable(corpus, report)
    items = manifest_items(corpus)
    if not items and not config.allow_empty_corpus:
        report.fail(
            f'found no document to sync under {config.source_dir}; pass --allow-empty-corpus to confirm that the '
            f'sync deletes every managed entry of {config.thread_id}',
            exit_code=EXIT_CONFIG,
        )
        return None
    try:
        slices = slice_manifest(items, max_request_bytes=config.max_request_bytes)
    except ValueError as error:
        report.fail(f'--max-request-bytes is too small for a plan request: {error}', exit_code=EXIT_CONFIG)
        return None
    return _Prepared(
        documents={document.source_id: document for document in corpus.documents},
        slices=slices,
        withhold_deletes=bool(corpus.directory_failures),
    )


def _record_unusable(corpus: Corpus, report: RunReport) -> None:
    """Record every file the run keeps as stored or leaves out, and every directory it could not list."""
    for failure in corpus.file_failures:
        if failure.keep:
            report.add(DocumentOutcome(source_id=failure.source_id, outcome='kept', context_id=None, error=failure.reason))
            continue
        report.add(DocumentOutcome(source_id=failure.source_id, outcome='failed', context_id=None, error=failure.reason))
        report.fail(f'file left out of the manifest: {failure.source_id}: {failure.reason}', exit_code=EXIT_PARTIAL)
    for directory in corpus.directory_failures:
        report.fail(f'directory not listed: {directory.relative_path}: {directory.reason}', exit_code=EXIT_PARTIAL)


@dataclass(frozen=True)
class _BatchOutcome:
    """What one batch call did.

    Attributes:
        outcomes: One outcome per entry of the batch, in the batch's order.
        pending: How many of the written entries the server reported with their generation pending.
    """

    outcomes: list[DocumentOutcome]
    pending: int


@dataclass
class _Run:
    """The legs of one run over its sessions."""

    sessions: RunSessions
    config: RunConfig
    report: RunReport
    prepared: _Prepared
    synced_at: str

    async def run(self) -> None:
        """Plan every range; unless the run is dry, then create, update, delete and verify."""
        plans = [await self._plan(piece) for piece in self.prepared.slices]
        self._record_plans(plans)
        if self.config.dry_run:
            return
        await self._create(plans)
        await self._update(plans)
        await self._delete()
        await self._verify()

    async def _plan(self, piece: ManifestSlice) -> PlanContextSyncDict:
        """Plan one range of the manifest.

        Returns:
            The range's plan.

        Raises:
            IngestExitError: The server refused the plan or answered with something
                that is not a plan of this range.
        """
        arguments: dict[str, object] = {'thread_id': self.config.thread_id, 'manifest': piece.items}
        if piece.source_id_after is not None:
            arguments['source_id_after'] = piece.source_id_after
        if piece.source_id_through is not None:
            arguments['source_id_through'] = piece.source_id_through
        if not piece.items:
            # An empty manifest reaches the plan only when --allow-empty-corpus confirmed it.
            arguments['allow_empty_manifest'] = True
        result = await self.sessions.first.call_tool(PLAN_TOOL, arguments)
        if isinstance(result, ToolFailure):
            raise IngestExitError(EXIT_CONFIG, f'{PLAN_TOOL} refused the plan of {_describe(piece)}: {result.message}')
        try:
            plan = _PLAN.validate_python(result)
        except ValidationError as error:
            raise IngestExitError(EXIT_CONFIG, _unexpected(PLAN_TOOL, error)) from error
        indices = [*plan['creates'], *(update['index'] for update in plan['updates'])]
        if any(not 0 <= index < len(piece.items) or piece.items[index].get('keep') is True for index in indices):
            raise IngestExitError(
                EXIT_CONFIG,
                f'the server answered {PLAN_TOOL} with a manifest index that names no document of {_describe(piece)}',
            )
        return plan

    def _record_plans(self, plans: list[PlanContextSyncDict]) -> None:
        """Record the unchanged documents and the foreign members, print the plan, and in a dry run record it."""
        for piece, plan in zip(self.prepared.slices, plans, strict=True):
            planned = {*plan['creates'], *(update['index'] for update in plan['updates'])}
            for index, item in enumerate(piece.items):
                if index not in planned and item.get('keep') is not True:
                    self._add(cast('str', item['source_id']), 'unchanged')
            if self.config.dry_run:
                self._record_dry_run(piece, plan)
        foreign = sum(plan['foreign_count'] for plan in plans)
        self.report.count('foreign', foreign)
        _say(
            f'plan: {sum(len(plan["creates"]) for plan in plans)} to create, '
            f'{sum(len(plan["updates"]) for plan in plans)} to update, '
            f'{sum(plan["deletes_total"] for plan in plans)} to delete, '
            f'{sum(plan["unchanged_count"] for plan in plans)} unchanged, '
            f'{sum(plan["kept_count"] for plan in plans)} kept, {foreign} foreign',
        )
        if foreign:
            print(
                f'warning: {self.config.thread_id} holds {foreign} managed entries another principal owns; this run '
                'manages only the entries of the principal it authenticates as and leaves them untouched',
                file=sys.stderr,
                flush=True,
            )

    def _record_dry_run(self, piece: ManifestSlice, plan: PlanContextSyncDict) -> None:
        """Print and record what applying one range's plan would do."""
        for index in plan['creates']:
            source_id = self._document(piece, index).source_id
            _say(f'would create {source_id}')
            self._add(source_id, 'created')
        for update in plan['updates']:
            source_id = self._document(piece, update['index']).source_id
            _say(f'would update {source_id} {update["context_id"]} ({update["change"]})')
            self._add(source_id, 'updated', context_id=update['context_id'])
        self._list_deletes(piece, plan)

    def _list_deletes(self, piece: ManifestSlice, plan: PlanContextSyncDict) -> None:
        """Print and count a range's planned deletes: as withheld when a directory was not listed, else as dry-run deletes.

        A plan lists at most 100 deletes of its range, so the listed ones are printed
        and recorded one by one and the count covers every planned delete.
        """
        withheld = self.prepared.withhold_deletes
        for delete in plan['deletes']:
            _say(f'{"withheld delete" if withheld else "would delete"} {_delete_line(delete)}')
            if not withheld:
                self._add(delete['source_id'], 'deleted', context_id=delete['context_id'])
        unlisted = plan['deletes_total'] - len(plan['deletes'])
        if unlisted:
            _say(f'... and {unlisted} more planned deletes in {_describe(piece)}')
        if withheld:
            self.report.count('withheld', plan['deletes_total'])
        else:
            self.report.count('deleted', unlisted)

    async def _create(self, plans: list[PlanContextSyncDict]) -> None:
        """Store every planned create as a fresh entry owned by the caller."""
        entries: list[dict[str, object]] = []
        source_ids: list[str] = []
        for piece, plan in zip(self.prepared.slices, plans, strict=True):
            for index in plan['creates']:
                document = self._document(piece, index)
                entries.append(document.create_entry(thread_id=self.config.thread_id, synced_at=self.synced_at))
                source_ids.append(document.source_id)
        if entries:
            done = await self._send(STORE_TOOL, 'entries', entries, source_ids, outcome='created')
            _say(f'created {done} of {len(entries)} documents')

    async def _update(self, plans: list[PlanContextSyncDict]) -> None:
        """Bring every planned update's stored entry in line with its document."""
        entries: list[dict[str, object]] = []
        source_ids: list[str] = []
        for piece, plan in zip(self.prepared.slices, plans, strict=True):
            for update in plan['updates']:
                document = self._document(piece, update['index'])
                entries.append(document.update_entry(
                    context_id=update['context_id'],
                    thread_id=self.config.thread_id,
                    synced_at=self.synced_at,
                    change=update['change'],
                ))
                source_ids.append(document.source_id)
        if entries:
            done = await self._send(UPDATE_TOOL, 'updates', entries, source_ids, outcome='updated')
            _say(f'updated {done} of {len(entries)} documents')

    async def _send(
        self,
        tool: str,
        key: Literal['entries', 'updates'],
        entries: list[dict[str, object]],
        source_ids: list[str],
        *,
        outcome: Literal['created', 'updated'],
    ) -> int:
        """Send entries through a batch tool in packed, non-atomic batches, up to ``--concurrency`` at once.

        The batches start in order, each on a session of its own while it is in flight,
        and their outcomes and pending entries are recorded in batch order when the leg
        ends, so the report reads as a sequential run's however the batches finish. A
        generating server can take minutes per batch, so every batch that finishes prints
        a line with the documents applied so far, its position, its failed entries, the
        batches done and the seconds since the leg began.

        Returns:
            How many entries the server applied.
        """
        started = time.monotonic()
        batches = pack(entries, batch_size=self.config.batch_size, max_request_bytes=self.config.max_request_bytes)
        offsets = list(itertools.accumulate((len(batch) for batch in batches), initial=0))
        settled: dict[int, _BatchOutcome] = {}
        applied = 0

        async def send(session: SyncSession, position: int) -> None:
            nonlocal applied
            batch = batches[position]
            batch_sources = source_ids[offsets[position]:offsets[position + 1]]
            result = await self._send_batch(tool, key, batch, batch_sources, session=session, outcome=outcome)
            settled[position] = result
            done = sum(item.outcome == outcome for item in result.outcomes)
            applied += done
            _say(
                f'{outcome} {applied} of {len(entries)} documents (batch {position + 1} of {len(batches)}, '
                f'{len(batch) - done} failed, {len(settled)} of {len(batches)} batches done, '
                f'{round(time.monotonic() - started)} s)',
            )

        sessions = await self.sessions.take(min(self.config.concurrency, len(batches)))
        try:
            await send_over(sessions, count=len(batches), send=send)
        finally:
            for position in sorted(settled):
                for item in settled[position].outcomes:
                    self.report.add(item)
                self.report.count('generation_pending', settled[position].pending)
        return applied

    @staticmethod
    async def _send_batch(
        tool: str,
        key: Literal['entries', 'updates'],
        batch: list[dict[str, object]],
        batch_sources: list[str],
        *,
        session: SyncSession,
        outcome: Literal['created', 'updated'],
    ) -> _BatchOutcome:
        """Send one batch over a session and return its entries' outcomes in the batch's order.

        The server's per-entry results are matched to the entries by their index, so the
        order in which the server lists them never matters.

        Returns:
            One outcome per entry of the batch, and how many of the written entries the
            server reported with their generation pending.

        Raises:
            IngestExitError: The server answered with something that is not the tool's response.
        """
        result = await session.call_tool(tool, {key: batch, 'atomic': False})
        if isinstance(result, ToolFailure):
            failed = [
                DocumentOutcome(source_id=source_id, outcome='failed', context_id=None, error=result.message)
                for source_id in batch_sources
            ]
            return _BatchOutcome(outcomes=failed, pending=0)
        try:
            response = _BATCH_RESPONSE.validate_python(result)
        except ValidationError as error:
            raise IngestExitError(EXIT_CONFIG, _unexpected(tool, error)) from error
        results = {item['index']: item for item in response['results']}
        outcomes: list[DocumentOutcome] = []
        pending = 0
        for index, source_id in enumerate(batch_sources):
            item = results.get(index)
            if item is None:
                unanswered = f'{tool} returned no result for this entry'
                outcomes.append(DocumentOutcome(source_id=source_id, outcome='failed', context_id=None, error=unanswered))
                continue
            applied = item['success']
            outcomes.append(DocumentOutcome(
                source_id=source_id,
                outcome=outcome if applied else 'failed',
                context_id=item['context_id'],
                error=None if applied else item['error'],
            ))
            if applied and item.get('generation_status') == 'pending':
                pending += 1
        return _BatchOutcome(outcomes=outcomes, pending=pending)

    async def _delete(self) -> None:
        """Plan every range again and delete its planned deletes until none remain, unless deletes are withheld."""
        before = self.report.counts['deleted']
        for piece in self.prepared.slices:
            plan = await self._plan(piece)
            if self.prepared.withhold_deletes:
                self._list_deletes(piece, plan)
            else:
                await self._delete_range(piece, plan)
        if self.prepared.withhold_deletes:
            if self.report.counts['withheld']:
                _say(f'withheld {self.report.counts["withheld"]} planned deletes: a directory could not be listed')
        elif self.report.counts['deleted'] > before:
            _say(f'deleted {self.report.counts["deleted"] - before} entries')

    async def _delete_range(self, piece: ManifestSlice, plan: PlanContextSyncDict) -> None:
        """Delete the listed deletes of a range and plan again, until the plan holds none or a round removes nothing."""
        while plan['deletes_total'] > 0:
            listed = plan['deletes']
            arguments: dict[str, object] = {'context_ids': [delete['context_id'] for delete in listed]}
            result = await self.sessions.first.call_tool(DELETE_TOOL, arguments)
            if isinstance(result, ToolFailure):
                for delete in listed:
                    self._add(delete['source_id'], 'failed', context_id=delete['context_id'], error=result.message)
                return
            plan = await self._plan(piece)
            remaining = {delete['context_id'] for delete in plan['deletes']}
            removed = [delete for delete in listed if delete['context_id'] not in remaining]
            for delete in removed:
                self._add(delete['source_id'], 'deleted', context_id=delete['context_id'])
            if not removed:
                for delete in listed:
                    self._add(delete['source_id'], 'failed', context_id=delete['context_id'], error=_NOT_DELETED)
                self.report.fail(
                    f'the deletes in {_describe(piece)} made no progress: {plan["deletes_total"]} planned deletes remain',
                    exit_code=EXIT_PARTIAL,
                )
                return

    async def _verify(self) -> None:
        """Plan every range a last time; anything left to create, update or delete means the run did not converge."""
        creates = updates = deletes = 0
        for piece in self.prepared.slices:
            plan = await self._plan(piece)
            creates += len(plan['creates'])
            updates += len(plan['updates'])
            if not self.prepared.withhold_deletes:
                deletes += plan['deletes_total']
        if creates or updates or deletes:
            self.report.fail(
                f'{self.config.thread_id} did not converge on the corpus: the final plan still holds {creates} creates, '
                f'{updates} updates and {deletes} deletes; rerun the sync',
                exit_code=EXIT_PARTIAL,
            )
        else:
            _say(f'verified: {self.config.thread_id} matches the corpus')

    def _document(self, piece: ManifestSlice, index: int) -> Document:
        """Return the document a manifest index of a range names."""
        return self.prepared.documents[cast('str', piece.items[index]['source_id'])]

    def _add(
        self,
        source_id: str,
        outcome: Literal['created', 'updated', 'deleted', 'unchanged', 'failed'],
        *,
        context_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record what happened to one document or stored entry."""
        self.report.add(DocumentOutcome(source_id=source_id, outcome=outcome, context_id=context_id, error=error))


def _say(line: str) -> None:
    """Print one line of the run's progress."""
    print(line, flush=True)


def _describe(piece: ManifestSlice) -> str:
    """Return the source-id range of a manifest slice as messages name it."""
    if piece.source_id_after is None and piece.source_id_through is None:
        return 'the corpus'
    after = repr(piece.source_id_after) if piece.source_id_after is not None else 'the start'
    through = repr(piece.source_id_through) if piece.source_id_through is not None else 'the end'
    return f'the source ids after {after} through {through}'


def _delete_line(delete: SyncDeleteDict) -> str:
    """Return a planned delete as a progress line names it."""
    return f'{delete["source_id"]} {delete["context_id"]} ({delete["reason"]})'


def _unexpected(tool: str, error: ValidationError) -> str:
    """Return the message for a response of a tool that does not have the tool's shape."""
    problems = '; '.join(
        f'{".".join(str(part) for part in problem["loc"]) or "result"}: {problem["msg"]}'
        for problem in error.errors()[:_NAMED_PROBLEMS]
    )
    return f'the server answered {tool} with an unexpected result ({problems}): is it a context server of this version?'
