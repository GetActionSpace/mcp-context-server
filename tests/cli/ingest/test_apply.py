"""Tests for the legs of a sync run: plan, creates, updates, deletes and the final verification.

Each run talks to an in-memory server that plans with the server's own diff, so a
sequence of runs over a changing directory shows what every leg sends and that the
thread converges on the corpus.
"""

import os
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import httpx2
import pytest

from app.cli.ingest import corpus
from app.cli.ingest.apply import RunConfig
from app.cli.ingest.apply import run_sync
from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import EXIT_TRANSIENT
from app.cli.ingest.client import IngestClient
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import TokenFileAuth
from app.cli.ingest.client import ToolFailure
from app.cli.ingest.corpus import DirectoryFailure
from app.cli.ingest.manifest import meta_digest
from app.cli.ingest.manifest import request_bytes
from app.cli.ingest.report import RunReport
from app.ingest import INGEST_KEY
from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import ingest_block_error
from tests.cli.ingest._fake_server import FakeServer
from tests.cli.ingest._fake_server import Json
from tests.cli.ingest._stand_in import Exchange
from tests.cli.ingest._stand_in import failing
from tests.cli.ingest._stand_in import ingest_client
from tests.cli.ingest._stand_in import stream_ending
from tests.cli.ingest._sync_run import DELETE
from tests.cli.ingest._sync_run import PLAN
from tests.cli.ingest._sync_run import SERVER
from tests.cli.ingest._sync_run import STORE
from tests.cli.ingest._sync_run import THREAD_ID
from tests.cli.ingest._sync_run import UPDATE
from tests.cli.ingest._sync_run import managed_metadata
from tests.cli.ingest._sync_run import sources_with
from tests.cli.ingest._sync_run import sync_directory
from tests.cli.ingest._sync_run import write_document


def _sent(server: FakeServer, tool: str, key: str) -> list[Json]:
    """Return every entry the run sent through a batch tool, in order."""
    return [entry for call in server.calls_to(tool) for entry in cast('list[Json]', call.arguments[key])]


def _block(entry: Json) -> Json:
    """Return the ingest block of a sent entry."""
    return cast('Json', cast('Json', entry['metadata'])[INGEST_KEY])


def _failures(report: RunReport) -> list[tuple[str, str | None]]:
    """Return the failed documents with their errors."""
    return [(item.source_id, item.error) for item in report.outcomes if item.outcome == 'failed']


def _refuse_listing(monkeypatch: pytest.MonkeyPatch, refused: Path) -> None:
    """Make os.scandir fail for one directory, as a directory without read permission does."""
    real_scandir = os.scandir

    def scandir(path: str) -> Iterator[os.DirEntry[str]]:
        if Path(path) == refused:
            raise PermissionError(13, 'Permission denied', path)
        return real_scandir(path)

    monkeypatch.setattr(os, 'scandir', scandir)


@pytest.fixture
def server() -> FakeServer:
    """Return an empty in-memory server."""
    return FakeServer()


class TestCreates:
    """New documents are stored as fresh entries, never merged, with the visibility the digest covers."""

    @pytest.mark.asyncio
    async def test_bootstrap_stores_every_document_without_deduplication(self, server: FakeServer, tmp_path: Path) -> None:
        """Every create goes through a non-atomic batch with deduplicate false and an explicit visibility."""
        write_document(tmp_path, 'a.md', '# Alpha\n\nfirst')
        write_document(tmp_path, 'guides/b.md', 'second')

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [call.arguments['atomic'] for call in server.calls_to(STORE)] == [False]
        entries = _sent(server, STORE, 'entries')
        assert [_block(entry)['source_id'] for entry in entries] == ['a.md', 'guides/b.md']
        for entry in entries:
            assert entry['deduplicate'] is False
            assert entry['visibility'] == 'private'
            assert entry['thread_id'] == THREAD_ID
            assert entry['source'] == 'user'
            assert ingest_block_error(_block(entry)) is None
            assert _block(entry)['scope'] == THREAD_ID
        assert sorted(server.by_source(THREAD_ID)) == ['a.md', 'guides/b.md']
        assert sources_with(report, 'created') == ['a.md', 'guides/b.md']
        assert {item.context_id for item in report.outcomes} == set(server.entries)
        assert server.calls_to(DELETE) == []
        assert report.errors == []

    @pytest.mark.asyncio
    async def test_sent_visibility_is_the_hashed_visibility(self, server: FakeServer, tmp_path: Path) -> None:
        """The digest each entry carries covers exactly the metadata, tags and visibility it is sent with."""
        write_document(tmp_path, 'a.md', 'alpha')

        report = await sync_directory(server, tmp_path, visibility='public', tags=('Guide', 'ops'), metadata={'team': 'docs'})

        assert report.exit_code == EXIT_CONVERGED
        (entry,) = _sent(server, STORE, 'entries')
        metadata = {key: value for key, value in cast('Json', entry['metadata']).items() if key != INGEST_KEY}
        assert entry['visibility'] == 'public'
        assert metadata == {'title': 'a', 'team': 'docs'}
        assert _block(entry)['meta_digest'] == meta_digest(metadata, cast('list[str]', entry['tags']), 'public')

    @pytest.mark.asyncio
    async def test_creates_are_packed_by_count_and_by_bytes(self, server: FakeServer, tmp_path: Path) -> None:
        """A batch holds at most --batch-size entries and never more bytes than the request budget."""
        for name in 'abcde':
            write_document(tmp_path, f'{name}.md', name * 2000)

        by_count = await sync_directory(server, tmp_path, batch_size=2)

        assert by_count.exit_code == EXIT_CONVERGED
        assert [len(cast('list[Json]', call.arguments['entries'])) for call in server.calls_to(STORE)] == [2, 2, 1]

        bytes_server = FakeServer()
        by_bytes = await sync_directory(bytes_server, tmp_path, max_request_bytes=6000)

        assert by_bytes.exit_code == EXIT_CONVERGED
        batches = [cast('list[Json]', call.arguments['entries']) for call in bytes_server.calls_to(STORE)]
        assert len(batches) == 3
        assert all(request_bytes(batch) <= 6000 for batch in batches)
        assert len(bytes_server.by_source(THREAD_ID)) == 5

    @pytest.mark.asyncio
    async def test_a_refused_entry_fails_alone_and_leaves_the_run_unconverged(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """A per-entry failure is that document's failure; the rest are stored and the final plan still shows it."""
        write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        server.failing_sources['b.md'] = 'Generation failed after exhausting configured retries'

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert sources_with(report, 'created') == ['a.md']
        assert _failures(report) == [('b.md', 'Generation failed after exhausting configured retries')]
        assert any('did not converge' in error for error in report.errors)

    @pytest.mark.asyncio
    async def test_a_refused_batch_call_fails_its_entries_and_the_run_goes_on(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """A batch the server or a proxy refused fails its own entries; the next batch is still sent."""
        write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        server.fail_next(STORE, ToolFailure('the request was too large for the server (HTTP 413)'))

        report = await sync_directory(server, tmp_path, batch_size=1)

        assert report.exit_code == EXIT_PARTIAL
        assert len(server.calls_to(STORE)) == 2
        assert _failures(report) == [('a.md', 'the request was too large for the server (HTTP 413)')]
        assert sources_with(report, 'created') == ['b.md']

    @pytest.mark.asyncio
    async def test_an_entry_the_response_does_not_answer_fails(self, server: FakeServer, tmp_path: Path) -> None:
        """An entry without a per-entry result is a failure of that document."""
        write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        server.answer_next(STORE, {
            'success': True, 'total': 2, 'succeeded': 1, 'failed': 0, 'message': 'Stored 1',
            'results': [{
                'index': 0, 'success': True, 'context_id': '0190' + '0' * 28, 'error': None, 'generation_status': 'ready',
            }],
        })

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert sources_with(report, 'created') == ['a.md']
        assert _failures(report) == [('b.md', 'store_context_batch returned no result for this entry')]

    @pytest.mark.asyncio
    async def test_a_batch_response_of_another_shape_exits_2(self, server: FakeServer, tmp_path: Path) -> None:
        """A batch answer without the batch response's fields ends the run as a configuration error."""
        write_document(tmp_path, 'a.md', 'alpha')
        server.answer_next(STORE, {'stored': 1})

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any('store_context_batch with an unexpected result' in error for error in report.errors)


class TestUpdates:
    """A changed document updates its stored entry, replacing the metadata in full."""

    @pytest.mark.asyncio
    async def test_a_rerun_without_changes_writes_nothing(self, server: FakeServer, tmp_path: Path) -> None:
        """Every document is unchanged, so no batch tool is called."""
        write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path)
        server.calls.clear()

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert {call.tool for call in server.calls} == {'preflight', PLAN}
        assert sources_with(report, 'unchanged') == ['a.md', 'b.md']

    @pytest.mark.asyncio
    async def test_a_text_change_sends_the_text_and_the_full_metadata(self, server: FakeServer, tmp_path: Path) -> None:
        """The update carries the text, the whole metadata with a fresh block, the tags and the visibility."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        await sync_directory(server, tmp_path)
        (context_id,) = server.entries
        path.write_text('alpha, revised', encoding='utf-8')

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [call.arguments['atomic'] for call in server.calls_to(UPDATE)] == [False]
        (update,) = _sent(server, UPDATE, 'updates')
        assert set(update) == {'context_id', 'text', 'metadata', 'tags', 'visibility'}
        assert (update['context_id'], update['text'], update['visibility']) == (context_id, 'alpha, revised', 'private')
        assert ingest_block_error(_block(update)) is None
        assert server.entries[context_id].text == 'alpha, revised'
        assert [(item.source_id, item.context_id) for item in report.outcomes if item.outcome == 'updated'] == [
            ('a.md', context_id),
        ]

    @pytest.mark.asyncio
    async def test_a_metadata_change_sends_no_text(self, server: FakeServer, tmp_path: Path) -> None:
        """A tag change replaces the metadata, tags and visibility without regenerating the text."""
        write_document(tmp_path, 'a.md', 'alpha')
        await sync_directory(server, tmp_path)

        report = await sync_directory(server, tmp_path, tags=('guide',))

        assert report.exit_code == EXIT_CONVERGED
        (update,) = _sent(server, UPDATE, 'updates')
        assert set(update) == {'context_id', 'metadata', 'tags', 'visibility'}
        assert update['tags'] == ['guide']

    @pytest.mark.asyncio
    async def test_a_text_and_metadata_change_sends_both(self, server: FakeServer, tmp_path: Path) -> None:
        """A change of both sends the text together with the full metadata."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        await sync_directory(server, tmp_path)
        path.write_text('alpha, revised', encoding='utf-8')

        report = await sync_directory(server, tmp_path, tags=('guide',))

        assert report.exit_code == EXIT_CONVERGED
        (update,) = _sent(server, UPDATE, 'updates')
        assert (update['text'], update['tags']) == ('alpha, revised', ['guide'])

    @pytest.mark.asyncio
    async def test_a_refused_update_is_a_failure(self, server: FakeServer, tmp_path: Path) -> None:
        """A per-entry not-authorized answer fails that document."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        await sync_directory(server, tmp_path)
        (context_id,) = server.entries
        server.failing_updates[context_id] = f'Not authorized to modify context entry {context_id}'
        path.write_text('alpha, revised', encoding='utf-8')

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert _failures(report) == [('a.md', f'Not authorized to modify context entry {context_id}')]


class TestDeletes:
    """Entries of removed documents and duplicates are deleted by id, after every create and update."""

    @pytest.mark.asyncio
    async def test_a_removed_document_is_deleted_by_id_after_the_updates(self, server: FakeServer, tmp_path: Path) -> None:
        """The delete names the entry's id alone and follows the update of the remaining document."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        removed = write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path)
        (removed_id,) = [entry.context_id for entry in server.by_source(THREAD_ID)['b.md']]
        removed.unlink()
        path.write_text('alpha, revised', encoding='utf-8')
        server.calls.clear()

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        tools = [call.tool for call in server.calls]
        assert tools.index(UPDATE) < tools.index(DELETE)
        assert [call.arguments for call in server.calls_to(DELETE)] == [{'context_ids': [removed_id]}]
        assert [(item.source_id, item.context_id) for item in report.outcomes if item.outcome == 'deleted'] == [
            ('b.md', removed_id),
        ]
        assert sorted(server.by_source(THREAD_ID)) == ['a.md']

    @pytest.mark.asyncio
    async def test_deletes_repeat_until_the_plan_lists_none(self, server: FakeServer, tmp_path: Path) -> None:
        """A plan lists at most 100 deletes, so the leg deletes, plans again and deletes the rest."""
        write_document(tmp_path, 'a.md', 'alpha')
        for number in range(150):
            server.seed(thread_id=THREAD_ID, text=f'gone {number}', metadata=managed_metadata(f'gone/{number:03}.md'))

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [len(cast('list[str]', call.arguments['context_ids'])) for call in server.calls_to(DELETE)] == [100, 50]
        assert report.counts['deleted'] == 150
        assert sorted(server.by_source(THREAD_ID)) == ['a.md']

    @pytest.mark.asyncio
    async def test_each_range_is_planned_with_its_bounds_and_deletes_only_its_own_entries(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """A small request budget splits the manifest into ranges, and a removal is deleted once, by its range."""
        for name in 'abcdef':
            write_document(tmp_path, f'{name}.md', name)
        await sync_directory(server, tmp_path, max_request_bytes=2600)
        (removed_id,) = [entry.context_id for entry in server.by_source(THREAD_ID)['c.md']]
        (tmp_path / 'c.md').unlink()
        server.calls.clear()

        report = await sync_directory(server, tmp_path, max_request_bytes=2600)

        assert report.exit_code == EXIT_CONVERGED
        bounds = [
            (call.arguments.get('source_id_after'), call.arguments.get('source_id_through')) for call in server.calls_to(PLAN)
        ]
        assert bounds[0][0] is None
        assert bounds[0][1] is not None
        assert any(after is not None and through is None for after, through in bounds)
        assert [call.arguments for call in server.calls_to(DELETE)] == [{'context_ids': [removed_id]}]
        assert sorted(server.by_source(THREAD_ID)) == ['a.md', 'b.md', 'd.md', 'e.md', 'f.md']

    @pytest.mark.asyncio
    async def test_the_delete_loop_stops_when_a_round_makes_no_progress(self, server: FakeServer, tmp_path: Path) -> None:
        """A delete that removes nothing is not repeated, and the run reports it as not converged."""
        write_document(tmp_path, 'a.md', 'alpha')
        stuck = server.seed(thread_id=THREAD_ID, text='gone', metadata=managed_metadata('gone.md'))
        server.undeletable.add(stuck)

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert [call.arguments for call in server.calls_to(DELETE)] == [{'context_ids': [stuck]}]
        assert any('no progress' in error for error in report.errors)
        assert any('did not converge' in error for error in report.errors)
        assert sources_with(report, 'deleted') == []

    @pytest.mark.asyncio
    async def test_a_duplicate_is_deleted_and_the_newest_copy_survives(self, server: FakeServer, tmp_path: Path) -> None:
        """Two stored copies of one document converge on one: the plan's survivor."""
        write_document(tmp_path, 'a.md', 'alpha')
        await sync_directory(server, tmp_path)
        (original,) = server.entries
        newer = server.duplicate(original)

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [entry.context_id for entry in server.by_source(THREAD_ID)['a.md']] == [newer]
        assert [(item.source_id, item.context_id) for item in report.outcomes if item.outcome == 'deleted'] == [
            ('a.md', original),
        ]

    @pytest.mark.asyncio
    async def test_a_refused_delete_call_fails_its_entries(self, server: FakeServer, tmp_path: Path) -> None:
        """A delete the server refuses fails the entries it named and leaves them stored."""
        write_document(tmp_path, 'a.md', 'alpha')
        removed = write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path)
        removed.unlink()
        server.fail_next(DELETE, ToolFailure('Not authorized to delete context entries'))

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert _failures(report) == [('b.md', 'Not authorized to delete context entries')]
        assert sorted(server.by_source(THREAD_ID)) == ['a.md', 'b.md']


class TestUnusableFiles:
    """A file the run cannot use never turns into a deletion, and an over-long path is never sent."""

    @pytest.mark.asyncio
    async def test_an_unreadable_stored_document_is_kept(self, server: FakeServer, tmp_path: Path) -> None:
        """An undecodable file is a keep item: its stored entry is neither updated nor deleted."""
        write_document(tmp_path, 'a.md', 'alpha')
        broken = write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path)
        broken.write_bytes(b'\xff\xfe not utf-8 \x80')
        server.calls.clear()

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        manifests = [cast('list[Json]', call.arguments['manifest']) for call in server.calls_to(PLAN)]
        assert {'source_id': 'b.md', 'keep': True} in manifests[0]
        assert server.calls_to(UPDATE) == []
        assert server.calls_to(DELETE) == []
        assert [entry.text for entry in server.by_source(THREAD_ID)['b.md']] == ['beta']
        assert sources_with(report, 'kept') == ['b.md']
        assert not any('did not converge' in error for error in report.errors)

    @pytest.mark.asyncio
    async def test_an_overlong_path_is_never_sent(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A source id above the bound is in no manifest and no batch, and the run names it."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        too_long = 'x' * (MAX_SOURCE_ID_LENGTH + 1)

        def walk(
            root: Path, *, includes: Sequence[str], excludes: Sequence[str],
        ) -> tuple[list[tuple[str, Path]], list[DirectoryFailure]]:
            del root, includes, excludes
            return [('a.md', path), (too_long, path)], []

        monkeypatch.setattr(corpus, 'walk', walk)

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        manifests = [cast('list[Json]', call.arguments['manifest']) for call in server.calls_to(PLAN)]
        assert {item['source_id'] for manifest in manifests for item in manifest} == {'a.md'}
        assert [_block(entry)['source_id'] for entry in _sent(server, STORE, 'entries')] == ['a.md']
        assert sources_with(report, 'failed') == [too_long]
        assert any(too_long in error for error in report.errors)


class TestDirectoryFailures:
    """A directory the walk cannot list withholds every delete of the run, and nothing else."""

    @pytest.mark.asyncio
    async def test_creates_and_updates_run_and_every_delete_is_withheld(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run creates and updates, never calls the delete tool, counts the withheld deletes and names the directory."""
        top = write_document(tmp_path, 'top.md', 'top')
        write_document(tmp_path, 'locked/inner.md', 'inner')
        gone = write_document(tmp_path, 'gone.md', 'gone')
        await sync_directory(server, tmp_path)
        gone.unlink()
        top.write_text('top, revised', encoding='utf-8')
        write_document(tmp_path, 'new.md', 'new')
        _refuse_listing(monkeypatch, tmp_path / 'locked')

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert server.calls_to(DELETE) == []
        assert (report.counts['withheld'], report.counts['updated'], report.counts['created']) == (2, 1, 1)
        assert 'directory not listed: locked: Permission denied' in report.errors
        assert not any('did not converge' in error for error in report.errors)
        assert sorted(server.by_source(THREAD_ID)) == ['gone.md', 'locked/inner.md', 'new.md', 'top.md']

    @pytest.mark.asyncio
    async def test_a_dry_run_reports_the_deletes_as_withheld(
        self,
        server: FakeServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A dry run under a directory failure prints the planned deletes as withheld and writes nothing."""
        write_document(tmp_path, 'top.md', 'top')
        write_document(tmp_path, 'locked/inner.md', 'inner')
        await sync_directory(server, tmp_path)
        (inner_id,) = [entry.context_id for entry in server.by_source(THREAD_ID)['locked/inner.md']]
        _refuse_listing(monkeypatch, tmp_path / 'locked')
        capsys.readouterr()

        report = await sync_directory(server, tmp_path, dry_run=True)

        assert report.exit_code == EXIT_PARTIAL
        assert report.counts['withheld'] == 1
        assert sources_with(report, 'deleted') == []
        assert f'withheld delete locked/inner.md {inner_id} (absent)' in capsys.readouterr().out.splitlines()
        assert inner_id in server.entries

    @pytest.mark.asyncio
    @pytest.mark.parametrize('dry_run', [False, True])
    async def test_withheld_deletes_past_the_listed_ones_count_as_withheld_alone(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool,
    ) -> None:
        """Every planned delete, listed or not, is withheld, and none counts as deleted."""
        write_document(tmp_path, 'top.md', 'top')
        write_document(tmp_path, 'locked/inner.md', 'inner')
        await sync_directory(server, tmp_path)
        for number in range(104):
            server.seed(thread_id=THREAD_ID, text=f'gone {number}', metadata=managed_metadata(f'gone/{number:03}.md'))
        _refuse_listing(monkeypatch, tmp_path / 'locked')

        report = await sync_directory(server, tmp_path, dry_run=dry_run)

        assert report.exit_code == EXIT_PARTIAL
        assert (report.counts['withheld'], report.counts['deleted']) == (105, 0)
        assert server.calls_to(DELETE) == []


class TestRunEnding:
    """A failure that ends the run stops it at once, and nothing it may have done is sent again."""

    @pytest.mark.asyncio
    async def test_a_timed_out_call_is_never_resent(self, server: FakeServer, tmp_path: Path) -> None:
        """A store whose answer never came ends the run; the next run finds what it committed and converges."""
        write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        timeout = IngestExitError(EXIT_TRANSIENT, 'the call to store_context_batch got no answer within --call-timeout')
        server.fail_next(STORE, timeout, commit=True)

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_TRANSIENT
        assert report.errors == [timeout.message]
        assert len(server.calls_to(STORE)) == 1
        assert server.calls[-1].tool == STORE
        server.calls.clear()

        rerun = await sync_directory(server, tmp_path)

        assert rerun.exit_code == EXIT_CONVERGED
        assert server.calls_to(STORE) == []
        assert {source: len(entries) for source, entries in server.by_source(THREAD_ID).items()} == {'a.md': 1, 'b.md': 1}

    @pytest.mark.asyncio
    async def test_a_refused_plan_exits_2_before_any_write(self, server: FakeServer, tmp_path: Path) -> None:
        """The plan tool's error is a plan error, reported, and nothing is written."""
        write_document(tmp_path, 'a.md', 'alpha')
        refusal = "manifest repeats source_id values (each may appear once): 'a.md'"
        server.fail_next(PLAN, ToolFailure(refusal))

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any(refusal in error for error in report.errors)
        assert server.calls_to(STORE) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('answer', 'message'),
        [
            ({'thread_id': THREAD_ID, 'creates': []}, 'with an unexpected result'),
            (
                {
                    'thread_id': THREAD_ID, 'creates': [5], 'updates': [], 'deletes': [], 'deletes_total': 0,
                    'unchanged_count': 0, 'kept_count': 0, 'foreign_count': 0,
                },
                'a manifest index that names no document',
            ),
        ],
    )
    async def test_a_plan_of_another_shape_exits_2(
        self, server: FakeServer, tmp_path: Path, answer: Json, message: str,
    ) -> None:
        """A plan without the plan's fields, or naming an index past the manifest, ends the run before any write."""
        write_document(tmp_path, 'a.md', 'alpha')
        server.answer_next(PLAN, answer)

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any(message in error for error in report.errors)
        assert server.calls_to(STORE) == []

    @pytest.mark.asyncio
    async def test_a_failed_preflight_ends_the_run_with_its_code(self, server: FakeServer, tmp_path: Path) -> None:
        """A server without the sync tools is refused before the first plan."""
        write_document(tmp_path, 'a.md', 'alpha')
        server.preflight_exit = IngestExitError(EXIT_CONFIG, 'the server does not offer plan_context_sync')

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert report.errors == ['the server does not offer plan_context_sync']
        assert server.calls_to(PLAN) == []

    @pytest.mark.asyncio
    async def test_the_token_file_is_read_before_the_server_is_contacted(self, server: FakeServer, tmp_path: Path) -> None:
        """An empty token file exits 2 without a connection; a usable one becomes the session's auth."""
        corpus_root = tmp_path / 'corpus'
        write_document(corpus_root, 'a.md', 'alpha')
        token = tmp_path / 'token'
        token.write_text('\n', encoding='utf-8')

        refused = await sync_directory(server, corpus_root, token_file=token)

        assert refused.exit_code == EXIT_CONFIG
        assert refused.errors == [f'the token file is empty: {token}']
        assert server.connections == []

        token.write_text('ingest-token\n', encoding='utf-8')
        accepted = await sync_directory(server, corpus_root, token_file=token)

        assert accepted.exit_code == EXIT_CONVERGED
        (auth,) = server.connections
        assert isinstance(auth, TokenFileAuth)
        assert auth.path == token

    @pytest.mark.asyncio
    async def test_a_request_budget_below_one_plan_call_exits_2(self, server: FakeServer, tmp_path: Path) -> None:
        """A --max-request-bytes too small for a plan request is a configuration error found before contact."""
        write_document(tmp_path, 'a.md', 'alpha')

        report = await sync_directory(server, tmp_path, max_request_bytes=500)

        assert report.exit_code == EXIT_CONFIG
        assert any('--max-request-bytes' in error for error in report.errors)
        assert server.connections == []

    @pytest.mark.asyncio
    async def test_a_connection_lost_during_a_call_exits_69_with_a_report(self, exchange: Exchange, tmp_path: Path) -> None:
        """A connection dropped during the plan call ends the run with exit 69, and the report says why."""
        write_document(tmp_path, 'a.md', 'alpha')
        exchange.script(PLAN, failing(httpx2.ReadError))

        def connect(config: RunConfig, auth: httpx2.Auth | None) -> IngestClient:
            return ingest_client(exchange, auth=auth, call_timeout=config.call_timeout, server=config.server)

        report = await run_sync(RunConfig(server=SERVER, corpus='handbook', source_dir=tmp_path), connect=connect)

        assert report.exit_code == EXIT_TRANSIENT
        assert report.errors == [
            (
                f'the call to {PLAN} lost its connection to {SERVER} (ReadError): '
                "the call's outcome is unknown; rerun the sync, which plans from what the server stored"
            ),
        ]
        assert [seen.tool for seen in exchange.seen if seen.tool is not None] == [PLAN]

    @pytest.mark.asyncio
    async def test_a_response_stream_lost_during_a_call_exits_69_with_a_report(
        self, exchange: Exchange, tmp_path: Path,
    ) -> None:
        """A plan call whose SSE stream ends after the server accepted it ends the run with exit 69 and says why."""
        write_document(tmp_path, 'a.md', 'alpha')
        exchange.script(PLAN, stream_ending(httpx2.RemoteProtocolError))

        def connect(config: RunConfig, auth: httpx2.Auth | None) -> IngestClient:
            return ingest_client(exchange, auth=auth, call_timeout=config.call_timeout, server=config.server)

        report = await run_sync(RunConfig(server=SERVER, corpus='handbook', source_dir=tmp_path), connect=connect)

        assert report.exit_code == EXIT_TRANSIENT
        assert report.errors == [
            (
                f'the call to {PLAN} lost its response stream from {SERVER} after the server accepted it '
                "(SSE stream ended without a response): the call's outcome is unknown; rerun the sync, which plans "
                'from what the server stored; when calls keep losing their stream after the same time, raise the '
                'response timeout of the proxy in front of the server to at least --call-timeout; see '
                "docs/knowledge-base.md, 'Limits'"
            ),
        ]


class TestEmptyCorpus:
    """Deleting a whole corpus takes an explicit flag."""

    @pytest.mark.asyncio
    async def test_an_empty_walk_exits_2_without_contact(self, server: FakeServer, tmp_path: Path) -> None:
        """A directory without documents is refused before the server is contacted."""
        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any('--allow-empty-corpus' in error for error in report.errors)
        assert server.connections == []

    @pytest.mark.asyncio
    async def test_the_flag_deletes_every_managed_entry(self, server: FakeServer, tmp_path: Path) -> None:
        """With the flag the empty manifest is confirmed and every entry of the corpus is deleted."""
        first = write_document(tmp_path, 'a.md', 'alpha')
        second = write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path)
        first.unlink()
        second.unlink()
        server.calls.clear()

        report = await sync_directory(server, tmp_path, allow_empty_corpus=True)

        assert report.exit_code == EXIT_CONVERGED
        plans = server.calls_to(PLAN)
        assert all(call.arguments['manifest'] == [] for call in plans)
        assert all(call.arguments['allow_empty_manifest'] is True for call in plans)
        assert server.by_source(THREAD_ID) == {}
        assert report.counts['deleted'] == 2
