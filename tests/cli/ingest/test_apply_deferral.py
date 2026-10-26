"""Tests for a sync run that defers generation: the key on every write, the preflight and the pending count.

Each run talks to an in-memory server that answers every write with the entry's
generation state, so the tests read what the run sent, what it asked the preflight
to check and how many written entries its report counts as pending.
"""

from pathlib import Path
from typing import cast

import pytest

from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from tests.cli.ingest._fake_server import FakeServer
from tests.cli.ingest._fake_server import Json
from tests.cli.ingest._sync_run import STORE
from tests.cli.ingest._sync_run import UPDATE
from tests.cli.ingest._sync_run import sources_with
from tests.cli.ingest._sync_run import sync_directory
from tests.cli.ingest._sync_run import write_document


def _sent(server: FakeServer, tool: str, key: str) -> list[Json]:
    """Return every entry the run sent through a batch tool, in order."""
    return [entry for call in server.calls_to(tool) for entry in cast('list[Json]', call.arguments[key])]


@pytest.fixture
def server() -> FakeServer:
    """Return an empty in-memory server."""
    return FakeServer()


class TestDeferredWrites:
    """A deferring run asks the server to defer generation on every create and update it sends."""

    @pytest.mark.asyncio
    async def test_every_create_asks_to_defer_generation(self, server: FakeServer, tmp_path: Path) -> None:
        """Each create entry carries defer_generation beside deduplicate false."""
        write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')

        report = await sync_directory(server, tmp_path, defer_generation=True)

        assert report.exit_code == EXIT_CONVERGED
        entries = _sent(server, STORE, 'entries')
        assert [(entry['deduplicate'], entry['defer_generation']) for entry in entries] == [(False, True), (False, True)]

    @pytest.mark.asyncio
    async def test_every_update_asks_to_defer_generation(self, server: FakeServer, tmp_path: Path) -> None:
        """A text update and a metadata update both carry defer_generation."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path)
        path.write_text('alpha, revised', encoding='utf-8')

        report = await sync_directory(server, tmp_path, defer_generation=True, tags=('guide',))

        assert report.exit_code == EXIT_CONVERGED
        updates = _sent(server, UPDATE, 'updates')
        assert [('text' in update, update['defer_generation']) for update in updates] == [(True, True), (False, True)]

    @pytest.mark.asyncio
    async def test_a_run_without_deferral_never_sends_the_key(self, server: FakeServer, tmp_path: Path) -> None:
        """Without the flag no create or update entry names defer_generation."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        await sync_directory(server, tmp_path)
        path.write_text('alpha, revised', encoding='utf-8')

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        sent = [*_sent(server, STORE, 'entries'), *_sent(server, UPDATE, 'updates')]
        assert len(sent) == 2
        assert all('defer_generation' not in entry for entry in sent)


class TestPreflight:
    """The run tells the preflight whether it defers, so a server that cannot defer stops it before any write."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('defer_generation', [True, False])
    async def test_the_preflight_learns_whether_the_run_defers(
        self, server: FakeServer, tmp_path: Path, defer_generation: bool,
    ) -> None:
        """The first call of the run is the preflight, told whether the run defers generation."""
        write_document(tmp_path, 'a.md', 'alpha')

        await sync_directory(server, tmp_path, defer_generation=defer_generation)

        assert (server.calls[0].tool, server.calls[0].arguments) == ('preflight', {'defer_generation': defer_generation})


class TestPendingCount:
    """The report counts the written entries the server reported pending; deferral never changes the exit code."""

    @pytest.mark.asyncio
    async def test_deferred_creates_are_counted_pending_and_the_run_converges(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """Every deferred create is pending, and the final plan still finds the thread matching the corpus."""
        for name in ('a.md', 'b.md', 'c.md'):
            write_document(tmp_path, name, f'{name} body')

        report = await sync_directory(server, tmp_path, defer_generation=True, batch_size=2)

        assert (report.exit_code, report.errors) == (EXIT_CONVERGED, [])
        assert (report.counts['created'], report.counts['generation_pending']) == (3, 3)

    @pytest.mark.asyncio
    async def test_the_count_follows_what_the_server_reports(self, server: FakeServer, tmp_path: Path) -> None:
        """A generating text update leaves its entry ready, while a metadata update keeps a pending entry pending."""
        path = write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path, defer_generation=True)
        path.write_text('alpha, revised', encoding='utf-8')

        report = await sync_directory(server, tmp_path, tags=('guide',))

        assert report.exit_code == EXIT_CONVERGED
        assert (report.counts['updated'], report.counts['generation_pending']) == (2, 1)

    @pytest.mark.asyncio
    async def test_a_run_without_deferral_counts_nothing_pending(self, server: FakeServer, tmp_path: Path) -> None:
        """A generating server answers every write ready."""
        write_document(tmp_path, 'a.md', 'alpha')

        report = await sync_directory(server, tmp_path)

        assert (report.counts['created'], report.counts['generation_pending']) == (1, 0)

    @pytest.mark.asyncio
    async def test_a_failed_entry_is_never_counted_pending(self, server: FakeServer, tmp_path: Path) -> None:
        """An entry the server refused is a failure of the run, never a pending entry."""
        write_document(tmp_path, 'a.md', 'alpha')
        write_document(tmp_path, 'b.md', 'beta')
        server.failing_sources['b.md'] = 'refused'

        report = await sync_directory(server, tmp_path, defer_generation=True)

        assert report.exit_code == EXIT_PARTIAL
        assert sources_with(report, 'failed') == ['b.md']
        assert (report.counts['created'], report.counts['generation_pending']) == (1, 1)

    @pytest.mark.asyncio
    async def test_a_dry_run_counts_nothing_pending(self, server: FakeServer, tmp_path: Path) -> None:
        """A dry run writes nothing, so nothing it planned is pending."""
        write_document(tmp_path, 'a.md', 'alpha')

        report = await sync_directory(server, tmp_path, defer_generation=True, dry_run=True)

        assert (report.counts['created'], report.counts['generation_pending']) == (1, 0)
        assert server.entries == {}
