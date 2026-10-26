"""Tests for the response helpers in app.tools._responses.

The generation status reads run against a SQLite database built from the server's
schema; the message builders are pure.
"""

import sqlite3
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Literal

import pytest

from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.tools._responses import attach_generation_status
from app.tools._responses import build_batch_store_response_message
from app.tools._responses import build_batch_update_response_message
from app.tools._responses import build_store_response_message
from app.tools._responses import build_update_response_message
from app.tools._responses import generation_statuses
from app.types import GenerationStatus
from tests.helpers import LOCAL_SCOPE
from tests.repositories._access_scope_layouts import SQLITE_TEST_VARIABLE_LIMIT
from tests.repositories._access_scope_layouts import limit_sqlite_variables


async def _insert_entries(backend: StorageBackend, count: int) -> list[str]:
    """Store ``count`` bare entries of the local principal as the table owner and return their ids in order."""
    ids = [generate_id() for _ in range(count)]

    def _insert(conn: sqlite3.Connection) -> None:
        conn.executemany(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
            "VALUES (?, 'responses', 'agent', 'text', ?, 'local')",
            [(context_id, f'entry {position}') for position, context_id in enumerate(ids)],
        )

    await backend.execute_write(_insert, access_scope=MAINTENANCE)
    return ids


async def _insert_jobs(backend: StorageBackend, statuses: dict[str, str]) -> None:
    """Give each entry a generation job in the stored status mapped to its id."""

    def _insert(conn: sqlite3.Connection) -> None:
        conn.executemany(
            'INSERT INTO generation_jobs (context_entry_id, status) VALUES (?, ?)', list(statuses.items()),
        )

    await backend.execute_write(_insert, access_scope=MAINTENANCE)


def _record_state_reads(monkeypatch: pytest.MonkeyPatch, repos: RepositoryContainer) -> list[list[str]]:
    """Record the ids of every state read the job repository runs, letting each read through."""
    reads: list[list[str]] = []
    read_states = repos.generation_jobs.states_for_ids

    async def _recording_read(context_ids: Sequence[str], *, scope: BackendAccess) -> dict[str, GenerationStatus]:
        reads.append(list(context_ids))
        return await read_states(context_ids, scope=scope)

    monkeypatch.setattr(repos.generation_jobs, 'states_for_ids', _recording_read)
    return reads


class TestGenerationStatuses:
    """Test generation_statuses, the one generation state read of a response."""

    @pytest.mark.asyncio
    async def test_reports_the_state_of_every_entry(self, async_db_initialized: StorageBackend) -> None:
        """No job reads ready, a waiting or running job pending, and a failed job failed."""
        ready, waiting, running, failed = await _insert_entries(async_db_initialized, 4)
        await _insert_jobs(async_db_initialized, {waiting: 'pending', running: 'running', failed: 'failed'})

        statuses = await generation_statuses(
            [ready, waiting, running, failed], RepositoryContainer(async_db_initialized), scope=LOCAL_SCOPE,
        )

        assert statuses == {ready: 'ready', waiting: 'pending', running: 'pending', failed: 'failed'}

    @pytest.mark.asyncio
    async def test_reads_each_entry_once_in_one_read(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An entry a response carries several times, as grep match rows do, is read once."""
        first, second = await _insert_entries(async_db_initialized, 2)
        await _insert_jobs(async_db_initialized, {second: 'pending'})
        repos = RepositoryContainer(async_db_initialized)
        reads = _record_state_reads(monkeypatch, repos)

        statuses = await generation_statuses([first, second, first, second], repos, scope=LOCAL_SCOPE)

        assert reads == [[first, second]]
        assert statuses == {first: 'ready', second: 'pending'}

    @pytest.mark.asyncio
    async def test_reads_more_entries_than_one_statement_binds(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A response of more entries than the lowest SQLite variable limit gets the state of every one."""
        limit_sqlite_variables(monkeypatch, SQLITE_TEST_VARIABLE_LIMIT)
        ids = await _insert_entries(async_db_initialized, SQLITE_TEST_VARIABLE_LIMIT + 1)
        await _insert_jobs(async_db_initialized, {ids[0]: 'pending', ids[-1]: 'failed'})

        statuses = await generation_statuses(ids, RepositoryContainer(async_db_initialized), scope=LOCAL_SCOPE)

        assert list(statuses) == ids
        assert (statuses[ids[0]], statuses[ids[-1]]) == ('pending', 'failed')
        assert {statuses[context_id] for context_id in ids[1:-1]} == {'ready'}


class TestAttachGenerationStatus:
    """Test attach_generation_status, which sets the state on each entry of a response."""

    @pytest.mark.asyncio
    async def test_sets_the_state_of_every_entry_from_one_read(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each entry gains its own generation_status and keeps every other field."""
        ready, failed = await _insert_entries(async_db_initialized, 2)
        await _insert_jobs(async_db_initialized, {failed: 'failed'})
        repos = RepositoryContainer(async_db_initialized)
        reads = _record_state_reads(monkeypatch, repos)
        entries: list[dict[str, Any]] = [{'id': ready, 'thread_id': 'responses'}, {'id': failed, 'thread_id': 'responses'}]

        await attach_generation_status(entries, repos, scope=LOCAL_SCOPE)

        assert entries == [
            {'id': ready, 'thread_id': 'responses', 'generation_status': 'ready'},
            {'id': failed, 'thread_id': 'responses', 'generation_status': 'failed'},
        ]
        assert reads == [[ready, failed]]

    @pytest.mark.asyncio
    async def test_reads_nothing_for_a_response_without_entries(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty result, such as a search without hits, runs no statement."""
        reads: list[object] = []
        execute_read = async_db_initialized.execute_read

        async def _recording_execute_read(operation: Callable[..., object], *, access_scope: BackendAccess) -> object:
            reads.append(operation)
            return await execute_read(operation, access_scope=access_scope)

        monkeypatch.setattr(async_db_initialized, 'execute_read', _recording_execute_read)

        await attach_generation_status([], RepositoryContainer(async_db_initialized), scope=LOCAL_SCOPE)

        assert reads == []


class TestBuildStoreResponseMessage:
    """Test build_store_response_message shared function."""

    def test_basic_stored(self) -> None:
        """Basic store with no extras produces simple message."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False, generation_status='ready',
        )
        assert msg == 'Context stored'

    def test_stored_with_images(self) -> None:
        """Store with images includes image count."""
        msg = build_store_response_message(
            action='stored', image_count=3,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False, generation_status='ready',
        )
        assert msg == 'Context stored with 3 images'

    def test_embedding_generated_and_stored(self) -> None:
        """Embedding generated and stored shows 'embedding generated'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=True, embedding_stored=True,
            summary_generated=False, summary_preserved=False, generation_status='ready',
        )
        assert 'embedding generated' in msg
        assert 'not stored' not in msg

    def test_embedding_generated_not_stored(self) -> None:
        """Embedding generated but not stored shows duplicate message."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=True, embedding_stored=False,
            summary_generated=False, summary_preserved=False, generation_status='ready',
        )
        assert 'embedding generated but not stored - duplicate' in msg

    def test_summary_generated(self) -> None:
        """Summary generated shows 'summary generated'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=True, summary_preserved=False, generation_status='ready',
        )
        assert 'summary generated' in msg

    def test_summary_preserved(self) -> None:
        """Summary preserved shows 'summary preserved'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=True, generation_status='ready',
        )
        assert 'summary preserved' in msg

    def test_all_parts(self) -> None:
        """All flags set produces message with all parts."""
        msg = build_store_response_message(
            action='stored', image_count=2,
            embedding_generated=True, embedding_stored=True,
            summary_generated=True, summary_preserved=False, generation_status='ready',
        )
        assert 'Context stored with 2 images' in msg
        assert 'embedding generated' in msg
        assert 'summary generated' in msg

    def test_no_parts(self) -> None:
        """No flags set produces no parenthetical."""
        msg = build_store_response_message(
            action='updated', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False, generation_status='ready',
        )
        assert msg == 'Context updated'
        assert '(' not in msg

    @pytest.mark.parametrize('status', ['pending', 'failed'])
    def test_an_outstanding_generation_state_is_named(self, status: Literal['pending', 'failed']) -> None:
        """A store whose entry still waits for its outputs, or gave up on them, says so."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False, generation_status=status,
        )
        assert msg == f'Context stored (generation {status})'

    def test_a_ready_entry_names_no_generation_state(self) -> None:
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=True, summary_preserved=False, generation_status='ready',
        )
        assert msg == 'Context stored (summary generated)'


class TestBuildUpdateResponseMessage:
    """Test build_update_response_message shared function."""

    def test_basic_update(self) -> None:
        """Basic update with no extras."""
        msg = build_update_response_message(
            updated_fields_count=3,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=False,
            generation_status='ready',
        )
        assert msg == 'Successfully updated 3 field(s)'

    def test_embedding_regenerated(self) -> None:
        """Embedding regenerated shows in message."""
        msg = build_update_response_message(
            updated_fields_count=2,
            embedding_generated=True,
            summary_generated=False,
            summary_cleared=False,
            generation_status='ready',
        )
        assert 'embedding regenerated' in msg

    def test_summary_regenerated(self) -> None:
        """Summary regenerated shows in message."""
        msg = build_update_response_message(
            updated_fields_count=2,
            embedding_generated=False,
            summary_generated=True,
            summary_cleared=False,
            generation_status='ready',
        )
        assert 'summary regenerated' in msg

    def test_summary_cleared(self) -> None:
        """Summary cleared shows in message."""
        msg = build_update_response_message(
            updated_fields_count=1,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=True,
            generation_status='ready',
        )
        assert 'summary cleared' in msg

    def test_all_parts(self) -> None:
        """All flags set produces all parts."""
        msg = build_update_response_message(
            updated_fields_count=5,
            embedding_generated=True,
            summary_generated=True,
            summary_cleared=False,
            generation_status='ready',
        )
        assert 'embedding regenerated' in msg
        assert 'summary regenerated' in msg
        assert '5 field(s)' in msg

    def test_a_pending_generation_follows_the_cleared_summary(self) -> None:
        """A deferred text update clears the old summary and leaves the new one to its job."""
        msg = build_update_response_message(
            updated_fields_count=2,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=True,
            generation_status='pending',
        )
        assert msg == 'Successfully updated 2 field(s) (summary cleared, generation pending)'

    def test_a_failed_generation_is_named(self) -> None:
        msg = build_update_response_message(
            updated_fields_count=1,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=False,
            generation_status='failed',
        )
        assert msg == 'Successfully updated 1 field(s) (generation failed)'


class TestBuildBatchStoreResponseMessage:
    """Test build_batch_store_response_message shared function."""

    def test_basic_batch_message(self) -> None:
        """Basic batch store message."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0, embeddings_stored_count=0,
            summaries_generated_count=0, summaries_preserved_count=0,
            generation_pending_count=0, generation_failed_count=0,
        )
        assert msg == 'Stored 3/3 entries successfully'

    def test_with_embeddings_not_stored(self) -> None:
        """Batch with some embeddings not stored shows duplicate count."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=3, embeddings_stored_count=1,
            summaries_generated_count=0, summaries_preserved_count=0,
            generation_pending_count=0, generation_failed_count=0,
        )
        assert 'embeddings generated (2 not stored - duplicates)' in msg

    def test_with_summaries_preserved(self) -> None:
        """Batch with preserved summaries shows count."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0, embeddings_stored_count=0,
            summaries_generated_count=2, summaries_preserved_count=1,
            generation_pending_count=0, generation_failed_count=0,
        )
        assert 'summaries generated' in msg
        assert 'summaries preserved' in msg

    def test_outstanding_generation_counts_are_named(self) -> None:
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0, embeddings_stored_count=0,
            summaries_generated_count=0, summaries_preserved_count=0,
            generation_pending_count=2, generation_failed_count=1,
        )
        assert msg == 'Stored 3/3 entries successfully (generation pending for 2 entries, generation failed for 1 entry)'


class TestBuildBatchUpdateResponseMessage:
    """Test build_batch_update_response_message shared function."""

    def test_basic_batch_update(self) -> None:
        """Basic batch update message."""
        msg = build_batch_update_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0,
            summaries_generated_count=0,
            summaries_cleared_count=0,
            generation_pending_count=0,
            generation_failed_count=0,
        )
        assert msg == 'Updated 3/3 entries successfully'

    def test_with_summaries_cleared(self) -> None:
        """Batch with cleared summaries shows count."""
        msg = build_batch_update_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0,
            summaries_generated_count=0,
            summaries_cleared_count=2,
            generation_pending_count=0,
            generation_failed_count=0,
        )
        assert 'summaries cleared' in msg

    def test_outstanding_generation_counts_are_named(self) -> None:
        msg = build_batch_update_response_message(
            succeeded=2, total=2,
            embeddings_generated_count=0,
            summaries_generated_count=0,
            summaries_cleared_count=2,
            generation_pending_count=1,
            generation_failed_count=0,
        )
        assert msg == 'Updated 2/2 entries successfully (summaries cleared, generation pending for 1 entry)'
