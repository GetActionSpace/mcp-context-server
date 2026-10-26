"""Tests for the generation job executor in app/generation_jobs/executor.py.

Most cases run a claimed job against a seeded SQLite database the server's own preparation
built, with the generation replaced by a fake: the legs follow the outputs the entry lacks,
a completion stores every output and deletes the job in one transaction, a text change, a
lost claim or a deleted entry discards the outputs, a failed attempt returns the job to the
queue with a backoff or fails it at the cap, and the lease is renewed while the outputs are
generated, a lost renewal cancelling the generation at once.
"""

import asyncio
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastmcp.exceptions import ToolError

import app.generation_jobs.executor as executor
from app import digests
from app.backends.base import MAINTENANCE
from app.generation_jobs.executor import JobOutcome
from app.generation_jobs.executor import run_job
from app.repositories.context_repository.records import GenerationSource
from app.repositories.generation_job_repository import ClaimedJob
from app.repositories.generation_job_repository import RecordedFailure
from app.repositories.generation_job_repository import retry_delay_s
from tests.generation_jobs.conftest import InstallGeneration
from tests.generation_jobs.conftest import chunk_for
from tests.generation_jobs.conftest import node_for
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._generation_job_cases import CLOCK_SLACK_S
from tests.repositories._generation_job_cases import MAX_ATTEMPTS
from tests.repositories._generation_job_cases import claim
from tests.repositories._generation_job_cases import enqueue
from tests.repositories._generation_job_cases import read_job
from tests.repositories._generation_job_cases import replace_job
from tests.repositories._generation_job_cases import require_job
from tests.repositories.context_repository._generation_cases import make_update_time_comparable
from tests.repositories.context_repository._generation_cases import read_entry

pytestmark = pytest.mark.usefixtures('active_legs')

LONG_TEXT = '# Section\n\n' + 'A sentence long enough to be summarized. ' * 30
OTHER_TOKEN = 'f' * 32


async def _claimed(
    db: ScopedDb,
    label: str,
    *,
    text: str = LONG_TEXT,
    angle: float | None = None,
    summary: str | None = None,
    index_node: bool = False,
) -> ClaimedJob:
    """Store an entry owned by alice with the given outputs, enqueue its job and claim it."""
    context_id = await db.add_entry(
        label, owner='alice', visibility='private', text=text, angle=angle, summary=summary, index_node=index_node,
    )
    await enqueue(db, context_id)
    job = await claim(db)
    assert job is not None
    assert job.context_id == context_id
    return job


async def _run(db: ScopedDb, job: ClaimedJob, *, max_attempts: int = MAX_ATTEMPTS) -> JobOutcome:
    """Run the job on the database's request channel as the maintenance identity."""
    return await run_job(db.repos, job, access=MAINTENANCE, max_attempts=max_attempts)


async def _steal_claim(db: ScopedDb, context_id: str) -> None:
    """Give the entry's job another claim token, as a reclaim by another worker does."""
    await db.execute('UPDATE generation_jobs SET claim_token = ? WHERE context_entry_id = ?', [OTHER_TOKEN, context_id])


class TestLegs:
    """The legs a job runs are the outputs its entry lacks."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('angle', 'summary', 'index_node', 'legs'),
        [
            pytest.param(None, None, False, {'embedding': True, 'summary': True, 'nodes': True}, id='nothing-stored'),
            pytest.param(0.0, None, False, {'embedding': False, 'summary': True, 'nodes': True}, id='embedded'),
            pytest.param(None, 'Stored.', False, {'embedding': True, 'summary': False, 'nodes': True}, id='summarized'),
            pytest.param(None, None, True, {'embedding': True, 'summary': True, 'nodes': False}, id='with-nodes'),
        ],
    )
    async def test_each_missing_output_runs_its_leg(
        self,
        db: ScopedDb,
        generation: InstallGeneration,
        angle: float | None,
        summary: str | None,
        index_node: bool,
        legs: dict[str, bool],
    ) -> None:
        """An entry's stored outputs switch their legs off; the missing ones run."""
        fake = generation()
        job = await _claimed(db, 'doc', angle=angle, summary=summary, index_node=index_node)

        assert await _run(db, job) is JobOutcome.COMPLETED

        assert fake.calls == [{'text': LONG_TEXT, 'source': 'agent', **legs}]

    @pytest.mark.asyncio
    async def test_a_text_below_the_summary_minimum_runs_no_summary_leg(
        self, db: ScopedDb, generation: InstallGeneration, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A text shorter than SUMMARY_MIN_CONTENT_LENGTH is never summarized, as on the synchronous path."""
        summary = executor.settings.summary.model_copy(update={'min_content_length': len(LONG_TEXT) + 1})
        monkeypatch.setattr(executor, 'settings', executor.settings.model_copy(update={'summary': summary}))
        fake = generation()
        job = await _claimed(db, 'short')

        await _run(db, job)

        assert fake.calls[0]['summary'] is False
        assert fake.calls[0]['embedding'] is True

    @pytest.mark.asyncio
    async def test_inactive_providers_run_no_leg_and_complete_the_job(
        self, db: ScopedDb, generation: InstallGeneration, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a provider no leg runs: the job completes without generating anything."""
        monkeypatch.setattr(executor, 'get_embedding_provider', lambda: None)
        monkeypatch.setattr(executor, 'get_summary_provider', lambda: None)
        monkeypatch.setattr(executor, 'node_layer_active', lambda: False)
        fake = generation()
        job = await _claimed(db, 'doc')

        assert await _run(db, job) is JobOutcome.COMPLETED

        assert fake.calls == []
        assert await read_job(db, job.context_id) is None

    @pytest.mark.asyncio
    async def test_an_entry_holding_every_output_completes_without_generating(
        self, db: ScopedDb, generation: InstallGeneration,
    ) -> None:
        """Nothing to generate: the job is deleted and the entry's update time advances."""
        fake = generation()
        job = await _claimed(db, 'complete', angle=0.0, summary='Stored.', index_node=True)
        await make_update_time_comparable(db, job.context_id)
        before = await read_entry(db, job.context_id)

        assert await _run(db, job) is JobOutcome.COMPLETED

        after = await read_entry(db, job.context_id)
        assert fake.calls == []
        assert await read_job(db, job.context_id) is None
        assert after.summary == 'Stored.'
        assert after.updated_at > before.updated_at


class TestCompletion:
    """A completion stores the generated outputs and deletes the job in one transaction."""

    @pytest.mark.asyncio
    async def test_every_output_is_stored_and_the_job_deleted(self, db: ScopedDb, generation: InstallGeneration) -> None:
        """The embeddings, the summary and the node rows land; the version stays and the update time advances."""
        chunk = await chunk_for(db, LONG_TEXT)
        generation(outputs=([chunk], 'A generated summary.', [node_for(LONG_TEXT)]))
        job = await _claimed(db, 'doc')
        await make_update_time_comparable(db, job.context_id)
        before = await read_entry(db, job.context_id)

        assert await _run(db, job) is JobOutcome.COMPLETED

        after = await read_entry(db, job.context_id)
        source = await db.repos.context.read_generation_source(job.context_id, scope=MAINTENANCE)
        assert source is not None
        assert (source.has_embeddings, source.has_summary, source.node_count) == (True, True, 1)
        assert after.summary == 'A generated summary.'
        assert after.version == before.version
        assert after.updated_at > before.updated_at
        assert await read_job(db, job.context_id) is None

    @pytest.mark.asyncio
    async def test_a_node_leg_without_rows_keeps_the_stored_nodes(self, db: ScopedDb, generation: InstallGeneration) -> None:
        """A node leg that produced nothing leaves the node rows alone and still completes the job."""
        chunk = await chunk_for(db, LONG_TEXT)
        generation(outputs=([chunk], 'A generated summary.', None))
        job = await _claimed(db, 'doc')

        assert await _run(db, job) is JobOutcome.COMPLETED

        source = await db.repos.context.read_generation_source(job.context_id, scope=MAINTENANCE)
        assert source is not None
        assert (source.has_embeddings, source.node_count) == (True, 0)
        assert await read_job(db, job.context_id) is None

    @pytest.mark.asyncio
    async def test_a_fault_after_the_chunks_rolls_every_write_back(
        self, db: ScopedDb, generation: InstallGeneration, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A write failing after the chunks undoes the summary, the chunks and the job deletion, and counts as a failure."""
        chunk = await chunk_for(db, LONG_TEXT)
        generation(outputs=([chunk], 'A generated summary.', [node_for(LONG_TEXT)]))
        monkeypatch.setattr(
            db.repos.index_nodes, 'replace_nodes_for_context', AsyncMock(side_effect=RuntimeError('node write failed')),
        )
        job = await _claimed(db, 'doc')

        assert await _run(db, job) is JobOutcome.RETRYING

        source = await db.repos.context.read_generation_source(job.context_id, scope=MAINTENANCE)
        assert source is not None
        assert (source.has_embeddings, source.has_summary, source.node_count) == (False, False, 0)
        stored = await require_job(db, job.context_id)
        assert (stored.status, stored.attempts, stored.last_error) == ('pending', 1, 'node write failed')


class TestDiscard:
    """Outputs generated for text, a claim or an entry that no longer holds are never written."""

    @pytest.mark.asyncio
    async def test_a_text_change_during_generation_discards_the_outputs(
        self, db: ScopedDb, generation: InstallGeneration,
    ) -> None:
        """The replaced text keeps no summary and no embeddings; its fresh job is left as the update wrote it."""
        context_id = ''

        async def change_text() -> None:
            await db.execute(
                'UPDATE context_entries SET text_content = ?, content_hash = ? WHERE id = ?',
                ['New text.', digests.content_hash('New text.'), context_id],
            )
            await replace_job(db, context_id)

        chunk = await chunk_for(db, LONG_TEXT)
        generation(outputs=([chunk], 'A generated summary.', [node_for(LONG_TEXT)]), during=change_text)
        job = await _claimed(db, 'doc')
        context_id = job.context_id

        assert await _run(db, job) is JobOutcome.DISCARDED

        source = await db.repos.context.read_generation_source(context_id, scope=MAINTENANCE)
        assert source is not None
        assert (source.has_embeddings, source.has_summary, source.node_count) == (False, False, 0)
        replacement = await require_job(db, context_id)
        assert (replacement.status, replacement.attempts, replacement.claim_token) == ('pending', 0, None)

    @pytest.mark.asyncio
    async def test_a_lost_claim_rolls_the_summary_back(self, db: ScopedDb, generation: InstallGeneration) -> None:
        """A job claimed again meanwhile is not deleted, and the summary written before the check is undone."""
        context_id = ''

        async def steal() -> None:
            await _steal_claim(db, context_id)

        generation(outputs=(None, 'A generated summary.', None), during=steal)
        job = await _claimed(db, 'doc')
        context_id = job.context_id
        await make_update_time_comparable(db, context_id)
        before = await read_entry(db, context_id)

        assert await _run(db, job) is JobOutcome.DISCARDED

        after = await read_entry(db, context_id)
        assert (after.summary, after.updated_at) == (None, before.updated_at)
        assert (await require_job(db, context_id)).claim_token == OTHER_TOKEN

    @pytest.mark.asyncio
    async def test_an_entry_deleted_during_generation_discards_the_outputs(
        self, db: ScopedDb, generation: InstallGeneration,
    ) -> None:
        """A deleted entry takes its job with it, and the outputs go nowhere."""
        context_id = ''

        async def delete_entry() -> None:
            await db.execute('DELETE FROM context_entries WHERE id = ?', [context_id])

        generation(outputs=(None, 'A generated summary.', None), during=delete_entry)
        job = await _claimed(db, 'doc')
        context_id = job.context_id

        assert await _run(db, job) is JobOutcome.DISCARDED

        assert await read_job(db, context_id) is None

    @pytest.mark.asyncio
    async def test_a_missing_entry_drops_its_job_without_generating(self, generation: InstallGeneration) -> None:
        """A job whose entry is gone is deleted under its claim and runs no generation."""
        fake = generation()
        repos = MagicMock()
        repos.context.read_generation_source = AsyncMock(return_value=None)
        repos.generation_jobs.delete_job = AsyncMock(return_value=True)
        job = ClaimedJob('0190abcdef1234567890abcd00000001', 1, 'a' * 32)

        assert await run_job(repos, job, access=MAINTENANCE, max_attempts=MAX_ATTEMPTS) is JobOutcome.DISCARDED

        repos.generation_jobs.delete_job.assert_awaited_once_with(job.context_id, job.claim_token, scope=MAINTENANCE)
        assert fake.calls == []


class TestFailure:
    """A failed attempt returns the job to the queue with a backoff, or fails it after the last attempt."""

    @pytest.mark.asyncio
    async def test_a_failed_attempt_returns_the_job_after_the_backoff(
        self, db: ScopedDb, generation: InstallGeneration, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The job is pending again, claimable after the first backoff, with the error kept and a warning logged."""
        generation(error=ToolError('Generation failed after exhausting configured retries: embedding: boom'))
        job = await _claimed(db, 'doc')

        with caplog.at_level(logging.WARNING, logger=executor.__name__):
            assert await _run(db, job) is JobOutcome.RETRYING

        stored = await require_job(db, job.context_id)
        assert (stored.status, stored.attempts, stored.claim_token) == ('pending', 1, None)
        assert stored.last_error is not None
        assert 'embedding: boom' in stored.last_error
        assert abs(stored.available_in_s - retry_delay_s(1)) <= CLOCK_SLACK_S
        warning = next(record for record in caplog.records if record.levelno == logging.WARNING)
        assert job.context_id in warning.getMessage()
        assert f'attempt 1 of {MAX_ATTEMPTS}' in warning.getMessage()

    @pytest.mark.asyncio
    async def test_the_last_failed_attempt_fails_the_job_with_an_error_line(
        self, db: ScopedDb, generation: InstallGeneration, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The job stays failed and an ERROR line names the entry, the error and the remedy."""
        generation(error=ToolError('Generation failed after exhausting configured retries: summary: down'))
        job = await _claimed(db, 'doc')

        with caplog.at_level(logging.ERROR, logger=executor.__name__):
            assert await _run(db, job, max_attempts=1) is JobOutcome.FAILED

        stored = await require_job(db, job.context_id)
        assert (stored.status, stored.attempts, stored.claim_token) == ('failed', 1, None)
        message = next(record for record in caplog.records if record.levelno == logging.ERROR).getMessage()
        assert job.context_id in message
        assert 'summary: down' in message
        assert 'readable and FTS-searchable' in message
        assert 'mcp-context-server-migrate --embed-missing' in message

    @pytest.mark.asyncio
    async def test_the_failure_of_a_replaced_job_is_not_recorded(self, db: ScopedDb, generation: InstallGeneration) -> None:
        """A job replaced while its attempt failed keeps the replacement untouched."""
        context_id = ''

        async def replace() -> None:
            await replace_job(db, context_id)

        generation(error=ToolError('boom'), during=replace)
        job = await _claimed(db, 'doc')
        context_id = job.context_id

        assert await _run(db, job) is JobOutcome.DISCARDED

        replacement = await require_job(db, context_id)
        assert (replacement.status, replacement.attempts, replacement.last_error) == ('pending', 0, None)

    @pytest.mark.asyncio
    async def test_a_read_failure_is_recorded_as_a_failed_attempt(self, generation: InstallGeneration) -> None:
        """A failure before any generation spends the attempt like a generation failure."""
        fake = generation()
        repos = MagicMock()
        repos.context.read_generation_source = AsyncMock(side_effect=RuntimeError('database gone'))
        repos.generation_jobs.record_failure = AsyncMock(
            return_value=RecordedFailure(attempts=1, failed=False, retry_in_s=retry_delay_s(1)),
        )
        job = ClaimedJob('0190abcdef1234567890abcd00000001', 1, 'a' * 32)

        assert await run_job(repos, job, access=MAINTENANCE, max_attempts=MAX_ATTEMPTS) is JobOutcome.RETRYING

        repos.generation_jobs.record_failure.assert_awaited_once_with(
            job.context_id, job.claim_token, 'database gone', max_attempts=MAX_ATTEMPTS, scope=MAINTENANCE,
        )
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_a_failure_that_cannot_be_recorded_is_left_to_the_lease(
        self, generation: InstallGeneration, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When even the failure write fails, the job waits for its lease to end and a warning says so."""
        generation(error=ToolError('boom'))
        repos = MagicMock()
        repos.context.read_generation_source = AsyncMock(
            return_value=GenerationSource(
                text_content=LONG_TEXT, source='agent', content_hash=digests.content_hash(LONG_TEXT),
                has_summary=False, has_embeddings=False, node_count=0,
            ),
        )
        repos.generation_jobs.record_failure = AsyncMock(side_effect=RuntimeError('database gone'))
        job = ClaimedJob('0190abcdef1234567890abcd00000001', 1, 'a' * 32)

        with caplog.at_level(logging.WARNING, logger=executor.__name__):
            outcome = await run_job(repos, job, access=MAINTENANCE, max_attempts=MAX_ATTEMPTS)

        assert outcome is JobOutcome.UNRECORDED
        message = next(record for record in caplog.records if record.levelno == logging.WARNING).getMessage()
        assert 'boom' in message
        assert 'database gone' in message


class TestLease:
    """The lease is renewed while the outputs are generated; a lost renewal ends the generation."""

    @pytest.mark.asyncio
    async def test_the_lease_is_renewed_while_generating(
        self, db: ScopedDb, generation: InstallGeneration, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each renewal interval during a generation extends the claim's lease."""
        monkeypatch.setattr(executor, 'LEASE_RENEWAL_INTERVAL_S', 0.05)
        renew = AsyncMock(wraps=db.repos.generation_jobs.renew_lease)
        monkeypatch.setattr(db.repos.generation_jobs, 'renew_lease', renew)
        generation(delay_s=0.4)
        job = await _claimed(db, 'doc')

        assert await _run(db, job) is JobOutcome.COMPLETED

        assert renew.await_count >= 2
        renew.assert_awaited_with(job.context_id, job.claim_token, lease_s=executor.LEASE_S, scope=MAINTENANCE)

    @pytest.mark.asyncio
    async def test_a_lost_renewal_cancels_the_generation(
        self, db: ScopedDb, generation: InstallGeneration, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A renewal that finds the claim gone cancels the generation and discards the job's run."""
        monkeypatch.setattr(executor, 'LEASE_RENEWAL_INTERVAL_S', 0.05)
        context_id = ''

        async def steal() -> None:
            await _steal_claim(db, context_id)

        fake = generation(during=steal, block=True)
        job = await _claimed(db, 'doc')
        context_id = job.context_id

        outcome = await asyncio.wait_for(_run(db, job), timeout=5)

        assert outcome is JobOutcome.DISCARDED
        assert fake.cancelled is True
        stored = await require_job(db, context_id)
        assert (stored.claim_token, stored.attempts, stored.last_error) == (OTHER_TOKEN, 1, None)

    @pytest.mark.asyncio
    async def test_a_cancelled_run_records_nothing(self, db: ScopedDb, generation: InstallGeneration) -> None:
        """Cancelling the run cancels the generation and leaves the claim for the caller to release."""
        fake = generation(block=True)
        job = await _claimed(db, 'doc')

        running = asyncio.create_task(_run(db, job))
        await asyncio.wait_for(fake.started.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        assert fake.cancelled is True
        stored = await require_job(db, job.context_id)
        assert (stored.status, stored.claim_token, stored.attempts, stored.last_error) == (
            'running', job.claim_token, 1, None,
        )
