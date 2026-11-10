"""Tests for the generation block of the database statistics on both backends.

``get_database_statistics`` reports ``generation``, the number of entries the caller may
read whose generated outputs are still outstanding (``pending``, a running job included)
and whose generation gave up (``failed``). SQLite matches the job rows against the readable
parent ids, as it does the tag and image rows; PostgreSQL joins them to the routed entry
relation under the READ predicate.
"""

import sqlite3
from typing import cast

import pytest

from app.access_scope import AccessMode
from app.access_scope import build_access_predicate
from app.access_scope import build_readable_parent_predicate
from app.backends.base import MAINTENANCE
from app.backends.base import StorageBackend
from app.backends.base import base_relation
from app.repositories.statistics_repository import StatisticsRepository
from tests.helpers import LOCAL_SCOPE
from tests.repositories._statistics_recording import RecordingPgBackend
from tests.repositories._statistics_recording import RecordingSqliteBackend

_GENERATION_COLUMNS = 'SELECT gj.status AS status, COUNT(*) AS count FROM'
_GROUP_BY_STATUS = 'GROUP BY gj.status'

# Entries by id: their owner and the status of their generation job, None for no job.
_ENTRIES: tuple[tuple[str, str, str | None], ...] = (
    ('0190abcdef1234567890abcd00000001', 'local', 'pending'),
    ('0190abcdef1234567890abcd00000002', 'local', 'running'),
    ('0190abcdef1234567890abcd00000003', 'local', 'failed'),
    ('0190abcdef1234567890abcd00000004', 'local', 'failed'),
    ('0190abcdef1234567890abcd00000005', 'local', None),
    ('0190abcdef1234567890abcd00000006', 'someone-else', 'pending'),
    ('0190abcdef1234567890abcd00000007', 'someone-else', 'failed'),
)


def _insert_entries_with_jobs(conn: sqlite3.Connection) -> None:
    """Store every entry of :data:`_ENTRIES` as private, with its generation job."""
    for context_id, owner, status in _ENTRIES:
        conn.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
            "VALUES (?, 'thread', 'agent', 'text', ?, ?)",
            (context_id, f'Entry {context_id}', owner),
        )
        if status is not None:
            conn.execute('INSERT INTO generation_jobs (context_entry_id, status) VALUES (?, ?)', (context_id, status))


@pytest.mark.asyncio
async def test_no_job_reports_nothing_outstanding(stats_repo: StatisticsRepository) -> None:
    """A database without generation jobs reports zero pending and zero failed entries."""
    stats = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)

    assert stats['generation'] == {'pending': 0, 'failed': 0}


@pytest.mark.asyncio
async def test_counts_the_jobs_of_readable_entries_with_running_as_pending(
    stats_test_db: StorageBackend, stats_repo: StatisticsRepository,
) -> None:
    """The caller's pending and running jobs count as pending, its failed jobs as failed, and nobody else's count."""
    await stats_test_db.execute_write(_insert_entries_with_jobs, access_scope=MAINTENANCE)

    stats = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)

    assert stats['generation'] == {'pending': 2, 'failed': 2}


@pytest.mark.asyncio
async def test_sqlite_matches_the_jobs_against_the_readable_parent_ids(stats_test_db: StorageBackend) -> None:
    """SQLite counts the job rows whose entry id is among the readable parent ids, grouped by status."""
    await stats_test_db.execute_write(_insert_entries_with_jobs, access_scope=MAINTENANCE)
    backend = RecordingSqliteBackend(stats_test_db)

    stats = await StatisticsRepository(cast(StorageBackend, backend)).get_database_statistics(scope=LOCAL_SCOPE)

    parents = build_readable_parent_predicate(
        LOCAL_SCOPE, child_key='gj.context_entry_id', parent_key='id', backend_type='sqlite',
    )
    readable_jobs = f'{base_relation("generation_jobs", alias="gj")}{parents.where_clause()}'
    assert f'{_GENERATION_COLUMNS} {readable_jobs} {_GROUP_BY_STATUS}' in backend.queries, backend.queries
    assert stats['generation'] == {'pending': 2, 'failed': 2}


@pytest.mark.asyncio
async def test_postgresql_joins_the_jobs_to_the_routed_entries_under_read() -> None:
    """PostgreSQL counts the job rows joined to the routed entry relation under the READ predicate."""
    backend = RecordingPgBackend()

    stats = await StatisticsRepository(cast(StorageBackend, backend)).get_database_statistics(scope=LOCAL_SCOPE)

    read = build_access_predicate(LOCAL_SCOPE, mode=AccessMode.READ, backend_type='postgresql', outer='ce')
    jobs = base_relation('generation_jobs', alias='gj')
    entries = base_relation('context_entries', alias='ce')
    readable_jobs = f'{jobs} JOIN {entries} ON ce.id = gj.context_entry_id{read.where_clause()}'
    expected = f'{_GENERATION_COLUMNS} {readable_jobs} {_GROUP_BY_STATUS}'
    assert expected in backend.connection.queries, backend.connection.queries
    assert stats['generation'] == {'pending': 0, 'failed': 0}
