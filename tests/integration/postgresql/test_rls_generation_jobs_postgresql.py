"""The row-level security policies of the generation_jobs table against a real PostgreSQL server.

A job follows its entry. Every check connects as the request login role ``mcp_app`` over a
raw asyncpg connection, or as ``mcp_outsider``, binds an identity with the request profile's
bind statement inside a transaction that is rolled back, and issues SQL without any
application predicate, so the jobs it reaches are the policies' answer alone. The matrix
database of :mod:`tests.integration.postgresql._rls_matrix` gives every seed entry a pending
job; a check that needs an entry without one deletes it through the table owner's channel.
Every refusal is asserted as SQLSTATE 42501.
"""

import uuid
from collections.abc import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio

from app.access_scope import AccessScope
from tests.integration.postgresql._rls_matrix import MATRIX_SCOPES
from tests.integration.postgresql._rls_matrix import entry_uuid
from tests.integration.postgresql._rls_matrix import rls_matrix_db
from tests.integration.postgresql._rls_matrix import scoped_transaction
from tests.integration.postgresql.conftest import RlsUrls
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import ScopedDb

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

# Every scope that writes a job carries the write right, as the write tools bind it, so each
# refusal below comes from the rule it names; DAVE only reads.
ALICE = AccessScope('alice', frozenset(), may_write=True)
BOB = AccessScope('bob', frozenset(), may_write=True)
CAROL = AccessScope('carol', frozenset({'team-x'}), may_write=True)
DAVE = AccessScope('dave', frozenset())
REFUSED = asyncpg.exceptions.InsufficientPrivilegeError

# A pending job on one entry ($1), as a deferred write enqueues it.
INSERT_JOB = 'INSERT INTO generation_jobs (context_entry_id) VALUES ($1)'
SELECT_JOB = 'SELECT status, attempts, claim_token FROM generation_jobs WHERE context_entry_id = $1'
DELETE_JOB = 'DELETE FROM generation_jobs WHERE context_entry_id = $1'
# What a worker writes to every job: a claim, a failure and a release each change these columns.
UPDATE_JOBS = (
    "UPDATE generation_jobs SET status = 'running', attempts = attempts + 1, claim_token = 'probe', "
    "last_error = 'probe', available_at = now()"
)


@pytest_asyncio.fixture(params=['fp32'])
async def matrix_db(
    request: pytest.FixtureRequest, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """The policy matrix database in the requested layout (fp32 unless a test parametrizes it).

    Yields:
        The seeded database with a pending job on every seed entry and the policies installed.
    """
    async with rls_matrix_db(pg_rls_db, layout=request.param, monkeypatch=monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def app_conn(matrix_db: ScopedDb, pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A raw connection of the request login role ``mcp_app`` to the matrix database.

    Yields:
        The open connection, closed afterwards.
    """
    _ = matrix_db
    conn = await asyncpg.connect(pg_rls_db.app)
    try:
        yield conn
    finally:
        await conn.close()


async def _without_job(db: ScopedDb, label: str) -> uuid.UUID:
    """Delete the job of the entry ``label`` through the owner channel, so an insert of its job would otherwise pass.

    Returns:
        The entry's id.
    """
    await db.execute('DELETE FROM generation_jobs WHERE context_entry_id = ?', (db.ids[label],))
    return entry_uuid(db, label)


async def _job(conn: asyncpg.Connection, entry: uuid.UUID) -> tuple[object, ...] | None:
    """Return the status, attempts and claim token of the job of ``entry`` that ``conn`` reads, or None."""
    row = await conn.fetchrow(SELECT_JOB, entry)
    return None if row is None else tuple(row)


async def _job_count(db: ScopedDb, label: str) -> int:
    """Return how many jobs the entry ``label`` holds, read as the owner."""
    rows = await db.fetch_all('SELECT count(*) FROM generation_jobs WHERE context_entry_id = ?', (db.ids[label],))
    return int(str(rows[0][0]))


class TestGenerationJobs:
    """A job follows its entry: a writer replaces it, a reader only reads it, and only the table owner updates it."""

    @pytest.mark.asyncio
    async def test_a_writer_replaces_the_job_of_an_entry_it_may_modify(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """The owner and each write grantee delete the job and enqueue a new one in one transaction, as a text update does."""
        cases = [
            (ALICE, 'alice_private'),
            (BOB, 'bob_private'),
            (BOB, 'alice_private_bob_write'),
            (CAROL, 'alice_public_carol_write'),
        ]
        for scope, label in cases:
            entry = entry_uuid(matrix_db, label)
            async with scoped_transaction(app_conn, scope):
                assert await app_conn.execute(DELETE_JOB, entry) == 'DELETE 1', label
                assert await app_conn.execute(INSERT_JOB, entry) == 'INSERT 0 1', label
                assert await _job(app_conn, entry) == ('pending', 0, None), label

    @pytest.mark.asyncio
    async def test_a_reader_sees_the_job_but_neither_enqueues_nor_removes_one(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """A user read grant, a group read grant and a public row each show the job and refuse every write to it."""
        cases = [
            (BOB, 'alice_private_bob_read'),
            (CAROL, 'alice_private_team_read'),
            (DAVE, 'alice_public_dave_read'),
            (MATRIX_SCOPES['mallory'], 'alice_public'),
        ]
        for scope, label in cases:
            entry = entry_uuid(matrix_db, label)
            async with scoped_transaction(app_conn, scope):
                assert await _job(app_conn, entry) == ('pending', 0, None), label
                assert await app_conn.execute(DELETE_JOB, entry) == 'DELETE 0', label
            assert await _job_count(matrix_db, label) == 1, label

            entry = await _without_job(matrix_db, label)
            async with scoped_transaction(app_conn, scope):
                with pytest.raises(REFUSED, match='row-level security policy for table "generation_jobs"'):
                    await app_conn.execute(INSERT_JOB, entry)
            assert await _job_count(matrix_db, label) == 0, label

    @pytest.mark.parametrize('matrix_db', ['fp32', 'compressed'], indirect=True)
    @pytest.mark.asyncio
    async def test_no_request_identity_updates_a_job_and_the_table_owner_does(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """No identity, the owner of every entry included, updates a job it reads; the table owner updates them all."""
        for name, scope in MATRIX_SCOPES.items():
            async with scoped_transaction(app_conn, scope):
                assert await app_conn.fetchval('SELECT count(*) FROM generation_jobs') > 0, name
                assert await app_conn.execute(UPDATE_JOBS) == 'UPDATE 0', name

        await matrix_db.execute(UPDATE_JOBS)

        rows = await matrix_db.fetch_all('SELECT context_entry_id, status, attempts, claim_token FROM generation_jobs')
        assert {matrix_db.labels_of([str(row[0])])[0]: row[1:] for row in rows} == dict.fromkeys(
            SEED_LABELS, ('running', 1, 'probe'),
        )

    @pytest.mark.asyncio
    async def test_the_outsider_reaches_no_job_under_any_identity(
        self, matrix_db: ScopedDb, pg_rls_db: RlsUrls,
    ) -> None:
        """Every privilege on the table, granted out of band to a role outside the policy role, reads and writes no job."""
        target = await _without_job(matrix_db, 'alice_public')
        admin = await asyncpg.connect(pg_rls_db.admin)
        try:
            await admin.execute('GRANT ALL ON generation_jobs TO mcp_outsider')
        finally:
            await admin.close()

        conn = await asyncpg.connect(pg_rls_db.outsider)
        try:
            for name, scope in MATRIX_SCOPES.items():
                async with scoped_transaction(conn, scope):
                    assert await conn.fetchval('SELECT count(*) FROM generation_jobs') == 0, name
                    assert await conn.execute(UPDATE_JOBS) == 'UPDATE 0', name
                    assert await conn.execute('DELETE FROM generation_jobs') == 'DELETE 0', name
                async with scoped_transaction(conn, scope):
                    with pytest.raises(REFUSED, match='row-level security policy for table "generation_jobs"'):
                        await conn.execute(INSERT_JOB, target)
        finally:
            await conn.close()

        rows = await matrix_db.fetch_all('SELECT context_entry_id, status FROM generation_jobs')
        assert sorted(matrix_db.labels_of(str(row[0]) for row in rows)) == sorted(set(SEED_LABELS) - {'alice_public'})
        assert {row[1] for row in rows} == {'pending'}
