"""Two-principal access-scope cases on PostgreSQL, and the checks of the case scaffold itself.

Every case of :mod:`tests.repositories._access_scope_cases` runs once per scope it names, on
a freshly seeded, isolated database of the case's layout on the pgvector container of the
``pg_test_url`` fixture (``@requires_docker_postgres``, skipped cleanly without Docker). The
scaffold checks prove the seed on both layouts and the expected visibility tables against the
access predicate, including the one-parameter ``text[]`` group bind. They also prove the
row-level security scaffold: the ``pg_rls_db`` database is owned by ``mcp_owner`` and
reachable as each test role, and a layout prepared and seeded by its owner through the seed
channel, while the request role ``mcp_app`` holds no table privilege, is read back whole
through the request channel.

Every case runs again under row-level security, on the ``fp32_rls`` and ``compressed_rls``
layouts: ``mcp_owner`` prepares the ``pg_rls_db`` database with ``ENABLE_POSTGRESQL_RLS``,
which grants the policy role and installs the policies and the read view, and seeds it, and
each request scope runs its seams as ``mcp_app`` under the request profile, so every
statement is filtered by the database as well as by the application predicate. The cases
that only row-level security runs join them there (:data:`RLS_CASES`). The cases whose
observable the readable rows alone decide also run with every seam called as the system
scope on a backend that binds the request scope's identity, so the database's answer is the
only filter (:data:`READ_DECIDED_CASES`).
"""

import dataclasses
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import pytest
import pytest_asyncio
from _pytest.mark.structures import ParameterSet

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.backends.postgresql_backend.session_profiles import SessionProfile
from tests.integration.postgresql._rls_matrix import POLICY_ROLE
from tests.integration.postgresql._rls_matrix import PUBLIC_ROWS
from tests.integration.postgresql._rls_profiles import PinnedIdentityProfile
from tests.integration.postgresql._rls_urls import RlsUrls
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SCOPES
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import case_params
from tests.repositories._access_scope_cases import expected_visibility
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import observe_predicate_visibility
from tests.repositories._access_scope_cases import registry_problems
from tests.repositories._access_scope_cases import run_case
from tests.repositories._access_scope_cases import run_case_with_scope
from tests.repositories._access_scope_layouts import assert_seeded_layout
from tests.repositories._access_scope_layouts import postgresql_owner_seeded_db
from tests.repositories._access_scope_layouts import postgresql_rls_scoped_db
from tests.repositories._access_scope_layouts import postgresql_scoped_db
from tests.repositories._access_scope_registry import CASES
from tests.repositories._access_scope_registry import READ_DECIDED_CASES
from tests.repositories._access_scope_registry import RLS_CASES

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


@pytest_asyncio.fixture
async def scoped_db_fp32(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """Seeded isolated database with compression off and the fp32 vector table.

    Yields:
        The seeded database.
    """
    async with postgresql_scoped_db(pg_test_url, 'fp32', monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def scoped_db_compressed(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """Seeded isolated database with compression on from the first start and no fp32 table.

    Yields:
        The seeded database.
    """
    async with postgresql_scoped_db(pg_test_url, 'compressed', monkeypatch) as db:
        yield db


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(CASES, 'fp32', backend='postgresql'))
async def test_access_case_fp32(scoped_db_fp32: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each fp32-layout case yields the expected observable for the scope."""
    await run_case(scoped_db_fp32, case, scope_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(CASES, 'compressed', backend='postgresql'))
async def test_access_case_compressed(scoped_db_compressed: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each compressed-layout case yields the expected observable for the scope."""
    await run_case(scoped_db_compressed, case, scope_name)


class TestSeededLayoutsPostgreSQL:
    """Both layouts hold the seed rows, their children and the layout's vector storage."""

    @pytest.mark.asyncio
    async def test_fp32_layout_holds_the_seed(self, scoped_db_fp32: ScopedDb) -> None:
        """The fp32 layout stores every seed row with fp32 vectors and no compressed table."""
        await assert_seeded_layout(scoped_db_fp32)

    @pytest.mark.asyncio
    async def test_compressed_layout_holds_the_seed(self, scoped_db_compressed: ScopedDb) -> None:
        """The compressed layout stores every seed row with compressed payloads and no fp32 table."""
        await assert_seeded_layout(scoped_db_compressed)

    @pytest.mark.asyncio
    async def test_expected_tables_match_the_access_predicate(self, scoped_db_compressed: ScopedDb) -> None:
        """READABLE, WRITABLE and OWNED are what the access predicate selects for every scope."""
        assert await observe_predicate_visibility(scoped_db_compressed) == expected_visibility()

    @pytest.mark.asyncio
    async def test_raw_write_changes_what_the_predicate_admits(self, scoped_db_compressed: ScopedDb) -> None:
        """A row published through the raw-write helper becomes readable to bob."""
        db = scoped_db_compressed

        await db.execute('UPDATE context_entries SET visibility = ? WHERE id = ?', ('public', db.ids['alice_private']))

        assert await db.labels_selected_by(SCOPES['bob'], AccessMode.READ) == in_seed_order(
            (*READABLE['bob'], 'alice_private'),
        )


def _through_request_channel(db: ScopedDb) -> ScopedDb:
    """Return ``db`` with its raw statements and repository writes sent through its request channel."""
    return dataclasses.replace(db, seed_backend=db.backend, seed=db.repos)


class TestRlsDatabase:
    """``pg_rls_db`` is a fresh database owned by ``mcp_owner``, reachable as each test role."""

    @pytest.mark.asyncio
    async def test_each_url_logs_in_as_its_role_on_the_same_database(self, pg_rls_db: RlsUrls) -> None:
        """The four connection strings log in as mcp_test, mcp_owner, mcp_app and mcp_outsider on one database."""
        sessions: dict[str, tuple[str, str]] = {}
        for name, url in dataclasses.asdict(pg_rls_db).items():
            conn = await asyncpg.connect(url)
            try:
                row = await conn.fetchrow('SELECT current_user, current_database()')
            finally:
                await conn.close()
            assert row is not None
            sessions[name] = (row[0], row[1])

        database = sessions['admin'][1]
        assert database != 'mcp_test'
        assert sessions == {
            'admin': ('mcp_test', database),
            'maintenance': ('mcp_owner', database),
            'app': ('mcp_app', database),
            'outsider': ('mcp_outsider', database),
        }

    @pytest.mark.asyncio
    async def test_database_is_owned_by_mcp_owner_and_carries_only_pgvector(self, pg_rls_db: RlsUrls) -> None:
        """The database belongs to mcp_owner, has the vector extension and holds no relation yet."""
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            row = await conn.fetchrow(
                'SELECT pg_catalog.pg_get_userbyid(d.datdba), '
                "(SELECT COUNT(*) FROM pg_catalog.pg_extension WHERE extname = 'vector'), "
                '(SELECT COUNT(*) FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
                "WHERE n.nspname = 'public') "
                'FROM pg_catalog.pg_database d WHERE d.datname = current_database()',
            )
        finally:
            await conn.close()

        assert row is not None
        assert tuple(row) == ('mcp_owner', 1, 0)


@pytest_asyncio.fixture(params=['fp32', 'compressed'])
async def owner_seeded_db(
    request: pytest.FixtureRequest, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[ScopedDb]:
    """The ``pg_rls_db`` database prepared and seeded by mcp_owner, with mcp_app as the request channel.

    Yields:
        The seeded database, in the fp32 and in the compressed layout.
    """
    async with postgresql_owner_seeded_db(
        request_url=pg_rls_db.app, maintenance_url=pg_rls_db.maintenance, layout=request.param, monkeypatch=monkeypatch,
    ) as db:
        yield db


class TestOwnerSeededLayouts:
    """The seed channel writes as the table owner, and the request channel reads what it wrote."""

    @pytest.mark.asyncio
    async def test_seed_channel_runs_as_the_owner_and_requests_as_the_app_role(self, owner_seeded_db: ScopedDb) -> None:
        """Raw statements of the seed channel run as mcp_owner, those of the request channel as mcp_app."""
        db = owner_seeded_db

        assert await db.fetch_all('SELECT current_user') == [('mcp_owner',)]
        assert await _through_request_channel(db).fetch_all('SELECT current_user') == [('mcp_app',)]

    @pytest.mark.asyncio
    async def test_owner_owns_every_table_and_the_app_role_holds_no_privilege(self, owner_seeded_db: ScopedDb) -> None:
        """The layout belongs to mcp_owner, and mcp_app could not have written a single seed row."""
        db = owner_seeded_db

        owners = await db.fetch_all("SELECT DISTINCT tableowner FROM pg_catalog.pg_tables WHERE schemaname = 'public'")
        app_privileges = await db.fetch_all(
            'SELECT c.relname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
            "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm') "
            "AND pg_catalog.has_table_privilege('mcp_app', c.oid, 'SELECT, INSERT, UPDATE, DELETE')",
        )

        assert owners == [('mcp_owner',)]
        assert app_privileges == []

    @pytest.mark.asyncio
    async def test_request_channel_reads_the_seeded_layout(self, owner_seeded_db: ScopedDb) -> None:
        """Granted SELECT after the seed, mcp_app reads every seed row, child row and the layout's vector storage."""
        db = owner_seeded_db

        await db.execute('GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_app')

        await assert_seeded_layout(_through_request_channel(db))


def _request_scope(scope_name: str) -> AccessScope:
    """Return the request scope named ``scope_name``."""
    scope = SCOPES[scope_name]
    assert isinstance(scope, AccessScope)
    return scope


@asynccontextmanager
async def _rls_scoped_db(
    urls: RlsUrls, layout: Layout, profile: SessionProfile, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """Prepare and seed ``layout`` on the database of ``urls`` with row-level security on, requested as mcp_app.

    Args:
        urls: The ``pg_rls_db`` connection strings.
        layout: The layout to build.
        profile: The session profile of the request backend.
        monkeypatch: Restores the layout's environment and bindings after the test.

    Yields:
        The seeded database: its seed channel runs as mcp_owner, its request channel as mcp_app under ``profile``.
    """
    async with postgresql_rls_scoped_db(
        request_url=urls.app,
        maintenance_url=urls.maintenance,
        policy_role=POLICY_ROLE,
        request_profile=profile,
        layout=layout,
        monkeypatch=monkeypatch,
    ) as db:
        yield db


@pytest_asyncio.fixture
async def scoped_db_fp32_rls(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """The fp32 layout with row-level security on, requested as mcp_app under the request profile.

    Yields:
        The seeded database.
    """
    async with _rls_scoped_db(pg_rls_db, 'fp32', RlsRequestProfile(read_views=RLS_READ_VIEWS), monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def scoped_db_compressed_rls(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """The compressed layout with row-level security on, requested as mcp_app under the request profile.

    Yields:
        The seeded database.
    """
    async with _rls_scoped_db(pg_rls_db, 'compressed', RlsRequestProfile(read_views=RLS_READ_VIEWS), monkeypatch) as db:
        yield db


def _through_seed_channel(db: ScopedDb) -> ScopedDb:
    """Return ``db`` with its seams run on its seed channel."""
    return dataclasses.replace(db, backend=db.seed_backend, repos=db.seed)


def _channel_of(db: ScopedDb, scope_name: str) -> ScopedDb:
    """Return ``db`` with the request channel the scope runs its seams on under row-level security.

    A request scope runs on mcp_app under the request profile, which binds its identity in
    every transaction. The system scope carries no identity, so the request profile binds
    nothing for it and the database admits only public rows; on PostgreSQL the system scope
    serves schema maintenance on the backend that owns the tables, so it runs on the seed
    channel, logged in as mcp_owner.

    Args:
        db: A seeded row-level security layout.
        scope_name: The scope the case runs as.

    Returns:
        ``db`` itself for a request scope, otherwise ``db`` with the seed channel as its request channel.
    """
    if scope_name == 'system':
        return _through_seed_channel(db)
    return db


_RLS_RUN_CASES = (*CASES, *RLS_CASES)


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(_RLS_RUN_CASES, 'fp32', backend='postgresql'))
async def test_access_case_fp32_rls(scoped_db_fp32_rls: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each fp32-layout case yields the expected observable for the scope under row-level security."""
    await run_case(_channel_of(scoped_db_fp32_rls, scope_name), case, scope_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(_RLS_RUN_CASES, 'compressed', backend='postgresql'))
async def test_access_case_compressed_rls(scoped_db_compressed_rls: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each compressed-layout case yields the expected observable for the scope under row-level security."""
    await run_case(_channel_of(scoped_db_compressed_rls, scope_name), case, scope_name)


def _database_alone_params(layout: Layout) -> list[ParameterSet]:
    """Build one parameter set per read-decided case of ``layout`` and request scope it names.

    The system scope is left out: it has no identity to bind.

    Args:
        layout: The layout the parametrized test seeds.

    Returns:
        The parameter sets, with the ids of :func:`case_params`.
    """
    return [param for param in case_params(READ_DECIDED_CASES, layout, backend='postgresql') if param.values[1] != 'system']


async def _run_database_alone(
    urls: RlsUrls, layout: Layout, case: AccessCase, scope_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run ``case`` with every seam called as the system scope on a backend that binds the identity of ``scope_name``.

    The system scope renders no application predicate, so the rows the seams see are the
    database's answer for the bound identity alone; it must equal what the scope observes
    with the application predicate in place.

    Args:
        urls: The ``pg_rls_db`` connection strings.
        layout: The layout to build.
        case: The case.
        scope_name: The request scope whose identity is bound and whose observable is expected.
        monkeypatch: Restores the layout's environment and bindings after the test.
    """
    profile = PinnedIdentityProfile(_request_scope(scope_name))
    async with _rls_scoped_db(urls, layout, profile, monkeypatch) as db:
        await run_case_with_scope(db, case, scope_name, scope=SYSTEM_SCOPE)


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), _database_alone_params('fp32'))
async def test_database_alone_fp32_rls(
    pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch, case: AccessCase, scope_name: str,
) -> None:
    """With no application predicate, the database alone yields each read-decided fp32 case's observable."""
    await _run_database_alone(pg_rls_db, 'fp32', case, scope_name, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), _database_alone_params('compressed'))
async def test_database_alone_compressed_rls(
    pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch, case: AccessCase, scope_name: str,
) -> None:
    """With no application predicate, the database alone yields each read-decided compressed case's observable."""
    await _run_database_alone(pg_rls_db, 'compressed', case, scope_name, monkeypatch)


class TestRlsLayouts:
    """The row-level security layouts: seeded by the owner, requested as mcp_app under the policies."""

    def test_rls_run_cases_are_well_formed(self) -> None:
        """The cases of the row-level security run use known case ids and scope names and unique test ids."""
        assert registry_problems(_RLS_RUN_CASES) == []

    def test_read_decided_cases_name_request_scopes(self) -> None:
        """Every read-decided case runs as at least one request scope, so the database-alone run covers it."""
        assert all(set(case.expected) - {'system'} for case in READ_DECIDED_CASES)

    @pytest.mark.asyncio
    async def test_fp32_rls_layout_holds_the_seed(self, scoped_db_fp32_rls: ScopedDb) -> None:
        """The fp32 row-level security layout stores every seed row with fp32 vectors, read as the owner."""
        await assert_seeded_layout(scoped_db_fp32_rls)

    @pytest.mark.asyncio
    async def test_compressed_rls_layout_holds_the_seed(self, scoped_db_compressed_rls: ScopedDb) -> None:
        """The compressed row-level security layout stores every seed row with compressed payloads, read as the owner."""
        await assert_seeded_layout(scoped_db_compressed_rls)

    @pytest.mark.asyncio
    async def test_request_channel_reads_through_the_policies(self, scoped_db_compressed_rls: ScopedDb) -> None:
        """The request channel logs in as mcp_app and, with no identity bound, reads the public seed rows alone."""
        db = scoped_db_compressed_rls
        request = _through_request_channel(db)

        rows = await request.fetch_all('SELECT id FROM context_entries')

        assert await request.fetch_all('SELECT current_user') == [('mcp_app',)]
        assert in_seed_order(db.labels_of(str(row[0]) for row in rows)) == PUBLIC_ROWS

    @pytest.mark.asyncio
    async def test_system_scope_runs_on_the_seed_channel(self, scoped_db_compressed_rls: ScopedDb) -> None:
        """The system scope's seams run as mcp_owner, every request scope's as mcp_app."""
        db = scoped_db_compressed_rls

        assert _channel_of(db, 'system').backend is db.seed_backend
        assert all(_channel_of(db, name).backend is db.backend for name in SCOPES if name != 'system')
