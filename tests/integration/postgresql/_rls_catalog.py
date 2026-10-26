"""Catalog probes and database preparation for the row-level security migration tests.

:func:`configure_rls_layout` and :func:`maintenance_backend` let a test boot a layout on
the ``pg_rls_db`` database the way the server does with ``ENABLE_POSTGRESQL_RLS``:
``prepare_database`` on a maintenance backend logged in as the table owner ``mcp_owner``,
which ends with the row-level security migration.
The probes read, as the superuser, what the migration leaves behind: the policies of
every table in a schema, the OID of every policy, function, trigger and view of ours there,
whether row-level security is on, and the read view's columns and ACL.
:func:`install_ddl_log` records the command tag of every DDL statement anyone issues in
the database from then on, so a test can prove that a step issued none.
:func:`assert_served_with_row_level_security` proves after a server run that the server
served ``mcp_app`` under the policies.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

import asyncpg
import pytest

from app.backends import StorageBackend
from app.backends import create_maintenance_backend
from app.settings import AppSettings
from tests.integration.postgresql._rls_matrix import rls_settings
from tests.integration.postgresql.conftest import RlsUrls
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_layouts import configure_postgresql_layout

SCHEMA_INIT_LOCK_SQL: Final = "SELECT pg_advisory_lock(hashtext('mcp_context_schema_init'))"
SCHEMA_INIT_UNLOCK_SQL: Final = "SELECT pg_advisory_unlock(hashtext('mcp_context_schema_init'))"
ADVISORY_WAITERS_SQL: Final = "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"

_POLICIES_SQL: Final = (
    'SELECT c.relname, p.polname, p.oid, '
    'ARRAY(SELECT pg_get_userbyid(r) FROM unnest(p.polroles) AS r ORDER BY 1), '
    "obj_description(p.oid, 'pg_policy') "
    'FROM pg_policy p JOIN pg_class c ON c.oid = p.polrelid JOIN pg_namespace n ON n.oid = c.relnamespace '
    'WHERE n.nspname = $1'
)
_OBJECT_OIDS_SQL: Final = (
    "SELECT 'policy ' || c.relname || '.' || p.polname, p.oid FROM pg_policy p "
    'JOIN pg_class c ON c.oid = p.polrelid JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 '
    "UNION ALL SELECT 'function ' || p.proname, p.oid FROM pg_proc p "
    "JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = $1 AND p.proname LIKE 'mcp\\_context\\_%' "
    "UNION ALL SELECT 'trigger ' || c.relname || '.' || t.tgname, t.oid FROM pg_trigger t "
    'JOIN pg_class c ON c.oid = t.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace '
    "WHERE n.nspname = $1 AND t.tgname IN ('guard_context_entry_access_columns', 'reject_writes') "
    "UNION ALL SELECT 'view ' || c.relname, c.oid FROM pg_class c "
    "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relkind = 'v'"
)
_ROW_SECURITY_SQL: Final = (
    'SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c '
    "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relkind = 'r'"
)
_VIEW_COLUMNS_SQL: Final = (
    'SELECT attname FROM pg_attribute WHERE attrelid = to_regclass($1) AND attnum > 0 AND NOT attisdropped ORDER BY attnum'
)
_RELATIONS_SQL: Final = (
    'SELECT c.relname, c.relkind::text, pg_get_userbyid(c.relowner), c.relrowsecurity, c.relforcerowsecurity FROM pg_class c '
    "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relname = ANY($1::text[])"
)
# The tables every request reads or writes, and the view plain reads go through.
_REQUEST_RELATIONS: Final = ('context_entries', 'context_entry_grants', 'tags', 'image_attachments', 'generation_jobs')
_READ_VIEW: Final = 'context_entries_readable'
_VIEW_ACL_SQL: Final = (
    "SELECT COALESCE(r.rolname, 'PUBLIC'), a.privilege_type FROM pg_class c "
    'CROSS JOIN LATERAL aclexplode(c.relacl) AS a LEFT JOIN pg_roles r ON r.oid = a.grantee '
    'WHERE c.oid = to_regclass($1) AND a.grantee <> c.relowner ORDER BY 1, 2'
)


@dataclass(frozen=True, slots=True)
class Policy:
    """One policy as the catalog holds it.

    Attributes:
        oid: Its OID, which changes when it is re-created.
        roles: The roles it applies to, by name.
        comment: Its comment.
    """

    oid: int
    roles: tuple[str, ...]
    comment: str | None


async def policies(conn: asyncpg.Connection, schema_name: str) -> dict[tuple[str, str], Policy]:
    """Return every policy on a table of ``schema_name``, keyed by table and policy name."""
    return {
        (str(row[0]), str(row[1])): Policy(oid=int(row[2]), roles=tuple(row[3]), comment=row[4])
        for row in await conn.fetch(_POLICIES_SQL, schema_name)
    }


async def object_oids(conn: asyncpg.Connection, schema_name: str) -> dict[str, int]:
    """Return the OID of every policy, ``mcp_context_`` function, row-level security trigger and view of ``schema_name``."""
    return {str(row[0]): int(row[1]) for row in await conn.fetch(_OBJECT_OIDS_SQL, schema_name)}


async def row_security(conn: asyncpg.Connection, schema_name: str) -> dict[str, tuple[bool, bool]]:
    """Return whether row-level security is enabled and forced on each table of ``schema_name``."""
    return {str(row[0]): (bool(row[1]), bool(row[2])) for row in await conn.fetch(_ROW_SECURITY_SQL, schema_name)}


async def view_columns(conn: asyncpg.Connection, view: str) -> list[str]:
    """Return the columns of ``view``, schema-qualified, in order."""
    return [str(row[0]) for row in await conn.fetch(_VIEW_COLUMNS_SQL, view)]


async def view_acl(conn: asyncpg.Connection, view: str) -> list[tuple[str, str]]:
    """Return every grantee but the owner of ``view`` with each privilege it holds; PUBLIC by that name."""
    return [(str(row[0]), str(row[1])) for row in await conn.fetch(_VIEW_ACL_SQL, view)]


async def install_ddl_log(conn: asyncpg.Connection) -> None:
    """Record the command tag of every DDL statement issued in the database from now on.

    Args:
        conn: A superuser connection; event triggers need one.
    """
    await conn.execute('CREATE TABLE public.ddl_log (tag text NOT NULL)')
    await conn.execute(
        'CREATE FUNCTION public.log_ddl() RETURNS event_trigger LANGUAGE plpgsql SECURITY DEFINER '
        'SET search_path = pg_catalog, public AS $$ BEGIN INSERT INTO public.ddl_log VALUES (tg_tag); END $$',
    )
    await conn.execute('CREATE EVENT TRIGGER log_ddl ON ddl_command_start EXECUTE FUNCTION public.log_ddl()')


async def ddl_log(conn: asyncpg.Connection) -> list[str]:
    """Return the command tags :func:`install_ddl_log` recorded, oldest first."""
    return [str(row[0]) for row in await conn.fetch('SELECT tag FROM public.ddl_log')]


async def assert_served_with_row_level_security(urls: RlsUrls) -> None:
    """Assert that a server started on ``urls.app`` with ``RLS_CREDENTIALS`` served it with row-level security.

    ``mcp_owner`` owns the tables every request touches and the read view, and has enabled
    row-level security on the tables without forcing it. ``mcp_app`` owns nothing and bypasses
    nothing, so every request the server served as ``mcp_app`` passed the policies and the
    read view rather than the application's own conditions alone.

    Args:
        urls: The ``pg_rls_db`` connection strings; the probe reads the catalog as the superuser.
    """
    conn = await asyncpg.connect(urls.admin)
    try:
        rows = await conn.fetch(_RELATIONS_SQL, [*_REQUEST_RELATIONS, _READ_VIEW])
    finally:
        await conn.close()
    relations = {str(row[0]): (str(row[1]), str(row[2]), bool(row[3]), bool(row[4])) for row in rows}
    assert relations == {
        **dict.fromkeys(_REQUEST_RELATIONS, ('r', 'mcp_owner', True, False)),
        _READ_VIEW: ('v', 'mcp_owner', False, False),
    }


@asynccontextmanager
async def maintenance_backend(url: str) -> AsyncGenerator[StorageBackend]:
    """Yield an initialized maintenance backend logged in with the credentials of ``url``, shut down afterwards."""
    backend = create_maintenance_backend(connection_string=url)
    await backend.initialize()
    try:
        yield backend
    finally:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=10.0)


def configure_rls_layout(
    urls: RlsUrls,
    monkeypatch: pytest.MonkeyPatch,
    *,
    layout: Layout,
    overrides: Mapping[str, str],
    rls: bool,
) -> AppSettings:
    """Configure ``layout`` for a boot of the ``pg_rls_db`` database, before any backend is created.

    Args:
        urls: The ``pg_rls_db`` connection strings; requests would log in as ``mcp_app``.
        monkeypatch: Restores the environment and the bindings after the test.
        layout: The layout to build.
        overrides: Further environment values, such as a feature switched off.
        rls: Whether the boot runs with ENABLE_POSTGRESQL_RLS.

    Returns:
        The settings the boot runs with, to pass to ``prepare_database``.
    """
    settings = configure_postgresql_layout(monkeypatch, layout, urls.app, overrides=overrides)
    return rls_settings() if rls else settings
