"""Tests for the privileges the request role's policy role receives.

Covers app.migrations.request_role_grants: the grant plan issues only the
privileges the policy role does not hold (schema usage, the four DML privileges
on every existing managed table, SELECT on the compression provenance, USAGE and
SELECT on the sequences the managed tables own, EXECUTE on the merge-patch
function), never on the read view or on a sequence the managed tables do not own;
the catalog reader that feeds it; and ``apply_request_role_grants``, which runs
only on PostgreSQL with split roles, refuses a missing policy role with the
statements that create it, takes the schema-init lock only when something is
missing, re-reads under the lock and reports what it issued.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.errors import ConfigurationError
from app.migrations import request_role_grants as grants_module
from app.migrations.request_role_grants import OWNED_SEQUENCES_SQL
from app.migrations.request_role_grants import GrantReport
from app.migrations.request_role_grants import HeldPrivileges
from app.migrations.request_role_grants import apply_request_role_grants
from app.migrations.request_role_grants import plan_grants
from app.migrations.request_role_grants import read_held_privileges
from app.settings import AppSettings

SCHEMA = '"public"'
ROLE = '"mcp_context_request"'
NONE_HELD: frozenset[str] = frozenset()
TAGS_SEQUENCE = '"public"."tags_id_seq"'
NODES_SEQUENCE = '"public"."context_index_nodes_id_seq"'


def _held(
    *,
    schema_usage: bool = True,
    tables: dict[str, frozenset[str]] | None = None,
    sequences: dict[str, frozenset[str]] | None = None,
    merge_patch: bool | None = True,
) -> HeldPrivileges:
    return HeldPrivileges(
        schema_usage=schema_usage, tables=tables or {}, sequences=sequences or {}, merge_patch=merge_patch,
    )


ALL_HELD = _held(
    tables={
        'context_entries': frozenset({'SELECT', 'INSERT', 'UPDATE', 'DELETE'}),
        'compression_metadata': frozenset({'SELECT'}),
    },
    sequences={TAGS_SEQUENCE: frozenset({'USAGE', 'SELECT'})},
)


class TestPlan:
    """Only missing privileges become statements, in a fixed order."""

    def test_nothing_held_grants_every_needed_privilege(self) -> None:
        held = _held(
            schema_usage=False,
            tables={
                'generation_jobs': NONE_HELD, 'tags': NONE_HELD, 'compression_metadata': NONE_HELD,
                'context_entry_grants': NONE_HELD, 'context_entries': NONE_HELD,
            },
            sequences={TAGS_SEQUENCE: NONE_HELD, NODES_SEQUENCE: NONE_HELD},
            merge_patch=False,
        )

        assert plan_grants(held, schema=SCHEMA, role=ROLE) == [
            f'GRANT USAGE ON SCHEMA {SCHEMA} TO {ROLE}',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {SCHEMA}.context_entries TO {ROLE}',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {SCHEMA}.context_entry_grants TO {ROLE}',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {SCHEMA}.tags TO {ROLE}',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {SCHEMA}.generation_jobs TO {ROLE}',
            f'GRANT SELECT ON TABLE {SCHEMA}.compression_metadata TO {ROLE}',
            f'GRANT USAGE, SELECT ON SEQUENCE {TAGS_SEQUENCE} TO {ROLE}',
            f'GRANT USAGE, SELECT ON SEQUENCE {NODES_SEQUENCE} TO {ROLE}',
            f'GRANT EXECUTE ON FUNCTION {SCHEMA}.jsonb_merge_patch(jsonb, jsonb) TO {ROLE}',
        ]

    def test_held_privileges_are_left_out(self) -> None:
        held = _held(
            tables={
                'tags': frozenset({'SELECT', 'INSERT'}),
                'compression_metadata': frozenset({'SELECT'}),
                'context_index_nodes': frozenset({'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE'}),
            },
            sequences={TAGS_SEQUENCE: frozenset({'USAGE'})},
        )

        assert plan_grants(held, schema=SCHEMA, role=ROLE) == [
            f'GRANT UPDATE, DELETE ON TABLE {SCHEMA}.tags TO {ROLE}',
            f'GRANT SELECT ON SEQUENCE {TAGS_SEQUENCE} TO {ROLE}',
        ]

    def test_nothing_missing_plans_nothing(self) -> None:
        assert plan_grants(ALL_HELD, schema=SCHEMA, role=ROLE) == []

    def test_an_absent_merge_patch_function_is_not_granted(self) -> None:
        assert plan_grants(_held(merge_patch=None), schema=SCHEMA, role=ROLE) == []

    def test_no_statement_names_the_read_view_or_truncate_or_every_sequence(self) -> None:
        held = _held(
            schema_usage=False,
            tables={'context_entries': NONE_HELD, 'compression_metadata': NONE_HELD},
            sequences={TAGS_SEQUENCE: NONE_HELD},
            merge_patch=False,
        )

        ddl = '\n'.join(plan_grants(held, schema=SCHEMA, role=ROLE))
        assert 'context_entries_readable' not in ddl
        assert 'TRUNCATE' not in ddl
        assert 'ALL SEQUENCES' not in ddl
        assert 'ALL TABLES' not in ddl

    def test_owned_sequences_are_resolved_through_their_column_dependency(self) -> None:
        """The serial and identity sequences of the managed tables, as pg_get_serial_sequence reports them."""
        assert 'pg_catalog.pg_depend' in OWNED_SEQUENCES_SQL
        assert "d.deptype IN ('a', 'i')" in OWNED_SEQUENCES_SQL
        assert "s.relkind = 'S'" in OWNED_SEQUENCES_SQL


class _PrivilegeConnection:
    """Answers the catalog reads of read_held_privileges."""

    def __init__(self, *, function_held: bool | None) -> None:
        self.function_held = function_held
        self.reads: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, sql: str, *args: object) -> bool | None:
        self.reads.append((sql, args))
        if 'has_schema_privilege' in sql:
            return False
        assert 'has_function_privilege' in sql
        return self.function_held

    async def fetch(self, sql: str, *args: object) -> list[tuple[object, ...]]:
        self.reads.append((sql, args))
        if sql == OWNED_SEQUENCES_SQL:
            return [('public', 'tags_id_seq', True, False), ('odd "schema"', 'seq', False, False)]
        assert 'has_table_privilege' in sql
        return [('tags', True, False, True, False), ('compression_metadata', True, True, True, True)]


class TestReadHeldPrivileges:
    """The catalog reader reports per-object privileges of the policy role."""

    @pytest.mark.asyncio
    async def test_rows_become_held_privileges(self) -> None:
        conn = _PrivilegeConnection(function_held=True)

        held = await read_held_privileges(cast(Any, conn), schema_name='public', role_name='mcp_context_request')

        assert held == HeldPrivileges(
            schema_usage=False,
            tables={
                'tags': frozenset({'SELECT', 'UPDATE'}),
                'compression_metadata': frozenset({'SELECT', 'INSERT', 'UPDATE', 'DELETE'}),
            },
            sequences={TAGS_SEQUENCE: frozenset({'USAGE'}), '"odd ""schema"""."seq"': NONE_HELD},
            merge_patch=True,
        )
        for _sql, args in conn.reads:
            assert 'mcp_context_request' in args

    @pytest.mark.asyncio
    async def test_an_absent_function_reads_as_none(self) -> None:
        held = await read_held_privileges(
            cast(Any, _PrivilegeConnection(function_held=None)), schema_name='public', role_name='mcp_context_request',
        )

        assert held.merge_patch is None


def _settings(*, schema: str = 'public', policy_role: str = 'mcp_context_request') -> AppSettings:
    return cast(
        AppSettings,
        SimpleNamespace(
            access_control=SimpleNamespace(postgresql_rls_policy_role=policy_role),
            storage=SimpleNamespace(postgresql_schema=schema, postgresql_migration_timeout_s=300.0),
        ),
    )


class _RoleConnection:
    """Answers the role reads: the maintenance current_user and whether the policy role exists."""

    def __init__(self, *, current_user: str = 'mcp_owner', policy_role_exists: bool = True) -> None:
        self.current_user = current_user
        self.policy_role_exists = policy_role_exists
        self.reads: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, sql: str, *args: object) -> object:
        self.reads.append((sql, args))
        if 'to_regrole' in sql:
            assert args == ('"mcp_context_request"',)
            return self.policy_role_exists
        assert sql == 'SELECT current_user'
        return self.current_user


class _Backend:
    """A backend that runs the grant step's callable on the role connection."""

    def __init__(self, conn: _RoleConnection, backend_type: str = 'postgresql') -> None:
        self.conn = conn
        self.backend_type = backend_type
        self.scopes: list[BackendAccess] = []

    async def execute_write(
        self, operation: Callable[[object], Awaitable[GrantReport]], *, access_scope: BackendAccess,
    ) -> GrantReport:
        self.scopes.append(access_scope)
        return await operation(self.conn)


@pytest.fixture
def lifecycle(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[HeldPrivileges]]:
    """Record the lock and the statements issued, and serve successive privilege reads.

    Returns:
        The recorded statements (the lock as ``begin_migration``) and the queue of
        privilege states the reader returns, first read first.
    """
    statements: list[str] = []
    states: list[HeldPrivileges] = []

    async def _begin(_conn: object, timeout_s: float) -> None:
        assert timeout_s == 300.0
        statements.append('begin_migration')

    async def _execute(_conn: object, statement: str, timeout_s: float) -> None:
        assert timeout_s == 300.0
        statements.append(statement)

    async def _read(_conn: object, *, schema_name: str, role_name: str) -> HeldPrivileges:
        assert (schema_name, role_name) == ('public', 'mcp_context_request')
        return states.pop(0)

    monkeypatch.setattr(grants_module, 'begin_migration', _begin)
    monkeypatch.setattr(grants_module, 'execute_migration_ddl', _execute)
    monkeypatch.setattr(grants_module, 'read_held_privileges', _read)
    return statements, states


class TestApplyRequestRoleGrants:
    """The step runs for split roles only, refuses a missing policy role and locks only to grant."""

    @pytest.mark.asyncio
    async def test_no_request_role_skips_the_step(self, lifecycle: tuple[list[str], list[HeldPrivileges]]) -> None:
        backend = _Backend(_RoleConnection())

        report = await apply_request_role_grants(cast(StorageBackend, backend), _settings(), request_role=None)

        assert report == GrantReport(statements=[], locked=False)
        assert backend.scopes == []
        assert lifecycle[0] == []

    @pytest.mark.asyncio
    async def test_sqlite_skips_the_step(self, lifecycle: tuple[list[str], list[HeldPrivileges]]) -> None:
        backend = _Backend(_RoleConnection(), backend_type='sqlite')

        report = await apply_request_role_grants(cast(StorageBackend, backend), _settings(), request_role='mcp_app')

        assert report == GrantReport(statements=[], locked=False)
        assert backend.scopes == []
        assert lifecycle[0] == []

    @pytest.mark.asyncio
    async def test_identical_roles_grant_nothing(self, lifecycle: tuple[list[str], list[HeldPrivileges]]) -> None:
        conn = _RoleConnection(current_user='mcp_app')

        report = await apply_request_role_grants(cast(StorageBackend, _Backend(conn)), _settings(), request_role='mcp_app')

        assert report == GrantReport(statements=[], locked=False)
        assert [sql for sql, _ in conn.reads] == ['SELECT current_user']
        assert lifecycle[0] == []

    @pytest.mark.asyncio
    async def test_missing_policy_role_is_refused_with_the_statements_that_create_it(
        self, lifecycle: tuple[list[str], list[HeldPrivileges]],
    ) -> None:
        backend = _Backend(_RoleConnection(policy_role_exists=False))

        with pytest.raises(ConfigurationError) as refusal:
            await apply_request_role_grants(cast(StorageBackend, backend), _settings(), request_role='mcp app')

        assert str(refusal.value) == (
            'POSTGRESQL_RLS_POLICY_ROLE names the PostgreSQL role "mcp_context_request", which does not exist, '
            'so the request login role "mcp app" cannot be granted the privileges it needs. As a role administrator, '
            'create the role and make the login role its member, then restart: '
            'CREATE ROLE "mcp_context_request" NOLOGIN; GRANT "mcp_context_request" TO "mcp app"; '
            "(docs/database-backends.md, 'Row-level security (PostgreSQL)')."
        )
        assert lifecycle[0] == []

    @pytest.mark.asyncio
    async def test_unchanged_privileges_issue_nothing_and_take_no_lock(
        self, lifecycle: tuple[list[str], list[HeldPrivileges]],
    ) -> None:
        statements, states = lifecycle
        states.append(ALL_HELD)
        backend = _Backend(_RoleConnection())

        report = await apply_request_role_grants(cast(StorageBackend, backend), _settings(), request_role='mcp_app')

        assert report == GrantReport(statements=[], locked=False)
        assert statements == []
        assert states == []
        assert backend.scopes == [MAINTENANCE]

    @pytest.mark.asyncio
    async def test_missing_privileges_are_granted_under_the_lock_after_a_second_read(
        self, lifecycle: tuple[list[str], list[HeldPrivileges]],
    ) -> None:
        statements, states = lifecycle
        before_lock = _held(schema_usage=False, tables={'tags': NONE_HELD})
        under_lock = _held(tables={'tags': frozenset({'SELECT'})})
        states.extend([before_lock, under_lock])

        report = await apply_request_role_grants(
            cast(StorageBackend, _Backend(_RoleConnection())), _settings(), request_role='mcp_app',
        )

        granted = [f'GRANT INSERT, UPDATE, DELETE ON TABLE {SCHEMA}.tags TO {ROLE}']
        assert report == GrantReport(statements=granted, locked=True)
        assert statements == ['begin_migration', *granted]
        assert states == []

    @pytest.mark.asyncio
    async def test_a_peer_that_granted_everything_meanwhile_leaves_nothing_to_issue(
        self, lifecycle: tuple[list[str], list[HeldPrivileges]],
    ) -> None:
        statements, states = lifecycle
        states.extend([_held(schema_usage=False), ALL_HELD])

        report = await apply_request_role_grants(
            cast(StorageBackend, _Backend(_RoleConnection())), _settings(), request_role='mcp_app',
        )

        assert report == GrantReport(statements=[], locked=True)
        assert statements == ['begin_migration']

    @pytest.mark.asyncio
    async def test_a_failure_surfaces_as_the_migration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _read(_conn: object, *, schema_name: str, role_name: str) -> HeldPrivileges:
            _ = (schema_name, role_name)
            raise OSError('connection reset')

        monkeypatch.setattr(grants_module, 'read_held_privileges', _read)

        with pytest.raises(RuntimeError, match='Request role grants failed: connection reset'):
            await apply_request_role_grants(
                cast(StorageBackend, _Backend(_RoleConnection())), _settings(), request_role='mcp_app',
            )
