"""Tests for the shared test helpers in tests/helpers.py.

Covers ``postgresql_subprocess_env``: a server or command-line subprocess inherits the
parent environment for its provider configuration, while every PostgreSQL routing input
of the parent (each ``POSTGRESQL_*`` variable and ``ENABLE_POSTGRESQL_RLS``) is removed
before the helper pins the routing, the row-level security credentials when they are
given, and the caller's overrides. ``RlsCredentials.environment`` names the variables
that start a server with row-level security. A structural test reads the integration
tests and requires every environment that inherits the parent's to come from the helper.

Covers ``CapturingBackend``, the statement capture the query-plan tests share: on SQLite it records a statement sent
through the connection and one sent through a cursor the connection opened, on PostgreSQL every statement sent through
``fetch``, ``fetchrow``, ``fetchval`` and ``execute``, each with its bound values, while the read runs unchanged. And
importing the helpers loads neither the application nor its settings, which ``tests/run_server.py`` relies on: it
imports them before it configures the server's environment.
"""

import ast
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from tests.helpers import AsyncpgConnection
from tests.helpers import Captured
from tests.helpers import CapturingBackend
from tests.helpers import RecordingSqliteConnection
from tests.helpers import RlsCredentials
from tests.helpers import postgresql_subprocess_env

TEST_URL = 'postgresql://mcp_app:mcp_app@localhost:5433/mcp_rls'
RLS = RlsCredentials(migration_user='mcp_owner', migration_password='mcp_owner', policy_role='mcp_context_request')
RLS_KEYS = (
    'ENABLE_POSTGRESQL_RLS', 'POSTGRESQL_RLS_POLICY_ROLE', 'POSTGRESQL_MIGRATION_USER', 'POSTGRESQL_MIGRATION_PASSWORD',
)


def test_credentials_name_the_row_level_security_environment() -> None:
    assert RLS.environment() == {
        'ENABLE_POSTGRESQL_RLS': 'true',
        'POSTGRESQL_RLS_POLICY_ROLE': 'mcp_context_request',
        'POSTGRESQL_MIGRATION_USER': 'mcp_owner',
        'POSTGRESQL_MIGRATION_PASSWORD': 'mcp_owner',
    }


@pytest.fixture
def supabase_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parent environment routed to a Supabase database, row-level security credentials included."""
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_HOST', 'aws-0-eu-central-1.pooler.supabase.com')
    monkeypatch.setenv('POSTGRESQL_USER', 'postgres.projectref')
    monkeypatch.setenv('POSTGRESQL_PASSWORD', 'supabase-secret')
    monkeypatch.setenv('POSTGRESQL_SCHEMA', 'tenant')
    monkeypatch.setenv('POSTGRESQL_MIGRATION_USER', 'supabase_owner')
    monkeypatch.setenv('POSTGRESQL_RLS_POLICY_ROLE', 'supabase_requests')
    monkeypatch.setenv('ENABLE_POSTGRESQL_RLS', 'true')
    monkeypatch.setenv('EMBEDDING_MODEL', 'all-minilm')
    monkeypatch.setenv('DISABLED_TOOLS', 'delete_context')


@pytest.mark.usefixtures('supabase_parent')
class TestPostgresqlSubprocessEnv:
    """The parent's PostgreSQL routing never reaches the subprocess; its provider configuration does."""

    def test_only_the_connection_string_survives_of_the_postgresql_family(self) -> None:
        env = postgresql_subprocess_env(TEST_URL)

        assert {key for key in env if key.upper().startswith('POSTGRESQL_')} == {'POSTGRESQL_CONNECTION_STRING'}
        assert env['POSTGRESQL_CONNECTION_STRING'] == TEST_URL
        assert env['STORAGE_BACKEND'] == 'postgresql'
        assert env['MCP_TEST_MODE'] == '1'
        assert env['DISABLED_TOOLS'] == ''

    def test_provider_configuration_is_inherited(self) -> None:
        assert postgresql_subprocess_env(TEST_URL)['EMBEDDING_MODEL'] == 'all-minilm'

    def test_row_level_security_is_absent_unless_credentials_are_given(self) -> None:
        env = postgresql_subprocess_env(TEST_URL)

        assert not set(RLS_KEYS) & set(env)

    def test_credentials_set_exactly_the_row_level_security_keys(self) -> None:
        env = postgresql_subprocess_env(TEST_URL, rls=RLS)
        without = postgresql_subprocess_env(TEST_URL)

        assert {key: env[key] for key in RLS_KEYS} == {
            'ENABLE_POSTGRESQL_RLS': 'true',
            'POSTGRESQL_RLS_POLICY_ROLE': 'mcp_context_request',
            'POSTGRESQL_MIGRATION_USER': 'mcp_owner',
            'POSTGRESQL_MIGRATION_PASSWORD': 'mcp_owner',
        }
        assert {key: value for key, value in env.items() if key not in RLS_KEYS} == without

    def test_overrides_come_last(self) -> None:
        env = postgresql_subprocess_env(TEST_URL, rls=RLS, ENABLE_POSTGRESQL_RLS='false', LOG_LEVEL='INFO')

        assert env['ENABLE_POSTGRESQL_RLS'] == 'false'
        assert env['LOG_LEVEL'] == 'INFO'
        assert env['POSTGRESQL_MIGRATION_USER'] == 'mcp_owner'


def _spreads_parent_and_routes_postgresql(node: ast.Dict) -> bool:
    """Whether a dict literal spreads ``os.environ`` and sets ``POSTGRESQL_CONNECTION_STRING``."""
    spreads_parent = any(
        key is None and isinstance(value, ast.Attribute) and value.attr == 'environ'
        and isinstance(value.value, ast.Name) and value.value.id == 'os'
        for key, value in zip(node.keys, node.values, strict=True)
    )
    routes = any(isinstance(key, ast.Constant) and key.value == 'POSTGRESQL_CONNECTION_STRING' for key in node.keys)
    return spreads_parent and routes


def test_every_postgresql_subprocess_environment_comes_from_the_helper() -> None:
    """No integration test builds an inheriting PostgreSQL environment by hand.

    A dict that spreads ``os.environ`` and sets the connection string keeps the parent's
    other ``POSTGRESQL_*`` variables and ``ENABLE_POSTGRESQL_RLS``, which
    ``postgresql_subprocess_env`` removes.
    """
    tests_root = Path(__file__).resolve().parent
    hand_built = [
        f'{path.relative_to(tests_root).as_posix()}:{node.lineno}'
        for path in sorted((tests_root / 'integration').rglob('*.py'))
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
        if isinstance(node, ast.Dict) and _spreads_parent_and_routes_postgresql(node)
    ]

    assert hand_built == []


class TestCapturingBackendOnSqlite:
    """Reads of a real SQLite backend, run through the capture."""

    @pytest.mark.asyncio
    async def test_records_a_statement_sent_through_the_connection(self, async_db_initialized: StorageBackend) -> None:
        capturing = CapturingBackend(async_db_initialized)

        def _read(conn: sqlite3.Connection, offset: int, *, label: str) -> list[tuple[object, ...]]:
            return [tuple(row) for row in conn.execute('SELECT ? + ?, ?', (offset, 2, label)).fetchall()]

        rows = await capturing.execute_read(_read, 1, access_scope=MAINTENANCE, label='x')

        assert rows == [(3, 'x')]
        assert capturing.statements == [Captured('SELECT ? + ?, ?', (1, 2, 'x'))]

    @pytest.mark.asyncio
    async def test_records_a_statement_sent_through_a_cursor(self, async_db_initialized: StorageBackend) -> None:
        capturing = CapturingBackend(async_db_initialized)

        def _read(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
            cursor = conn.cursor()
            cursor.execute('SELECT ?', ('first',))
            first = [tuple(row) for row in cursor.fetchall()]
            second = [tuple(row) for row in cursor.execute('SELECT ?, ?', ['second', 2]).fetchall()]
            return first + second

        rows = await capturing.execute_read(_read, access_scope=MAINTENANCE)

        assert rows == [('first',), ('second', 2)]
        assert capturing.statements == [Captured('SELECT ?', ('first',)), Captured('SELECT ?, ?', ('second', 2))]


class _AsyncpgReadBackend:
    """A PostgreSQL-typed backend whose reads hand the operation one fixed connection."""

    backend_type = 'postgresql'

    def __init__(self, conn: object) -> None:
        self.conn = conn

    async def execute_read(self, operation: object, *args: object, access_scope: BackendAccess, **kwargs: object) -> object:
        assert access_scope is MAINTENANCE
        return await cast('AsyncMock', operation)(self.conn, *args, **kwargs)


@pytest.mark.asyncio
async def test_records_every_asyncpg_statement_and_passes_everything_else_through() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = ['row']
    conn.fetchrow.return_value = 'one row'
    conn.fetchval.return_value = 'one value'
    conn.execute.return_value = 'SET'
    conn.transaction = MagicMock(return_value='a transaction')
    capturing = CapturingBackend(cast(StorageBackend, _AsyncpgReadBackend(conn)))

    async def _read(recording: AsyncpgConnection, label: str, *, flag: bool) -> list[object]:
        return [
            await recording.fetch('SELECT $1', 1),
            await recording.fetchrow('SELECT $1, $2', 2, 'b'),
            await recording.fetchval('SELECT 3'),
            await recording.execute('SET LOCAL x = $1', 4),
            recording.transaction(),
            label,
            flag,
        ]

    result = await capturing.execute_read(_read, 'label', access_scope=MAINTENANCE, flag=True)

    assert result == [['row'], 'one row', 'one value', 'SET', 'a transaction', 'label', True]
    assert capturing.statements == [
        Captured('SELECT $1', (1,)),
        Captured('SELECT $1, $2', (2, 'b')),
        Captured('SELECT 3', ()),
        Captured('SET LOCAL x = $1', (4,)),
    ]
    conn.fetchrow.assert_awaited_once_with('SELECT $1, $2', 2, 'b')


def test_importing_the_helpers_loads_neither_the_application_nor_its_settings() -> None:
    """A fresh interpreter that imports the helpers holds ``app.access_scope`` and no other application module."""
    probe = (
        'import json, sys, tests.helpers; '
        "print(json.dumps(sorted(name for name in sys.modules if name == 'app' or name.startswith('app.'))))"
    )
    repository_root = Path(__file__).resolve().parent.parent

    completed = subprocess.run([sys.executable, '-c', probe], cwd=repository_root, capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == ['app', 'app.access_scope']


def test_a_recording_sqlite_connection_runs_on_the_connection_it_wraps() -> None:
    statements: list[Captured] = []
    with sqlite3.connect(':memory:') as conn:
        recording = RecordingSqliteConnection(conn, statements)

        assert recording.execute('SELECT 7').fetchall() == [(7,)]

    assert statements == [Captured('SELECT 7', ())]
