"""The row-level security policy matrix: extra rows, request scopes and the rows each may reach.

The matrix database is the access-scope seed (:data:`~tests.repositories._access_scope_cases.SEED_ROWS`)
prepared and seeded by the table owner ``mcp_owner`` on the ``pg_rls_db`` database, plus
:data:`MATRIX_ROWS`: a private row owned by the empty principal, a row granted to the empty
principal and the empty group, rows granted to groups whose ids hold a comma, a double quote,
braces and a backslash, and a decoy row granted to the fragments a lossy group serialization
would produce. Every seed entry also carries a pending generation job, so the jobs table, like
every other child table, holds rows each scope may reach and rows it may not. The policy role
then receives its privileges and the policies are installed, so ``mcp_app`` connections reach
rows through the policies alone.

:data:`MATRIX_SCOPES` names every identity the policy tests bind, ``unset`` (no bind at all)
included, and :data:`MATRIX_VISIBILITY` holds the rows each one reads, modifies and owns.
:func:`scoped_transaction` opens a transaction on a raw connection that is always rolled
back, binding the scope first with :func:`bind`, which runs the request profile's own bind
statement.
"""

import uuid
from collections.abc import AsyncGenerator
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Final

import asyncpg
import pytest

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.backends.postgresql_backend.session_profiles import BIND_IDENTITY_STATEMENT
from app.migrations import apply_request_role_grants
from app.migrations import apply_rls
from app.settings import AppSettings
from app.settings import get_settings
from tests.integration.postgresql.conftest import RlsUrls
from tests.repositories._access_scope_cases import OWNED
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SCOPES
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import WRITABLE
from tests.repositories._access_scope_cases import Grant
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import SeedRow
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_layouts import postgresql_owner_seeded_db

REQUEST_ROLE: Final = 'mcp_app'
POLICY_ROLE: Final = 'mcp_context_request'

COMMA_GROUP: Final = 'a,b'
QUOTE_GROUP: Final = 'q"t'
BRACE_GROUP: Final = '{br}'
BACKSLASH_GROUP: Final = 'back\\slash'
ODD_GROUPS: Final = frozenset({COMMA_GROUP, QUOTE_GROUP, BRACE_GROUP, BACKSLASH_GROUP})
# What a group list serialized as comma-separated text, or as an array literal read back
# without unescaping, would turn the odd group ids into.
DECOY_GROUPS: Final = ('a', 'b', '"a,b"', 'q', 'q\\"t', 'br', 'back', 'slash', 'back\\\\slash')

MATRIX_ROWS: Final = (
    SeedRow('blank_owner_private', '', 'private'),
    SeedRow('alice_private_blank_read', 'alice', 'private', grants=(Grant('user', '', 'read'), Grant('group', '', 'write'))),
    SeedRow('alice_private_comma_read', 'alice', 'private', grants=(Grant('group', COMMA_GROUP, 'read'),)),
    SeedRow('alice_private_quote_read', 'alice', 'private', grants=(Grant('group', QUOTE_GROUP, 'read'),)),
    SeedRow('alice_private_brace_write', 'alice', 'private', grants=(Grant('group', BRACE_GROUP, 'write'),)),
    SeedRow('alice_private_backslash_write', 'alice', 'private', grants=(Grant('group', BACKSLASH_GROUP, 'write'),)),
    SeedRow('alice_private_decoy', 'alice', 'private', grants=tuple(Grant('group', group, 'write') for group in DECOY_GROUPS)),
)


# A pending generation job on one entry; an entry that already holds a job keeps it.
SEED_JOB: Final = 'INSERT INTO generation_jobs (context_entry_id) VALUES (?) ON CONFLICT (context_entry_id) DO NOTHING'


def _request_scope(name: str) -> AccessScope:
    scope = SCOPES[name]
    assert isinstance(scope, AccessScope)
    return scope


# Every identity a policy test binds; None binds nothing, as a transaction of a scope without a principal does.
MATRIX_SCOPES: Final[dict[str, AccessScope | None]] = {
    'alice': _request_scope('alice'),
    'bob': _request_scope('bob'),
    'carol': _request_scope('carol'),
    'dave': _request_scope('dave'),
    'mallory': AccessScope('mallory', frozenset({'team-y'})),
    'unset': None,
    'blank': AccessScope('', frozenset()),
    'oscar': AccessScope('oscar', ODD_GROUPS),
}

PUBLIC_ROWS: Final = ('alice_public', 'alice_public_carol_write', 'alice_public_dave_read')
_ALICE_MATRIX_ROWS: Final = tuple(row.label for row in MATRIX_ROWS if row.owner == 'alice')
_ODD_READ_ROWS: Final = ('alice_private_comma_read', 'alice_private_quote_read')
_ODD_WRITE_ROWS: Final = ('alice_private_brace_write', 'alice_private_backslash_write')


def _rows(*groups: Iterable[str]) -> tuple[str, ...]:
    return in_seed_order(label for group in groups for label in group)


# The rows each identity reads, modifies and owns. Neither the empty principal nor the empty
# group matches anything, so ``blank`` reaches exactly what ``unset`` does: the public rows.
MATRIX_VISIBILITY: Final[dict[AccessMode, dict[str, tuple[str, ...]]]] = {
    AccessMode.READ: {
        'alice': _rows(READABLE['alice'], _ALICE_MATRIX_ROWS),
        'bob': _rows(READABLE['bob']),
        'carol': _rows(READABLE['carol']),
        'dave': _rows(READABLE['dave']),
        'mallory': _rows(PUBLIC_ROWS),
        'unset': _rows(PUBLIC_ROWS),
        'blank': _rows(PUBLIC_ROWS),
        'oscar': _rows(PUBLIC_ROWS, _ODD_READ_ROWS, _ODD_WRITE_ROWS),
    },
    AccessMode.WRITE: {
        'alice': _rows(WRITABLE['alice'], _ALICE_MATRIX_ROWS),
        'bob': _rows(WRITABLE['bob']),
        'carol': _rows(WRITABLE['carol']),
        'dave': (),
        'mallory': (),
        'unset': (),
        'blank': (),
        'oscar': _rows(_ODD_WRITE_ROWS),
    },
    AccessMode.OWNER: {
        'alice': _rows(OWNED['alice'], _ALICE_MATRIX_ROWS),
        'bob': _rows(OWNED['bob']),
        'carol': (),
        'dave': (),
        'mallory': (),
        'unset': (),
        'blank': (),
        'oscar': (),
    },
}

# The statement that reaches the rows of each mode through its policy: SELECT through the read
# policy, a no-op UPDATE through the update policy and a DELETE through the delete policy.
MODE_STATEMENTS: Final[dict[AccessMode, str]] = {
    AccessMode.READ: 'SELECT id FROM context_entries',
    AccessMode.WRITE: 'UPDATE context_entries SET text_content = text_content RETURNING id',
    AccessMode.OWNER: 'DELETE FROM context_entries RETURNING id',
}


def rls_settings(*, policy_role: str = POLICY_ROLE) -> AppSettings:
    """Return the current settings with row-level security on and ``policy_role`` as the policy role.

    Args:
        policy_role: The NOLOGIN role the policies apply to.

    Returns:
        A copy of the current settings.
    """
    settings = get_settings()
    access_control = settings.access_control.model_copy(
        update={'postgresql_rls_enabled': True, 'postgresql_rls_policy_role': policy_role},
    )
    return settings.model_copy(update={'access_control': access_control})


@asynccontextmanager
async def rls_matrix_db(urls: RlsUrls, *, layout: Layout, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """Seed the matrix and its jobs on ``urls``' database as its owner, grant the policy role and install the policies.

    Args:
        urls: The ``pg_rls_db`` connection strings.
        layout: The database layout.
        monkeypatch: Restores the layout's environment and bindings after the test.

    Yields:
        The seeded database; its seed channel runs as ``mcp_owner``, which bypasses the policies.
    """
    async with postgresql_owner_seeded_db(
        request_url=urls.app, maintenance_url=urls.maintenance, layout=layout, monkeypatch=monkeypatch,
    ) as db:
        for row in MATRIX_ROWS:
            await db.add_entry(row.label, owner=row.owner, visibility=row.visibility, grants=row.grants)
        for label in SEED_LABELS:
            await db.execute(SEED_JOB, (db.ids[label],))
        settings = rls_settings()
        await apply_request_role_grants(db.seed_backend, settings, request_role=REQUEST_ROLE)
        await apply_rls(db.seed_backend, settings)
        yield db


@asynccontextmanager
async def scoped_transaction(conn: asyncpg.Connection, scope: AccessScope | None) -> AsyncGenerator[asyncpg.Connection]:
    """Open a transaction on ``conn`` bound to ``scope`` and roll it back on exit.

    Args:
        conn: A raw connection of the request role.
        scope: The identity to bind with the request profile's bind statement; None binds nothing.

    Yields:
        ``conn`` inside the transaction.
    """
    transaction = conn.transaction()
    await transaction.start()
    try:
        if scope is not None:
            await bind(conn, scope)
        yield conn
    finally:
        await transaction.rollback()


async def bind(conn: asyncpg.Connection, scope: AccessScope) -> None:
    """Bind ``scope`` for the rest of ``conn``'s open transaction with the request profile's bind statement.

    Args:
        conn: A raw connection of the request role inside a transaction.
        scope: The identity to bind.
    """
    await conn.execute(
        BIND_IDENTITY_STATEMENT,
        scope.principal_id,
        scope.groups_list(),
        'on' if scope.may_publish else 'off',
        'on' if scope.may_write else 'off',
    )


def entry_uuid(db: ScopedDb, label: str) -> uuid.UUID:
    """Return the id of the entry ``label`` as the UUID asyncpg binds to a UUID column.

    Args:
        db: The seeded database.
        label: The entry's label.

    Returns:
        The entry id.
    """
    return uuid.UUID(hex=db.ids[label])


def labels(db: ScopedDb, rows: Iterable[asyncpg.Record]) -> tuple[str, ...]:
    """Translate rows whose first column is an entry id into labels in seed order.

    Args:
        db: The seeded database.
        rows: Rows holding an entry id first: a UUID from a raw connection, or the
            32-character hex string a backend's pooled connection decodes it to.

    Returns:
        The labels in seed order.
    """
    return in_seed_order(db.labels_of(row[0].hex if isinstance(row[0], uuid.UUID) else str(row[0]) for row in rows))


async def policy_visibility(conn: asyncpg.Connection, db: ScopedDb) -> dict[AccessMode, dict[str, tuple[str, ...]]]:
    """Return the rows each matrix scope reaches through the policies in every mode.

    Each statement runs in its own rolled-back transaction with no application predicate.

    Args:
        conn: A raw connection of the request role.
        db: The seeded database.

    Returns:
        The labels keyed like :data:`MATRIX_VISIBILITY`.
    """
    observed: dict[AccessMode, dict[str, tuple[str, ...]]] = {}
    for mode, statement in MODE_STATEMENTS.items():
        observed[mode] = {}
        for name, scope in MATRIX_SCOPES.items():
            async with scoped_transaction(conn, scope):
                observed[mode][name] = labels(db, await conn.fetch(statement))
    return observed
