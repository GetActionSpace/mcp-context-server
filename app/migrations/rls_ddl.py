"""Rendering of the PostgreSQL row-level security objects: statement text only, no I/O.

Row-level security enforces the access model of :mod:`app.access_scope` in the database
for the NOLOGIN policy role every request login role is a member of. This module renders
the objects that do it:

- four helper functions that hand the transaction's bound identity to the policies, the
  statements that bind the names in their bodies to pg_catalog while they are created,
  and the grant that lets the policy role execute them;
- per-command PERMISSIVE policies on ``context_entries`` (READ for SELECT, the owner
  stamp, the publish gate and the write right for INSERT, WRITE for UPDATE, OWNER for
  DELETE), on ``context_entry_grants`` (reads of the caller's own grants and of every grant
  on the caller's entries; inserts and deletes by the entry's owner; no update policy,
  because no grant is ever updated) and on every child table (reads follow the parent's
  read policy; inserts and deletes need WRITE on the parent; no update policy, because no
  request updates a child row: a request replaces a generation job by deleting it and
  inserting another, and only the table owner updates a job's state);
- the guard trigger that keeps ``owner_id`` immutable, visibility changes owner-only and
  publishing gated, which a policy cannot express because WITH CHECK does not see the old
  row;
- the read view of ``context_entries`` that request roles read through, the statements
  that leave SELECT for the policy role as the only privilege on it, and the function and
  INSTEAD OF trigger that keep it read-only.

Every policy expression of ``context_entries`` and of a child's parent comes from
:func:`app.access_scope.build_policy_predicate`, and the read view's condition from
:func:`app.access_scope.build_gated_read_predicate`; both use the composer the application
predicate uses, so they cannot drift apart. Policies apply to the policy role only, never
to PUBLIC, so every other role without the table owner's bypass reads nothing.
Row-level security is enabled without FORCE: the table owner, which runs migrations and
the command-line tools, keeps reading every row, and so does the read view it owns, which
carries the read predicate, gated on membership of the policy role, as an ordinary
condition.

Every object carries a fingerprint comment: the ``sha`` of its rendered statement, which
changes whenever this module renders something different, and the digest of the server's
own deparse of the object, which changes when the object is altered out of band. The
comment of each table's read policy also records whether row-level security was already
on before the migration first enabled it, which tells removal whether to switch it off.

Schema and role names are taken already quoted as SQL identifiers.
"""

import hashlib
import re
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final
from typing import NamedTuple

from app.access_scope import GROUPS_FUNCTION
from app.access_scope import MAY_PUBLISH_FUNCTION
from app.access_scope import MAY_WRITE_FUNCTION
from app.access_scope import PRINCIPAL_FUNCTION
from app.access_scope import READ_VIEW
from app.access_scope import AccessMode
from app.access_scope import build_gated_read_predicate
from app.access_scope import build_policy_predicate
from app.access_scope import policy_operands

FINGERPRINT_PREFIX: Final = 'mcp-context-server rls v1'

CONTEXT_TABLE: Final = 'context_entries'
GRANTS_TABLE: Final = 'context_entry_grants'
# Every table whose rows belong to one context entry, with the column naming that entry.
CHILD_PARENT_KEYS: Final[Mapping[str, str]] = MappingProxyType({
    'tags': 'context_entry_id',
    'image_attachments': 'context_entry_id',
    'embedding_metadata': 'context_id',
    'vec_context_embeddings': 'context_id',
    'vec_context_embeddings_compressed': 'context_id',
    'context_index_nodes': 'context_id',
    'generation_jobs': 'context_entry_id',
})
# The tables row-level security governs; each layout and feature set provisions a subset.
MANAGED_TABLES: Final[tuple[str, ...]] = (CONTEXT_TABLE, GRANTS_TABLE, *CHILD_PARENT_KEYS)
# The read policy of each managed table, whose fingerprint records whether row-level security
# was on before the migration first enabled it.
_READ_POLICIES: Final = frozenset(f'{table}_read' for table in MANAGED_TABLES)

GUARD_FUNCTION: Final = 'mcp_context_guard_access_columns'
GUARD_TRIGGER: Final = 'guard_context_entry_access_columns'
REJECT_FUNCTION: Final = 'mcp_context_reject_view_write'
REJECT_TRIGGER: Final = 'reject_writes'

# Alias of the parent entry in the subqueries of the grants and child policies, and of
# context_entries in the read view.
_PARENT: Final = 'ce'
_HEX_DIGEST = re.compile(r'[0-9a-f]{64}')
_FINGERPRINT = re.compile(
    rf'{re.escape(FINGERPRINT_PREFIX)} sha=([0-9a-f]{{64}}) def=([0-9a-f]{{64}})(?: preenabled=(true|false))?',
)
# The privileges a table's or a view's ACL holds, as GRANT and REVOKE name them, and ALL.
_ACL_PRIVILEGES: Final = frozenset({
    'ALL', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER', 'MAINTAIN',
})

# Each helper returns one transaction-local setting the request profile binds. An unset
# setting reads NULL and a setting cleared by COMMIT or RESET ALL reads '', and NULLIF turns
# both into no principal, no groups, no publish right and no write right.
_HELPER_BODIES: Final = (
    (PRINCIPAL_FUNCTION, 'text', "NULLIF(pg_catalog.current_setting('app.principal_id', true), '')"),
    (
        GROUPS_FUNCTION,
        'text[]',
        "COALESCE(NULLIF(pg_catalog.current_setting('app.groups', true), '')::pg_catalog.text[], '{}'::pg_catalog.text[])",
    ),
    (
        MAY_PUBLISH_FUNCTION,
        'boolean',
        "COALESCE(NULLIF(pg_catalog.current_setting('app.may_publish', true), '') = 'on', false)",
    ),
    (
        MAY_WRITE_FUNCTION,
        'boolean',
        "COALESCE(NULLIF(pg_catalog.current_setting('app.may_write', true), '') = 'on', false)",
    ),
)
# Run before the helpers are created, so every name in their bodies, the operators and casts
# NULLIF and COALESCE apply included, resolves in pg_catalog alone, whatever search path the
# migration connection uses. SET LOCAL ends with the transaction.
PIN_SEARCH_PATH: Final = 'SET LOCAL search_path = pg_catalog, pg_temp'


class RlsObjectKind(StrEnum):
    """The kinds of database object row-level security creates, as SQL names them."""

    FUNCTION = 'FUNCTION'
    POLICY = 'POLICY'
    TRIGGER = 'TRIGGER'
    VIEW = 'VIEW'


class ObjectKey(NamedTuple):
    """What identifies a row-level security object in the catalog, without rendering it.

    Attributes:
        kind: Function, policy, trigger or view.
        name: The object's own name.
        anchor: For a function, its schema-qualified signature; for a view, its
            schema-qualified name; for a policy or a trigger, the schema-qualified table
            or view it is defined on.
    """

    kind: RlsObjectKind
    name: str
    anchor: str

    @property
    def target(self) -> str:
        """The object as COMMENT ON and DROP name it, such as ``POLICY tags_read ON "public".tags``."""
        if self.kind in (RlsObjectKind.FUNCTION, RlsObjectKind.VIEW):
            return f'{self.kind} {self.anchor}'
        return f'{self.kind} {self.name} ON {self.anchor}'

    @property
    def drop_statement(self) -> str:
        """The statement that drops the object when it exists."""
        kind, _, reference = self.target.partition(' ')
        return f'DROP {kind} IF EXISTS {reference}'


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """A fingerprint comment as read back from the catalog.

    Attributes:
        sha: The digest of the statement the object was created from.
        definition: The digest of the server's deparse right after it was created.
        preenabled: On a table's read policy, whether row-level security was on before
            the migration first enabled it; None on every other object.
    """

    sha: str
    definition: str
    preenabled: bool | None


@dataclass(frozen=True, slots=True)
class RlsObject:
    """One row-level security object and the statement that creates it.

    Attributes:
        kind: Function, policy, trigger or view.
        name: The object's own name.
        anchor: For a function, its schema-qualified signature; for a view, its
            schema-qualified name; for a policy or a trigger, the schema-qualified table
            or view it is defined on.
        statement: The statement that creates the object; functions are created or
            replaced, policies, triggers and the view are created after their drop.
    """

    kind: RlsObjectKind
    name: str
    anchor: str
    statement: str

    @property
    def sha(self) -> str:
        """The digest of :attr:`statement`, which embeds the schema and the policy role."""
        return digest(self.statement)

    @property
    def key(self) -> ObjectKey:
        """What identifies the object in the catalog."""
        return ObjectKey(self.kind, self.name, self.anchor)

    @property
    def target(self) -> str:
        """The object as COMMENT ON and DROP name it, such as ``POLICY tags_read ON "public".tags``."""
        return self.key.target

    @property
    def drop_statement(self) -> str:
        """The statement that drops the object when it exists."""
        return self.key.drop_statement


def digest(text: str) -> str:
    """Return the lowercase hex SHA-256 of ``text``.

    Args:
        text: The text to digest.

    Returns:
        64 hex characters.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def fingerprint(*, sha: str, definition: str, preenabled: bool | None) -> str:
    """Return the fingerprint comment of an object.

    Args:
        sha: The digest of the object's rendered statement.
        definition: The digest of the server's deparse of the object.
        preenabled: For a table's read policy, whether row-level security was on before
            the migration first enabled it; None for every other object.

    Returns:
        ``'mcp-context-server rls v1 sha=<sha> def=<definition>'``, followed by
        ``' preenabled=true'`` or ``' preenabled=false'`` when ``preenabled`` is given.
    """
    comment = f'{FINGERPRINT_PREFIX} sha={sha} def={definition}'
    if preenabled is None:
        return comment
    return f'{comment} preenabled={"true" if preenabled else "false"}'


def parse_fingerprint(comment: str | None) -> Fingerprint | None:
    """Read a fingerprint comment back.

    Args:
        comment: An object's comment, or None when it has none.

    Returns:
        The fingerprint, or None when the comment is not one :func:`fingerprint` writes.
    """
    match = None if comment is None else _FINGERPRINT.fullmatch(comment)
    if match is None:
        return None
    preenabled = None if match[3] is None else match[3] == 'true'
    return Fingerprint(sha=match[1], definition=match[2], preenabled=preenabled)


def records_preenabled(obj: RlsObject) -> bool:
    """Return whether ``obj`` is a table's read policy, whose fingerprint records ``preenabled``.

    Args:
        obj: The object.

    Returns:
        True for the read policy of a managed table.
    """
    return obj.kind is RlsObjectKind.POLICY and obj.name in _READ_POLICIES


def comment_statement(obj: RlsObject, *, definition: str, preenabled: bool | None) -> str:
    """Return the COMMENT that attaches ``obj``'s fingerprint to it.

    Args:
        obj: The object.
        definition: The digest of the server's deparse of the object.
        preenabled: For a table's read policy, whether row-level security was on before
            the migration first enabled it; None for every other object.

    Returns:
        The COMMENT ON statement.

    Raises:
        ValueError: If ``definition`` is not a lowercase hex SHA-256 digest, which is
            interpolated into a string literal, or if ``preenabled`` is missing on a read
            policy or given for any other object.
    """
    if _HEX_DIGEST.fullmatch(definition) is None:
        raise ValueError(f'{definition!r} is not a SHA-256 hex digest')
    if (preenabled is None) == records_preenabled(obj):
        raise ValueError(f'{obj.target} must record preenabled exactly when it is a read policy of a managed table')
    return f"COMMENT ON {obj.target} IS '{fingerprint(sha=obj.sha, definition=definition, preenabled=preenabled)}'"


def helper_functions(schema: str) -> tuple[RlsObject, RlsObject, RlsObject, RlsObject]:
    """Return the principal, groups, publish-right and write-right helpers the policy operands call.

    They are STABLE and PARALLEL SAFE SQL functions running as the invoker, each with a
    SQL-standard ``RETURN`` body and no SET clause, so the planner inlines every call into
    the expression around it: a policy evaluated once per row pays no function call. A
    SQL-standard body is parsed when the function is created, so its names stay bound to
    what they resolved to then; the migration creates the helpers after
    :data:`PIN_SEARCH_PATH`, and no search path a caller sets can change them.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The four helper functions, in that order.
    """
    principal, groups, may_publish, may_write = (
        RlsObject(
            kind=RlsObjectKind.FUNCTION,
            name=name,
            anchor=f'{schema}.{name}()',
            statement=(
                f'CREATE OR REPLACE FUNCTION {schema}.{name}() RETURNS {returns} '
                f'LANGUAGE sql STABLE PARALLEL SAFE RETURN {body}'
            ),
        )
        for name, returns, body in _HELPER_BODIES
    )
    return principal, groups, may_publish, may_write


def restore_search_path(search_path: str) -> str:
    """Return the statement that sets the search path back after :data:`PIN_SEARCH_PATH`, until the transaction ends.

    The statements that follow the helpers in the migration transaction read back the
    server's deparse of what they create, which names a relation outside the search path
    with its schema, so they must run under the connection's own search path.

    Args:
        search_path: The search path as ``current_setting('search_path')`` read it before the pin.

    Returns:
        The ``set_config`` call, the value as a string literal.
    """
    return f"SELECT pg_catalog.set_config('search_path', {_string_literal(search_path)}, true)"


def helper_grant(schema: str, role: str) -> str:
    """Return the grant that lets the policy role execute the helper functions.

    Args:
        schema: The schema of the tables, quoted.
        role: The policy role, quoted.

    Returns:
        The GRANT EXECUTE statement.
    """
    functions = ', '.join(helper.anchor for helper in helper_functions(schema))
    return f'GRANT EXECUTE ON FUNCTION {functions} TO {role}'


def guard_function(schema: str) -> RlsObject:
    """Return the trigger function guarding the access columns of ``context_entries``.

    A transaction that binds no principal passes through: the request role then matches no
    update policy, so its updates never reach the trigger, and a maintenance transaction,
    such as the owner reassignment of the command line, must pass.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The trigger function.
    """
    return RlsObject(
        kind=RlsObjectKind.FUNCTION,
        name=GUARD_FUNCTION,
        anchor=f'{schema}.{GUARD_FUNCTION}()',
        statement=(
            f'CREATE OR REPLACE FUNCTION {schema}.{GUARD_FUNCTION}() RETURNS trigger\n'
            '    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$\n'
            'DECLARE\n'
            f'    principal text := {schema}.{PRINCIPAL_FUNCTION}();\n'
            'BEGIN\n'
            '    IF principal IS NULL THEN\n'
            '        RETURN NEW;\n'
            '    END IF;\n'
            '    IF NEW.owner_id IS DISTINCT FROM OLD.owner_id THEN\n'
            "        RAISE EXCEPTION 'owner_id of context entry % is immutable', OLD.id USING ERRCODE = '42501';\n"
            '    END IF;\n'
            '    IF NEW.visibility IS DISTINCT FROM OLD.visibility THEN\n'
            '        IF OLD.owner_id IS DISTINCT FROM principal THEN\n'
            "            RAISE EXCEPTION 'only the owner may change the visibility of context entry %', OLD.id "
            "USING ERRCODE = '42501';\n"
            '        END IF;\n'
            f"        IF NEW.visibility = 'public' AND NOT {schema}.{MAY_PUBLISH_FUNCTION}() THEN\n"
            "            RAISE EXCEPTION 'publishing context entry % requires the publish role', OLD.id "
            "USING ERRCODE = '42501';\n"
            '        END IF;\n'
            '    END IF;\n'
            '    RETURN NEW;\n'
            'END\n'
            '$$'
        ),
    )


def guard_trigger(schema: str) -> RlsObject:
    """Return the trigger that runs the guard for updates naming ``owner_id`` or ``visibility``.

    Updates that set neither column, such as text and metadata edits, never fire it.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The trigger.
    """
    return RlsObject(
        kind=RlsObjectKind.TRIGGER,
        name=GUARD_TRIGGER,
        anchor=f'{schema}.{CONTEXT_TABLE}',
        statement=(
            f'CREATE TRIGGER {GUARD_TRIGGER} BEFORE UPDATE OF owner_id, visibility ON {schema}.{CONTEXT_TABLE} '
            f'FOR EACH ROW EXECUTE FUNCTION {schema}.{GUARD_FUNCTION}()'
        ),
    )


def policy_names(table: str) -> tuple[str, ...]:
    """Return the names of ``table``'s policies.

    Args:
        table: A managed table.

    Returns:
        The table name joined with each command the table has a policy for.

    Raises:
        ValueError: If ``table`` is not managed.
    """
    if table == CONTEXT_TABLE:
        commands: tuple[str, ...] = ('read', 'insert', 'update', 'delete')
    elif table == GRANTS_TABLE or table in CHILD_PARENT_KEYS:
        commands = ('read', 'insert', 'delete')
    else:
        raise ValueError(f'{table!r} is not a table row-level security manages')
    return tuple(f'{table}_{command}' for command in commands)


def table_policies(table: str, *, schema: str, role: str) -> tuple[RlsObject, ...]:
    """Return ``table``'s policies, in the order of :func:`policy_names`.

    Args:
        table: A managed table; :func:`policy_names` refuses any other.
        schema: The schema of the tables, quoted.
        role: The policy role, quoted.

    Returns:
        The policies.
    """
    names = policy_names(table)
    if table == CONTEXT_TABLE:
        clauses = _context_entries_clauses(schema)
    elif table == GRANTS_TABLE:
        clauses = _grants_clauses(schema)
    else:
        clauses = _child_clauses(table, CHILD_PARENT_KEYS[table], schema)
    return tuple(
        RlsObject(
            kind=RlsObjectKind.POLICY,
            name=name,
            anchor=f'{schema}.{table}',
            statement=f'CREATE POLICY {name} ON {schema}.{table} AS PERMISSIVE FOR {command} TO {role}{clause}',
        )
        for name, (command, clause) in zip(names, clauses, strict=True)
    )


def enable_statement(table: str, *, schema: str) -> str:
    """Return the statement that enables row-level security on ``table``, never forcing it.

    Args:
        table: A managed table.
        schema: The schema of the tables, quoted.

    Returns:
        The ALTER TABLE statement.

    Raises:
        ValueError: If ``table`` is not managed.
    """
    if table not in MANAGED_TABLES:
        raise ValueError(f'{table!r} is not a table row-level security manages')
    return f'ALTER TABLE {schema}.{table} ENABLE ROW LEVEL SECURITY'


def disable_statement(table: str, *, schema: str) -> str:
    """Return the statement that disables row-level security on ``table``.

    Args:
        table: A managed table.
        schema: The schema of the tables, quoted.

    Returns:
        The ALTER TABLE statement.

    Raises:
        ValueError: If ``table`` is not managed.
    """
    if table not in MANAGED_TABLES:
        raise ValueError(f'{table!r} is not a table row-level security manages')
    return f'ALTER TABLE {schema}.{table} DISABLE ROW LEVEL SECURITY'


def _context_entries_clauses(schema: str) -> tuple[tuple[str, str], ...]:
    """Return the command and the USING and WITH CHECK clauses of each context_entries policy."""
    read = build_policy_predicate(AccessMode.READ, outer=CONTEXT_TABLE, schema=schema)
    write = build_policy_predicate(AccessMode.WRITE, outer=CONTEXT_TABLE, schema=schema)
    owner = build_policy_predicate(AccessMode.OWNER, outer=CONTEXT_TABLE, schema=schema)
    operands = policy_operands(schema)
    insertable = f"{owner} AND ({CONTEXT_TABLE}.visibility <> 'public' OR {operands.may_publish}) AND {operands.may_write}"
    return (
        ('SELECT', f' USING ({read})'),
        ('INSERT', f' WITH CHECK ({insertable})'),
        ('UPDATE', f' USING ({write}) WITH CHECK ({write})'),
        ('DELETE', f' USING ({owner})'),
    )


def _grants_clauses(schema: str) -> tuple[tuple[str, str], ...]:
    """Return the command and clause of each context_entry_grants policy.

    Three arms key on the bound identity: a grantee reads the grants naming it or one of its
    groups, and the entry's owner reads, inserts and deletes every grant on the entry. No
    policy holds a subquery. Every scoped read reads this table, both through the
    application's grant arm and through the read policy of ``context_entries``, so a policy
    here that read ``context_entries`` would reach that table's read policy, which reads this
    table again, and PostgreSQL rejects such a policy as infinitely recursive. The owner arms
    therefore compare the grant's own ``entry_owner_id`` with the bound principal: the
    composite foreign key from ``(context_entry_id, entry_owner_id)`` onto the entry's
    ``(id, owner_id)`` keeps that column equal to the entry's owner, so a row whose recorded
    owner is the caller is a grant on the caller's entry. A function looking the owner up
    could not stand in for the column, because a function that reads a table the caller
    may not read must run as its owner, and the planner never inlines such a function.

    The policies call the helpers directly rather than through their scalar subqueries, and
    the planner inlines every call, so a policy evaluated once per grant row calls no
    function. The insert check also pins the recorded grantor to the bound principal; with
    no update policy, no request can change a grantor afterwards.

    Returns:
        The SELECT, INSERT and DELETE commands with their clauses.
    """
    principal, groups, _, _ = helper_functions(schema)
    own_entry = f'entry_owner_id = {principal.anchor}'
    readable = (
        f"(principal_type = 'user' AND principal_id = {principal.anchor}) "
        f"OR (principal_type = 'group' AND principal_id = ANY({groups.anchor})) "
        f'OR {own_entry}'
    )
    return (
        ('SELECT', f' USING ({readable})'),
        ('INSERT', f' WITH CHECK (granted_by = {principal.anchor} AND {own_entry})'),
        ('DELETE', f' USING ({own_entry})'),
    )


def _child_clauses(table: str, key: str, schema: str) -> tuple[tuple[str, str], ...]:
    """Return the command and clause of each policy of the child ``table``, whose ``key`` names its entry.

    The read policy's subquery reads the parent through the parent's own read policy,
    because policy subqueries run as the invoker. An invisible parent and a missing one
    refuse an insert alike, so the foreign key reveals nothing.

    Returns:
        The SELECT, INSERT and DELETE commands with their clauses.
    """
    parent = f'EXISTS (SELECT 1 FROM {schema}.{CONTEXT_TABLE} {_PARENT} WHERE {_PARENT}.id = {table}.{key}'
    writable = f'{parent} AND {build_policy_predicate(AccessMode.WRITE, outer=_PARENT, schema=schema)})'
    return (
        ('SELECT', f' USING ({parent}))'),
        ('INSERT', f' WITH CHECK ({writable})'),
        ('DELETE', f' USING ({writable})'),
    )


def read_view(columns: Sequence[str], *, schema: str, role_name: str) -> RlsObject:
    """Return the view through which request roles read ``context_entries``.

    The view lists ``columns`` in order and keeps a row when the reading role is a member
    of the policy role and the bound identity may read the row. It carries no option: it
    runs as its owner, the table owner, whom the policies do not filter, and the planner
    pulls it up into the query, so its conditions and the query's own compete by cost
    alone and the query's conditions keep their indexes. The membership test sits inside
    the identity operands and the public arm of the read predicate
    (:func:`app.access_scope.build_gated_read_predicate`), each a scalar subquery the
    planner evaluates once per statement, so it never stands above a join of the view as
    a condition of its own. For a role outside the policy role the identity is NULL, the
    groups are empty and the public arm is false, so such a role that obtains SELECT on
    the view reads nothing, as the policies give it nothing.

    Args:
        columns: Every column of ``context_entries``, quoted, in attribute order.
        schema: The schema of the tables, quoted.
        role_name: The policy role's own name, which the membership test takes as a
            string literal.

    Returns:
        The view.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if not columns:
        raise ValueError('the read view needs at least one column of context_entries')
    select_list = ', '.join(f'{_PARENT}.{column}' for column in columns)
    gate = f"pg_catalog.pg_has_role(CURRENT_USER, {_string_literal(role_name)}, 'USAGE')"
    readable = build_gated_read_predicate(outer=_PARENT, schema=schema, gate=gate)
    return RlsObject(
        kind=RlsObjectKind.VIEW,
        name=READ_VIEW,
        anchor=f'{schema}.{READ_VIEW}',
        statement=(
            f'CREATE VIEW {schema}.{READ_VIEW} AS SELECT {select_list} '
            f'FROM {schema}.{CONTEXT_TABLE} AS {_PARENT} WHERE {readable}'
        ),
    )


def read_view_drop(schema: str) -> str:
    """Return the statement that drops the read view, and its trigger with it, when it exists.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The DROP VIEW statement.
    """
    return f'DROP VIEW IF EXISTS {schema}.{READ_VIEW}'


def read_view_grant(schema: str, role: str) -> str:
    """Return the grant that lets the policy role read through the view, and nothing more.

    Args:
        schema: The schema of the tables, quoted.
        role: The policy role, quoted.

    Returns:
        The GRANT SELECT statement.
    """
    return f'GRANT SELECT ON {schema}.{READ_VIEW} TO {role}'


def read_view_revoke(schema: str, grantee: str, privileges: Sequence[str]) -> str:
    """Return the statement that revokes ``privileges`` on the read view from ``grantee``.

    Args:
        schema: The schema of the tables, quoted.
        grantee: The role, quoted, or ``PUBLIC``.
        privileges: The privileges as the view's ACL names them, or ``ALL``.

    Returns:
        The REVOKE statement.

    Raises:
        ValueError: If ``privileges`` is empty or names anything but a table privilege;
            each is interpolated into the statement.
    """
    if not privileges or any(privilege not in _ACL_PRIVILEGES for privilege in privileges):
        raise ValueError(f'{list(privileges)!r} is not a list of table privileges')
    return f'REVOKE {", ".join(privileges)} ON {schema}.{READ_VIEW} FROM {grantee}'


def reject_function(schema: str) -> RlsObject:
    """Return the trigger function that refuses every write through the read view.

    The view is automatically updatable, and a write through it would run as its owner,
    past every write policy, so the function raises insufficient privilege instead, should
    a later grant ever give anyone a write privilege on the view.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The trigger function.
    """
    return RlsObject(
        kind=RlsObjectKind.FUNCTION,
        name=REJECT_FUNCTION,
        anchor=f'{schema}.{REJECT_FUNCTION}()',
        statement=(
            f'CREATE OR REPLACE FUNCTION {schema}.{REJECT_FUNCTION}() RETURNS trigger\n'
            '    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$\n'
            'BEGIN\n'
            f"    RAISE EXCEPTION '{READ_VIEW} is read-only' USING ERRCODE = '42501';\n"
            'END\n'
            '$$'
        ),
    )


def reject_trigger(schema: str) -> RlsObject:
    """Return the trigger that runs :func:`reject_function` instead of every write on the read view.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The trigger.
    """
    return RlsObject(
        kind=RlsObjectKind.TRIGGER,
        name=REJECT_TRIGGER,
        anchor=f'{schema}.{READ_VIEW}',
        statement=(
            f'CREATE TRIGGER {REJECT_TRIGGER} INSTEAD OF INSERT OR UPDATE OR DELETE ON {schema}.{READ_VIEW} '
            f'FOR EACH ROW EXECUTE FUNCTION {schema}.{REJECT_FUNCTION}()'
        ),
    )


def _string_literal(value: str) -> str:
    """Return ``value`` as a SQL string literal, doubling embedded quotes."""
    return "'" + value.replace("'", "''") + "'"
