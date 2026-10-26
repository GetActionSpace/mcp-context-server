"""Tests for the rendering of the PostgreSQL row-level security objects.

Covers app.migrations.rls_ddl: the helper functions that hand the bound identity
to the policies and their EXECUTE grant, the per-command policies of
``context_entries``, the grants table and every child table, the guard trigger,
the policy names, the ``sha`` of every rendered statement, the fingerprint
comment and the COMMENT, DROP and ENABLE statements. Policy expressions come
from the one predicate composer in app.access_scope, so they are asserted
against its output rather than restated.
"""

import hashlib
import inspect
import re

import pytest

from app.access_scope import AccessMode
from app.access_scope import build_policy_predicate
from app.access_scope import policy_operands
from app.backends.postgresql_backend.session_profiles import BIND_IDENTITY_STATEMENT
from app.migrations import rls_ddl
from app.migrations.rls_ddl import CHILD_PARENT_KEYS
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import PIN_SEARCH_PATH
from app.migrations.rls_ddl import Fingerprint
from app.migrations.rls_ddl import ObjectKey
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import disable_statement
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import fingerprint
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import parse_fingerprint
from app.migrations.rls_ddl import policy_names
from app.migrations.rls_ddl import records_preenabled
from app.migrations.rls_ddl import restore_search_path
from app.migrations.rls_ddl import table_policies

SCHEMA = '"public"'
ROLE = '"mcp_context_request"'
P = f'(SELECT {SCHEMA}.mcp_context_principal_id())'
G = f'(SELECT {SCHEMA}.mcp_context_groups())::text[]'
M = f'(SELECT {SCHEMA}.mcp_context_may_publish())'
W = f'(SELECT {SCHEMA}.mcp_context_may_write())'
# The principal helper called directly, as the grants policies call it.
PRINCIPAL_CALL = f'{SCHEMA}.mcp_context_principal_id()'


def _read(outer: str) -> str:
    return build_policy_predicate(AccessMode.READ, outer=outer, schema=SCHEMA)


def _write(outer: str) -> str:
    return build_policy_predicate(AccessMode.WRITE, outer=outer, schema=SCHEMA)


def _owner(outer: str) -> str:
    return build_policy_predicate(AccessMode.OWNER, outer=outer, schema=SCHEMA)


def _statements(table: str) -> dict[str, str]:
    return {policy.name: policy.statement for policy in table_policies(table, schema=SCHEMA, role=ROLE)}


# ============================================================================
# Managed tables
# ============================================================================


def test_managed_tables_are_the_context_table_the_grants_table_and_every_child_in_order() -> None:
    """Every table whose rows belong to an entry is managed, the context table first."""
    assert MANAGED_TABLES == (
        'context_entries', 'context_entry_grants', 'tags', 'image_attachments', 'embedding_metadata',
        'vec_context_embeddings', 'vec_context_embeddings_compressed', 'context_index_nodes', 'generation_jobs',
    )
    assert dict(CHILD_PARENT_KEYS) == {
        'tags': 'context_entry_id',
        'image_attachments': 'context_entry_id',
        'embedding_metadata': 'context_id',
        'vec_context_embeddings': 'context_id',
        'vec_context_embeddings_compressed': 'context_id',
        'context_index_nodes': 'context_id',
        'generation_jobs': 'context_entry_id',
    }


# ============================================================================
# Helper functions
# ============================================================================


def test_helper_functions_read_the_bound_identity_and_fail_closed() -> None:
    """Unset and RESET values read as no principal, no groups, no publish right and no write right."""
    statements = {helper.name: helper.statement for helper in helper_functions(SCHEMA)}

    assert statements == {
        'mcp_context_principal_id': (
            f'CREATE OR REPLACE FUNCTION {SCHEMA}.mcp_context_principal_id() RETURNS text '
            'LANGUAGE sql STABLE PARALLEL SAFE '
            "RETURN NULLIF(pg_catalog.current_setting('app.principal_id', true), '')"
        ),
        'mcp_context_groups': (
            f'CREATE OR REPLACE FUNCTION {SCHEMA}.mcp_context_groups() RETURNS text[] '
            'LANGUAGE sql STABLE PARALLEL SAFE '
            "RETURN COALESCE(NULLIF(pg_catalog.current_setting('app.groups', true), '')::pg_catalog.text[], "
            "'{}'::pg_catalog.text[])"
        ),
        'mcp_context_may_publish': (
            f'CREATE OR REPLACE FUNCTION {SCHEMA}.mcp_context_may_publish() RETURNS boolean '
            'LANGUAGE sql STABLE PARALLEL SAFE '
            "RETURN COALESCE(NULLIF(pg_catalog.current_setting('app.may_publish', true), '') = 'on', false)"
        ),
        'mcp_context_may_write': (
            f'CREATE OR REPLACE FUNCTION {SCHEMA}.mcp_context_may_write() RETURNS boolean '
            'LANGUAGE sql STABLE PARALLEL SAFE '
            "RETURN COALESCE(NULLIF(pg_catalog.current_setting('app.may_write', true), '') = 'on', false)"
        ),
    }
    for helper in helper_functions(SCHEMA):
        assert helper.kind is RlsObjectKind.FUNCTION
        assert 'SECURITY DEFINER' not in helper.statement


def test_helper_functions_have_inlinable_bodies_bound_at_creation() -> None:
    """A SQL-standard body and no SET clause: the planner inlines the call, and the body's names are bound when it is created.

    A ``SET`` clause keeps PostgreSQL from inlining a SQL function, and a string body is
    parsed again, under the caller's search path, every time it is inlined.
    """
    for helper in helper_functions(SCHEMA):
        assert ' RETURN ' in helper.statement
        assert ' SET ' not in helper.statement
        assert '$$' not in helper.statement
        assert ' AS ' not in helper.statement


def test_helpers_are_created_under_a_search_path_of_pg_catalog_alone() -> None:
    """The statement that pins name resolution while the helpers are created changes nothing past the transaction."""
    assert PIN_SEARCH_PATH == 'SET LOCAL search_path = pg_catalog, pg_temp'


@pytest.mark.parametrize(
    ('search_path', 'expected'),
    [
        ('"tenant one", public', "SELECT pg_catalog.set_config('search_path', '\"tenant one\", public', true)"),
        ("\"o'brien\", public", "SELECT pg_catalog.set_config('search_path', '\"o''brien\", public', true)"),
    ],
)
def test_the_search_path_is_restored_transaction_locally_as_a_string_literal(search_path: str, expected: str) -> None:
    """The restore sets the value read before the pin for the rest of the transaction, its quotes doubled."""
    assert restore_search_path(search_path) == expected


def test_helpers_read_exactly_the_settings_the_bind_statement_sets() -> None:
    """Each helper reads one transaction-local setting the request profile binds."""
    settings = ('app.principal_id', 'app.groups', 'app.may_publish', 'app.may_write')
    for helper, setting in zip(helper_functions(SCHEMA), settings, strict=True):
        assert f"current_setting('{setting}', true)" in helper.statement
        assert f"set_config('{setting}', " in BIND_IDENTITY_STATEMENT


def test_helpers_are_the_functions_the_policy_operands_call() -> None:
    """The predicate composer's operands call the functions this module creates."""
    operands = policy_operands(SCHEMA)
    principal, groups, may_publish, may_write = helper_functions(SCHEMA)

    assert operands.principal == f'(SELECT {principal.anchor})' == P
    assert operands.groups == f'(SELECT {groups.anchor})::text[]' == G
    assert operands.may_publish == f'(SELECT {may_publish.anchor})' == M
    assert operands.may_write == f'(SELECT {may_write.anchor})' == W


def test_helper_grant_gives_the_policy_role_execute_on_every_helper() -> None:
    """One GRANT names the four helpers and the policy role."""
    assert helper_grant(SCHEMA, ROLE) == (
        f'GRANT EXECUTE ON FUNCTION {SCHEMA}.mcp_context_principal_id(), {SCHEMA}.mcp_context_groups(), '
        f'{SCHEMA}.mcp_context_may_publish(), {SCHEMA}.mcp_context_may_write() TO {ROLE}'
    )


# ============================================================================
# context_entries policies
# ============================================================================


def test_context_entries_policies_render_the_composer_predicates_per_command() -> None:
    """SELECT is READ, INSERT stamps the caller and needs the publish and write rights, UPDATE is WRITE, DELETE is OWNER."""
    head = f'ON {SCHEMA}.context_entries AS PERMISSIVE FOR'

    assert _statements('context_entries') == {
        'context_entries_read': (
            f'CREATE POLICY context_entries_read {head} SELECT TO {ROLE} USING ({_read("context_entries")})'
        ),
        'context_entries_insert': (
            f'CREATE POLICY context_entries_insert {head} INSERT TO {ROLE} WITH CHECK '
            f"({_owner('context_entries')} AND (context_entries.visibility <> 'public' OR {M}) AND {W})"
        ),
        'context_entries_update': (
            f'CREATE POLICY context_entries_update {head} UPDATE TO {ROLE} '
            f'USING ({_write("context_entries")}) WITH CHECK ({_write("context_entries")})'
        ),
        'context_entries_delete': (
            f'CREATE POLICY context_entries_delete {head} DELETE TO {ROLE} USING ({_owner("context_entries")})'
        ),
    }


# ============================================================================
# context_entry_grants policies
# ============================================================================


def test_grants_policies_let_grantees_read_their_grants_and_the_entry_owner_read_insert_and_delete() -> None:
    """Every arm keys on the bound principal or groups, the owner arms on the grant's recorded entry owner."""
    head = f'ON {SCHEMA}.context_entry_grants AS PERMISSIVE FOR'

    assert _statements('context_entry_grants') == {
        'context_entry_grants_read': (
            f'CREATE POLICY context_entry_grants_read {head} SELECT TO {ROLE} USING '
            f"((principal_type = 'user' AND principal_id = {PRINCIPAL_CALL}) "
            f"OR (principal_type = 'group' AND principal_id = ANY({SCHEMA}.mcp_context_groups())) "
            f'OR entry_owner_id = {PRINCIPAL_CALL})'
        ),
        'context_entry_grants_insert': (
            f'CREATE POLICY context_entry_grants_insert {head} INSERT TO {ROLE} WITH CHECK '
            f'(granted_by = {PRINCIPAL_CALL} AND entry_owner_id = {PRINCIPAL_CALL})'
        ),
        'context_entry_grants_delete': (
            f'CREATE POLICY context_entry_grants_delete {head} DELETE TO {ROLE} USING (entry_owner_id = {PRINCIPAL_CALL})'
        ),
    }


def test_grants_policies_hold_no_subquery() -> None:
    """No grants policy reads another relation, so the policy expansion ends at the grants table.

    PostgreSQL rejects as infinitely recursive any policy with a subquery that is reached
    again while its own table's policies expand: every scoped read reads the grants table,
    and a grants policy that read ``context_entries`` would reach that table's read policy,
    which reads the grants table again. Each policy calls the helpers directly, not even
    through the scalar subquery around a helper, and the foreign key from
    ``(context_entry_id, entry_owner_id)`` keeps the recorded owner exact without one.
    """
    for policy in table_policies('context_entry_grants', schema=SCHEMA, role=ROLE):
        clause = policy.statement.partition(f' TO {ROLE} ')[2]
        assert 'SELECT' not in clause, policy.name
        assert 'context_entries' not in clause, policy.name


# ============================================================================
# Child policies
# ============================================================================


@pytest.mark.parametrize(('table', 'key'), list(CHILD_PARENT_KEYS.items()))
def test_child_policies_inherit_read_and_require_write_on_the_parent(table: str, key: str) -> None:
    """SELECT follows the parent's read policy; INSERT and DELETE need WRITE on the parent; no UPDATE policy."""
    head = f'ON {SCHEMA}.{table} AS PERMISSIVE FOR'
    parent = f'EXISTS (SELECT 1 FROM {SCHEMA}.context_entries ce WHERE ce.id = {table}.{key}'
    writable = f'{parent} AND {_write("ce")})'

    assert _statements(table) == {
        f'{table}_read': f'CREATE POLICY {table}_read {head} SELECT TO {ROLE} USING ({parent}))',
        f'{table}_insert': f'CREATE POLICY {table}_insert {head} INSERT TO {ROLE} WITH CHECK ({writable})',
        f'{table}_delete': f'CREATE POLICY {table}_delete {head} DELETE TO {ROLE} USING ({writable})',
    }


# ============================================================================
# Policy shape and names
# ============================================================================


@pytest.mark.parametrize('table', MANAGED_TABLES)
def test_policy_names_are_the_table_name_and_the_command(table: str) -> None:
    """The names are fixed per table, so they do not depend on the policy role setting."""
    expected = {
        'context_entries': (
            'context_entries_read', 'context_entries_insert', 'context_entries_update', 'context_entries_delete',
        ),
        'context_entry_grants': (
            'context_entry_grants_read', 'context_entry_grants_insert', 'context_entry_grants_delete',
        ),
    }.get(table, (f'{table}_read', f'{table}_insert', f'{table}_delete'))

    assert policy_names(table) == expected
    assert tuple(policy.name for policy in table_policies(table, schema=SCHEMA, role=ROLE)) == expected


@pytest.mark.parametrize('table', ['embedding_chunks', 'compression_metadata', 'context_entries_readable'])
def test_a_table_outside_the_managed_set_has_no_policies(table: str) -> None:
    """Asking for an unmanaged table's policies is a programming error."""
    with pytest.raises(ValueError, match=table):
        policy_names(table)
    with pytest.raises(ValueError, match=table):
        table_policies(table, schema=SCHEMA, role=ROLE)


@pytest.mark.parametrize('table', MANAGED_TABLES)
def test_every_policy_is_permissive_and_applies_to_the_policy_role_only(table: str) -> None:
    """No RESTRICTIVE policy and no policy TO PUBLIC: every other role is denied by default."""
    for policy in table_policies(table, schema=SCHEMA, role=ROLE):
        assert policy.kind is RlsObjectKind.POLICY
        assert policy.anchor == f'{SCHEMA}.{table}'
        assert ' AS PERMISSIVE FOR ' in policy.statement
        assert f' TO {ROLE} ' in policy.statement
        assert ' TO PUBLIC' not in policy.statement.upper()
        assert 'RESTRICTIVE' not in policy.statement


# ============================================================================
# Guard trigger
# ============================================================================


def test_guard_function_keeps_owner_immutable_and_visibility_owner_only_and_publish_gated() -> None:
    """Every refusal raises 42501, and a transaction without a bound principal passes through."""
    function = guard_function(SCHEMA)

    assert function.kind is RlsObjectKind.FUNCTION
    assert function.name == 'mcp_context_guard_access_columns'
    assert function.anchor == f'{SCHEMA}.mcp_context_guard_access_columns()'
    assert function.statement == (
        f'CREATE OR REPLACE FUNCTION {SCHEMA}.mcp_context_guard_access_columns() RETURNS trigger\n'
        '    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$\n'
        'DECLARE\n'
        f'    principal text := {SCHEMA}.mcp_context_principal_id();\n'
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
        f"        IF NEW.visibility = 'public' AND NOT {SCHEMA}.mcp_context_may_publish() THEN\n"
        "            RAISE EXCEPTION 'publishing context entry % requires the publish role', OLD.id "
        "USING ERRCODE = '42501';\n"
        '        END IF;\n'
        '    END IF;\n'
        '    RETURN NEW;\n'
        'END\n'
        '$$'
    )
    assert 'SECURITY DEFINER' not in function.statement


def test_guard_trigger_fires_only_for_updates_naming_the_access_columns() -> None:
    """A text or metadata edit never fires it; an owner or visibility change always does."""
    trigger = guard_trigger(SCHEMA)

    assert trigger.kind is RlsObjectKind.TRIGGER
    assert trigger.name == 'guard_context_entry_access_columns'
    assert trigger.anchor == f'{SCHEMA}.context_entries'
    assert trigger.statement == (
        f'CREATE TRIGGER guard_context_entry_access_columns BEFORE UPDATE OF owner_id, visibility '
        f'ON {SCHEMA}.context_entries FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.mcp_context_guard_access_columns()'
    )


# ============================================================================
# Fingerprints, comments, drops and enable
# ============================================================================


def _all_objects(schema: str, role: str) -> list[RlsObject]:
    objects = [*helper_functions(schema), guard_function(schema), guard_trigger(schema)]
    for table in MANAGED_TABLES:
        objects.extend(table_policies(table, schema=schema, role=role))
    return objects


def test_sha_is_the_digest_of_the_rendered_statement() -> None:
    """The sha detects any change in what this code renders for the object."""
    for obj in _all_objects(SCHEMA, ROLE):
        assert obj.sha == hashlib.sha256(obj.statement.encode('utf-8')).hexdigest()


def test_sha_embeds_the_schema_and_the_policy_role() -> None:
    """Changing the schema changes every sha; changing the policy role changes every policy's sha."""
    base = _all_objects(SCHEMA, ROLE)
    other_schema = _all_objects('"tenant one"', ROLE)
    other_role = _all_objects(SCHEMA, '"other_request"')

    assert [obj.sha for obj in base] == [obj.sha for obj in _all_objects(SCHEMA, ROLE)]
    assert all(a.sha != b.sha for a, b in zip(base, other_schema, strict=True))
    for a, b in zip(base, other_role, strict=True):
        assert (a.sha != b.sha) is (a.kind is RlsObjectKind.POLICY)


def test_fingerprint_names_the_version_and_both_digests() -> None:
    """The fingerprint carries the rendered sha and the server deparse digest, and preenabled when recorded."""
    sha = 'a' * 64
    definition = 'b' * 64

    assert fingerprint(sha=sha, definition=definition, preenabled=None) == (
        f'mcp-context-server rls v1 sha={sha} def={definition}'
    )
    assert fingerprint(sha=sha, definition=definition, preenabled=False) == (
        f'mcp-context-server rls v1 sha={sha} def={definition} preenabled=false'
    )
    assert fingerprint(sha=sha, definition=definition, preenabled=True) == (
        f'mcp-context-server rls v1 sha={sha} def={definition} preenabled=true'
    )


@pytest.mark.parametrize('preenabled', [None, False, True])
def test_a_fingerprint_reads_back_as_it_was_written(preenabled: bool | None) -> None:
    """Parsing a fingerprint returns both digests and the preenabled value it carries."""
    sha = hashlib.sha256(b'statement').hexdigest()
    definition = hashlib.sha256(b'deparse').hexdigest()

    assert parse_fingerprint(fingerprint(sha=sha, definition=definition, preenabled=preenabled)) == Fingerprint(
        sha=sha, definition=definition, preenabled=preenabled,
    )


@pytest.mark.parametrize(
    'comment',
    [
        None,
        '',
        'policy installed by hand',
        f'mcp-context-server rls v1 sha={"a" * 64}',
        f'mcp-context-server rls v1 sha={"a" * 63} def={"b" * 64}',
        f'mcp-context-server rls v1 sha={"A" * 64} def={"b" * 64}',
        f'mcp-context-server rls v2 sha={"a" * 64} def={"b" * 64}',
        f'mcp-context-server rls v1 sha={"a" * 64} def={"b" * 64} preenabled=maybe',
        f'mcp-context-server rls v1 sha={"a" * 64} def={"b" * 64} trailing',
    ],
)
def test_anything_but_a_fingerprint_reads_as_no_fingerprint(comment: str | None) -> None:
    """A missing, foreign or damaged comment is no fingerprint, so the object counts as changed out of band."""
    assert parse_fingerprint(comment) is None


def test_each_tables_read_policy_and_nothing_else_records_preenabled() -> None:
    """Only the read policy of a managed table carries whether row-level security was on before the migration."""
    for table in MANAGED_TABLES:
        read, *others = table_policies(table, schema=SCHEMA, role=ROLE)
        assert records_preenabled(read), read.name
        assert not any(records_preenabled(policy) for policy in others), table
    non_policies = (*helper_functions(SCHEMA), guard_function(SCHEMA), guard_trigger(SCHEMA))
    assert not any(records_preenabled(obj) for obj in non_policies)


def test_comment_statement_carries_preenabled_on_read_policies_only() -> None:
    """A read policy's comment must say whether row-level security was on before; no other comment may."""
    definition = hashlib.sha256(b'deparse').hexdigest()
    read, insert, _ = table_policies('tags', schema=SCHEMA, role=ROLE)

    assert comment_statement(read, definition=definition, preenabled=True) == (
        f'COMMENT ON POLICY tags_read ON {SCHEMA}.tags IS '
        f"'mcp-context-server rls v1 sha={read.sha} def={definition} preenabled=true'"
    )
    with pytest.raises(ValueError, match='preenabled'):
        comment_statement(read, definition=definition, preenabled=None)
    with pytest.raises(ValueError, match='preenabled'):
        comment_statement(insert, definition=definition, preenabled=False)
    with pytest.raises(ValueError, match='preenabled'):
        comment_statement(guard_trigger(SCHEMA), definition=definition, preenabled=True)


def test_an_object_key_names_and_drops_the_object_without_rendering_it() -> None:
    """The key of an object, which the catalog read builds too, yields its target and its drop."""
    policy = table_policies('tags', schema=SCHEMA, role=ROLE)[0]
    key = ObjectKey(RlsObjectKind.POLICY, 'tags_read', f'{SCHEMA}.tags')

    assert policy.key == key
    assert key.target == policy.target == f'POLICY tags_read ON {SCHEMA}.tags'
    assert key.drop_statement == policy.drop_statement == f'DROP POLICY IF EXISTS tags_read ON {SCHEMA}.tags'
    assert guard_function(SCHEMA).key.drop_statement == (
        f'DROP FUNCTION IF EXISTS {SCHEMA}.mcp_context_guard_access_columns()'
    )


@pytest.mark.parametrize(
    ('obj', 'target'),
    [
        pytest.param(helper_functions(SCHEMA)[0], f'FUNCTION {SCHEMA}.mcp_context_principal_id()', id='function'),
        pytest.param(
            guard_trigger(SCHEMA), f'TRIGGER guard_context_entry_access_columns ON {SCHEMA}.context_entries', id='trigger',
        ),
        pytest.param(table_policies('tags', schema=SCHEMA, role=ROLE)[0], f'POLICY tags_read ON {SCHEMA}.tags', id='policy'),
    ],
)
def test_comment_and_drop_statements_name_the_object(obj: RlsObject, target: str) -> None:
    """COMMENT ON attaches the fingerprint, and DROP ... IF EXISTS removes the same object."""
    definition = hashlib.sha256(b'deparse').hexdigest()

    preenabled = True if records_preenabled(obj) else None
    suffix = ' preenabled=true' if preenabled else ''

    assert obj.target == target
    assert comment_statement(obj, definition=definition, preenabled=preenabled) == (
        f"COMMENT ON {target} IS 'mcp-context-server rls v1 sha={obj.sha} def={definition}{suffix}'"
    )
    kind, _, reference = target.partition(' ')
    assert obj.drop_statement == f'DROP {kind} IF EXISTS {reference}'


def test_comment_statement_accepts_only_a_hex_digest() -> None:
    """The deparse digest is interpolated into a literal, so anything but 64 hex characters is refused."""
    obj = guard_trigger(SCHEMA)

    with pytest.raises(ValueError, match='digest'):
        comment_statement(obj, definition="x' OR '1", preenabled=None)


def test_enable_statement_enables_without_force() -> None:
    """The table owner keeps bypassing the policies, so migrations and the command line read every row."""
    statement = enable_statement('tags', schema=SCHEMA)

    assert statement == f'ALTER TABLE {SCHEMA}.tags ENABLE ROW LEVEL SECURITY'
    assert 'FORCE' not in statement
    with pytest.raises(ValueError, match='compression_metadata'):
        enable_statement('compression_metadata', schema=SCHEMA)


def test_disable_statement_disables_a_managed_table_only() -> None:
    """Removal turns row-level security off again only on the tables it manages."""
    assert disable_statement('tags', schema=SCHEMA) == f'ALTER TABLE {SCHEMA}.tags DISABLE ROW LEVEL SECURITY'
    with pytest.raises(ValueError, match='compression_metadata'):
        disable_statement('compression_metadata', schema=SCHEMA)


def test_no_rendered_statement_forces_row_level_security() -> None:
    """Nothing this module renders forces row-level security on the owner."""
    for obj in _all_objects(SCHEMA, ROLE):
        assert re.search(r'\bFORCE\b', obj.statement) is None


def test_module_imports_nothing_that_performs_io() -> None:
    """Rendering is pure: the module imports neither a driver nor a backend."""
    source = inspect.getsource(rls_ddl)

    assert 'import asyncpg' not in source
    assert 'app.backends' not in source
