"""Structural completeness of access scoping, read from the source of ``app/``.

The unit of classification is the method: a function defined directly under a class or a
module of ``app/repositories``, with every closure nested inside it counted as part of it.
The classification and the source parser live in
:mod:`tests.repositories._access_scope_completeness_model`, which the backend-access rules
share. Six rules hold:

1. Detection: every method whose non-docstring string constants carry a statement against
   ``context_entries`` or one of its child tables, or that reads one through the read seam
   ``_readable``, is a key of :data:`REGISTRY`, and every key names a method of the package.
2. Classification: :data:`REGISTRY` assigns each method a :class:`Seam`.
3. Predicate reference: every scoped, write-authorized and owner-keyed method builds the
   access predicate, hands its scope to a predicate fragment helper, or is a dispatcher
   that forwards its scope to every executor it names; every owner-stamped method binds
   ``owner_id`` from its own scope, in its INSERT or through the INSERT builder, and every
   method that inserts a grant binds ``granted_by`` the same way, and neither names the
   system or the maintenance identity.
4. Closure list: the closures of the two ranked searches are classified one by one.
5. Signatures and cases: a guarded method takes a required keyword-only ``scope``; every
   guarded entry, and every child entry that names cases, names two-principal cases that
   exist in the case registry and run on both backends, with a filtered variant when the
   method accepts client filters; each dispatcher names the spy tests that prove it
   forwards the scope on every branch.
6. Call sites and sweeps: every attribute reference to a child reader or writer, and every
   reference to the shared writer of generated outputs, is a pinned site; every reference
   to a worker method sits in the generation worker, the CLI
   or another worker method; no context-table SQL exists outside the repositories, migrations, CLI and
   schemas; only the FTS migration uses the system scope; no tool exposes ``scope`` or
   ``owner_id``; every tool resolves its caller exactly once; no string constant names
   the ``shared`` visibility.

Rule 3 proves that no method lost its predicate wholesale. Whether every per-backend
executor inside a method applies it is behavioral: every registered case runs on both
backends, so an executor that drops the predicate fails its case there.
"""

import ast
import inspect
import re
import textwrap
from collections import Counter

import pytest

import app.tools
from app.access_scope import AccessScope
from app.access_scope import Scope
from tests.repositories._access_scope_cases import KNOWN_CASE_IDS
from tests.repositories._access_scope_completeness_model import CHILD
from tests.repositories._access_scope_completeness_model import CONTEXT_TABLE_SQL
from tests.repositories._access_scope_completeness_model import GUARDED
from tests.repositories._access_scope_completeness_model import OWNER_STAMPED_BUILDERS
from tests.repositories._access_scope_completeness_model import PREDICATE_SEAMS
from tests.repositories._access_scope_completeness_model import REGISTRY
from tests.repositories._access_scope_completeness_model import ROOT
from tests.repositories._access_scope_completeness_model import ParsedModule
from tests.repositories._access_scope_completeness_model import Seam
from tests.repositories._access_scope_completeness_model import app_modules
from tests.repositories._access_scope_completeness_model import callee_name
from tests.repositories._access_scope_completeness_model import direct_closures
from tests.repositories._access_scope_completeness_model import hands_over_scope
from tests.repositories._access_scope_completeness_model import qualified_references
from tests.repositories._access_scope_completeness_model import receiver_name
from tests.repositories._access_scope_completeness_model import registry_keys
from tests.repositories._access_scope_completeness_model import repository_methods
from tests.repositories._access_scope_completeness_model import resolve_target
from tests.repositories._access_scope_completeness_model import runs_context_sql
from tests.repositories._access_scope_completeness_model import scope_parameter
from tests.repositories._access_scope_completeness_model import string_constants
from tests.repositories._access_scope_completeness_model import worker_site_violations
from tests.repositories._access_scope_registry import CASES

# Top-level directories of app/ whose SQL is reviewed by classification or is system-level by
# construction, and the one module outside them that emits SQL: the predicate builder.
SQL_HOMES = frozenset({'repositories', 'migrations', 'cli', 'schemas'})
PREDICATE_MODULE = 'access_scope.py'

# The access-scope functions that turn a scope into a predicate.
PREDICATE_BUILDERS = frozenset({'build_access_predicate', 'build_readable_parent_predicate'})

# Helpers that build the access predicate for the method that hands them its scope.
PREDICATE_HELPERS = frozenset({
    '_candidate_sql', '_interleave_sql', '_dedup_update_sql', '_build_context_filter_clause', '_grant_insert_sql',
})

# The statement builders a grant-inserting method may insert through instead of holding the INSERT itself.
GRANTOR_STAMPED_BUILDERS = frozenset({'_grant_insert_sql'})

RESOLVERS = frozenset({'resolve_access_scope', 'resolve_effective_principal'})
SYSTEM_SCOPE_NAMES = frozenset({'SYSTEM_SCOPE', 'SystemScope'})
MAINTENANCE_NAMES = frozenset({'MAINTENANCE', 'MaintenanceAccess'})

# An INSERT into the context table and the column list it names.
CONTEXT_INSERT = re.compile(r'INSERT INTO context_entries\s*\((?P<columns>[^)]*)\)', re.IGNORECASE)

# An INSERT into the grants table and the column list it names.
GRANT_INSERT = re.compile(r'INSERT INTO context_entry_grants\s*\((?P<columns>[^)]*)\)', re.IGNORECASE)

# Tool-layer helpers that carry the caller's scope to the repositories.
SCOPE_CARRYING_HELPERS = (
    'app.tools.search.legs:semantic_search_raw',
    'app.tools.search.legs:fts_search_raw',
    'app.tools._transactions:execute_store_in_transaction',
    'app.tools._transactions:execute_update_in_transaction',
    'app.tools._transactions:reread_entry_version',
    'app.tools._transactions:_leave_deferred_store_job',
    'app.tools._transactions:_outputs_missing',
    'app.tools.batch._store_generation:generate_store_outputs',
    'app.tools.batch._store_commit:commit_stores_atomically',
    'app.tools.batch._store_commit:commit_stores_individually',
    'app.tools.batch._store_commit:_store_entry',
    'app.tools._delete_cleanup:delete_entries_with_cleanup',
    'app.tools._delete_cleanup:cleanup_embeddings_for_delete',
    'app.tools.sharing:_run_owned_grant_change',
    'app.tools._responses:generation_statuses',
    'app.tools._responses:attach_generation_status',
    'app.tools.search.ranking:apply_search_display_format',
    'app.ids:resolve_prefix',
    'app.ids:resolve_or_normalize_id',
    'app.ids:resolve_or_normalize_ids',
)


# Part 1: detection.


def test_every_context_sql_method_is_classified() -> None:
    """A repository method that runs SQL against a context table fails until it is classified."""
    detected = {key for key, (node, module) in repository_methods().items() if runs_context_sql(node, module)}

    assert sorted(detected - REGISTRY.keys()) == []


def test_every_registry_key_names_a_repository_method() -> None:
    """A renamed or deleted method leaves no stale classification behind."""
    assert sorted(REGISTRY.keys() - repository_methods().keys()) == []


# Part 3: predicate reference.


@pytest.mark.parametrize('key', registry_keys(*PREDICATE_SEAMS))
def test_guarded_method_applies_the_access_predicate(key: str) -> None:
    """A guarded method builds the predicate, hands its scope to a fragment helper, or forwards it."""
    entry = REGISTRY[key]
    node, _module = repository_methods()[key]
    calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)]
    if entry.forwards_to:
        forwarded = [call for call in calls if callee_name(call) in entry.forwards_to]
        assert {callee_name(call) for call in forwarded} == set(entry.forwards_to)
        assert all(hands_over_scope(call) for call in forwarded)
        return
    builds = any(
        isinstance(child, ast.Name | ast.Attribute)
        and (child.id if isinstance(child, ast.Name) else child.attr) in PREDICATE_BUILDERS
        for child in ast.walk(node)
    )
    helper_calls = [call for call in calls if callee_name(call) in PREDICATE_HELPERS]

    assert builds or helper_calls, f'{key} neither builds the access predicate nor calls a fragment helper'
    assert all(hands_over_scope(call) for call in helper_calls)


def _names_identity(node: ast.AST, names: frozenset[str]) -> bool:
    """Whether ``node``'s subtree names one of ``names`` as a name, an attribute or an import."""
    return any(
        (isinstance(child, ast.Name) and child.id in names)
        or (isinstance(child, ast.Attribute) and child.attr in names)
        or (isinstance(child, ast.alias) and child.name in names)
        for child in ast.walk(node)
    )


def _reads_scope_principal(node: ast.AST) -> bool:
    """Whether ``node``'s subtree reads ``scope.principal_id``."""
    return any(
        isinstance(child, ast.Attribute)
        and child.attr == 'principal_id'
        and isinstance(child.value, ast.Name)
        and child.value.id == 'scope'
        for child in ast.walk(node)
    )


def owner_stamped_violations(node: ast.AST, module: ParsedModule) -> list[str]:
    """List how an owner-stamped method fails to bind ``owner_id`` from its own scope.

    The method either holds the INSERT itself, naming ``owner_id`` among its columns and
    reading ``scope.principal_id``, or holds no INSERT and inserts through an owner-stamped
    builder, handing every call its own ``scope``. Either way it never names the system or
    the maintenance identity.

    Args:
        node: The method.
        module: The parsed module that defines it.

    Returns:
        One message per violation, empty for a method that passes.
    """
    violations: list[str] = []
    inserts = [match for _line, value in string_constants(node, module) for match in CONTEXT_INSERT.finditer(value)]
    builder_calls = [
        call for call in ast.walk(node) if isinstance(call, ast.Call) and callee_name(call) in OWNER_STAMPED_BUILDERS
    ]
    if inserts:
        if not all('owner_id' in (column.strip() for column in match['columns'].split(',')) for match in inserts):
            violations.append('an INSERT does not name owner_id')
        if not _reads_scope_principal(node):
            violations.append('the INSERT does not bind owner_id from scope.principal_id')
    elif builder_calls:
        if not all(hands_over_scope(call) for call in builder_calls):
            violations.append('a builder call does not hand over the scope')
    else:
        violations.append('neither holds an INSERT nor calls an owner-stamped builder')
    if _names_identity(node, SYSTEM_SCOPE_NAMES | MAINTENANCE_NAMES):
        violations.append('names the system or the maintenance identity')
    return violations


@pytest.mark.parametrize('key', registry_keys(Seam.OWNER_STAMPED))
def test_owner_stamped_method_binds_the_owner_from_its_scope(key: str) -> None:
    """An owner-stamped insert is owned by the caller's principal and by nothing else."""
    node, module = repository_methods()[key]

    assert owner_stamped_violations(node, module) == []


def _snippet(source: str) -> tuple[ast.AST, ParsedModule]:
    """Parse one function of ``source`` as the owner-stamped rule reads a repository method."""
    tree = ast.parse(textwrap.dedent(source))
    return tree.body[0], ParsedModule('snippet.py', 'snippet', tree, frozenset())


@pytest.mark.parametrize(
    ('source', 'violation'),
    [
        pytest.param(
            '''
            def insert(scope):
                return 'INSERT INTO context_entries (id, owner_id) VALUES (?, ?)', ['new-id', 'local']
            ''',
            'the INSERT does not bind owner_id from scope.principal_id',
            id='literal-owner',
        ),
        pytest.param(
            '''
            def insert(scope):
                return 'INSERT INTO context_entries (id, thread_id) VALUES (?, ?)', [scope.principal_id, 't']
            ''',
            'an INSERT does not name owner_id',
            id='owner-column-missing',
        ),
        pytest.param(
            '''
            async def insert(self, scope):
                return _insert_entry_sql('sqlite', context_id='new-id')
            ''',
            'a builder call does not hand over the scope',
            id='builder-without-scope',
        ),
        pytest.param(
            '''
            async def insert(self, scope):
                sql, values = _insert_entry_sql('sqlite', scope=scope, context_id='new-id')
                return await self.backend.execute_write(sql, access_scope=MAINTENANCE)
            ''',
            'names the system or the maintenance identity',
            id='maintenance-identity',
        ),
        pytest.param(
            '''
            def insert(scope):
                return 'SELECT 1', []
            ''',
            'neither holds an INSERT nor calls an owner-stamped builder',
            id='no-insert',
        ),
    ],
)
def test_owner_stamped_rule_rejects_an_owner_not_bound_from_the_scope(source: str, violation: str) -> None:
    """Each way of stamping an owner other than the caller's own scope is reported."""
    node, module = _snippet(source)

    assert owner_stamped_violations(node, module) == [violation]


def grantor_stamped_violations(node: ast.AST, module: ParsedModule) -> list[str]:
    """List how a grant-inserting method fails to bind ``granted_by`` from its own scope.

    The method either holds the INSERT itself, naming ``granted_by`` among its columns and
    reading ``scope.principal_id``, or holds no INSERT and inserts through a grantor-stamped
    builder, handing every call its own ``scope``. Either way it never names the system or
    the maintenance identity.

    Args:
        node: The method.
        module: The parsed module that defines it.

    Returns:
        One message per violation, empty for a method that passes.
    """
    violations: list[str] = []
    inserts = [match for _line, value in string_constants(node, module) for match in GRANT_INSERT.finditer(value)]
    builder_calls = [
        call for call in ast.walk(node) if isinstance(call, ast.Call) and callee_name(call) in GRANTOR_STAMPED_BUILDERS
    ]
    if inserts:
        if not all('granted_by' in (column.strip() for column in match['columns'].split(',')) for match in inserts):
            violations.append('an INSERT does not name granted_by')
        if not _reads_scope_principal(node):
            violations.append('the INSERT does not bind granted_by from scope.principal_id')
    elif builder_calls:
        if not all(hands_over_scope(call) for call in builder_calls):
            violations.append('a builder call does not hand over the scope')
    else:
        violations.append('neither holds an INSERT nor calls a grantor-stamped builder')
    if _names_identity(node, SYSTEM_SCOPE_NAMES | MAINTENANCE_NAMES):
        violations.append('names the system or the maintenance identity')
    return violations


def _grant_inserting_methods() -> list[str]:
    """The repository methods that hold an INSERT into the grants table or call a grantor-stamped builder, sorted."""
    return sorted(
        key for key, (node, module) in repository_methods().items()
        if any(GRANT_INSERT.search(value) for _line, value in string_constants(node, module))
        or any(isinstance(call, ast.Call) and callee_name(call) in GRANTOR_STAMPED_BUILDERS for call in ast.walk(node))
    )


def test_the_grant_builder_is_a_grant_inserting_method() -> None:
    """The rule below reads the builder itself, so it never passes over an empty set of methods."""
    assert 'app.repositories.grant_repository:GrantRepository._grant_insert_sql' in _grant_inserting_methods()


@pytest.mark.parametrize('key', _grant_inserting_methods())
def test_grant_insert_binds_the_grantor_from_its_scope(key: str) -> None:
    """A grant is recorded as granted by the caller's principal and by nothing else."""
    node, module = repository_methods()[key]

    assert grantor_stamped_violations(node, module) == []


@pytest.mark.parametrize(
    ('source', 'violation'),
    [
        pytest.param(
            '''
            def insert(scope):
                return 'INSERT INTO context_entry_grants (context_entry_id, granted_by) VALUES (?, ?)', ['id', 'local']
            ''',
            'the INSERT does not bind granted_by from scope.principal_id',
            id='literal-grantor',
        ),
        pytest.param(
            '''
            def insert(scope):
                return 'INSERT INTO context_entry_grants (context_entry_id, principal_id) VALUES (?, ?)', [scope.principal_id]
            ''',
            'an INSERT does not name granted_by',
            id='grantor-column-missing',
        ),
        pytest.param(
            '''
            async def insert(self, scope):
                return self._grant_insert_sql('sqlite', 'id', 'group', 'team-x', 'read')
            ''',
            'a builder call does not hand over the scope',
            id='builder-without-scope',
        ),
        pytest.param(
            '''
            async def insert(self, scope):
                sql, values = self._grant_insert_sql('sqlite', 'id', 'group', 'team-x', 'read', scope=scope)
                return await self.backend.execute_write(sql, access_scope=MAINTENANCE)
            ''',
            'names the system or the maintenance identity',
            id='maintenance-identity',
        ),
        pytest.param(
            '''
            def insert(scope):
                return 'SELECT 1', []
            ''',
            'neither holds an INSERT nor calls a grantor-stamped builder',
            id='no-insert',
        ),
    ],
)
def test_grantor_stamped_rule_rejects_a_grantor_not_bound_from_the_scope(source: str, violation: str) -> None:
    """Each way of recording a grantor other than the caller's own scope is reported."""
    node, module = _snippet(source)

    assert grantor_stamped_violations(node, module) == [violation]


# Part 4: closure list.


@pytest.mark.parametrize('key', sorted(key for key, entry in REGISTRY.items() if entry.closures))
def test_ranked_search_closures_are_classified(key: str) -> None:
    """Every closure of a ranked search is listed, and only the compute closures run no SQL."""
    entry = REGISTRY[key]
    node, module = repository_methods()[key]
    closures = direct_closures(node)

    assert sorted(closures) == sorted(entry.closures)
    for name, closure in closures.items():
        assert runs_context_sql(closure, module) is (entry.closures[name] is not Seam.COMPUTE), name


# Part 5: signatures, dispatchers and cases.


@pytest.mark.parametrize('key', registry_keys(*GUARDED))
def test_guarded_method_requires_a_scope(key: str) -> None:
    """A guarded method takes a keyword-only ``scope`` with no default, typed per its class."""
    parameter, annotation = scope_parameter(key)
    expected = AccessScope if REGISTRY[key].seam in {Seam.OWNER_KEYED, Seam.OWNER_STAMPED} else Scope

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert annotation is expected


@pytest.mark.parametrize('target', SCOPE_CARRYING_HELPERS)
def test_scope_carrying_helper_requires_a_scope(target: str) -> None:
    """A tool-layer helper that reaches a repository takes a required keyword-only scope."""
    parameter, annotation = scope_parameter(target)

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert annotation in {Scope, AccessScope}


@pytest.mark.parametrize('key', sorted(key for key, entry in REGISTRY.items() if entry.forwards_to))
def test_dispatcher_names_its_spy_tests(key: str) -> None:
    """Each dispatcher names the tests that spy on its branches, and they exist."""
    spies = REGISTRY[key].spies
    assert spies
    for spy in spies:
        path, name = spy.split('::')
        tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
        assert name in {node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}, spy


# The methods two-principal cases prove: every guarded method, and each child method that names cases.
CASE_PROVEN_KEYS = sorted({*registry_keys(*GUARDED), *(key for key, entry in REGISTRY.items() if entry.cases)})


@pytest.mark.parametrize('key', CASE_PROVEN_KEYS)
def test_guarded_method_names_registered_cases(key: str) -> None:
    """A case-proven method names cases that run on both backends, filtered when it accepts filters."""
    names = REGISTRY[key].cases
    accepts_filters = 'metadata_filters' in inspect.signature(resolve_target(key)[1]).parameters

    assert names
    assert set(names) <= KNOWN_CASE_IDS
    for name in names:
        assert any(case.case_id == name and case.backend is None for case in CASES), f'{name} has no case on both backends'
    if accepts_filters:
        assert any(case.case_id in names and case.filtered and case.backend is None for case in CASES)


def test_every_registered_case_proves_a_classified_method() -> None:
    """Every case id in the case registry is claimed by a guarded method."""
    claimed = {name for entry in REGISTRY.values() for name in entry.cases}

    assert {case.case_id for case in CASES} == claimed


# Part 6: call sites and sweeps.


@pytest.mark.parametrize('key', registry_keys(*CHILD))
def test_child_reader_and_writer_sites_are_pinned(key: str) -> None:
    """A new reference to a child reader or writer fails until its id source is reviewed."""
    entry = REGISTRY[key]
    name = key.rsplit('.', 1)[1]
    sites = Counter(
        f'{module.path}:{function}'
        for module in app_modules()
        for function, attribute in qualified_references(module)
        if attribute.attr == name and (entry.receiver is None or receiver_name(attribute.value) == entry.receiver)
    )

    assert sites == Counter(entry.sites)


# The shared writer of an entry's generated outputs runs the chunk and node writers for the id it is
# handed: the update path calls it for the entry its write gate admitted, and the generation worker for
# the entry of the job it claimed.
GENERATION_OUTPUT_WRITER = 'write_generation_outputs'
GENERATION_OUTPUT_WRITER_SITES = (
    'tools/_transactions.py:execute_update_in_transaction',
    'generation_jobs/executor.py:_complete',
)


def _name_references(module: ParsedModule, name: str) -> list[str]:
    """Return the qualified enclosing function of every read of ``name``, bare or as an attribute, in ``module``."""
    found: list[str] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                visit(child, (*scope, child.name))
                continue
            bare = isinstance(child, ast.Name) and child.id == name
            attribute = isinstance(child, ast.Attribute) and child.attr == name
            if (bare or attribute) and isinstance(getattr(child, 'ctx', None), ast.Load):
                found.append('.'.join(scope))
            visit(child, scope)

    visit(module.tree, ())
    return found


def test_generation_output_writer_sites_are_pinned() -> None:
    """A new caller of the shared output writer fails until the id it hands over is reviewed."""
    sites = Counter(
        f'{module.path}:{function}'
        for module in app_modules()
        for function in _name_references(module, GENERATION_OUTPUT_WRITER)
    )

    assert sites == Counter(GENERATION_OUTPUT_WRITER_SITES)


def test_output_writer_references_count_bare_and_attribute_reads() -> None:
    """Both a bare call and a call through a module attribute count as a site; the import does not."""
    module = _module(
        'tools/context/store.py',
        '''
        from app.tools._transactions import write_generation_outputs
        import app.tools._transactions as transactions

        async def store(repos, txn):
            await write_generation_outputs(repos, txn, 'id', None, None, embedding_model='m', scope=None)
            await transactions.write_generation_outputs(repos, txn, 'id', None, None, embedding_model='m', scope=None)
        ''',
    )

    assert _name_references(module, GENERATION_OUTPUT_WRITER) == ['store', 'store']


def test_worker_methods_are_referenced_only_outside_requests() -> None:
    """A worker method is referenced only by the generation worker, the CLI and other worker methods."""
    assert worker_site_violations(app_modules()) == []


def _module(path: str, source: str) -> ParsedModule:
    """Parse ``source`` as the module of ``app/`` at ``path``, as the worker site rule reads it."""
    tree = ast.parse(textwrap.dedent(source))
    return ParsedModule(path, 'app.' + path.removesuffix('.py').replace('/', '.'), tree, frozenset())


_CLAIMING_FUNCTION = '''
async def drain(repos, scope):
    return await repos.generation_jobs.claim_next(max_attempts=3, lease_s=120, scope=scope)
'''


@pytest.mark.parametrize(
    ('path', 'violations'),
    [
        pytest.param('tools/context/store.py', ['tools/context/store.py:drain'], id='tool'),
        pytest.param('startup/generation_worker.py', ['startup/generation_worker.py:drain'], id='startup'),
        pytest.param('generation_jobs/worker.py', [], id='worker'),
        pytest.param('cli/migrate_embeddings.py', [], id='cli'),
    ],
)
def test_worker_site_rule_admits_only_the_worker_and_the_cli(path: str, violations: list[str]) -> None:
    """A worker method referenced from a tool or any other request-side module is reported."""
    assert worker_site_violations([_module(path, _CLAIMING_FUNCTION)]) == violations


def test_worker_site_rule_admits_only_worker_methods_inside_the_repositories() -> None:
    """Inside the repositories one worker method may reference another; any other method is reported."""
    module = _module(
        'repositories/generation_job_repository.py',
        '''
        class GenerationJobRepository:
            async def delete_job(self, context_id, claim_token, *, scope):
                return await self.delete_claimed(context_id, claim_token, txn=None)

            async def states_for_ids(self, context_ids, *, scope):
                return await self.reset_failed(scope=scope)
        ''',
    )

    assert worker_site_violations([module]) == [
        'repositories/generation_job_repository.py:GenerationJobRepository.states_for_ids',
    ]


def test_worker_site_rule_counts_record_failure_only_through_the_job_repository() -> None:
    """A circuit breaker's own ``record_failure`` is no worker call; the job repository's is."""
    module = _module(
        'tools/context/store.py',
        '''
        async def store(self, repos, scope):
            self.circuit_breaker.record_failure()
            await repos.generation_jobs.record_failure('id', 'token', 'error', max_attempts=3, scope=scope)
        ''',
    )

    assert worker_site_violations([module]) == ['tools/context/store.py:store']


def test_no_context_sql_outside_the_repositories() -> None:
    """Context-table SQL lives only in the repositories, migrations, CLI, schemas and the predicate builder."""
    stray = [
        f'{module.path}:{line}'
        for module in app_modules()
        if module.path.split('/')[0] not in SQL_HOMES and module.path != PREDICATE_MODULE
        for line, value in string_constants(module.tree, module)
        if CONTEXT_TABLE_SQL.search(value)
    ]

    assert stray == []


def test_system_scope_is_used_only_by_the_fts_migration() -> None:
    """No request path can reach the system scope: its one consumer is the FTS migration."""
    users = {
        module.path
        for module in app_modules()
        if module.path != PREDICATE_MODULE
        and any(
            (isinstance(node, ast.Name) and node.id in SYSTEM_SCOPE_NAMES)
            or (isinstance(node, ast.Attribute) and node.attr in SYSTEM_SCOPE_NAMES)
            or (isinstance(node, ast.alias) and node.name in SYSTEM_SCOPE_NAMES)
            for node in ast.walk(module.tree)
        )
    }

    assert users == {'migrations/fts.py'}


@pytest.mark.parametrize('name', sorted(app.tools.TOOL_ANNOTATIONS))
def test_tool_signature_hides_scope_and_owner(name: str) -> None:
    """The server resolves the caller; no tool lets a client pass a scope or an owner."""
    parameters = inspect.signature(getattr(app.tools, name)).parameters

    assert 'scope' not in parameters
    assert 'owner_id' not in parameters


@pytest.mark.parametrize('name', sorted(app.tools.TOOL_ANNOTATIONS))
def test_tool_resolves_its_caller_once(name: str) -> None:
    """Every tool resolves the calling principal exactly once."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(getattr(app.tools, name)))))
    resolutions = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and callee_name(node) in RESOLVERS]

    assert len(resolutions) == 1


def test_no_string_constant_names_the_shared_visibility() -> None:
    """The visibility vocabulary is exactly ``private`` and ``public``."""
    hits = [
        f'{module.path}:{line}'
        for module in app_modules()
        for line, value in string_constants(module.tree, module)
        if value == 'shared'
    ]

    assert hits == []
