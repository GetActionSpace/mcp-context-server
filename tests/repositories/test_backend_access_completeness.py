"""Structural completeness of the backend access seam, read from the source of ``app/``.

Every entry point of a storage backend (``execute_read``, ``execute_write`` and
``begin_transaction``) takes the identity it runs as through a required keyword-only
``access_scope``, and ``readable_relation`` names the relation a plain read selects from.
Eight rules hold:

1. Every call of an entry point under ``app/`` passes ``access_scope``.
2. A repository method that calls an entry point takes a required keyword-only ``scope``
   and hands that same ``scope`` to every such call.
3. Such a method is typed by its class: a guarded method keeps the scope type its
   predicate needs, and a child, worker or system method takes the wider ``BackendAccess``.
4. No repository method carries a ``scope`` it neither hands to a call nor reads, and no
   guarded ``scope`` admits the maintenance identity.
5. The maintenance identity never appears under ``app/tools`` or ``app/repositories``:
   only migrations, startup, the CLI and the compression provenance run as it.
6. Nothing outside ``app/backends`` calls ``get_connection``: a raw connection runs
   outside any transaction the backend opens, so it cannot carry an identity.
7. Outside ``app/backends`` only the repository base class asks a backend for a readable
   relation, so every repository names its relations through one seam.
8. The routable tables are ``context_entries`` and every table that references it except
   the grants table, which only the access predicate reads, by its own name.
"""

import ast
import inspect
import re
import types
import typing

import pytest

from app.access_scope import AccessScope
from app.access_scope import SystemScope
from app.backends.base import ROUTABLE_TABLES
from app.backends.base import BackendAccess
from app.backends.base import MaintenanceAccess
from tests.repositories._access_scope_completeness_model import APP
from tests.repositories._access_scope_completeness_model import GUARDED
from tests.repositories._access_scope_completeness_model import FunctionNode
from tests.repositories._access_scope_completeness_model import app_modules
from tests.repositories._access_scope_completeness_model import registry_keys
from tests.repositories._access_scope_completeness_model import repository_methods
from tests.repositories._access_scope_completeness_model import scope_parameter

# The backend methods that bind the caller's identity before running anything.
BACKEND_ENTRY_POINTS = frozenset({'execute_read', 'execute_write', 'begin_transaction'})

# The one module outside app/backends that asks a backend for a readable relation.
RELATION_SEAM_MODULE = 'repositories/base.py'

# The directories whose DDL creates the tables, and the table the access predicate reads by name.
DDL_HOMES = ('schemas', 'migrations')
GRANTS_TABLE = 'context_entry_grants'
CREATE_TABLE = re.compile(r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*(?:\(|USING\b)')

# The names of the maintenance identity and the directories it must never reach.
MAINTENANCE_NAMES = frozenset({'MAINTENANCE', 'MaintenanceAccess'})
REQUEST_PATH_DIRECTORIES = frozenset({'tools', 'repositories'})


def _entry_point_calls(node: ast.AST) -> list[ast.Call]:
    """Return every call of a backend entry point in ``node``'s subtree."""
    return [
        child for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        and child.func.attr in BACKEND_ENTRY_POINTS
    ]


def _access_scope_argument(call: ast.Call) -> ast.expr | None:
    """Return the value a call passes as ``access_scope``, or None when it passes none."""
    return next((keyword.value for keyword in call.keywords if keyword.arg == 'access_scope'), None)


def _backend_callers() -> list[str]:
    """The repository methods whose subtree calls a backend entry point, sorted."""
    return sorted(key for key, (node, _module) in repository_methods().items() if _entry_point_calls(node))


def _scope_carriers() -> list[str]:
    """The repository methods that declare a ``scope`` parameter, sorted."""
    return sorted(
        key for key, (node, _module) in repository_methods().items()
        if any(argument.arg == 'scope' for argument in (*node.args.args, *node.args.kwonlyargs))
    )


def _consumes_scope(node: FunctionNode) -> bool:
    """Whether ``node``'s subtree hands the bare name ``scope`` to a call or reads an attribute of it.

    Binding it to a throwaway name does not count: that discards the identity.

    Returns:
        True when the scope reaches a call or an attribute read.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            arguments = (*child.args, *(keyword.value for keyword in child.keywords))
            if any(isinstance(argument, ast.Name) and argument.id == 'scope' for argument in arguments):
                return True
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == 'scope':
            return True
    return False


def _union_members(annotation: object) -> set[object]:
    """Expand type aliases and unions into the set of types an annotation admits."""
    if isinstance(annotation, typing.TypeAliasType):
        return _union_members(annotation.__value__)
    if typing.get_origin(annotation) in {types.UnionType, typing.Union}:
        return set().union(*(_union_members(member) for member in typing.get_args(annotation)))
    return {annotation}


def test_backend_access_admits_every_identity_and_nothing_else() -> None:
    """``BackendAccess`` is exactly the request scopes plus the maintenance identity."""
    assert _union_members(BackendAccess) == {AccessScope, SystemScope, MaintenanceAccess}


def test_every_backend_entry_point_call_passes_the_access_scope() -> None:
    """A read or write that reaches the backend without naming its identity fails until it does."""
    missing = [
        f'{module.path}:{call.lineno}'
        for module in app_modules()
        for call in _entry_point_calls(module.tree)
        if _access_scope_argument(call) is None
    ]

    assert missing == []


@pytest.mark.parametrize('key', _backend_callers())
def test_backend_calling_method_forwards_its_scope(key: str) -> None:
    """A repository method that reaches the backend requires a scope and hands it to the backend."""
    parameter, _annotation = scope_parameter(key)
    node, _module = repository_methods()[key]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    for call in _entry_point_calls(node):
        argument = _access_scope_argument(call)
        assert isinstance(argument, ast.Name), f'{key}:{call.lineno}'
        assert argument.id == 'scope', f'{key}:{call.lineno}'


@pytest.mark.parametrize('key', [key for key in _backend_callers() if key not in registry_keys(*GUARDED)])
def test_child_and_system_backend_callers_take_the_backend_access(key: str) -> None:
    """A child, worker or system method renders no predicate, so it accepts every identity the backend binds."""
    _parameter, annotation = scope_parameter(key)

    assert annotation is BackendAccess


@pytest.mark.parametrize('key', _scope_carriers())
def test_repository_scope_is_consumed(key: str) -> None:
    """A repository method never carries a scope it neither hands on nor reads."""
    node, _module = repository_methods()[key]

    assert _consumes_scope(node), key


@pytest.mark.parametrize('key', registry_keys(*GUARDED))
def test_guarded_scope_never_admits_the_maintenance_identity(key: str) -> None:
    """A method that renders the access predicate cannot be handed the maintenance identity."""
    _parameter, annotation = scope_parameter(key)

    assert MaintenanceAccess not in _union_members(annotation)


def test_maintenance_identity_stays_off_the_request_paths() -> None:
    """The tools and repositories run as the caller; only non-request paths name the maintenance identity."""
    users = sorted(
        f'{module.path}:{getattr(node, "lineno", 0)}'
        for module in app_modules()
        if module.path.split('/')[0] in REQUEST_PATH_DIRECTORIES
        for node in ast.walk(module.tree)
        if (isinstance(node, ast.Name) and node.id in MAINTENANCE_NAMES)
        or (isinstance(node, ast.Attribute) and node.attr in MAINTENANCE_NAMES)
        or (isinstance(node, ast.alias) and node.name in MAINTENANCE_NAMES)
    )

    assert users == []


def _attribute_calls(name: str) -> list[tuple[str, int]]:
    """Return the module path and line of every ``x.<name>(...)`` call under ``app/``, sorted."""
    return sorted(
        (module.path, node.lineno)
        for module in app_modules()
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name
    )


def test_no_raw_connection_outside_the_backends() -> None:
    """A raw connection carries no identity, so only the backends themselves open one."""
    stray = [f'{path}:{line}' for path, line in _attribute_calls('get_connection') if not path.startswith('backends/')]

    assert stray == []


def test_readable_relation_is_asked_only_through_the_repository_base() -> None:
    """Every repository names its relations through the base class, never by asking the backend itself."""
    stray = [
        f'{path}:{line}'
        for path, line in _attribute_calls('readable_relation')
        if not path.startswith('backends/') and path != RELATION_SEAM_MODULE
    ]

    assert stray == []


def _tables_referencing_the_context_table() -> set[str]:
    """Return every table whose DDL under the schema and migration directories references ``context_entries``."""
    tables: set[str] = set()
    for home in DDL_HOMES:
        for file in sorted((APP / home).rglob('*')):
            if file.suffix not in {'.sql', '.py'}:
                continue
            text = file.read_text(encoding='utf-8')
            matches = list(CREATE_TABLE.finditer(text))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                if 'REFERENCES context_entries(id)' in text[match.end():end]:
                    tables.add(match.group(1))
    return tables


def test_routable_tables_are_the_context_table_and_its_children() -> None:
    """A new child table fails until it is routable, so a profile can map it to a readable relation."""
    children = _tables_referencing_the_context_table()
    expected = {'context_entries'} | (children - {GRANTS_TABLE})

    assert GRANTS_TABLE in children
    assert expected == ROUTABLE_TABLES
