"""Structural completeness of read routing, read from the source of ``app/``.

A plain read names every routable table it selects from through the repository seam
``BaseRepository._readable``, which asks the backend for the relation to read. Four
rules hold:

1. No guarded method and no id-keyed child reader names a routable table after ``FROM``
   or ``JOIN`` in a string constant. A delete names its target table, and the write gate
   ``entry_exists`` keeps the base name because it locks the row it reads.
2. Every ``_readable`` call passes string literals for a routable table and its alias,
   so the relation a statement reads is visible in its source.
3. Child writers, worker methods and system methods never call ``_readable``: they write,
   or run outside any request, and name their tables directly.
4. In the methods rule 1 covers, every relation an f-string interpolates after ``FROM``
   or ``JOIN`` is a name bound to a ``self._readable(...)`` call; to an f-string that
   starts with such a relation and reads only such relations after its own ``FROM`` and
   ``JOIN`` keywords; or to a parameter of a module-level helper that every call site
   fills with such a name.

Rule 1 rejects a routable table written into a statement, and rule 4 rejects any other
expression in a relation's place, such as a name bound to a string constant: together
they prove that each plain read a routed method builds from string literals and
f-strings names its relations through the seam.
"""

import ast
import functools
import re
from collections.abc import Iterator

import pytest

from app.backends.base import ROUTABLE_TABLES
from tests.repositories._access_scope_completeness_model import GUARDED
from tests.repositories._access_scope_completeness_model import READ_SEAM
from tests.repositories._access_scope_completeness_model import REGISTRY
from tests.repositories._access_scope_completeness_model import FunctionNode
from tests.repositories._access_scope_completeness_model import ParsedModule
from tests.repositories._access_scope_completeness_model import Seam
from tests.repositories._access_scope_completeness_model import app_modules
from tests.repositories._access_scope_completeness_model import callee_name
from tests.repositories._access_scope_completeness_model import direct_closures
from tests.repositories._access_scope_completeness_model import registry_keys
from tests.repositories._access_scope_completeness_model import repository_methods

# A FROM or JOIN item over a routable table; the optional DELETE marks a delete target.
ROUTABLE_ITEM = re.compile(
    rf'\b(DELETE\s+)?(?:FROM|JOIN)\s+({"|".join(sorted(ROUTABLE_TABLES))})\b',
    re.IGNORECASE,
)

# An f-string part after which the next interpolation is the relation a FROM or JOIN reads.
RELATION_FOLLOWS = re.compile(r'\b(?:FROM|JOIN)\s*$', re.IGNORECASE)

# The write gate reads under a row lock on PostgreSQL, so it keeps the base name.
LOCKING_READ = 'app.repositories.context_repository.reads:ContextReadMixin.entry_exists'

ROUTED = frozenset({*GUARDED, Seam.CHILD_BY_TRUSTED_ID})
UNROUTED = frozenset({Seam.CHILD_WRITER, Seam.WORKER, Seam.SYSTEM})
ROUTED_KEYS = [key for key in registry_keys(*ROUTED) if key != LOCKING_READ]

ALIAS = re.compile(r'[a-z_][a-z0-9_]*')

# Nodes that open a scope of their own, so the names they bind are not the enclosing function's.
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _excluded(key: str) -> frozenset[int]:
    """Return the node ids of the classified closures of ``key`` whose class does not route."""
    closures = direct_closures(repository_methods()[key][0])
    return frozenset(
        id(child)
        for name, seam in REGISTRY[key].closures.items() if seam not in ROUTED
        for child in ast.walk(closures[name])
    )


def _routed_constants(key: str) -> Iterator[tuple[int, str]]:
    """Yield the line and value of every string constant of a method whose plain reads must route.

    The whole method counts, except the classified closures of a ranked search whose class
    does not route.

    Yields:
        The line and value of each non-docstring string constant.
    """
    node, module = repository_methods()[key]
    excluded = _excluded(key)
    for line, value, constant in _constants(node, module):
        if id(constant) not in excluded:
            yield line, value


def _constants(node: ast.AST, module: ParsedModule) -> Iterator[tuple[int, str, ast.Constant]]:
    """Yield the line, value and node of every non-docstring string constant of ``node``'s subtree."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and id(child) not in module.docstrings:
            yield child.lineno, child.value, child


def _seam_calls(node: ast.AST) -> list[ast.Call]:
    """Return every call of the read-routing seam in ``node``'s subtree."""
    return [child for child in ast.walk(node) if isinstance(child, ast.Call) and callee_name(child) == READ_SEAM]


@pytest.mark.parametrize('key', ROUTED_KEYS)
def test_routed_method_names_no_routable_table(key: str) -> None:
    """A plain read of a guarded method or child reader names its relations through the seam."""
    hits = [
        f'{line}: {match.group(0)!r}'
        for line, value in _routed_constants(key)
        for match in ROUTABLE_ITEM.finditer(value)
        if match.group(1) is None
    ]

    assert hits == []


@functools.cache
def _parents() -> dict[int, ast.AST]:
    """Map the id of every node of ``app/`` to its parent node."""
    return {
        id(child): node
        for module in app_modules()
        for node in ast.walk(module.tree)
        for child in ast.iter_child_nodes(node)
    }


def _enclosing_functions(node: ast.AST) -> Iterator[FunctionNode]:
    """Yield the functions enclosing ``node``, innermost first."""
    parent = _parents().get(id(node))
    while parent is not None:
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
            yield parent
        parent = _parents().get(id(parent))


def _own_scope(function: FunctionNode) -> Iterator[ast.AST]:
    """Yield the nodes of ``function``'s body that belong to its own scope."""
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, SCOPES):
            pending.extend(ast.iter_child_nodes(node))


def _relation_items(fstring: ast.JoinedStr) -> list[ast.expr]:
    """Return what ``fstring`` interpolates right after each ``FROM`` or ``JOIN``.

    A ``FROM`` or ``JOIN`` that ends the f-string, or that a literal follows, contributes
    the part after it (or its own part), which never resolves to a relation.

    Returns:
        The expression, or the literal part, standing in each relation's place.
    """
    items: list[ast.expr] = []
    parts = fstring.values
    for index, part in enumerate(parts):
        if isinstance(part, ast.Constant) and isinstance(part.value, str) and RELATION_FOLLOWS.search(part.value):
            follower = parts[index + 1] if index + 1 < len(parts) else part
            items.append(follower.value if isinstance(follower, ast.FormattedValue) else follower)
    return items


def _is_relation(expression: ast.expr, seen: frozenset[int]) -> bool:
    """Whether ``expression``, where it stands in ``app/``, is a relation the seam produced.

    The name resolves in the innermost enclosing function that binds it, as Python
    resolves it: there it is a parameter, or every binding of it assigns a relation.

    Returns:
        True for such a name, False for any other expression.
    """
    if not isinstance(expression, ast.Name):
        return False
    for function in _enclosing_functions(expression):
        arguments = function.args
        parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
        if expression.id in {parameter.arg for parameter in parameters if parameter is not None}:
            return _is_filled_parameter(function, expression.id, seen)
        bindings = [
            node for node in _own_scope(function)
            if isinstance(node, ast.Name) and node.id == expression.id and not isinstance(node.ctx, ast.Load)
        ]
        if bindings:
            return all(_binds_relation(binding, seen) for binding in bindings)
    return False


def _binds_relation(target: ast.Name, seen: frozenset[int]) -> bool:
    """Whether the binding ``target`` assigns a relation the seam produced.

    Returns:
        True when ``target`` is the only target of an assignment whose value is a
        ``self._readable(...)`` call, or an f-string that starts with a relation and reads
        only relations after its own ``FROM`` and ``JOIN`` keywords.
    """
    statement = _parents()[id(target)]
    if id(statement) in seen or not isinstance(statement, ast.Assign | ast.AnnAssign):
        return False
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    value = statement.value
    if targets != [target] or value is None:
        return False
    if isinstance(value, ast.Call):
        method = value.func
        return (
            isinstance(method, ast.Attribute) and method.attr == READ_SEAM
            and isinstance(method.value, ast.Name) and method.value.id == 'self'
        )
    if not isinstance(value, ast.JoinedStr) or not isinstance(value.values[0], ast.FormattedValue):
        return False
    deeper = seen | {id(statement)}
    return all(_is_relation(item, deeper) for item in [value.values[0].value, *_relation_items(value)])


def _passed_argument(call: ast.Call, function: FunctionNode, name: str) -> ast.expr | None:
    """Return what ``call`` passes for the parameter ``name`` of ``function``, by keyword or by position."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    positional = [parameter.arg for parameter in (*function.args.posonlyargs, *function.args.args)]
    if name in positional and positional.index(name) < len(call.args):
        return call.args[positional.index(name)]
    return None


def _is_filled_parameter(function: FunctionNode, name: str, seen: frozenset[int]) -> bool:
    """Whether every call site of the module-level ``function`` passes a relation as ``name``.

    Returns:
        True when ``function`` is defined at module level, is called somewhere under
        ``app/``, and each call passes a relation the seam produced for ``name``.
    """
    if id(function) in seen or not isinstance(_parents()[id(function)], ast.Module):
        return False
    calls = [
        node
        for module in app_modules()
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function.name
    ]
    deeper = seen | {id(function)}
    arguments = [_passed_argument(call, function, name) for call in calls]
    return bool(calls) and all(argument is not None and _is_relation(argument, deeper) for argument in arguments)


@pytest.mark.parametrize('key', ROUTED_KEYS)
def test_routed_method_reads_only_seam_relations(key: str) -> None:
    """Each relation a routed method interpolates after FROM or JOIN comes from the seam."""
    node = repository_methods()[key][0]
    excluded = _excluded(key)
    unresolved = [
        f'{item.lineno}: {ast.unparse(item)!r}'
        for fstring in ast.walk(node)
        if isinstance(fstring, ast.JoinedStr) and id(fstring) not in excluded
        for item in _relation_items(fstring)
        if not _is_relation(item, frozenset())
    ]

    assert unresolved == []


@functools.cache
def _seam_calls_by_location() -> dict[str, ast.Call]:
    """Map ``<path under app>:<line>:<column>`` to every call of the seam under ``app/``."""
    return {
        f'{module.path}:{call.lineno}:{call.col_offset}': call
        for module in app_modules()
        for call in _seam_calls(module.tree)
    }


def test_the_seam_is_called() -> None:
    """The literal-argument rule inspects real calls, so the seam has callers."""
    assert _seam_calls_by_location()


@pytest.mark.parametrize('location', sorted(_seam_calls_by_location()))
def test_seam_call_passes_literal_table_and_alias(location: str) -> None:
    """Each call names a routable table and an alias as string literals and nothing else."""
    call = _seam_calls_by_location()[location]
    assert len(call.args) == 1
    assert [keyword.arg for keyword in call.keywords] == ['alias']
    table = call.args[0]
    alias = call.keywords[0].value
    assert isinstance(table, ast.Constant)
    assert table.value in ROUTABLE_TABLES
    assert isinstance(alias, ast.Constant)
    assert isinstance(alias.value, str)
    assert ALIAS.fullmatch(alias.value) is not None


@pytest.mark.parametrize('key', registry_keys(*UNROUTED))
def test_child_writers_and_system_methods_never_route(key: str) -> None:
    """A writer, a worker method or a system method names its tables directly, never through the seam."""
    node: FunctionNode = repository_methods()[key][0]

    assert _seam_calls(node) == []
