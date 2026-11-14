"""Plans PostgreSQL chooses for a statement a repository read sent, shared by the plan-shape tests.

``explain_custom`` plans a captured statement with its arguments bound, which the planner
folds in as constants. ``explain_generic`` plans it as a prepared statement held to its
generic plan, which no argument value shapes and which a prepared statement may settle on
after its first five executions. Both return the parsed ``EXPLAIN (VERBOSE, FORMAT JSON)``
document and take the connection first, so a backend read can run them as its operation,
and ``plan_nodes`` lists the nodes of that document or of any part of it.
"""

import json
from typing import cast

from tests.helpers import AsyncpgConnection
from tests.helpers import Captured

_PREPARED_NAME = 'plan_shape'


def plan_nodes(node: object) -> list[dict[str, object]]:
    """Return every plan node of a parsed ``EXPLAIN (FORMAT JSON)`` document or of any part of one.

    Returns:
        The nodes, InitPlans and SubPlans included, each before the nodes beneath it.
    """
    found: list[dict[str, object]] = []
    if isinstance(node, dict):
        plan = cast('dict[str, object]', node)
        if 'Node Type' in plan:
            found.append(plan)
        for child in plan.values():
            found.extend(plan_nodes(child))
    elif isinstance(node, list):
        for item in cast('list[object]', node):
            found.extend(plan_nodes(item))
    return found


def sql_literal(value: object) -> str:
    """Render a bound argument of a captured statement as a SQL literal for ``EXECUTE``.

    Returns:
        The literal: an integer as written, a string quoted, a list of strings as a ``text[]`` array.

    Raises:
        TypeError: When the value is a boolean or of a type the repositories do not bind.
    """
    if isinstance(value, bool):
        raise TypeError(f'no boolean argument is expected: {value!r}')
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, list):
        elements = ', '.join(sql_literal(element) for element in cast('list[object]', value))
        return f'ARRAY[{elements}]::text[]'
    raise TypeError(f'unexpected argument type {type(value).__name__}: {value!r}')


async def explain_custom(conn: AsyncpgConnection, statement: Captured) -> object:
    """Plan the statement with its arguments bound.

    Returns:
        The parsed ``EXPLAIN (VERBOSE, FORMAT JSON)`` document.
    """
    return json.loads(str(await conn.fetchval(f'EXPLAIN (VERBOSE, FORMAT JSON) {statement.sql}', *statement.params)))


async def explain_generic(conn: AsyncpgConnection, statement: Captured) -> object:
    """Plan the statement as a prepared statement held to its generic plan, then deallocate it.

    Returns:
        The parsed ``EXPLAIN (VERBOSE, FORMAT JSON)`` document.
    """
    arguments = ', '.join(sql_literal(value) for value in statement.params)
    try:
        async with conn.transaction():
            await conn.execute(f'PREPARE {_PREPARED_NAME} AS {statement.sql}')
            await conn.execute('SET LOCAL plan_cache_mode = force_generic_plan')
            document = await conn.fetchval(f'EXPLAIN (VERBOSE, FORMAT JSON) EXECUTE {_PREPARED_NAME}({arguments})')
            return json.loads(str(document))
    finally:
        await conn.execute(f'DEALLOCATE {_PREPARED_NAME}')
