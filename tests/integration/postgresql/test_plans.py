"""Tests for the PostgreSQL plan readers in ``_plans.py``.

``plan_nodes`` lists every node of a parsed ``EXPLAIN (FORMAT JSON)`` document or of a part
of one, the nodes nested under ``Plans`` at any depth included, and nothing that is not a
node. ``sql_literal`` renders the arguments a captured statement binds as the literals
``EXECUTE`` takes, and refuses a value it has no literal for. These need no database.

``explain_custom`` and ``explain_generic`` plan one read of a system catalog against the test
container: the custom plan compares with the bound value folded in as a constant, the
generic plan with the parameter ``$1``, which only a plan no argument value shaped keeps.
``explain_generic`` deallocates its prepared statement, so it plans again on the same
connection.
"""

import datetime as dt

import asyncpg
import pytest

from tests.helpers import Captured
from tests.integration.postgresql._plans import explain_custom
from tests.integration.postgresql._plans import explain_generic
from tests.integration.postgresql._plans import plan_nodes
from tests.integration.postgresql._plans import sql_literal

pytestmark = [pytest.mark.integration]

# A read of a catalog every database carries, bound to one argument.
_CATALOG_READ = Captured(sql='SELECT oid FROM pg_catalog.pg_class WHERE relname = $1', params=('pg_class',))


def test_plan_nodes_lists_every_nested_node_in_document_order() -> None:
    document: object = [{
        'Plan': {
            'Node Type': 'Limit',
            'Output': ['id'],
            'Plans': [
                {'Node Type': 'Result', 'Parent Relationship': 'InitPlan', 'Output': ['true']},
                {
                    'Node Type': 'Nested Loop',
                    'Plans': [
                        {'Node Type': 'Index Scan', 'Index Name': 'context_entries_pkey', 'Index Cond': '(id >= $1)'},
                        {'Node Type': 'Seq Scan', 'Relation Name': 'tags'},
                    ],
                },
            ],
        },
        'Planning': {'Shared Hit Blocks': 3},
    }]

    nodes = plan_nodes(document)

    assert [node['Node Type'] for node in nodes] == ['Limit', 'Result', 'Nested Loop', 'Index Scan', 'Seq Scan']
    assert nodes[3]['Index Cond'] == '(id >= $1)'
    assert [node['Node Type'] for node in plan_nodes(nodes[2]['Plans'])] == ['Index Scan', 'Seq Scan']


def test_plan_nodes_of_a_document_without_nodes_is_empty() -> None:
    assert plan_nodes([{'Planning': {'Shared Hit Blocks': 0}}]) == []


@pytest.mark.parametrize(
    ('value', 'literal'),
    [
        pytest.param(42, '42', id='int'),
        pytest.param('alice', "'alice'", id='str'),
        pytest.param("o'brien", "'o''brien'", id='str-with-quote'),
        pytest.param(['team-x', "o'brien"], "ARRAY['team-x', 'o''brien']::text[]", id='list'),
        pytest.param([], 'ARRAY[]::text[]', id='empty-list'),
    ],
)
def test_sql_literal_renders_each_bound_argument(value: object, literal: str) -> None:
    assert sql_literal(value) == literal


@pytest.mark.parametrize(
    'value',
    [
        pytest.param(True, id='bool'),
        pytest.param(1.5, id='float'),
        pytest.param(None, id='none'),
        pytest.param(dt.datetime(2026, 1, 1, tzinfo=dt.UTC), id='datetime'),
        pytest.param(['team-x', True], id='list-with-bool'),
    ],
)
def test_sql_literal_refuses_a_value_it_has_no_literal_for(value: object) -> None:
    with pytest.raises(TypeError):
        sql_literal(value)


def _catalog_condition(document: object) -> str:
    """Return the condition the one scan of ``pg_class`` in a plan applies, as an index or as a filter."""
    scans = [node for node in plan_nodes(document) if node.get('Relation Name') == 'pg_class']
    assert len(scans) == 1, document
    return str(scans[0].get('Index Cond') or scans[0].get('Filter'))


@pytest.mark.requires_docker_postgres
@pytest.mark.asyncio
async def test_explain_generic_keeps_the_parameter_that_explain_custom_binds(pg_test_url: str) -> None:
    conn = await asyncpg.connect(pg_test_url)
    try:
        custom = _catalog_condition(await explain_custom(conn, _CATALOG_READ))
        generic = [_catalog_condition(await explain_generic(conn, _CATALOG_READ)) for _ in range(2)]
    finally:
        await conn.close()

    assert "'pg_class'" in custom, custom
    assert '$1' not in custom, custom
    for condition in generic:
        assert '$1' in condition, generic
        assert "'pg_class'" not in condition, generic
