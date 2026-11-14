"""Tests for the PostgreSQL plan readers in ``_plans.py`` that need no database.

``plan_nodes`` lists every node of a parsed ``EXPLAIN (FORMAT JSON)`` document or of a part
of one, the nodes nested under ``Plans`` at any depth included, and nothing that is not a
node. ``sql_literal`` renders the arguments a captured statement binds as the literals
``EXECUTE`` takes, and refuses a value it has no literal for.
"""

import pytest

from tests.integration.postgresql._plans import plan_nodes
from tests.integration.postgresql._plans import sql_literal

pytestmark = [pytest.mark.integration]


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
    'value', [pytest.param(True, id='bool'), pytest.param(1.5, id='float'), pytest.param(None, id='none')],
)
def test_sql_literal_refuses_a_value_it_has_no_literal_for(value: object) -> None:
    with pytest.raises(TypeError):
        sql_literal(value)
