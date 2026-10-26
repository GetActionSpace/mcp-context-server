"""The ranked search tools report each result's visibility, ownership and generation state, never its owner.

Runs semantic_search_context, fts_search_context and hybrid_search_context against a real
SQLite database as two principals. The repositories compute the ownership flag in SQL as an
integer; every tool returns it as a bool, because clients validate it against the boolean
output schema. The generation state of a response's results comes from one state read.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

import pytest

from app.startup import ensure_repositories
from app.tools.search.fts import fts_search_context
from app.tools.search.hybrid import hybrid_search_context
from app.tools.search.semantic import semantic_search_context
from tests.conftest import requires_semantic_search
from tests.helpers import as_principal
from tests.tools._generation_jobs import record_state_reads
from tests.tools._generation_jobs import seed_generation_jobs
from tests.tools.search.conftest import RANKED_KEYWORD
from tests.tools.search.conftest import RANKED_THREAD

type RankedTool = Callable[..., Awaitable[dict[str, Any]]]

_TOOLS = pytest.mark.parametrize(
    'tool',
    [semantic_search_context, fts_search_context, hybrid_search_context],
    ids=['semantic', 'fts', 'hybrid'],
)


async def _read_back(tool: RankedTool, principal_id: str) -> dict[str, tuple[object, object]]:
    """Search as the principal, checking that every result carries a bool ownership flag and no owner.

    Returns:
        Each result's ``(visibility, is_owner)`` keyed by its id.
    """
    with as_principal(principal_id):
        response = await tool(query=RANKED_KEYWORD, thread_id=RANKED_THREAD, limit=10)
    results: list[dict[str, Any]] = response['results']
    for entry in results:
        assert type(entry['is_owner']) is bool, entry
        assert 'owner_id' not in entry
    return {entry['id']: (entry['visibility'], entry['is_owner']) for entry in results}


@requires_semantic_search
@pytest.mark.asyncio
@_TOOLS
async def test_owner_reads_back_ownership_of_its_entries(tool: RankedTool, ranked_entries: dict[str, str]) -> None:
    """Alice reads both of her entries as their owner and never sees bob's private one."""
    assert await _read_back(tool, 'alice') == {
        ranked_entries['alice_private']: ('private', True),
        ranked_entries['alice_public']: ('public', True),
    }


@requires_semantic_search
@pytest.mark.asyncio
@_TOOLS
async def test_reader_reads_back_another_owners_public_entry(tool: RankedTool, ranked_entries: dict[str, str]) -> None:
    """Bob reads alice's public entry without owning it, and his own private entry as its owner."""
    assert await _read_back(tool, 'bob') == {
        ranked_entries['alice_public']: ('public', False),
        ranked_entries['bob_private']: ('private', True),
    }


@requires_semantic_search
@pytest.mark.asyncio
@_TOOLS
async def test_results_report_their_generation_state(
    tool: RankedTool, ranked_entries: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed or running job reads failed or pending, no job reads ready, from one state read per response."""
    repos = await ensure_repositories()
    await seed_generation_jobs(
        repos.context.backend, {ranked_entries['alice_private']: 'failed', ranked_entries['alice_public']: 'running'},
    )
    reads = record_state_reads(monkeypatch, repos)

    with as_principal('bob'):
        response = await tool(query=RANKED_KEYWORD, thread_id=RANKED_THREAD, limit=10)
    with as_principal('alice'):
        owned = await tool(query=RANKED_KEYWORD, thread_id=RANKED_THREAD, limit=10)

    assert {entry['id']: entry['generation_status'] for entry in response['results']} == {
        ranked_entries['alice_public']: 'pending', ranked_entries['bob_private']: 'ready',
    }
    assert {entry['id']: entry['generation_status'] for entry in owned['results']} == {
        ranked_entries['alice_private']: 'failed', ranked_entries['alice_public']: 'pending',
    }
    assert [sorted(read) for read in reads] == [
        sorted([ranked_entries['alice_public'], ranked_entries['bob_private']]),
        sorted([ranked_entries['alice_private'], ranked_entries['alice_public']]),
    ]
