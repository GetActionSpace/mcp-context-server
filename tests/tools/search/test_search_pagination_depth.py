"""Ranked search pagination: every page is a window into ONE ordering.

semantic, FTS, and hybrid search decide their final order after the database
returns rows (cross-encoder reranking, RRF fusion, or both). When the candidate
window was sized from the requested page, page N and page N + 1 were cut from
different candidate pools, so a document that entered only the larger pool could
outrank rows already served: adjacent pages repeated rows while other rows were
never returned at all. These tests page through a fixed corpus and assert the
concatenated pages reproduce the unpaginated prefix exactly.
"""

from types import ModuleType
from typing import Any

import pytest

import app.tools.search.fts as search_fts
import app.tools.search.hybrid as search_hybrid
import app.tools.search.limits as search_limits
import app.tools.search.semantic as search_semantic
from app.backends.base import BackendAccess
from tests.tools._generation_jobs import InMemoryGenerationJobs

CORPUS_SIZE = 25

TOOL_MODULES: dict[str, ModuleType] = {
    'semantic_search_context': search_semantic,
    'fts_search_context': search_fts,
    'hybrid_search_context': search_hybrid,
}


def _corpus() -> list[dict[str, Any]]:
    """Build a deterministic corpus ordered by descending first-stage score.

    Returns:
        Rows shaped like repository search output, best first-stage match first.
    """
    return [
        {
            'id': f'{index:032x}',
            'thread_id': 'pagination-thread',
            'source': 'agent',
            'content_type': 'text',
            'text_content': f'document number {index}',
            'metadata': None,
            'score': float(CORPUS_SIZE - index),
            'distance': float(index) / 100,
            'visibility': 'private',
            'is_owner': 1,
        }
        for index in range(CORPUS_SIZE)
    ]


class _ReverseReranker:
    """Reranker whose order is the exact reverse of the first-stage order.

    Inverting the order makes the candidate pool size decisive: the rows the
    cross-encoder promotes are the ones that enter last, so a pool that grows
    with the requested offset produces a visibly different ranking per page.
    """

    provider_name = 'reverse-test-reranker'

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score candidates so later first-stage rows win.

        Args:
            query: Unused; the ordering is deterministic.
            results: Candidates carrying 'id' and 'text'.
            limit: Maximum number of candidates to return.

        Returns:
            The candidates in reverse input order with a rerank_score.
        """
        del query
        ordered = [
            {'id': item.get('id'), 'rerank_score': float(position)}
            for position, item in enumerate(reversed(results))
        ]
        return ordered[:limit] if limit else ordered


class _FakeTagsRepo:
    async def get_tags_for_context(self, context_id: str, *, scope: BackendAccess) -> list[str]:
        """Return no tags.

        Args:
            context_id: Unused.
            scope: Unused.

        Returns:
            An empty tag list.
        """
        del context_id, scope
        return []


class _FakeRepos:
    tags = _FakeTagsRepo()
    generation_jobs = InMemoryGenerationJobs()


@pytest.fixture
def raw_search_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch both raw searches to serve a fixed corpus prefix, recording call kwargs.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        The list the patched raw searches append their keyword arguments to.
    """
    calls: list[dict[str, Any]] = []
    corpus = _corpus()

    async def fake_raw(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        calls.append(kwargs)
        limit = int(kwargs['limit'])
        offset = int(kwargs.get('offset') or 0)
        rows = [dict(row) for row in corpus[offset:offset + limit]]
        return rows, {'execution_time_ms': 0.0}

    async def fake_ensure_repositories() -> _FakeRepos:
        return _FakeRepos()

    for module in (search_semantic, search_hybrid):
        monkeypatch.setattr(module, 'semantic_search_raw', fake_raw)
        monkeypatch.setattr(module, 'get_embedding_provider', lambda: object())
    for module in (search_fts, search_hybrid):
        monkeypatch.setattr(module, 'fts_search_raw', fake_raw)
    for module in TOOL_MODULES.values():
        monkeypatch.setattr(module, 'ensure_repositories', fake_ensure_repositories)
        monkeypatch.setattr(module, 'get_reranking_provider', lambda: _ReverseReranker())
    return calls


async def _ids(response: dict[str, Any]) -> list[str]:
    """Extract the result ids from a search response.

    Args:
        response: A search tool response.

    Returns:
        The ids in response order.
    """
    return [str(result['id']) for result in response['results']]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool_name',
    ['semantic_search_context', 'fts_search_context', 'hybrid_search_context'],
)
@pytest.mark.usefixtures('raw_search_calls')
async def test_pages_reproduce_the_unpaginated_prefix(tool_name: str) -> None:
    """Four pages of two rows return exactly the first eight ranked rows."""
    tool = getattr(TOOL_MODULES[tool_name], tool_name)

    whole = await _ids(await tool(query='anything', limit=8, offset=0))

    paged: list[str] = []
    for offset in (0, 2, 4, 6):
        paged.extend(await _ids(await tool(query='anything', limit=2, offset=offset)))

    assert len(whole) == 8
    assert paged == whole
    assert len(set(paged)) == len(paged), 'a row was served on two different pages'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool_name',
    ['semantic_search_context', 'fts_search_context', 'hybrid_search_context'],
)
async def test_candidate_depth_is_independent_of_the_requested_offset(
    tool_name: str,
    raw_search_calls: list[dict[str, Any]],
) -> None:
    """The retrieval depth is the same for every page of the same query."""
    tool = getattr(TOOL_MODULES[tool_name], tool_name)

    for offset in (0, 2, 40):
        await tool(query='anything', limit=2, offset=offset)

    limits = {int(call['limit']) for call in raw_search_calls}
    offsets = {int(call.get('offset') or 0) for call in raw_search_calls}
    assert len(limits) == 1, f'candidate depth varied with the page: {limits}'
    assert offsets == {0}, 'the ranked window is sliced in Python, not by the database'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool_name',
    ['semantic_search_context', 'fts_search_context', 'hybrid_search_context'],
)
@pytest.mark.usefixtures('raw_search_calls')
async def test_window_past_the_ranked_depth_is_reported(tool_name: str) -> None:
    """A page reaching past the ranked depth carries the rank_depth_limit hint."""
    tool = getattr(TOOL_MODULES[tool_name], tool_name)

    inside = await tool(query='anything', limit=5, offset=0)
    assert 'rank_depth_limit' not in inside

    at_edge = await tool(query='anything', limit=search_limits.RANKED_SEARCH_DEPTH, offset=0)
    assert 'rank_depth_limit' not in at_edge

    beyond = await tool(query='anything', limit=5, offset=search_limits.RANKED_SEARCH_DEPTH - 1)
    assert beyond['rank_depth_limit'] == {
        'requested_offset': search_limits.RANKED_SEARCH_DEPTH - 1,
        'requested_limit': 5,
        'rank_depth': search_limits.RANKED_SEARCH_DEPTH,
    }
    assert beyond['results'] == []


def test_ranked_depth_covers_the_largest_servable_page() -> None:
    """The ranked ordering is deep enough for any single request to be served whole."""
    assert search_limits.RANKED_SEARCH_DEPTH >= search_limits.MAX_SEARCH_LIMIT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool_name',
    ['semantic_search_context', 'fts_search_context', 'hybrid_search_context'],
)
async def test_page_past_the_depth_runs_no_search_at_all(
    tool_name: str,
    raw_search_calls: list[dict[str, Any]],
) -> None:
    """An offset at or past the ranked depth is answered without touching a provider.

    The page is empty by arithmetic alone -- every ranked page is a window into an
    ordering at most RANKED_SEARCH_DEPTH rows deep -- so retrieving and cross-encoder
    scoring that whole ordering only to discard every row of the slice is pure waste. On
    the two tools with an embedding leg it is a network round trip to the provider.

    Args:
        tool_name: The ranked search tool under test.
        raw_search_calls: Recorder for the raw searches, which must stay empty.
    """
    tool = getattr(TOOL_MODULES[tool_name], tool_name)

    response = await tool(query='anything', limit=10, offset=search_limits.RANKED_SEARCH_DEPTH)

    assert raw_search_calls == []
    assert response['results'] == []
    assert response['count'] == 0
    assert response['rank_depth_limit'] == {
        'requested_offset': search_limits.RANKED_SEARCH_DEPTH,
        'requested_limit': 10,
        'rank_depth': search_limits.RANKED_SEARCH_DEPTH,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool_name',
    ['semantic_search_context', 'fts_search_context', 'hybrid_search_context'],
)
async def test_an_invalid_filter_yields_the_normal_path_past_the_depth(
    tool_name: str,
    raw_search_calls: list[dict[str, Any]],
) -> None:
    """A structurally invalid filter takes precedence over the empty-page shortcut.

    Answering "no rows" would hide a request that can never return rows for a different
    reason, leaving the client re-sending the same broken filter forever. The shortcut
    therefore yields, and the request takes the ordinary path that reports the filter --
    which, for the semantic leg, still rejects it BEFORE the embedding call.

    Args:
        tool_name: The ranked search tool under test.
        raw_search_calls: Recorder proving the ordinary path ran.
    """
    tool = getattr(TOOL_MODULES[tool_name], tool_name)

    await tool(
        query='anything',
        limit=10,
        offset=search_limits.RANKED_SEARCH_DEPTH,
        metadata_filters=[{'key': 'priority', 'operator': 'bogus_op', 'value': 5}],
    )

    assert raw_search_calls, 'the empty-page shortcut swallowed an invalid filter'
