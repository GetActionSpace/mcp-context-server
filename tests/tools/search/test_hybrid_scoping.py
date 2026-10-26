"""hybrid_search_context runs both of its legs as one scope, the caller's.

The tool resolves the caller's scope once and hands the very same object to the full-text
and the semantic leg, so the fused ranking, the counts and the explain stats all derive from
the rows the caller may read.
"""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.access_scope import AccessScope
from tests.helpers import as_principal
from tests.tools._generation_jobs import NoGenerationJobs


@pytest.mark.asyncio
async def test_both_legs_receive_the_same_scope_object() -> None:
    """The full-text and the semantic leg run as the caller's scope, one object for both."""
    from app.tools.search.hybrid import hybrid_search_context

    repos = MagicMock()
    repos.tags.get_tags_for_context = AsyncMock(return_value=[])
    repos.generation_jobs = NoGenerationJobs()
    fts_rows: list[dict[str, Any]] = [
        {
            'id': 'a' * 32, 'thread_id': 't', 'source': 'agent', 'content_type': 'text',
            'text_content': 'lattice', 'score': 1.0, 'metadata': None,
            'created_at': '2026-01-01T00:00:00Z', 'updated_at': '2026-01-01T00:00:00Z',
            'visibility': 'private', 'is_owner': 1,
        },
    ]
    fts_leg = AsyncMock(return_value=(fts_rows, {}))
    semantic_leg = AsyncMock(return_value=([], {}))

    with (
        patch('app.tools.search.hybrid.ensure_repositories', AsyncMock(return_value=repos)),
        patch('app.tools.search.hybrid.get_embedding_provider', return_value=object()),
        patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
        patch('app.tools.search.hybrid.fts_search_raw', fts_leg),
        patch('app.tools.search.hybrid.semantic_search_raw', semantic_leg),
        as_principal('bob', groups=['team-x']),
    ):
        result = await hybrid_search_context(query='lattice', thread_id='t')

    assert fts_leg.await_args is not None
    assert semantic_leg.await_args is not None
    fts_scope = fts_leg.await_args.kwargs['scope']
    assert fts_scope == AccessScope('bob', frozenset({'team-x'}))
    assert semantic_leg.await_args.kwargs['scope'] is fts_scope
    assert result['search_modes_used'] == ['fts', 'semantic']
    assert [entry['id'] for entry in result['results']] == ['a' * 32]
