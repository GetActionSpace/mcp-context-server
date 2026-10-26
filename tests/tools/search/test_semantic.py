"""semantic_search_context statistics, the validation-error response, and the caller's scope."""

from typing import Any

import pytest

from app.backends.base import BackendAccess
from tests.helpers import LOCAL_SCOPE
from tests.tools._generation_jobs import NoGenerationJobs


class TestSemanticEmbeddingGenerationStat:
    """The measured query-embedding duration is surfaced in the semantic stats.

    ``semantic_search_raw`` times the ``embed_query`` call and injects
    ``embedding_generation_ms`` (rounded, milliseconds) into the returned stats
    dict; the standalone ``semantic_search_context`` tool passes that dict through
    as its ``stats`` payload, and hybrid search inherits it via ``semantic_stats``.
    """

    @staticmethod
    def _patch_provider_and_repo(
        monkeypatch: pytest.MonkeyPatch,
        rows: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Stub the embedding provider and repository so no real embedding runs, returning both."""
        import app.tools.search.semantic as search_semantic

        class _FakeEmbeddingProvider:
            async def embed_query(self, _query: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        class _FakeEmbeddingsRepo:
            async def search(self, **_kwargs: object) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                # Return a fresh copy so the injected key is observed on the returned dict.
                return rows, dict(stats)

        class _FakeTagsRepo:
            async def get_tags_for_context(self, _context_id: str, *, scope: BackendAccess) -> list[str]:
                assert scope == LOCAL_SCOPE
                return []

        class _FakeRepos:
            embeddings = _FakeEmbeddingsRepo()
            tags = _FakeTagsRepo()
            generation_jobs = NoGenerationJobs()

        provider = _FakeEmbeddingProvider()
        repos = _FakeRepos()

        async def _fake_ensure_repositories() -> _FakeRepos:
            return repos

        monkeypatch.setattr(search_semantic, 'get_embedding_provider', lambda: provider)
        monkeypatch.setattr(search_semantic, 'ensure_repositories', _fake_ensure_repositories)
        # No reranking provider so the tool takes the plain (non-overfetch-rerank) path.
        monkeypatch.setattr(search_semantic, 'get_reranking_provider', lambda: None)
        return provider, repos

    @pytest.mark.asyncio
    async def test_raw_search_injects_embedding_generation_ms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """semantic_search_raw adds a non-negative embedding_generation_ms to stats."""
        from app.tools.search.legs import semantic_search_raw

        provider, repos = self._patch_provider_and_repo(monkeypatch, rows=[], stats={'rows_returned': 0})

        _results, stats = await semantic_search_raw(
            query='hi', limit=5, explain_query=True, repos=repos, embedding_provider=provider,
            scope=LOCAL_SCOPE,
        )

        assert 'embedding_generation_ms' in stats
        elapsed = stats['embedding_generation_ms']
        assert isinstance(elapsed, (int, float))
        assert elapsed >= 0
        # Pre-existing repository stats are preserved alongside the injected key.
        assert stats['rows_returned'] == 0

    @pytest.mark.asyncio
    async def test_tool_stats_carry_embedding_generation_ms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """semantic_search_context surfaces embedding_generation_ms under stats when
        explain_query=True (the documented HybridSemanticStatsDict field)."""
        from app.tools.search.semantic import semantic_search_context

        rows = [
            {
                'id': 'a' * 32, 'thread_id': 't', 'source': 'user', 'content_type': 'text',
                'text_content': 'hello world', 'distance': 0.1, 'metadata': None,
                'visibility': 'private', 'is_owner': 1,
            },
        ]
        self._patch_provider_and_repo(monkeypatch, rows=rows, stats={'rows_returned': 1})

        response = await semantic_search_context(query='hello', limit=5, explain_query=True)

        assert 'stats' in response
        stats = response['stats']
        assert 'embedding_generation_ms' in stats
        assert stats['embedding_generation_ms'] >= 0

    @pytest.mark.asyncio
    async def test_embedding_generation_ms_absent_without_explain_query(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without explain_query the tool emits no stats block at all."""
        from app.tools.search.semantic import semantic_search_context

        self._patch_provider_and_repo(monkeypatch, rows=[], stats={'rows_returned': 0})

        response = await semantic_search_context(query='hello', limit=5, explain_query=False)

        assert 'stats' not in response


class TestSemanticValidationErrorStats:
    """The semantic validation-error response carries a full stats dict when explain_query=True.

    A MetadataFilterValidationError (invalid metadata filter) returns a structured
    error response instead of raising. When explain_query is requested, that
    response's ``stats`` dict must include the ``backend`` key alongside the zeroed
    counters, mirroring the FTS error path and every success path so the shape is
    uniform for the client.
    """

    @pytest.mark.asyncio
    async def test_validation_error_stats_include_backend(self) -> None:
        """The error-path stats dict includes backend (the active storage backend type)."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from app.repositories.embedding_repository.records import MetadataFilterValidationError
        from app.tools.search.semantic import semantic_search_context

        with (
            patch('app.tools.search.semantic.get_reranking_provider', return_value=None),
            patch('app.tools.search.semantic.ensure_repositories', new=AsyncMock(return_value=MagicMock())),
            patch(
                'app.tools.search.semantic.semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', ['bad operator: nope'])),
            ),
        ):
            response = await semantic_search_context(
                query='python async',
                explain_query=True,
            )

        assert response['count'] == 0
        assert response['results'] == []
        assert response['error'] == 'Invalid filters'
        assert response['validation_errors'] == ['bad operator: nope']
        assert 'stats' in response
        stats = response['stats']
        # The backend key must be present and match the backend the tool actually
        # resolves (the module-level settings binding the production code reads),
        # so the error-path stats shape matches every other stats path.
        import app.tools.search.limits as search_limits

        assert 'backend' in stats
        assert stats['backend'] == search_limits.settings.storage.backend_type
        # The other documented error-path stat keys accompany it, including the
        # semantic shape's embedding timing counter (zeroed: no query executed).
        assert stats['execution_time_ms'] == 0.0
        assert stats['embedding_generation_ms'] == 0.0
        assert stats['filters_applied'] == 0
        assert stats['rows_returned'] == 0

    @pytest.mark.asyncio
    async def test_validation_error_omits_stats_without_explain_query(self) -> None:
        """Without explain_query the validation-error response carries no stats block."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from app.repositories.embedding_repository.records import MetadataFilterValidationError
        from app.tools.search.semantic import semantic_search_context

        with (
            patch('app.tools.search.semantic.get_reranking_provider', return_value=None),
            patch('app.tools.search.semantic.ensure_repositories', new=AsyncMock(return_value=MagicMock())),
            patch(
                'app.tools.search.semantic.semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', ['bad operator: nope'])),
            ),
        ):
            response = await semantic_search_context(
                query='python async',
                explain_query=False,
            )

        assert response['count'] == 0
        assert response['error'] == 'Invalid filters'
        assert 'stats' not in response


class TestSemanticSearchScoping:
    """semantic_search_context runs the vector search as the caller's scope."""

    @pytest.mark.asyncio
    async def test_scope_reaches_the_repository(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The caller's principal and groups reach the vector search; its filter count passes through."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        import app.tools.search.semantic as search_semantic
        from app.access_scope import AccessScope
        from app.tools.search.semantic import semantic_search_context
        from tests.helpers import as_principal

        repos = MagicMock()
        repos.embeddings.search = AsyncMock(return_value=([], {'filters_applied': 2, 'rows_returned': 0}))
        repos.generation_jobs = NoGenerationJobs()
        provider = MagicMock()
        provider.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        monkeypatch.setattr(search_semantic, 'get_embedding_provider', lambda: provider)
        monkeypatch.setattr(search_semantic, 'ensure_repositories', AsyncMock(return_value=repos))
        monkeypatch.setattr(search_semantic, 'get_reranking_provider', lambda: None)

        with as_principal('bob', groups=['team-x']):
            response = await semantic_search_context(query='q', thread_id='t', source='agent', explain_query=True)

        call = repos.embeddings.search.await_args
        assert call is not None
        assert call.kwargs['scope'] == AccessScope('bob', frozenset({'team-x'}))
        assert response['stats']['filters_applied'] == 2
