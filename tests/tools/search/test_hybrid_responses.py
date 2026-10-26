"""Tests for hybrid_search_context response semantics.

Covers search_modes_used, the structured error when every available mode fails validation, the
filter-caps validation error, and partial degradation when one sub-search fails.
"""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from tests.tools._generation_jobs import NoGenerationJobs


class TestSearchModesUsedSemantics:
    """Test that search_modes_used reflects execution, not results.

    Verifies that modes_used tracks whether a search mode executed
    successfully (no error), regardless of whether it returned results.
    """

    @staticmethod
    def _compute_modes_used(
        available_modes: list[str],
        fts_error: str | None,
        semantic_error: str | None,
    ) -> list[str]:
        """Replicate the production modes_used logic for testing."""
        modes_used: list[str] = []
        if 'fts' in available_modes and not fts_error:
            modes_used.append('fts')
        if 'semantic' in available_modes and not semantic_error:
            modes_used.append('semantic')
        return modes_used

    def test_fts_executed_zero_results_still_in_modes(self) -> None:
        """FTS executes with zero results -> still included in modes_used.

        When FTS runs successfully but finds no matching documents,
        it should still appear in modes_used because it was executed.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error=None,
            semantic_error=None,
        )

        assert modes_used == ['fts', 'semantic'], (
            f'Expected both modes when both executed, got {modes_used}'
        )

    def test_fts_error_excluded_from_modes(self) -> None:
        """FTS errors during execution -> excluded from modes_used.

        When FTS encounters an error, it should NOT appear in
        modes_used.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error='FTS search failed: connection timeout',
            semantic_error=None,
        )

        assert modes_used == ['semantic'], (
            f'Expected only semantic when FTS errored, got {modes_used}'
        )

    def test_semantic_error_excluded_from_modes(self) -> None:
        """Semantic errors during execution -> excluded from modes_used.

        When semantic search encounters an error, it should NOT
        appear in modes_used.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error=None,
            semantic_error='Embedding provider unavailable',
        )

        assert modes_used == ['fts'], (
            f'Expected only fts when semantic errored, got {modes_used}'
        )

    def test_both_error_empty_modes(self) -> None:
        """Both modes error -> empty modes_used.

        When both search modes encounter errors, modes_used should
        be empty.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error='FTS failed',
            semantic_error='Semantic failed',
        )

        assert modes_used == [], (
            f'Expected empty modes when both errored, got {modes_used}'
        )

    def test_semantic_only_mode_available(self) -> None:
        """Only semantic mode available and succeeds -> ['semantic'].

        When FTS is not in available_modes (not enabled), only semantic
        should appear regardless of fts_error state.
        """
        modes_used = self._compute_modes_used(
            available_modes=['semantic'],
            fts_error=None,
            semantic_error=None,
        )

        assert modes_used == ['semantic'], (
            f'Expected only semantic when FTS not available, got {modes_used}'
        )


class TestHybridAllModesFailedValidationResponse:
    """The structured error return when every available mode fails validation.

    Exercises the ACTUAL error branch of hybrid_search_context (not a local
    re-implementation): the response must use the documented
    search_modes_used key, and identical validation messages produced by both
    sub-searches over the same filters must be deduplicated.
    """

    @pytest.mark.asyncio
    async def test_error_response_keys_and_deduplicated_messages(self) -> None:
        from app.repositories.embedding_repository.records import MetadataFilterValidationError
        from app.repositories.fts_repository.faults import FtsValidationError
        from app.tools.search.hybrid import hybrid_search_context

        shared_messages = [
            "Invalid metadata filter {'key': 'a', 'operator': 'nope'}: unsupported operator",
        ]

        with (
            patch('app.tools.search.hybrid.ensure_repositories', AsyncMock(return_value=MagicMock())),
            patch('app.tools.search.hybrid.get_embedding_provider', return_value=object()),
            patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
            patch(
                'app.tools.search.hybrid.fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', list(shared_messages))),
            ),
            patch(
                'app.tools.search.hybrid.semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', list(shared_messages))),
            ),
        ):
            result = await hybrid_search_context(
                query='anything',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        assert result['count'] == 0
        assert result['results'] == []
        # The documented key, matching the success path, the docstring, and
        # the TypedDict -- not a stray spelling unique to the error branch.
        assert result['search_modes_used'] == []
        assert 'modes_used' not in result
        error_text = result['error']
        assert isinstance(error_text, str)
        assert error_text.startswith('All available search modes failed')
        # Both sub-searches validated the same filters and produced identical
        # messages; the client must see each defect once.
        assert result['validation_errors'] == shared_messages

    @pytest.mark.asyncio
    async def test_error_response_carries_fusion_method_and_counts(self) -> None:
        """The all-modes-failed response carries fusion_method, fts_count, and
        semantic_count -- the always-present response-shape keys the success path,
        the docstring, and HybridSearchResponseDict declare, so a client reading
        them never hits a KeyError on the error branch."""
        from app.repositories.embedding_repository.records import MetadataFilterValidationError
        from app.repositories.fts_repository.faults import FtsValidationError
        from app.tools.search.hybrid import hybrid_search_context

        messages = ['bad operator: nope']

        with (
            patch('app.tools.search.hybrid.ensure_repositories', AsyncMock(return_value=MagicMock())),
            patch('app.tools.search.hybrid.get_embedding_provider', return_value=object()),
            patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
            patch(
                'app.tools.search.hybrid.fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', list(messages))),
            ),
            patch(
                'app.tools.search.hybrid.semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', list(messages))),
            ),
        ):
            result = await hybrid_search_context(
                query='anything',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        assert result['fusion_method'] == 'rrf'
        assert result['fts_count'] == 0
        assert result['semantic_count'] == 0
        # No explain_query -> no stats block, mirroring the success path's gating.
        assert 'stats' not in result

    @pytest.mark.asyncio
    async def test_error_response_attaches_stats_under_explain_query(self) -> None:
        """With explain_query=True the all-modes-failed response carries the same
        stats keys the success path builds: real elapsed time, the (None) sub-search
        stats, a zeroed fusion_stats with the resolved rrf_k, and the adaptive FTS
        mode -- so a client reading response['stats'] under explain_query never hits
        a KeyError on the error branch."""
        from app.repositories.embedding_repository.records import MetadataFilterValidationError
        from app.repositories.fts_repository.faults import FtsValidationError
        from app.tools.search.hybrid import hybrid_search_context

        messages = ['bad operator: nope']

        with (
            patch('app.tools.search.hybrid.ensure_repositories', AsyncMock(return_value=MagicMock())),
            patch('app.tools.search.hybrid.get_embedding_provider', return_value=object()),
            patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
            patch(
                'app.tools.search.hybrid.fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', list(messages))),
            ),
            patch(
                'app.tools.search.hybrid.semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', list(messages))),
            ),
        ):
            result = await hybrid_search_context(
                query='anything',
                rrf_k=77,
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
                explain_query=True,
            )

        assert result['validation_errors'] == messages
        assert 'stats' in result
        stats = result['stats']
        assert isinstance(stats['execution_time_ms'], float)
        assert stats['execution_time_ms'] >= 0.0
        # Both sub-searches failed validation, so no sub-search stats were captured.
        assert stats['fts_stats'] is None
        assert stats['semantic_stats'] is None
        # The zeroed fusion block carries the client-resolved rrf_k.
        assert stats['fusion_stats'] == {
            'rrf_k': 77,
            'total_unique_documents': 0,
            'documents_in_both': 0,
            'documents_fts_only': 0,
            'documents_semantic_only': 0,
        }
        assert stats['adaptive_fts_mode'] in ('match', 'boolean')


class TestHybridFilterCapsValidationStats:
    """The hybrid boundary-caps validation error carries stats under explain_query.

    The caps early return runs before any sub-search, so its stats block is fully
    zeroed: 0.0 elapsed, no sub-search stats, a zeroed fusion_stats with the
    resolved rrf_k (resolved BEFORE the caps check), and the default 'match'
    adaptive mode. Both standalone siblings attach stats on this exact failure
    class; hybrid must match.
    """

    @pytest.mark.asyncio
    async def test_tags_cap_error_attaches_stats_under_explain_query(self) -> None:
        from app.tools.search.hybrid import hybrid_search_context
        from app.tools.search.limits import MAX_FILTER_TAGS

        oversized = [f'tag-{i}' for i in range(MAX_FILTER_TAGS + 1)]
        result = await hybrid_search_context(query='anything', tags=oversized, rrf_k=42, explain_query=True)

        assert result['count'] == 0
        assert 'exceeds the maximum' in result['error']
        assert result['validation_errors'] == [result['error']]
        assert result['fusion_method'] == 'rrf'
        assert result['fts_count'] == 0
        assert result['semantic_count'] == 0
        assert result['stats'] == {
            'execution_time_ms': 0.0,
            'fts_stats': None,
            'semantic_stats': None,
            'fusion_stats': {
                'rrf_k': 42,
                'total_unique_documents': 0,
                'documents_in_both': 0,
                'documents_fts_only': 0,
                'documents_semantic_only': 0,
            },
            'adaptive_fts_mode': 'match',
        }

    @pytest.mark.asyncio
    async def test_tags_cap_error_omits_stats_without_explain_query(self) -> None:
        from app.tools.search.hybrid import hybrid_search_context
        from app.tools.search.limits import MAX_FILTER_TAGS

        oversized = [f'tag-{i}' for i in range(MAX_FILTER_TAGS + 1)]
        result = await hybrid_search_context(query='anything', tags=oversized)

        assert 'exceeds the maximum' in result['error']
        assert 'stats' not in result

    @pytest.mark.asyncio
    async def test_metadata_filters_cap_error_attaches_stats_under_explain_query(self) -> None:
        from app.tools.search.hybrid import hybrid_search_context
        from app.tools.search.limits import MAX_METADATA_FILTERS

        oversized = [{'key': 'status', 'operator': 'eq', 'value': 'x'}] * (MAX_METADATA_FILTERS + 1)
        result = await hybrid_search_context(query='anything', metadata_filters=oversized, explain_query=True)

        assert result['count'] == 0
        assert 'exceeds the maximum' in result['error']
        assert 'stats' in result
        assert result['stats']['fusion_stats']['total_unique_documents'] == 0


class TestHybridPartialDegradationResponse:
    """One sub-search fails validation while the other succeeds (graceful degradation).

    Exercises the ACTUAL partial-degradation path of hybrid_search_context: the
    surviving mode's results are returned, but the response also carries the
    specific failure text in ``warnings`` and the per-filter details under the
    same ``validation_errors`` key the all-failed branch uses, so a client can
    correct an invalid sub-query even when results were still produced.
    """

    @staticmethod
    def _repos_with_tags() -> MagicMock:
        """A repository container whose tag lookup returns an empty list."""
        repos = MagicMock()
        repos.tags = MagicMock()
        repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        repos.generation_jobs = NoGenerationJobs()
        return repos

    @pytest.mark.asyncio
    async def test_semantic_failure_surfaces_warning_and_validation_errors(self) -> None:
        """Semantic fails validation, FTS succeeds: results returned, warning + details present."""
        from app.repositories.embedding_repository.records import MetadataFilterValidationError
        from app.tools.search.hybrid import hybrid_search_context

        semantic_messages = [
            "Invalid metadata filter {'key': 'a', 'operator': 'nope'}: unsupported operator",
        ]
        fts_rows: list[dict[str, Any]] = [
            {
                'id': 'a' * 32, 'thread_id': 't1', 'source': 'agent', 'content_type': 'text',
                'text_content': 'python async guide', 'score': 9.0, 'metadata': None,
                'created_at': '2025-01-01T00:00:00Z', 'updated_at': '2025-01-01T00:00:00Z',
                'visibility': 'private', 'is_owner': 1,
            },
        ]

        with (
            patch('app.tools.search.hybrid.ensure_repositories', AsyncMock(return_value=self._repos_with_tags())),
            patch('app.tools.search.hybrid.get_embedding_provider', return_value=object()),
            patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
            patch('app.tools.search.hybrid.fts_search_raw', AsyncMock(return_value=(fts_rows, {}))),
            patch(
                'app.tools.search.hybrid.semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', list(semantic_messages))),
            ),
        ):
            result = await hybrid_search_context(
                query='python async',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        # FTS survived, so results are still returned (graceful degradation).
        assert result['count'] == 1
        assert result['results'][0]['id'] == 'a' * 32
        assert result['search_modes_used'] == ['fts']

        # The warning embeds the SPECIFIC semantic failure text, not a generic notice.
        warnings = result['warnings']
        assert isinstance(warnings, list)
        assert len(warnings) == 1
        assert 'Semantic:' in warnings[0]
        assert 'Invalid filters' in warnings[0]

        # The per-filter details ride under the same validation_errors key the
        # all-failed branch uses, so the client can fix the invalid sub-query.
        assert result['validation_errors'] == semantic_messages
        # No top-level error on partial degradation -- the request partly succeeded.
        assert 'error' not in result

    @pytest.mark.asyncio
    async def test_fts_failure_surfaces_warning_and_validation_errors(self) -> None:
        """FTS fails validation, semantic succeeds: mirror case with FTS-prefixed warning."""
        from app.repositories.fts_repository.faults import FtsValidationError
        from app.tools.search.hybrid import hybrid_search_context

        fts_messages = ['invalid boolean expression near ")"']
        semantic_rows: list[dict[str, Any]] = [
            {
                'id': 'b' * 32, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'python async guide', 'distance': 0.2, 'metadata': None,
                'created_at': '2025-01-01T00:00:00Z', 'updated_at': '2025-01-01T00:00:00Z',
                'visibility': 'private', 'is_owner': 1,
            },
        ]

        with (
            patch('app.tools.search.hybrid.ensure_repositories', AsyncMock(return_value=self._repos_with_tags())),
            patch('app.tools.search.hybrid.get_embedding_provider', return_value=object()),
            patch('app.tools.search.hybrid.get_reranking_provider', return_value=None),
            patch(
                'app.tools.search.hybrid.fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', list(fts_messages))),
            ),
            patch('app.tools.search.hybrid.semantic_search_raw', AsyncMock(return_value=(semantic_rows, {}))),
        ):
            result = await hybrid_search_context(
                query='python async',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        assert result['count'] == 1
        assert result['results'][0]['id'] == 'b' * 32
        assert result['search_modes_used'] == ['semantic']

        warnings = result['warnings']
        assert isinstance(warnings, list)
        assert len(warnings) == 1
        assert 'FTS:' in warnings[0]
        assert 'Invalid filters' in warnings[0]

        assert result['validation_errors'] == fts_messages
        assert 'error' not in result
