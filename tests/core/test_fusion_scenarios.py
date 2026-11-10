"""Scenario tests for reciprocal rank fusion and result counting.

Covers realistic mixes of FTS and semantic rankings, metadata preservation, large result sets, the k
parameter, results without an id, limit and offset slicing, and the score fields of fused results.
"""

from typing import Any

from app.fusion import count_unique_results
from app.fusion import reciprocal_rank_fusion
from app.types import HybridFusedResultDict


def _fuse(
    fts_results: list[dict[str, Any]], semantic_results: list[dict[str, Any]], *, k: int, limit: int,
) -> list[HybridFusedResultDict]:
    """Fuse leg rows that carry the access fields every ranked statement selects.

    The scenarios vary ids, scores and text; each row gains the visibility and the
    ownership flag that every row of both legs carries.

    Returns:
        The fused results.
    """

    def _with_access(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{'visibility': 'private', 'is_owner': 1, **row} for row in rows]

    return reciprocal_rank_fusion(_with_access(fts_results), _with_access(semantic_results), k=k, limit=limit)


class TestRRFIntegration:
    """Test RRF algorithm with realistic data scenarios."""

    def test_rrf_with_diverse_rankings(self) -> None:
        """Test RRF with documents having different rankings in each source."""
        # Simulate FTS results (ranked by relevance score)
        fts_results: list[dict[str, Any]] = [
            {'id': '1', 'score': 10.0, 'text_content': 'Python programming tutorial', 'thread_id': 't1'},
            {'id': '2', 'score': 8.5, 'text_content': 'Python data science guide', 'thread_id': 't1'},
            {'id': '3', 'score': 7.0, 'text_content': 'Machine learning basics', 'thread_id': 't1'},
            {'id': '4', 'score': 5.5, 'text_content': 'Deep learning neural networks', 'thread_id': 't1'},
        ]

        # Simulate semantic results (ranked by distance - lower is better)
        semantic_results: list[dict[str, Any]] = [
            {'id': '3', 'distance': 0.1, 'text_content': 'Machine learning basics', 'thread_id': 't1'},
            {'id': '4', 'distance': 0.2, 'text_content': 'Deep learning neural networks', 'thread_id': 't1'},
            {'id': '2', 'distance': 0.3, 'text_content': 'Python data science guide', 'thread_id': 't1'},
            {'id': '5', 'distance': 0.4, 'text_content': 'AI fundamentals', 'thread_id': 't1'},
        ]

        results = _fuse(fts_results, semantic_results, k=60, limit=10)

        # Verify all unique documents are present
        result_ids = [r.get('id') for r in results]
        assert len(result_ids) == 5  # 5 unique documents

        # Documents appearing in both should have higher scores
        # Doc 2, 3, 4 appear in both
        overlap_ids = {'2', '3', '4'}
        for r in results:
            if r.get('id') in overlap_ids:
                scores = r.get('scores', {})
                # Should have both ranks
                assert scores.get('fts_rank') is not None
                assert scores.get('semantic_rank') is not None

        # Verify scores structure
        for r in results:
            scores = r.get('scores', {})
            assert 'rrf' in scores
            assert scores['rrf'] > 0

    def test_rrf_preserves_metadata(self) -> None:
        """Test that RRF preserves metadata from both sources."""
        fts_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'score': 10.0,
                'text_content': 'Test content',
                'thread_id': 'thread-1',
                'source': 'agent',
                'content_type': 'text',
                'created_at': '2024-01-01T00:00:00Z',
            },
        ]

        semantic_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'distance': 0.1,
                'text_content': 'Test content',
                'metadata': {'key': 'value', 'priority': 5},
                'tags': ['tag1', 'tag2'],
            },
        ]

        results = _fuse(fts_results, semantic_results, k=60, limit=10)

        assert len(results) == 1
        result = results[0]

        # Should have data from FTS
        assert result.get('thread_id') == 'thread-1'
        assert result.get('source') == 'agent'
        assert result.get('content_type') == 'text'
        assert result.get('created_at') == '2024-01-01T00:00:00Z'

        # Should have data from semantic
        assert result.get('metadata') == {'key': 'value', 'priority': 5}
        assert result.get('tags') == ['tag1', 'tag2']

    def test_rrf_handles_large_result_sets(self) -> None:
        """Test RRF performance with larger result sets."""
        # Create 100 FTS results
        fts_results: list[dict[str, Any]] = [
            {'id': i, 'score': 100 - i, 'text_content': f'Document {i}', 'thread_id': 't1'}
            for i in range(1, 101)
        ]

        # Create 100 semantic results with 50% overlap
        semantic_results: list[dict[str, Any]] = [
            {'id': i + 50, 'distance': i * 0.01, 'text_content': f'Document {i + 50}', 'thread_id': 't1'}
            for i in range(1, 101)
        ]

        results = _fuse(fts_results, semantic_results, k=60, limit=50)

        # Should return exactly 50 results
        assert len(results) == 50

        # Results should be sorted by RRF score
        scores = [r.get('scores', {}).get('rrf', 0) for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_k_parameter_impact(self) -> None:
        """Test how k parameter affects ranking and score distribution."""
        fts_results: list[dict[str, Any]] = [
            {'id': '1', 'score': 10.0, 'text_content': 'Top FTS result'},
            {'id': '2', 'score': 5.0, 'text_content': 'Second FTS result'},
        ]

        semantic_results: list[dict[str, Any]] = [
            {'id': '2', 'distance': 0.1, 'text_content': 'Top semantic (also #2 in FTS)'},
            {'id': '3', 'distance': 0.2, 'text_content': 'Second semantic only'},
        ]

        # With small k, top ranks matter more
        results_small_k = _fuse(fts_results, semantic_results, k=1, limit=10)
        # With large k, ranks matter less (more uniform)
        results_large_k = _fuse(fts_results, semantic_results, k=100, limit=10)

        # Document 2 appears in both - should be top in both cases
        assert results_small_k[0].get('id') == '2'
        assert results_large_k[0].get('id') == '2'

        # Verify k affects score magnitudes
        small_k_scores = {r.get('id'): r.get('scores', {}).get('rrf', 0) for r in results_small_k}
        large_k_scores = {r.get('id'): r.get('scores', {}).get('rrf', 0) for r in results_large_k}

        # Scores should be smaller with large k (1/(k+rank) decreases as k increases)
        assert small_k_scores['2'] > large_k_scores['2']
        assert small_k_scores['1'] > large_k_scores['1']

        # With small k, the absolute difference between rank 1 and rank 2 is larger
        # k=1: rank1 = 1/2 = 0.5, rank2 = 1/3 = 0.333, diff = 0.167
        # k=100: rank1 = 1/101 ≈ 0.0099, rank2 = 1/102 ≈ 0.0098, diff ≈ 0.0001
        small_k_diff = 1 / (1 + 1) - 1 / (1 + 2)  # 0.5 - 0.333 = 0.167
        large_k_diff = 1 / (100 + 1) - 1 / (100 + 2)  # much smaller

        assert small_k_diff > large_k_diff


class TestCountUniqueResultsIntegration:
    """Test count_unique_results with various scenarios."""

    def test_count_with_realistic_data(self) -> None:
        """Test counting with realistic search results."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 10.0},
            {'id': 2, 'score': 8.0},
            {'id': 3, 'score': 6.0},
            {'id': 4, 'score': 4.0},
            {'id': 5, 'score': 2.0},
        ]

        semantic_results: list[dict[str, Any]] = [
            {'id': 3, 'distance': 0.1},  # Overlap
            {'id': 4, 'distance': 0.2},  # Overlap
            {'id': 6, 'distance': 0.3},
            {'id': 7, 'distance': 0.4},
        ]

        fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)

        assert fts_only == 3  # IDs 1, 2, 5
        assert semantic_only == 2  # IDs 6, 7
        assert overlap == 2  # IDs 3, 4


class TestHybridSearchToolIntegration:
    """Test hybrid_search_context tool integration.

    Note: These tests focus on the fusion algorithm and response structure.
    Full integration tests with the MCP server are in test_real_server.py.
    """

    def test_fusion_single_source_fts_only(self) -> None:
        """Test fusion with only FTS results (semantic unavailable scenario)."""
        fts_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'thread_id': 't1',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Python tutorial',
                'score': 10.0,
                'metadata': None,
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z',
            },
        ]

        results = _fuse(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        assert results[0].get('id') == '1'
        scores = results[0].get('scores', {})
        assert scores.get('fts_rank') == 1
        assert scores.get('semantic_rank') is None

    def test_fusion_single_source_semantic_only(self) -> None:
        """Test fusion with only semantic results (FTS unavailable scenario)."""
        semantic_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'thread_id': 't1',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Python tutorial',
                'distance': 0.1,
                'metadata': {'priority': 5},
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z',
            },
        ]

        results = _fuse([], semantic_results, k=60, limit=10)

        assert len(results) == 1
        assert results[0].get('id') == '1'
        scores = results[0].get('scores', {})
        assert scores.get('fts_rank') is None
        assert scores.get('semantic_rank') == 1

    def test_fusion_metadata_json_string_handling(self) -> None:
        """Test that fusion preserves metadata regardless of format."""
        fts_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'text_content': 'Test content',
                'score': 10.0,
                'metadata': {'priority': 5},  # Already a dict
            },
        ]

        results = _fuse(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        # Metadata should be preserved as-is
        assert results[0].get('metadata') == {'priority': 5}


class TestRRFEdgeCases:
    """Test edge cases in RRF fusion algorithm."""

    def test_rrf_skips_semantic_results_with_none_id(self) -> None:
        """Test RRF skips semantic results where id is None."""
        semantic_results: list[dict[str, Any]] = [
            {'id': None, 'distance': 0.1, 'text_content': 'No ID entry'},
            {'id': '1', 'distance': 0.2, 'text_content': 'Valid entry'},
        ]
        fts_results: list[dict[str, Any]] = []

        results = _fuse(fts_results, semantic_results, k=60, limit=10)

        # Only the entry with valid ID should be in results
        assert len(results) == 1
        assert results[0].get('id') == '1'

    def test_rrf_skips_fts_results_with_none_id(self) -> None:
        """Test RRF skips FTS results where id is None."""
        fts_results: list[dict[str, Any]] = [
            {'id': None, 'score': 10.0, 'text_content': 'No ID'},
            {'id': '2', 'score': 8.0, 'text_content': 'Valid'},
        ]

        results = _fuse(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        assert results[0].get('id') == '2'

    def test_rrf_skips_both_sources_with_none_ids(self) -> None:
        """Test RRF skips None IDs from both FTS and semantic results."""
        fts_results: list[dict[str, Any]] = [
            {'id': None, 'score': 10.0, 'text_content': 'FTS no ID'},
            {'id': '1', 'score': 8.0, 'text_content': 'FTS valid'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': None, 'distance': 0.1, 'text_content': 'Semantic no ID'},
            {'id': '2', 'distance': 0.2, 'text_content': 'Semantic valid'},
        ]

        results = _fuse(fts_results, semantic_results, k=60, limit=10)

        # Only entries with valid IDs should be in results
        result_ids = {r.get('id') for r in results}
        assert result_ids == {'1', '2'}
        assert len(results) == 2

    def test_rrf_all_none_ids_returns_empty(self) -> None:
        """Test RRF returns empty list when all IDs are None."""
        fts_results: list[dict[str, Any]] = [
            {'id': None, 'score': 10.0, 'text_content': 'No ID 1'},
            {'id': None, 'score': 8.0, 'text_content': 'No ID 2'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': None, 'distance': 0.1, 'text_content': 'No ID 3'},
        ]

        results = _fuse(fts_results, semantic_results, k=60, limit=10)

        assert results == []


class TestHybridSearchPagination:
    """Test hybrid search offset and limit handling."""

    def test_limit_applied_correctly(self) -> None:
        """Test that limit is applied after RRF fusion."""
        # Create 10 FTS results
        fts_results: list[dict[str, Any]] = [
            {'id': str(i), 'score': 10.0 - i, 'text_content': f'Doc {i}', 'thread_id': 't1'}
            for i in range(1, 11)
        ]

        # Request limit=3
        results = _fuse(fts_results, [], k=60, limit=3)

        assert len(results) == 3
        # Results should be top 3 by RRF score
        result_ids = [r.get('id') for r in results]
        assert result_ids == ['1', '2', '3']

    def test_limit_exceeds_available_results(self) -> None:
        """Test limit larger than available results returns all."""
        fts_results: list[dict[str, Any]] = [
            {'id': str(i), 'score': 10.0 - i, 'text_content': f'Doc {i}'}
            for i in range(1, 6)  # Only 5 results
        ]

        # Request limit=20 but only 5 exist
        results = _fuse(fts_results, [], k=60, limit=20)

        assert len(results) == 5

    def test_pagination_simulation_with_offset(self) -> None:
        """Test pagination by simulating offset with list slicing.

        In real usage, offset is applied after RRF fusion via result slicing.
        This tests the behavior of paginating through fused results.
        """
        # Create 10 FTS results
        fts_results: list[dict[str, Any]] = [
            {'id': str(i), 'score': 10.0 - (i * 0.1), 'text_content': f'Doc {i}', 'thread_id': 't1'}
            for i in range(1, 11)
        ]

        # Get all results first
        all_results = _fuse(fts_results, [], k=60, limit=10)

        # Simulate offset=3, limit=3 (skip first 3, return next 3)
        paginated = all_results[3:6]

        assert len(paginated) == 3
        # First 3 IDs (1, 2, 3) should be skipped
        paginated_ids = [r.get('id') for r in paginated]
        assert '1' not in paginated_ids
        assert '2' not in paginated_ids
        assert '3' not in paginated_ids


class TestHybridSearchResponseStructure:
    """Test hybrid search response TypedDict structure."""

    def test_response_has_required_fields(self) -> None:
        """Test that RRF results have all required fields."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 10.0, 'text_content': 'Test'},
        ]

        results = _fuse(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        result = results[0]

        # Check all required fields are present
        required_fields = [
            'id',
            'thread_id',
            'source',
            'content_type',
            'text_content',
            'metadata',
            'created_at',
            'updated_at',
            'tags',
            'scores',
        ]

        for field in required_fields:
            assert field in result, f'Missing required field: {field}'

        # Check scores structure - use .get() for TypedDict compatibility
        scores = result.get('scores', {})
        score_fields = ['rrf', 'fts_rank', 'semantic_rank', 'fts_score', 'semantic_distance', 'rerank_score']
        for field in score_fields:
            assert field in scores, f'Missing score field: {field}'

    def test_response_scores_type_consistency(self) -> None:
        """Test that score fields have consistent types."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 10.0, 'text_content': 'Test'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': 1, 'distance': 0.5, 'text_content': 'Test'},
        ]

        results = _fuse(fts_results, semantic_results, k=60, limit=10)

        assert len(results) == 1
        scores = results[0].get('scores', {})

        # rrf should always be a float > 0
        rrf_value = scores.get('rrf')
        assert isinstance(rrf_value, float)
        assert rrf_value > 0

        # When present, ranks should be integers
        fts_rank = scores.get('fts_rank')
        semantic_rank = scores.get('semantic_rank')
        assert isinstance(fts_rank, int)
        assert isinstance(semantic_rank, int)

        # When present, scores should be floats
        fts_score = scores.get('fts_score')
        semantic_distance = scores.get('semantic_distance')
        assert isinstance(fts_score, float)
        assert isinstance(semantic_distance, float)
