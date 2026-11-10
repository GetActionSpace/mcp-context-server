"""Fusion algorithms for hybrid search.

This module provides implementations of result fusion algorithms
for combining results from multiple search methods (FTS, semantic).
"""


from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from app.types import HybridFusedResultDict
    from app.types import HybridScoresDict


def reciprocal_rank_fusion(
    fts_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    k: int = 60,
    limit: int = 50,
) -> 'list[HybridFusedResultDict]':
    """Combine FTS and semantic search results using Reciprocal Rank Fusion (RRF).

    RRF Formula: score(d) = sum(1 / (k + rank_i(d))) for each result list i

    Documents appearing in both result sets score higher due to the additive
    nature of the formula. The k parameter (default 60) controls how much
    emphasis is placed on top-ranked documents vs. lower-ranked ones.

    Every row of either list carries the entry's 'visibility' and 'is_owner', as every
    ranked search statement selects them; a row without them raises KeyError.

    Args:
        fts_results: Results from full-text search with 'id' and 'score' fields.
        semantic_results: Results from semantic search with 'id' and 'distance' fields.
        k: RRF smoothing constant (default 60). Higher values give more weight
           to lower-ranked documents. Standard values range from 10 to 100.
        limit: Maximum number of results to return.

    Returns:
        Combined results sorted by RRF score (descending), ties broken by ascending
        context id so the fused order is reproducible across executions, with scores
        breakdown.

    Example:
        >>> access = {'visibility': 'private', 'is_owner': 1}
        >>> fts = [{'id': '0190abcdef1234567890abcdef123401', 'score': 2.5, **access}]
        >>> semantic = [{'id': '0190abcdef1234567890abcdef123401', 'distance': 0.3, **access}]
        >>> results = reciprocal_rank_fusion(fts, semantic, k=60)
        >>> # The document appears in both sources, so it ranks higher
    """
    # Build document registry with scores from each source, keyed by the
    # 32-char lowercase hex UUIDv7 context id.
    doc_registry: dict[str, dict[str, Any]] = {}

    # Process FTS results (rank 1 = best score, highest relevance)
    for rank, result in enumerate(fts_results, start=1):
        doc_id = result.get('id')
        if doc_id is None:
            continue

        if doc_id not in doc_registry:
            doc_registry[doc_id] = {
                'data': result.copy(),
                'fts_rank': None,
                'semantic_rank': None,
                'fts_score': None,
                'semantic_distance': None,
                'rrf_score': 0.0,
            }

        doc_registry[doc_id]['fts_rank'] = rank
        doc_registry[doc_id]['fts_score'] = result.get('score')
        doc_registry[doc_id]['rrf_score'] += 1.0 / (k + rank)

    # Process semantic results (rank 1 = lowest distance, most similar)
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result.get('id')
        if doc_id is None:
            continue

        if doc_id not in doc_registry:
            doc_registry[doc_id] = {
                'data': result.copy(),
                'fts_rank': None,
                'semantic_rank': None,
                'fts_score': None,
                'semantic_distance': None,
                'rrf_score': 0.0,
            }

        # Update data if we have semantic result (may have richer info)
        if doc_registry[doc_id]['semantic_rank'] is None:
            # Merge semantic data into existing, preserving FTS data
            for key, value in result.items():
                if key not in doc_registry[doc_id]['data'] or doc_registry[doc_id]['data'].get(key) is None:
                    doc_registry[doc_id]['data'][key] = value

        doc_registry[doc_id]['semantic_rank'] = rank
        doc_registry[doc_id]['semantic_distance'] = result.get('distance')
        doc_registry[doc_id]['rrf_score'] += 1.0 / (k + rank)

    # Sort by RRF score (descending) with the context id as an explicit, unique
    # secondary key. RRF ties are common rather than exotic -- a document ranked 1
    # in the FTS leg only and a different document ranked 1 in the semantic leg only
    # both score 1/(k+1) -- and a score-only sort leaves their relative order at the
    # mercy of dict insertion order, which follows the two legs' (themselves
    # write-order sensitive) result orders. Without the tiebreak the same query run
    # twice can emit tied documents in different positions, so a client paging with
    # limit/offset silently skips or duplicates one.
    sorted_docs = [
        doc
        for _doc_id, doc in sorted(
            doc_registry.items(),
            key=lambda item: (-float(item[1]['rrf_score']), item[0]),
        )
    ][:limit]

    # Build result list with scores breakdown
    results: list[HybridFusedResultDict] = []
    for doc in sorted_docs:
        data = doc['data']

        # Build scores object (includes rerank_score when reranking is enabled)
        scores: HybridScoresDict = {
            'rrf': doc['rrf_score'],
            'fts_rank': doc['fts_rank'],
            'semantic_rank': doc['semantic_rank'],
            'fts_score': doc['fts_score'],
            'semantic_distance': doc['semantic_distance'],
            'rerank_score': None,  # Will be populated by apply_reranking (app/tools/search/ranking.py)
        }

        # Build result entry
        result_entry: HybridFusedResultDict = {
            'id': data.get('id'),
            'thread_id': data.get('thread_id', ''),
            'source': data.get('source', ''),
            'content_type': data.get('content_type', 'text'),
            'text_content': data.get('text_content', ''),
            'metadata': data.get('metadata'),
            'created_at': data.get('created_at', ''),
            'updated_at': data.get('updated_at', ''),
            'tags': data.get('tags', []),
            'scores': scores,
            'summary': data.get('summary'),  # Preserve for search display formatting
            'rerank_text': data.get('rerank_text'),  # Preserve for chunk-aware reranking
            # Every leg row carries the entry's access fields; search display formatting shapes them
            'visibility': data['visibility'],
            'is_owner': data['is_owner'],
        }
        results.append(result_entry)

    return results


def count_unique_results(
    fts_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Count unique and overlapping results between FTS and semantic search.

    Args:
        fts_results: Results from full-text search.
        semantic_results: Results from semantic search.

    Returns:
        Tuple of (fts_only_count, semantic_only_count, overlap_count).
    """
    fts_ids = {r.get('id') for r in fts_results if r.get('id') is not None}
    semantic_ids = {r.get('id') for r in semantic_results if r.get('id') is not None}

    overlap = fts_ids & semantic_ids
    fts_only = fts_ids - semantic_ids
    semantic_only = semantic_ids - fts_ids

    return len(fts_only), len(semantic_only), len(overlap)
