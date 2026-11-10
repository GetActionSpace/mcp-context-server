"""
Backend-specific tool description generators.

This module provides functions to generate dynamic tool descriptions
based on the storage backend type and language configuration.
Descriptions are generated at server initialization when the backend is known.
"""

from typing import Literal


def generate_fts_description(
    backend_type: Literal['sqlite', 'postgresql'],
    language: str,
) -> str:
    """Generate FTS tool description based on storage backend and language.

    The description format is UNIFORM across both backends with only values changing.
    This ensures AI agents can understand FTS behavior regardless of backend.

    Args:
        backend_type: Either 'sqlite' or 'postgresql'
        language: The configured FTS_LANGUAGE value (e.g., 'english', 'german', 'simple')

    Returns:
        Complete tool description string for the fts_search_context tool
    """
    if backend_type == 'sqlite':
        return _generate_sqlite_fts_description(language)
    return _generate_postgresql_fts_description(language)


def _generate_sqlite_fts_description(language: str) -> str:
    """Generate SQLite FTS5 description based on language configuration.

    SQLite FTS5 has limited language support:
    - Stemming ONLY works for English (Porter algorithm)
    - Stop words are NEVER removed
    - For all other languages, unicode61 tokenizer provides tokenization only

    Args:
        language: The configured FTS_LANGUAGE value

    Returns:
        Complete tool description string for SQLite FTS5
    """
    language_lower = language.lower()

    # SQLite: Porter stemmer ONLY works for English
    if language_lower == 'english':
        stemming_status = 'ENABLED'
        stemming_detail = 'for English text (Porter algorithm)'
        stemming_example = '"running" WILL match "run", "runs", "runner"'
    else:
        stemming_status = 'DISABLED'
        stemming_detail = '(Porter stemmer is English-only)'
        stemming_example = '"running" does NOT match "run" - exact token matching required'

    # SQLite: Stop words are NEVER removed
    stopwords_status = 'DISABLED'
    stopwords_detail = 'FTS5 does not remove stop words'
    stopwords_example = '"the", "is", "a" MUST match literally in queries'

    # Non-language text behavior
    non_language_behavior = 'Tokenized by unicode61 but NOT stemmed'

    # bm25() reads corpus statistics (row count, average length, per-term document
    # counts) from the whole FTS5 index, so entries outside the caller's scope
    # still weigh on the scores of the entries it may read.
    fts_score_scope = (
        'The BM25 score draws on statistics of the whole FTS5 index, entries the caller cannot read '
        "included, so the scores of readable entries and their relative order can shift as other principals' "
        'entries change, while the set of matching entries and the number of results per page depend only '
        'on what the caller can read.'
    )

    return _FTS_DESCRIPTION_TEMPLATE.format(
        backend='SQLite',
        engine='FTS5',
        language=language,
        stemming_status=stemming_status,
        stemming_detail=stemming_detail,
        stemming_example=stemming_example,
        stopwords_status=stopwords_status,
        stopwords_detail=stopwords_detail,
        stopwords_example=stopwords_example,
        non_language_behavior=non_language_behavior,
        mode_descriptions=_SQLITE_MODE_DESCRIPTIONS,
        fts_score_scope=fts_score_scope,
    )


def _generate_postgresql_fts_description(language: str) -> str:
    """Generate PostgreSQL tsvector description based on language configuration.

    PostgreSQL has full language support with Snowball stemmers:
    - Stemming works for all 28 non-simple languages (Snowball algorithm)
    - Stop words are removed for all non-simple languages
    - 'simple' configuration: no stemming, no stop words

    Args:
        language: The configured FTS_LANGUAGE value

    Returns:
        Complete tool description string for PostgreSQL tsvector
    """
    language_lower = language.lower()

    # PostgreSQL: Snowball stemmer works for all languages except 'simple'
    if language_lower == 'simple':
        stemming_status = 'DISABLED'
        stemming_detail = '(simple configuration - tokenization only)'
        stemming_example = '"running" does NOT match "run" - exact token matching required'
        stopwords_status = 'DISABLED'
        stopwords_detail = 'simple configuration - no stop word list'
        stopwords_example = '"the", "is", "a" MUST match literally in queries'
    else:
        stemming_status = 'ENABLED'
        stemming_detail = f'for {language} text (Snowball algorithm)'
        stemming_example = '"running" WILL match "run", "runs", "runner"'
        stopwords_status = 'ENABLED'
        stopwords_detail = f'for {language} (language-specific list)'
        stopwords_example = '"the", "is", "a" are REMOVED and ignored in searches'

    # Non-language text behavior
    non_language_behavior = 'Tokenized on spaces but NOT stemmed (CJK languages require extensions)'

    # ts_rank_cd() reads only the scored entry's own tsvector.
    fts_score_scope = 'ts_rank_cd scores each entry on its own, so a score depends only on the entry and the query.'

    return _FTS_DESCRIPTION_TEMPLATE.format(
        backend='PostgreSQL',
        engine='tsvector',
        language=language,
        stemming_status=stemming_status,
        stemming_detail=stemming_detail,
        stemming_example=stemming_example,
        stopwords_status=stopwords_status,
        stopwords_detail=stopwords_detail,
        stopwords_example=stopwords_example,
        non_language_behavior=non_language_behavior,
        mode_descriptions=_POSTGRESQL_MODE_DESCRIPTIONS,
        fts_score_scope=fts_score_scope,
    )


# Uniform template for both backends - structure is IDENTICAL, only values change
_FTS_DESCRIPTION_TEMPLATE = '''Full-text search using {engine} ({backend} backend).

FTS provides keyword-based search with relevance scoring.

IMPORTANT - {backend} {engine} characteristics (FTS_LANGUAGE={language}):
- Stemming: {stemming_status} {stemming_detail}
  {stemming_example}
- Stop words: {stopwords_status} ({stopwords_detail})
  {stopwords_example}
- Case-insensitive matching
- Non-{language} text: {non_language_behavior}

{mode_descriptions}

Filtering options (all combinable):
- thread_id/source: Basic entry filtering
- content_type: Filter by text or multimodal entries
- tags: OR logic (matches ANY of provided tags)
- start_date/end_date: Date range filtering (ISO 8601)
- metadata: Simple key=value equality matching
- metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)

Only entries the caller may read are matched, returned and counted.

The `scores` object contains:
- fts_score: BM25/ts_rank relevance (HIGHER = better match). {fts_score_scope}
- fts_rank: Always null for standalone FTS search
- rerank_score: Cross-encoder relevance (HIGHER = better), present when reranking enabled

Pagination:
- A query has ONE ranking, at most 100 rows deep; limit and offset select a window
  inside it, so paging never repeats a row on two pages nor skips one entirely.
- A window reaching past that depth is served short (empty when the offset itself is
  past it) and the response carries rank_depth_limit.

Returns:
    Dict with query (str), mode (str), results (list with id, thread_id, source,
    text_content (truncated), summary, is_text_content_truncated,
    metadata, scores, highlighted, tags, visibility, is_owner (whether the
    caller owns the entry), generation_status ('ready' once the entry's
    embeddings and summaries are generated, 'pending' while their generation
    waits or runs, 'failed' when it gave up after its retries; the full-text
    index covers an entry as soon as it is stored)), count (int), language (str),
    stats (only when explain_query=True), and rank_depth_limit (only when the
    requested window reaches past the ranked depth).

Raises:
    ToolError: If FTS is not available or search operation fails.'''


# SQLite-specific mode descriptions
_SQLITE_MODE_DESCRIPTIONS = '''Search modes:

match (default): Natural language query. Words joined with implicit AND.
  Example: "python async" finds docs containing BOTH exact words.

prefix: Wildcard search with * suffix. Matches word beginnings.
  Example: "search*" matches "searching", "searched", "searchable"
  Note: Hyphenated words are split: "full-text" becomes "full* AND text*"

phrase: Exact phrase matching. Words must appear in exact order.
  Example: "exact phrase" matches only "exact phrase", not "phrase exact"
  Note: All words including stop words must be present.

boolean: Boolean operators for precise control.
  Syntax: AND, OR, NOT (UPPERCASE required)
  Example: "python AND (async OR await) NOT blocking"
  Grouping: Full parentheses support
  Note: Use double quotes for phrases within boolean: '"error handling" AND python'
  Note: Malformed syntax (e.g. unbalanced parentheses) degrades to a best-effort term match rather than raising an error.'''


# PostgreSQL-specific mode descriptions
_POSTGRESQL_MODE_DESCRIPTIONS = '''Search modes:

match (default): Natural language query with stemming. Words joined with implicit AND.
  Example: "running tests" finds docs with "run", "test" (stemmed forms).

prefix: Wildcard search with * suffix. Matches word beginnings after stemming.
  Example: "search*" matches "searching", "searched" (also "search" via stem)
  Note: Hyphenated words are split: "full-text" becomes "full* AND text*"

phrase: Phrase matching with stemming and stop word handling.
  Example: "running tests" matches "run test" (stemmed forms in sequence)
  Note: Stop words are replaced with positional distance operators.
  "over the rainbow" matches "over ... rainbow" with correct distance.

boolean: Boolean operators with websearch syntax.
  Syntax:
    - space = AND (implicit, no "AND" keyword exists)
    - "or" = OR operator (case-insensitive: "or", "OR", "Or" all work)
    - "-" = NOT operator (minus prefix, no "NOT" keyword exists)
  Example: "python (async or await) -blocking"
  Grouping: Parentheses for grouping
  IMPORTANT:
    - There is NO "AND" or "NOT" keyword. Use space for AND, "-" for NOT.
    - Words "and"/"not" typed as terms are removed as stop words under English (and configs whose
      ASCII words use the English stop list); other languages keep them as ordinary terms.
  Use double quotes for phrases: '"error handling" python\''''
