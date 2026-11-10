"""
Backend-specific tool description generators.

This module provides functions to generate dynamic tool descriptions
based on the storage backend type and language configuration.
Descriptions are generated at server initialization when the backend is known.
"""

from typing import Literal

from app.repositories.fts_repository.query import ENGLISH_STOPWORD_FTS_LANGUAGES

# The FTS_LANGUAGE values whose PostgreSQL text search dictionary carries a stop word list.
# Every other non-simple configuration removes no word of its own language from a query.
_POSTGRESQL_STOP_WORD_LANGUAGES = frozenset({
    'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian',
    'nepali', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish',
})


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
    - Stemming exists only under english (the porter tokenizer), where it runs on every word
    - FTS5 removes no stop words; the query builder drops the words and/or/not from match and
      prefix queries for the languages in ENGLISH_STOPWORD_FTS_LANGUAGES
    - For all other languages, unicode61 tokenizer provides tokenization only

    Args:
        language: The configured FTS_LANGUAGE value

    Returns:
        Complete tool description string for SQLite FTS5
    """
    language_lower = language.lower()

    # SQLite: the porter tokenizer is used for English only, and it stems every token it sees
    if language_lower == 'english':
        stemming_status = 'ENABLED'
        stemming_detail = 'for English text (Porter algorithm)'
        stemming_example = '"running" WILL match "run" and "runs", but not "runner"'
        non_language_behavior = 'stemmed by the same Porter rules, which run on every word: "maison" matches "maisons"'
    else:
        stemming_status = 'DISABLED'
        stemming_detail = '(Porter stemmer is English-only)'
        stemming_example = '"running" does NOT match "run" - exact token matching required'
        non_language_behavior = 'NOT stemmed either (unicode61 tokenization only)'

    # SQLite: FTS5 removes no stop words; the query builder drops and/or/not for some languages
    stopwords_status = 'DISABLED'
    stopwords_detail = 'FTS5 does not remove stop words'
    stopwords_example = '"the", "is", "a" MUST match literally in queries'
    if language_lower in ENGLISH_STOPWORD_FTS_LANGUAGES:
        stopwords_example += '; only "and", "or", "not" are dropped, from match and prefix queries'

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
    - Stemming works for all 28 non-simple languages (Snowball algorithm), on every word
    - Stop words are removed for the languages in _POSTGRESQL_STOP_WORD_LANGUAGES only
    - hindi and russian send ASCII words to the English stemmer and its stop word list
    - 'simple' configuration: no stemming, no stop words

    Args:
        language: The configured FTS_LANGUAGE value

    Returns:
        Complete tool description string for PostgreSQL tsvector
    """
    language_lower = language.lower()
    has_stop_list = language_lower in _POSTGRESQL_STOP_WORD_LANGUAGES
    ascii_words_are_english = language_lower != 'english' and language_lower in ENGLISH_STOPWORD_FTS_LANGUAGES

    # PostgreSQL: Snowball stemmer works for all languages except 'simple'
    if language_lower == 'simple':
        stemming_status = 'DISABLED'
        stemming_detail = '(simple configuration - tokenization only)'
        stemming_example = '"running" does NOT match "run" - exact token matching required'
        stopwords_status = 'DISABLED'
        stopwords_detail = 'simple configuration - no stop word list'
        stopwords_example = '"the", "is", "a" MUST match literally in queries'
        non_language_behavior = 'NOT stemmed either, like every word under the simple configuration'
    elif language_lower == 'english':
        stemming_status = 'ENABLED'
        stemming_detail = f'for {language} text (Snowball algorithm)'
        stemming_example = '"running" WILL match "run" and "runs", but not "runner"'
        stopwords_status = 'ENABLED'
        stopwords_detail = f'for {language} (language-specific list)'
        stopwords_example = '"the", "is", "a" are REMOVED and ignored in searches'
        non_language_behavior = (
            f'treated by the same {language} stemming rules and stop list: "maison" matches "maisons"'
        )
    else:
        # Every language has its own stems and its own stop word list, if PostgreSQL ships one:
        # the English examples do not hold for it.
        stemming_status = 'ENABLED'
        stemming_detail = f'for {language} text (Snowball algorithm)'
        stemming_example = f'Inflected forms of a {language} word match each other'
        if has_stop_list:
            stopwords_status = 'ENABLED'
            stopwords_detail = f'for {language} (language-specific list)'
            stopwords_example = f'The most common {language} words are REMOVED and ignored in searches'
            non_language_behavior = (
                f'treated by the same {language} stemming rules and stop list, whatever their language'
            )
        else:
            stopwords_status = 'DISABLED'
            stopwords_detail = f'PostgreSQL has no {language} stop word list'
            stopwords_example = f'No {language} word is ignored: a common word in a query must match like any other'
            non_language_behavior = f'stemmed by the same {language} rules, whatever their language'
        if ascii_words_are_english:
            # The configuration maps ASCII words to english_stem, which carries the English stop word list.
            if has_stop_list:
                stopwords_detail += '; the English list for ASCII words'
                stopwords_example += ', and so are "the", "is", "a"'
            else:
                stopwords_status = 'ENABLED'
                stopwords_detail = f'the English list, for ASCII words only: PostgreSQL has no {language} stop word list'
                stopwords_example = f'"the", "is", "a" are REMOVED and ignored in searches; no {language} word is ignored'
            non_language_behavior = (
                f'ASCII words follow the English stemming rules and stop list, all other words the {language} rules'
            )

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
- Words of other languages: {non_language_behavior}
- Text written without spaces between words (such as Chinese or Japanese) is not split into words:
  each unbroken run is indexed as one word

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
    metadata, scores, highlighted (set only when highlight=True), tags, visibility,
    is_owner (whether the caller owns the entry), generation_status ('ready' once the
    entry's embeddings and summaries are generated, 'pending' while their generation
    waits or runs, 'failed' when it gave up after its retries; the full-text
    index covers an entry as soon as it is stored)), count (int), language (str),
    stats (only when explain_query=True), and rank_depth_limit (only when the
    requested window reaches past the ranked depth).

Raises:
    ToolError: If FTS is not available or search operation fails.'''


# SQLite-specific mode descriptions
_SQLITE_MODE_DESCRIPTIONS = '''Search modes:

match (default): Natural language query. Words joined with implicit AND.
  Example: "python async" finds docs containing BOTH words (stemmed forms when Stemming is ENABLED).

prefix: Every word is matched as a word beginning, all words required; a trailing * is optional.
  Example: "search*" matches "searching", "searched", "searchable"
  Note: When Stemming is ENABLED, indexed words and the typed prefix are both stemmed first, so a prefix
  that is not a word stem can miss: "runn*" does not match "running" (indexed as "run").
  Note: Hyphenated words are split: "full-text" becomes "full* AND text*"

phrase: Phrase matching. Words must appear next to each other, in this order (stemmed forms when Stemming is ENABLED).
  Example: "exact phrase" matches "exact phrase", not "phrase exact"
  Note: All words including stop words must be present.

boolean: Boolean operators for precise control.
  Syntax: AND, OR, NOT (UPPERCASE required); NOT takes a term on each side: "python NOT blocking"
  Example: "python AND (async OR await) NOT blocking"
  Grouping: Full parentheses support
  Note: Use double quotes for phrases within boolean: '"error handling" AND python'
  Note: Malformed syntax (e.g. unbalanced parentheses) degrades to a best-effort term match rather than raising an error.'''


# PostgreSQL-specific mode descriptions
_POSTGRESQL_MODE_DESCRIPTIONS = '''Search modes (the examples show English text under FTS_LANGUAGE=english):

match (default): Natural language query. Words joined with implicit AND.
  Example: "running tests" finds docs with "run", "test" (stemmed forms, when Stemming is ENABLED).

prefix: Every word is matched as a word beginning, all words required; a trailing * is optional.
  Example: "search*" matches "searching", "searched", "searchable"
  Note: When Stemming is ENABLED, indexed words and the typed prefix are both stemmed first, so a prefix
  that is not a word stem can miss: "runn*" does not match "running" (indexed as "run").
  Note: Hyphenated words are split: "full-text" becomes "full* AND text*"

phrase: Phrase matching: the words in sequence, with the stemming and stop word handling above.
  Example: "running tests" matches "run test" (stemmed forms in sequence)
  Note: Stop words are replaced with positional distance operators.
  "lord of the rings" matches "lord ... rings" with correct distance.

boolean: Boolean operators with websearch syntax.
  Syntax:
    - space = AND (implicit, no "AND" keyword exists)
    - "or" = OR operator (case-insensitive: "or", "OR", "Or" all work)
    - "-" = NOT operator (minus prefix on one term, no "NOT" keyword exists)
  Example: "python async -blocking" (python AND async, without blocking); "async or await" (either word)
  Grouping: none. Parentheses are ignored, and "or" binds looser than AND and "-":
    "python async or python await" means (python AND async) OR (python AND await),
    so repeat a shared term on each side of "or".
  IMPORTANT:
    - There is NO "AND" or "NOT" keyword. Use space for AND, "-" for NOT.
    - Words "and"/"not" typed as terms are removed as stop words under English (and configs whose
      ASCII words use the English stop list); other languages keep them as ordinary terms.
  Use double quotes for phrases: '"error handling" python\''''
