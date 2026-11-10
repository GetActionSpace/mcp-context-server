"""Pure full-text query handling shared by the FTS search paths and their callers.

Sanitizes client tokens into safe FTS5 literals, validates that a query can be
bound on both backends, maps ``FTS_LANGUAGE`` to the SQLite FTS5 tokenizer, and
turns a query into its per-backend form for each search mode.
"""

import re
from typing import Literal

from app.metadata_types import pg_bind_reject_reason

# Tokenizer for FTS5 sanitization: a "quoted phrase" OR a run of non-whitespace.
_FTS_TOKEN_RE = re.compile(r'"[^"]*"|\S+')


# PostgreSQL text-search configs whose ``asciiword`` tokens route through the ``english_stem``
# dictionary, so ``plainto_tsquery`` drops the English operator barewords and/or/not as stopwords.
# The SQLite sanitizer mirrors that drop ONLY for these languages to preserve cross-backend recall
# parity; for EVERY other configured language PostgreSQL keeps and/or/not as ordinary lexemes, so
# SQLite must keep them too (as literal terms) or the two backends return different result sets.
# Verified against PostgreSQL 18: ``plainto_tsquery(cfg, 'and or not')`` is empty ONLY for english,
# hindi, and russian; all other configs (including 'simple') return the three lexemes.
ENGLISH_STOPWORD_FTS_LANGUAGES = frozenset({'english', 'hindi', 'russian'})


def sanitize_sqlite_fts_terms(tokens: list[str], language: str = 'english') -> list[str]:
    """Sanitize query tokens into crash-safe SQLite FTS5 terms.

    SQLite FTS5 MATCH treats AND/OR/NOT/NEAR as case-sensitive operators and rejects bare
    special characters, so an unsanitized token can raise ``fts5: syntax error`` in match,
    prefix, or boolean-OR contexts. Each token is made safe and aligned with PostgreSQL's
    plainto_tsquery for the CONFIGURED language (which ANDs the remaining lexemes and, for a
    language whose ASCII words route through ``english_stem``, drops the operator stopwords):

    - A quoted-phrase token (``"..."``) is preserved as-is.
    - An operator bareword (and/or/not, case-insensitive) is DROPPED only when ``language`` is one
      whose PostgreSQL config treats them as stopwords (english/hindi/russian -- see
      ``ENGLISH_STOPWORD_FTS_LANGUAGES``); for every other language it is KEPT as a literal term,
      because PostgreSQL keeps it as a required lexeme and dropping it on SQLite alone would make
      the two backends return different rows.
    - A bare term carrying an embedded double quote is SPLIT on that quote into one FTS5 string
      literal per fragment (``cat"dog`` -> ``"cat" "dog"``), which FTS5 ANDs. Escaping the quote by
      doubling it instead (``"cat""dog"``) does NOT neutralize it: FTS5 re-tokenizes the content of
      a string literal, the escape decodes back to a literal ``"``, and the tokenizer treats it as
      a word boundary -- silently turning an ordinary token into a strict two-word ADJACENCY
      phrase. PostgreSQL's plainto_tsquery splits the same input into independently-ANDed lexemes
      with no adjacency requirement, so only the split form keeps the backends returning the same
      rows.
    - A HYPHEN inside a bare term stays a space inside ONE literal (``full-text`` ->
      ``"full text"``, an adjacency phrase) on purpose: PostgreSQL emits the compound lexeme
      ``full-text`` plus its parts, which requires the two words to be adjacent in the document.
      FTS5's unicode61/porter tokenizer drops hyphens entirely, so the document side cannot
      distinguish ``full-text`` from ``full text`` at all; the adjacency phrase is the closest
      expressible approximation, and ANDing the parts separately would widen recall further.
      This rule is scoped to the modes THIS function serves (match and the hybrid OR join),
      where PostgreSQL's compound lexeme is what sets the adjacency target. PREFIX mode has a
      different target -- ``_handle_hyphenated_prefix_postgresql`` emits AND-ed prefix lexemes
      (``full:* & text:*``) with no adjacency requirement -- so the prefix branch of
      ``transform_query_sqlite`` splits hyphens into separate wildcarded literals instead.
    - Every other bare term is wrapped as an FTS5 string literal, so an operator-cased NEAR or a
      special char ( ( ) : * ^ ) is a LITERAL term, never syntax. A quoted single word is still
      tokenized/stemmed, so ordinary-word recall is unchanged.

    Shared by the standalone FTS transform (``transform_query_sqlite`` match/prefix modes) and
    the hybrid adaptive query builder so the two never diverge.

    Args:
        tokens: Raw whitespace/quote tokens from the query.
        language: The configured FTS_LANGUAGE, deciding whether and/or/not are dropped as
            stopwords (to mirror PostgreSQL's plainto_tsquery for that language).

    Returns:
        The list of safe FTS5 term strings (operator barewords removed only for the
        stopword-dropping languages).
    """
    drop_operator_barewords = language.lower() in ENGLISH_STOPWORD_FTS_LANGUAGES
    terms: list[str] = []
    for token in tokens:
        # Require >= 2 chars: a LONE '"' satisfies both startswith AND endswith (they test the
        # SAME character), so without the length check it would pass through as a "balanced
        # phrase" and produce an unterminated FTS5 string literal (MATCH syntax error). With the
        # check it falls through to the splitting path below, which yields no fragment at all.
        if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
            terms.append(token)
            continue
        for fragment in token.replace('-', ' ').split('"'):
            clean = fragment.strip()
            if not clean:
                continue
            if drop_operator_barewords and clean.lower() in ('and', 'or', 'not'):
                continue
            terms.append(f'"{clean}"')
    return terms


def fts_query_validation_errors(query: str) -> list[str] | None:
    """Return the validation messages for a query the FTS path cannot bind, else None.

    An embedded NUL (U+0000) and an unpaired UTF-16 surrogate both pass JSON and Pydantic
    validation yet are fatal on the query path: FTS5's MATCH parser reads the bound query
    as a NUL-terminated C string (truncating a quoted literal into an unterminated-string
    grammar error that quoting cannot neutralize), and asyncpg rejects any NUL-carrying or
    non-UTF-8-encodable text bind on PostgreSQL. The shared ``pg_bind_reject_reason`` probe
    catches both sequences, including the lone surrogate a bare ``\\x00`` scan misses.

    Kept as a standalone predicate so the repository's own guard and any caller that wants
    to decide the same thing without running a search share one wording and one rule.

    Args:
        query: The client-supplied search query.

    Returns:
        The validation messages, or None when the query is bindable on both backends.
    """
    reason = pg_bind_reject_reason(query)
    if reason is None:
        return None
    return [f'Query contains {reason}, which full-text search cannot parse']


def desired_sqlite_fts_tokenizer(language: str) -> str:
    """Map an FTS_LANGUAGE setting to the SQLite FTS5 tokenizer.

    The single source of truth for the language->tokenizer rule, shared by the FTS
    migration (server startup), FtsRepository.get_desired_tokenizer (the rebuild check),
    and the migration CLI's SQLite target initialization, so they cannot drift:
    English benefits from the Porter stemmer; other languages use plain unicode61
    (proper Unicode tokenization, no English stemming).

    Args:
        language: The FTS_LANGUAGE setting value.

    Returns:
        The FTS5 tokenizer string ('porter unicode61' for English, else 'unicode61').
    """
    if language.lower() == 'english':
        return 'porter unicode61'
    return 'unicode61'


def _escape_double_quotes(text: str) -> str:
    """Escape double quotes for FTS5 phrase literals.

    FTS5 requires double quotes to be escaped by doubling them.

    Args:
        text: Text that may contain double quotes

    Returns:
        Text with double quotes escaped as ""
    """
    return text.replace('"', '""')


def _handle_hyphenated_prefix_postgresql(word: str) -> str:
    """Handle hyphenated words for PostgreSQL prefix mode.

    Splits hyphenated words into AND-ed prefix terms.
    "full-text" -> "full:* & text:*"

    Args:
        word: Single word that may contain hyphens

    Returns:
        PostgreSQL prefix query fragment
    """
    # to_tsquery() is STRICT: a bare tsquery operator (&, |, !, parens), a stray ':'
    # or '*', or an unterminated quote raises "syntax error in tsquery" -- unlike
    # plainto/phraseto/websearch_to_tsquery, which never raise. Extract only word
    # characters (Unicode-aware) and emit each as an AND-ed prefix lexeme 'sub:*', so
    # an adversarial token ('cat(', 'foo:bar', '"x') degrades to safe literal
    # prefixes. A hyphenated/punctuated word splits into AND-ed prefixes
    # ("full-text" -> "full:* & text:*"); a token with no word characters yields ''
    # (dropped by the caller). A user trailing '*'/':' is naturally excluded.
    parts = re.findall(r'\w+', word)
    return ' & '.join(f'{part}:*' for part in parts)


def transform_query_sqlite(
    query: str,
    mode: Literal['match', 'prefix', 'phrase', 'boolean'],
    language: str = 'english',
) -> str:
    """Transform query string for SQLite FTS5 based on mode.

    Args:
        query: Original search query
        mode: Search mode
        language: Configured FTS_LANGUAGE; governs whether and/or/not operator barewords are
            dropped as stopwords (to mirror PostgreSQL's plainto_tsquery for that language).

    Returns:
        Transformed query for FTS5 MATCH
    """
    # Clean the query
    query = query.strip()
    # Drop the English operator barewords and/or/not ONLY for a language whose PostgreSQL
    # config treats them as stopwords; otherwise keep them as literal terms (see
    # sanitize_sqlite_fts_terms / ENGLISH_STOPWORD_FTS_LANGUAGES) so SQLite and PostgreSQL
    # return the same rows.
    drop_operator_barewords = language.lower() in ENGLISH_STOPWORD_FTS_LANGUAGES

    if mode == 'phrase':
        # Exact phrase matching - wrap in double quotes
        # Escape any existing double quotes first
        escaped = _escape_double_quotes(query)
        return f'"{escaped}"'

    if mode == 'prefix':
        # Prefix (autocomplete): wrap each token as a safe FTS5 string literal, then add the
        # prefix wildcard ( "term"* matches terms starting with the token). Operator barewords
        # are dropped and special chars neutralized (so 'cat(' / 'foo:bar' are literal
        # prefixes, never an FTS5 syntax error); a user-supplied trailing '*' is stripped so
        # it is not doubled. A quoted-phrase token keeps its phrase and gets the wildcard.
        # An embedded double quote SPLITS the token into separate wildcarded literals, exactly
        # like sanitize_sqlite_fts_terms and PostgreSQL's AND-ed prefix lexemes ('cat:* &
        # dog:*'): doubling the quote instead leaves it a word boundary inside the literal,
        # which FTS5 reads as an adjacency phrase (see sanitize_sqlite_fts_terms).
        # A HYPHEN splits the same way HERE, unlike the match-mode sanitizer: PostgreSQL's
        # prefix transform emits AND-ed prefix lexemes ('full:* & text:*') with no adjacency
        # requirement, so keeping the parts in one literal ('"full text"*') would make SQLite
        # demand adjacency for a query PostgreSQL answers without it -- a divergence that has
        # no match-mode counterpart, where PostgreSQL's own compound lexeme does require it.
        prefix_terms: list[str] = []
        for token in _FTS_TOKEN_RE.findall(query):
            # >= 2 chars so a lone '"' is not mistaken for a balanced phrase (see
            # sanitize_sqlite_fts_terms) -- otherwise it yields an unterminated FTS5 string.
            if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
                prefix_terms.append(f'{token}*')
                continue
            for fragment in token.rstrip('*').split('"'):
                for word in fragment.replace('-', ' ').split():
                    if drop_operator_barewords and word.lower() in ('and', 'or', 'not'):
                        continue
                    prefix_terms.append(f'"{word}"*')
        if not prefix_terms:
            # Every token was an operator bareword: match NOTHING (empty result), in
            # parity with PostgreSQL's empty to_tsquery for the same input. '' is the
            # caller's "match nothing" sentinel (see _search_sqlite). The earlier
            # literal-phrase fallback ('"and"') was WRONG: FTS5's porter/unicode61
            # tokenizer keeps stopwords as tokens, so it matched every document containing
            # the dropped word while PostgreSQL returned zero -- a cross-backend divergence.
            return ''
        return ' '.join(prefix_terms)

    if mode == 'boolean':
        # Boolean mode passes the query through UNCHANGED so the user's native FTS5 boolean
        # syntax (AND/OR/NOT uppercase, parentheses, quoted phrases) reaches MATCH intact --
        # the one SQLite mode that is NOT pre-sanitized here. A malformed boolean query would
        # make FTS5 raise a grammar error; _search_sqlite catches that at execution and
        # degrades to the safe sanitized term match, so a valid query is unaffected while a
        # malformed one returns best-effort results instead of erroring (parity with
        # PostgreSQL's tolerant websearch_to_tsquery). The user is responsible for quoting
        # hyphenated words.
        return query

    # 'match' - default (AND logic). Sanitize each token to a safe FTS5 string literal so a
    # bare FTS5 operator (AND/OR/NOT, case-sensitive) or special char ((): "*^) becomes a
    # LITERAL term -- never 'fts5: syntax error'. This matches PostgreSQL's plainto_tsquery
    # (drops operator stopwords for the stopword-dropping languages, keeps them as literal
    # terms otherwise, and ANDs the remaining lexemes); a quoted single word is still stemmed,
    # so ordinary-word recall is unchanged. Shared with the hybrid query builder.
    terms = sanitize_sqlite_fts_terms(_FTS_TOKEN_RE.findall(query), language)
    if not terms:
        # Every token was an operator/stopword bareword: match NOTHING (empty result), in
        # parity with PostgreSQL's empty plainto_tsquery for the same input. '' is the
        # caller's "match nothing" sentinel (see _search_sqlite). A literal-phrase fallback
        # ('"and"') would instead MATCH every document containing the dropped word (FTS5
        # porter/unicode61 keeps stopwords as tokens), diverging from PostgreSQL.
        return ''
    return ' '.join(terms)


def transform_query_postgresql(
    query: str,
    mode: Literal['match', 'prefix', 'phrase', 'boolean'],
) -> str:
    """Transform query string for PostgreSQL tsquery based on mode.

    For prefix mode, transforms "hello world" to "hello:* & world:*"
    to work correctly with to_tsquery().

    Args:
        query: Original search query
        mode: Search mode

    Returns:
        Transformed query for PostgreSQL tsquery
    """
    # Clean the query
    query = query.strip()

    if mode == 'prefix':
        # Prefix matching: sanitize each whitespace token into safe AND-ed prefix
        # lexemes for the STRICT to_tsquery (the SQLite prefix path is already
        # sanitized; this keeps PostgreSQL from RAISING on adversarial input rather
        # than degrading gracefully). Drop tokens that sanitize to nothing; an
        # all-punctuation query yields '' -- an empty tsquery that simply matches
        # nothing, never a syntax error.
        words = query.split()
        prefix_terms = [frag for word in words if (frag := _handle_hyphenated_prefix_postgresql(word))]
        return ' & '.join(prefix_terms)

    # For other modes, return query as-is
    # - match: plainto_tsquery discards punctuation (safe)
    # - phrase: phraseto_tsquery discards punctuation (safe)
    # - boolean: websearch_to_tsquery treats - as NOT (by design)
    return query


def get_tsquery_function(
    mode: Literal['match', 'prefix', 'phrase', 'boolean'],
    language: str,
) -> str:
    """Get the appropriate PostgreSQL tsquery function for the search mode.

    Args:
        mode: Search mode
        language: Language for text search

    Returns:
        SQL function call string for tsquery generation
    """
    if mode == 'phrase':
        return f"phraseto_tsquery('{language}', "
    if mode == 'prefix':
        # For prefix, we use to_tsquery which supports :* for prefix
        return f"to_tsquery('{language}', "
    if mode == 'boolean':
        # websearch supports Google-like syntax with OR, -, quotes
        return f"websearch_to_tsquery('{language}', "
    # 'match' - default
    # plainto_tsquery handles natural language input
    return f"plainto_tsquery('{language}', "
