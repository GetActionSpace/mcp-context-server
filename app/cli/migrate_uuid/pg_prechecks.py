"""Pre-flight checks for values a PostgreSQL target cannot store or index.

A source row whose text or JSON PostgreSQL rejects, or whose indexed column
exceeds the B-tree item budget, is reported before the INSERT would abort the run.
"""

import json
from collections.abc import Iterable
from typing import cast

from app.metadata_types import PG_INDEX_TEXT_CASTS
from app.metadata_types import non_finite_metadata_error
from app.metadata_types import pg_indexed_cast_error
from app.metadata_types import pg_indexed_metadata_text
from app.metadata_types import unstorable_string_error


def pg_unstorable_column_reason(value: str | None, *, is_jsonb: bool) -> str | None:
    """Return why a SQLite value cannot be stored on the PostgreSQL target, else None.

    An embedded NUL (U+0000) or unpaired UTF-16 surrogate is legal in a SQLite
    TEXT value but fatal on PostgreSQL: asyncpg rejects a NUL text bind
    (``CharacterNotInRepertoireError``, SQLSTATE 22021) and an unpaired surrogate
    is not UTF-8-encodable at all. A ``jsonb`` column has a second failure mode --
    the store path serializes a metadata NUL as the JSON escape ``\\u0000`` (six
    ASCII characters, not a literal byte), which passes the raw-string check yet
    is rejected by PostgreSQL's jsonb parser (SQLSTATE 22P05). A ``jsonb`` column
    has a THIRD failure mode: a non-finite JSON number. ``json.loads`` accepts the
    non-standard tokens ``NaN``/``Infinity``/``-Infinity`` AND silently converts a
    standard-but-overflowing literal such as ``1e400`` into ``inf``, while
    ``json.dumps`` -- which :meth:`MetadataIdRewriter.rewrite` applies to every
    parseable ``metadata`` value on the copy path -- re-emits those values as the
    invalid tokens the jsonb parser rejects. Such a value is also unstorable
    through the server's own store boundary, which runs the same
    :func:`app.metadata_types.non_finite_metadata_error` guard, so a v3 target must
    not receive it either way.

    Accordingly this runs the shared
    :func:`app.metadata_types.unstorable_string_error` walker on the raw
    serialized string (catching a literal NUL/surrogate in the bind) and, for a
    ``jsonb`` column, ALSO on the decoded structure (``json.loads`` restores the
    ``\\u0000`` escape to a real U+0000 the walker detects) plus the shared
    non-finite-number guard on that same decoded structure. For a ``jsonb`` column
    a value ``json.loads`` cannot parse is itself unstorable: the target binds it
    through a ``$n::jsonb`` cast
    that PostgreSQL rejects mid-transaction (SQLSTATE 22P02), aborting the whole
    migration with no row identification -- the exact failure this pre-check
    exists to convert into a skip-and-warn -- so malformed JSON returns a reason
    rather than passing as storable. (A raw TEXT column has no such cast, so
    unparseable content is not rejected there.)

    Args:
        value: The raw SQLite column value (already serialized to a JSON string
            for a ``jsonb`` column), or ``None``.
        is_jsonb: True when ``value`` is bound into a PostgreSQL ``jsonb`` column,
            enabling the decoded-escape and non-finite-number checks in addition
            to the raw-string check.

    Returns:
        The shared guard's operator-facing message for the first offending
        sequence or value found, else None.
    """
    if value is None:
        return None
    raw_reason = unstorable_string_error(value)
    if raw_reason is not None:
        return raw_reason
    if not is_jsonb:
        return None
    try:
        decoded: object = json.loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        return (
            f'value is not valid JSON, so the target rejects the jsonb bind '
            f'mid-transaction (SQLSTATE 22P02): {exc}. SQLite stores the same '
            f'malformed metadata verbatim.'
        )
    string_reason = unstorable_string_error(decoded)
    if string_reason is not None:
        return string_reason
    return non_finite_metadata_error(decoded)


# PostgreSQL refuses to index a btree tuple larger than roughly a third of an 8KB
# page: BTMaxItemSize is 2704 bytes on btree version 4.
#
# For thread_id and tags this matters only for a SQLite SOURCE: SQLite indexes those
# columns with no size limit, while every PostgreSQL database declares idx_thread_id
# and idx_tags_tag in its base schema and therefore cannot already hold a value that
# breaches the ceiling. An INDEXED METADATA value is different, and a PostgreSQL
# source can hold one: the metadata expression indexes are deliberately NOT in the
# base schema (see app/schemas/postgresql_schema.sql -- a database initialized only by
# this CLI has none until its first server startup), so a source whose
# METADATA_INDEXED_FIELDS never covered a field, or which was never started as a
# server, can carry a value the target's index rejects.
#
# Bound at the source, such a value aborts the INSERT mid-transaction, ROLLBACKs the
# entire run, and reports a raw driver error naming no source row -- the exact failure
# the NUL/surrogate pre-check exists to convert into a per-row skip-and-warn.
#
# The budget is deliberately measured against the UNCOMPRESSED value. PostgreSQL
# compresses an index attribute larger than 512 bytes in line, so a highly repetitive
# oversized value can still fit while an incompressible one of the same length cannot.
# Modeling that compression is not possible from here, and the two errors are not
# symmetric: over-accepting costs the WHOLE run, while over-skipping costs one row that
# is named in the errors and migrates on a rerun once the value is shortened.
PG_BTREE_MAX_ITEM_BYTES = 2704


# What an index tuple costs BESIDES the payload of the value being checked, for any
# index shape:
#   16 bytes  IndexTupleData header (8 bytes), MAXALIGNed to 16 once a nullable
#             trailing column adds the null bitmap
#    4 bytes  the long varlena header carried by a text datum past the 126-byte
#             short-header threshold
#    8 bytes  one MAXALIGN quantum of slack on the assembled tuple, covering the
#             inter-attribute padding a fixed-width trailing column can introduce
_PG_INDEX_TUPLE_FIXED_BYTES = 16 + 4 + 8


# Budget for a value that is indexed ON ITS OWN: idx_tags_tag on ``tags(tag)`` and the
# metadata expression indexes ``idx_metadata_<field>`` on
# ``context_entries((metadata->>'<field>'))`` that handle_metadata_indexes provisions
# for every string-typed METADATA_INDEXED_FIELDS entry.
PG_MAX_INDEXED_VALUE_BYTES = PG_BTREE_MAX_ITEM_BYTES - _PG_INDEX_TUPLE_FIXED_BYTES


# thread_id needs a SMALLER budget because it is the leading column of
# idx_context_entries_dedup_hash (thread_id, source, content_hash), which the base
# schema declares on every PostgreSQL target: the tuple must also hold 6 bytes of
# source ('agent' as a short-header varlena) and 65 bytes of content_hash (a 64-character
# SHA-256 hex string, likewise short-header). The other compound indexes thread_id feeds
# (idx_thread_source, idx_thread_created) have narrower trailing columns than that, so
# the dedup index sets the ceiling.
PG_MAX_INDEXED_THREAD_ID_BYTES = PG_MAX_INDEXED_VALUE_BYTES - (6 + 65)


def pg_unindexable_column_reason(value: str | None, max_bytes: int) -> str | None:
    """Return why a value is too large for a PostgreSQL btree index, else None.

    Args:
        value: The candidate value for an INDEXED target column.
        max_bytes: Payload budget for the widest index this value feeds --
            :data:`PG_MAX_INDEXED_THREAD_ID_BYTES` for thread_id (a compound index
            whose trailing columns share the tuple), :data:`PG_MAX_INDEXED_VALUE_BYTES`
            for a value indexed on its own.

    Returns:
        A reason string when the encoded value exceeds the index-tuple budget,
        else None.
    """
    if value is None:
        return None
    encoded_bytes = len(value.encode('utf-8'))
    if encoded_bytes <= max_bytes:
        return None
    return (
        f'the value is {encoded_bytes} UTF-8 bytes, which exceeds the PostgreSQL btree '
        f'index-tuple budget of {max_bytes} bytes for this indexed '
        f'column; SQLite indexes it without a size limit, so this row is skipped -- shorten '
        f'the value in the source database and rerun to migrate it'
    )


def first_pg_unindexable_column(
    columns: Iterable[tuple[str, str | None, int]],
) -> tuple[str, str] | None:
    """Return the first ``(column, reason)`` a PostgreSQL btree index cannot hold, else None.

    Args:
        columns: Ordered ``(column_name, value, max_bytes)`` candidates for one row,
            limited to columns the target schema actually indexes. ``max_bytes`` is the
            payload budget of the widest index that column feeds.

    Returns:
        The ``(column_name, reason)`` of the first oversized column, else None.
    """
    for name, value, max_bytes in columns:
        reason = pg_unindexable_column_reason(value, max_bytes)
        if reason is not None:
            return name, reason
    return None


def first_pg_unindexable_metadata_field(metadata_json: str | None) -> tuple[str, str] | None:
    """Return the first indexed metadata field whose value a PostgreSQL target refuses, else None.

    A ``METADATA_INDEXED_FIELDS`` key gets an expression btree index
    ``idx_metadata_<field>`` on ``context_entries`` (app.migrations.metadata),
    evaluated on every INSERT. SQLite's equivalent ``json_extract`` index has neither
    a size limit nor a cast, so a source can hold a value the target cannot index at
    all -- aborting the whole run when the target already carries the index, or
    breaking the target's first server startup (which creates the index) when the CLI
    initialized the target itself -- or a value of another type than the field
    declares, which the running server refuses to store. Each is checked, in the
    order it would fail:

    * WIDTH, for a ``string``-typed field, whose TEXT btree entry is bounded by the
      index-tuple ceiling that also bounds thread_id and tags. The width is measured on
      the text the expression YIELDS (:func:`~app.metadata_types.pg_indexed_metadata_text`),
      so a list or object is measured as the whole serialized JSON ``->>`` renders it as.
    * DECLARED TYPE, for an ``integer``/``boolean``/``float``-typed field, whose value
      must parse as that type. This is the same check the write boundary applies
      (:func:`~app.metadata_types.pg_indexed_cast_error`), so a value the running server
      would refuse to store is a value the migration refuses to import -- SQLite happily
      holds ``{"priority": "high"}`` under an integer-typed field. For ``integer`` and
      ``boolean`` the target's index expression also carries a hard SQL cast the value
      would fail on INSERT, and the skip reason says so; the ``float`` index casts JSON
      numbers only, so its skip reason names the declared type.

    ``array``/``object``-typed fields are exempt from both: they build no expression
    index at all, being served by the always-present jsonb_path_ops GIN index, which
    hashes its entries. Only top-level keys are inspected, because
    ``metadata->>'<field>'`` addresses top-level keys only. Unparseable or non-object
    metadata returns None: the jsonb bind itself rejects it, which the unstorable
    pre-check reports with a more specific reason.

    Args:
        metadata_json: The metadata JSON string about to be bound into the target's
            ``jsonb`` column, or None when the row has no metadata.

    Returns:
        The ``('metadata.<field>', reason)`` of the first unindexable value, else None.
    """
    if metadata_json is None:
        return None
    try:
        parsed: object = json.loads(metadata_json)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    from app.settings import get_settings

    indexed_fields = get_settings().storage.metadata_indexed_fields
    for key, value in cast(dict[str, object], parsed).items():
        type_hint = indexed_fields.get(key)
        if type_hint is None:
            continue
        if type_hint == 'string':
            reason = pg_unindexable_column_reason(pg_indexed_metadata_text(value), PG_MAX_INDEXED_VALUE_BYTES)
            if reason is not None:
                return f'metadata.{key}', reason
            continue
        cast_error = pg_indexed_cast_error(key, value, type_hint)
        if cast_error is not None:
            cause = (
                'the PostgreSQL expression index evaluates that cast on every INSERT while '
                'SQLite indexes the value uncast'
                if type_hint in PG_INDEX_TEXT_CASTS
                else f'METADATA_INDEXED_FIELDS declares the field {type_hint}, and the server '
                f'refuses such a value on both backends'
            )
            return (
                f'metadata.{key}',
                (
                    f'{cast_error}; {cause}, so this row is skipped -- correct the value in the '
                    f'source database and rerun to migrate it'
                ),
            )
    return None


def first_pg_unstorable_column(
    columns: Iterable[tuple[str, str | None, bool]],
) -> tuple[str, str] | None:
    """Return the first ``(column, reason)`` a PostgreSQL target cannot store, else None.

    Each candidate is a ``(name, value, is_jsonb)`` triple. Columns are checked in
    the given order and the first offending one short-circuits, so a caller can
    identify the exact column that would abort the row's INSERT on PostgreSQL.

    Args:
        columns: Ordered ``(column_name, value, is_jsonb)`` candidates for one row.

    Returns:
        The ``(column_name, reason)`` of the first unstorable column, else None.
    """
    for name, value, is_jsonb in columns:
        reason = pg_unstorable_column_reason(value, is_jsonb=is_jsonb)
        if reason is not None:
            return name, reason
    return None
