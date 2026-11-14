"""Pre-flight checks for values a PostgreSQL target cannot store or that exceed the index budgets set here.

A source row whose text or JSON PostgreSQL rejects is reported before its INSERT
would abort the run; a row whose indexed column exceeds the byte budget set here is
reported and skipped too.
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
    TEXT value but fatal on PostgreSQL: PostgreSQL refuses a NUL in a text value,
    and an unpaired surrogate is not UTF-8-encodable at all. A
    ``jsonb`` column has a second failure mode -- the store path serializes a
    metadata NUL as the JSON escape ``\\u0000`` (six
    ASCII characters, not a literal byte), which passes the raw-string check yet
    is rejected by PostgreSQL's jsonb parser (SQLSTATE 22P05). A ``jsonb`` column
    has a THIRD failure mode: a non-finite JSON number. ``json.loads`` accepts the
    non-standard tokens ``NaN``/``Infinity``/``-Infinity`` AND silently converts a
    standard literal with a fraction or an exponent that is too large for a double,
    such as ``1e400``, into ``inf``.
    :meth:`~app.cli.migrate_uuid.metadata_ids.MetadataIdRewriter.rewrite` re-encodes
    with ``allow_nan=False`` and, when it cannot re-encode, keeps the source text
    verbatim, so the text bound on the copy path still holds the non-standard token,
    which the jsonb parser rejects, or the literal read as ``inf``, which the server's own
    store boundary refuses through the same
    :func:`app.metadata_types.non_finite_metadata_error` guard. This check catches both
    before the bind, so a v3 target receives neither.

    Accordingly this runs the shared
    :func:`app.metadata_types.unstorable_string_error` walker on the raw
    serialized string (catching a literal NUL/surrogate in the bind) and, for a
    ``jsonb`` column, ALSO on the decoded structure (``json.loads`` restores the
    ``\\u0000`` escape to a real U+0000 the walker detects) plus the shared
    non-finite-number guard on that same decoded structure. For a ``jsonb`` column
    a value ``json.loads`` cannot parse is itself unstorable: the target binds it
    through a ``$n::jsonb`` cast that PostgreSQL rejects mid-transaction
    (SQLSTATE 22P02), aborting the whole migration with no row identification --
    the exact failure this pre-check exists to convert into a per-row skip
    recorded as an error -- so malformed JSON returns a reason rather than passing
    as storable. (A raw TEXT column has no such cast, so unparseable content is
    not rejected there.)

    Args:
        value: The raw SQLite column value (already serialized to a JSON string
            for a ``jsonb`` column), or ``None``.
        is_jsonb: True when ``value`` is bound into a PostgreSQL ``jsonb`` column,
            enabling the decoded-escape and non-finite-number checks in addition
            to the raw-string check.

    Returns:
        The reason for the first offending value: a shared guard's message, or
        this function's own for ``jsonb`` text that is not valid JSON; else None.
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


# The largest btree index tuple, in bytes, the budgets below let a value produce on the
# PostgreSQL target. Each budget is compared with the value's UTF-8 bytes, uncompressed.
#
# For thread_id and tags only a SQLite SOURCE is checked: every PostgreSQL database
# declares idx_thread_id and idx_tags_tag in its base schema, so a PostgreSQL source
# already holds its thread_id and tag values in btree indexes. An INDEXED METADATA value
# is checked in both directions that write to PostgreSQL: the metadata expression
# indexes are deliberately NOT in the base schema (see app/schemas/postgresql_schema.sql
# -- a database initialized only by this CLI has none until its first server startup),
# so a PostgreSQL source whose METADATA_INDEXED_FIELDS never covered a field, or which
# was never started as a server, can carry a value over its budget.
PG_BTREE_MAX_ITEM_BYTES = 2704


# The bytes the budgets reserve in every index tuple beside the checked value: 16 for
# the tuple header, 4 for the value's length header and 8 of slack.
_PG_INDEX_TUPLE_FIXED_BYTES = 16 + 4 + 8


# Budget for a tag and for the value of a string-typed METADATA_INDEXED_FIELDS entry.
# The metadata value is indexed on its own, by the expression index idx_metadata_<field>
# on ``context_entries((metadata->>'<field>'))`` that handle_metadata_indexes provisions.
# A tag is indexed by idx_tags_tag on ``tags(tag)`` and by the unique idx_tags_entry_tag
# on ``tags(context_entry_id, tag)``; this budget reserves no room for context_entry_id.
PG_MAX_INDEXED_VALUE_BYTES = PG_BTREE_MAX_ITEM_BYTES - _PG_INDEX_TUPLE_FIXED_BYTES


# thread_id gets a smaller budget because it leads idx_context_entries_dedup_hash
# (thread_id, source, content_hash), which the base schema declares: the budget also
# reserves 6 bytes for source and 65 for content_hash.
PG_MAX_INDEXED_THREAD_ID_BYTES = PG_MAX_INDEXED_VALUE_BYTES - (6 + 65)


def pg_unindexable_column_reason(value: str | None, max_bytes: int) -> str | None:
    """Return why a value exceeds its PostgreSQL btree index budget, else None.

    Args:
        value: The candidate value for an INDEXED target column.
        max_bytes: Payload budget in UTF-8 bytes --
            :data:`PG_MAX_INDEXED_THREAD_ID_BYTES` for thread_id,
            :data:`PG_MAX_INDEXED_VALUE_BYTES` for a tag or a string-typed metadata value.

    Returns:
        A reason string when the encoded value exceeds ``max_bytes``, else None.
    """
    if value is None:
        return None
    encoded_bytes = len(value.encode('utf-8'))
    if encoded_bytes <= max_bytes:
        return None
    return (
        f"the value is {encoded_bytes} UTF-8 bytes, over the pre-check's budget of {max_bytes} "
        f'bytes for this indexed column, so this row is skipped -- shorten the value in the '
        f'source database and rerun to migrate it'
    )


def first_pg_unindexable_column(
    columns: Iterable[tuple[str, str | None, int]],
) -> tuple[str, str] | None:
    """Return the first ``(column, reason)`` over its PostgreSQL btree index budget, else None.

    Args:
        columns: Ordered ``(column_name, value, max_bytes)`` candidates for one row,
            limited to columns the target schema actually indexes. ``max_bytes`` is the
            column's payload budget in UTF-8 bytes.

    Returns:
        The ``(column_name, reason)`` of the first oversized column, else None.
    """
    for name, value, max_bytes in columns:
        reason = pg_unindexable_column_reason(value, max_bytes)
        if reason is not None:
            return name, reason
    return None


def first_pg_unindexable_metadata_field(metadata_json: str | None) -> tuple[str, str] | None:
    """Return the first indexed metadata field whose value fails its field's check, else None.

    A ``METADATA_INDEXED_FIELDS`` key gets an expression btree index
    ``idx_metadata_<field>`` on ``context_entries`` (app.migrations.metadata),
    evaluated on every INSERT. SQLite's equivalent ``json_extract`` index applies no
    cast, so a source can hold a value the target's index cast refuses -- aborting the
    whole run when the target already carries the index, or breaking the target's
    first server startup (which creates the index) when the CLI initialized the target
    itself. A source can also hold a value over the width budget below, or a value of
    another type than the field declares, which the running server refuses to store.
    Each is checked by the field's type hint:

    * WIDTH, for a ``string``-typed field, against :data:`PG_MAX_INDEXED_VALUE_BYTES`,
      the budget a tag also gets. The width is measured on the text the expression
      YIELDS (:func:`~app.metadata_types.pg_indexed_metadata_text`),
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
    index of their own. Only top-level keys are inspected, because
    ``metadata->>'<field>'`` addresses top-level keys only. Unparseable text returns
    None because :func:`pg_unstorable_column_reason` reports it with its own reason.
    Non-object metadata, such as a JSON array or scalar, returns None because
    ``metadata->>'<field>'`` reads NULL from a non-object, so no expression index sees a
    value; jsonb stores such metadata as it is.

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
