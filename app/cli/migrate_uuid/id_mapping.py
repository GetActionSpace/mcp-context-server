"""Integer-to-UUIDv7 ID mapping and the timestamp coercion it relies on.

Each migrated row receives a UUIDv7 whose timestamp comes from its ``created_at``;
the helpers here coerce source timestamps and anchor unusable ones.
"""

import logging
import sqlite3
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from datetime import timedelta

# UUIDv7 generation for integer-keyed rows uses the timestamp parameter of
# uuid_utils.uuid7() in UNIX seconds (with optional nanos for sub-second
# precision). Upstream tracker on the parameter's units:
# https://github.com/aminalaee/uuid-utils/issues/73
from app.ids import generate_id_with_timestamp

logger = logging.getLogger(__name__)


def _coerce_datetime(value: object) -> datetime:
    """Coerce SQLite-side timestamp values to :class:`datetime.datetime`.

    SQLite stores timestamps as TEXT or naive Python datetimes. The
    function accepts either form and returns a timezone-aware datetime
    (assuming UTC for naive inputs and ISO-format text).

    Args:
        value: A SQLite timestamp value (str or datetime).

    Returns:
        A timezone-aware :class:`datetime.datetime`.

    Raises:
        ValueError: If ``value`` cannot be parsed.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = datetime.strptime(text, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    # A bool is a degenerate int and is NOT a valid timestamp -- reject it explicitly.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Some non-app source databases store created_at as Unix epoch SECONDS. Coerce via
        # epoch + timedelta (NOT datetime.fromtimestamp, which raises OSError for negative /
        # out-of-range epochs on Windows) so a numeric -- including pre-1970 -- created_at
        # never aborts the whole migration; created_at_for_id anchors a resulting pre-1970
        # value for id derivation while the stored created_at value is preserved verbatim.
        try:
            return NULL_CREATED_AT_ANCHOR + timedelta(seconds=float(value))
        except (OverflowError, ValueError):
            # An out-of-range epoch (extreme future/past) cannot derive a datetime: anchor
            # it rather than abort, matching the NULL / pre-1970 handling.
            return NULL_CREATED_AT_ANCHOR
    raise ValueError(f'unsupported created_at value: {value!r}')


# A schema-legal NULL created_at in an arbitrary non-app source row cannot derive
# a UUIDv7 id (the id embeds a timestamp), so it is anchored to a fixed epoch
# rather than aborting the whole migration. The stored created_at VALUE is kept
# NULL (not invented); only id derivation uses the anchor.
NULL_CREATED_AT_ANCHOR = datetime(1970, 1, 1, tzinfo=UTC)


def sqlite_timestamp(value: datetime | None) -> str | None:
    """Render a source datetime as SQLite's canonical TEXT timestamp.

    SQLite stores created_at/updated_at as "YYYY-MM-DD HH:MM:SS" (UTC, the
    CURRENT_TIMESTAMP form the server writes) and orders/filters them as TEXT.
    Writing ``datetime.isoformat()`` ("YYYY-MM-DDTHH:MM:SS+00:00") instead stores a
    'T'/offset form that mis-sorts under SQLite's TEXT comparison and ``ORDER BY
    created_at`` (and skews date-range filters), so normalize to the space form in
    UTC. ``None`` is preserved (schema-legal NULL).

    Returns:
        The canonical "YYYY-MM-DD HH:MM:SS" UTC string, or ``None`` for ``None``.
    """
    if value is None:
        return None
    dt = value.astimezone(UTC) if value.tzinfo is not None else value
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def created_at_for_id(value: object) -> datetime:
    """Coerce a source ``created_at`` to a datetime for UUIDv7 id derivation.

    Unlike :func:`_coerce_datetime`, a missing (NULL) ``created_at`` -- and any
    other value that cannot be parsed (a malformed non-ISO / non-epoch string, an
    out-of-range epoch) -- is tolerated: it falls back to
    :data:`NULL_CREATED_AT_ANCHOR` so one bad row in an arbitrary non-app source
    database cannot abort the entire migration. The stored ``created_at`` value is
    preserved verbatim by the callers that bind it (see
    :func:`stored_datetime_or_none`); only the derived id timestamp is anchored.

    Args:
        value: A source ``created_at`` value (datetime, ISO/epoch text, numeric
            epoch, None, or an unparseable value).

    Returns:
        A timezone-aware :class:`datetime.datetime`, or the epoch anchor when
        ``value`` is None or cannot be parsed.
    """
    if value is None:
        return NULL_CREATED_AT_ANCHOR
    try:
        coerced = _coerce_datetime(value)
    except (ValueError, OverflowError):
        # A malformed non-NULL created_at (e.g. a non-ISO string like '2024/01/01'
        # or '15-06-2024', for which both datetime.fromisoformat and the
        # '%Y-%m-%d %H:%M:%S' fallback raise) must NOT abort the whole migration:
        # anchor its derived id like the NULL / pre-1970 / out-of-range-epoch cases
        # while the binding callers preserve the stored value verbatim (or NULL).
        return NULL_CREATED_AT_ANCHOR
    # A pre-1970 (negative-epoch) timestamp makes uuid_utils.uuid7 raise
    # OverflowError on the negative seconds; anchor it like NULL for id
    # derivation while the stored created_at value is preserved verbatim.
    if coerced < NULL_CREATED_AT_ANCHOR:
        return NULL_CREATED_AT_ANCHOR
    return coerced


def stored_datetime_or_none(value: object) -> datetime | None:
    """Coerce a stored ``created_at`` / ``updated_at`` for a verbatim target bind.

    Mirrors :func:`created_at_for_id`'s tolerance for the value bound INTO the
    target timestamp column (not the derived id): a NULL or otherwise unparseable
    value yields ``None`` (stored as SQL NULL on the target, exactly as a NULL
    source already does) rather than aborting the whole migration on one bad row
    in an arbitrary non-app source database. A well-formed value is coerced to a
    timezone-aware datetime.

    Args:
        value: A stored timestamp value (datetime, ISO/epoch text, numeric epoch,
            None, or an unparseable value).

    Returns:
        A timezone-aware :class:`datetime.datetime`, or ``None`` when the value is
        NULL or cannot be parsed.
    """
    if value is None:
        return None
    try:
        return _coerce_datetime(value)
    except (ValueError, OverflowError):
        return None


def build_id_mapping(source_rows: Iterable[sqlite3.Row]) -> dict[int, str]:
    """Construct the integer-to-UUIDv7 mapping table.

    For each source row, generates a UUIDv7 from the row's ``created_at``
    timestamp via :func:`app.ids.generate_id_with_timestamp`. The embedded
    48-bit timestamp field is deterministic at millisecond precision; the
    lower 74 random bits are not.

    Args:
        source_rows: Iterable of source ``context_entries`` rows
            containing at minimum the columns ``id`` (integer) and
            ``created_at`` (timestamp).

    Returns:
        Dictionary mapping each source integer ID to a 32-character
        lowercase hex UUIDv7 string.
    """
    mapping: dict[int, str] = {}
    null_created_at = 0
    for row in source_rows:
        source_id = int(row['id'])
        if row['created_at'] is None:
            null_created_at += 1
        created_at = created_at_for_id(row['created_at'])
        mapping[source_id] = generate_id_with_timestamp(created_at)
    if null_created_at:
        logger.warning(
            '%d source context_entries row(s) had NULL created_at; their ids '
            'were anchored to %s (the stored created_at is preserved as NULL)',
            null_created_at,
            NULL_CREATED_AT_ANCHOR.isoformat(),
        )
    return mapping
