"""Rewrite of entry ids at metadata paths, and the stamp of a migrated entry's old id.

Every migration direction rewrites each entry's metadata through ``MetadataIdRewriter``
once the complete table of old and new ids is built, so an entry citing a newer entry is
rewritten like one citing an older entry.
"""

import json
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Self
from typing import cast

from app.cli.migrate_uuid.id_paths import NUMERIC_ID
from app.cli.migrate_uuid.id_paths import IdPath
from app.cli.migrate_uuid.id_paths import Location
from app.cli.migrate_uuid.id_paths import locations
from app.cli.migrate_uuid.id_paths import render_location
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import MetadataOutcome
from app.cli.migrate_uuid.records import StampOutcome


def _id_number(value: object) -> int | None:
    """Return the entry id a value names, or None when it names none.

    A JSON integer names itself. A string names an integer only in the exact text JSON
    writes for one (``NUMERIC_ID``): ASCII digits without a leading zero, after an
    optional leading minus, so ``"-5"`` reads as -5. Strings with leading zeros or
    surrounding whitespace, a leading ``+``, ``"-0"``, digit separators, a decimal point
    or non-ASCII digits read as codes rather than ids and are not taken for ids. A boolean
    is never an id.

    Args:
        value: A value found at a path.

    Returns:
        The integer the value names, or None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and NUMERIC_ID.fullmatch(value):
        return int(value)
    return None


def _json_type(value: object) -> str:
    """Name the JSON type of a parsed value that is not an object.

    Args:
        value: A parsed JSON value.

    Returns:
        ``array``, ``string``, ``boolean``, ``null`` or ``number``.
    """
    if isinstance(value, list):
        return 'array'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, bool):
        return 'boolean'
    if value is None:
        return 'null'
    return 'number'


def _value_at(root: object, location: Location) -> object:
    """Return the value a location reaches from the metadata root.

    Args:
        root: Parsed metadata.
        location: Object keys and list indexes from the root.

    Returns:
        The value at the location.
    """
    node = root
    for step in location:
        node = cast(list[object], node)[step] if isinstance(step, int) else cast(dict[str, object], node)[step]
    return node


class MetadataIdRewriter:
    """Rewrites the entry ids found at the given metadata paths, then stamps the old id.

    A value found at a path is a candidate id; a list found there is a list of candidates.
    An integer, or a string in the form JSON writes an integer, that is the id of an entry
    the run migrates becomes that entry's new 32-character id. Every other value stays as
    written: an integer or such a string naming no migrated entry is counted
    ``not_migrated`` with a warning, anything else is counted ``kept``. When several
    paths find the same value, the first path given rewrites and counts it. Metadata
    that does not parse, or that holds a number Python cannot write back as JSON, is
    returned unchanged with an error: no id in it is rewritten, every path count is
    zero, and with a legacy id key the stamp outcome is ``not_stamped``.
    """

    def __init__(
        self,
        id_paths: Sequence[IdPath],
        id_mapping: Mapping[int, str],
        *,
        legacy_id_key: str | None,
        unmigrated_ids: AbstractSet[int] = frozenset(),
    ) -> None:
        """Create a rewriter.

        Args:
            id_paths: The paths to rewrite, in command-line order.
            id_mapping: Every source entry's old integer id and its new id.
            legacy_id_key: The top-level key the old id is stamped under, or None.
            unmigrated_ids: Old ids in ``id_mapping`` of entries the run does not
                store; values naming them are kept, not rewritten.
        """
        self._id_paths = tuple(id_paths)
        self._id_mapping = id_mapping
        self._legacy_id_key = legacy_id_key
        self._unmigrated_ids = frozenset(unmigrated_ids)

    def excluding(self, unmigrated_ids: AbstractSet[int]) -> Self:
        """Return a rewriter that also treats the given old ids as not migrated.

        The new rewriter shares this one's id table; this one is unchanged.

        Args:
            unmigrated_ids: Old ids of entries the run skips.

        Returns:
            The new rewriter.
        """
        return type(self)(
            self._id_paths,
            self._id_mapping,
            legacy_id_key=self._legacy_id_key,
            unmigrated_ids=self._unmigrated_ids | unmigrated_ids,
        )

    def rewrite(self, metadata_json: str | None, old_id: int) -> MetadataOutcome:
        """Rewrite the ids in one entry's metadata and stamp its old id.

        Re-encoding uses ``allow_nan=False``. ``json.loads`` accepts the non-standard
        ``NaN``/``Infinity``/``-Infinity`` and silently turns a standard literal with a
        fraction or an exponent that is too large for a double, such as ``1e400``, into
        ``inf``, which a default ``json.dumps`` would write back as a token no RFC 8259
        parser accepts, in place of the source text. With
        ``allow_nan=False`` the encoder raises instead, the rewrite is discarded and the
        original text is kept verbatim, so every path count of the outcome is zero, it
        carries no warning, and with a legacy id key its stamp is ``not_stamped``. A
        PostgreSQL target written from a SQLite source then skips the row
        through its non-finite pre-check, so the text never reaches the jsonb bind; every
        other target stores the text as the source held it, since a SQLite column carries
        no JSON check and a PostgreSQL source's jsonb text is text a PostgreSQL target's
        jsonb accepts. Either way the error makes the command exit non-zero so the
        operator can repair the value.

        Args:
            metadata_json: The source row's metadata text, or None.
            old_id: The source row's integer id, used in messages and the stamp.

        Returns:
            The metadata to store, with the counts per path, the stamp outcome and the
            messages for this row.
        """
        counts = {path.text: IdPathCounts() for path in self._id_paths}
        key = self._legacy_id_key
        if metadata_json is None:
            if key is None:
                return MetadataOutcome(None, counts, None, [], [])
            return MetadataOutcome(json.dumps({key: old_id}), counts, 'stamped', [], [])
        try:
            root: object = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            message = f'row {old_id}: metadata JSON parse failed ({exc}); no id in it was rewritten'
            if key is not None:
                message += f', and the old id was not stamped under {key!r}'
            return self._preserved(metadata_json, message)
        warnings: list[str] = []
        owners: dict[Location, str] = {}
        for path in self._id_paths:
            for location in locations(root, path):
                owners.setdefault(location, path.text)
        for location, text in owners.items():
            self._rewrite_location(root, location, counts[text], warnings, old_id)
        stamp, stamp_error = self._stamp(root, old_id)
        try:
            metadata = json.dumps(root, ensure_ascii=False, allow_nan=False)
        except ValueError as exc:
            message = (
                f'row {old_id}: metadata contains a number Python cannot round-trip as JSON ({exc}); '
                'no id in it was rewritten'
            )
            if key is not None:
                message += f', and the old id was not stamped under {key!r}'
            return self._preserved(metadata_json, message)
        return MetadataOutcome(metadata, counts, stamp, warnings, [] if stamp_error is None else [stamp_error])

    def _preserved(self, metadata_json: str, error: str) -> MetadataOutcome:
        """Return the outcome of a row whose metadata is kept verbatim.

        Args:
            metadata_json: The original metadata text.
            error: Why it was kept.

        Returns:
            The original text with zero counts, no warnings, the error, and
            ``not_stamped`` when a stamp key is given.
        """
        return MetadataOutcome(
            metadata=metadata_json,
            counts={path.text: IdPathCounts() for path in self._id_paths},
            stamp=None if self._legacy_id_key is None else 'not_stamped',
            warnings=[],
            errors=[error],
        )

    def _rewrite_location(
        self,
        root: object,
        location: Location,
        counts: IdPathCounts,
        warnings: list[str],
        old_id: int,
    ) -> None:
        """Rewrite the value at one location, or each element when it is a list.

        Args:
            root: Parsed metadata, changed in place.
            location: Where the value sits; its last step is an object key.
            counts: The owning path's counts for this row.
            warnings: The row's warnings.
            old_id: The source row's integer id.
        """
        parent = cast(dict[str, object], _value_at(root, location[:-1]))
        key = cast(str, location[-1])
        value = parent[key]
        if isinstance(value, list):
            elements = cast(list[object], value)
            for index, element in enumerate(elements):
                elements[index] = self._rewrite_value(element, (*location, index), counts, warnings, old_id)
        else:
            parent[key] = self._rewrite_value(value, location, counts, warnings, old_id)

    def _rewrite_value(
        self,
        value: object,
        location: Location,
        counts: IdPathCounts,
        warnings: list[str],
        old_id: int,
    ) -> object:
        """Classify one candidate and return the value to store in its place.

        Args:
            value: The candidate.
            location: Where it sits, list index included, for the warning.
            counts: The owning path's counts for this row.
            warnings: The row's warnings.
            old_id: The source row's integer id.

        Returns:
            The new id when the value names a migrated entry, else the value itself.
        """
        number = _id_number(value)
        if number is None:
            counts.kept += 1
            return value
        new_id = None if number in self._unmigrated_ids else self._id_mapping.get(number)
        if new_id is not None:
            counts.rewritten += 1
            return new_id
        counts.not_migrated += 1
        reason = (
            'the id of an entry this run skipped'
            if number in self._unmigrated_ids
            else 'which is not the id of a migrated entry'
        )
        warnings.append(f'row {old_id}: {render_location(location)} holds {value!r}, {reason}; not rewritten')
        return value

    def _stamp(self, root: object, old_id: int) -> tuple[StampOutcome | None, str | None]:
        """Stamp the old id under the legacy id key, after the row's rewrite.

        Args:
            root: Parsed and rewritten metadata, changed in place.
            old_id: The source row's integer id.

        Returns:
            The stamp outcome (None without a key) and the error of a stamp that failed.
        """
        key = self._legacy_id_key
        if key is None:
            return None, None
        if not isinstance(root, dict):
            return 'not_stamped', (
                f'row {old_id}: metadata is a JSON {_json_type(root)}, not an object, '
                f'so the old id was not stamped under {key!r}'
            )
        metadata = cast(dict[str, object], root)
        if key in metadata:
            return 'already_present', None
        metadata[key] = old_id
        return 'stamped', None
