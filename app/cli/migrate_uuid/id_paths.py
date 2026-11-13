"""Metadata paths that hold entry ids, and the key a migrated entry's old id is stamped under.

A path is its keys joined by single dots, read from the top of the metadata object. Each key is
compared exactly (case-sensitive, no trimming); ``*`` as a whole key stands for any one key; a
leading ``**.`` lets the rest of the path start at any object at any depth, objects reached
through list elements included. A key whose name contains a dot or a star cannot be addressed,
just as the server's metadata filters cannot address a dotted key.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self
from typing import cast

from app.ingest import INGEST_KEY
from app.metadata_sql import is_safe_key

DEFAULT_ID_PATH = '**.references.context_ids'

# The text Python and JSON write for an integer: no padding, no '+', no leading zero, no
# separators, ASCII digits only. Use ``fullmatch``: ``match`` would accept the '0' of '007'.
NUMERIC_ID = re.compile(r'0|-?[1-9][0-9]*')

ANY_KEY = '*'
ANY_DEPTH = '**'

# Steps from the metadata root to a value: an object key (str) or a list index (int).
type Location = tuple[str | int, ...]


@dataclass(frozen=True)
class IdPath:
    """A parsed ``--rewrite-ids-at`` path.

    Attributes:
        text: The path as given, which names it in the summary and the report.
        any_depth: True when the path starts with ``**.``.
        keys: The keys after ``**.``, or all keys when it is absent; ``*`` is any one key.
    """

    text: str
    any_depth: bool
    keys: tuple[str, ...]

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse a path text.

        Args:
            text: The path, keys separated by single dots.

        Returns:
            The parsed path.

        Raises:
            ValueError: The text has an empty key, ``**`` anywhere but first or with no key
                after it, or ``*`` as part of a key.
        """
        keys = text.split('.')
        if '' in keys:
            raise ValueError(f'{text!r} has an empty key: keys are separated by single dots, with none at either end')
        any_depth = keys[0] == ANY_DEPTH
        if any_depth:
            keys = keys[1:]
        if not keys or ANY_DEPTH in keys:
            raise ValueError(f'{text!r}: "**" may appear only as the first key, followed by at least one key')
        if any(ANY_KEY in key and key != ANY_KEY for key in keys):
            raise ValueError(f'{text!r}: "*" stands for a whole key and cannot be part of one')
        return cls(text=text, any_depth=any_depth, keys=tuple(keys))


def _objects(root: object) -> Iterator[tuple[dict[str, object], Location]]:
    """Yield every object in ``root`` with its location, ``root`` included, in document order.

    Objects are reached through object values and list elements at any depth.

    Args:
        root: Parsed metadata.

    Yields:
        Each object and the location it sits at.
    """
    pending: list[tuple[object, Location]] = [(root, ())]
    while pending:
        node, location = pending.pop()
        if isinstance(node, dict):
            obj = cast(dict[str, object], node)
            yield obj, location
            children = [(value, (*location, key)) for key, value in obj.items()]
        elif isinstance(node, list):
            children = [(value, (*location, index)) for index, value in enumerate(cast(list[object], node))]
        else:
            continue
        pending.extend(reversed(children))


def _match(obj: dict[str, object], location: Location, keys: tuple[str, ...]) -> Iterator[Location]:
    """Yield the locations ``keys`` reach from ``obj``, stepping only into object values.

    Args:
        obj: The object the first key is looked up in.
        location: Where ``obj`` sits in the metadata.
        keys: The keys still to follow, at least one.

    Yields:
        The location of each value the last key reaches.
    """
    head, rest = keys[0], keys[1:]
    if head == ANY_KEY:
        steps = list(obj.items())
    elif head in obj:
        steps = [(head, obj[head])]
    else:
        return
    for key, value in steps:
        if not rest:
            yield (*location, key)
        elif isinstance(value, dict):
            yield from _match(cast(dict[str, object], value), (*location, key), rest)


def locations(root: object, path: IdPath) -> Iterator[Location]:
    """Yield every location ``path`` finds in parsed metadata, each once.

    Without ``**`` the match starts at the root, which must be an object. With it, the match
    starts at every object in ``root``, the root included, also when the root is a list.
    Between keys only object values are stepped into: a list, a scalar or null ends that branch.
    The last key's value is found whatever it holds. Locations come per starting object, the
    objects in the order they appear in the metadata, so messages about them follow that order.

    Args:
        root: Parsed metadata.
        path: The path to match.

    Yields:
        The location of each value found.
    """
    if path.any_depth:
        for obj, location in _objects(root):
            yield from _match(obj, location, path.keys)
    elif isinstance(root, dict):
        yield from _match(cast(dict[str, object], root), (), path.keys)


def render_location(location: Location) -> str:
    """Render a location for a message, such as ``metadata.items[0].references.context_ids[2]``.

    Args:
        location: Keys and list indexes from the metadata root.

    Returns:
        ``metadata`` followed by each key after a dot and each list index in brackets.
    """
    return 'metadata' + ''.join(f'[{step}]' if isinstance(step, int) else f'.{step}' for step in location)


def check_legacy_id_key(name: str) -> str:
    """Check the top-level metadata key a migrated entry's old id is stamped under.

    The stamp exists to be found by the server's metadata filters, so the key must be one they
    address at the top level: their key rule with no dot, which they read as nesting.

    Args:
        name: The key given to ``--legacy-id-key``.

    Returns:
        The key, unchanged.

    Raises:
        ValueError: The filters cannot address the key at the top level, or it is the
            corpus-sync block key.
    """
    if not is_safe_key(name) or '.' in name:
        raise ValueError(
            f'{name!r} cannot be stamped: the metadata filters address a top-level key '
            'made of letters, digits, "_" and "-" only',
        )
    if name == INGEST_KEY:
        raise ValueError(f'{name!r} is reserved for the corpus-sync block')
    return name
