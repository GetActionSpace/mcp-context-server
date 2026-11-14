"""Centralized identifier handling for context entries.

Public identifier format
    Context-entry IDs are UUIDv7 values displayed as a 32-character lowercase
    hex string without hyphens (matching the regex ``^[0-9a-f]{32}$``). All
    callers exchange identifiers in this canonical form.

Accepted input formats
    Tool boundaries accept either the canonical 32-char hex form or the
    36-char hyphenated UUID form. Whitespace is stripped and case is folded
    to lowercase before validation. Prefix lookups require 8 to 31 hex
    characters.

Lowercase output is required
    ``normalize_id`` always emits lowercase hex. Under SQLite TEXT BINARY
    collation, uppercase ``A-F`` sorts before lowercase ``a-f``; mixing
    cases would corrupt any ``id > ?`` ordering comparison performed by the
    storage layer.

Module contents
    - :data:`ContextId` -- type alias for the canonical hex string.
    - :func:`generate_id` -- produce a fresh UUIDv7 in canonical form.
    - :func:`generate_id_with_timestamp` -- produce a UUIDv7 anchored to a
      specific :class:`datetime.datetime`.
    - :func:`normalize_id` -- validate and canonicalize any accepted input.
    - :func:`is_id_prefix` -- predicate for partial-hex prefix strings.
    - :func:`id_prefix_bounds` -- the lowest and the highest full ID that
      start with a prefix, the bounds of a range read over the ordered ID.
    - :func:`resolve_prefix` -- resolve a prefix to a unique full ID via a
      repository implementing the :class:`_PrefixResolverRepo` protocol,
      matching only the entries the caller's scope may read.
    - :func:`resolve_or_normalize_id` -- single boundary helper that either
      normalizes a full ID or resolves an 8-31 char hex prefix.
    - :func:`resolve_or_normalize_ids` -- the list form of
      :func:`resolve_or_normalize_id`, preserving input order.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from typing import runtime_checkable

import uuid_utils

from app.access_scope import Scope

type ContextId = str  # PEP 695 type alias


@runtime_checkable
class _PrefixResolverRepo(Protocol):
    """Repository capability required by :func:`resolve_prefix`.

    Defined here as a Protocol so that :mod:`app.ids` has no hard
    dependency on the repository layer.
    """

    async def find_ids_by_prefix(self, prefix: ContextId, limit: int = 2, *, scope: Scope) -> list[ContextId]:
        """Return up to ``limit`` IDs of entries the scope may read that start with ``prefix``.

        Returning more than one indicates ambiguity. Implementations should
        cap the result at ``limit`` (default 2) since the caller only needs
        to distinguish unique from ambiguous matches, and apply the scope
        before that cap, so an entry the scope may not read never counts.
        """
        ...


def generate_id() -> ContextId:
    """Generate a fresh UUIDv7 as a 32-character lowercase hex string.

    Returns:
        A string matching ``^[0-9a-f]{32}$``.
    """
    return uuid_utils.uuid7().hex


def generate_id_with_timestamp(created_at: datetime) -> ContextId:
    """Generate a UUIDv7 whose embedded timestamp matches ``created_at``.

    Produces deterministic ordering for callers that need to assign IDs
    to records with predetermined creation times.

    Args:
        created_at: A timezone-aware datetime (UTC recommended). The
            timezone offset is honored by :meth:`datetime.timestamp`.

    Returns:
        A 32-character lowercase hex string matching ``^[0-9a-f]{32}$``.
        The embedded UUIDv7 timestamp decodes to the same calendar
        millisecond as ``created_at``.
    """
    # uuid_utils.uuid7 expects ``timestamp`` in UNIX SECONDS (an integer)
    # and ``nanos`` as the nanosecond fraction within that second. Passing
    # milliseconds here would place the embedded timestamp roughly 1000
    # times further into the future, producing UUIDs whose year decodes to
    # ~50,000 AD instead of the input year.
    # Upstream tracker on the timestamp parameter's units: https://github.com/aminalaee/uuid-utils/issues/73
    seconds = int(created_at.timestamp())
    nanos = created_at.microsecond * 1_000  # microseconds -> nanoseconds
    return uuid_utils.uuid7(timestamp=seconds, nanos=nanos).hex


def normalize_id(value: str) -> ContextId:
    """Validate ``value`` and return it in canonical 32-char lowercase hex form.

    Accepts either the 32-character hex form or the 36-character hyphenated
    UUID form. Whitespace is stripped and case is folded to lowercase
    before validation. The function is idempotent on already-normalized
    input.

    The lowercase output is required: under SQLite TEXT BINARY collation,
    uppercase ``A-F`` sorts before lowercase ``a-f``, so any ``id > ?``
    ordering comparison would behave incorrectly if uppercase IDs were
    accepted.

    Args:
        value: A UUID in 32-char hex form (e.g.
            ``'0190abcdef1234567890abcdef123456'``) or 36-char hyphenated
            form (e.g. ``'0190abcd-ef12-3456-7890-abcdef123456'``).

    Returns:
        A 32-character lowercase hex string matching ``^[0-9a-f]{32}$``.

    Raises:
        ValueError: If ``value`` is not a valid 32-char hex or 36-char
            hyphenated UUID after normalization.
    """
    s = value.strip().lower()
    if len(s) == 36 and s.count('-') == 4:
        s = s.replace('-', '')
    if len(s) != 32:
        raise ValueError(f'Invalid UUID identifier length (expected 32 hex chars after normalization): {value!r}')
    if any(c not in '0123456789abcdef' for c in s):
        raise ValueError(f'Invalid UUID identifier (non-hex character): {value!r}')
    return s


def is_id_prefix(value: str) -> bool:
    """Return ``True`` if ``value`` is a UUID hex prefix of length 8 to 31.

    Used at tool boundaries to decide whether to call :func:`normalize_id`
    (full UUID) or :func:`resolve_prefix` (partial hex prefix). Whitespace
    is stripped and case is folded internally; callers do not need to
    pre-normalize.

    Args:
        value: Candidate prefix string.

    Returns:
        ``True`` iff ``value`` has length 8 through 31 inclusive and
        contains only hex characters (after lowercasing). ``False`` for
        full 32-char UUIDs, 36-char hyphenated UUIDs, prefixes shorter
        than 8 characters, and any string containing non-hex characters.
    """
    s = value.strip().lower()
    if not (8 <= len(s) < 32):
        return False
    return all(c in '0123456789abcdef' for c in s)


def id_prefix_bounds(prefix: str) -> tuple[ContextId, ContextId]:
    """Return the lowest and the highest full ID that start with ``prefix``.

    Every ID starting with the prefix lies between the two bounds, inclusive, and every 32-character lowercase
    hex string between them starts with the prefix, so a range read over the ordered ID answers a prefix
    lookup. The order is the same on both backends: lowercase hex text compared byte by byte on SQLite, the
    16 bytes of a native ``uuid`` on PostgreSQL.

    Args:
        prefix: An 8-31 character hex prefix; whitespace is stripped and case is folded.

    Returns:
        The prefix padded to 32 characters with ``0``, and padded with ``f``.

    Raises:
        ValueError: If ``prefix`` is not an 8-31 character hex prefix.
    """
    s = prefix.strip().lower()
    if not is_id_prefix(s):
        raise ValueError(f'Invalid UUID prefix (must be 8-31 hex chars): {prefix!r}')
    return s.ljust(32, '0'), s.ljust(32, 'f')


async def resolve_prefix(prefix: str, repo: _PrefixResolverRepo, *, scope: Scope) -> ContextId:
    """Resolve an 8-31 char hex prefix to the unique full UUID of an entry the scope may read.

    Uniqueness is decided over the entries the scope may read, so an entry the
    scope may not read never makes a prefix ambiguous, and a prefix that only
    such entries carry matches nothing, exactly like a prefix no entry carries.

    Args:
        prefix: A hex string 8 through 31 characters long (validated via
            :func:`is_id_prefix`). Whitespace is stripped and case is
            folded internally.
        repo: Object implementing the :class:`_PrefixResolverRepo`
            protocol; provides :meth:`find_ids_by_prefix`.
        scope: The caller's scope.

    Returns:
        A 32-character lowercase hex string identifying the unique readable
        context entry whose ``id`` starts with ``prefix``.

    Raises:
        ValueError: If ``prefix`` is not a valid 8-31 char hex prefix
            (length below 8, length 32 or above, contains non-hex
            characters), if no readable entry matches, or if more than one
            readable entry matches the prefix.
    """
    s = prefix.strip().lower()
    if not is_id_prefix(s):
        raise ValueError(f'Invalid UUID prefix (must be 8-31 hex chars): {prefix!r}')
    matches = await repo.find_ids_by_prefix(s, limit=2, scope=scope)
    if not matches:
        raise ValueError(f'No context entry matches prefix {prefix!r}')
    if len(matches) > 1:
        raise ValueError(f'Ambiguous prefix {prefix!r} matches multiple entries')
    return matches[0]


async def resolve_or_normalize_id(value: str, repo: _PrefixResolverRepo, *, scope: Scope) -> ContextId:
    """Canonicalize a full ID, or resolve an 8-31 char hex prefix to its unique full ID.

    Single boundary helper for every tool that accepts a context-entry ID from
    outside the storage layer. A full 32-char hex or 36-char hyphenated identifier
    is validated and lowercased via :func:`normalize_id`; an 8-31 hex-char prefix
    is resolved to its unique full ID via :func:`resolve_prefix`. Routing every
    ID-accepting tool through this one helper keeps prefix acceptance uniform
    across ``get_context_by_ids``, ``update_context``, ``delete_context`` and
    their batch variants.

    Propagates :class:`ValueError` from :func:`normalize_id` or
    :func:`resolve_prefix` when ``value`` is neither a valid full identifier nor
    a prefix that resolves to exactly one entry (no match or ambiguous match).

    Args:
        value: A full UUID (32-char hex or 36-char hyphenated) or an 8-31 char
            hex prefix.
        repo: Object implementing the :class:`_PrefixResolverRepo` protocol.
        scope: The caller's scope; a prefix resolves over the entries it may read.

    Returns:
        A 32-character lowercase hex string matching ``^[0-9a-f]{32}$``.
    """
    if is_id_prefix(value):
        return await resolve_prefix(value, repo, scope=scope)
    return normalize_id(value)


async def resolve_or_normalize_ids(
    values: Sequence[str], repo: _PrefixResolverRepo, *, scope: Scope,
) -> list[ContextId]:
    """Apply :func:`resolve_or_normalize_id` to each item, preserving order.

    Propagates :class:`ValueError` from the first item that is neither a valid
    full identifier nor a prefix that resolves to exactly one entry.

    Args:
        values: Context-entry IDs and/or 8-31 char hex prefixes.
        repo: Object implementing the :class:`_PrefixResolverRepo` protocol.
        scope: The caller's scope; every prefix resolves over the entries it may read.

    Returns:
        The canonical 32-char lowercase hex IDs in the same order as ``values``.
    """
    return [await resolve_or_normalize_id(value, repo, scope=scope) for value in values]
