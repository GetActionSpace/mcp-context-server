"""Two-principal cases of the by-id reads, the existence probes and id-prefix resolution.

Each seam admits only the rows the caller may read, so a row the caller may not read is
indistinguishable from an absent one: it is omitted from a read, probes like a missing id
and never matches a prefix.

- S3: ``get_by_ids`` returns exactly the readable rows of the requested ids, also when
  the id list spans several statement chunks, each with its stored visibility and whether
  the caller owns it.
- S4: ``find_ids_by_prefix`` matches readable rows only, before its LIMIT, so hidden rows
  ordered first never crowd out a readable match; prefix resolution through
  :func:`app.ids.resolve_or_normalize_id` turns that into a unique match, a no-match or an
  ambiguity computed over readable rows alone. Rows at both edges of a prefix's range and
  the first id past it match exactly the prefixes they start with, at the shortest (8), an
  odd (9) and the longest (31) prefix length.
- S5: ``check_entry_exists`` reports a readable row with its source, version, owner and
  whether the caller may modify it, and an unreadable row exactly like a missing one.
- S7: ``probe_ids`` maps every readable requested id to whether the caller may modify and
  owns it, inside or outside a transaction and across statement chunks, and omits the rest.
"""

from collections.abc import Sequence

from app.access_scope import Scope
from app.ids import generate_id
from app.ids import resolve_or_normalize_id
from tests.helpers import insert_grant
from tests.repositories._access_scope_cases import OWNED
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import SEED_ROWS
from tests.repositories._access_scope_cases import SEED_SOURCE
from tests.repositories._access_scope_cases import TEAM_GROUP
from tests.repositories._access_scope_cases import WRITABLE
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Grant
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import Visibility
from tests.repositories._access_scope_cases import expected_read_back
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import per_scope

type ProbeRow = tuple[str, bool, str | None, int | None, str | None, bool]
type AccessRow = tuple[str, bool, bool]
type ReadBack = dict[str, tuple[object, object]]
type PrefixMatches = tuple[tuple[str, tuple[str, ...]], ...]
type PrefixResolution = tuple[tuple[str, str], ...]

# An id no row carries; it stands for every absent id the seams are asked about.
ABSENT_LABEL = 'absent'
ABSENT_ID = '0' * 31 + '1'

# Requested ids of the chunked variants: enough to span two statement chunks, with seed ids
# placed on both sides of the first chunk boundary.
_CHUNKED_ID_COUNT = 1200
_CHUNKED_SEED_POSITIONS = (0, 450, 898, 899, 900, 901, 1100, 1199)

PREFIX_THREAD = 'access-prefixes'
# Each group shares one 8-character prefix; the absent prefix matches no row.
PREFIX_A = 'a5a5a5a5'
PREFIX_B = 'b6b6b6b6'
PREFIX_C = 'c7c7c7c7'
PREFIX_D = 'd7d7d7d7'
PREFIX_ABSENT = 'd9d9d9d9'
# The fourth group holds the lowest and the highest id that start with PREFIX_D and the
# first id past them, so a range read whose bounds or comparisons are off by one digit
# drops an edge row or takes in the neighbor.
PREFIX_D_LOW = PREFIX_D + '0' * 23
PREFIX_D_HIGH = PREFIX_D + 'f' * 23
PREFIX_D_ODD = PREFIX_D + 'f'
PREFIX_D_NEXT = 'd7d7d7d8'
PREFIXES = (PREFIX_A, PREFIX_B, PREFIX_C, PREFIX_D, PREFIX_D_LOW, PREFIX_D_HIGH, PREFIX_D_ODD, PREFIX_D_NEXT, PREFIX_ABSENT)

# The prefixed rows in id order: (label, id, owner, visibility, grants).
_PREFIXED_ROWS: tuple[tuple[str, str, str, Visibility, tuple[Grant, ...]], ...] = (
    # Two private rows ordered before a public one: a LIMIT applied before the access
    # predicate would fill the two-row lookup with rows only alice may read.
    ('prefix_a_alice_private_1', PREFIX_A + '1' * 24, 'alice', 'private', ()),
    ('prefix_a_alice_private_2', PREFIX_A + '2' * 24, 'alice', 'private', ()),
    ('prefix_a_alice_public', PREFIX_A + '3' * 24, 'alice', 'public', ()),
    ('prefix_b_alice_private', PREFIX_B + '1' * 24, 'alice', 'private', ()),
    ('prefix_b_alice_bob_read', PREFIX_B + '2' * 24, 'alice', 'private', (Grant('user', 'bob', 'read'),)),
    ('prefix_c_alice_team_read', PREFIX_C + '1' * 24, 'alice', 'private', (Grant('group', TEAM_GROUP, 'read'),)),
    ('prefix_c_bob_private', PREFIX_C + '2' * 24, 'bob', 'private', ()),
    # Public, so every scope reads the edge rows and the neighbor alike.
    ('prefix_d_low', PREFIX_D + '0' * 24, 'alice', 'public', ()),
    ('prefix_d_high', PREFIX_D + 'f' * 24, 'alice', 'public', ()),
    ('prefix_d_next', PREFIX_D_NEXT + '0' * 24, 'alice', 'public', ()),
)

# What a two-row lookup of each prefix of the fourth group returns, the same for every scope.
_PREFIX_D_MATCHES: PrefixMatches = (
    (PREFIX_D, ('prefix_d_low', 'prefix_d_high')),
    (PREFIX_D_LOW, ('prefix_d_low',)),
    (PREFIX_D_HIGH, ('prefix_d_high',)),
    (PREFIX_D_ODD, ('prefix_d_high',)),
    (PREFIX_D_NEXT, ('prefix_d_next',)),
)

# What a two-row lookup of each prefix returns per scope, in id order.
_PREFIX_MATCHES: dict[str, PrefixMatches] = {
    'alice': (
        (PREFIX_A, ('prefix_a_alice_private_1', 'prefix_a_alice_private_2')),
        (PREFIX_B, ('prefix_b_alice_private', 'prefix_b_alice_bob_read')),
        (PREFIX_C, ('prefix_c_alice_team_read',)),
        *_PREFIX_D_MATCHES,
        (PREFIX_ABSENT, ()),
    ),
    'bob': (
        (PREFIX_A, ('prefix_a_alice_public',)),
        (PREFIX_B, ('prefix_b_alice_bob_read',)),
        (PREFIX_C, ('prefix_c_bob_private',)),
        *_PREFIX_D_MATCHES,
        (PREFIX_ABSENT, ()),
    ),
    'carol': (
        (PREFIX_A, ('prefix_a_alice_public',)),
        (PREFIX_B, ()),
        (PREFIX_C, ('prefix_c_alice_team_read',)),
        *_PREFIX_D_MATCHES,
        (PREFIX_ABSENT, ()),
    ),
    'dave': (
        (PREFIX_A, ('prefix_a_alice_public',)),
        (PREFIX_B, ()),
        (PREFIX_C, ()),
        *_PREFIX_D_MATCHES,
        (PREFIX_ABSENT, ()),
    ),
    'anonymous': (
        (PREFIX_A, ('prefix_a_alice_public',)),
        (PREFIX_B, ()),
        (PREFIX_C, ()),
        *_PREFIX_D_MATCHES,
        (PREFIX_ABSENT, ()),
    ),
    'system': (
        (PREFIX_A, ('prefix_a_alice_private_1', 'prefix_a_alice_private_2')),
        (PREFIX_B, ('prefix_b_alice_private', 'prefix_b_alice_bob_read')),
        (PREFIX_C, ('prefix_c_alice_team_read', 'prefix_c_bob_private')),
        *_PREFIX_D_MATCHES,
        (PREFIX_ABSENT, ()),
    ),
}


def _seed_ids_with_absent(db: ScopedDb) -> list[str]:
    """Return the ids of every seed row followed by the absent id."""
    return [*(db.ids[label] for label in SEED_LABELS), ABSENT_ID]


def _chunked_ids(db: ScopedDb) -> list[str]:
    """Return absent ids spanning two statement chunks, with the seed ids around the first boundary."""
    ids = [generate_id() for _ in range(_CHUNKED_ID_COUNT)]
    for position, label in zip(_CHUNKED_SEED_POSITIONS, SEED_LABELS, strict=True):
        ids[position] = db.ids[label]
    return ids


async def _read_labels(db: ScopedDb, scope: Scope, context_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the labels of the rows ``get_by_ids`` returns for the ids, in seed order."""
    rows = await db.repos.context.get_by_ids(list(context_ids), scope=scope)
    return in_seed_order(db.labels_of(str(row['id']) for row in rows))


async def _get_seed_and_absent(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
    """Read every seed row and the absent id as the scope."""
    return await _read_labels(db, scope, _seed_ids_with_absent(db))


async def _get_chunked(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
    """Read the seed rows through an id list that spans two statement chunks."""
    return await _read_labels(db, scope, _chunked_ids(db))


async def _read_back(db: ScopedDb, scope: Scope, context_ids: Sequence[str]) -> ReadBack:
    """Return the ``(visibility, is_owner)`` pair ``get_by_ids`` reports for each row it returns, keyed by label."""
    rows = await db.repos.context.get_by_ids(list(context_ids), scope=scope)
    labels = db.labels_of(str(row['id']) for row in rows)
    return {label: (row['visibility'], row['is_owner']) for label, row in zip(labels, rows, strict=True)}


async def _read_back_seed_and_absent(db: ScopedDb, scope: Scope) -> ReadBack:
    """Read back every seed row and the absent id as the scope."""
    return await _read_back(db, scope, _seed_ids_with_absent(db))


async def _read_back_chunked(db: ScopedDb, scope: Scope) -> ReadBack:
    """Read back the seed rows through an id list that spans two statement chunks."""
    return await _read_back(db, scope, _chunked_ids(db))


async def _add_prefixed_rows(db: ScopedDb) -> None:
    """Insert the prefixed rows under their fixed ids, with their grants."""
    for label, context_id, owner, visibility, grants in _PREFIXED_ROWS:
        await db.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (context_id, PREFIX_THREAD, SEED_SOURCE, 'text', f'Prefixed entry {label}', owner, visibility),
        )
        db.ids[label] = context_id
        for grant in grants:
            await insert_grant(db.seed_backend, context_id, grant.principal_type, grant.principal_id, grant.permission, owner)


async def _find_by_prefix(db: ScopedDb, scope: Scope) -> PrefixMatches:
    """Run a two-row prefix lookup of every prefix as the scope."""
    matches: list[tuple[str, tuple[str, ...]]] = []
    for prefix in PREFIXES:
        found = await db.repos.context.find_ids_by_prefix(prefix, limit=2, scope=scope)
        matches.append((prefix, tuple(db.labels_of(found))))
    return tuple(matches)


async def _resolve_prefixes(db: ScopedDb, scope: Scope) -> PrefixResolution:
    """Resolve every prefix as the scope to the matching label or the resolver's error message."""
    resolved: list[tuple[str, str]] = []
    for prefix in PREFIXES:
        try:
            context_id = await resolve_or_normalize_id(prefix, db.repos.context, scope=scope)
        except ValueError as error:
            resolved.append((prefix, str(error)))
        else:
            resolved.append((prefix, db.labels_of([context_id])[0]))
    return tuple(resolved)


def _expected_resolution(scope_name: str) -> PrefixResolution:
    """Turn the expected lookup of each prefix into the resolver's outcome."""
    resolution: list[tuple[str, str]] = []
    for prefix, labels in _PREFIX_MATCHES[scope_name]:
        if len(labels) == 1:
            resolution.append((prefix, labels[0]))
        elif labels:
            resolution.append((prefix, f'Ambiguous prefix {prefix!r} matches multiple entries'))
        else:
            resolution.append((prefix, f'No context entry matches prefix {prefix!r}'))
    return tuple(resolution)


async def _probe_each(db: ScopedDb, scope: Scope) -> tuple[ProbeRow, ...]:
    """Probe every seed row and the absent id as the scope."""
    observed: list[ProbeRow] = []
    for label in (*SEED_LABELS, ABSENT_LABEL):
        probe = await db.repos.context.check_entry_exists(db.ids.get(label, ABSENT_ID), scope=scope)
        observed.append((label, probe.exists, probe.source, probe.version, probe.owner_id, probe.can_write))
    return tuple(observed)


def _expected_probes(scope_name: str) -> tuple[ProbeRow, ...]:
    """A readable seed row probes with its source, version 0, owner and write access; anything else as missing."""
    missing = (False, None, None, None, False)
    observed: list[ProbeRow] = []
    for row in SEED_ROWS:
        if row.label in READABLE[scope_name]:
            observed.append((row.label, True, SEED_SOURCE, 0, row.owner, row.label in WRITABLE[scope_name]))
        else:
            observed.append((row.label, *missing))
    observed.append((ABSENT_LABEL, *missing))
    return tuple(observed)


async def _access_rows(
    db: ScopedDb, scope: Scope, context_ids: Sequence[str], *, in_transaction: bool,
) -> tuple[AccessRow, ...]:
    """Return ``(label, can_write, is_owner)`` for every id ``probe_ids`` reports, in seed order."""
    if in_transaction:
        async with db.backend.begin_transaction(access_scope=scope) as txn:
            access = await db.repos.context.probe_ids(list(context_ids), scope=scope, txn=txn)
    else:
        access = await db.repos.context.probe_ids(list(context_ids), scope=scope)
    by_label = dict(zip(db.labels_of(access), access.values(), strict=True))
    return tuple((label, by_label[label].can_write, by_label[label].is_owner) for label in in_seed_order(by_label))


async def _probe_ids(db: ScopedDb, scope: Scope) -> tuple[AccessRow, ...]:
    """Probe the seed ids and the absent id in one call as the scope."""
    return await _access_rows(db, scope, _seed_ids_with_absent(db), in_transaction=False)


async def _probe_ids_in_transaction(db: ScopedDb, scope: Scope) -> tuple[AccessRow, ...]:
    """Probe the seed ids and the absent id on a transaction connection as the scope."""
    return await _access_rows(db, scope, _seed_ids_with_absent(db), in_transaction=True)


async def _probe_ids_chunked(db: ScopedDb, scope: Scope) -> tuple[AccessRow, ...]:
    """Probe the seed ids through an id list that spans two statement chunks."""
    return await _access_rows(db, scope, _chunked_ids(db), in_transaction=False)


def _expected_access(scope_name: str) -> tuple[AccessRow, ...]:
    """Every readable seed row, with whether the scope may modify and owns it."""
    return tuple(
        (label, label in WRITABLE[scope_name], label in OWNED[scope_name]) for label in READABLE[scope_name]
    )


READ_CASES: tuple[AccessCase, ...] = (
    AccessCase('S3', _get_seed_and_absent, per_scope(READABLE.__getitem__)),
    AccessCase('S3', _get_chunked, per_scope(READABLE.__getitem__), variant='chunked'),
    AccessCase(
        'S3', _read_back_seed_and_absent, per_scope(expected_read_back), variant='read-back', reports_write_access=True,
    ),
    AccessCase(
        'S3', _read_back_chunked, per_scope(expected_read_back), variant='read-back-chunked', reports_write_access=True,
    ),
    AccessCase('S4', _find_by_prefix, per_scope(_PREFIX_MATCHES.__getitem__), setup=_add_prefixed_rows),
    AccessCase('S4', _resolve_prefixes, per_scope(_expected_resolution), variant='resolve', setup=_add_prefixed_rows),
    AccessCase('S5', _probe_each, per_scope(_expected_probes), reports_write_access=True),
    AccessCase('S7', _probe_ids, per_scope(_expected_access), reports_write_access=True),
    AccessCase(
        'S7', _probe_ids_in_transaction, per_scope(_expected_access), variant='in-transaction', reports_write_access=True,
    ),
    AccessCase('S7', _probe_ids_chunked, per_scope(_expected_access), variant='chunked', reports_write_access=True),
)
