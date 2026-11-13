"""The answer a short-id lookup owes, computed from the stored rows and grants alone.

The SQLite and PostgreSQL plan tests of ``find_ids_by_prefix`` compare the lookup with it,
so the comparison does not rest on the access predicate or the statement under test.
"""

from collections.abc import Iterable

from app.access_scope import AccessScope
from app.ids import resolve_or_normalize_id
from tests.repositories._access_scope_cases import ScopedDb

# The prefix lengths the comparison cuts from each sampled id: both ends of the accepted
# 8-31 range, odd and even lengths, and lengths inside every group of the hyphenated form.
PREFIX_LENGTHS = (8, 9, 12, 16, 23, 31)
# The length of the prefix the case-folding step sends.
_FOLDED_PREFIX_LENGTH = 16


def prefixes_of(context_ids: Iterable[str]) -> set[str]:
    """Return every prefix of :data:`PREFIX_LENGTHS` characters cut from the ids."""
    return {context_id[:length] for context_id in context_ids for length in PREFIX_LENGTHS}


async def readable_ids(db: ScopedDb, scope: AccessScope) -> list[str]:
    """Return every id the scope may read, in id order, read through the seed channel.

    An entry is readable when the scope's principal owns it, when it is public, or when a
    grant of either permission names the principal or one of its groups.

    Returns:
        The readable ids as 32-character lowercase hex, sorted.
    """
    grants = await db.fetch_all('SELECT context_entry_id, principal_type, principal_id FROM context_entry_grants')
    granted = {
        str(context_id)
        for context_id, principal_type, principal_id in grants
        if (principal_type == 'user' and principal_id == scope.principal_id)
        or (principal_type == 'group' and principal_id in scope.groups)
    }
    rows = await db.fetch_all('SELECT id, owner_id, visibility FROM context_entries')
    return sorted(
        str(context_id)
        for context_id, owner, visibility in rows
        if owner == scope.principal_id or visibility == 'public' or str(context_id) in granted
    )


def _unique_prefix_with_a_letter(readable: list[str]) -> str:
    """Return a prefix exactly one readable id starts with and whose case folding changes it."""
    length = _FOLDED_PREFIX_LENGTH
    for position, context_id in enumerate(readable):
        prefix = context_id[:length]
        shared_before = position > 0 and readable[position - 1].startswith(prefix)
        shared_after = position + 1 < len(readable) and readable[position + 1].startswith(prefix)
        if not shared_before and not shared_after and prefix != prefix.upper():
            return prefix
    raise AssertionError('no readable id carries a prefix of its own with a hex letter in it')


async def assert_lookup_matches_readable_ids(db: ScopedDb, scope: AccessScope, prefixes: Iterable[str]) -> None:
    """Assert the lookup returns the first two readable ids that start with each prefix, and folds case.

    Every prefix is looked up as the scope with ``limit=2`` through the request channel and
    compared with the readable ids that start with it, in id order. Then a prefix exactly
    one readable id carries is resolved twice through
    :func:`app.ids.resolve_or_normalize_id`, as given and uppercase between spaces, and
    both must name that id.

    Args:
        db: The database, seeded.
        scope: The caller.
        prefixes: Lowercase hex prefixes of 8-31 characters.
    """
    readable = await readable_ids(db, scope)
    compared = sorted(set(prefixes))
    assert compared, 'no prefix to compare'
    for prefix in compared:
        found = await db.repos.context.find_ids_by_prefix(prefix, limit=2, scope=scope)
        expected = [context_id for context_id in readable if context_id.startswith(prefix)][:2]
        assert found == expected, (prefix, found, expected)

    prefix = _unique_prefix_with_a_letter(readable)
    carrier = next(context_id for context_id in readable if context_id.startswith(prefix))
    resolved = await resolve_or_normalize_id(prefix, db.repos.context, scope=scope)
    folded = await resolve_or_normalize_id(f'  {prefix.upper()}  ', db.repos.context, scope=scope)
    assert (resolved, folded) == (carrier, carrier), (prefix, resolved, folded)
