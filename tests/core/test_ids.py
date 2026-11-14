"""Unit tests for app.ids module.

Covers the centralized identifier handling primitives:
  - ``generate_id()`` -- produces a 32-character lowercase hex UUIDv7.
  - ``generate_id_with_timestamp(created_at)`` -- deterministic generation
    that interprets the timestamp parameter in SECONDS (not milliseconds).
  - ``normalize_id()`` -- accepts 32-char hex and 36-char hyphenated UUIDs
    and always emits lowercase hex output.
  - ``is_id_prefix()`` -- predicate for 8 to 31 char hex prefixes.
  - ``id_prefix_bounds()`` -- the lowest and highest full id sharing a prefix,
    the bounds of the range read a prefix lookup runs on the id index.
  - ``resolve_prefix()`` -- repository-backed unique-prefix resolution, decided
    over the rows the caller's scope may read.
  - ``resolve_or_normalize_id()`` / ``resolve_or_normalize_ids()`` -- the tool
    boundary helpers that normalize full ids and resolve prefixes.
"""

import re
import time
import uuid
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pytest

from app.access_scope import AccessScope
from app.access_scope import Scope
from app.ids import generate_id
from app.ids import generate_id_with_timestamp
from app.ids import id_prefix_bounds
from app.ids import is_id_prefix
from app.ids import normalize_id
from app.ids import resolve_or_normalize_id
from app.ids import resolve_or_normalize_ids
from app.ids import resolve_prefix
from tests.helpers import LOCAL_SCOPE

HEX_32_RE = re.compile(r'^[0-9a-f]{32}$')


# ============================================================================
# generate_id()
# ============================================================================


class TestGenerateId:
    """Tests for ``generate_id()``."""

    def test_returns_32_char_lowercase_hex(self) -> None:
        """``generate_id()`` produces 32-char lowercase hex matching ``^[0-9a-f]{32}$``."""
        for _ in range(20):
            result = generate_id()
            assert isinstance(result, str)
            assert len(result) == 32
            assert HEX_32_RE.match(result), f'Result {result!r} does not match 32-char lowercase hex'

    def test_produces_unique_values(self) -> None:
        """Successive calls produce distinct values."""
        ids = {generate_id() for _ in range(50)}
        assert len(ids) == 50, 'Expected 50 unique IDs in 50 calls'

    def test_uuid_version_is_7(self) -> None:
        """The embedded UUID version is 7 (UUIDv7 per RFC 9562)."""
        result = generate_id()
        # Reconstruct hyphenated form for stdlib uuid parsing
        hyphenated = (
            f'{result[0:8]}-{result[8:12]}-{result[12:16]}-{result[16:20]}-{result[20:32]}'
        )
        parsed = uuid.UUID(hyphenated)
        assert parsed.version == 7

    def test_generation_is_monotonic(self) -> None:
        """UUIDv7 lex-string ordering matches creation time at millisecond granularity."""
        # Per RFC 9562 paragraph 5.7: UUIDv7 lex-string ordering matches creation
        # order at MILLISECOND granularity. Within the same millisecond, ordering
        # depends on the generator's own monotonic strategy, which nothing in the
        # project relies on; we sample IDs 2 ms apart and verify SORTED order at
        # that granularity.
        ids: list[str] = []
        for _ in range(5):
            ids.append(generate_id())
            time.sleep(0.002)  # 2 ms gap -> guaranteed cross-ms ordering
        assert ids == sorted(ids), f'IDs not monotonic across 2ms gaps: {ids}'


# ============================================================================
# generate_id_with_timestamp()
# ============================================================================


class TestGenerateIdWithTimestamp:
    """Tests for ``generate_id_with_timestamp()``.

    Verifies the timestamp parameter is interpreted in SECONDS (not
    milliseconds) and that the resulting UUIDv7 embeds the same calendar
    millisecond as the supplied :class:`datetime`.
    """

    def test_returns_32_char_lowercase_hex(self) -> None:
        """Result format matches ``^[0-9a-f]{32}$`` (same shape as ``generate_id``)."""
        result = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        assert HEX_32_RE.match(result)

    def test_uses_seconds_not_milliseconds(self) -> None:
        """The timestamp parameter is interpreted in seconds.

        If the implementation incorrectly multiplied the timestamp by 1000,
        the embedded UUIDv7 timestamp would land roughly in year ~50,000
        AD instead of the input year. This test extracts the 48-bit
        ``unix_ts_ms`` field from the UUID and asserts the decoded year
        matches the input year.

        UUIDv7 layout (RFC 9562):
          - bits 0..47:  ``unix_ts_ms`` (48-bit milliseconds since epoch)
          - bits 48..51: version (must be 7)
          - bits 52..63: ``random_a`` (12 bits)
          - bits 64..65: variant (10)
          - bits 66..127: ``random_b`` (62 bits)
        """
        created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        hex_id = generate_id_with_timestamp(created_at)
        # Reconstruct UUID and extract first 48 bits as ms-since-epoch
        as_int = int(hex_id, 16)
        unix_ts_ms = as_int >> 80  # top 48 bits
        # Convert ms back to a datetime for assertion
        decoded_dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=unix_ts_ms)
        # Year MUST be 2024 (NOT 50,000+ AD)
        assert decoded_dt.year == 2024, (
            f'Decoded year {decoded_dt.year} is not 2024 -- '
            f'generate_id_with_timestamp likely passed milliseconds to '
            f'uuid_utils.uuid7 instead of seconds.'
        )
        # Tighter assertion: same calendar second
        assert decoded_dt.replace(microsecond=0) == created_at.replace(microsecond=0), (
            f'Decoded datetime {decoded_dt} does not match input {created_at}'
        )

    def test_microseconds_propagate_via_nanos(self) -> None:
        """Microsecond precision in ``created_at`` is preserved at millisecond granularity in the UUID."""
        # 123 ms = 123000 microseconds (exactly representable as ms in the UUID timestamp)
        created_at = datetime(2024, 6, 15, 10, 30, 45, 123_000, tzinfo=UTC)
        hex_id = generate_id_with_timestamp(created_at)
        as_int = int(hex_id, 16)
        unix_ts_ms = as_int >> 80
        decoded_dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=unix_ts_ms)
        # Within 1 ms of input
        delta_ms = abs((decoded_dt - created_at).total_seconds() * 1000)
        assert delta_ms <= 1.0, f'Decoded {decoded_dt} differs from input {created_at} by {delta_ms} ms'

    def test_uuid_version_is_7(self) -> None:
        """Generated UUID has version 7."""
        hex_id = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        hyphenated = (
            f'{hex_id[0:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:32]}'
        )
        parsed = uuid.UUID(hyphenated)
        assert parsed.version == 7


# ============================================================================
# normalize_id()
# ============================================================================


class TestNormalizeId:
    """Tests for ``normalize_id()``.

    Verifies that output is always 32-char lowercase hex regardless of
    input case or hyphenation, and that invalid inputs raise
    :class:`ValueError`.
    """

    def test_accepts_32_char_lowercase_hex(self) -> None:
        """32-char lowercase hex passes through unchanged."""
        v = '0190abcdef1234567890abcdef123456'
        assert normalize_id(v) == v

    def test_accepts_36_char_hyphenated_uuid(self) -> None:
        """36-char hyphenated UUID is converted to 32-char hex."""
        v = '0190abcd-ef12-3456-7890-abcdef123456'
        assert normalize_id(v) == '0190abcdef1234567890abcdef123456'

    def test_lowercase_invariant_enforced(self) -> None:
        """Output is always lowercase regardless of input case.

        Lowercase output is required because SQLite TEXT BINARY collation
        sorts uppercase ``A-F`` before lowercase ``a-f``; any ``id > ?``
        ordering comparison in the storage layer would behave incorrectly
        if uppercase IDs were accepted.
        """
        # Mixed-case 32-char hex
        result = normalize_id('0190ABCDEF1234567890abcdef123456')
        assert result == '0190abcdef1234567890abcdef123456'
        assert result == result.lower()
        assert all(c in '0123456789abcdef' for c in result)
        # Mixed-case 36-char hyphenated
        result2 = normalize_id('0190ABCD-EF12-3456-7890-ABCDEF123456')
        assert result2 == '0190abcdef1234567890abcdef123456'
        assert result2 == result2.lower()

    def test_strips_whitespace(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert normalize_id('  0190abcdef1234567890abcdef123456  ') == '0190abcdef1234567890abcdef123456'
        assert normalize_id('\t0190abcd-ef12-3456-7890-abcdef123456\n') == '0190abcdef1234567890abcdef123456'

    def test_idempotent_on_normalized_input(self) -> None:
        """``normalize_id(normalize_id(x)) == normalize_id(x)``."""
        v = '0190abcd-ef12-3456-7890-abcdef123456'
        once = normalize_id(v)
        twice = normalize_id(once)
        assert once == twice

    def test_rejects_too_short(self) -> None:
        """31-char hex raises :class:`ValueError`."""
        with pytest.raises(ValueError, match='Invalid UUID identifier length'):
            normalize_id('0190abcdef1234567890abcdef12345')  # 31 chars

    def test_rejects_too_long(self) -> None:
        """33-char hex raises :class:`ValueError`."""
        with pytest.raises(ValueError, match='Invalid UUID identifier length'):
            normalize_id('0190abcdef1234567890abcdef1234567')  # 33 chars

    def test_rejects_non_hex_character(self) -> None:
        """32-char string with a non-hex character raises :class:`ValueError`."""
        with pytest.raises(ValueError, match='non-hex character'):
            normalize_id('0190abcdef1234567890abcdef12345Z')  # Z is not hex (lowercased to 'z')

    def test_rejects_wrong_hyphenated_form(self) -> None:
        """36-char string with wrong hyphen positions raises :class:`ValueError`."""
        # 36 chars but only 3 hyphens -> s.count('-') != 4 -> branch not taken -> length check fails
        with pytest.raises(ValueError, match='Invalid UUID identifier length'):
            normalize_id('0190abcdef1234567890abcdef12345-6789-abcd')

    def test_rejects_empty_string(self) -> None:
        """Empty string raises :class:`ValueError`."""
        with pytest.raises(ValueError, match='Invalid UUID identifier length'):
            normalize_id('')

    def test_rejects_integer_string(self) -> None:
        """A short integer-style string (not 32 hex chars) is rejected."""
        with pytest.raises(ValueError, match='Invalid UUID identifier length'):
            normalize_id('8944')


# ============================================================================
# is_id_prefix()
# ============================================================================


class TestIsIdPrefix:
    """Tests for ``is_id_prefix()``."""

    def test_8_char_hex_is_prefix(self) -> None:
        """Exactly 8 hex chars is a valid prefix."""
        assert is_id_prefix('0190abcd') is True

    def test_31_char_hex_is_prefix(self) -> None:
        """Exactly 31 hex chars is a valid prefix."""
        assert is_id_prefix('0190abcdef1234567890abcdef12345') is True

    def test_7_char_hex_is_not_prefix(self) -> None:
        """7 hex chars is below the 8-char minimum."""
        assert is_id_prefix('0190abc') is False

    def test_32_char_hex_is_not_prefix(self) -> None:
        """Exactly 32 hex chars is a full UUID, not a prefix."""
        assert is_id_prefix('0190abcdef1234567890abcdef123456') is False

    def test_36_char_hyphenated_is_not_prefix(self) -> None:
        """Hyphenated UUIDs are not prefixes (length 36 exceeds the 31-char ceiling)."""
        assert is_id_prefix('0190abcd-ef12-3456-7890-abcdef123456') is False

    def test_uppercase_hex_recognized(self) -> None:
        """Case-insensitive hex recognition (after internal lowercasing)."""
        assert is_id_prefix('0190ABCD') is True

    def test_non_hex_character_rejected(self) -> None:
        """Non-hex character disqualifies even if length is correct."""
        assert is_id_prefix('0190abcZ') is False  # Z is not hex (z after lowercasing)

    def test_empty_string_rejected(self) -> None:
        """Empty string is not a prefix."""
        assert is_id_prefix('') is False

    def test_whitespace_stripped(self) -> None:
        """Surrounding whitespace is stripped before length and charset checks."""
        assert is_id_prefix('  0190abcd  ') is True


# ============================================================================
# id_prefix_bounds()
# ============================================================================


_HEX_DIGITS = '0123456789abcdef'


class TestIdPrefixBounds:
    """Tests for ``id_prefix_bounds()``: the inclusive range of full ids that start with a prefix."""

    def test_eight_characters_pad_to_the_lowest_and_highest_id(self) -> None:
        """The shortest prefix pads its remaining 24 digits with ``0`` and with ``f``."""
        assert id_prefix_bounds('0190abcd') == ('0190abcd' + '0' * 24, '0190abcd' + 'f' * 24)

    def test_thirty_one_characters_leave_one_digit(self) -> None:
        """The longest prefix leaves only the last digit open, ``0`` at the low end and ``f`` at the high end."""
        prefix = '0190abcdef1234567890abcdef12345'
        assert id_prefix_bounds(prefix) == (prefix + '0', prefix + 'f')

    def test_odd_length_pads_from_the_next_digit(self) -> None:
        """A prefix ending inside a byte pads from the very next hex digit, not from the next byte."""
        assert id_prefix_bounds('0190abcd1') == ('0190abcd1' + '0' * 23, '0190abcd1' + 'f' * 23)

    def test_case_and_whitespace_are_folded(self) -> None:
        """Surrounding whitespace is stripped and uppercase hex is folded, so both bounds are canonical ids."""
        assert id_prefix_bounds('  0190ABCD  ') == ('0190abcd' + '0' * 24, '0190abcd' + 'f' * 24)

    def test_every_id_with_the_prefix_lies_within_the_bounds(self) -> None:
        """Ids sharing the prefix lie inside the bounds and the nearest ids with another prefix lie outside.

        The bounds are checked both as the hex text SQLite compares and as the ``uuid`` values PostgreSQL
        compares. For each prefix, the ids whose last prefix digit is one lower or one higher are the
        closest ids that do not share it: the highest id below the range and the lowest id above it.
        """
        sample = [generate_id() for _ in range(20)]
        sample += ['0' * 32, 'f' * 32, '0190abcdef1234567890abcdef123456']
        for full_id in sample:
            for length in range(8, 32):
                prefix = full_id[:length]
                low, high = id_prefix_bounds(prefix)
                assert HEX_32_RE.match(low)
                assert HEX_32_RE.match(high)
                assert low.startswith(prefix)
                assert high.startswith(prefix)
                assert low <= full_id <= high
                assert uuid.UUID(hex=low) <= uuid.UUID(hex=full_id) <= uuid.UUID(hex=high)

                last = _HEX_DIGITS.index(prefix[-1])
                if last > 0:
                    below = (prefix[:-1] + _HEX_DIGITS[last - 1]).ljust(32, 'f')
                    assert below < low
                    assert uuid.UUID(hex=below) < uuid.UUID(hex=low)
                if last < 15:
                    above = (prefix[:-1] + _HEX_DIGITS[last + 1]).ljust(32, '0')
                    assert above > high
                    assert uuid.UUID(hex=above) > uuid.UUID(hex=high)

    @pytest.mark.parametrize(
        'value',
        [
            pytest.param('0190abc', id='seven-characters'),
            pytest.param('0190abcdef1234567890abcdef123456', id='full-id'),
            pytest.param('0190abcd-ef12-3456-7890-abcdef123456', id='hyphenated-id'),
            pytest.param('0190abcd-ef12', id='hyphenated-fragment'),
            pytest.param('0190abcZ', id='non-hex-character'),
        ],
    )
    def test_rejects_what_is_not_a_prefix(self, value: str) -> None:
        """Anything other than an 8-31 character hex prefix raises :class:`ValueError` naming the input."""
        with pytest.raises(ValueError, match=re.escape(f'Invalid UUID prefix (must be 8-31 hex chars): {value!r}')):
            id_prefix_bounds(value)


# ============================================================================
# resolve_prefix() -- async; uses a stub repo
# ============================================================================


class _StubRepo:
    """In-memory stub of the prefix-resolver repository protocol that records each lookup's scope.

    ``readable`` limits what a scope may match, keyed by principal id: a scope whose
    principal is listed matches only those ids; any other scope matches every id.
    """

    def __init__(self, ids: list[str], readable: dict[str, list[str]] | None = None) -> None:
        self._ids = ids
        self._readable = readable or {}
        self.scopes: list[Scope] = []

    async def find_ids_by_prefix(self, prefix: str, limit: int = 2, *, scope: Scope) -> list[str]:
        self.scopes.append(scope)
        visible = self._ids
        if isinstance(scope, AccessScope) and scope.principal_id in self._readable:
            visible = self._readable[scope.principal_id]
        matches = [i for i in visible if i.startswith(prefix)]
        return matches[:limit]


class TestResolvePrefix:
    """Tests for ``resolve_prefix()`` -- async repository-backed lookup."""

    @pytest.mark.asyncio
    async def test_unique_prefix_resolves(self) -> None:
        """A prefix matching exactly one ID returns that full ID."""
        repo = _StubRepo([
            '0190abcdef1234567890abcdef123456',
            '0190ffffeeeeeeeeeeeeeeeeeeeeeeee',
        ])
        result = await resolve_prefix('0190abcd', repo, scope=LOCAL_SCOPE)
        assert result == '0190abcdef1234567890abcdef123456'

    @pytest.mark.asyncio
    async def test_ambiguous_prefix_raises(self) -> None:
        """A prefix matching multiple IDs raises :class:`ValueError` with 'Ambiguous'."""
        repo = _StubRepo([
            '0190abcdef1234567890abcdef123456',
            '0190abcdaabbccddeeff0011223344556',  # also starts with '0190abcd'
        ])
        with pytest.raises(ValueError, match='Ambiguous prefix'):
            await resolve_prefix('0190abcd', repo, scope=LOCAL_SCOPE)

    @pytest.mark.asyncio
    async def test_no_match_raises(self) -> None:
        """A prefix matching zero IDs raises :class:`ValueError` with 'No context entry'."""
        repo = _StubRepo([
            '0190ffffeeeeeeeeeeeeeeeeeeeeeeee',
        ])
        with pytest.raises(ValueError, match='No context entry matches prefix'):
            await resolve_prefix('0190abcd', repo, scope=LOCAL_SCOPE)

    @pytest.mark.asyncio
    async def test_short_prefix_rejected(self) -> None:
        """A prefix shorter than 8 chars raises :class:`ValueError` before querying the repo."""
        repo = _StubRepo(['0190abcdef1234567890abcdef123456'])
        with pytest.raises(ValueError, match='Invalid UUID prefix'):
            await resolve_prefix('0190abc', repo, scope=LOCAL_SCOPE)  # 7 chars
        assert repo.scopes == []

    @pytest.mark.asyncio
    async def test_full_uuid_rejected(self) -> None:
        """A 32-char full UUID is rejected (callers must use ``normalize_id`` instead)."""
        repo = _StubRepo(['0190abcdef1234567890abcdef123456'])
        with pytest.raises(ValueError, match='Invalid UUID prefix'):
            await resolve_prefix('0190abcdef1234567890abcdef123456', repo, scope=LOCAL_SCOPE)

    @pytest.mark.asyncio
    async def test_non_hex_prefix_rejected(self) -> None:
        """A prefix with non-hex characters raises :class:`ValueError` before querying the repo."""
        repo = _StubRepo(['0190abcdef1234567890abcdef123456'])
        with pytest.raises(ValueError, match='Invalid UUID prefix'):
            await resolve_prefix('0190abcZ', repo, scope=LOCAL_SCOPE)  # Z is not hex

    @pytest.mark.asyncio
    async def test_uppercase_prefix_normalized(self) -> None:
        """An uppercase-hex prefix is folded to lowercase before querying the repo."""
        repo = _StubRepo([
            '0190abcdef1234567890abcdef123456',
        ])
        result = await resolve_prefix('0190ABCD', repo, scope=LOCAL_SCOPE)
        assert result == '0190abcdef1234567890abcdef123456'

    @pytest.mark.asyncio
    async def test_lookup_runs_under_the_given_scope(self) -> None:
        """The repository lookup receives exactly the scope the caller passed."""
        bob = AccessScope('bob', frozenset({'team-x'}))
        repo = _StubRepo(['0190abcdef1234567890abcdef123456'])
        await resolve_prefix('0190abcd', repo, scope=bob)
        assert repo.scopes == [bob]

    @pytest.mark.asyncio
    async def test_match_set_is_the_scoped_lookup(self) -> None:
        """Uniqueness and no-match are decided over what the scope may read, not over every stored id."""
        visible_id = '0190abcdef1234567890abcdef123456'
        hidden_id = '0190abcdaabbccddeeff001122334455'
        repo = _StubRepo([visible_id, hidden_id], readable={'bob': [visible_id], 'dave': []})

        with pytest.raises(ValueError, match='Ambiguous prefix'):
            await resolve_prefix('0190abcd', repo, scope=LOCAL_SCOPE)
        assert await resolve_prefix('0190abcd', repo, scope=AccessScope('bob', frozenset())) == visible_id
        with pytest.raises(ValueError, match="No context entry matches prefix '0190abcd'"):
            await resolve_prefix('0190abcd', repo, scope=AccessScope('dave', frozenset()))


class TestResolveOrNormalizeId:
    """Tests for the tool-boundary helpers ``resolve_or_normalize_id()`` and ``resolve_or_normalize_ids()``."""

    @pytest.mark.asyncio
    async def test_full_id_is_normalized_without_a_lookup(self) -> None:
        """A full id is canonicalized locally; the repository is never asked."""
        repo = _StubRepo([])
        result = await resolve_or_normalize_id('  0190ABCDEF1234567890ABCDEF123456 ', repo, scope=LOCAL_SCOPE)
        assert result == '0190abcdef1234567890abcdef123456'
        assert repo.scopes == []

    @pytest.mark.asyncio
    async def test_prefix_resolves_under_the_given_scope(self) -> None:
        """A prefix resolves through the repository under the caller's scope."""
        bob = AccessScope('bob', frozenset())
        repo = _StubRepo(['0190abcdef1234567890abcdef123456'])
        assert await resolve_or_normalize_id('0190abcd', repo, scope=bob) == '0190abcdef1234567890abcdef123456'
        assert repo.scopes == [bob]

    @pytest.mark.asyncio
    async def test_list_form_keeps_order_and_forwards_the_scope_per_prefix(self) -> None:
        """Every prefix of the list resolves under the same scope; full ids need no lookup."""
        bob = AccessScope('bob', frozenset())
        repo = _StubRepo(['0190abcdef1234567890abcdef123456', '0190ffffeeeeeeeeeeeeeeeeeeeeeeee'])
        result = await resolve_or_normalize_ids(
            ['0190ffff', '0190ABCDEF1234567890ABCDEF123456', '0190abcd'], repo, scope=bob,
        )
        assert result == [
            '0190ffffeeeeeeeeeeeeeeeeeeeeeeee',
            '0190abcdef1234567890abcdef123456',
            '0190abcdef1234567890abcdef123456',
        ]
        assert repo.scopes == [bob, bob]
