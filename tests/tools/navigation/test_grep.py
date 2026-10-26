"""Tool-level tests for grep_context (SQLite backend): output modes, Unicode-aware case-insensitivity,
thread scoping, bounded output, regex handling and its aggregate timeout, and server-side clamps of
client-requested bounds.
"""

import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.repositories.context_repository.records import GrepScanRow
from app.services.grep_service import GrepEntryResult
from app.startup import ensure_repositories
from app.tools.navigation.grep import grep_context
from tests.helpers import as_principal
from tests.tools._generation_jobs import record_state_reads
from tests.tools._generation_jobs import seed_generation_jobs
from tests.tools.navigation._navigation import grep_as_dict
from tests.tools.navigation._navigation import store_entry

if TYPE_CHECKING:
    from app.settings import AppSettings


class TestGrepContextModes:
    """Output modes and their row shapes."""

    @pytest.mark.asyncio
    async def test_files_with_matches_default(self, nav_backend: StorageBackend) -> None:
        hit = await store_entry(nav_backend, 'the needle is here', offset_seconds=1)
        await store_entry(nav_backend, 'only hay', offset_seconds=2)
        result = await grep_as_dict(pattern='needle', thread_id='t')
        assert result['mode'] == 'files_with_matches'
        ids = {row['context_id'] for row in result['results']}
        assert ids == {hit}

    @pytest.mark.asyncio
    async def test_content_mode_returns_offsets_and_context(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'first line\nsecond needle line\nthird line')
        result = await grep_as_dict(pattern='needle', thread_id='t', output_mode='content', context_lines=1)
        assert result['total_matches'] == 1
        match = result['results'][0]
        assert match['line_number'] == 2
        assert match['line'] == 'second needle line'
        assert match['before'] == ['first line']
        assert match['after'] == ['third line']
        assert match['match_end'] > match['match_start']

    @pytest.mark.asyncio
    async def test_count_mode(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'x x x on one line\nand x again')
        result = await grep_as_dict(pattern='x', thread_id='t', output_mode='count', case_sensitive=True)
        assert result['results'][0]['count'] == 4


class TestGrepContextMatching:
    """Matching semantics: case, literal vs regex, Unicode, scoping."""

    @pytest.mark.asyncio
    async def test_case_insensitive_default(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'Contains ERROR token')
        result = await grep_as_dict(pattern='error', thread_id='t')
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_case_sensitive_excludes(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'Contains ERROR token')
        result = await grep_as_dict(pattern='error', thread_id='t', case_sensitive=True)
        assert result['results'] == []

    @pytest.mark.asyncio
    async def test_cyrillic_case_insensitive(self, nav_backend: StorageBackend) -> None:
        # Stored lower-case Cyrillic; query upper-case. Requires Python re
        # IGNORECASE (the ASCII-only SQL pre-narrow is skipped for non-ASCII).
        lower = ''.join(chr(c) for c in (0x043F, 0x0440, 0x0438, 0x0432, 0x0435, 0x0442))
        upper = lower.upper()
        await store_entry(nav_backend, f'message: {lower}')
        result = await grep_as_dict(pattern=upper, thread_id='t')
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_literal_does_not_treat_dot_as_wildcard(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'value axb here')
        result = await grep_as_dict(pattern='a.b', thread_id='t')
        assert result['results'] == []

    @pytest.mark.asyncio
    async def test_regex_mode(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'value axb here')
        result = await grep_as_dict(pattern='a.b', thread_id='t', is_regex=True)
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_invalid_regex_raises(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'anything')
        with pytest.raises(ToolError):
            await grep_as_dict(pattern='a(b', thread_id='t', is_regex=True)

    @pytest.mark.asyncio
    async def test_thread_scoping(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'needle in t', thread_id='t')
        await store_entry(nav_backend, 'needle in other', thread_id='other')
        result = await grep_as_dict(pattern='needle', thread_id='t')
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_max_matches_truncates(self, nav_backend: StorageBackend) -> None:
        await store_entry(nav_backend, 'm\nm\nm\nm\nm')
        result = await grep_as_dict(pattern='m', thread_id='t', output_mode='content', max_matches=2, case_sensitive=True)
        assert result['total_matches'] == 2
        assert result['truncated'] is True

    @pytest.mark.asyncio
    async def test_no_spurious_truncation_at_exact_cap(self, nav_backend: StorageBackend) -> None:
        # An entry with exactly max_matches matches, then a trailing (older,
        # visited-after) candidate with ZERO matches, must NOT report truncated --
        # a zero-match candidate is not a dropped match.
        await store_entry(nav_backend, 'no hits here', thread_id='t', offset_seconds=0)  # 0 matches, visited last
        await store_entry(nav_backend, 'm\nm', thread_id='t', offset_seconds=1)          # 2 matches, visited first
        result = await grep_as_dict(pattern='m', thread_id='t', output_mode='content', max_matches=2, case_sensitive=True)
        assert result['total_matches'] == 2
        assert result['truncated'] is False

    @pytest.mark.asyncio
    async def test_truncation_when_real_extra_match_exists(self, nav_backend: StorageBackend) -> None:
        # A genuine extra match beyond the cap (in a later entry) -> truncated True,
        # output trimmed back to exactly max_matches.
        await store_entry(nav_backend, 'm', thread_id='t', offset_seconds=0)     # 1 match, visited last
        await store_entry(nav_backend, 'm\nm', thread_id='t', offset_seconds=1)  # 2 matches, visited first
        result = await grep_as_dict(pattern='m', thread_id='t', output_mode='content', max_matches=2, case_sensitive=True)
        assert result['total_matches'] == 2
        assert result['truncated'] is True

    @pytest.mark.asyncio
    async def test_regex_timeout_surfaced_in_response(
        self, nav_backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A regex-timed-out entry is skipped (match_count 0) but surfaced via
        # timed_out_context_ids so the caller knows the result is incomplete for it.
        cid = await store_entry(nav_backend, 'aaaa', thread_id='t')

        async def _timeout(context_id: str, *_args: Any, **_kwargs: Any) -> GrepEntryResult:
            return GrepEntryResult(context_id=context_id, matches=(), match_count=0, timed_out=True)

        monkeypatch.setattr('app.tools.navigation.grep.match_entry', _timeout)
        result = await grep_as_dict(pattern='a+', thread_id='t', is_regex=True)
        assert result.get('timed_out_context_ids') == [cid]


class _FakeClock:
    """Deterministic ``monotonic`` source for the aggregate-deadline tests."""

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._i = 0

    def monotonic(self) -> float:
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


class TestGrepRegexAggregateTimeout:
    """The is_regex scan is bounded by an AGGREGATE wall-clock budget, not only the
    per-entry timeout: it stops early with truncated=True instead of running the
    per-entry timeout against every one of up to max_entries_scanned rows."""

    @pytest.mark.asyncio
    async def test_aggregate_deadline_stops_scan_and_flags_truncated(
        self, nav_backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Three matching entries; a fake clock under the deadline for the first
        # entry's pre-check and past it for the second, so the scan stops after ONE
        # entry with truncated=True even though more matches exist. Calls:
        # [deadline-calc, iter1 pre-check (under), iter2 pre-check (over)].
        for i in range(3):
            await store_entry(nav_backend, f'entry {i} HIT', thread_id='tdl', offset_seconds=i + 1)
        monkeypatch.setattr('app.tools.navigation.grep.time', _FakeClock([100.0, 100.0, 200.0]))

        result = await grep_as_dict(pattern='HIT', thread_id='tdl', is_regex=True, case_sensitive=True)

        assert result['truncated'] is True
        # Only the first scanned entry was matched before the budget fired; without
        # the aggregate budget all three would be scanned (results len 3).
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_regex_scan_under_budget_is_not_falsely_truncated(
        self, nav_backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A clock that never crosses the deadline must NOT truncate a normal regex
        # scan: all three entries are scanned and truncated stays False.
        for i in range(3):
            await store_entry(nav_backend, f'entry {i} HIT', thread_id='tub', offset_seconds=i + 1)
        monkeypatch.setattr('app.tools.navigation.grep.time', _FakeClock([100.0]))

        result = await grep_as_dict(pattern='HIT', thread_id='tub', is_regex=True, case_sensitive=True)

        assert result['truncated'] is False
        assert len(result['results']) == 3


class TestGrepCaseFoldParity:
    """The ASCII pre-narrow must not drop non-ASCII case-folds on the default
    case-insensitive path."""

    @pytest.mark.asyncio
    async def test_case_insensitive_ascii_letter_matches_unicode_fold(self, nav_backend: StorageBackend) -> None:
        # The stored text's only 's'-like code point is U+017F (LONG S), which
        # Python re.IGNORECASE folds to 's'. A case-insensitive grep for 's' must
        # still find it: the ASCII LIKE pre-narrow is skipped for ASCII letters
        # under case-insensitive matching, falling back to a full Python scan.
        # A SQL LIKE '%s%' pre-narrow would exclude this row (0 hits).
        await store_entry(nav_backend, 'meaſure')
        result = await grep_as_dict(pattern='s', thread_id='t', output_mode='content')
        assert result['total_matches'] == 1


class TestGrepServerSideClamps:
    """grep_context clamps client-requested bounds to the configured server caps.

    The wire schema deliberately admits values far above the defaults
    (max_matches up to 10000, context_lines up to 100, max_entries_scanned up to
    1000000) so an operator can raise the caps; the tool body then clamps every
    request to whatever the server is actually configured for. Without a test
    passing an ABOVE-CAP value, pass-through and clamping are indistinguishable,
    and dropping a clamp would let one call return ten thousand matches.
    """

    @staticmethod
    def _settings_with_grep_caps(
        *, max_matches: int, context_lines: int, entries_scanned: int,
    ) -> 'AppSettings':
        """Build a settings copy with tightened grep caps.

        Returns:
            An AppSettings copy whose grep_context caps are the given values.
        """
        from app.settings import get_settings

        base = get_settings()
        return base.model_copy(update={
            'grep_context': base.grep_context.model_copy(update={
                'max_matches_cap': max_matches,
                'max_context_lines': context_lines,
                'max_entries_scanned': entries_scanned,
            }),
        })

    @pytest.mark.asyncio
    async def test_above_cap_request_is_clamped_to_server_caps(
        self, nav_backend: StorageBackend,
    ) -> None:
        """max_matches and context_lines are clamped down to their configured caps."""
        import app.tools.navigation.grep as grep_module

        # Five entries, each with five matching lines separated by filler lines.
        for _ in range(5):
            await store_entry(nav_backend, '\n'.join(['needle', 'filler'] * 5))

        tightened = self._settings_with_grep_caps(
            max_matches=3, context_lines=1, entries_scanned=100,
        )
        captured: dict[str, Any] = {}
        from app.services.grep_service import match_entry as real_match_entry

        async def spy_match_entry(*args: Any, **kwargs: Any) -> GrepEntryResult:
            captured.setdefault('context_lines', kwargs['context_lines'])
            return await real_match_entry(*args, **kwargs)

        with (
            patch.object(grep_module, 'settings', tightened),
            patch.object(grep_module, 'match_entry', spy_match_entry),
        ):
            result = await grep_context(
                pattern='needle',
                thread_id='t',
                output_mode='content',
                max_matches=10000,
                context_lines=100,
                max_entries_scanned=1000000,
            )

        payload = cast('dict[str, Any]', result)
        # Clamped to the cap, not the requested 10000.
        assert payload['total_matches'] == 3
        assert payload['truncated'] is True
        assert len(payload['results']) == 3
        # context_lines clamped to the cap before reaching the matcher.
        assert captured['context_lines'] == 1
        for row in payload['results']:
            assert len(row['before']) <= 1
            assert len(row['after']) <= 1

    @pytest.mark.asyncio
    async def test_max_entries_scanned_is_clamped(self, nav_backend: StorageBackend) -> None:
        """The scan-width request is clamped before it reaches the repository."""
        import app.tools.navigation.grep as grep_module

        for _ in range(5):
            await store_entry(nav_backend, 'needle here')

        tightened = self._settings_with_grep_caps(
            max_matches=1000, context_lines=20, entries_scanned=2,
        )
        captured: dict[str, Any] = {}
        repos = await ensure_repositories()
        real_scan = repos.context.grep_scan_text_contents

        async def spy_scan(**kwargs: Any) -> tuple[list[GrepScanRow], dict[str, Any]]:
            captured['max_entries_scanned'] = kwargs['max_entries_scanned']
            return await real_scan(**kwargs)

        with (
            patch.object(grep_module, 'settings', tightened),
            patch.object(repos.context, 'grep_scan_text_contents', spy_scan),
        ):
            await grep_context(pattern='needle', thread_id='t', max_entries_scanned=1000000)

        assert captured['max_entries_scanned'] == 2


async def _publish(backend: StorageBackend, context_id: str) -> None:
    """Make one stored entry public."""

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE context_entries SET visibility = 'public' WHERE id = ?", (context_id,))

    await backend.execute_write(_write, access_scope=MAINTENANCE)


class TestGrepContextReadBack:
    """Every result row reports its entry's visibility and whether the caller owns the entry, never the owner."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('output_mode', ['files_with_matches', 'count', 'content'])
    async def test_rows_carry_visibility_and_ownership(self, nav_backend: StorageBackend, output_mode: str) -> None:
        """The caller's private entry reads back as owned and alice's public entry as not owned, in every mode."""
        own = await store_entry(nav_backend, 'own needle\nanother needle', offset_seconds=1)
        foreign = await store_entry(nav_backend, 'public needle', offset_seconds=2, owner='alice')
        await _publish(nav_backend, foreign)

        payload = await grep_as_dict(pattern='needle', thread_id='t', output_mode=output_mode)

        expected = {own: ('private', True), foreign: ('public', False)}
        rows = payload['results']
        assert len(rows) == (3 if output_mode == 'content' else 2)
        for row in rows:
            assert (row['visibility'], row['is_owner']) == expected[row['context_id']]
            assert type(row['is_owner']) is bool
            assert 'owner_id' not in row

    @pytest.mark.asyncio
    @pytest.mark.parametrize('output_mode', ['files_with_matches', 'count', 'content'])
    async def test_rows_carry_generation_state(
        self, nav_backend: StorageBackend, output_mode: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every row reports its entry's generation state, read once for the matched entries only."""
        ready = await store_entry(nav_backend, 'ready needle\nanother needle', offset_seconds=1)
        pending = await store_entry(nav_backend, 'pending needle', offset_seconds=2)
        failed = await store_entry(nav_backend, 'failed needle', offset_seconds=3)
        unmatched = await store_entry(nav_backend, 'only hay', offset_seconds=4)
        await seed_generation_jobs(nav_backend, {pending: 'running', failed: 'failed', unmatched: 'pending'})
        reads = record_state_reads(monkeypatch, await ensure_repositories())

        payload = await grep_as_dict(pattern='needle', thread_id='t', output_mode=output_mode)

        expected = {ready: 'ready', pending: 'pending', failed: 'failed'}
        rows = payload['results']
        assert len(rows) == (4 if output_mode == 'content' else 3)
        for row in rows:
            assert row['generation_status'] == expected[row['context_id']]
        assert [sorted(read) for read in reads] == [sorted(expected)]


class TestGrepContextScoping:
    """grep_context scans only the entries the caller may read."""

    @pytest.mark.asyncio
    async def test_unreadable_entry_never_matches(self, nav_backend: StorageBackend) -> None:
        """Another principal's private entry yields the same empty result as no entry at all."""
        await store_entry(nav_backend, 'the needle is hidden', owner='alice')

        hidden = await grep_as_dict(pattern='needle', thread_id='t')
        absent = await grep_as_dict(pattern='needle', thread_id='absent-thread')

        assert hidden == absent
        assert hidden['total_matches'] == 0

    @pytest.mark.asyncio
    async def test_owner_matches_their_private_entry(self, nav_backend: StorageBackend) -> None:
        """The owner of a private entry finds its match."""
        cid = await store_entry(nav_backend, 'the needle is hers', owner='alice')

        with as_principal('alice'):
            payload = await grep_as_dict(pattern='needle', thread_id='t')

        assert [row['context_id'] for row in payload['results']] == [cid]

    @pytest.mark.asyncio
    async def test_hidden_entries_neither_take_scan_slots_nor_flag_truncation(self, nav_backend: StorageBackend) -> None:
        """Hidden entries newer and older than the caller's never fill the scan cap or mark the scan truncated."""
        await store_entry(nav_backend, 'old hidden needle', offset_seconds=0, owner='alice')
        own = [await store_entry(nav_backend, f'own needle {index}', offset_seconds=10 + index) for index in range(2)]
        for index in range(3):
            await store_entry(nav_backend, f'new hidden needle {index}', offset_seconds=20 + index, owner='alice')

        payload = await grep_as_dict(pattern='needle', thread_id='t', max_entries_scanned=2)

        assert [row['context_id'] for row in payload['results']] == own[::-1]
        assert payload['truncated'] is False

    @pytest.mark.asyncio
    async def test_scope_reaches_the_repository(self, nav_backend: StorageBackend) -> None:
        """The caller's principal and groups reach grep_scan_text_contents as its scope."""
        del nav_backend
        captured: dict[str, Any] = {}
        repos = await ensure_repositories()
        real_scan = repos.context.grep_scan_text_contents

        async def spy_scan(**kwargs: Any) -> tuple[list[GrepScanRow], dict[str, Any]]:
            captured['scope'] = kwargs['scope']
            return await real_scan(**kwargs)

        with as_principal('bob', groups=['team-x']), patch.object(repos.context, 'grep_scan_text_contents', spy_scan):
            await grep_context(pattern='needle', thread_id='t')

        assert captured['scope'] == AccessScope('bob', frozenset({'team-x'}))
