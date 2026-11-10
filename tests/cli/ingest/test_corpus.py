"""Tests for the corpus walk and the per-document build of the ingest client."""

import os
import sys
import unicodedata
from collections.abc import Iterator
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import pytest

from app.cli.ingest import corpus
from app.cli.ingest.corpus import Corpus
from app.cli.ingest.corpus import DirectoryFailure
from app.cli.ingest.corpus import Document
from app.cli.ingest.corpus import Visibility
from app.cli.ingest.corpus import build_corpus
from app.cli.ingest.corpus import format_synced_at
from app.cli.ingest.corpus import walk
from app.cli.ingest.manifest import meta_digest
from app.cli.ingest.manifest import request_bytes
from app.cli.ingest.manifest import text_digest
from app.digests import content_hash
from app.ingest import INGEST_KEY
from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import SyncChange
from app.ingest import ingest_block_error

THREAD_ID = 'kb:test'
BUDGET = 3_145_728
SYNCED_AT = '2026-10-18T12:00:00Z'

# A file or directory mode denies access only on a POSIX host and only to a user
# other than root.
if sys.platform == 'win32':
    MODES_DENY_ACCESS = False
else:
    MODES_DENY_ACCESS = os.geteuid() != 0
requires_denying_modes = pytest.mark.skipif(
    not MODES_DENY_ACCESS,
    reason='file modes deny access only on a POSIX host to a user other than root',
)


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    """Create a file under the root, creating its parent directories."""
    path = root.joinpath(*relative.split('/'))
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding='utf-8')
    return path


def _build(
    root: Path,
    *,
    includes: Sequence[str] = ('**/*.md',),
    excludes: Sequence[str] = (),
    static_metadata: dict[str, object] | None = None,
    tags: Sequence[str] = (),
    visibility: Visibility = 'private',
    defer_generation: bool = False,
    max_request_bytes: int = BUDGET,
) -> Corpus:
    """Build a corpus with the defaults the command uses."""
    return build_corpus(
        root,
        thread_id=THREAD_ID,
        includes=includes,
        excludes=excludes,
        static_metadata=static_metadata or {},
        tags=tags,
        visibility=visibility,
        defer_generation=defer_generation,
        max_request_bytes=max_request_bytes,
    )


def _only_document(result: Corpus) -> Document:
    """Return the single document of a corpus that holds exactly one."""
    assert result.file_failures == []
    assert result.directory_failures == []
    assert len(result.documents) == 1
    return result.documents[0]


def _source_ids(result: Corpus) -> list[str]:
    """Return the source ids of a corpus's documents in order."""
    return [document.source_id for document in result.documents]


def _try_symlink(link: Path, target: Path, *, directory: bool) -> None:
    """Create a symbolic link, skipping the test where the host does not permit one."""
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f'this host does not permit creating symbolic links: {error}')


class TestWalkPruning:
    """The walk never enters ``.git`` and never follows a symbolic link."""

    def test_git_directories_are_never_walked(self, tmp_path: Path) -> None:
        """A ``.git`` directory at any depth is pruned, even when an include selects everything."""
        _write(tmp_path, '.git/HEAD.md', 'x')
        _write(tmp_path, 'sub/.git/config.md', 'x')
        _write(tmp_path, 'sub/kept.md', 'x')
        _write(tmp_path, 'top.md', 'x')

        files, failures = walk(tmp_path, includes=['**'], excludes=[])

        assert failures == []
        assert [source_id for source_id, _ in files] == ['sub/kept.md', 'top.md']

    def test_symbolic_links_are_not_followed(self, tmp_path: Path) -> None:
        """Neither a linked directory nor a linked file outside the root is read."""
        root = tmp_path / 'root'
        outside = tmp_path / 'outside'
        _write(outside, 'secret.md', '# Secret\n')
        _write(root, 'own.md', '# Own\n')
        _try_symlink(root / 'linked-dir', outside, directory=True)
        _try_symlink(root / 'linked.md', outside / 'secret.md', directory=False)

        result = _build(root)

        assert _source_ids(result) == ['own.md']
        assert result.file_failures == []


class TestTextNormalization:
    """Text is decoded as UTF-8, line endings become LF, and the stored strip applies."""

    def test_byte_order_mark_is_dropped(self, tmp_path: Path) -> None:
        """A UTF-8 byte order mark is not part of the text."""
        _write(tmp_path, 'bom.md', b'\xef\xbb\xbf# Title\n\nbody')

        document = _only_document(_build(tmp_path))

        assert document.text == '# Title\n\nbody'
        assert document.metadata['title'] == 'Title'

    def test_crlf_and_cr_become_lf(self, tmp_path: Path) -> None:
        """Windows and classic Mac line endings both become LF."""
        _write(tmp_path, 'endings.md', b'one\r\ntwo\rthree\n\r\nfour')

        document = _only_document(_build(tmp_path))

        assert document.text == 'one\ntwo\nthree\n\nfour'

    def test_surrounding_whitespace_is_stripped(self, tmp_path: Path) -> None:
        """The text is stored and hashed without surrounding whitespace."""
        _write(tmp_path, 'padded.md', '\n\n  body text \n\n')

        document = _only_document(_build(tmp_path))

        assert document.text == 'body text'
        assert document.text_digest == content_hash('body text')
        assert document.text_digest == text_digest(document.text)

    def test_non_ascii_text_is_kept(self, tmp_path: Path) -> None:
        """Characters outside ASCII, astral ones included, survive the decode."""
        _write(tmp_path, 'unicode.md', '# Привет 🌍\n\nтекст')

        document = _only_document(_build(tmp_path))

        assert document.text == '# Привет 🌍\n\nтекст'
        assert document.metadata['title'] == 'Привет 🌍'


class TestKeepItems:
    """A file the client cannot use becomes a keep failure, and the rest of the corpus still builds."""

    def test_whitespace_only_file_is_kept(self, tmp_path: Path) -> None:
        """A file whose normalized text is empty is represented by a keep item."""
        path = _write(tmp_path, 'blank.md', ' \r\n\t\n')
        _write(tmp_path, 'fine.md', 'body')

        result = _build(tmp_path)

        assert _source_ids(result) == ['fine.md']
        assert [(failure.source_id, failure.path, failure.keep) for failure in result.file_failures] == [
            ('blank.md', path, True),
        ]
        assert 'empty' in result.file_failures[0].reason

    def test_undecodable_file_is_kept(self, tmp_path: Path) -> None:
        """Bytes that are not UTF-8 make a keep failure naming the decode error."""
        _write(tmp_path, 'latin1.md', 'café'.encode('latin-1'))
        _write(tmp_path, 'fine.md', 'body')

        result = _build(tmp_path)

        assert _source_ids(result) == ['fine.md']
        failure = result.file_failures[0]
        assert (failure.source_id, failure.keep) == ('latin1.md', True)
        assert 'UTF-8' in failure.reason

    def test_unreadable_file_is_kept(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file that cannot be opened makes a keep failure naming the operating-system error."""
        locked = _write(tmp_path, 'locked.md', 'body')
        _write(tmp_path, 'fine.md', 'body')
        real_read_bytes = Path.read_bytes

        def read_bytes(self: Path) -> bytes:
            if self == locked:
                raise PermissionError(13, 'Permission denied', str(self))
            return real_read_bytes(self)

        monkeypatch.setattr(Path, 'read_bytes', read_bytes)

        result = _build(tmp_path)

        assert _source_ids(result) == ['fine.md']
        assert [(failure.source_id, failure.keep) for failure in result.file_failures] == [('locked.md', True)]
        assert 'Permission denied' in result.file_failures[0].reason

    @requires_denying_modes
    def test_file_without_read_permission_is_kept(self, tmp_path: Path) -> None:
        """A file whose mode denies reading makes a keep failure."""
        locked = _write(tmp_path, 'locked.md', 'body')
        locked.chmod(0)
        try:
            result = _build(tmp_path)
        finally:
            locked.chmod(0o600)

        assert result.documents == []
        assert [(failure.source_id, failure.keep) for failure in result.file_failures] == [('locked.md', True)]


class TestOversizedEntries:
    """A document whose batch entry exceeds the request budget alone becomes a keep failure."""

    def test_entry_at_the_budget_builds(self, tmp_path: Path) -> None:
        """A create entry of exactly the budget still fits one request."""
        _write(tmp_path, 'big.md', '# Big\n\n' + 'x' * 5000)
        document = _only_document(_build(tmp_path))
        size = request_bytes([document.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)])

        result = _build(tmp_path, max_request_bytes=size)

        assert _source_ids(result) == ['big.md']

    def test_entry_over_the_budget_is_kept(self, tmp_path: Path) -> None:
        """One byte over the budget turns the document into a keep failure that names the budget."""
        _write(tmp_path, 'big.md', '# Big\n\n' + 'x' * 5000)
        _write(tmp_path, 'small.md', 'tiny')
        document = _build(tmp_path).documents[0]
        size = request_bytes([document.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)])

        result = _build(tmp_path, max_request_bytes=size - 1)

        assert _source_ids(result) == ['small.md']
        failure = result.file_failures[0]
        assert (failure.source_id, failure.keep) == ('big.md', True)
        assert f'{size - 1:,}' in failure.reason
        assert "docs/knowledge-base.md, 'Limits'" in failure.reason

    def test_update_entries_fit_whenever_the_create_entry_fits(self, tmp_path: Path) -> None:
        """Every update shape of a built document is within the budget its create entry met."""
        _write(tmp_path, 'big.md', '# Big\n\n' + 'x' * 5000)
        document = _only_document(_build(tmp_path))
        size = request_bytes([document.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)])
        document = _only_document(_build(tmp_path, max_request_bytes=size))

        for change in ('text', 'metadata', 'both'):
            entry = document.update_entry(context_id='f' * 32, thread_id=THREAD_ID, synced_at=SYNCED_AT, change=change)
            assert request_bytes([entry]) <= size

    def test_a_deferred_entry_is_measured_with_its_deferral(self, tmp_path: Path) -> None:
        """The deferral key counts toward the budget, so an entry that fits exactly without it is kept with it."""
        _write(tmp_path, 'big.md', '# Big\n\n' + 'x' * 5000)
        document = _only_document(_build(tmp_path))
        size = request_bytes([document.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)])

        result = _build(tmp_path, defer_generation=True, max_request_bytes=size)

        assert result.documents == []
        assert [(failure.source_id, failure.keep) for failure in result.file_failures] == [('big.md', True)]

    def test_non_ascii_text_is_measured_in_bytes(self, tmp_path: Path) -> None:
        """The budget counts UTF-8 bytes, so four-byte characters weigh four bytes each."""
        _write(tmp_path, 'emoji.md', '😀' * 1000)
        document = _only_document(_build(tmp_path))
        size = request_bytes([document.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)])

        assert size > 4000
        assert _build(tmp_path, max_request_bytes=size - 1).documents == []


class TestOverlongSourceIds:
    """A source id longer than the protocol bound is excluded from the manifest, never kept."""

    @staticmethod
    def _walk_yielding(
        monkeypatch: pytest.MonkeyPatch,
        root: Path,
        files: list[tuple[str, Path]],
    ) -> None:
        """Make build_corpus see the given files, whose names the host may not be able to hold."""

        def fake_walk(
            walked_root: Path,
            *,
            includes: Sequence[str],
            excludes: Sequence[str],
        ) -> tuple[list[tuple[str, Path]], list[DirectoryFailure]]:
            assert walked_root == root
            assert list(includes) == ['**/*.md']
            assert list(excludes) == []
            return files, []

        monkeypatch.setattr(corpus, 'walk', fake_walk)

    def test_bound_counts_code_points(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """256 four-byte characters are within the bound; 257 ASCII characters are not."""
        path = _write(tmp_path, 'doc.md', 'body')
        astral = '😀' * MAX_SOURCE_ID_LENGTH
        too_long = 'd/' + 'x' * (MAX_SOURCE_ID_LENGTH - 4) + '.md'
        assert len(too_long) == MAX_SOURCE_ID_LENGTH + 1
        self._walk_yielding(monkeypatch, tmp_path, [(astral, path), (too_long, path)])

        result = _build(tmp_path)

        assert _source_ids(result) == [astral]
        assert [(failure.source_id, failure.keep) for failure in result.file_failures] == [(too_long, False)]
        assert '257' in result.file_failures[0].reason

    def test_overlong_unreadable_file_is_excluded_not_kept(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The length rule applies before the file is read, so an over-long id is never a keep item."""
        missing = tmp_path / 'missing.md'
        too_long = 'y' * (MAX_SOURCE_ID_LENGTH + 1)
        self._walk_yielding(monkeypatch, tmp_path, [(too_long, missing)])

        result = _build(tmp_path)

        assert result.documents == []
        assert [(failure.source_id, failure.path, failure.keep) for failure in result.file_failures] == [
            (too_long, missing, False),
        ]


class TestSourceIds:
    """Source ids are root-relative POSIX paths in NFC, case preserved."""

    def test_nested_paths_use_forward_slashes(self, tmp_path: Path) -> None:
        """No backslash reaches a source id on any host."""
        _write(tmp_path, 'Guides/Setup/First Steps.md', 'body')

        result = _build(tmp_path)

        assert _source_ids(result) == ['Guides/Setup/First Steps.md']

    def test_decomposed_names_become_nfc(self, tmp_path: Path) -> None:
        """A name stored in decomposed form gets the composed source id."""
        name = unicodedata.normalize('NFD', 'café.md')
        assert len(name) == len('café.md') + 1
        _write(tmp_path, name, 'body')

        result = _build(tmp_path)

        assert _source_ids(result) == [unicodedata.normalize('NFC', name)]
        assert result.documents[0].metadata['title'] == 'café'

    def test_documents_are_sorted_by_code_point(self, tmp_path: Path) -> None:
        """Documents come out in code-point order of their source ids."""
        for name in ('b.md', 'B2.md', 'a/z.md', 'é.md'):
            _write(tmp_path, name, 'body')

        result = _build(tmp_path)

        assert _source_ids(result) == ['B2.md', 'a/z.md', 'b.md', 'é.md']


class TestMetadata:
    """Metadata holds the title and the run-wide metadata; the digest covers tags and visibility."""

    def test_title_is_the_first_level_one_heading(self, tmp_path: Path) -> None:
        """Deeper headings are skipped, and the first level-one heading wins."""
        _write(tmp_path, 'doc.md', 'intro\n## Section\n# Real Title #\n# Second Title\n')

        document = _only_document(_build(tmp_path))

        assert document.metadata == {'title': 'Real Title'}

    @pytest.mark.parametrize(
        ('first_line', 'title'),
        [
            ('   # Indented', 'Indented'),
            ('#\tTabbed', 'Tabbed'),
            ('# Closed ##', 'Closed'),
            ('# Keeps#hash', 'Keeps#hash'),
            ('# Spaced   ', 'Spaced'),
            ('    # Code block', 'notes'),
            ('#NoSpace', 'notes'),
            ('#', 'notes'),
            ('# #', 'notes'),
            ('## Level two', 'notes'),
        ],
    )
    def test_heading_forms(self, tmp_path: Path, first_line: str, title: str) -> None:
        """Only an ATX level-one heading with text gives the title; otherwise the stem does.

        The line follows an introduction, because the stored text loses the indentation
        of its first line.
        """
        _write(tmp_path, 'notes.md', f'intro\n{first_line}\nbody\n')

        document = _only_document(_build(tmp_path))

        assert document.metadata['title'] == title

    def test_title_falls_back_to_the_stem(self, tmp_path: Path) -> None:
        """A document without a level-one heading takes its file stem as the title."""
        _write(tmp_path, 'guide.v2.md', 'no heading here')

        document = _only_document(_build(tmp_path))

        assert document.metadata == {'title': 'guide.v2'}

    def test_run_metadata_is_merged_and_its_title_wins(self, tmp_path: Path) -> None:
        """Run-wide metadata applies to every document, and its title replaces the derived one."""
        _write(tmp_path, 'doc.md', '# Derived\n')

        merged = _only_document(_build(tmp_path, static_metadata={'project': 'handbook'}))
        overridden = _only_document(_build(tmp_path, static_metadata={'title': 'Fixed', 'project': 'handbook'}))

        assert merged.metadata == {'title': 'Derived', 'project': 'handbook'}
        assert overridden.metadata == {'title': 'Fixed', 'project': 'handbook'}

    def test_tags_are_normalized(self, tmp_path: Path) -> None:
        """Tags are trimmed, lower-cased and deduplicated in first-seen order."""
        _write(tmp_path, 'doc.md', 'body')

        document = _only_document(_build(tmp_path, tags=[' Beta', 'alpha', 'beta', ' ']))

        assert document.tags == ['beta', 'alpha']

    def test_meta_digest_covers_metadata_tags_and_visibility(self, tmp_path: Path) -> None:
        """The document digest is the manifest digest of what the client sends besides the text."""
        _write(tmp_path, 'doc.md', '# Title\n')

        document = _only_document(_build(tmp_path, static_metadata={'project': 'p'}, tags=['b', 'a']))

        assert document.meta_digest == meta_digest({'title': 'Title', 'project': 'p'}, ['b', 'a'], 'private')

    def test_visibility_reaches_the_digest(self, tmp_path: Path) -> None:
        """Changing only the visibility changes the digest, so the change reaches stored entries."""
        _write(tmp_path, 'doc.md', 'body')

        private = _only_document(_build(tmp_path, visibility='private'))
        public = _only_document(_build(tmp_path, visibility='public'))

        assert (private.visibility, public.visibility) == ('private', 'public')
        assert private.meta_digest != public.meta_digest
        assert public.meta_digest == meta_digest(public.metadata, public.tags, 'public')

    def test_tag_changes_reach_the_digest(self, tmp_path: Path) -> None:
        """Adding a tag changes the digest; reordering or recasing tags does not."""
        _write(tmp_path, 'doc.md', 'body')

        base = _only_document(_build(tmp_path, tags=['a', 'b']))
        reordered = _only_document(_build(tmp_path, tags=['B', 'a']))
        extended = _only_document(_build(tmp_path, tags=['a', 'b', 'c']))

        assert base.meta_digest == reordered.meta_digest
        assert base.meta_digest != extended.meta_digest


class TestEntries:
    """A document renders the batch entries the client sends, with the ingest block attached."""

    def test_entry_metadata_carries_a_valid_block(self, tmp_path: Path) -> None:
        """The block names the corpus, the source id, the digest and the run time, and passes validation."""
        _write(tmp_path, 'doc.md', '# Title\n')
        document = _only_document(_build(tmp_path, static_metadata={'project': 'p'}))

        metadata = document.entry_metadata(thread_id=THREAD_ID, synced_at=SYNCED_AT)

        assert metadata == {
            'title': 'Title',
            'project': 'p',
            INGEST_KEY: {
                'scope': THREAD_ID,
                'source_id': 'doc.md',
                'meta_digest': document.meta_digest,
                'synced_at': SYNCED_AT,
            },
        }
        assert ingest_block_error(metadata[INGEST_KEY]) is None
        assert INGEST_KEY not in document.metadata

    def test_create_entry_never_deduplicates(self, tmp_path: Path) -> None:
        """A create entry inserts a fresh user entry with the explicit visibility."""
        _write(tmp_path, 'doc.md', 'body')
        document = _only_document(_build(tmp_path, tags=['t'], visibility='public'))

        entry = document.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)

        assert entry == {
            'thread_id': THREAD_ID,
            'source': 'user',
            'text': 'body',
            'metadata': document.entry_metadata(thread_id=THREAD_ID, synced_at=SYNCED_AT),
            'tags': ['t'],
            'visibility': 'public',
            'deduplicate': False,
        }

    @pytest.mark.parametrize(('change', 'has_text'), [('text', True), ('both', True), ('metadata', False)])
    def test_update_entry_replaces_metadata(self, tmp_path: Path, change: SyncChange, has_text: bool) -> None:
        """An update entry always sends the full metadata, tags and visibility, and the text when it changed."""
        _write(tmp_path, 'doc.md', 'body')
        document = _only_document(_build(tmp_path))
        context_id = '0' * 32

        entry = document.update_entry(context_id=context_id, thread_id=THREAD_ID, synced_at=SYNCED_AT, change=change)

        expected: dict[str, object] = {
            'context_id': context_id,
            'metadata': document.entry_metadata(thread_id=THREAD_ID, synced_at=SYNCED_AT),
            'tags': [],
            'visibility': 'private',
        }
        if has_text:
            expected['text'] = 'body'
        assert entry == expected

    def test_a_deferring_create_entry_asks_to_defer_generation(self, tmp_path: Path) -> None:
        """With deferral a create entry carries defer_generation beside deduplicate; without it the key is absent."""
        _write(tmp_path, 'doc.md', 'body')
        immediate = _only_document(_build(tmp_path))
        deferring = _only_document(_build(tmp_path, defer_generation=True))

        entry = deferring.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)

        assert entry == {**immediate.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT), 'defer_generation': True}
        assert 'defer_generation' not in immediate.create_entry(thread_id=THREAD_ID, synced_at=SYNCED_AT)

    @pytest.mark.parametrize('change', ['text', 'both', 'metadata'])
    def test_a_deferring_update_entry_asks_to_defer_generation(self, tmp_path: Path, change: SyncChange) -> None:
        """With deferral every update entry carries defer_generation; the server applies it only to new text."""
        _write(tmp_path, 'doc.md', 'body')
        immediate = _only_document(_build(tmp_path))
        deferring = _only_document(_build(tmp_path, defer_generation=True))
        context_id = '0' * 32

        entry = deferring.update_entry(context_id=context_id, thread_id=THREAD_ID, synced_at=SYNCED_AT, change=change)

        plain = immediate.update_entry(context_id=context_id, thread_id=THREAD_ID, synced_at=SYNCED_AT, change=change)
        assert entry == {**plain, 'defer_generation': True}
        assert 'defer_generation' not in plain

    def test_deferral_leaves_the_digests_alone(self, tmp_path: Path) -> None:
        """Deferral is how a run writes, not what it stores, so the manifest digests do not change with it."""
        _write(tmp_path, 'doc.md', 'body')
        immediate = _only_document(_build(tmp_path))
        deferring = _only_document(_build(tmp_path, defer_generation=True))

        assert (deferring.text_digest, deferring.meta_digest) == (immediate.text_digest, immediate.meta_digest)


class TestSyncedAt:
    """The run time is written in UTC with whole seconds."""

    def test_whole_seconds_in_utc(self) -> None:
        """Fractions are dropped and the zone is converted to UTC."""
        moment = datetime(2026, 10, 18, 15, 0, 7, 999_999, tzinfo=timezone(timedelta(hours=3)))

        assert format_synced_at(moment) == '2026-10-18T12:00:07Z'

    def test_every_instant_has_the_same_width(self) -> None:
        """The width the size check assumes holds for any run time."""
        widths = {len(format_synced_at(datetime(year, 1, 1, tzinfo=UTC))) for year in (1000, 2026, 9999)}

        assert widths == {len(SYNCED_AT)}

    def test_value_passes_block_validation(self) -> None:
        """The server accepts the formatted run time in a block."""
        block = {
            'scope': THREAD_ID,
            'source_id': 'a.md',
            'meta_digest': '0' * 64,
            'synced_at': format_synced_at(datetime.now(UTC)),
        }

        assert ingest_block_error(block) is None


class TestDirectoryFailures:
    """A directory the walk cannot list is recorded, and the rest of the tree is still walked."""

    @staticmethod
    def _refuse_listing(monkeypatch: pytest.MonkeyPatch, refused: Path) -> None:
        """Make os.scandir fail for one directory, as a directory without read permission does."""
        real_scandir = os.scandir

        def scandir(path: str) -> Iterator[os.DirEntry[str]]:
            if Path(path) == refused:
                raise PermissionError(13, 'Permission denied', path)
            return real_scandir(path)

        monkeypatch.setattr(os, 'scandir', scandir)

    @staticmethod
    def _tree(root: Path) -> None:
        """Create files at the top, inside a directory to refuse and inside an open one."""
        _write(root, 'top.md', 'body')
        _write(root, 'locked/inner.md', 'body')
        _write(root, 'locked/deeper/leaf.md', 'body')
        _write(root, 'open/visible.md', 'body')

    def test_unlistable_directory_is_recorded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The refused directory is named with its reason; every other file is selected."""
        self._tree(tmp_path)
        self._refuse_listing(monkeypatch, tmp_path / 'locked')

        files, failures = walk(tmp_path, includes=['**/*.md'], excludes=[])

        assert [source_id for source_id, _ in files] == ['open/visible.md', 'top.md']
        assert failures == [DirectoryFailure(relative_path='locked', reason='Permission denied')]

    def test_corpus_carries_the_directory_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The corpus reports the directory and builds the documents outside it."""
        self._tree(tmp_path)
        self._refuse_listing(monkeypatch, tmp_path / 'locked' / 'deeper')

        result = _build(tmp_path)

        assert _source_ids(result) == ['locked/inner.md', 'open/visible.md', 'top.md']
        assert result.file_failures == []
        assert result.directory_failures == [DirectoryFailure(relative_path='locked/deeper', reason='Permission denied')]

    def test_unlistable_root_is_named_dot(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A root that cannot be listed is reported as '.'."""
        self._tree(tmp_path)
        self._refuse_listing(monkeypatch, tmp_path)

        files, failures = walk(tmp_path, includes=['**/*.md'], excludes=[])

        assert files == []
        assert failures == [DirectoryFailure(relative_path='.', reason='Permission denied')]

    def test_missing_root_is_recorded(self, tmp_path: Path) -> None:
        """A root that does not exist is a directory failure, not an empty corpus."""
        files, failures = walk(tmp_path / 'absent', includes=['**/*.md'], excludes=[])

        assert files == []
        assert [failure.relative_path for failure in failures] == ['.']

    @requires_denying_modes
    def test_directory_without_permissions_is_recorded(self, tmp_path: Path) -> None:
        """A directory whose mode denies listing is recorded."""
        self._tree(tmp_path)
        locked = tmp_path / 'locked'
        locked.chmod(0)
        try:
            files, failures = walk(tmp_path, includes=['**/*.md'], excludes=[])
        finally:
            locked.chmod(0o700)

        assert [source_id for source_id, _ in files] == ['open/visible.md', 'top.md']
        assert [failure.relative_path for failure in failures] == ['locked']
