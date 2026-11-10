"""Tests for the ingest client's manifest: digests, items, range slices and the byte-measuring packer."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.cli.ingest.corpus import Corpus
from app.cli.ingest.corpus import Document
from app.cli.ingest.corpus import FileFailure
from app.cli.ingest.manifest import ManifestSlice
from app.cli.ingest.manifest import manifest_items
from app.cli.ingest.manifest import meta_digest
from app.cli.ingest.manifest import pack
from app.cli.ingest.manifest import plan_request_bytes
from app.cli.ingest.manifest import request_bytes
from app.cli.ingest.manifest import slice_manifest
from app.cli.ingest.manifest import text_digest
from app.digests import canonical_json_digest
from app.digests import content_hash
from app.digests import normalize_stored_text
from app.ingest import MAX_MANIFEST_ITEMS
from app.models import MAX_THREAD_ID_LENGTH
from app.models import normalize_tag_list

DIGEST = 'a' * 64


def _compact_size(value: object) -> int:
    """Return the UTF-8 length of a value's compact JSON form, as the client sends it."""
    return len(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


def _document(source_id: str) -> Document:
    """Return a document with fixed digests, for manifest tests that need no files."""
    return Document(
        source_id=source_id,
        text='body',
        metadata={'title': source_id},
        tags=[],
        visibility='private',
        defer_generation=False,
        text_digest=DIGEST,
        meta_digest=DIGEST,
    )


def _items(count: int, *, width: int = 8) -> list[dict[str, object]]:
    """Return sorted ordinary manifest items with zero-padded source ids."""
    return [{'source_id': f'{n:0{width}d}.md', 'text_digest': DIGEST, 'meta_digest': DIGEST} for n in range(count)]


def _ids(items: Sequence[dict[str, object]]) -> list[object]:
    """Return the source ids of manifest items in order."""
    return [item['source_id'] for item in items]


class TestDigests:
    """The manifest digests are the shared server digests over what the client sends."""

    @pytest.mark.parametrize('text', ['body', '  padded \n', 'Привет 🌍', ''])
    def test_text_digest_is_the_stored_content_hash(self, text: str) -> None:
        """The text digest equals the hash the server records for the stored text."""
        assert text_digest(text) == content_hash(normalize_stored_text(text))

    def test_meta_digest_formula(self) -> None:
        """The metadata digest hashes metadata, sorted normalized tags and the sent visibility."""
        metadata: dict[str, object] = {'title': 'T', 'nested': {'b': 1, 'a': [1, 2]}}
        tags = ['Zeta', ' alpha ', 'zeta']

        expected = canonical_json_digest({
            'metadata': metadata,
            'tags': sorted(normalize_tag_list(tags)),
            'visibility': 'public',
        })

        assert meta_digest(metadata, tags, 'public') == expected

    def test_tag_order_and_case_do_not_matter(self) -> None:
        """The same tag set in any order or case gives the same digest."""
        assert meta_digest({}, ['b', 'A'], 'private') == meta_digest({}, ['a', ' B ', 'b'], 'private')

    def test_metadata_key_order_does_not_matter(self) -> None:
        """The digest is computed over canonical JSON."""
        assert meta_digest({'a': 1, 'b': 2}, [], 'private') == meta_digest({'b': 2, 'a': 1}, [], 'private')

    def test_every_input_changes_the_digest(self) -> None:
        """A change of metadata, tags or visibility each changes the digest."""
        base = meta_digest({'title': 'T'}, ['a'], 'private')

        assert meta_digest({'title': 'U'}, ['a'], 'private') != base
        assert meta_digest({'title': 'T'}, ['a', 'b'], 'private') != base
        assert meta_digest({'title': 'T'}, ['a'], 'public') != base


class TestManifestItems:
    """Documents become digest items, kept files become keep items, excluded files become nothing."""

    def test_items_by_kind(self, tmp_path: Path) -> None:
        """Each document and each kept file appears once; an excluded file does not appear."""
        corpus = Corpus(
            documents=[_document('b.md')],
            file_failures=[
                FileFailure(source_id='a.md', path=tmp_path / 'a.md', reason='not valid UTF-8', keep=True),
                FileFailure(source_id='x' * 257, path=tmp_path / 'long.md', reason='too long', keep=False),
            ],
            directory_failures=[],
        )

        assert manifest_items(corpus) == [
            {'source_id': 'a.md', 'keep': True},
            {'source_id': 'b.md', 'text_digest': DIGEST, 'meta_digest': DIGEST},
        ]

    def test_items_are_sorted_by_code_point(self) -> None:
        """Sorting follows code points, so an astral character sorts after U+FFFF as on the server."""
        source_ids = ['😀.md', 'b.md', '￿.md', 'é.md', 'B.md', 'a/z.md']
        corpus = Corpus(documents=[_document(source_id) for source_id in source_ids], file_failures=[], directory_failures=[])

        assert _ids(manifest_items(corpus)) == ['B.md', 'a/z.md', 'b.md', 'é.md', '￿.md', '😀.md']

    def test_empty_corpus_has_no_items(self) -> None:
        """A corpus without documents or kept files yields an empty manifest."""
        assert manifest_items(Corpus(documents=[], file_failures=[], directory_failures=[])) == []


class TestRequestBytes:
    """A batch request is measured as the call envelope plus each entry and its framing."""

    def test_empty_request_is_the_envelope(self) -> None:
        """A request without entries costs the call envelope alone."""
        assert request_bytes([]) == 391

    def test_each_entry_adds_its_framing_and_compact_json(self) -> None:
        """Every entry adds 66 bytes plus its compact UTF-8 JSON."""
        entries: list[dict[str, object]] = [{'text': 'abc', 'tags': ['x']}, {'text': 'Привет 😀'}]

        assert request_bytes(entries) == 391 + sum(66 + _compact_size(entry) for entry in entries)

    def test_non_ascii_counts_utf8_bytes(self) -> None:
        """Characters are measured in raw UTF-8, never as escapes."""
        one = request_bytes([{'text': 'я'}])
        four = request_bytes([{'text': '😀'}])

        assert four - one == 2


class TestPlanRequestBytes:
    """A plan request is measured with the widest thread id the server accepts."""

    def test_any_thread_id_fits_the_allowance(self) -> None:
        """The worst thread id, every character escaped to six bytes, stays within the measure."""
        items = _items(3)
        arguments = {
            'thread_id': '\x1f' * MAX_THREAD_ID_LENGTH,
            'manifest': items,
            'source_id_after': 'é' * 256,
            'source_id_through': '😀' * 256,
            'allow_empty_manifest': False,
        }

        measured = plan_request_bytes(items, source_id_after='é' * 256, source_id_through='😀' * 256)

        assert request_bytes([]) + _compact_size(arguments) <= measured

    def test_bounds_are_measured_in_utf8_bytes(self) -> None:
        """A wider bound makes a larger request."""
        narrow = plan_request_bytes([], source_id_after='a', source_id_through=None)
        wide = plan_request_bytes([], source_id_after='😀', source_id_through=None)

        assert wide - narrow == 3

    def test_items_are_measured_like_entries(self) -> None:
        """Each manifest item adds what it would add to a batch request."""
        items = _items(5)

        with_items = plan_request_bytes(items, source_id_after=None, source_id_through=None)
        without = plan_request_bytes([], source_id_after=None, source_id_through=None)

        assert with_items - without == request_bytes(items) - request_bytes([])


class TestSliceManifest:
    """Slices partition the sorted manifest into consecutive ranges that cover every source id."""

    @staticmethod
    def _assert_partition(slices: list[ManifestSlice], items: list[dict[str, object]]) -> None:
        """Assert the slices hold every item once, in order, with chained bounds open at both ends."""
        assert [item for part in slices for item in part.items] == items
        assert slices[0].source_id_after is None
        assert slices[-1].source_id_through is None
        for previous, following in zip(slices, slices[1:], strict=False):
            assert previous.source_id_through == previous.items[-1]['source_id']
            assert following.source_id_after == previous.source_id_through

    def test_empty_manifest_is_one_unbounded_slice(self) -> None:
        """With no items, one slice covers every stored source id."""
        assert slice_manifest([], max_request_bytes=3_145_728) == [
            ManifestSlice(items=[], source_id_after=None, source_id_through=None),
        ]

    def test_small_manifest_is_one_unbounded_slice(self) -> None:
        """A manifest within both caps is planned in one call with no bounds."""
        items = _items(3)

        assert slice_manifest(items, max_request_bytes=3_145_728) == [
            ManifestSlice(items=items, source_id_after=None, source_id_through=None),
        ]

    def test_item_cap_splits_into_ranges(self) -> None:
        """2,500 items become slices of 1,000, 1,000 and 500 whose bounds chain."""
        items = _items(2500)

        slices = slice_manifest(items, max_request_bytes=3_145_728)

        assert [len(part.items) for part in slices] == [MAX_MANIFEST_ITEMS, MAX_MANIFEST_ITEMS, 500]
        assert [(part.source_id_after, part.source_id_through) for part in slices] == [
            (None, '00000999.md'),
            ('00000999.md', '00001999.md'),
            ('00001999.md', None),
        ]
        self._assert_partition(slices, items)

    def test_custom_item_cap(self) -> None:
        """A smaller item cap gives more slices."""
        items = _items(7)

        slices = slice_manifest(items, max_items=3, max_request_bytes=3_145_728)

        assert [len(part.items) for part in slices] == [3, 3, 1]
        self._assert_partition(slices, items)

    def test_byte_budget_splits_into_ranges(self) -> None:
        """Under a tight budget every slice's plan request fits, and no slice could take the next item."""
        items = _items(200, width=200)
        budget = 20_000

        slices = slice_manifest(items, max_request_bytes=budget)

        assert len(slices) > 1
        self._assert_partition(slices, items)
        for index, part in enumerate(slices):
            after = part.source_id_after
            assert plan_request_bytes(part.items, source_id_after=after, source_id_through=part.source_id_through) <= budget
            if index + 1 < len(slices):
                grown = [*part.items, slices[index + 1].items[0]]
                assert plan_request_bytes(grown, source_id_after=after, source_id_through=str(grown[-1]['source_id'])) > budget

    def test_unsorted_items_are_sorted_first(self) -> None:
        """Slices follow code-point order whatever order the items arrive in."""
        items = _items(5)

        slices = slice_manifest(list(reversed(items)), max_items=2, max_request_bytes=3_145_728)

        self._assert_partition(slices, items)

    def test_keep_items_are_ordinary_members(self) -> None:
        """A keep item takes its sorted place and can bound a range."""
        items: list[dict[str, object]] = [
            {'source_id': 'a.md', 'text_digest': DIGEST, 'meta_digest': DIGEST},
            {'source_id': 'b.md', 'keep': True},
            {'source_id': 'c.md', 'text_digest': DIGEST, 'meta_digest': DIGEST},
        ]

        slices = slice_manifest(items, max_items=2, max_request_bytes=3_145_728)

        assert [(part.source_id_after, part.source_id_through) for part in slices] == [(None, 'b.md'), ('b.md', None)]

    def test_item_too_large_for_the_budget_is_refused(self) -> None:
        """A budget that cannot hold even one item's plan request is a usage error."""
        with pytest.raises(ValueError, match='max_request_bytes'):
            slice_manifest(_items(2), max_request_bytes=500)

    def test_excluded_files_bound_no_slice(self, tmp_path: Path) -> None:
        """A file excluded from the manifest is in no slice and bounds no range."""
        excluded = 'b/' + 'x' * 300
        corpus = Corpus(
            documents=[_document('a.md'), _document('c.md')],
            file_failures=[FileFailure(source_id=excluded, path=tmp_path / 'long.md', reason='too long', keep=False)],
            directory_failures=[],
        )

        slices = slice_manifest(manifest_items(corpus), max_items=1, max_request_bytes=3_145_728)

        assert [(part.source_id_after, part.source_id_through) for part in slices] == [(None, 'a.md'), ('a.md', None)]
        assert excluded not in [source_id for part in slices for source_id in _ids(part.items)]


class TestPack:
    """Batches respect both the entry count and the byte budget, in order."""

    def test_packs_by_count(self) -> None:
        """25 small entries with a batch size of 10 become 10, 10 and 5."""
        entries: list[dict[str, object]] = [{'text': str(n)} for n in range(25)]

        batches = pack(entries, batch_size=10, max_request_bytes=3_145_728)

        assert [len(batch) for batch in batches] == [10, 10, 5]
        assert [entry for batch in batches for entry in batch] == entries

    def test_packs_by_bytes(self) -> None:
        """A budget holding two large entries splits the batch even below the count."""
        entries: list[dict[str, object]] = [{'text': 'x' * 1000} for _ in range(5)]
        budget = request_bytes(entries[:2])

        batches = pack(entries, batch_size=10, max_request_bytes=budget)

        assert [len(batch) for batch in batches] == [2, 2, 1]
        assert all(request_bytes(batch) <= budget for batch in batches)

    def test_one_byte_short_starts_a_new_batch(self) -> None:
        """A batch one byte over the budget is split."""
        entries: list[dict[str, object]] = [{'text': 'x' * 1000} for _ in range(2)]

        assert len(pack(entries, batch_size=10, max_request_bytes=request_bytes(entries))) == 1
        assert len(pack(entries, batch_size=10, max_request_bytes=request_bytes(entries) - 1)) == 2

    def test_creates_and_updates_pack_alike(self) -> None:
        """Create and update entries of mixed sizes are packed by the same measure."""
        create: dict[str, object] = {
            'thread_id': 'kb:x',
            'source': 'user',
            'text': 'y' * 3000,
            'metadata': {},
            'tags': [],
            'visibility': 'private',
            'deduplicate': False,
        }
        update: dict[str, object] = {
            'context_id': '0' * 32,
            'text': 'z' * 3000,
            'metadata': {},
            'tags': [],
            'visibility': 'private',
        }
        entries = [create, update, create, update]
        budget = request_bytes([create, update])

        batches = pack(entries, batch_size=100, max_request_bytes=budget)

        assert batches == [[create, update], [create, update]]

    def test_no_entries_no_batches(self) -> None:
        """Nothing to send makes no request."""
        assert pack([], batch_size=10, max_request_bytes=3_145_728) == []

    def test_entry_too_large_for_the_budget_is_refused(self) -> None:
        """An entry that cannot fit a request alone is refused rather than sent to fail."""
        with pytest.raises(ValueError, match='max_request_bytes'):
            pack([{'text': 'x' * 1000}], batch_size=10, max_request_bytes=1000)
