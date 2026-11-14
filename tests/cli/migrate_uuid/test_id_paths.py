"""Tests for the metadata id path grammar, its matcher, and the stamp key rule."""

import pytest

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.id_paths import NUMERIC_ID
from app.cli.migrate_uuid.id_paths import IdPath
from app.cli.migrate_uuid.id_paths import Location
from app.cli.migrate_uuid.id_paths import check_legacy_id_key
from app.cli.migrate_uuid.id_paths import locations
from app.cli.migrate_uuid.id_paths import render_location

EMPTY_KEY = 'has an empty key: keys are separated by single dots, with none at either end'
ANY_DEPTH_PLACEMENT = '"**" may appear only as the first key, followed by at least one key'
PARTIAL_WILDCARD = '"*" stands for a whole key and cannot be part of one'
STAMP_CHARACTERS = (
    'cannot be stamped: the metadata filters address a top-level key made of letters, digits, "_" and "-" only'
)


def _found(root: object, text: str) -> set[Location]:
    """Return the locations a path finds, asserting that none is found twice."""
    found = list(locations(root, IdPath.parse(text)))
    assert len(found) == len(set(found))
    return set(found)


class TestParse:
    """Accepted and refused path texts."""

    @pytest.mark.parametrize(
        ('text', 'any_depth', 'keys'),
        [
            ('references.context_ids', False, ('references', 'context_ids')),
            ('**.references.context_ids', True, ('references', 'context_ids')),
            ('links.*', False, ('links', '*')),
            ('*', False, ('*',)),
            ('**.*', True, ('*',)),
            ('*.context_ids', False, ('*', 'context_ids')),
            ('Links.Parent-Id_2', False, ('Links', 'Parent-Id_2')),
            (' links . parent', False, (' links ', ' parent')),
            ('0.1', False, ('0', '1')),
        ],
    )
    def test_accepted(self, text: str, any_depth: bool, keys: tuple[str, ...]) -> None:
        """A valid path keeps its text and splits into exact keys, with no trimming."""
        path = IdPath.parse(text)
        assert path == IdPath(text=text, any_depth=any_depth, keys=keys)

    def test_default_path(self) -> None:
        """The default path finds references.context_ids at any depth."""
        assert IdPath.parse(DEFAULT_ID_PATH) == IdPath(
            text='**.references.context_ids',
            any_depth=True,
            keys=('references', 'context_ids'),
        )

    @pytest.mark.parametrize(
        ('text', 'reason'),
        [
            ('', EMPTY_KEY),
            ('.a', EMPTY_KEY),
            ('a.', EMPTY_KEY),
            ('a..b', EMPTY_KEY),
            ('**.', EMPTY_KEY),
            ('**', ANY_DEPTH_PLACEMENT),
            ('links.**', ANY_DEPTH_PLACEMENT),
            ('a.**.b', ANY_DEPTH_PLACEMENT),
            ('**.**.a', ANY_DEPTH_PLACEMENT),
            ('a*', PARTIAL_WILDCARD),
            ('*x', PARTIAL_WILDCARD),
            ('***', PARTIAL_WILDCARD),
            ('**x.a', PARTIAL_WILDCARD),
            ('**.links.a*b', PARTIAL_WILDCARD),
        ],
    )
    def test_refused(self, text: str, reason: str) -> None:
        """An invalid path is refused with a message that quotes it and names the rule."""
        with pytest.raises(ValueError, match='.') as caught:
            IdPath.parse(text)
        separator = ' ' if reason == EMPTY_KEY else ': '
        assert str(caught.value) == f'{text!r}{separator}{reason}'


class TestLocations:
    """Which locations a path finds in parsed metadata."""

    def test_default_path_at_the_root(self) -> None:
        """references.context_ids at the root is found."""
        assert _found({'references': {'context_ids': [1]}}, DEFAULT_ID_PATH) == {('references', 'context_ids')}

    def test_default_path_inside_lists_of_objects_at_any_depth(self) -> None:
        """Objects reached through list elements and nested lists are searched too."""
        metadata = {
            'items': [{'references': {'context_ids': [1]}}],
            'a': {'b': [[{'references': {'context_ids': [2]}}]]},
            'references': {'context_ids': [3]},
        }
        assert _found(metadata, DEFAULT_ID_PATH) == {
            ('references', 'context_ids'),
            ('items', 0, 'references', 'context_ids'),
            ('a', 'b', 0, 0, 'references', 'context_ids'),
        }

    def test_default_path_when_the_root_is_a_list(self) -> None:
        """A list root is searched element by element."""
        assert _found([{'references': {'context_ids': [1]}}, 5], DEFAULT_ID_PATH) == {(0, 'references', 'context_ids')}

    def test_default_path_nested_inside_a_found_list(self) -> None:
        """An object inside a found list is searched like any other object."""
        metadata = {'references': {'context_ids': [{'references': {'context_ids': [1]}}]}}
        assert _found(metadata, DEFAULT_ID_PATH) == {
            ('references', 'context_ids'),
            ('references', 'context_ids', 0, 'references', 'context_ids'),
        }

    def test_locations_come_object_by_object_in_document_order(self) -> None:
        """Matches are yielded per starting object, objects in the order they appear in the metadata."""
        metadata = {
            'first': {'references': {'context_ids': [1]}},
            'references': {'context_ids': [{'references': {'context_ids': [2]}}]},
            'last': [{'references': {'context_ids': [3]}}, {'references': {'context_ids': [4]}}],
        }
        assert list(locations(metadata, IdPath.parse(DEFAULT_ID_PATH))) == [
            ('references', 'context_ids'),
            ('first', 'references', 'context_ids'),
            ('references', 'context_ids', 0, 'references', 'context_ids'),
            ('last', 0, 'references', 'context_ids'),
            ('last', 1, 'references', 'context_ids'),
        ]

    def test_lists_are_not_stepped_through_between_named_keys(self) -> None:
        """A list between two named keys ends the branch, also under the default path."""
        metadata = {'references': [{'context_ids': [1]}]}
        assert _found(metadata, DEFAULT_ID_PATH) == set()
        assert _found(metadata, 'references.context_ids') == set()
        assert _found({'items': [{'references': 1}]}, 'items.references') == set()

    @pytest.mark.parametrize('value', [5, '5', None, [], {}, 'd' * 32])
    def test_any_value_at_the_last_key_is_a_location(self, value: object) -> None:
        """The last key's value is found whatever it holds, a scalar or null included."""
        assert _found({'references': {'context_ids': value}}, DEFAULT_ID_PATH) == {('references', 'context_ids')}

    @pytest.mark.parametrize(
        'metadata',
        [
            {'references': 'y'},
            {'references': None},
            {'references': {'other': [1]}},
            {'context_ids': [1]},
            {},
        ],
    )
    def test_nothing_found_when_a_key_is_missing_or_not_an_object(self, metadata: object) -> None:
        """A missing key or a non-object between keys yields no location."""
        assert _found(metadata, DEFAULT_ID_PATH) == set()

    @pytest.mark.parametrize('root', [[{'references': {'context_ids': [1]}}], 'text', 7, None, True])
    def test_anchored_path_needs_an_object_root(self, root: object) -> None:
        """A path without a leading any-depth key starts only at an object root."""
        assert _found(root, 'references.context_ids') == set()

    def test_anchored_path_starts_at_the_root_only(self) -> None:
        """A path without a leading any-depth key does not match deeper objects."""
        metadata = {
            'items': [{'references': {'context_ids': [1]}}],
            'nested': {'references': {'context_ids': [2]}},
            'references': {'context_ids': [3]},
        }
        assert _found(metadata, 'references.context_ids') == {('references', 'context_ids')}

    def test_keys_are_compared_exactly(self) -> None:
        """Keys match case-sensitively and without trimming."""
        metadata = {'references': {'context_ids': [1]}}
        assert _found(metadata, 'References.context_ids') == set()
        assert _found(metadata, ' references.context_ids') == set()

    def test_a_key_holding_a_dot_cannot_be_addressed(self) -> None:
        """A dot always separates keys, so a path cannot name a key with a dot in its name."""
        assert _found({'a.b': [1], 'a': {'c': 1}}, 'a.b') == set()

    def test_wildcards_reach_keys_a_path_cannot_name(self) -> None:
        """A wildcard finds a key holding a dot or a star, and any depth starts inside its value."""
        assert _found({'links': {'a.b': 1, 'c*d': 2}}, 'links.*') == {('links', 'a.b'), ('links', 'c*d')}
        assert _found({'x.y': {'references': {'context_ids': [1]}}}, DEFAULT_ID_PATH) == {
            ('x.y', 'references', 'context_ids'),
        }

    def test_wildcard_finds_every_key_of_an_object(self) -> None:
        """A wildcard key finds every value of the object it steps into."""
        metadata = {
            'links': {'parent': 1, 'closed_by': [2, 99], 'github_prs': ['5']},
            'references': {'context_ids': [7]},
        }
        assert _found(metadata, 'links.*') == {
            ('links', 'parent'),
            ('links', 'closed_by'),
            ('links', 'github_prs'),
        }
        assert _found(metadata, '*') == {('links',), ('references',)}

    def test_wildcard_between_named_keys(self) -> None:
        """A wildcard steps only into object values before the next key."""
        metadata = {
            'references': {'context_ids': [1]},
            'other': {'context_ids': 2},
            'listed': [{'context_ids': 3}],
            'scalar': 4,
        }
        assert _found(metadata, '*.context_ids') == {('references', 'context_ids'), ('other', 'context_ids')}

    def test_wildcard_over_a_list_finds_nothing(self) -> None:
        """A wildcard does not enumerate list elements."""
        assert _found({'links': [1, 2]}, 'links.*') == set()

    def test_any_depth_at_every_object(self) -> None:
        """An any-depth path is matched from the root and from every nested object."""
        metadata = {'context_ids': [3], 'references': {'context_ids': [1]}, 'items': [{'context_ids': 2}]}
        assert _found(metadata, '**.context_ids') == {
            ('context_ids',),
            ('references', 'context_ids'),
            ('items', 0, 'context_ids'),
        }

    def test_any_depth_wildcard_finds_every_key(self) -> None:
        """Any depth followed by a wildcard finds every key of every object."""
        metadata = {'a': {'b': 1}, 'c': [{'d': 2}, [{'e': 3}]]}
        assert _found(metadata, '**.*') == {('a',), ('a', 'b'), ('c',), ('c', 0, 'd'), ('c', 1, 0, 'e')}

    def test_any_depth_on_a_scalar_root_finds_nothing(self) -> None:
        """A scalar or null root holds no object to search."""
        assert _found(7, DEFAULT_ID_PATH) == set()
        assert _found(None, '**.*') == set()


class TestRenderLocation:
    """How a location reads in a message."""

    @pytest.mark.parametrize(
        ('location', 'text'),
        [
            ((), 'metadata'),
            (('references', 'context_ids'), 'metadata.references.context_ids'),
            (('items', 0, 'references', 'context_ids', 2), 'metadata.items[0].references.context_ids[2]'),
            ((0, 'references'), 'metadata[0].references'),
            (('a', 0, 1, 'b'), 'metadata.a[0][1].b'),
        ],
    )
    def test_render(self, location: Location, text: str) -> None:
        """Keys are joined by dots and list indexes are bracketed after the metadata prefix."""
        assert render_location(location) == text


class TestLegacyIdKey:
    """Names --legacy-id-key accepts and refuses."""

    @pytest.mark.parametrize('name', ['legacy_id', 'old-id', 'ID2', '0', 'references', 'Ingest'])
    def test_accepted(self, name: str) -> None:
        """A key the metadata filters can address at the top level is returned unchanged."""
        assert check_legacy_id_key(name) == name

    @pytest.mark.parametrize('name', ['', ' ', 'a.b', 'with space', 'legacy_id\n', 'café', 'a/b', '٣'])
    def test_refused_characters(self, name: str) -> None:
        """A key the metadata filters cannot address at the top level is refused."""
        with pytest.raises(ValueError, match='.') as caught:
            check_legacy_id_key(name)
        assert str(caught.value) == f'{name!r} {STAMP_CHARACTERS}'

    def test_ingest_is_reserved(self) -> None:
        """The corpus-sync block key is refused."""
        with pytest.raises(ValueError, match='.') as caught:
            check_legacy_id_key('ingest')
        assert str(caught.value) == "'ingest' is reserved for the corpus-sync block"


class TestNumericId:
    """The canonical decimal form a numeric string must have to be read as an id."""

    @pytest.mark.parametrize('text', ['0', '7', '-5', '10', '999', '36717'])
    def test_canonical_forms_match(self, text: str) -> None:
        """The text Python and JSON write for an integer matches."""
        assert NUMERIC_ID.fullmatch(text) is not None

    @pytest.mark.parametrize(
        'text',
        ['', '-', '-0', '00', '007', ' 7', '7 ', '7\n', '+7', '7.0', '1_0', '1e3', '٧', 'd' * 32],
    )
    def test_other_forms_do_not_match(self, text: str) -> None:
        """The empty string, leading zeros, surrounding whitespace, a leading '+', '-0' or a bare '-', digit separators,
        decimals, exponents, non-ASCII digits and a hex id do not match.
        """
        assert NUMERIC_ID.fullmatch(text) is None
