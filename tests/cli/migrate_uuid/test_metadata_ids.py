"""Tests for the rewrite of entry ids at metadata paths and the stamp of a migrated entry's old id."""

import json
from collections.abc import Mapping
from collections.abc import Sequence

import pytest

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.id_paths import IdPath
from app.cli.migrate_uuid.metadata_ids import MetadataIdRewriter
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import MetadataOutcome

A = 'a' * 32
B = 'b' * 32
C = 'c' * 32
E = 'e' * 32
MAPPING: dict[int, str] = {1: A, 2: B, 5: C, 7: E}
SKIPPED_WORDING = 'the id of an entry this run skipped; not rewritten'
NOT_MIGRATED_WORDING = 'which is not the id of a migrated entry; not rewritten'


def _rewriter(
    paths: Sequence[str] = (DEFAULT_ID_PATH,),
    mapping: Mapping[int, str] = MAPPING,
    *,
    key: str | None = None,
) -> MetadataIdRewriter:
    """Build a rewriter over the given path texts."""
    return MetadataIdRewriter([IdPath.parse(text) for text in paths], mapping, legacy_id_key=key)


def _rewrite(
    metadata: object,
    paths: Sequence[str] = (DEFAULT_ID_PATH,),
    mapping: Mapping[int, str] = MAPPING,
    *,
    key: str | None = None,
    old_id: int = 3,
) -> tuple[object, MetadataOutcome]:
    """Rewrite JSON-encoded metadata and return the parsed result with the outcome."""
    outcome = _rewriter(paths, mapping, key=key).rewrite(json.dumps(metadata), old_id)
    assert outcome.metadata is not None
    return json.loads(outcome.metadata), outcome


def _counts(rewritten: int = 0, not_migrated: int = 0, kept: int = 0) -> IdPathCounts:
    """Shorthand for one path's expected counts."""
    return IdPathCounts(rewritten=rewritten, not_migrated=not_migrated, kept=kept)


class TestValues:
    """What becomes of each value found at a path."""

    def test_integers_of_migrated_entries_become_new_ids(self) -> None:
        """An integer list at the default path is rewritten to the new 32-hex ids."""
        result, outcome = _rewrite({'references': {'context_ids': [1, 7]}})
        assert result == {'references': {'context_ids': [A, E]}}
        assert outcome.counts == {DEFAULT_ID_PATH: _counts(rewritten=2)}
        assert outcome.warnings == []
        assert outcome.errors == []

    def test_references_inside_lists_of_objects_are_rewritten(self) -> None:
        """The default path finds references in objects held by lists at any depth."""
        metadata = {
            'history': [{'note': 'first'}, {'references': {'context_ids': [5]}}],
            'a': {'b': [[{'references': {'context_ids': [2]}}]]},
        }
        result, outcome = _rewrite(metadata)
        assert result == {
            'history': [{'note': 'first'}, {'references': {'context_ids': [C]}}],
            'a': {'b': [[{'references': {'context_ids': [B]}}]]},
        }
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(rewritten=2)

    def test_top_level_list_metadata_is_rewritten(self) -> None:
        """Metadata that is a list still has its objects searched by the default path."""
        result, outcome = _rewrite([{'references': {'context_ids': [1]}}])
        assert result == [{'references': {'context_ids': [A]}}]
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(rewritten=1)

    def test_references_nested_inside_context_ids_are_rewritten(self) -> None:
        """An object element is kept as a value, and the references inside it are rewritten."""
        result, outcome = _rewrite({'references': {'context_ids': [{'references': {'context_ids': [1]}}]}})
        assert result == {'references': {'context_ids': [{'references': {'context_ids': [A]}}]}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(rewritten=1, kept=1)

    def test_references_held_in_a_list_are_not_found(self) -> None:
        """Named keys never step through a list, so a list under ``references`` is left alone."""
        metadata = {'references': [{'context_ids': [1]}]}
        result, outcome = _rewrite(metadata)
        assert result == metadata
        assert outcome.counts[DEFAULT_ID_PATH] == _counts()

    def test_non_numeric_string_is_kept(self) -> None:
        """A string that is not an integer's text stays as written and is counted as kept."""
        existing_id = 'd' * 32
        result, outcome = _rewrite({'references': {'context_ids': [existing_id]}}, mapping={})
        assert result == {'references': {'context_ids': [existing_id]}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(kept=1)
        assert outcome.warnings == []

    def test_numeric_string_of_a_migrated_entry_is_rewritten(self) -> None:
        """A numeric string naming a migrated entry becomes that entry's new id."""
        result, outcome = _rewrite({'references': {'context_ids': ['5']}}, mapping={5: C})
        assert result == {'references': {'context_ids': [C]}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(rewritten=1)

    def test_numeric_string_without_an_entry_is_not_migrated(self) -> None:
        """A numeric string naming no migrated entry stays a string and is warned about."""
        result, outcome = _rewrite({'references': {'context_ids': ['7']}}, mapping={})
        assert result == {'references': {'context_ids': ['7']}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(not_migrated=1)
        assert outcome.warnings == [
            f"row 3: metadata.references.context_ids[0] holds '7', {NOT_MIGRATED_WORDING}",
        ]

    def test_integer_without_an_entry_stays_an_integer(self) -> None:
        """An integer naming no migrated entry is kept as an integer, with one warning."""
        result, outcome = _rewrite({'references': {'context_ids': [9999]}}, mapping={}, old_id=42)
        assert result == {'references': {'context_ids': [9999]}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(not_migrated=1)
        assert outcome.warnings == [
            f'row 42: metadata.references.context_ids[0] holds 9999, {NOT_MIGRATED_WORDING}',
        ]
        assert outcome.errors == []

    @pytest.mark.parametrize(
        ('value', 'outcome_name'),
        [
            ('7', 'rewritten'),
            ('007', 'kept'),
            (' 7', 'kept'),
            ('7 ', 'kept'),
            ('+7', 'kept'),
            ('7.0', 'kept'),
            ('٧', 'kept'),
            ('1_0', 'kept'),
            ('-0', 'kept'),
            ('0x7', 'kept'),
            ('', 'kept'),
            ('0', 'not_migrated'),
            ('-5', 'not_migrated'),
            ('999', 'not_migrated'),
        ],
    )
    def test_only_the_canonical_integer_text_is_an_id(self, value: str, outcome_name: str) -> None:
        """A string counts as an id only in the form JSON writes an integer."""
        result, outcome = _rewrite({'references': {'context_ids': [value]}})
        expected = {'rewritten': E, 'kept': value, 'not_migrated': value}[outcome_name]
        assert result == {'references': {'context_ids': [expected]}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(**{outcome_name: 1})

    def test_values_that_are_not_ids_are_kept_without_an_error(self) -> None:
        """Booleans, floats, null, lists and objects stay as they are and count as kept."""
        elements = [True, False, 1.5, 5.0, None, [1], {'a': 1}]
        result, outcome = _rewrite({'references': {'context_ids': elements}})
        assert result == {'references': {'context_ids': elements}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(kept=7)
        assert outcome.warnings == []
        assert outcome.errors == []

    def test_mixed_list_counts_each_element(self) -> None:
        """Every element of a found list is one candidate."""
        elements = [1, 2, 99, 'd' * 32, '2', True, 1.5, None, [1], {'a': 1}]
        result, outcome = _rewrite({'references': {'context_ids': elements}})
        assert result == {'references': {'context_ids': [A, B, 99, 'd' * 32, B, True, 1.5, None, [1], {'a': 1}]}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(rewritten=3, not_migrated=1, kept=6)

    @pytest.mark.parametrize(('value', 'expected'), [(5, C), ('5', C), (99, 99), ('x', 'x'), (None, None)])
    def test_a_single_found_value_is_one_candidate(self, value: object, expected: object) -> None:
        """A path whose value is not a list finds one candidate, rewritten in place."""
        result, outcome = _rewrite({'references': {'context_ids': value}})
        assert result == {'references': {'context_ids': expected}}
        assert sum(vars(outcome.counts[DEFAULT_ID_PATH]).values()) == 1
        assert outcome.errors == []

    def test_absent_or_non_object_references_find_nothing(self) -> None:
        """A path that reaches no value counts nothing."""
        metadata = {'x': {'references': 'y'}, 'z': {'references': {}}}
        result, outcome = _rewrite(metadata)
        assert result == metadata
        assert outcome.counts[DEFAULT_ID_PATH] == _counts()

    def test_warning_names_the_location_walked(self) -> None:
        """A warning renders keys with dots and list indexes in brackets."""
        _, outcome = _rewrite({'items': [{'references': {'context_ids': [1, 404]}}]}, old_id=8)
        assert outcome.warnings == [
            f'row 8: metadata.items[0].references.context_ids[1] holds 404, {NOT_MIGRATED_WORDING}',
        ]


class TestPaths:
    """Paths the operator names, and which path owns a value."""

    def test_given_paths_replace_the_default(self) -> None:
        """Only the named paths are rewritten."""
        metadata = {'references': {'context_ids': [1]}, 'links': {'closed_by': [2], 'parent': 5}}
        result, outcome = _rewrite(metadata, ['links.closed_by'])
        assert result == {'references': {'context_ids': [1]}, 'links': {'closed_by': [B], 'parent': 5}}
        assert outcome.counts == {'links.closed_by': _counts(rewritten=1)}

    def test_wildcard_rewrites_every_key(self) -> None:
        """``*`` finds every key, numeric strings included."""
        metadata = {'links': {'parent': 1, 'closed_by': [2, 99], 'github_prs': ['5'], 'urls': ['https://x/7']}}
        result, outcome = _rewrite(metadata, ['links.*'])
        assert result == {'links': {'parent': A, 'closed_by': [B, 99], 'github_prs': [C], 'urls': ['https://x/7']}}
        assert outcome.counts == {'links.*': _counts(rewritten=3, not_migrated=1, kept=1)}

    def test_first_path_owns_a_location(self) -> None:
        """A location two paths find is rewritten once and counted under the first path given."""
        metadata = {'references': {'context_ids': [7]}, 'items': [{'references': {'context_ids': [1]}}]}
        result, outcome = _rewrite(metadata, ['references.context_ids', DEFAULT_ID_PATH])
        assert result == {'references': {'context_ids': [E]}, 'items': [{'references': {'context_ids': [A]}}]}
        assert outcome.counts == {
            'references.context_ids': _counts(rewritten=1),
            DEFAULT_ID_PATH: _counts(rewritten=1),
        }

    def test_first_path_owns_a_location_in_either_order(self) -> None:
        """Reversing the paths moves the shared location to the other path's counts."""
        metadata = {'references': {'context_ids': [7]}}
        _, outcome = _rewrite(metadata, ['**.context_ids', 'references.context_ids'])
        assert outcome.counts == {
            '**.context_ids': _counts(rewritten=1),
            'references.context_ids': _counts(),
        }

    def test_every_given_path_is_counted(self) -> None:
        """A path that finds nothing still has zeroed counts, in the order given."""
        _, outcome = _rewrite({'a': 1}, ['links.parent', DEFAULT_ID_PATH])
        assert list(outcome.counts) == ['links.parent', DEFAULT_ID_PATH]
        assert all(counts == _counts() for counts in outcome.counts.values())


class TestExcluding:
    """Ids of entries a run skips are not rewritten."""

    def test_excluded_id_is_not_migrated(self) -> None:
        """An id the rewriter is told was skipped stays as written, with the skipped wording."""
        rewriter = _rewriter().excluding(frozenset({2}))
        outcome = rewriter.rewrite(json.dumps({'references': {'context_ids': [1, 2, '2']}}), 9)
        assert outcome.metadata is not None
        assert json.loads(outcome.metadata) == {'references': {'context_ids': [A, 2, '2']}}
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(rewritten=1, not_migrated=2)
        assert outcome.warnings == [
            f'row 9: metadata.references.context_ids[1] holds 2, {SKIPPED_WORDING}',
            f"row 9: metadata.references.context_ids[2] holds '2', {SKIPPED_WORDING}",
        ]

    def test_excluding_leaves_the_original_rewriter_unchanged(self) -> None:
        """Excluding returns a new rewriter and the original still rewrites the id."""
        rewriter = _rewriter()
        rewriter.excluding(frozenset({2}))
        outcome = rewriter.rewrite(json.dumps({'references': {'context_ids': [2]}}), 9)
        assert outcome.metadata is not None
        assert json.loads(outcome.metadata) == {'references': {'context_ids': [B]}}

    def test_excluding_adds_to_ids_already_excluded(self) -> None:
        """Excluding twice keeps both sets of ids unmigrated."""
        rewriter = _rewriter().excluding(frozenset({1})).excluding(frozenset({2}))
        outcome = rewriter.rewrite(json.dumps({'references': {'context_ids': [1, 2, 5]}}), 9)
        assert outcome.counts[DEFAULT_ID_PATH] == _counts(rewritten=1, not_migrated=2)


class TestLegacyIdStamp:
    """The old integer id stamped under ``--legacy-id-key``."""

    def test_null_metadata_is_stamped(self) -> None:
        """An entry with no metadata gets an object holding only the stamp."""
        outcome = _rewriter(key='legacy_id').rewrite(None, 4)
        assert outcome.metadata == '{"legacy_id": 4}'
        assert outcome.stamp == 'stamped'
        assert outcome.counts == {DEFAULT_ID_PATH: _counts()}
        assert outcome.errors == []

    def test_null_metadata_without_a_key_stays_null(self) -> None:
        """An entry with no metadata keeps none when no key is given."""
        outcome = _rewriter().rewrite(None, 4)
        assert outcome == MetadataOutcome(
            metadata=None,
            counts={DEFAULT_ID_PATH: _counts()},
            stamp=None,
            warnings=[],
            errors=[],
        )

    def test_object_gets_the_stamp_last_as_an_integer(self) -> None:
        """The stamp is a JSON integer appended after the existing keys."""
        outcome = _rewriter(key='legacy_id').rewrite('{"b": 1, "a": 2}', 36717)
        assert outcome.metadata == '{"b": 1, "a": 2, "legacy_id": 36717}'
        assert outcome.stamp == 'stamped'

    def test_empty_object_is_stamped(self) -> None:
        """An empty object gets the stamp."""
        outcome = _rewriter(key='legacy_id').rewrite('{}', 2)
        assert outcome.metadata == '{"legacy_id": 2}'
        assert outcome.stamp == 'stamped'

    @pytest.mark.parametrize('existing', ['kept-as-is', 7, None, {'x': 1}])
    def test_existing_key_keeps_its_value(self, existing: object) -> None:
        """An entry that already holds the key keeps whatever value it holds."""
        result, outcome = _rewrite({'legacy_id': existing}, key='legacy_id', old_id=5)
        assert result == {'legacy_id': existing}
        assert outcome.stamp == 'already_present'
        assert outcome.errors == []

    def test_stamp_follows_the_rewrite_of_the_same_row(self) -> None:
        """A stamp under a key a path reaches is written after the rewrite, so it stays the old id."""
        result, outcome = _rewrite({'note': 'x'}, ['*'], key='legacy_id', old_id=1)
        assert result == {'note': 'x', 'legacy_id': 1}
        assert outcome.counts == {'*': _counts(kept=1)}
        assert outcome.stamp == 'stamped'

    def test_unparseable_metadata_is_not_stamped(self) -> None:
        """Text that does not parse is preserved verbatim and reported as not stamped."""
        outcome = _rewriter(key='legacy_id').rewrite('not json', 6)
        assert outcome.metadata == 'not json'
        assert outcome.stamp == 'not_stamped'
        assert outcome.counts == {DEFAULT_ID_PATH: _counts()}
        assert len(outcome.errors) == 1
        assert outcome.errors[0].startswith('row 6: metadata JSON parse failed (')
        assert outcome.errors[0].endswith("); no id in it was rewritten, and the old id was not stamped under 'legacy_id'")

    def test_unparseable_metadata_without_a_key_is_reported(self) -> None:
        """Text that does not parse is preserved verbatim with an error, and nothing is stamped."""
        outcome = _rewriter().rewrite('not json', 6)
        assert outcome.metadata == 'not json'
        assert outcome.stamp is None
        assert len(outcome.errors) == 1
        assert outcome.errors[0].startswith('row 6: metadata JSON parse failed (')
        assert outcome.errors[0].endswith('); no id in it was rewritten')

    @pytest.mark.parametrize(
        ('metadata', 'json_type'),
        [('[1, 2]', 'array'), ('"text"', 'string'), ('5', 'number'), ('1.5', 'number'), ('true', 'boolean'),
         ('null', 'null')],
    )
    def test_non_object_metadata_is_not_stamped(self, metadata: str, json_type: str) -> None:
        """Metadata that parses to anything but an object is kept and reported as not stamped."""
        outcome = _rewriter(key='legacy_id').rewrite(metadata, 6)
        assert outcome.metadata == metadata
        assert outcome.stamp == 'not_stamped'
        assert outcome.errors == [
            f"row 6: metadata is a JSON {json_type}, not an object, so the old id was not stamped under 'legacy_id'",
        ]

    def test_non_object_metadata_is_still_rewritten(self) -> None:
        """A list root that cannot be stamped is still rewritten by the default path."""
        outcome = _rewriter(key='legacy_id').rewrite('[{"references": {"context_ids": [1]}}]', 6)
        assert outcome.metadata == f'[{{"references": {{"context_ids": ["{A}"]}}}}]'
        assert outcome.counts == {DEFAULT_ID_PATH: _counts(rewritten=1)}
        assert outcome.stamp == 'not_stamped'
        assert len(outcome.errors) == 1


class TestReencode:
    """Metadata holding a number Python cannot write back as JSON is preserved verbatim."""

    def test_overflowing_number_is_preserved_not_re_emitted(self) -> None:
        """The original literal survives instead of the invalid token ``Infinity``."""
        outcome = _rewriter().rewrite('{"score": 1e400}', 1)
        assert outcome.metadata == '{"score": 1e400}'
        assert len(outcome.errors) == 1
        assert outcome.errors[0].startswith('row 1: metadata contains a number Python cannot round-trip as JSON (')
        assert outcome.errors[0].endswith('); no id in it was rewritten')

    @pytest.mark.parametrize('metadata', ['{"score": Infinity}', '{"a": NaN}', '{"score": -Infinity}'])
    def test_non_standard_tokens_are_preserved(self, metadata: str) -> None:
        """A source already holding a non-standard token keeps it byte for byte, with an error."""
        outcome = _rewriter().rewrite(metadata, 7)
        assert outcome.metadata == metadata
        assert len(outcome.errors) == 1
        assert outcome.errors[0].startswith('row 7: ')

    def test_discarded_rewrites_are_not_counted_or_warned(self) -> None:
        """Counts and warnings describe stored values, so a discarded rewrite carries none."""
        metadata = '{"references": {"context_ids": [1, 2, 404]}, "score": 1e400}'
        outcome = _rewriter(key='legacy_id').rewrite(metadata, 5)
        assert outcome.metadata == metadata
        assert outcome.counts == {DEFAULT_ID_PATH: _counts()}
        assert outcome.warnings == []
        assert outcome.stamp == 'not_stamped'
        assert len(outcome.errors) == 1
        assert outcome.errors[0].endswith(", and the old id was not stamped under 'legacy_id'")

    def test_non_object_root_reports_one_error_when_re_encode_fails(self) -> None:
        """A list root holding an overflowing number reports the preserved text once."""
        outcome = _rewriter(key='legacy_id').rewrite('[1e400]', 5)
        assert outcome.metadata == '[1e400]'
        assert outcome.stamp == 'not_stamped'
        assert len(outcome.errors) == 1
        assert 'cannot round-trip as JSON' in outcome.errors[0]

    def test_ids_beside_a_large_finite_number_are_rewritten_and_counted(self) -> None:
        """A large but finite number re-encodes, so the ids beside it are rewritten and counted."""
        result, outcome = _rewrite({'references': {'context_ids': [1, 2]}, 'score': 1e300})
        assert result == {'references': {'context_ids': [A, B]}, 'score': 1e300}
        assert outcome.counts == {DEFAULT_ID_PATH: _counts(rewritten=2)}
        assert outcome.errors == []

    def test_finite_metadata_still_round_trips(self) -> None:
        """Large but finite numbers and non-ASCII text are written back unchanged."""
        outcome = _rewriter().rewrite('{"score": 1e300, "name": "café", "a": [1, 2]}', 3)
        assert outcome.metadata is not None
        assert json.loads(outcome.metadata) == {'score': 1e300, 'name': 'café', 'a': [1, 2]}
        assert 'café' in outcome.metadata
        assert outcome.errors == []
