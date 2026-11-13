"""Tests for the PostgreSQL storability and index-size pre-check helpers.

A NUL (U+0000), an unpaired UTF-16 surrogate or a non-finite JSON number is legal in
a SQLite value but fatal in its PostgreSQL column, and SQLite indexes a value of any
size while PostgreSQL refuses an index tuple larger than BTMaxItemSize (2704 bytes on
btree version 4). These helpers name such a value per column so the cross-backend
copy loops skip its row instead of aborting the run. The index-size tests pin two
properties:

* the payload budget is derived from the widest index each value feeds, so a value
  that passes really is indexable (thread_id shares its tuple with the ``source`` and
  ``content_hash`` columns of idx_context_entries_dedup_hash, while a tag and a
  string-typed metadata field are indexed on their own);
* every column the target schema indexes is covered, including the values stored
  under string-typed ``METADATA_INDEXED_FIELDS`` keys, whose expression index
  ``idx_metadata_<field>`` carries the same ceiling.
"""

import json

import pytest

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.id_paths import IdPath
from app.cli.migrate_uuid.metadata_ids import MetadataIdRewriter
from app.cli.migrate_uuid.pg_prechecks import PG_BTREE_MAX_ITEM_BYTES
from app.cli.migrate_uuid.pg_prechecks import PG_MAX_INDEXED_THREAD_ID_BYTES
from app.cli.migrate_uuid.pg_prechecks import PG_MAX_INDEXED_VALUE_BYTES
from app.cli.migrate_uuid.pg_prechecks import first_pg_unindexable_metadata_field
from app.cli.migrate_uuid.pg_prechecks import pg_unindexable_column_reason
from app.cli.migrate_uuid.pg_prechecks import pg_unstorable_column_reason
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import MetadataOutcome
from app.settings import get_settings


def _rewrite(metadata_json: str, mapping: dict[int, str], old_id: int) -> MetadataOutcome:
    """Rewrite one row's metadata at the default path, as every migration direction does."""
    return MetadataIdRewriter([IdPath.parse(DEFAULT_ID_PATH)], mapping, legacy_id_key=None).rewrite(metadata_json, old_id)


class TestUnstorableColumnReason:
    """Unit coverage for the per-column detection helper, including the jsonb escape."""

    def test_raw_nul_and_surrogate_detected_clean_passes(self) -> None:
        """A raw TEXT NUL/surrogate is flagged; clean text and None pass."""
        assert pg_unstorable_column_reason('a\x00b', is_jsonb=False) is not None
        assert pg_unstorable_column_reason('\ud800', is_jsonb=False) is not None
        assert pg_unstorable_column_reason('clean value', is_jsonb=False) is None
        assert pg_unstorable_column_reason(None, is_jsonb=False) is None

    def test_jsonb_escape_needs_the_decoded_check(self) -> None:
        """A metadata NUL serializes to the \\u0000 escape (no literal byte); only the
        decoded-structure check under is_jsonb=True catches it."""
        escaped = json.dumps({'note': 'has\x00nul'}, ensure_ascii=False)
        assert '\x00' not in escaped  # stored as the six-char escape, not a literal NUL
        # The raw-string check alone misses the escape ...
        assert pg_unstorable_column_reason(escaped, is_jsonb=False) is None
        # ... but the jsonb path decodes and detects it.
        assert pg_unstorable_column_reason(escaped, is_jsonb=True) is not None

    def test_jsonb_literal_nul_and_malformed_json(self) -> None:
        """A literal NUL in the serialized jsonb is flagged; unparseable JSON destined for a
        jsonb column is itself unstorable (the ``::jsonb`` cast rejects it mid-transaction),
        while the same unparseable content in a raw TEXT column has no cast and passes."""
        assert pg_unstorable_column_reason('{"k": "a\x00b"}', is_jsonb=True) is not None
        assert pg_unstorable_column_reason('{not valid json', is_jsonb=True) is not None
        assert pg_unstorable_column_reason('{not valid json', is_jsonb=False) is None


class TestNonFiniteJsonNumbers:
    """Unit coverage for the jsonb non-finite-number branch of the same helper."""

    @pytest.mark.parametrize(
        'metadata_json',
        [
            '{"score": 1e400}',        # standard JSON that json.loads turns into inf
            '{"score": Infinity}',     # non-standard token json.loads accepts
            '{"score": -Infinity}',
            '{"a": NaN}',
            '{"a": {"b": [1.0, NaN]}}',  # nested, reached by the walker
        ],
    )
    def test_non_finite_number_is_unstorable_in_jsonb(self, metadata_json: str) -> None:
        """Every spelling that decodes to a non-finite float is flagged for a jsonb column."""
        assert pg_unstorable_column_reason(metadata_json, is_jsonb=True) is not None

    def test_finite_numbers_and_clean_json_pass(self) -> None:
        """Ordinary finite numbers stay storable."""
        assert pg_unstorable_column_reason('{"score": 1.5}', is_jsonb=True) is None
        assert pg_unstorable_column_reason('{"score": 1e300}', is_jsonb=True) is None
        assert pg_unstorable_column_reason('{"a": [1, 2, 3]}', is_jsonb=True) is None

    def test_message_names_the_offending_value(self) -> None:
        """The recorded reason identifies the non-finite float so the row can be repaired."""
        reason = pg_unstorable_column_reason('{"score": 1e400}', is_jsonb=True)
        assert reason is not None
        assert 'Non-finite float' in reason

    def test_rewrite_never_manufactures_an_invalid_token(self) -> None:
        """Re-serialization preserves the source literal instead of emitting Infinity.

        Re-encoding is where the invalid token would be created, on ALL FOUR migration
        directions: json.loads turns 1e400 into inf and a default json.dumps writes it
        back as the token ``Infinity``, which no RFC 8259 parser accepts. That would
        convert valid source metadata into invalid target metadata on every SQLite
        target (json_valid flips to 0) and abort the transaction on every PostgreSQL
        target.
        """
        outcome = _rewrite('{"score": 1e400}', {}, 1)

        assert outcome.metadata == '{"score": 1e400}'
        assert json.loads(outcome.metadata)  # still parseable JSON
        assert 'Infinity' not in outcome.metadata
        # The skipped rewrite is reported, and a non-empty error list exits non-zero.
        assert len(outcome.errors) == 1
        assert 'row 1' in outcome.errors[0]

    def test_discarded_rewrites_are_not_reported_as_rewritten(self) -> None:
        """Rewrites the encoder rejects are not counted as rewrites that landed.

        The rewrite counts each value as it changes the parsed structure, but the
        re-encode then fails and the ORIGINAL metadata -- still carrying the integer ids
        -- is what reaches the target. Counting those values makes the run summary and
        the --report JSON claim rewrites the target does not have, contradicting the error
        the same branch records.
        """
        metadata_json = '{"references": {"context_ids": [1, 2, 3]}, "score": 1e400}'
        mapping = {1: 'a' * 32, 2: 'b' * 32, 3: 'c' * 32}

        outcome = _rewrite(metadata_json, mapping, 5)

        assert outcome.metadata == metadata_json
        assert json.loads(outcome.metadata)['references']['context_ids'] == [1, 2, 3]
        assert outcome.counts == {DEFAULT_ID_PATH: IdPathCounts()}
        assert len(outcome.errors) == 1
        assert 'row 5' in outcome.errors[0]

    def test_successful_rewrites_are_still_counted(self) -> None:
        """The discarded counts are confined to the re-encode failure."""
        outcome = _rewrite(
            '{"references": {"context_ids": [1, 2]}, "score": 1e300}',
            {1: 'a' * 32, 2: 'b' * 32},
            6,
        )

        assert outcome.metadata is not None
        assert json.loads(outcome.metadata)['references']['context_ids'] == ['a' * 32, 'b' * 32]
        assert outcome.counts == {DEFAULT_ID_PATH: IdPathCounts(rewritten=2)}
        assert outcome.errors == []

    @pytest.mark.parametrize(
        'metadata_json',
        ['{"score": Infinity}', '{"a": NaN}', '{"score": -Infinity}'],
    )
    def test_non_standard_source_tokens_are_preserved_not_re_emitted(self, metadata_json: str) -> None:
        """A source already carrying a non-standard token is preserved and reported.

        Rewriting it would re-emit the same invalid token; preserving it verbatim keeps
        the target byte-identical to the source, and the recorded error tells the
        operator to repair the value.
        """
        outcome = _rewrite(metadata_json, {}, 7)

        assert outcome.metadata == metadata_json
        assert len(outcome.errors) == 1
        assert 'row 7' in outcome.errors[0]

    def test_finite_metadata_still_round_trips(self) -> None:
        """The guard does not disturb ordinary metadata."""
        outcome = _rewrite('{"score": 1e300, "a": [1, 2]}', {}, 3)

        assert outcome.metadata is not None
        assert json.loads(outcome.metadata) == {'score': 1e300, 'a': [1, 2]}
        assert outcome.errors == []


# Bytes an index tuple spends on things other than the checked payload, used by the
# arithmetic assertions below: the MAXALIGNed IndexTupleData header with a null bitmap
# (16) plus the long varlena header a text datum past 126 bytes carries (4).
_INDEX_TUPLE_HEADER_BYTES = 16 + 4


# The trailing columns of idx_context_entries_dedup_hash(thread_id, source, content_hash):
# 'agent' as a short-header varlena (6) and a 64-character SHA-256 hex string (65).
_DEDUP_TRAILING_COLUMN_BYTES = 6 + 65


class TestBtreeBudgets:
    """The accepted payload sizes really fit the index tuples they land in."""

    def test_thread_id_budget_fits_the_dedup_index_tuple(self) -> None:
        """A thread_id at the budget still fits idx_context_entries_dedup_hash.

        The base schema declares that index on (thread_id, source, content_hash), so the
        thread_id payload shares its tuple with two more datums. A budget that accounts
        only for the tuple header lets a value pass the guard and then abort the whole
        run with ``index row size ... exceeds btree version 4 maximum 2704``.
        """
        widest_tuple = (
            _INDEX_TUPLE_HEADER_BYTES + PG_MAX_INDEXED_THREAD_ID_BYTES + _DEDUP_TRAILING_COLUMN_BYTES
        )
        assert widest_tuple <= PG_BTREE_MAX_ITEM_BYTES

    def test_single_column_budget_fits_its_index_tuple(self) -> None:
        """A tag or string-typed metadata value at the budget fits its own index tuple."""
        assert _INDEX_TUPLE_HEADER_BYTES + PG_MAX_INDEXED_VALUE_BYTES <= PG_BTREE_MAX_ITEM_BYTES

    def test_thread_id_budget_is_the_narrower_one(self) -> None:
        """thread_id gets less room than a value indexed on its own."""
        assert PG_MAX_INDEXED_THREAD_ID_BYTES < PG_MAX_INDEXED_VALUE_BYTES

    def test_boundary_values_are_accepted_and_rejected(self) -> None:
        """Each budget accepts its exact size and rejects one byte more."""
        for budget in (PG_MAX_INDEXED_THREAD_ID_BYTES, PG_MAX_INDEXED_VALUE_BYTES):
            assert pg_unindexable_column_reason('a' * budget, budget) is None
            reason = pg_unindexable_column_reason('a' * (budget + 1), budget)
            assert reason is not None
            assert str(budget) in reason

    def test_multibyte_values_are_measured_in_utf8_bytes(self) -> None:
        """A short string of wide code points can still exceed the byte budget."""
        # Each code point encodes to 3 UTF-8 bytes, so half the budget in characters
        # is one and a half budgets in bytes.
        value = '中' * PG_MAX_INDEXED_THREAD_ID_BYTES
        assert pg_unindexable_column_reason(value, PG_MAX_INDEXED_THREAD_ID_BYTES) is not None

    def test_none_and_ordinary_values_pass(self) -> None:
        """Absent and normally sized values are never flagged."""
        assert pg_unindexable_column_reason(None, PG_MAX_INDEXED_VALUE_BYTES) is None
        assert pg_unindexable_column_reason('thread-1', PG_MAX_INDEXED_VALUE_BYTES) is None


class TestIndexedMetadataFields:
    """Values under string-typed METADATA_INDEXED_FIELDS keys are checked too."""

    def test_oversized_string_under_an_indexed_key_is_flagged(self) -> None:
        """A default indexed field carrying an oversized value names the field."""
        metadata = json.dumps({'task_name': 'a' * 3000})
        found = first_pg_unindexable_metadata_field(metadata)

        assert found is not None
        column, reason = found
        assert column == 'metadata.task_name'
        assert '3000 UTF-8 bytes' in reason

    def test_oversized_value_under_a_non_indexed_key_passes(self) -> None:
        """Only indexed keys are capped; jsonb itself imposes no such limit."""
        metadata = json.dumps({'notes': 'a' * 5000})
        assert first_pg_unindexable_metadata_field(metadata) is None

    def test_array_and_object_fields_are_not_expression_indexed(self) -> None:
        """The default ``references``/``technologies`` fields are served by the GIN index.

        They are excluded from expression indexing on both backends, so a large value
        under them is perfectly storable and must not cost the row its migration.
        """
        metadata = json.dumps(
            {
                'references': {'context_ids': [f'{index:032x}' for index in range(300)]},
                'technologies': ['python'] * 2000,
            },
        )
        assert len(metadata) > PG_MAX_INDEXED_VALUE_BYTES
        assert first_pg_unindexable_metadata_field(metadata) is None

    def test_container_under_a_string_typed_key_is_measured_as_serialized_text(self) -> None:
        """``metadata->>'<field>'`` yields a container's serialized form, which is indexed."""
        metadata = json.dumps({'project': ['a' * 100] * 40})
        found = first_pg_unindexable_metadata_field(metadata)

        assert found is not None
        assert found[0] == 'metadata.project'

    def test_typed_field_holding_an_uncastable_value_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value the target's index cast rejects loses its row, not the whole run.

        SQLite indexes ``{"priority": "high"}`` under an integer-typed field without a
        cast and stores it happily; the PostgreSQL expression index evaluates
        ``(metadata->>'priority')::INTEGER`` on every INSERT and refuses it, aborting
        the transaction with a raw driver error that names no source row.

        Args:
            monkeypatch: Used to configure a typed indexed field.
        """
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'priority:integer')
        get_settings.cache_clear()

        found = first_pg_unindexable_metadata_field(json.dumps({'priority': 'high'}))

        assert found is not None
        column, reason = found
        assert column == 'metadata.priority'
        assert 'not a valid integer' in reason
        assert 'the PostgreSQL expression index evaluates that cast on every INSERT' in reason
        assert 'skipped' in reason

    def test_float_field_holding_a_non_number_is_flagged_by_its_declared_type(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-number under a float-typed field loses its row because of the declared type.

        The float index casts JSON numbers only, so the target could store the value;
        the row is skipped because the running server refuses a value that is not a
        float under a field declared float, and the reason says so instead of citing
        an index cast.

        Args:
            monkeypatch: Used to configure a float-typed indexed field.
        """
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'score:float')
        get_settings.cache_clear()

        found = first_pg_unindexable_metadata_field(json.dumps({'score': 'high'}))

        assert found is not None
        column, reason = found
        assert column == 'metadata.score'
        assert 'not a valid float' in reason
        assert 'METADATA_INDEXED_FIELDS declares the field float' in reason
        assert 'expression index' not in reason
        assert 'skipped' in reason

    def test_float_field_holding_a_number_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A JSON number and a numeric string under a float-typed field migrate untouched.

        Args:
            monkeypatch: Used to configure a float-typed indexed field.
        """
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'score:float')
        get_settings.cache_clear()

        assert first_pg_unindexable_metadata_field(json.dumps({'score': 3.5})) is None
        assert first_pg_unindexable_metadata_field(json.dumps({'score': '3.5'})) is None

    def test_typed_field_holding_a_castable_value_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value the cast accepts migrates untouched.

        Args:
            monkeypatch: Used to configure a typed indexed field.
        """
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'priority:integer')
        get_settings.cache_clear()

        assert first_pg_unindexable_metadata_field(json.dumps({'priority': 7})) is None
        assert first_pg_unindexable_metadata_field(json.dumps({'priority': '7'})) is None

    def test_json_null_and_small_values_pass(self) -> None:
        """A JSON null indexes as SQL NULL (excluded by the index predicate)."""
        assert first_pg_unindexable_metadata_field(json.dumps({'task_name': None})) is None
        assert first_pg_unindexable_metadata_field(json.dumps({'task_name': 'audit'})) is None
        assert first_pg_unindexable_metadata_field(json.dumps({'status': 42})) is None

    def test_absent_unparseable_and_non_object_metadata_pass(self) -> None:
        """Shapes the jsonb bind rejects on its own are left to the unstorable check."""
        assert first_pg_unindexable_metadata_field(None) is None
        assert first_pg_unindexable_metadata_field('{not json') is None
        assert first_pg_unindexable_metadata_field('[1, 2, 3]') is None
