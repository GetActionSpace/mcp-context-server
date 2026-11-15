"""Tests for the migration's option and statistics records."""

import json

import pytest

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.id_paths import IdPath
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import LegacyIdCounts
from app.cli.migrate_uuid.records import MetadataOutcome
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.records import StampOutcome


def _options(*paths: str, key: str | None = None) -> MigrationOptions:
    """Options naming the given paths and stamp key."""
    return MigrationOptions(
        source_url='source.db',
        target_url='target.db',
        id_paths=tuple(IdPath.parse(text) for text in paths),
        legacy_id_key=key,
    )


def _outcome(
    counts: dict[str, IdPathCounts],
    *,
    stamp: StampOutcome | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> MetadataOutcome:
    """An outcome carrying the given counts, stamp and messages."""
    return MetadataOutcome(
        metadata='{}',
        counts=counts,
        stamp=stamp,
        warnings=warnings or [],
        errors=errors or [],
    )


class TestMigrationOptions:
    """Defaults the runners rely on."""

    def test_defaults_rewrite_the_default_path_and_stamp_nothing(self) -> None:
        """Options built without id_paths or legacy_id_key rewrite the default path only and stamp nothing."""
        options = MigrationOptions(source_url='source.db', target_url='target.db')
        assert options.id_paths == (IdPath.parse(DEFAULT_ID_PATH),)
        assert options.legacy_id_key is None


class TestStartMetadataCounts:
    """Zeroed counts for every given path, and the stamp record only with a key."""

    def test_one_zeroed_entry_per_path_in_order(self) -> None:
        """Every path is present, also one that will match nothing, in command-line order."""
        stats = MigrationStats()
        stats.start_metadata_counts(_options('links.*', DEFAULT_ID_PATH))
        assert list(stats.id_paths) == ['links.*', DEFAULT_ID_PATH]
        assert all(counts == IdPathCounts() for counts in stats.id_paths.values())
        assert stats.legacy_id is None

    def test_stamp_record_only_with_a_key(self) -> None:
        """A stamp key starts a zeroed stamp record under that key."""
        stats = MigrationStats()
        stats.start_metadata_counts(_options(DEFAULT_ID_PATH, key='legacy_id'))
        assert stats.legacy_id == LegacyIdCounts(key='legacy_id')


class TestRecordMetadata:
    """Counts describe stored rows only; messages are always kept."""

    def test_migrated_row_adds_counts_and_stamp(self) -> None:
        """A row that reaches the target adds its per-path counts and its stamp outcome."""
        stats = MigrationStats()
        stats.start_metadata_counts(_options('links.*', DEFAULT_ID_PATH, key='legacy_id'))
        stats.record_metadata(
            _outcome(
                {'links.*': IdPathCounts(rewritten=2, kept=1), DEFAULT_ID_PATH: IdPathCounts(not_migrated=1)},
                stamp='stamped',
                warnings=['w1'],
            ),
            migrated=True,
        )
        stats.record_metadata(
            _outcome({'links.*': IdPathCounts(rewritten=1), DEFAULT_ID_PATH: IdPathCounts()}, stamp='already_present'),
            migrated=True,
        )
        assert stats.id_paths == {
            'links.*': IdPathCounts(rewritten=3, kept=1),
            DEFAULT_ID_PATH: IdPathCounts(not_migrated=1),
        }
        assert stats.legacy_id == LegacyIdCounts(key='legacy_id', stamped=1, already_present=1)
        assert stats.warnings == ['w1']

    def test_unmigrated_row_adds_only_its_messages(self) -> None:
        """A row the target does not receive adds no counts, but keeps its warnings and errors."""
        stats = MigrationStats()
        stats.start_metadata_counts(_options(DEFAULT_ID_PATH, key='legacy_id'))
        stats.record_metadata(
            _outcome({DEFAULT_ID_PATH: IdPathCounts(rewritten=4)}, stamp='stamped', warnings=['w'], errors=['e']),
            migrated=False,
        )
        assert stats.id_paths == {DEFAULT_ID_PATH: IdPathCounts()}
        assert stats.legacy_id == LegacyIdCounts(key='legacy_id')
        assert stats.warnings == ['w']
        assert stats.errors == ['e']

    def test_not_stamped_is_counted(self) -> None:
        """A row whose stamp failed counts as not stamped and keeps its error."""
        stats = MigrationStats()
        stats.start_metadata_counts(_options(DEFAULT_ID_PATH, key='legacy_id'))
        stats.record_metadata(
            _outcome({DEFAULT_ID_PATH: IdPathCounts()}, stamp='not_stamped', errors=['row 6: ...']),
            migrated=True,
        )
        assert stats.legacy_id == LegacyIdCounts(key='legacy_id', not_stamped=1)
        assert stats.errors == ['row 6: ...']

    def test_stamp_outcome_needs_a_started_stamp_record(self) -> None:
        """A stamp outcome recorded before the stamp record was started is refused."""
        stats = MigrationStats()
        stats.start_metadata_counts(_options(DEFAULT_ID_PATH))
        with pytest.raises(ValueError, match='start_metadata_counts'):
            stats.record_metadata(_outcome({DEFAULT_ID_PATH: IdPathCounts()}, stamp='stamped'), migrated=True)


class TestToDict:
    """The report renders the nested records as objects."""

    def test_report_renders_paths_and_stamp(self) -> None:
        """``id_paths`` maps each path to its counts and ``legacy_id`` is an object."""
        stats = MigrationStats(rows_migrated=3)
        stats.start_metadata_counts(_options(DEFAULT_ID_PATH, key='legacy_id'))
        stats.record_metadata(
            _outcome({DEFAULT_ID_PATH: IdPathCounts(rewritten=4, not_migrated=1, kept=1)}, stamp='stamped'),
            migrated=True,
        )
        report = json.loads(json.dumps(stats.to_dict()))
        assert report['id_paths'] == {DEFAULT_ID_PATH: {'rewritten': 4, 'not_migrated': 1, 'kept': 1}}
        assert report['legacy_id'] == {'key': 'legacy_id', 'stamped': 1, 'already_present': 0, 'not_stamped': 0}
        assert list(report)[:3] == ['rows_migrated', 'id_paths', 'legacy_id']

    def test_report_without_a_stamp_key_has_null_legacy_id(self) -> None:
        """``legacy_id`` is null when no stamp key is given."""
        stats = MigrationStats()
        stats.start_metadata_counts(_options(DEFAULT_ID_PATH))
        report = json.loads(json.dumps(stats.to_dict()))
        assert report['legacy_id'] is None
        assert report['id_paths'] == {DEFAULT_ID_PATH: {'rewritten': 0, 'not_migrated': 0, 'kept': 0}}
