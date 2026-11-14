"""Source rows and expectations shared by the acceptance tests of the metadata id rewrite.

Every migration direction migrates the same rows and must produce the same metadata, so
the rows and what the target must hold afterward are written once here. Each row's
``text_content`` is ``entry-<old id>``, which finds the row on the target, where it has a
new id.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH

# A value that already is a 32-character id: kept as written.
HEX_ID = '0123456789abcdef0123456789abcdef'


def source_row(old_id: int, metadata: object, created_at: str) -> dict[str, object]:
    """Build one source row, whose text is ``entry-<old id>``, in the shape ``seed_source_db`` takes."""
    return {
        'id': old_id,
        'thread_id': 'thread-ids',
        'source': 'agent',
        'content_type': 'text',
        'text_content': f'entry-{old_id}',
        'metadata': metadata,
        'created_at': created_at,
    }


# Entry 1, the oldest, cites entry 3, a newer one, so its rewrite needs the complete id
# table. Entry 2 cites entry 1 by a numeric string and holds ids under ``links``, which
# the default path does not reach. Entry 3 holds the id of a migrated entry as an
# integer and as a numeric string (both rewritten), an integer naming no entry
# (not_migrated), and a 32-hex string, a zero-padded numeric string and a boolean
# (kept), plus a single id at depth. Floats, null, lists and objects at a path are
# covered by ``TestValues`` in ``test_metadata_ids.py``. Entry 4 has no metadata; entry
# 5 already holds ``legacy_id``.
SCENARIO_ROWS: list[dict[str, object]] = [
    source_row(1, {'references': {'context_ids': [3]}}, '2024-01-01 10:00:00'),
    source_row(
        2,
        {'references': {'context_ids': ['1']}, 'links': {'closed_by': [3], 'github_prs': ['1']}},
        '2024-02-01 10:00:00',
    ),
    source_row(
        3,
        {
            'references': {'context_ids': [1, '2', 999, HEX_ID, '0042', True]},
            'items': [{'references': {'context_ids': 2}}],
        },
        '2024-03-01 10:00:00',
    ),
    source_row(4, None, '2024-04-01 10:00:00'),
    source_row(5, {'legacy_id': 'kept-as-is'}, '2024-05-01 10:00:00'),
]

# The one value the default path finds that names no source entry.
NOT_MIGRATED_WARNING = (
    'row 3: metadata.references.context_ids[2] holds 999, which is not the id of a migrated entry; preserved'
)


@dataclass(frozen=True)
class TargetEntry:
    """One migrated entry as the target stores it.

    Attributes:
        id: The entry's new id, 32 lowercase hex characters.
        metadata_text: The stored metadata text, or None.
    """

    id: str
    metadata_text: str | None

    @property
    def metadata(self) -> object:
        """The stored metadata, parsed."""
        return None if self.metadata_text is None else json.loads(self.metadata_text)


def read_sqlite_target(path: Path) -> dict[str, TargetEntry]:
    """Read every entry of a migrated SQLite target, keyed by its text.

    Args:
        path: The target database file.

    Returns:
        Each entry's id and metadata by its ``text_content``.
    """
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute('SELECT id, text_content, metadata FROM context_entries').fetchall()
    finally:
        conn.close()
    return {text: TargetEntry(id=entry_id, metadata_text=metadata) for entry_id, text, metadata in rows}


def row_warnings(report: dict[str, object]) -> list[str]:
    """Return the warnings of a report that name a source row."""
    return [message for message in _strings(report['warnings']) if message.startswith('row ')]


def _strings(value: object) -> list[str]:
    """Return a report's message list as strings."""
    assert isinstance(value, list)
    return [str(message) for message in value]


def _metadata(rows: dict[str, TargetEntry], old_id: int) -> dict[str, object]:
    """Return the parsed metadata object of a migrated entry."""
    metadata = rows[f'entry-{old_id}'].metadata
    assert isinstance(metadata, dict)
    return metadata


def new_id(rows: dict[str, TargetEntry], old_id: int) -> str:
    """Return the new id of the entry migrated from ``old_id``."""
    return rows[f'entry-{old_id}'].id


def assert_default_rewrite(rows: dict[str, TargetEntry], report: dict[str, object]) -> None:
    """Check the target and report of a run that names no path, with or without a stamp.

    Args:
        rows: The migrated entries by text.
        report: The run's ``--report`` content.
    """
    assert len(rows) == len(SCENARIO_ROWS)
    assert _metadata(rows, 1)['references'] == {'context_ids': [new_id(rows, 3)]}
    assert _metadata(rows, 2)['references'] == {'context_ids': [new_id(rows, 1)]}
    assert _metadata(rows, 2)['links'] == {'closed_by': [3], 'github_prs': ['1']}
    entry_3 = _metadata(rows, 3)
    assert entry_3['references'] == {
        'context_ids': [new_id(rows, 1), new_id(rows, 2), 999, HEX_ID, '0042', True],
    }
    assert entry_3['items'] == [{'references': {'context_ids': new_id(rows, 2)}}]
    assert _metadata(rows, 5)['legacy_id'] == 'kept-as-is'
    assert report['id_paths'] == {DEFAULT_ID_PATH: {'rewritten': 5, 'not_migrated': 1, 'kept': 3}}
    assert row_warnings(report) == [NOT_MIGRATED_WARNING]


def assert_named_paths(rows: dict[str, TargetEntry], report: dict[str, object]) -> None:
    """Check a run given ``--rewrite-ids-at links.closed_by`` alone.

    The given path replaces the default, so only ``links.closed_by`` is rewritten and
    every ``references.context_ids`` value stays as the source wrote it.

    Args:
        rows: The migrated entries by text.
        report: The run's ``--report`` content.
    """
    assert _metadata(rows, 2)['links'] == {'closed_by': [new_id(rows, 3)], 'github_prs': ['1']}
    assert _metadata(rows, 1)['references'] == {'context_ids': [3]}
    assert _metadata(rows, 2)['references'] == {'context_ids': ['1']}
    entry_3 = _metadata(rows, 3)
    assert entry_3['references'] == {'context_ids': [1, '2', 999, HEX_ID, '0042', True]}
    assert entry_3['items'] == [{'references': {'context_ids': 2}}]
    assert report['id_paths'] == {'links.closed_by': {'rewritten': 1, 'not_migrated': 0, 'kept': 0}}
    assert row_warnings(report) == []


def assert_legacy_stamp(rows: dict[str, TargetEntry], report: dict[str, object]) -> None:
    """Check a run given ``--legacy-id-key legacy_id``.

    Args:
        rows: The migrated entries by text.
        report: The run's ``--report`` content.
    """
    for old_id in (1, 2, 3):
        stamp = _metadata(rows, old_id)['legacy_id']
        assert stamp == old_id
        assert type(stamp) is int
    assert rows['entry-4'].metadata == {'legacy_id': 4}
    assert _metadata(rows, 5)['legacy_id'] == 'kept-as-is'
    assert report['legacy_id'] == {'key': 'legacy_id', 'stamped': 4, 'already_present': 1, 'not_stamped': 0}
