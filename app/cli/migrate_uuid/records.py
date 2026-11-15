"""Statistics and options records of the integer-to-UUIDv7 migration."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal
from typing import cast

from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.cli.migrate_uuid.id_paths import IdPath

# What became of a migrated entry's old id under ``--legacy-id-key``: written under the key,
# left alone because the entry already held the key, or not written because the metadata
# could not take it (an error says why).
type StampOutcome = Literal['stamped', 'already_present', 'not_stamped']


@dataclass
class IdPathCounts:
    """The values one ``--rewrite-ids-at`` path found and how each was treated.

    The counts cover one row's rewrite, or the sum over rows.

    Attributes:
        rewritten: Values that became a migrated entry's new id.
        not_migrated: Integers, or strings holding an integer in the form JSON writes one,
            that are not the id of a migrated entry, kept as written.
        kept: Every other value found at the path, kept as written.
    """

    rewritten: int = 0
    not_migrated: int = 0
    kept: int = 0

    def add(self, other: 'IdPathCounts') -> None:
        """Add another set of counts to these.

        Args:
            other: The counts to add.
        """
        self.rewritten += other.rewritten
        self.not_migrated += other.not_migrated
        self.kept += other.kept


@dataclass
class LegacyIdCounts:
    """What ``--legacy-id-key`` did to the entries that reached the target.

    Attributes:
        key: The top-level metadata key the old id is stamped under.
        stamped: Entries that received the old id under the key.
        already_present: Entries that already held the key and so received no stamp.
        not_stamped: Entries whose metadata could not take the stamp; each has an error.
    """

    key: str
    stamped: int = 0
    already_present: int = 0
    not_stamped: int = 0

    def add(self, outcome: StampOutcome) -> None:
        """Count one entry's stamp outcome.

        Args:
            outcome: What became of the entry's old id.
        """
        match outcome:
            case 'stamped':
                self.stamped += 1
            case 'already_present':
                self.already_present += 1
            case 'not_stamped':
                self.not_stamped += 1


@dataclass
class MetadataOutcome:
    """The rewritten metadata of one entry and what the rewrite found.

    Attributes:
        metadata: The metadata text to store on the target, or None for none.
        counts: Per path text, every given path present, the values this row's rewrite
            counted at it; all zero when the metadata is kept verbatim.
        stamp: What became of the old id under ``--legacy-id-key``; None when no key is given.
        warnings: One per value counted ``not_migrated``; none when the metadata is kept verbatim.
        errors: Why the metadata was preserved verbatim or not stamped.
    """

    metadata: str | None
    counts: dict[str, IdPathCounts]
    stamp: StampOutcome | None
    warnings: list[str]
    errors: list[str]


@dataclass
class MigrationOptions:
    """Parsed CLI arguments.

    Attributes:
        source_url: The source database: a SQLite file path or URL, or a
            PostgreSQL URL, as ``app.cli._database_url.parse_backend_url``
            classifies it.
        target_url: The target database, in the same forms as ``source_url``.
        dry_run: When True, run the full migration logic in memory but
            issue no INSERT statements against the target.
        report_path: Optional path. When set, write the migration
            statistics as JSON to this file at end of run.
        id_paths: The metadata paths whose entry ids are rewritten, in
            command-line order; a location two paths find belongs to the first.
        legacy_id_key: The top-level metadata key each migrated entry's old
            integer id is stamped under, or None to stamp nothing.
    """

    source_url: str
    target_url: str
    dry_run: bool = False
    report_path: Path | None = None
    id_paths: tuple[IdPath, ...] = field(default_factory=lambda: (IdPath.parse(DEFAULT_ID_PATH),))
    legacy_id_key: str | None = None


@dataclass
class MigrationStats:
    """Counters and warning/error log for a migration run.

    Attributes:
        rows_migrated: Number of ``context_entries`` rows copied to the
            target.
        id_paths: Per ``--rewrite-ids-at`` path text, in command-line order,
            what the path found in the metadata of the rows copied to the target.
        legacy_id: What ``--legacy-id-key`` did to the rows copied to the
            target, or None when no key is given.
        tags_migrated: Number of tag rows copied.
        images_migrated: Number of ``image_attachments`` rows copied.
        embedding_metadata_migrated: Number of ``embedding_metadata`` rows
            copied.
        embedding_chunks_migrated: Number of ``embedding_chunks`` rows
            copied (SQLite).
        vec_rows_migrated: Number of ``vec_context_embeddings`` rows
            copied.
        fts_rebuilt: Whether the FTS5 index rebuild succeeded on the
            target.
        warnings: Free-form warning messages.
        errors: Free-form error messages.
    """

    rows_migrated: int = 0
    id_paths: dict[str, IdPathCounts] = field(default_factory=dict)
    legacy_id: LegacyIdCounts | None = None
    tags_migrated: int = 0
    images_migrated: int = 0
    embedding_metadata_migrated: int = 0
    embedding_chunks_migrated: int = 0
    vec_rows_migrated: int = 0
    fts_rebuilt: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def start_metadata_counts(self, options: MigrationOptions) -> None:
        """Zero the counts of every given path and, with a stamp key, the stamp counts.

        Every path gets an entry, so the summary and the report list a path that
        matched nothing too.

        Args:
            options: The run's options, naming the paths and the stamp key.
        """
        self.id_paths = {path.text: IdPathCounts() for path in options.id_paths}
        self.legacy_id = None if options.legacy_id_key is None else LegacyIdCounts(key=options.legacy_id_key)

    def record_metadata(self, outcome: MetadataOutcome, *, migrated: bool) -> None:
        """Add one row's metadata outcome.

        The warnings and errors are always added. The counts are added only when
        the row reaches the target, so they describe stored values.

        Args:
            outcome: The row's rewrite outcome.
            migrated: Whether the row is written to the target (or would be, in
                a dry run).

        Raises:
            ValueError: The outcome carries a stamp but ``start_metadata_counts``
                was not given a stamp key.
        """
        self.warnings.extend(outcome.warnings)
        self.errors.extend(outcome.errors)
        if not migrated:
            return
        for text, counts in outcome.counts.items():
            self.id_paths[text].add(counts)
        if outcome.stamp is not None:
            if self.legacy_id is None:
                raise ValueError('a stamp outcome was recorded before start_metadata_counts was given a stamp key')
            self.legacy_id.add(outcome.stamp)

    def to_dict(self) -> dict[str, object]:
        """Return a plain-dict view suitable for :func:`json.dump`.

        Returns:
            Dictionary with the same key ordering as the dataclass field
            declaration order.
        """
        return cast(dict[str, object], asdict(self))
