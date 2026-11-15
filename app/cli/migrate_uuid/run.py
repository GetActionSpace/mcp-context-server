"""Entry point of the integer-to-UUIDv7 migration: backend dispatch, summary and report."""

import asyncio
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from app.cli._database_url import mask_credentials
from app.cli._database_url import parse_backend_url
from app.cli.migrate_uuid.id_paths import IdPath
from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql
from app.cli.migrate_uuid.postgresql_to_sqlite import run_migration_mixed_postgresql_to_sqlite
from app.cli.migrate_uuid.records import IdPathCounts
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.sqlite_to_postgresql import run_migration_mixed_sqlite_to_postgresql
from app.cli.migrate_uuid.sqlite_to_sqlite import run_migration_sqlite_to_sqlite
from app.errors import ConfigurationError

logger = logging.getLogger(__name__)


def _id_path_line(text: str, counts: IdPathCounts) -> str:
    """Render the summary line of one ``--rewrite-ids-at`` path."""
    not_migrated = 'not the id of a migrated entry' if counts.not_migrated == 1 else 'not ids of migrated entries'
    kept = 'other value kept' if counts.kept == 1 else 'other values kept'
    return (
        f'  ids at {text}: {counts.rewritten} rewritten, {counts.not_migrated} {not_migrated}, '
        f'{counts.kept} {kept}'
    )


def print_summary(stats: MigrationStats, source_url: str, target_url: str, dry_run: bool = False) -> None:
    """Print a human-readable summary of the migration to stdout."""
    source_display = mask_credentials(source_url)
    target_display = mask_credentials(target_url)
    print('Migration summary')
    print(f'  source: {source_display}')
    print(f'  target: {target_display}')
    print(f'  rows migrated: {stats.rows_migrated}')
    for text, counts in stats.id_paths.items():
        print(_id_path_line(text, counts))
    if stats.legacy_id is not None:
        legacy = stats.legacy_id
        print(
            f'  legacy id key {legacy.key!r}: {legacy.stamped} stamped, '
            f'{legacy.already_present} already present, {legacy.not_stamped} not stamped',
        )
    print(f'  tags migrated: {stats.tags_migrated}')
    print(f'  images migrated: {stats.images_migrated}')
    print(f'  embedding_metadata migrated: {stats.embedding_metadata_migrated}')
    print(f'  embedding_chunks migrated: {stats.embedding_chunks_migrated}')
    print(f'  vec rows migrated: {stats.vec_rows_migrated}')
    print(f'  FTS rebuilt: {stats.fts_rebuilt}')
    if stats.warnings:
        print(f'  warnings: {len(stats.warnings)}')
        for message in stats.warnings:
            print(f'    - {message}')
    if stats.errors:
        print(f'  errors: {len(stats.errors)}')
        for message in stats.errors:
            print(f'    - {message}')
    if dry_run:
        print('Dry run: no changes were written to the target.')
    elif stats.rows_migrated > 0 and not stats.errors:
        print(
            'Next steps: point the server at the new target database '
            '(DB_PATH=... for SQLite or POSTGRESQL_CONNECTION_STRING=... for PostgreSQL).',
        )


def run_uuid_migration(
    source_url: str,
    target_url: str,
    *,
    dry_run: bool,
    report_path: Path | None,
    id_paths: tuple[IdPath, ...],
    legacy_id_key: str | None,
) -> int:
    """Run the integer-to-UUIDv7 migration and report its outcome.

    Dispatches on the backend pair of ``source_url`` and ``target_url``, prints
    the summary, and writes the JSON report when ``report_path`` is set.

    Args:
        source_url: Database URL passed to ``--source-url``.
        target_url: Database URL passed to ``--target-url``.
        dry_run: When True, run the full migration logic but issue no writes
            against the target.
        report_path: Optional path. When set, write the migration statistics
            as JSON to this file.
        id_paths: The metadata paths whose entry ids are rewritten, in
            command-line order.
        legacy_id_key: The top-level metadata key each migrated entry's old id
            is stamped under, or None to stamp nothing.

    Returns:
        Process exit code: 0 on success, 1 on an invalid URL, an unsupported
        backend combination or recorded errors, 2 on unrecoverable migration
        failure, 78 (EX_CONFIG) on a settings ValidationError.
        A connection row-level security filters, or a privilege error, is not a
        failure of this mode: it propagates to the dispatcher, which prints the
        owner remediation and exits 1.
    """
    options = MigrationOptions(
        source_url=source_url,
        target_url=target_url,
        dry_run=dry_run,
        report_path=report_path,
        id_paths=id_paths,
        legacy_id_key=legacy_id_key,
    )

    try:
        src_kind, _ = parse_backend_url(options.source_url)
        tgt_kind, _ = parse_backend_url(options.target_url)
    except ValueError as exc:
        logger.error('invalid database URL: %s', exc)
        return 1

    try:
        if src_kind == 'sqlite' and tgt_kind == 'sqlite':
            stats = run_migration_sqlite_to_sqlite(options)
        elif src_kind == 'postgresql' and tgt_kind == 'postgresql':
            stats = asyncio.run(run_migration_postgresql(options))
        elif src_kind == 'sqlite' and tgt_kind == 'postgresql':
            stats = asyncio.run(run_migration_mixed_sqlite_to_postgresql(options))
        elif src_kind == 'postgresql' and tgt_kind == 'sqlite':
            stats = asyncio.run(run_migration_mixed_postgresql_to_sqlite(options))
        else:
            logger.error('unsupported backend combination: %s -> %s', src_kind, tgt_kind)
            return 1
    except ValidationError as exc:
        # Same classification as the in-place dispatch in app.cli.migrate: a settings
        # ValidationError is a permanent misconfiguration (EX_CONFIG), not a
        # migration failure worth the generic exit 2.
        print(f'Configuration invalid: {exc}', file=sys.stderr)
        return ConfigurationError.EXIT_CODE
    except Exception as exc:
        from app.cli._backend import owner_refusal

        if owner_refusal(exc) is not None:
            raise
        logger.exception('migration failed: %s', exc)
        return 2

    print_summary(stats, options.source_url, options.target_url, options.dry_run)
    if options.report_path is not None:
        try:
            options.report_path.write_text(
                json.dumps(stats.to_dict(), indent=2),
                encoding='utf-8',
            )
        except OSError as exc:
            logger.error('failed to write report: %s', exc)
            stats.errors.append(f'failed to write report: {exc}')

    return 0 if not stats.errors else 1
