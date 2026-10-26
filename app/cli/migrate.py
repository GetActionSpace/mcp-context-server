"""Console-script dispatcher for ``mcp-context-server-migrate``.

Parses the command line and routes each invocation to its mode:

- the integer-to-UUIDv7 migration (``--source-url`` plus ``--target-url``):
  :mod:`app.cli.migrate_uuid`;
- ``--compress`` and ``--decompress``: :mod:`app.cli.migrate_compression`;
- ``--re-embed``: :mod:`app.cli.migrate_reembed`;
- ``--embed-missing``, standalone or after ``--compress``:
  :mod:`app.cli.migrate_embeddings`;
- ``--reassign-owner``: :mod:`app.cli.migrate_reassign_owner`;
- ``--disable-rls``, with ``--force``: :mod:`app.cli.migrate_disable_rls`.

Every PostgreSQL mode must connect with the table owner's credentials. A mode that finds
its role filtered by row-level security, or that the database refuses a statement with
insufficient privilege, ends here with the owner remediation and exit code 1.
"""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from app.cli.migrate_uuid.run import run_uuid_migration
from app.errors import ConfigurationError

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``mcp-context-server-migrate``.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(
        prog='mcp-context-server-migrate',
        description=(
            'Migrate an integer-keyed MCP context database to the UUIDv7 '
            'schema, compress/decompress an existing UUIDv7 database with '
            'TurboQuant embedding compression, re-embed an existing '
            'database under a new model, reassign the owner of existing '
            'context entries, or remove the PostgreSQL row-level security '
            "the server installed. PostgreSQL modes need the table owner's "
            'credentials in --source-url.'
        ),
    )
    parser.add_argument(
        '--source-url',
        required=True,
        help='Source database URL or filesystem path (sqlite:/// or postgresql://).',
    )
    parser.add_argument(
        '--target-url',
        required=False,
        default=None,
        help=(
            'Target database URL or filesystem path. Required for the v2->v3 '
            'migration; ignored by the in-place modes (--compress, '
            '--decompress, --re-embed, --embed-missing, --reassign-owner, '
            '--disable-rls).'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the full migration logic in memory but issue no writes against the target.',
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=None,
        metavar='PATH',
        help='Write a JSON migration report to PATH.',
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--compress',
        action='store_true',
        help=(
            'Compress an existing database with fp32 embeddings. Requires '
            'ENABLE_EMBEDDING_COMPRESSION=true in the environment. Reads '
            'from --source-url; --target-url is ignored. Use --dry-run to '
            'preview. May be combined with --embed-missing to also backfill '
            'entries lacking embeddings (compress runs first, then backfill).'
        ),
    )
    mode_group.add_argument(
        '--decompress',
        action='store_true',
        help=(
            'Decompress a database with compressed embeddings back to fp32 '
            '(lossy reconstruction). Requires ENABLE_EMBEDDING_COMPRESSION '
            'to be unset or false. Reads from --source-url; --target-url is '
            'ignored. Use --dry-run to preview. Not combinable with '
            '--embed-missing (a co-passed --embed-missing is ignored; run it '
            'separately after decompressing).'
        ),
    )
    mode_group.add_argument(
        '--re-embed',
        action='store_true',
        help=(
            'Re-embed EVERY context_entries row using the currently '
            'configured EMBEDDING_PROVIDER/EMBEDDING_MODEL, deleting existing '
            'embeddings first. The one-command path for switching the '
            'embedding MODEL on an existing database. Works for fp32 and '
            'compressed layouts. Requires ENABLE_EMBEDDING_GENERATION=true. '
            'Reads from --source-url; --target-url is ignored. First completes '
            'the generation jobs that writes made with defer_generation left '
            'outstanding, failed jobs included, calling the summary provider '
            'too when summary generation is enabled; an entry whose job '
            'remains (held by a running server, or failed again) is not '
            're-embedded. Use --dry-run to preview the entry and job counts '
            'without calling a provider. Refuses '
            'a dimension change (a different EMBEDDING_DIM than stored): a '
            'dimension change requires the documented rebuild. A co-passed '
            '--embed-missing is ignored because --re-embed already covers '
            'every entry.'
        ),
    )
    mode_group.add_argument(
        '--reassign-owner',
        nargs=2,
        metavar=('FROM', 'TO'),
        default=None,
        help=(
            'Reassign every context_entries row owned by FROM to TO, for '
            'example after switching to MCP_AUTH_PROVIDER=jwt when the legacy '
            'rows belong to ACCESS_CONTROL_DEFAULT_PRINCIPAL. Stop the server '
            'first. Grants are not changed. Reads from --source-url; '
            '--target-url is ignored. Use --dry-run to print the row count. '
            'Not combinable with --embed-missing.'
        ),
    )
    mode_group.add_argument(
        '--disable-rls',
        action='store_true',
        help=(
            'Remove the PostgreSQL row-level security this server installed '
            '(policies, guard trigger, read view and helper functions) and '
            'disable row-level security on the tables it enabled. Stop every '
            "server instance first and connect with the table owner's "
            'credentials; set ENABLE_POSTGRESQL_RLS=false before starting '
            'again. Refuses while a non-superuser member of the policy role '
            'is connected unless --force. Reads from --source-url; '
            '--target-url is ignored. Use --dry-run to print what would be '
            'removed. Not combinable with --embed-missing.'
        ),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help=(
            'With --disable-rls only: remove row-level security even while '
            'sessions of policy-role members are connected.'
        ),
    )
    # --embed-missing is intentionally OUTSIDE mode_group: it composes with
    # --compress (one-shot compress+backfill) AND runs standalone (fp32-only
    # backfill or compressed-only backfill, depending on the env var state).
    parser.add_argument(
        '--embed-missing',
        action='store_true',
        help=(
            'Generate embeddings for context_entries rows that lack an '
            'embedding_metadata row, calling the configured embedding '
            'provider (EMBEDDING_PROVIDER, EMBEDDING_MODEL). Works '
            'standalone (against the existing storage layout) or composed '
            'with --compress (compress first, then backfill into the '
            'compressed table). First completes the generation jobs that '
            'writes made with defer_generation left outstanding, failed '
            'jobs included, calling the summary provider too when summary '
            'generation is enabled; an entry whose job remains (held by a '
            'running server, or failed again) is not backfilled. Reads from '
            '--source-url; --target-url is ignored. Use --dry-run to preview '
            'the missing-entry and job counts without calling a provider.'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``mcp-context-server-migrate`` script.

    Args:
        argv: Optional override for ``sys.argv[1:]`` (used by tests).

    Returns:
        Process exit code: 0 on success, 1 on user error, recorded errors,
        or a connection the command may not run on (row-level security
        filters its role, or the database refuses a statement with
        insufficient privilege), 2 on unrecoverable migration failure, 78
        (EX_CONFIG) on a settings ValidationError -- the same classification
        the server's guarded import applies.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.force and not args.disable_rls:
        parser.error('--force is valid only with --disable-rls')

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Single-backend in-place operations: --reassign-owner, --disable-rls,
    # --compress, --decompress, --re-embed, --embed-missing. All dispatch on
    # --source-url alone; --target-url is ignored. Composition rule: --compress
    # and --embed-missing can be combined (--compress runs first, then
    # --embed-missing against the compressed layout). --reassign-owner,
    # --disable-rls, --compress, --decompress, and --re-embed are mutually
    # exclusive (enforced by argparse mode_group). --reassign-owner,
    # --disable-rls, --decompress and --re-embed return before the
    # --embed-missing check in _dispatch, so a co-passed --embed-missing is
    # silently superseded: --re-embed already re-embeds every entry (gaps
    # included), and --reassign-owner, --disable-rls and --decompress are
    # documented as not combinable with --embed-missing (run it separately
    # afterward). Imported lazily so callers running the v2->v3 migration do
    # not pay the compression/numpy import cost.
    #
    # Settings validation surfaces on these paths at the first get_settings()
    # call (transitively, at backend-module import), never through
    # app.server's guarded import: mcp-context-server-migrate is its own
    # console script and never imports app.server. Classify it here exactly
    # like the server does -- a permanent misconfiguration exits EX_CONFIG
    # (78) with the pydantic detail on stderr, instead of an unhandled
    # traceback and the generic exit 1 supervisors cannot distinguish from a
    # transient failure.
    #
    # A mode that finds its connection filtered by row-level security, or whose
    # statement the database refuses with insufficient privilege (42501), lets
    # that reach this point rather than reporting a generic failure: the
    # remediation, connecting with the table owner's credentials, is the same for
    # every mode.
    try:
        return _dispatch(args)
    except ValidationError as e:
        print(f'Configuration invalid: {e}', file=sys.stderr)
        return ConfigurationError.EXIT_CODE
    except Exception as exc:
        from app.cli._backend import owner_refusal

        refusal = owner_refusal(exc)
        if refusal is None:
            raise
        print(refusal, file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    """Run the mode ``args`` select and return its exit code.

    Args:
        args: The parsed command line.

    Returns:
        The mode's exit code.
    """
    if args.reassign_owner:
        from app.cli.migrate_reassign_owner import run_reassign_owner
        from_principal, to_principal = args.reassign_owner
        return run_reassign_owner(args.source_url, from_principal, to_principal, dry_run=args.dry_run)
    if args.disable_rls:
        from app.cli.migrate_disable_rls import run_disable_rls
        return run_disable_rls(args.source_url, dry_run=args.dry_run, force=args.force)
    if args.compress:
        from app.cli.migrate_compression.compress import run_compress
        rc = run_compress(args.source_url, dry_run=args.dry_run)
        if rc != 0:
            return rc
        if args.embed_missing:
            from app.cli.migrate_embeddings import run_embed_missing
            return run_embed_missing(args.source_url, dry_run=args.dry_run)
        return 0
    if args.decompress:
        from app.cli.migrate_compression.decompress import run_decompress
        return run_decompress(args.source_url, dry_run=args.dry_run)
    if args.re_embed:
        from app.cli.migrate_reembed import run_reembed
        return run_reembed(args.source_url, dry_run=args.dry_run)
    if args.embed_missing:
        from app.cli.migrate_embeddings import run_embed_missing
        return run_embed_missing(args.source_url, dry_run=args.dry_run)

    if not args.target_url:
        logger.error(
            '--target-url is required for the v2->v3 migration. '
            'For an in-place operation against --source-url, pass one of '
            '--compress, --decompress, --re-embed, --embed-missing, '
            '--reassign-owner, or --disable-rls (none of which use --target-url).',
        )
        return 1

    return run_uuid_migration(
        args.source_url,
        args.target_url,
        dry_run=args.dry_run,
        report_path=args.report,
    )


if __name__ == '__main__':
    sys.exit(main())
