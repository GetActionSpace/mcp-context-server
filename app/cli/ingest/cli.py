"""The mcp-context-ingest command: read the flags, run one sync of a corpus directory and report it.

The command prints its progress and the run's summary on stdout and every
run-level error on stderr, writes the JSON report ``--report`` names, and exits
with 0 when the thread matches the corpus (after ``--dry-run``, when every slice
was planned and every file and directory was usable), 1 when the run completed
with problems, 2 on a configuration, authentication or plan error, and 69 on a
transient failure. A bearer token is read from the file ``--token-file`` names or
from the environment variable ``--token-env`` names, never from the command line.
"""

import argparse
import asyncio
import io
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from app.cli.ingest.apply import DEFAULT_BATCH_SIZE
from app.cli.ingest.apply import DEFAULT_CALL_TIMEOUT
from app.cli.ingest.apply import DEFAULT_CONCURRENCY
from app.cli.ingest.apply import DEFAULT_INCLUDES
from app.cli.ingest.apply import DEFAULT_MAX_REQUEST_BYTES
from app.cli.ingest.apply import MAX_BATCH_SIZE
from app.cli.ingest.apply import MAX_CONCURRENCY
from app.cli.ingest.apply import MAX_REQUEST_BYTES
from app.cli.ingest.apply import THREAD_PREFIX
from app.cli.ingest.apply import RunConfig
from app.cli.ingest.apply import run_sync
from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import strip_userinfo
from app.cli.ingest.corpus import matches
from app.cli.ingest.report import render_summary
from app.cli.ingest.report import write_json
from app.ingest import INGEST_KEY
from app.models import MAX_THREAD_ID_LENGTH

PROG = 'mcp-context-ingest'
_CORPUS_NAME = re.compile(r'[a-z0-9][a-z0-9._-]*')
_HTTP_SCHEMES = frozenset({'http', 'https'})
_ENDPOINT_EXAMPLE = 'http://localhost:8000/mcp'


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the command's flags.

    Returns:
        A parser whose usage errors exit with code 2.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            'Synchronize a directory of documents into the knowledge-base thread kb:<corpus> of a context server, '
            'over MCP Streamable HTTP.'
        ),
        epilog=(
            'Exit codes: 0 the thread matches the corpus (with --dry-run: every slice planned and every file and '
            'directory usable), 1 completed with problems, 2 configuration, authentication or plan error, '
            '69 transient failure. '
            "See docs/knowledge-base.md, 'The mcp-context-ingest CLI'."
        ),
    )
    parser.add_argument(
        '--server', required=True, metavar='URL', type=_server_url,
        help=f"the server's Streamable HTTP endpoint, such as {_ENDPOINT_EXAMPLE}",
    )
    parser.add_argument(
        '--corpus', required=True, metavar='NAME', type=_corpus_name,
        help=f'the corpus; its thread is {THREAD_PREFIX}NAME (lowercase letters, digits, dots, underscores, hyphens)',
    )
    parser.add_argument(
        '--source-dir', required=True, metavar='PATH', type=_source_dir,
        help='the corpus root, walked recursively; symbolic links are not followed and .git is skipped',
    )
    parser.add_argument(
        '--include', action='append', metavar='GLOB', type=_glob,
        help=f'select files matching GLOB, case-sensitively, relative to the root (repeatable; default {DEFAULT_INCLUDES[0]})',
    )
    parser.add_argument(
        '--exclude', action='append', default=[], metavar='GLOB', type=_glob,
        help='leave out files matching GLOB (repeatable)',
    )
    parser.add_argument('--tag', action='append', default=[], metavar='TAG', help='a tag for every document (repeatable)')
    parser.add_argument(
        '--metadata-json', metavar='JSON', type=_metadata_object,
        help='a JSON object of metadata for every document; its title replaces the one taken from the file',
    )
    parser.add_argument(
        '--visibility', choices=('private', 'public'), default='private',
        help='the visibility sent with every create and update (default private)',
    )
    parser.add_argument(
        '--defer-generation', action='store_true',
        help=(
            'ask the server to store every create and update at once and generate its embeddings and summaries '
            'in the background; the summary counts the entries the server reports pending'
        ),
    )
    tokens = parser.add_mutually_exclusive_group()
    tokens.add_argument(
        '--token-file', metavar='PATH', type=Path,
        help='read the bearer token from PATH before every request, so a refresher can rotate it',
    )
    tokens.add_argument('--token-env', metavar='NAME', help='read the bearer token once from the environment variable NAME')
    parser.add_argument('--dry-run', action='store_true', help='plan and print the plan; write nothing')
    parser.add_argument(
        '--allow-empty-corpus', action='store_true',
        help='let a directory without documents delete every managed entry of the corpus',
    )
    parser.add_argument('--report', metavar='PATH', type=Path, help='write a JSON report of the run to PATH')
    parser.add_argument(
        '--batch-size', metavar='N', type=_whole_number(1, MAX_BATCH_SIZE), default=DEFAULT_BATCH_SIZE,
        help=f'the most entries per batch call, 1 to {MAX_BATCH_SIZE} (default {DEFAULT_BATCH_SIZE})',
    )
    parser.add_argument(
        '--max-request-bytes', metavar='N', type=_whole_number(1, MAX_REQUEST_BYTES), default=DEFAULT_MAX_REQUEST_BYTES,
        help=f'the request budget per call, at most {MAX_REQUEST_BYTES:,} (default {DEFAULT_MAX_REQUEST_BYTES:,})',
    )
    parser.add_argument(
        '--call-timeout', metavar='SECONDS', type=_seconds, default=DEFAULT_CALL_TIMEOUT,
        help=f'how long one call may take (default {DEFAULT_CALL_TIMEOUT:g})',
    )
    parser.add_argument(
        '--concurrency', metavar='N', type=_whole_number(1, MAX_CONCURRENCY), default=DEFAULT_CONCURRENCY,
        help=(
            f'the most create and update batch calls in flight at once, 1 to {MAX_CONCURRENCY} '
            f'(default {DEFAULT_CONCURRENCY}); deletes and plans run one at a time'
        ),
    )
    return parser


def parse_config(argv: list[str] | None = None) -> RunConfig:
    """Parse and validate the command's flags.

    A missing or invalid flag ends the process with a usage error and exit code 2,
    and ``--help`` ends it with exit code 0, both through ``argparse``.

    Args:
        argv: The arguments after the command name; None reads them from ``sys.argv``.

    Returns:
        The run's settings.
    """
    args = build_parser().parse_args(argv)
    return RunConfig(
        server=args.server,
        corpus=args.corpus,
        source_dir=args.source_dir,
        includes=tuple(args.include) if args.include else DEFAULT_INCLUDES,
        excludes=tuple(args.exclude),
        tags=tuple(args.tag),
        metadata=args.metadata_json if args.metadata_json is not None else {},
        visibility=args.visibility,
        defer_generation=args.defer_generation,
        token_file=args.token_file,
        token_env=args.token_env,
        dry_run=args.dry_run,
        allow_empty_corpus=args.allow_empty_corpus,
        report=args.report,
        batch_size=args.batch_size,
        max_request_bytes=args.max_request_bytes,
        call_timeout=args.call_timeout,
        concurrency=args.concurrency,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: The arguments after the command name; None reads them from ``sys.argv``.

    Returns:
        The exit code: 0, 1, 2 or 69.
    """
    _escape_unencodable_output()
    config = parse_config(argv)
    report = asyncio.run(run_sync(config))
    if config.report is not None:
        try:
            write_json(report, config.report)
        except OSError as error:
            report.fail(f'cannot write the report {config.report}: {error.strerror or error}', exit_code=EXIT_CONFIG)
    print(render_summary(report), flush=True)
    for message in report.errors:
        print(f'error: {message}', file=sys.stderr, flush=True)
    return report.exit_code


def _escape_unencodable_output() -> None:
    """Print characters the output encoding cannot represent as escapes instead of failing on them.

    A redirected stream on Windows encodes with the ANSI code page, which cannot
    represent most non-ASCII paths the run prints.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors='backslashreplace')


def _server_url(value: str) -> str:
    """Accept an http or https URL with a host; a refused value is named without its userinfo."""
    try:
        parts = urlsplit(value)
        host = parts.hostname
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a valid URL; use an endpoint such as {_ENDPOINT_EXAMPLE}') from None
    if parts.scheme not in _HTTP_SCHEMES or not host:
        raise argparse.ArgumentTypeError(
            f'{strip_userinfo(value)!r} is not an http or https URL with a host, such as {_ENDPOINT_EXAMPLE}',
        )
    return value


def _corpus_name(value: str) -> str:
    """Accept a corpus name whose thread id the server can store."""
    if _CORPUS_NAME.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            f'{value!r} is not a corpus name: use lowercase letters, digits, dots, underscores and hyphens, '
            'starting with a letter or digit',
        )
    if len(THREAD_PREFIX + value) > MAX_THREAD_ID_LENGTH:
        raise argparse.ArgumentTypeError(
            f'the corpus name is {len(value)} characters long; {THREAD_PREFIX}NAME must stay within '
            f'{MAX_THREAD_ID_LENGTH} characters',
        )
    return value


def _source_dir(value: str) -> Path:
    """Accept an existing directory, so a mistyped path never reads as an empty corpus."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'{value} is not a directory')
    return path


def _glob(value: str) -> str:
    """Accept a glob pattern the corpus matcher can compile."""
    try:
        matches(value, '')
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _metadata_object(value: str) -> dict[str, object]:
    """Accept a JSON object of plain JSON values that leaves the ingest block to the sync."""
    try:
        parsed: object = json.loads(value, parse_constant=_refuse_constant)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f'not valid JSON: {error}') from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError('must be a JSON object')
    metadata = cast('dict[str, object]', parsed)
    if INGEST_KEY in metadata:
        raise argparse.ArgumentTypeError(f'must not set {INGEST_KEY!r}: the sync writes the ingest block itself')
    return metadata


def _refuse_constant(constant: str) -> object:
    """Refuse the NaN and infinity literals the JSON decoder accepts, which no JSON document may carry."""
    raise argparse.ArgumentTypeError(f'{constant} is not a JSON value')


def _whole_number(low: int, high: int) -> Callable[[str], int]:
    """Return a parser of whole numbers from low to high."""

    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{value!r} is not a whole number') from None
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f'{number:,} is outside {low:,} to {high:,}')
        return number

    return parse


def _seconds(value: str) -> float:
    """Accept a positive, finite number of seconds."""
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not a number of seconds') from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(f'{value!r} is not a positive number of seconds')
    return seconds


if __name__ == '__main__':
    sys.exit(main())
