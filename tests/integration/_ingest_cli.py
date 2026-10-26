"""End-to-end scenarios of the mcp-context-ingest command against live HTTP servers.

The SQLite and PostgreSQL entry points (``tests/integration/sqlite/test_ingest_cli.py``
and ``tests/integration/postgresql/test_ingest_cli.py``) start each server with
:func:`tests.integration._http_jwt.running_http_server` from an environment that
``build_env`` or ``build_jwt_env`` builds on the MCP SDK's ``get_default_environment()``,
and hand the scenario the server's origin. A scenario writes a corpus directory, runs
the command against the server and reads the result back through the sync tools as
the principal the command ran as. Every server runs with generation off except the
deferred-generation scenario's, which ``stub_generation_env`` points at an embedding
endpoint the test process serves (:mod:`tests.integration._ollama_stub`).

The command runs as a child process of the test's own interpreter, as
``python -m app.cli.ingest.cli`` or, in one run, as the installed console script; never
through ``uv run``, whose children outlive their parent on Windows. Its environment is
``get_default_environment()`` plus the one variable ``--token-env`` names, so nothing
from the developer's shell reaches it. Every run writes its JSON report, which the
scenarios read for the exit code, the counts and each document's outcome.

The token-rotation scenario runs the sync in process instead: it paces the exchanges
through a substituted response recorder, so a token that expires after the first
exchange is refused exactly once and replaced before the retry.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any
from typing import cast
from typing import override

import httpx2
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from mcp.client.stdio import get_default_environment

from app.cli.ingest import client as ingest_client
from app.cli.ingest.apply import MAX_REQUEST_BYTES
from app.cli.ingest.apply import RunConfig
from app.cli.ingest.apply import run_sync
from app.cli.ingest.client import StatusRecorder
from app.cli.ingest.report import COUNT_KEYS
from app.digests import content_hash
from app.digests import normalize_stored_text
from app.ingest import MAX_INGEST_STATE_PAGE
from tests.integration._http_jwt import ALICE
from tests.integration._http_jwt import BOB
from tests.integration._http_jwt import JWT_AUDIENCE
from tests.integration._http_jwt import JWT_ISSUER
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import jwt_client
from tests.integration._http_jwt import mint_token
from tests.integration._ollama_stub import OllamaStub

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_COMMAND = (sys.executable, '-m', 'app.cli.ingest.cli')
# The environment's scripts directory is the interpreter's directory on every platform.
CONSOLE_SCRIPT = Path(sys.executable).with_name('mcp-context-ingest' + ('.exe' if os.name == 'nt' else ''))
TOKEN_VARIABLE = 'INGEST_TOKEN'
CLI_TIMEOUT_S = 300
# The server environment overrides of a server that does not offer corpus sync.
SYNC_DISABLED = {'ENABLE_CONTEXT_SYNC': 'false'}
# The length of the vectors the stub embedding endpoint returns, a multiple of 64.
STUB_EMBEDDING_DIM = 64
# How long a scenario waits for the server's worker to finish the generation a run deferred.
GENERATION_TIMEOUT_S = 60
GENERATION_POLL_S = 0.2
# Long enough that the run's first exchange lands well before the token expires, even
# on a loaded machine; the rotating recorder waits the rest of it out.
ROTATION_TOKEN_LIFETIME_S = 5
EXPIRY_MARGIN_S = 0.5
IDENTICAL_TEXT = '# Shared page\n\nThe same bytes stored under two names.\n'

# A file or directory mode denies access only on a POSIX host and only to a user
# other than root.
if sys.platform == 'win32':
    MODES_DENY_ACCESS = False
else:
    MODES_DENY_ACCESS = os.geteuid() != 0
requires_denying_modes = pytest.mark.skipif(
    not MODES_DENY_ACCESS,
    reason='file modes deny access only on a POSIX host to a user other than root',
)


@dataclass(frozen=True)
class CliRun:
    """One finished run of the command.

    Attributes:
        exit_code: The command's exit code.
        stdout: What it printed on standard output.
        stderr: What it printed on standard error.
        report: The JSON report it wrote.
    """

    exit_code: int
    stdout: str
    stderr: str
    report: dict[str, Any]

    def expect(self, exit_code: int, **counts: int) -> None:
        """Assert the exit code and the report's counts; a count not named must be zero.

        Args:
            exit_code: The expected exit code.
            counts: The expected non-zero counts by name.
        """
        output = f'stdout:\n{self.stdout}\nstderr:\n{self.stderr}'
        expected: dict[str, int] = {**dict.fromkeys(COUNT_KEYS, 0), **counts}
        assert self.exit_code == exit_code, f'exit {self.exit_code}, expected {exit_code}\n{output}'
        assert self.report['exit_code'] == exit_code, output
        assert self.report['counts'] == expected, output

    def outcomes(self, outcome: str) -> list[str]:
        """Return the source ids the report records with one outcome.

        Args:
            outcome: An outcome such as ``kept`` or ``deleted``.

        Returns:
            The source ids in order.
        """
        documents: list[dict[str, Any]] = self.report['documents']
        return sorted(str(document['source_id']) for document in documents if document['outcome'] == outcome)


def run_cli(
    base_url: str,
    corpus: str,
    source_dir: Path,
    *flags: str,
    work: Path,
    token: str | None = None,
    console_script: bool = False,
) -> CliRun:
    """Run the command once against a server and read its report.

    Args:
        base_url: The server origin; the command talks to its ``/mcp`` endpoint.
        corpus: The corpus name.
        source_dir: The corpus directory.
        flags: Further command-line flags.
        work: The test's directory; reports are written under it, outside the corpus.
        token: A bearer token passed through ``--token-env``, or None for no token.
        console_script: Run the installed console script instead of the module.

    Returns:
        The finished run.
    """
    reports = work / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / f'{uuid.uuid4().hex}.json'
    command = [str(CONSOLE_SCRIPT)] if console_script else list(MODULE_COMMAND)
    command += ['--server', f'{base_url}/mcp', '--corpus', corpus, '--source-dir', str(source_dir), '--report', str(report)]
    command += flags
    env = get_default_environment()
    if token is not None:
        env[TOKEN_VARIABLE] = token
        command += ['--token-env', TOKEN_VARIABLE]
    completed = subprocess.run(
        command,
        env=env,
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=CLI_TIMEOUT_S,
        check=False,
    )
    stdout = completed.stdout.decode('utf-8', errors='replace')
    stderr = completed.stderr.decode('utf-8', errors='replace')
    assert report.is_file(), f'the command wrote no report (exit {completed.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}'
    return CliRun(completed.returncode, stdout, stderr, json.loads(report.read_text(encoding='utf-8')))


async def ingest(
    base_url: str,
    corpus: str,
    source_dir: Path,
    *flags: str,
    work: Path,
    token: str | None = None,
    console_script: bool = False,
) -> CliRun:
    """Run the command in a worker thread, so the event loop stays free while the child runs.

    Args:
        base_url: The server origin.
        corpus: The corpus name.
        source_dir: The corpus directory.
        flags: Further command-line flags.
        work: The test's directory.
        token: A bearer token passed through ``--token-env``, or None.
        console_script: Run the installed console script instead of the module.

    Returns:
        The finished run.
    """
    return await asyncio.to_thread(
        run_cli, base_url, corpus, source_dir, *flags, work=work, token=token, console_script=console_script,
    )


def write_files(root: Path, files: Mapping[str, str | bytes]) -> Path:
    """Write files under a corpus root, creating their directories.

    Args:
        root: The corpus root.
        files: File contents by root-relative POSIX path; text is written as UTF-8 with LF.

    Returns:
        The root.
    """
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding='utf-8', newline='\n')
    return root


def remove_file(root: Path, relative: str) -> None:
    """Remove one file of a corpus.

    Args:
        root: The corpus root.
        relative: The file's root-relative POSIX path.
    """
    (root / relative).unlink()


def make_directory(path: Path) -> Path:
    """Create a directory and its parents.

    Args:
        path: The directory.

    Returns:
        The directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_mode(path: Path, mode: int) -> None:
    """Set a file's or directory's permission bits.

    Args:
        path: The file or directory.
        mode: The permission bits.
    """
    path.chmod(mode)


def read_text(path: Path) -> str:
    """Return a file's UTF-8 text.

    Args:
        path: The file.

    Returns:
        Its text.
    """
    return path.read_text(encoding='utf-8')


def page(title: str, size: int) -> str:
    """Return a Markdown document of about ``size`` characters.

    Args:
        title: The level-one heading, which also makes the text unique.
        size: The approximate length of the text.

    Returns:
        The document.
    """
    filler = f'{title} keeps the corpus line {{}} of the page.\n'
    lines = [f'# {title}', '']
    while sum(len(line) + 1 for line in lines) < size:
        lines.append(filler.format(len(lines)).rstrip('\n'))
    return '\n'.join(lines) + '\n'


def stored_digest(text: str) -> str:
    """Return the text digest the server records for a document's text.

    Args:
        text: The file's text.

    Returns:
        The digest of the stored form of the text.
    """
    return content_hash(normalize_stored_text(text))


def open_client(base_url: str) -> Client[Any]:
    """Build an unconnected client of a server that requires no authentication.

    Args:
        base_url: The server origin.

    Returns:
        The client; requests run as the server's default principal.
    """
    return Client(StreamableHttpTransport(url=f'{base_url}/mcp'))


async def structured(client: Client[Any], tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool that must succeed and return its structured content.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        The structured content.
    """
    result = await client.call_tool(tool, arguments, raise_on_error=False)
    assert not result.is_error, f'{tool} failed: {result.content}'
    assert result.structured_content is not None, f'{tool} returned no structured content'
    return result.structured_content


async def managed_state(client: Client[Any], thread_id: str) -> list[dict[str, Any]]:
    """Return every managed entry of a corpus thread the calling principal owns.

    Args:
        client: The principal's client.
        thread_id: The corpus thread.

    Returns:
        The ``list_ingest_state`` entries of every page, in source id order.
    """
    entries: list[dict[str, Any]] = []
    arguments: dict[str, Any] = {'thread_id': thread_id, 'limit': MAX_INGEST_STATE_PAGE}
    while True:
        listed = await structured(client, 'list_ingest_state', arguments)
        entries.extend(listed['entries'])
        if listed['next_cursor'] is None:
            return entries
        arguments['cursor'] = listed['next_cursor']


async def readable_ids(client: Client[Any], context_ids: list[str]) -> list[str]:
    """Return which of some entries the calling principal can read.

    Args:
        client: The principal's client.
        context_ids: The entries to read.

    Returns:
        The ids of the entries it read, sorted.
    """
    rows = (await structured(client, 'get_context_by_ids', {'context_ids': context_ids}))['result']
    return sorted(str(row['id']) for row in rows)


def by_source(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Map each managed entry's source id to its context id.

    Args:
        entries: ``list_ingest_state`` entries with distinct source ids.

    Returns:
        The context ids by source id.
    """
    mapping = {str(entry['source_id']): str(entry['context_id']) for entry in entries}
    assert len(mapping) == len(entries), f'a source id is stored twice: {entries}'
    return mapping


def entry_of(entries: list[dict[str, Any]], source_id: str) -> dict[str, Any]:
    """Return the managed entry of one source id.

    Args:
        entries: ``list_ingest_state`` entries.
        source_id: The source id.

    Returns:
        Its only entry.
    """
    (entry,) = (entry for entry in entries if entry['source_id'] == source_id)
    return entry


async def verify_corpus_lifecycle(base_url: str, work: Path) -> None:
    """Bootstrap a corpus, rerun it, apply edits, a removal, a tag and a duplicate, then empty it.

    Args:
        base_url: Origin of a server that requires no authentication.
        work: The test's directory.
    """
    corpus, thread = make_directory(work / 'corpus'), 'kb:lifecycle'
    write_files(corpus, {
        'guide.md': '# Guide\n\nHow the corpus is organized.\n',
        'notes/setup.md': '# Setup\n\nInstall the tools first.\n',
        'copy-1.md': IDENTICAL_TEXT,
        'copy-2.md': IDENTICAL_TEXT,
    })

    (await ingest(base_url, 'lifecycle', corpus, work=work, console_script=True)).expect(0, created=4)
    async with open_client(base_url) as client:
        stored = await managed_state(client, thread)
    ids = by_source(stored)
    assert sorted(ids) == ['copy-1.md', 'copy-2.md', 'guide.md', 'notes/setup.md']
    assert ids['copy-1.md'] != ids['copy-2.md'], 'byte-identical files must stay two entries'

    (await ingest(base_url, 'lifecycle', corpus, work=work)).expect(0, unchanged=4)
    async with open_client(base_url) as client:
        assert await managed_state(client, thread) == stored, 'a rerun of an unchanged corpus wrote'

    edited = '# Guide\n\nHow the corpus is organized, edited.\n'
    write_files(corpus, {'guide.md': edited, 'new.md': '# New\n\nA page added later.\n'})
    remove_file(corpus, 'notes/setup.md')
    dry = await ingest(base_url, 'lifecycle', corpus, '--dry-run', work=work)
    dry.expect(0, created=1, updated=1, deleted=1, unchanged=2)
    assert dry.report['dry_run'] is True
    assert 'would create new.md' in dry.stdout
    assert f'would update guide.md {ids["guide.md"]} (text)' in dry.stdout
    assert f'would delete notes/setup.md {ids["notes/setup.md"]} (absent)' in dry.stdout
    async with open_client(base_url) as client:
        assert await managed_state(client, thread) == stored, 'a dry run wrote'

    (await ingest(base_url, 'lifecycle', corpus, work=work)).expect(0, created=1, updated=1, deleted=1, unchanged=2)
    async with open_client(base_url) as client:
        changed = await managed_state(client, thread)
    ids = by_source(changed)
    assert sorted(ids) == ['copy-1.md', 'copy-2.md', 'guide.md', 'new.md']
    assert ids['guide.md'] == by_source(stored)['guide.md'], 'a text edit must keep the entry'
    assert entry_of(changed, 'guide.md')['text_digest'] == stored_digest(edited)

    tagged = await ingest(base_url, 'lifecycle', corpus, '--tag', 'Guide', '--dry-run', work=work)
    tagged.expect(0, updated=4)
    for source_id, context_id in ids.items():
        assert f'would update {source_id} {context_id} (metadata)' in tagged.stdout, tagged.stdout
    (await ingest(base_url, 'lifecycle', corpus, '--tag', 'Guide', work=work)).expect(0, updated=4)
    async with open_client(base_url) as client:
        retagged = await managed_state(client, thread)
        rows = (await structured(client, 'get_context_by_ids', {'context_ids': list(ids.values())}))['result']
    assert by_source(retagged) == ids
    assert [entry['text_digest'] for entry in retagged] == [entry['text_digest'] for entry in changed]
    assert all(new['meta_digest'] != old['meta_digest'] for new, old in zip(retagged, changed, strict=True))
    assert [row['tags'] for row in rows] == [['guide']] * 4

    async with open_client(base_url) as client:
        (original,) = (await structured(client, 'get_context_by_ids', {'context_ids': [ids['guide.md']]}))['result']
        seeded = await structured(client, 'store_context', {
            'thread_id': thread,
            'source': 'user',
            'text': original['text_content'],
            'metadata': original['metadata'],
            'tags': original['tags'],
            'deduplicate': False,
        })
    deduplicated = await ingest(base_url, 'lifecycle', corpus, '--tag', 'Guide', work=work)
    deduplicated.expect(0, deleted=1, unchanged=4)
    assert deduplicated.outcomes('deleted') == ['guide.md']
    async with open_client(base_url) as client:
        survivors = by_source(await managed_state(client, thread))
    assert survivors == {**ids, 'guide.md': str(seeded['context_id'])}, 'the newest exact copy must survive'

    empty = make_directory(work / 'empty')
    refused = await ingest(base_url, 'lifecycle', empty, work=work)
    refused.expect(2)
    assert '--allow-empty-corpus' in refused.stderr
    async with open_client(base_url) as client:
        assert by_source(await managed_state(client, thread)) == survivors, 'a refused empty corpus deleted'
    (await ingest(base_url, 'lifecycle', empty, '--allow-empty-corpus', work=work)).expect(0, deleted=4)
    async with open_client(base_url) as client:
        assert await managed_state(client, thread) == []


async def verify_unusable_files_are_kept(base_url: str, work: Path) -> None:
    """Keep the stored entry of a file that is oversized, undecodable or unreadable, and store the rest.

    Args:
        base_url: Origin of a server that requires no authentication.
        work: The test's directory.
    """
    corpus, thread, budget = make_directory(work / 'corpus'), 'kb:kept', ['--max-request-bytes', '20000']
    small = {'alpha.md': page('Alpha', 400), 'beta.md': page('Beta', 400)}
    if MODES_DENY_ACCESS:
        small['locked.md'] = page('Locked', 400)
    write_files(corpus, {**small, 'huge.md': page('Huge', 40_000)})

    first = await ingest(base_url, 'kept', corpus, *budget, work=work)
    first.expect(1, created=len(small), kept=1)
    assert first.outcomes('kept') == ['huge.md']
    assert '--max-request-bytes' in first.stdout
    async with open_client(base_url) as client:
        stored = await managed_state(client, thread)
    assert sorted(by_source(stored)) == sorted(small), 'an oversized file on its first upload must not be stored'

    write_files(corpus, {'alpha.md': page('Alpha', 40_000), 'beta.md': b'# Beta\n\n\xff\xfe not UTF-8\n'})
    if MODES_DENY_ACCESS:
        set_mode(corpus / 'locked.md', 0)
    try:
        second = await ingest(base_url, 'kept', corpus, *budget, work=work)
    finally:
        if MODES_DENY_ACCESS:
            set_mode(corpus / 'locked.md', 0o600)
    second.expect(1, kept=len(small) + 1)
    assert second.outcomes('kept') == sorted([*small, 'huge.md'])
    async with open_client(base_url) as client:
        assert await managed_state(client, thread) == stored, 'a kept file must leave its stored entry intact'


async def verify_corpus_above_the_request_limit(base_url: str, work: Path) -> None:
    """Bootstrap a corpus larger than one HTTP request may carry with two batches in flight, then rewrite it.

    Args:
        base_url: Origin of a server that requires no authentication.
        work: The test's directory.
    """
    corpus, thread = make_directory(work / 'corpus'), 'kb:packed'
    files = {f'large-{number}.md': page(f'Large {number}', 800_000) for number in range(6)}
    assert sum(len(text.encode('utf-8')) for text in files.values()) > MAX_REQUEST_BYTES
    write_files(corpus, files)

    bootstrap = await ingest(base_url, 'packed', corpus, '--concurrency', '2', work=work)
    bootstrap.expect(0, created=6)
    assert '2 of 2 batches done' in bootstrap.stdout, bootstrap.stdout
    assert bootstrap.outcomes('created') == sorted(files)
    async with open_client(base_url) as client:
        ids = by_source(await managed_state(client, thread))
    assert sorted(ids) == sorted(files)

    edited = {name: text + 'Every large page changes at once.\n' for name, text in files.items()}
    write_files(corpus, edited)
    (await ingest(base_url, 'packed', corpus, work=work)).expect(0, updated=6)
    async with open_client(base_url) as client:
        rewritten = await managed_state(client, thread)
    assert by_source(rewritten) == ids
    assert {entry['source_id']: entry['text_digest'] for entry in rewritten} == {
        name: stored_digest(text) for name, text in edited.items()
    }


async def verify_sync_disabled_server(base_url: str, work: Path) -> None:
    """Stop at preflight with exit 2 on a server that does not offer corpus sync.

    Args:
        base_url: Origin of a server started with ``SYNC_DISABLED``.
        work: The test's directory.
    """
    corpus = write_files(make_directory(work / 'corpus'), {'alpha.md': page('Alpha', 200)})
    run = await ingest(base_url, 'disabled', corpus, work=work)
    run.expect(2)
    assert 'plan_context_sync' in run.stderr
    assert 'ENABLE_CONTEXT_SYNC=false' in run.stderr
    async with open_client(base_url) as client:
        assert (await structured(client, 'search_context', {'thread_id': 'kb:disabled'}))['count'] == 0


async def generation_states(client: Client[Any], context_ids: list[str]) -> dict[str, str]:
    """Return the generation state of some entries the calling principal can read.

    Args:
        client: The principal's client.
        context_ids: The entries.

    Returns:
        Each entry's ``generation_status`` by context id.
    """
    rows = (await structured(client, 'get_context_by_ids', {'context_ids': context_ids}))['result']
    return {str(row['id']): str(row['generation_status']) for row in rows}


async def await_generated(base_url: str, context_ids: list[str]) -> None:
    """Wait until the server's worker has finished the generation of every entry.

    Args:
        base_url: Origin of a server that requires no authentication.
        context_ids: The entries whose generation a run deferred.
    """
    deadline = time.monotonic() + GENERATION_TIMEOUT_S
    async with open_client(base_url) as client:
        while (states := await generation_states(client, context_ids)) != dict.fromkeys(context_ids, 'ready'):
            assert time.monotonic() < deadline, f'generation did not finish within {GENERATION_TIMEOUT_S} s: {states}'
            await asyncio.sleep(GENERATION_POLL_S)


async def verify_deferred_generation(base_url: str, embedder: OllamaStub, work: Path) -> None:
    """Store and update a corpus with deferred generation, converge at once, and let the worker generate.

    A deferring run stores every document before its embeddings exist: it converges with
    exit 0 and counts the pending entries, the server's worker then embeds them through
    the stub endpoint and reports them ready, and a text edit goes the same way. A run
    without the flag that only changes metadata leaves the ready entries ready.

    Args:
        base_url: Origin of a server that requires no authentication, started with
            ``stub_generation_env`` over ``embedder``.
        embedder: The stub endpoint the server embeds through.
        work: The test's directory.
    """
    corpus, thread = make_directory(work / 'corpus'), 'kb:deferred'
    write_files(corpus, {'one.md': page('Deferred one', 600), 'two.md': page('Deferred two', 600)})
    embedded_before = len(embedder.inputs)

    bootstrap = await ingest(base_url, 'deferred', corpus, '--defer-generation', work=work)
    bootstrap.expect(0, created=2, generation_pending=2)
    assert 'generation pending 2' in bootstrap.stdout, bootstrap.stdout
    async with open_client(base_url) as client:
        ids = by_source(await managed_state(client, thread))
    await await_generated(base_url, sorted(ids.values()))
    embedded = embedder.inputs[embedded_before:]
    for title in ('# Deferred one', '# Deferred two'):
        assert any(text.startswith(title) for text in embedded), f'the worker never embedded {title!r}: {embedded}'

    write_files(corpus, {'one.md': page('Deferred one, edited', 600)})
    edited = await ingest(base_url, 'deferred', corpus, '--defer-generation', work=work)
    edited.expect(0, updated=1, unchanged=1, generation_pending=1)
    await await_generated(base_url, sorted(ids.values()))
    assert any(text.startswith('# Deferred one, edited') for text in embedder.inputs), 'the edited text was never embedded'

    (await ingest(base_url, 'deferred', corpus, '--tag', 'guide', work=work)).expect(0, updated=2)
    (await ingest(base_url, 'deferred', corpus, '--tag', 'guide', '--defer-generation', work=work)).expect(0, unchanged=2)
    async with open_client(base_url) as client:
        assert by_source(await managed_state(client, thread)) == ids, 'a deferred update must keep the entry'


async def verify_unreachable_server(work: Path) -> None:
    """Exit 69 when nothing listens at the server URL.

    Args:
        work: The test's directory.
    """
    corpus = write_files(make_directory(work / 'corpus'), {'alpha.md': page('Alpha', 200)})
    run = await ingest(f'http://127.0.0.1:{free_port()}', 'unreachable', corpus, work=work)
    run.expect(69)
    assert 'cannot reach' in run.stderr


async def verify_unlistable_directory(base_url: str, work: Path) -> None:
    """Withhold every delete while a directory cannot be listed, and converge once it can.

    Args:
        base_url: Origin of a server that requires no authentication.
        work: The test's directory.
    """
    corpus, thread = make_directory(work / 'corpus'), 'kb:unlistable'
    write_files(corpus, {'open/a.md': page('Open', 200), 'locked/b.md': page('Bee', 200), 'locked/c.md': page('Sea', 200)})
    (await ingest(base_url, 'unlistable', corpus, work=work)).expect(0, created=3)
    async with open_client(base_url) as client:
        ids = by_source(await managed_state(client, thread))

    write_files(corpus, {'open/a.md': page('Open, edited', 200)})
    set_mode(corpus / 'locked', 0)
    try:
        run = await ingest(base_url, 'unlistable', corpus, work=work)
    finally:
        set_mode(corpus / 'locked', 0o700)
    run.expect(1, updated=1, withheld=2)
    assert 'directory not listed: locked' in run.stderr
    async with open_client(base_url) as client:
        assert by_source(await managed_state(client, thread)) == ids, 'an unlistable directory must delete nothing'

    (await ingest(base_url, 'unlistable', corpus, work=work)).expect(0, unchanged=3)


async def verify_principals_manage_their_own_copy(base_url: str, key_pair: RSAKeyPair, work: Path) -> None:
    """Refuse missing and forged tokens; a second principal copies a public corpus without touching it.

    alice publishes the corpus; bob, who reads it, syncs the same directory, is warned about
    alice's entries, stores his own copy and deletes nothing of hers. alice's rerun without
    ``--visibility`` makes her entries private again, after which bob cannot read them.

    Args:
        base_url: Origin of a server started with ``build_jwt_env``.
        key_pair: The key pair the server verifies tokens against.
        work: The test's directory.
    """
    corpus, thread = make_directory(work / 'corpus'), 'kb:shared'
    write_files(corpus, {'one.md': page('One', 300), 'two.md': page('Two', 300)})
    alice_token, bob_token = mint_token(key_pair, ALICE), mint_token(key_pair, BOB)

    anonymous = await ingest(base_url, 'shared', corpus, work=work)
    anonymous.expect(2)
    assert 'requires a bearer token' in anonymous.stderr
    forged_token = RSAKeyPair.generate().create_token(subject=ALICE, issuer=JWT_ISSUER, audience=JWT_AUDIENCE)
    forged = await ingest(base_url, 'shared', corpus, work=work, token=forged_token)
    forged.expect(2)
    assert 'refused the bearer token' in forged.stderr

    (await ingest(base_url, 'shared', corpus, '--visibility', 'public', work=work, token=alice_token)).expect(0, created=2)
    async with jwt_client(base_url, alice_token, 'auto') as alice, jwt_client(base_url, bob_token, 'auto') as bob:
        published = await managed_state(alice, thread)
        alice_ids = sorted(by_source(published).values())
        assert await readable_ids(bob, alice_ids) == alice_ids, 'bob cannot read the entries alice published'

    copied = await ingest(base_url, 'shared', corpus, work=work, token=bob_token)
    copied.expect(0, created=2, foreign=2)
    assert 'another principal owns' in copied.stderr
    async with jwt_client(base_url, alice_token, 'auto') as alice, jwt_client(base_url, bob_token, 'auto') as bob:
        assert await managed_state(alice, thread) == published, "bob's sync changed alice's entries"
        bob_ids = by_source(await managed_state(bob, thread))
    assert sorted(bob_ids) == ['one.md', 'two.md']
    assert set(bob_ids.values()).isdisjoint(alice_ids)

    (await ingest(base_url, 'shared', corpus, work=work, token=alice_token)).expect(0, updated=2)
    async with jwt_client(base_url, alice_token, 'auto') as alice, jwt_client(base_url, bob_token, 'auto') as bob:
        assert by_source(await managed_state(alice, thread)) == by_source(published)
        assert await readable_ids(bob, alice_ids) == [], 'bob still reads the entries alice made private'
        assert by_source(await managed_state(bob, thread)) == bob_ids


def token_expiry(token: str) -> float:
    """Return the ``exp`` claim of a JWT, read without verifying it.

    Args:
        token: The encoded token.

    Returns:
        The expiry as a POSIX timestamp.
    """
    payload = token.split('.')[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
    return float(claims['exp'])


def rotating_recorder(token_file: Path, fresh_token: str, *, expires_at: float, statuses: list[int]) -> type[StatusRecorder]:
    """Return a response recorder that lets the file's token expire and then rotates it.

    After the first successful response the recorder waits until the token has
    expired, so the next request is refused; on that first refusal it writes the
    fresh token to the file, which the client's single retry then sends.

    Args:
        token_file: The file the run reads its token from.
        fresh_token: The valid token written on the first 401.
        expires_at: When the file's first token expires.
        statuses: Receives the status of every response, in order.

    Returns:
        The recorder class to substitute for the client's own.
    """

    class RotatingRecorder(StatusRecorder):
        @override
        async def on_response(self, response: httpx2.Response) -> None:
            await super().on_response(response)
            statuses.append(response.status_code)
            if response.status_code == HTTPStatus.OK and statuses.count(HTTPStatus.OK) == 1:
                await asyncio.sleep(max(0.0, expires_at - time.time()) + EXPIRY_MARGIN_S)
            elif response.status_code == HTTPStatus.UNAUTHORIZED and statuses.count(HTTPStatus.UNAUTHORIZED) == 1:
                write_files(token_file.parent, {token_file.name: fresh_token})

    return RotatingRecorder


async def verify_token_rotation(base_url: str, key_pair: RSAKeyPair, work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry the one request refused after the token file's token expired, with the token the file holds now.

    Args:
        base_url: Origin of a server started with ``build_jwt_env``.
        key_pair: The key pair the server verifies tokens against.
        work: The test's directory.
        monkeypatch: Substitutes the client's response recorder.
    """
    corpus = write_files(make_directory(work / 'corpus'), {'r1.md': page('Rotation 1', 200), 'r2.md': page('Rotation 2', 200)})
    expiring = key_pair.create_token(
        subject=ALICE, issuer=JWT_ISSUER, audience=JWT_AUDIENCE, expires_in_seconds=ROTATION_TOKEN_LIFETIME_S,
    )
    fresh = mint_token(key_pair, ALICE)
    token_file = write_files(make_directory(work / 'secrets'), {'token': expiring}) / 'token'
    statuses: list[int] = []
    recorder = rotating_recorder(token_file, fresh, expires_at=token_expiry(expiring), statuses=statuses)
    monkeypatch.setattr(ingest_client, 'StatusRecorder', recorder)

    report = await run_sync(RunConfig(server=f'{base_url}/mcp', corpus='rotation', source_dir=corpus, token_file=token_file))

    assert report.exit_code == 0, report.errors
    assert cast('dict[str, int]', report.counts)['created'] == 2
    assert statuses.count(HTTPStatus.UNAUTHORIZED) == 1, statuses
    refused = statuses.index(HTTPStatus.UNAUTHORIZED)
    assert HTTPStatus.OK in statuses[:refused], f'no exchange succeeded with the first token: {statuses}'
    assert HTTPStatus.OK <= statuses[refused + 1] < HTTPStatus.MULTIPLE_CHOICES, f'the retry failed: {statuses}'
    assert read_text(token_file) == fresh
    async with jwt_client(base_url, fresh, 'auto') as alice:
        assert sorted(by_source(await managed_state(alice, 'kb:rotation'))) == ['r1.md', 'r2.md']
