"""Helpers for running a sync of a corpus directory against the in-memory server.

The sync-run tests write corpus files into a temporary directory, run the command's
legs against ``FakeServer`` and read the report and the calls the server received.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict
from typing import Unpack

from app.cli.ingest.apply import RunConfig
from app.cli.ingest.apply import run_sync
from app.cli.ingest.corpus import Visibility
from app.cli.ingest.report import RunReport
from app.ingest import INGEST_KEY
from tests.cli.ingest._fake_server import FakeServer
from tests.cli.ingest._fake_server import Json

THREAD_ID = 'kb:handbook'
SERVER = 'http://127.0.0.1:8765/mcp'
STORE = 'store_context_batch'
UPDATE = 'update_context_batch'
DELETE = 'delete_context_batch'
PLAN = 'plan_context_sync'
SYNCED_AT = '2026-10-18T12:00:00Z'


def write_document(root: Path, relative: str, text: str) -> Path:
    """Write a corpus file, creating its directories."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


class RunSettings(TypedDict, total=False):
    """The run settings a test changes from their defaults."""

    tags: tuple[str, ...]
    metadata: Mapping[str, object]
    visibility: Visibility
    defer_generation: bool
    token_file: Path
    token_env: str
    dry_run: bool
    allow_empty_corpus: bool
    batch_size: int
    max_request_bytes: int
    concurrency: int


async def sync_directory(server: FakeServer, root: Path, **settings: Unpack[RunSettings]) -> RunReport:
    """Run a sync of a directory against the fake server."""
    config = RunConfig(server=SERVER, corpus='handbook', source_dir=root, **settings)
    return await run_sync(config, connect=server.connect)


def managed_metadata(source_id: str) -> Json:
    """Return the metadata of an entry an earlier run stored for a source."""
    return {INGEST_KEY: {'scope': THREAD_ID, 'source_id': source_id, 'meta_digest': '0' * 64, 'synced_at': SYNCED_AT}}


def sources_with(report: RunReport, outcome: str) -> list[str]:
    """Return the source ids of the documents with one outcome."""
    return [item.source_id for item in report.outcomes if item.outcome == outcome]
