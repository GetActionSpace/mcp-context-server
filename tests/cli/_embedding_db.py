"""SQLite databases for the embedding CLI tests, built and seeded with plain SQL.

:func:`bootstrap_schema` applies the base schema, which holds the generation job table, and
creates ``embedding_metadata`` directly, so the CLI's structural checks pass without loading
sqlite-vec. The seed helpers write entries, embedding markers and generation jobs the way
the server leaves them, and :func:`seed_jobs_database` builds the layout the drain cases
share: an entry without a job and one each with a pending, a failed and a held job.
:class:`DrainAndPass` replaces the generation the drain runs and the embedding the CLI's own
pass runs with recorders, so a case sees which entry each of them reached, in order, without
a provider.
"""

import sqlite3
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import app.generation_jobs.executor as executor
import app.tools._generation as generation
from app.schemas import load_schema
from app.settings import get_settings
from app.startup import get_summary_provider

type JobStatus = Literal['pending', 'running', 'failed']

# The claim token of a job a running server holds.
HELD_TOKEN = 'e' * 32

_EMBEDDING_METADATA_SQL = '''
CREATE TABLE IF NOT EXISTS embedding_metadata (
    context_id TEXT NOT NULL PRIMARY KEY,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
);
'''

_JOB_SQL: dict[JobStatus, str] = {
    'pending': 'INSERT INTO generation_jobs (context_entry_id) VALUES (?)',
    'running': (
        "INSERT INTO generation_jobs (context_entry_id, status, attempts, claim_token, available_at) "
        f"VALUES (?, 'running', 1, '{HELD_TOKEN}', datetime('now', '+120 seconds'))"
    ),
    'failed': (
        'INSERT INTO generation_jobs (context_entry_id, status, attempts, last_error) '
        "VALUES (?, 'failed', 3, 'provider down')"
    ),
}


def bootstrap_schema(path: Path, *, with_embedding_metadata: bool = True) -> None:
    """Apply the SQLite base schema and, optionally, the ``embedding_metadata`` table to ``path``."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(load_schema('sqlite'))
        if with_embedding_metadata:
            conn.executescript(_EMBEDDING_METADATA_SQL)
    finally:
        conn.close()


def _write(path: Path, sql: str, params: tuple[object, ...] = ()) -> None:
    """Run one write statement against the database at ``path`` and commit it."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def seed_entry(path: Path, entry_id: str, text: str) -> None:
    """Insert one context entry with ``text``, no embeddings and no job."""
    _write(
        path,
        'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
        "VALUES (?, 'thread-a', 'user', 'text', ?, 'local')",
        (entry_id, text),
    )


def seed_embedding_metadata(path: Path, context_id: str, model: str, dim: int) -> None:
    """Insert the ``embedding_metadata`` row that marks an entry as embedded."""
    _write(
        path,
        'INSERT INTO embedding_metadata (context_id, model_name, dimensions, chunk_count) VALUES (?, ?, ?, 1)',
        (context_id, model, dim),
    )


def seed_job(path: Path, context_id: str, status: JobStatus) -> None:
    """Give an entry a generation job: pending, running under a live lease, or failed after its attempts."""
    _write(path, _JOB_SQL[status], (context_id,))


def drop_jobs_table(path: Path) -> None:
    """Drop the generation job table, as a database prepared before deferred generation lacks it."""
    _write(path, 'DROP TABLE generation_jobs')


def job_statuses(path: Path) -> dict[str, str]:
    """Return the status of every generation job, by entry id."""
    conn = sqlite3.connect(str(path))
    try:
        return {str(row[0]): str(row[1]) for row in conn.execute('SELECT context_entry_id, status FROM generation_jobs')}
    finally:
        conn.close()


def provider_double(*, available: bool = True) -> MagicMock:
    """Return an embedding or summary provider double that initializes, reports ``available`` and shuts down."""
    provider = MagicMock()
    provider.provider_name = 'fake'
    provider.initialize = AsyncMock()
    provider.is_available = AsyncMock(return_value=available)
    provider.shutdown = AsyncMock()
    return provider


# An entry without a job, and one each whose job is pending, failed after its attempts, and
# held under a live lease by a running server.
ENTRY_WITHOUT_JOB = 'a' * 32
ENTRY_PENDING = 'b' * 32
ENTRY_FAILED = 'c' * 32
ENTRY_HELD = 'd' * 32

_JOB_LAYOUT: tuple[tuple[str, JobStatus | None], ...] = (
    (ENTRY_WITHOUT_JOB, None),
    (ENTRY_PENDING, 'pending'),
    (ENTRY_FAILED, 'failed'),
    (ENTRY_HELD, 'running'),
)


def text_of(entry_id: str) -> str:
    """Return the text :func:`seed_jobs_database` stores for ``entry_id``."""
    return f'doc {entry_id[0]}'


def seed_jobs_database(path: Path) -> None:
    """Build the database at ``path`` with the four entries of the job layout and their jobs."""
    bootstrap_schema(path)
    for entry_id, status in _JOB_LAYOUT:
        seed_entry(path, entry_id, text_of(entry_id))
        if status is not None:
            seed_job(path, entry_id, status)


def generation_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, summaries: bool = False) -> Path:
    """Configure embedding generation with compression off and return the database path to use.

    Args:
        monkeypatch: Sets the environment for the test.
        tmp_path: The test's directory.
        summaries: Whether summary generation is enabled.

    Returns:
        The path of the test's database.
    """
    db = tmp_path / 'test.db'
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('ENABLE_SUMMARY_GENERATION', 'true' if summaries else 'false')
    monkeypatch.setenv('DB_PATH', str(db))
    get_settings.cache_clear()
    return db


@dataclass
class DrainAndPass:
    """Records the entries the drain generates outputs for and the CLI's pass embeds, in order.

    Attributes:
        events: ``('drain', text)`` for every generation the drain ran and ``('pass', text)``
            for every entry the pass embedded.
        summary_providers: The summary provider registered at each generation the drain ran.
    """

    events: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    summary_providers: list[object] = field(default_factory=list[object])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace the drain's generation and the pass's embedding with this recorder's."""
        monkeypatch.setattr(executor, 'run_generation', self._generate)
        monkeypatch.setattr(generation, 'generate_embeddings_with_timeout', self._embed)

    def texts(self, stage: str) -> list[str]:
        """Return the texts of ``stage`` ('drain' or 'pass') in order."""
        return [text for event, text in self.events if event == stage]

    async def _generate(
        self, text: str, source: str, *, run_embedding: bool, run_summary: bool, run_nodes: bool,
    ) -> tuple[None, None, None]:
        """Record the drained entry and generate nothing."""
        del source, run_embedding, run_summary, run_nodes
        self.events.append(('drain', text))
        self.summary_providers.append(get_summary_provider())
        return None, None, None

    async def _embed(self, text: str) -> None:
        """Record the entry the pass reached and return no embeddings, so the pass skips it."""
        self.events.append(('pass', text))
