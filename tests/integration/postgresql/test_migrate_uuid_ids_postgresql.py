"""Acceptance tests of ``--rewrite-ids-at`` and ``--legacy-id-key`` in the migration directions through PostgreSQL.

Each test runs the command line the operator runs against databases created for it on the
test container, then reads the target. The default path, a given path and the stamp are
checked against the shared scenario in every direction, so each is held to what the
SQLite-to-SQLite migration produces. The tests of skipped rows and of the dry run write
their own expectations, because only a PostgreSQL target skips rows.
"""

import asyncio
import json
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import asyncpg
import pytest

from app.cli.migrate import main as cli_main
from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import get_settings
from tests.cli.migrate_uuid._id_rewrite_scenario import SCENARIO_ROWS
from tests.cli.migrate_uuid._id_rewrite_scenario import TargetEntry
from tests.cli.migrate_uuid._id_rewrite_scenario import assert_default_rewrite
from tests.cli.migrate_uuid._id_rewrite_scenario import assert_legacy_stamp
from tests.cli.migrate_uuid._id_rewrite_scenario import assert_named_paths
from tests.cli.migrate_uuid._id_rewrite_scenario import new_id
from tests.cli.migrate_uuid._id_rewrite_scenario import read_sqlite_target
from tests.cli.migrate_uuid._id_rewrite_scenario import row_warnings
from tests.cli.migrate_uuid._id_rewrite_scenario import source_row
from tests.cli.migrate_uuid._sources import seed_source_db
from tests.helpers import LOCAL_SCOPE
from tests.integration.postgresql._v2_sources import drop_isolated_db
from tests.integration.postgresql._v2_sources import make_isolated_db
from tests.integration.postgresql._v2_sources import seed_pg_v2_source

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

type Direction = Literal['sqlite_to_postgresql', 'postgresql_to_sqlite', 'postgresql_to_postgresql']
type CreateDatabase = Callable[[str], str]

_TO_POSTGRESQL: list[Direction] = ['sqlite_to_postgresql', 'postgresql_to_postgresql']


@pytest.fixture(autouse=True)
def status_is_the_only_indexed_field(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Index only ``status`` as a string, whatever the developer environment sets."""
    monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'status')
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


@pytest.fixture
def create_database(pg_test_url: str) -> Iterator[CreateDatabase]:
    """Create empty databases on the test container; drop them when the test ends."""
    created: list[str] = []

    def create(name: str) -> str:
        url = asyncio.run(make_isolated_db(pg_test_url, name))
        created.append(name)
        return url

    yield create
    for name in created:
        asyncio.run(drop_isolated_db(pg_test_url, name))


async def _read_postgresql_target(url: str) -> dict[str, TargetEntry]:
    """Read every entry of a migrated PostgreSQL target, keyed by its text."""
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch('SELECT id, text_content, metadata::text AS metadata FROM context_entries')
    finally:
        await conn.close()
    return {str(row['text_content']): TargetEntry(id=row['id'].hex, metadata_text=row['metadata']) for row in rows}


async def _has_entries_table(url: str) -> bool:
    """Whether the database at ``url`` has a ``context_entries`` table."""
    conn = await asyncpg.connect(url)
    try:
        return await conn.fetchval("SELECT to_regclass('context_entries') IS NOT NULL")
    finally:
        await conn.close()


async def _texts_found_by_legacy_id(url: str, old_id: int) -> list[str]:
    """Search a PostgreSQL target with the metadata filter ``search_context`` applies."""
    from app.backends.factory import create_backend
    from app.repositories.context_repository import ContextRepository

    backend = create_backend(backend_type='postgresql', connection_string=url)
    await backend.initialize()
    try:
        found, _ = await ContextRepository(backend).search_contexts(metadata={'legacy_id': old_id}, scope=LOCAL_SCOPE)
    finally:
        await backend.shutdown()
    return [str(row['text_content']) for row in found]


class _Endpoints:
    """The source and target of one migration, seeded and ready to run."""

    def __init__(
        self, direction: Direction, rows: list[dict[str, object]], tmp_path: Path, create_database: CreateDatabase,
    ) -> None:
        if direction == 'sqlite_to_postgresql':
            source_path = tmp_path / 'source.db'
            seed_source_db(source_path, rows)
            self.source_url = f'sqlite:///{source_path.as_posix()}'
        else:
            self.source_url = create_database('mcp_ids_rewrite_src')
            asyncio.run(seed_pg_v2_source(self.source_url, rows))
        self._target_path: Path | None = None
        if direction == 'postgresql_to_sqlite':
            self._target_path = tmp_path / 'target.db'
            self.target_url = f'sqlite:///{self._target_path.as_posix()}'
        else:
            self.target_url = create_database('mcp_ids_rewrite_tgt')

    def migrate(self, report: Path, *options: str) -> int:
        """Run the migration command line with a report."""
        return cli_main(
            ['--source-url', self.source_url, '--target-url', self.target_url, '--report', str(report), *options],
        )

    def read_target(self) -> dict[str, TargetEntry]:
        """Read every migrated entry by its text."""
        if self._target_path is not None:
            return read_sqlite_target(self._target_path)
        return asyncio.run(_read_postgresql_target(self.target_url))

    def target_written(self) -> bool:
        """Whether a run wrote the target: the SQLite file exists, or the PostgreSQL entries table does."""
        if self._target_path is not None:
            return self._target_path.exists()
        return asyncio.run(_has_entries_table(self.target_url))


def _report(path: Path) -> dict[str, object]:
    """Load a ``--report`` file."""
    loaded = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize('direction', ['sqlite_to_postgresql', 'postgresql_to_sqlite', 'postgresql_to_postgresql'])
def test_default_path_and_stamp(direction: Direction, tmp_path: Path, create_database: CreateDatabase) -> None:
    """The default path rewrites forward references and numeric strings, and every entry is stamped."""
    endpoints = _Endpoints(direction, SCENARIO_ROWS, tmp_path, create_database)
    report = tmp_path / 'report.json'

    assert endpoints.migrate(report, '--legacy-id-key', 'legacy_id') == 0

    rows, loaded = endpoints.read_target(), _report(report)
    assert_default_rewrite(rows, loaded)
    assert_legacy_stamp(rows, loaded)


@pytest.mark.parametrize('direction', ['sqlite_to_postgresql', 'postgresql_to_sqlite', 'postgresql_to_postgresql'])
def test_given_paths_replace_the_default(direction: Direction, tmp_path: Path, create_database: CreateDatabase) -> None:
    """A path given on the command line is rewritten in place of the default one."""
    endpoints = _Endpoints(direction, SCENARIO_ROWS, tmp_path, create_database)
    report = tmp_path / 'report.json'

    assert endpoints.migrate(report, '--rewrite-ids-at', 'links.closed_by') == 0

    assert_named_paths(endpoints.read_target(), _report(report))


@pytest.mark.parametrize('direction', _TO_POSTGRESQL)
def test_stamped_entry_is_found_by_metadata_filter(
    direction: Direction, tmp_path: Path, create_database: CreateDatabase,
) -> None:
    """The metadata filter ``search_context`` applies finds an entry by its old id on a PostgreSQL target."""
    endpoints = _Endpoints(direction, SCENARIO_ROWS, tmp_path, create_database)

    assert endpoints.migrate(tmp_path / 'report.json', '--legacy-id-key', 'legacy_id') == 0

    assert asyncio.run(_texts_found_by_legacy_id(endpoints.target_url, 2)) == ['entry-2']


# Entry 2 is the one the target cannot accept: on a SQLite source its text holds a NUL,
# which PostgreSQL cannot store; on a PostgreSQL source its indexed status is wider than
# the index accepts. Entry 1, older, and entry 3, newer, cite it; entry 2 cites entry 1.
_SKIPPED_COLUMN: dict[Direction, str] = {
    'sqlite_to_postgresql': 'text_content',
    'postgresql_to_postgresql': 'metadata.status',
}


def _rows_with_a_skipped_entry(direction: Direction) -> list[dict[str, object]]:
    """The source rows of a run whose target skips entry 2."""
    skipped = source_row(2, {'references': {'context_ids': [1]}}, '2024-02-01 10:00:00')
    if direction == 'sqlite_to_postgresql':
        skipped['text_content'] = 'entry-2\x00'
    else:
        skipped['metadata'] = {'references': {'context_ids': [1]}, 'status': 'x' * 3000}
    return [
        source_row(1, {'references': {'context_ids': [2]}}, '2024-01-01 10:00:00'),
        skipped,
        source_row(3, {'references': {'context_ids': [2, 1]}}, '2024-03-01 10:00:00'),
    ]


def _skip_error_prefix(direction: Direction) -> str:
    """The start of the error that reports the skip of entry 2."""
    return f"context_entries row id=2 thread_id='thread-ids' column '{_SKIPPED_COLUMN[direction]}' skipped"


@pytest.mark.parametrize('direction', _TO_POSTGRESQL)
def test_reference_to_a_skipped_row_is_not_rewritten(
    direction: Direction, tmp_path: Path, create_database: CreateDatabase,
) -> None:
    """A reference to an entry the target skips stays as written, whether the citing entry is older or newer."""
    endpoints = _Endpoints(direction, _rows_with_a_skipped_entry(direction), tmp_path, create_database)
    report = tmp_path / 'report.json'

    assert endpoints.migrate(report, '--legacy-id-key', 'legacy_id') == 1

    rows, loaded = endpoints.read_target(), _report(report)
    assert sorted(rows) == ['entry-1', 'entry-3']
    assert rows['entry-1'].metadata == {'references': {'context_ids': [2]}, 'legacy_id': 1}
    assert rows['entry-3'].metadata == {'references': {'context_ids': [2, new_id(rows, 1)]}, 'legacy_id': 3}
    # The skipped entry's own reference to entry 1 is not counted: it never reaches the target.
    assert loaded['id_paths'] == {DEFAULT_ID_PATH: {'rewritten': 1, 'not_migrated': 2, 'kept': 0}}
    assert loaded['legacy_id'] == {'key': 'legacy_id', 'stamped': 2, 'already_present': 0, 'not_stamped': 0}
    assert loaded['rows_migrated'] == 2
    assert row_warnings(loaded) == [
        'row 1: metadata.references.context_ids[0] holds 2, the id of an entry this run skipped; preserved',
        'row 3: metadata.references.context_ids[0] holds 2, the id of an entry this run skipped; preserved',
    ]
    errors = loaded['errors']
    assert isinstance(errors, list)
    assert len(errors) == 1
    assert str(errors[0]).startswith(_skip_error_prefix(direction))


@pytest.mark.parametrize('direction', _TO_POSTGRESQL)
def test_a_row_is_skipped_by_the_metadata_it_would_store_once_rewritten(
    direction: Direction, tmp_path: Path, create_database: CreateDatabase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row whose rewritten id breaks an integer index stays skipped, and a reference to it stays as written.

    Entry 1 is skipped for its wide status. Entry 2's integer-indexed ``parent`` cites
    entry 1: rewritten to a 32-character id it fails the integer index, so entry 2 is
    skipped too, although the value it ends up holding, entry 1's old id kept as written,
    would fit. Entry 3 cites entry 2, which the target never stores.
    """
    monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'status,parent:integer')
    get_settings.cache_clear()
    rows = [
        source_row(1, {'status': 'x' * 3000}, '2024-01-01 10:00:00'),
        source_row(2, {'parent': 1}, '2024-02-01 10:00:00'),
        source_row(3, {'references': {'context_ids': [2]}}, '2024-03-01 10:00:00'),
    ]
    endpoints = _Endpoints(direction, rows, tmp_path, create_database)
    report = tmp_path / 'report.json'

    exit_code = endpoints.migrate(
        report, '--rewrite-ids-at', 'parent', '--rewrite-ids-at', 'references.context_ids',
    )

    assert exit_code == 1
    migrated, loaded = endpoints.read_target(), _report(report)
    assert sorted(migrated) == ['entry-3']
    assert migrated['entry-3'].metadata == {'references': {'context_ids': [2]}}
    assert loaded['id_paths'] == {
        'parent': {'rewritten': 0, 'not_migrated': 0, 'kept': 0},
        'references.context_ids': {'rewritten': 0, 'not_migrated': 1, 'kept': 0},
    }
    assert row_warnings(loaded) == [
        'row 2: metadata.parent holds 1, the id of an entry this run skipped; preserved',
        'row 3: metadata.references.context_ids[0] holds 2, the id of an entry this run skipped; preserved',
    ]
    errors = [str(error) for error in loaded['errors']] if isinstance(loaded['errors'], list) else []
    assert len(errors) == 2
    assert errors[0].startswith("context_entries row id=1 thread_id='thread-ids' column 'metadata.status' skipped")
    assert errors[1].startswith("context_entries row id=2 thread_id='thread-ids' column 'metadata.parent' skipped")


@pytest.mark.parametrize('direction', ['sqlite_to_postgresql', 'postgresql_to_sqlite', 'postgresql_to_postgresql'])
def test_dry_run_matches_real_run(direction: Direction, tmp_path: Path, create_database: CreateDatabase) -> None:
    """A dry run reports what the real run that follows it writes, and writes nothing.

    A PostgreSQL target runs the rows with a skipped entry, so the skip, the references
    to it and the stamp outcomes are compared; the SQLite target runs the shared scenario.
    """
    to_postgresql = direction in _TO_POSTGRESQL
    rows = _rows_with_a_skipped_entry(direction) if to_postgresql else SCENARIO_ROWS
    exit_code = 1 if to_postgresql else 0
    endpoints = _Endpoints(direction, rows, tmp_path, create_database)
    dry, real = tmp_path / 'dry.json', tmp_path / 'real.json'

    assert endpoints.migrate(dry, '--dry-run', '--legacy-id-key', 'legacy_id') == exit_code
    assert endpoints.target_written() is False
    assert endpoints.migrate(real, '--legacy-id-key', 'legacy_id') == exit_code
    assert endpoints.target_written() is True

    dry_report, real_report = _report(dry), _report(real)
    for key in ('rows_migrated', 'id_paths', 'legacy_id', 'errors'):
        assert dry_report[key] == real_report[key], key
    assert row_warnings(real_report) != []
    assert row_warnings(dry_report) == row_warnings(real_report)
    assert real_report['rows_migrated'] == (2 if to_postgresql else len(SCENARIO_ROWS))


async def _seed_pg_embeddings(url: str, old_ids: list[int]) -> None:
    """Give each named entry of a seeded PostgreSQL source one embedding_metadata row and one vector."""
    conn = await asyncpg.connect(url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute(
            'CREATE TABLE embedding_metadata ('
            '  context_id BIGINT NOT NULL PRIMARY KEY, model_name TEXT NOT NULL, dimensions INTEGER NOT NULL,'
            '  chunk_count INTEGER NOT NULL DEFAULT 1,'
            '  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,'
            '  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);'
            'CREATE TABLE vec_context_embeddings ('
            '  id BIGSERIAL PRIMARY KEY, context_id BIGINT NOT NULL, embedding vector(3),'
            '  start_index INTEGER NOT NULL DEFAULT 0, end_index INTEGER NOT NULL DEFAULT 0);',
        )
        for old_id in old_ids:
            await conn.execute(
                "INSERT INTO embedding_metadata (context_id, model_name, dimensions) VALUES ($1, 'fake-model', 3)",
                old_id,
            )
            await conn.execute(
                "INSERT INTO vec_context_embeddings (context_id, embedding, end_index) VALUES ($1, '[0.1,0.2,0.3]', 7)",
                old_id,
            )
    finally:
        await conn.close()


def _warnings_with(report: dict[str, object], text: str) -> list[str]:
    """Return the warnings of a report that contain ``text``."""
    warnings = report['warnings']
    assert isinstance(warnings, list)
    return [str(message) for message in warnings if text in str(message)]


def test_dry_run_into_an_empty_postgresql_target_counts_the_embeddings_the_real_run_copies(
    tmp_path: Path, create_database: CreateDatabase,
) -> None:
    """A dry run into a target the real run initializes leaves out a skipped entry's embeddings, as the real run does."""
    direction: Direction = 'postgresql_to_postgresql'
    endpoints = _Endpoints(direction, _rows_with_a_skipped_entry(direction), tmp_path, create_database)
    asyncio.run(_seed_pg_embeddings(endpoints.source_url, [1, 2]))
    dry, real = tmp_path / 'dry.json', tmp_path / 'real.json'

    assert endpoints.migrate(dry, '--dry-run') == 1
    assert endpoints.target_written() is False
    assert endpoints.migrate(real) == 1

    dry_report, real_report = _report(dry), _report(real)
    assert real_report['embedding_metadata_migrated'] == 1
    assert real_report['vec_rows_migrated'] == 1
    for key in ('rows_migrated', 'embedding_metadata_migrated', 'vec_rows_migrated', 'errors'):
        assert dry_report[key] == real_report[key], key
    missing = 'references missing context_id'
    assert _warnings_with(dry_report, missing) == _warnings_with(real_report, missing) == [
        'embedding_metadata row references missing context_id=2; skipped',
        'vec_context_embeddings row references missing context_id=2; skipped',
    ]
    assert _warnings_with(dry_report, 'would be auto-initialized') != []
    assert _warnings_with(dry_report, 'initialize the target schema first') == []
