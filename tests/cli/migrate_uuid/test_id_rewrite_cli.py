"""Acceptance tests of ``--rewrite-ids-at`` and ``--legacy-id-key`` on a SQLite-to-SQLite migration.

Each test runs the command line the operator runs and reads the target database it wrote.
"""

import asyncio
import json
from pathlib import Path

import pytest

from app.cli.migrate import main as cli_main
from app.cli.migrate_uuid.id_paths import DEFAULT_ID_PATH
from tests.cli.migrate_uuid._id_rewrite_scenario import SCENARIO_ROWS
from tests.cli.migrate_uuid._id_rewrite_scenario import assert_default_rewrite
from tests.cli.migrate_uuid._id_rewrite_scenario import assert_legacy_stamp
from tests.cli.migrate_uuid._id_rewrite_scenario import assert_named_paths
from tests.cli.migrate_uuid._id_rewrite_scenario import new_id
from tests.cli.migrate_uuid._id_rewrite_scenario import read_sqlite_target
from tests.cli.migrate_uuid._id_rewrite_scenario import row_warnings
from tests.cli.migrate_uuid._sources import seed_source_db
from tests.helpers import LOCAL_SCOPE

IN_PLACE_REFUSAL = '--rewrite-ids-at and --legacy-id-key are valid only with the v2->v3 migration, not with an in-place mode'


@pytest.fixture
def scenario_source(tmp_path: Path) -> Path:
    """An integer-keyed source holding the shared scenario rows."""
    path = tmp_path / 'source.db'
    seed_source_db(path, SCENARIO_ROWS)
    return path


def _migrate(source: Path, target: Path, report: Path, *options: str) -> int:
    """Run the migration command line from ``source`` to ``target`` with a report."""
    return cli_main(
        [
            '--source-url', f'sqlite:///{source.as_posix()}',
            '--target-url', f'sqlite:///{target.as_posix()}',
            '--report', str(report),
            *options,
        ],
    )


def _report(path: Path) -> dict[str, object]:
    """Load a ``--report`` file."""
    loaded = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(loaded, dict)
    return loaded


class TestRewriteAtPaths:
    """Which values the default and the given paths rewrite."""

    def test_default_path_rewrites_forward_references_and_numeric_strings(
        self, scenario_source: Path, tmp_path: Path,
    ) -> None:
        """With no path given, every ``references.context_ids`` value naming a migrated entry is rewritten."""
        target, report = tmp_path / 'target.db', tmp_path / 'report.json'

        assert _migrate(scenario_source, target, report) == 0

        loaded = _report(report)
        assert_default_rewrite(read_sqlite_target(target), loaded)
        assert loaded['legacy_id'] is None

    def test_given_paths_replace_the_default(self, scenario_source: Path, tmp_path: Path) -> None:
        """A given path is the only one rewritten; the default no longer applies."""
        target, report = tmp_path / 'target.db', tmp_path / 'report.json'

        assert _migrate(scenario_source, target, report, '--rewrite-ids-at', 'links.closed_by') == 0

        assert_named_paths(read_sqlite_target(target), _report(report))

    def test_wildcard_and_any_depth_paths(self, scenario_source: Path, tmp_path: Path) -> None:
        """``*`` reaches every key at its level and a leading ``**.`` every depth."""
        target, report = tmp_path / 'target.db', tmp_path / 'report.json'

        exit_code = _migrate(
            scenario_source, target, report, '--rewrite-ids-at', 'links.*', '--rewrite-ids-at', '**.context_ids',
        )

        assert exit_code == 0
        rows = read_sqlite_target(target)
        entry_2 = rows['entry-2'].metadata
        assert isinstance(entry_2, dict)
        # The wildcard rewrites the numeric string under github_prs too: it equals a migrated id.
        assert entry_2['links'] == {'closed_by': [new_id(rows, 3)], 'github_prs': [new_id(rows, 1)]}
        assert entry_2['references'] == {'context_ids': [new_id(rows, 1)]}
        id_paths = _report(report)['id_paths']
        assert isinstance(id_paths, dict)
        assert list(id_paths) == ['links.*', '**.context_ids']
        assert id_paths == {
            'links.*': {'rewritten': 2, 'not_migrated': 0, 'kept': 0},
            '**.context_ids': {'rewritten': 5, 'not_migrated': 1, 'kept': 3},
        }

    def test_first_path_owns_a_location(self, scenario_source: Path, tmp_path: Path) -> None:
        """A value two paths find is rewritten once and counted under the path given first."""
        target, report = tmp_path / 'target.db', tmp_path / 'report.json'

        exit_code = _migrate(
            scenario_source, target, report,
            '--rewrite-ids-at', 'references.context_ids', '--rewrite-ids-at', DEFAULT_ID_PATH,
        )

        assert exit_code == 0
        rows = read_sqlite_target(target)
        entry_3 = rows['entry-3'].metadata
        assert isinstance(entry_3, dict)
        assert entry_3['items'] == [{'references': {'context_ids': new_id(rows, 2)}}]
        assert _report(report)['id_paths'] == {
            'references.context_ids': {'rewritten': 4, 'not_migrated': 1, 'kept': 3},
            DEFAULT_ID_PATH: {'rewritten': 1, 'not_migrated': 0, 'kept': 0},
        }


class TestLegacyIdKey:
    """The stamp of each migrated entry's old id."""

    def test_legacy_id_key_stamps_every_entry(self, scenario_source: Path, tmp_path: Path) -> None:
        """Every entry gets its old id under the key, except one that already holds the key."""
        target, report = tmp_path / 'target.db', tmp_path / 'report.json'

        assert _migrate(scenario_source, target, report, '--legacy-id-key', 'legacy_id') == 0

        rows, loaded = read_sqlite_target(target), _report(report)
        assert_legacy_stamp(rows, loaded)
        assert_default_rewrite(rows, loaded)

    def test_stamped_entry_is_found_by_metadata_filter(self, scenario_source: Path, tmp_path: Path) -> None:
        """The metadata filter ``search_context`` applies finds an entry by its old id."""
        from app.backends.sqlite_backend import SQLiteBackend
        from app.repositories.context_repository import ContextRepository

        target = tmp_path / 'target.db'
        assert _migrate(scenario_source, target, tmp_path / 'report.json', '--legacy-id-key', 'legacy_id') == 0

        async def search() -> list[str]:
            backend = SQLiteBackend(db_path=str(target))
            await backend.initialize()
            try:
                found, _ = await ContextRepository(backend).search_contexts(
                    metadata={'legacy_id': 2}, scope=LOCAL_SCOPE,
                )
            finally:
                await backend.shutdown()
            return [str(row['text_content']) for row in found]

        assert asyncio.run(search()) == ['entry-2']

    def test_unparseable_metadata_is_not_stamped(self, tmp_path: Path) -> None:
        """Metadata that does not parse is copied verbatim, reported, and makes the run fail."""
        source, target, report = tmp_path / 'source.db', tmp_path / 'target.db', tmp_path / 'report.json'
        seed_source_db(source, [SCENARIO_ROWS[3], {**SCENARIO_ROWS[4], 'metadata': 'not json'}])

        assert _migrate(source, target, report, '--legacy-id-key', 'legacy_id') == 1

        assert read_sqlite_target(target)['entry-5'].metadata_text == 'not json'
        loaded = _report(report)
        assert loaded['legacy_id'] == {'key': 'legacy_id', 'stamped': 1, 'already_present': 0, 'not_stamped': 1}
        errors = loaded['errors']
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert str(errors[0]).startswith('row 5: metadata JSON parse failed')
        assert str(errors[0]).endswith("so the old id was not stamped under 'legacy_id'")

    def test_non_object_metadata_is_not_stamped(self, tmp_path: Path) -> None:
        """Metadata that is not an object cannot take a key, which is reported and makes the run fail."""
        source, target, report = tmp_path / 'source.db', tmp_path / 'target.db', tmp_path / 'report.json'
        seed_source_db(source, [SCENARIO_ROWS[3], {**SCENARIO_ROWS[4], 'metadata': [1, 2]}])

        assert _migrate(source, target, report, '--legacy-id-key', 'legacy_id') == 1

        assert read_sqlite_target(target)['entry-5'].metadata == [1, 2]
        loaded = _report(report)
        assert loaded['legacy_id'] == {'key': 'legacy_id', 'stamped': 1, 'already_present': 0, 'not_stamped': 1}
        assert loaded['errors'] == [
            "row 5: metadata is a JSON array, not an object, so the old id was not stamped under 'legacy_id'",
        ]


class TestDryRun:
    """A dry run reports what a real run writes."""

    def test_dry_run_reports_what_a_real_run_writes(self, scenario_source: Path, tmp_path: Path) -> None:
        """The dry run's counts and row messages equal those of the real run that follows."""
        target = tmp_path / 'target.db'
        options = ('--rewrite-ids-at', DEFAULT_ID_PATH, '--rewrite-ids-at', 'links.*', '--legacy-id-key', 'legacy_id')

        assert _migrate(scenario_source, target, tmp_path / 'dry.json', '--dry-run', *options) == 0
        assert not target.exists()
        assert _migrate(scenario_source, target, tmp_path / 'real.json', *options) == 0

        dry, real = _report(tmp_path / 'dry.json'), _report(tmp_path / 'real.json')
        for key in ('rows_migrated', 'id_paths', 'legacy_id', 'errors'):
            assert dry[key] == real[key], key
        assert row_warnings(dry) == row_warnings(real)
        assert real['rows_migrated'] == len(SCENARIO_ROWS)


class TestOptionValidation:
    """Which option values and combinations the command refuses and which it accepts.

    A refused one exits 2 before any database is opened; a stamp key that a rewrite path
    also reaches is accepted, and the run migrates with it.
    """

    @pytest.mark.parametrize(
        ('options', 'message'),
        [
            (['--rewrite-ids-at', 'links.*', '--compress'], IN_PLACE_REFUSAL),
            (['--legacy-id-key', 'legacy_id', '--compress'], IN_PLACE_REFUSAL),
            (['--legacy-id-key', 'legacy_id', '--embed-missing'], IN_PLACE_REFUSAL),
            (['--rewrite-ids-at', 'links.*', '--reassign-owner', 'a', 'b'], IN_PLACE_REFUSAL),
            *[
                (['--rewrite-ids-at', path], f"argument --rewrite-ids-at: '{path}' has an empty key")
                for path in ('', '.a', 'a.', 'a..b')
            ],
            *[
                (
                    ['--rewrite-ids-at', path],
                    f"argument --rewrite-ids-at: '{path}': \"**\" may appear only as the first key",
                )
                for path in ('**', 'links.**', 'a.**.b')
            ],
            *[
                (
                    ['--rewrite-ids-at', path],
                    f"argument --rewrite-ids-at: '{path}': \"*\" stands for a whole key and cannot be part of one",
                )
                for path in ('a*', '*x')
            ],
            (
                ['--rewrite-ids-at', 'links.*', '--rewrite-ids-at', 'links.*'],
                "--rewrite-ids-at 'links.*' is given twice",
            ),
            (
                ['--legacy-id-key', 'ingest'],
                "argument --legacy-id-key: 'ingest' is reserved for the corpus-sync block",
            ),
            *[
                (
                    ['--legacy-id-key', name],
                    (
                        f"argument --legacy-id-key: '{name}' cannot be stamped: the metadata filters address "
                        'a top-level key made of letters, digits, "_" and "-" only'
                    ),
                )
                for name in ('a.b', '', 'with space')
            ],
        ],
    )
    def test_option_validation(
        self,
        options: list[str],
        message: str,
        scenario_source: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The command refuses the value or combination with the message and writes nothing."""
        target = tmp_path / 'target.db'

        with pytest.raises(SystemExit) as raised:
            cli_main(
                [
                    '--source-url', f'sqlite:///{scenario_source.as_posix()}',
                    '--target-url', f'sqlite:///{target.as_posix()}',
                    *options,
                ],
            )

        assert raised.value.code == 2
        assert message in capsys.readouterr().err
        assert not target.exists()

    def test_a_key_the_default_path_reaches_is_accepted(self, scenario_source: Path, tmp_path: Path) -> None:
        """A stamp key that a rewrite path also reaches is accepted, and existing values under it are kept."""
        target, report = tmp_path / 'target.db', tmp_path / 'report.json'

        assert _migrate(scenario_source, target, report, '--legacy-id-key', 'references') == 0

        rows, loaded = read_sqlite_target(target), _report(report)
        assert loaded['legacy_id'] == {'key': 'references', 'stamped': 2, 'already_present': 3, 'not_stamped': 0}
        assert rows['entry-4'].metadata == {'references': 4}
        assert_default_rewrite(rows, loaded)
