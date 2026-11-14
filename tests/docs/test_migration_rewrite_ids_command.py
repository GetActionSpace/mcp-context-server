"""The ``--rewrite-ids-at`` command line of docs/migration-v2-to-v3.md runs as written.

The "Rewriting Entry Ids in Metadata" section shows the command an operator types to name the metadata paths that hold
entry ids, and each of the two tests checks a different property of that command.

Outside quotes, a path holding ``*`` is a shell wildcard, which a shell replaces with the names of the files that match it.
The quoting test fails when a path of the command is not in single quotes, which make a shell pass the path unchanged.

The run test splits the command with ``shlex.split``, which applies a POSIX shell's quoting and word splitting but no
pathname expansion, so it never receives a path a shell replaced. It runs the command with only its source and target
URLs replaced, by a seeded source whose metadata holds ids at the command's paths and a temporary target; the report is
written under the name the command gives, in a temporary directory. The run must exit with status 0 and report every path
under ``id_paths``, each with at least one id rewritten, so a path of the command that finds no id of a migrated entry in
the seeded source, such as a mistyped path or one the seed does not cover, fails the check.
"""

import json
import re
import shlex
from pathlib import Path

import pytest

from app.cli.migrate import main as cli_main
from tests.cli.migrate_uuid._id_rewrite_scenario import SCENARIO_ROWS
from tests.cli.migrate_uuid._sources import seed_source_db

_DOC = Path(__file__).resolve().parents[2] / 'docs' / 'migration-v2-to-v3.md'
_SECTION = 'Rewriting Entry Ids in Metadata'
_HEADING = re.compile(r'^(?P<marks>#{1,6})[ \t]+(?P<text>.+?)[ \t]*$')
_FENCE = re.compile(r'^[ \t]*```(?P<info>\S*)')
_COMMAND = 'mcp-context-server-migrate'
_PATH_OPTION = '--rewrite-ids-at'
_SINGLE_QUOTED_PATH = re.compile(_PATH_OPTION + r"[ \t]+'(?P<path>[^']*)'")


def _section_lines() -> list[str]:
    """Return the lines of the "Rewriting Entry Ids in Metadata" section."""
    level: int | None = None
    section: list[str] = []
    in_fence = False
    for line in _DOC.read_text(encoding='utf-8').splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading is not None:
            depth = len(heading['marks'])
            if level is not None and depth <= level:
                break
            if level is None and heading['text'] == _SECTION:
                level = depth
                continue
        if level is not None:
            section.append(line)
    assert level is not None, f'{_DOC.name} has no "{_SECTION}" heading'
    return section


def _command() -> str:
    """Return the one bash block of the section that runs the CLI with ``--rewrite-ids-at``."""
    blocks: list[str] = []
    current: list[str] | None = None
    for line in _section_lines():
        fence = _FENCE.match(line)
        if current is None:
            if fence is not None and fence['info'] == 'bash':
                current = []
        elif fence is not None:
            blocks.append('\n'.join(current))
            current = None
        else:
            current.append(line)
    commands = [block for block in blocks if block.lstrip().startswith(_COMMAND) and _PATH_OPTION in block]
    assert len(commands) == 1, (
        f'expected one bash block running {_COMMAND} with {_PATH_OPTION} under "{_SECTION}", found {len(commands)}'
    )
    return commands[0]


def _shell_words(command: str) -> list[str]:
    """Split a command into the words a POSIX shell passes, joining continued lines first."""
    return shlex.split(command.replace('\\\n', ' '))


def _values(words: list[str], option: str) -> list[str]:
    """Return the value after each occurrence of ``option``, in order."""
    return [words[index + 1] for index, word in enumerate(words[:-1]) if word == option]


def _with_value(words: list[str], option: str, value: str) -> list[str]:
    """Return ``words`` with the value after the one ``option`` replaced by ``value``."""
    assert words.count(option) == 1, f'the command must give {option} once'
    index = words.index(option)
    return [*words[: index + 1], value, *words[index + 2:]]


def test_every_path_is_single_quoted() -> None:
    """Each ``--rewrite-ids-at`` path is in single quotes, and the command shows a path holding ``*``."""
    command = _command()

    paths = _values(_shell_words(command), _PATH_OPTION)

    assert paths
    assert [match['path'] for match in _SINGLE_QUOTED_PATH.finditer(command)] == paths
    assert any('*' in path for path in paths)


def test_command_runs_and_reports_each_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command exits 0 and reports each quoted path, unchanged, with an id rewritten there."""
    source, target = tmp_path / 'source.db', tmp_path / 'target.db'
    seed_source_db(source, SCENARIO_ROWS)
    words = _shell_words(_command())
    assert words[0] == _COMMAND
    arguments = _with_value(words[1:], '--source-url', f'sqlite:///{source.as_posix()}')
    arguments = _with_value(arguments, '--target-url', f'sqlite:///{target.as_posix()}')
    (report_name,) = _values(arguments, '--report')
    monkeypatch.chdir(tmp_path)

    assert cli_main(arguments) == 0

    loaded = json.loads((tmp_path / report_name).read_text(encoding='utf-8'))
    assert isinstance(loaded, dict)
    id_paths = loaded['id_paths']
    assert isinstance(id_paths, dict)
    assert list(id_paths) == _values(arguments, _PATH_OPTION)
    for path, counts in id_paths.items():
        assert isinstance(counts, dict)
        assert counts['rewritten'] > 0, f'{path} found no id of a migrated entry: {counts}'
